import ast
import logging
import math

import ase.io
import numpy as np
import torch
from e3nn import o3
from graph_longrange.gto_utils import gto_basis_kspace_cutoff
from graph_longrange.utils import FIELD_CONSTANT
from mace.modules import ScaleShiftMACE
from mace.modules.utils import prepare_graph
from mace.tools.scatter import scatter_sum
from torch import nn
from torch.nn import functional as F

RESPONSE_SCALE = .02
# A background density of states (e/V per atom) at all energies, so that every charge has one finite level.
DOS_FLOOR = 1.e-5
# Gaussian tails below exp(-TAIL) are dropped from the exact sums of the counter charge.
TAIL = 36.
# Form factors of the neutral atoms, P_Z(k)/K on a uniform grid in k (1/A).
FORM_FACTOR_SPACING, FORM_FACTOR_MAXIMUM = .01, 40.
FORM_FACTOR_GRID = np.arange(0., FORM_FACTOR_MAXIMUM+.5*FORM_FACTOR_SPACING, FORM_FACTOR_SPACING)


def positive(x):
    return RESPONSE_SCALE*(F.softplus(x+math.log(math.expm1(1.)))+1.e-6)


def gaussian(x):
    return torch.exp(-x*x)/math.sqrt(math.pi)


def gaussian_step(x):
    return .5*torch.erfc(-x)


def gaussian_ramp(x):
    return x*gaussian_step(x)+.5*gaussian(x)


def normal_density(x):
    return torch.exp(-.5*x*x)/math.sqrt(2*math.pi)


def normal_step(x):
    return .5*(1.+torch.erf(x/math.sqrt(2.)))


def window_moments(center, width, length):
    """Charge, first and second moment of a periodic unit Gaussian on the window [0, length), and its edge density."""
    mass = first = second = edge = torch.zeros_like(center)
    for n in (-1., 0., 1.):
        c = center+n*length
        a, b = -c/width, (length-c)/width
        pa, pb, part = normal_density(a), normal_density(b), normal_step(b)-normal_step(a)
        mass = mass+part
        first = first+c*part+width*(pa-pb)
        second = second+c.square()*part+2*c*width*(pa-pb)+width.square()*(part+a*pa-b*pb)
        edge = edge+pb/width
    return mass, first, second, edge


def window_derivatives(center, width, length, order):
    """The derivatives 0 to order of the periodic unit Gaussian of window_moments at the window ends 0 and length."""
    ends = []
    for end in (0., length):
        values = [torch.zeros_like(center) for _ in range(order+1)]
        for n in (-1., 0., 1.):
            x = (end-center-n*length)/width
            density, hermite = normal_density(x)/width, [torch.ones_like(x), x]
            for j in range(1, order):
                hermite.append(x*hermite[j]-j*hermite[j-1])
            values = [value+(-1./width)**j*hermite[j]*density for j, value in enumerate(values)]
        ends.append(values)
    return ends


def solid_harmonics(order, vectors):
    """The solid harmonics |x|^l Y_lm of e3nn for l = 0 to order, of norm |x|^l in every order; l = 1 is x."""
    return o3.spherical_harmonics(list(range(order+1)), vectors, False, normalization='norm')


def component_orders(order):
    """The order l of every component of the harmonics to order."""
    return [l for l in range(order+1) for _ in range(2*l+1)]


def phase_factors(order, sign, reference):
    """Real and imaginary part of (sign i)^l for every component of the harmonics to order."""
    cycle = reference.new_tensor([[1., 0.], [0., sign], [-1., 0.], [0., -sign]])
    factors = cycle[torch.tensor(component_orders(order), device=reference.device) % 4]
    return factors[:, 0], factors[:, 1]


def vector_asinh(vector):
    squared = vector.square().sum(-1, keepdim=True)
    threshold = math.sqrt(torch.finfo(vector.dtype).eps)
    norm = squared.clamp_min(threshold).sqrt()
    small = squared.clamp_max(threshold)
    series = 1.-small/6.+3.*small.square()/40.
    return vector*torch.where(squared < threshold, series, torch.asinh(norm)/norm)


def neighbour_transport(positions, edge_index, shifts, batch, slot, graphs, size, cutoff, width):
    """l = 0 and l = 1 averages over the neighbours on the MACE graph, without the atom itself."""
    sender, receiver = edge_index
    vector = positions[sender]-positions[receiver]-shifts.to(positions)
    square = vector.square().sum(-1)
    weights = torch.exp(-.5*square/width**2)*(1.-square/cutoff**2).clamp_min(0.).pow(3)
    index = (batch[receiver]*size+slot[receiver])*size+slot[sender]
    values = torch.cat((weights[:, None], weights[:, None]*vector/width), -1)
    raw = positions.new_zeros((graphs*size*size, 4)).index_add(0, index, values).reshape(graphs, size, size, 4)
    operator = (raw/(1.+raw[..., 0].sum(-1, keepdim=True))[..., None]).permute(0, 3, 1, 2)
    return (operator-torch.diag_embed(operator.sum(-1))).flatten(1, 2)


class AtomicResponse(nn.Module):
    """The response of every atom: bias, density of states on an energy grid, dd_dE, d0 and local amplitudes.

    It reads a learned mixture of the scalars and vectors of all num_layers layers, as wide as one layer. The
    multipoles of the atom reach order max_l and its local amplitudes order completion_max_l. Every order l >= 1
    has a permanent moment d0 and a response dd_dE, the moment per field projection of that order in units of
    the density width: for l = 1 dd_dE times the density width squared is the dipole per field. The outputs of
    order l are gated sums of the order l solid harmonics of the vector channels. The density of states is a
    sum of Gaussian bumps of width resolution on [-energy_range, energy_range] eV over a background of
    DOS_FLOOR, and starts at 1 e/V in the bands and 0.0025 e/V within 1 eV of the level. With fixed_point the
    response also reads the levels and fields of the atom and its neighbours; at zero levels and field that read
    is the response of the geometry.
    """

    def __init__(self, num_scalars, num_vectors, num_layers, num_elements, num_sources, num_receivers,
                 fixed_point=False, energy_range=6., resolution=.25, width=64, max_l=1, completion_max_l=1):
        super().__init__()
        # The mixture maps every layer linearly and sums them, normalized like an e3nn linear map of one layer.
        self.mixer_scalar = nn.Parameter(torch.randn(num_layers*num_scalars, num_scalars))
        self.mixer_vector = nn.Parameter(torch.randn(num_layers*num_vectors, num_vectors))
        self.max_l, self.completion_max_l = int(max_l), int(completion_max_l)
        self.vector_width = min(16, max(4, num_vectors))
        self.num_sources = num_sources
        self.chemical_scalar = nn.Linear(num_scalars, width)
        self.chemical_vector = nn.Linear(num_vectors, self.vector_width, bias=False)
        self.shared = nn.Sequential(nn.Linear(width+2*self.vector_width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.scalar_out = nn.Linear(width, 1+num_sources)
        self.vector_out = nn.Linear(width, 2*sum(self.outputs())*self.vector_width)
        self.dd_dE_readout = nn.Linear(width, self.max_l)
        centers = torch.linspace(-energy_range, energy_range, int(round(2*energy_range/resolution))+1)
        self.dos_width = float(centers[1]-centers[0])
        self.register_buffer('dos_centers', centers)
        self.register_buffer('dos_offsets', torch.where(centers.abs() < 1.-1.e-9, math.log(math.expm1(.0025)),
                                                        math.log(math.expm1(1.))))
        self.dos_readout = nn.Linear(width, len(centers))
        self.species_bias = nn.Parameter(torch.zeros(num_elements))
        self.species_source = nn.Parameter(torch.zeros(num_elements, num_sources))
        layers = [self.scalar_out, self.vector_out, self.dd_dE_readout, self.dos_readout]
        if fixed_point:
            self.state_scalar = nn.Linear(3*num_receivers, width, bias=False)
            self.state_vector = nn.Linear(3*num_receivers, self.vector_width, bias=False)
            layers += [self.state_scalar, self.state_vector]
        for layer in layers:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self.register_buffer('scalar_unit', torch.ones(num_scalars))
        self.register_buffer('vector_unit', torch.ones(num_vectors))

    def mix(self, scalars, vectors):
        """The mixture of the layers; a response saved before the mixture reads them side by side."""
        if 'mixer_scalar' not in self._parameters:
            return scalars, vectors
        return (scalars@self.mixer_scalar/math.sqrt(self.mixer_scalar.shape[1]),
                torch.einsum('...kc,kd->...dc', vectors, self.mixer_vector)/math.sqrt(self.mixer_vector.shape[1]))

    def features(self, scalars, vectors):
        scalars, vectors = self.mix(scalars, vectors)
        chemical = self.chemical_scalar(torch.asinh(scalars)/self.scalar_unit)
        vector = vector_asinh(vectors/self.vector_unit[:, None])
        return chemical, self.chemical_vector(vector.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, chemical, vector, attrs, mask, state=None):
        embedded = vector
        if state is not None:
            scalar, polar = state
            chemical = chemical+self.state_scalar(torch.asinh(scalar))
            embedded = vector+self.state_vector(vector_asinh(polar).transpose(-1, -2)).transpose(-1, -2)
        invariants = torch.cat((chemical, torch.asinh(embedded.square().sum(-1)),
                                torch.asinh((vector*embedded).sum(-1))), -1)
        hidden = self.shared(invariants)
        scalars = self.scalar_out(hidden)
        bias = (attrs@self.species_bias+scalars[..., 0])*mask
        dd_dE = positive(self.dd_dE_readout(hidden))*mask[..., None]
        coefficients = F.softplus(self.dos_readout(hidden)+self.dos_offsets)*mask[..., None]
        # The outputs of order l, d0 first, are gated sums of the order l solid harmonics of the vector channels.
        outputs, channels, max_l = self.outputs(), torch.cat((vector, embedded), -2), getattr(self, 'max_l', 1)
        harmonics = channels
        if len(outputs) > 1:
            harmonics = torch.cat((channels, solid_harmonics(len(outputs), channels)[..., 4:]), -1)
        gates = self.vector_out(hidden).reshape(*hidden.shape[:-1], sum(outputs), channels.shape[-2])
        d0, amplitudes, scale = [], [attrs@self.species_source+scalars[..., 1:]], math.sqrt(channels.shape[-2])
        for l, gate in enumerate(gates.split(outputs, -2) if outputs else (), 1):
            polar = torch.einsum('bnov,bnvc->bnoc', gate, harmonics[..., l*l-1:(l+1)**2-1])/scale
            if l <= max_l:
                d0.append(polar[..., 0, :])
            amplitudes.append(polar[..., int(l <= max_l):, :].flatten(-2))
        d0 = torch.cat(d0, -1) if d0 else bias.new_zeros(*bias.shape, 0)
        return bias, dd_dE, coefficients, d0*mask[..., None], torch.cat(amplitudes, -1)*mask[..., None]

    def outputs(self):
        """The number of equivariant outputs of every order l >= 1: d0 to max_l, the amplitudes to completion_max_l."""
        max_l, completion_max_l = getattr(self, 'max_l', 1), getattr(self, 'completion_max_l', 1)
        return [int(l <= max_l)+self.num_sources*int(l <= completion_max_l)
                for l in range(1, max(max_l, completion_max_l)+1)]

    def charge(self, coefficients, delta):
        """The charge of the atoms at the levels delta, and the density of states there."""
        w, centers = self.dos_width, self.dos_centers
        x = (delta[..., None]-centers)/w
        dq = (coefficients*(gaussian_step(x)-gaussian_step(-centers/w))).sum(-1)*w+DOS_FLOOR*delta
        return dq, (coefficients*gaussian(x)).sum(-1)+DOS_FLOOR

    def energy(self, coefficients, delta):
        """The integral of the charge from zero to delta."""
        w, centers = self.dos_width, self.dos_centers
        x = (delta[..., None]-centers)/w
        return (coefficients*(w*(gaussian_ramp(x)-gaussian_ramp(-centers/w))
                              - delta[..., None]*gaussian_step(-centers/w))).sum(-1)*w+.5*DOS_FLOOR*delta.square()


# The class of the atomic response in models saved before the renaming.
LocalLaw = AtomicResponse


def reciprocal_grid(cell, cutoff):
    """Odd FFT grid covering the sphere k < cutoff in every cell of the batch."""
    rcell = 2*math.pi*torch.linalg.inv(cell).transpose(-1, -2)
    limits = torch.ceil(torch.linalg.vector_norm(cell, dim=-1)*float(cutoff)/(2*math.pi)).long()
    shape = tuple(2*int(x)+1 for x in limits.max(0).values.detach().cpu())
    axes = [torch.fft.fftfreq(n, d=1./n, device=cell.device).to(cell) for n in shape]
    modes = torch.cartesian_prod(*axes)
    wave = torch.einsum("ki,gij->gkj", modes, rcell)
    k2 = wave.square().sum(-1)
    return wave, k2, k2 <= float(cutoff)**2, shape, modes


class FourierSpace:
    """The Fourier space of a batch: its operators, counter charge and boundary terms, packed as (graphs, atoms)."""

    def __init__(self, model, data, positions):
        self.cell = data['cell'].reshape(-1, 3, 3)
        self.volume = torch.linalg.det(self.cell).abs()
        pbc = data['pbc'].reshape(-1, 3).bool()
        if bool((pbc.sum(-1) < 2).any()):
            raise NotImplementedError('MACE-VOLT supports periodic bulk and two-periodic slabs')
        self.slab = pbc.sum(-1) == 2
        self.axes = (~pbc).long().argmax(-1)
        rows = torch.arange(len(self.cell), device=positions.device)
        normal = torch.linalg.cross(self.cell[rows, (self.axes+1) % 3], self.cell[rows, (self.axes+2) % 3])
        normal = normal*torch.sign((self.cell[rows, self.axes]*normal).sum(-1, keepdim=True))
        self.normal = normal/torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        self.length = (self.cell[rows, self.axes]*self.normal).sum(-1)
        self.wave, self.k2, self.mask, self.shape, self.modes = reciprocal_grid(self.cell, model.kspace_cutoff)
        self.ngrid = math.prod(self.shape)
        self.batch, ptr = data['batch'], data['ptr']
        self.counts = (ptr[1:]-ptr[:-1]).detach().cpu().tolist()
        self.size = max(self.counts)
        self.slot = torch.arange(len(positions), device=positions.device)-ptr[:-1][self.batch]
        self.present = positions.new_zeros((len(self.counts), self.size))
        self.present[self.batch, self.slot] = 1.
        self.width = model.density_width
        active = self.mask & (self.k2 > 0)
        self.coulomb = torch.where(active, FIELD_CONSTANT/self.k2.clamp_min(1.e-30), 0.)
        self.slab_factor = FIELD_CONSTANT/self.volume*self.slab
        self.external = data['external_field'].reshape(-1, 3)
        self.field = (self.external*self.normal).sum(-1)
        if bool(((self.external-self.field[:, None]*self.normal).abs().amax(-1) > 1.e-10).any()
                or ((self.field != 0) & ~self.slab).any()):
            raise ValueError('An applied field must be normal to a slab')
        # The dipole correction plane; the slab is centred half a cell from it.
        self.branch = data['dipole_correction_fraction'].reshape(-1).remainder(1.)
        self.branch_center = (self.branch+.5).remainder(1.)*self.slab
        self.origin = self.branch_center[:, None]*self.cell[rows, self.axes]*self.slab[:, None]
        self.planar = (self.modes[None]*pbc[:, None]).abs().sum(-1) == 0
        self.axial = self.planar & active & self.slab[:, None]
        self.order = self.modes[None].expand(len(self.counts), -1, -1).gather(
            -1, self.axes[:, None, None].expand(-1, len(self.modes), 1)).squeeze(-1)
        self.attrs = self.pack(data['node_attrs'])
        # The vacuum level of the neutral atoms in the zero-mean origin of periodic DFT.
        self.vacuum_reference = self.attrs.sum(1)@model.vacuum_reference_integrals/self.volume*self.slab
        self.counter_charge_sheet(model, data, positions)

        # Only the nonzero modes inside the cutoff enter the multipoles and the sampled fields, a half space of
        # them first: the Hessian of the multipoles, even in k, sums that half twice.
        n1, n2, n3 = self.modes.unbind(-1)
        half = active & ((n1 > 0) | ((n1 == 0) & ((n2 > 0) | ((n2 == 0) & (n3 > 0)))))
        size = active.shape[1]
        keys = torch.arange(size, device=positions.device)+size*((~half).long()+2*(~active).long())
        keys = keys.sort(-1).values[:, :int(active.sum(-1).max())]
        self.selected_mask = keys < 2*size
        self.selected_indices = keys % size
        self.half_count = int(half.sum(-1).max())
        self.half_mask = keys[:, :self.half_count] < size

        def selected(value):
            mask = self.selected_mask.reshape(*self.selected_mask.shape, *([1]*(value.ndim-2)))
            return value[rows[:, None], self.selected_indices]*mask
        # The orders of the multipoles and of the local amplitudes; models saved before them stop at l = 1.
        self.max_l, self.completion_max_l = getattr(model, 'max_l', 1), getattr(model, 'completion_max_l', 1)
        self.num_moments = (self.max_l+1)**2
        self.k, k2 = selected(self.wave), selected(self.k2)
        phase = torch.einsum('gkc,gnc->gkn', self.k, self.pack(positions))
        self.co = phase.cos()*self.present[:, None, :]*self.selected_mask[..., None]
        self.si = phase.sin()*self.present[:, None, :]*self.selected_mask[..., None]
        self.smooth = torch.exp(-.5*self.width**2*k2)*self.selected_mask
        self.harmonics = solid_harmonics(max(self.max_l, self.completion_max_l, 1), self.k)
        inside = self.selected_mask[..., None]
        self.radial, self.receiver = (torch.exp(-.5*k2[..., None]*widths.to(positions).square())*inside
                                      for widths in (model.source_widths, model.receiver_widths))
        self.poisson = -selected(self.coulomb)
        self.zn = ((self.pack(positions)-self.origin[:, None])*self.normal[:, None]).sum(-1)*self.present
        self.slab_ramp = selected(torch.view_as_real(self.ramp(-self.slab_factor)))
        # The fixed field of the solve: the applied field and the counter charge.
        self.fixed = selected(torch.view_as_real(self.ramp(self.field)+self.counter_potential))
        self.counter_coulomb = selected(torch.view_as_real(-self.coulomb*self.counter_density))
        self.kernels = coefficient_kernels(self, (2*math.pi)**1.5*self.width**3)

    def pack(self, values):
        result = values.new_zeros((len(self.counts), self.size, *values.shape[1:]))
        result[self.batch, self.slot] = values
        return result

    def unpack(self, values):
        return values[self.batch, self.slot]

    def ramp(self, field, width=None):
        """Fourier coefficients of the sawtooth of a normal field, smoothed to width, its jump at the plane."""
        safe = torch.where(self.axial, self.order, torch.ones_like(self.order))
        phase = -2*math.pi*safe*self.branch[:, None]
        amplitude = field[:, None]*self.length[:, None]/(2*math.pi*safe)
        width = self.width if width is None else width
        amplitude = amplitude*torch.exp(-.5*width**2*self.k2)*self.axial
        return torch.complex(-amplitude*phase.sin(), amplitude*phase.cos())

    def counter_charge_sheet(self, model, data, positions):
        """A Gaussian sheet of charge Q_cc parallel to the slab, with its self, slab and neutral atom energies."""
        n, zero = len(self.volume), self.volume.new_zeros(len(self.volume))
        self.counter_charge = data['counter_charge'].reshape(-1).to(zero)
        width = data['counter_charge_width'].reshape(-1).to(zero)
        present = (width > 0) & self.slab
        if bool(((self.counter_charge != 0) & ~present).any()):
            raise ValueError('A counter charge needs a slab and a counter_charge_width')
        width = torch.where(present, width, 1.)
        rows = torch.arange(n, device=zero.device)
        self.counter_center = data['counter_charge_fraction'].reshape(-1)[:, None]*self.cell[rows, self.axes]
        phase = -torch.einsum('gkc,gc->gk', self.wave, self.counter_center)
        profile = torch.exp(-.5*self.k2*width[:, None].square())*self.planar
        self.counter_density = ((self.counter_charge*present/self.volume)[:, None]*profile
                                * torch.complex(phase.cos(), phase.sin())*self.mask)
        c = ((self.counter_center*self.normal).sum(-1)/self.length-self.branch_center+.5).remainder(1.)-.5
        shift = (c[:, None]+torch.arange(-2., 3., device=zero.device))*self.length[:, None]
        a = (-.5*self.length[:, None]-shift)/width[:, None]
        b = (.5*self.length[:, None]-shift)/width[:, None]
        unit = (.5*shift*(torch.erf(b/math.sqrt(2))-torch.erf(a/math.sqrt(2)))
                + width[:, None]/math.sqrt(2*math.pi)*(torch.exp(-.5*a.square())-torch.exp(-.5*b.square()))).sum(-1)
        self.counter_moment = self.counter_charge*unit*present
        self.counter_width = width
        species, energy = data['node_attrs'].argmax(-1), zero.clone()
        table = model.neutral_atom_form_factors
        # In constant potential MD the sheet stands for ions and acts through electrostatics only.
        overlap = getattr(model, 'counter_charge_overlap', True)
        for i in torch.nonzero(present).flatten().tolist():
            sigma = float(width[i])
            atoms = data['batch'] == i
            rows = table[species[atoms].unique()]
            covered = FORM_FACTOR_SPACING*(rows.shape[-1]-1-(rows != 0).flip(-1).int().argmax(-1))
            if overlap and (bool((rows[:, 0] == 0).any()) or bool((covered < math.sqrt(2*TAIL)/sigma).any())):
                raise ValueError('A counter charge needs the neutral atoms of every element: train on frames '
                                 'with fourier_proto_potential')
            m = torch.arange(1, int(math.sqrt(2*TAIL)/sigma*float(self.length[i])/(2*math.pi))+2,
                             device=zero.device).to(zero)
            waves = (2*math.pi*m/self.length[i])[:, None]*self.normal[i]
            k2 = waves.square().sum(-1)
            # The self energy and the overlap -Q_cc <phi_neutral>, over the +k half of the modes.
            energy[i] = FIELD_CONSTANT*self.counter_charge[i].square()/self.volume[i]*(torch.exp(-sigma**2*k2)/k2).sum()
            if not overlap:
                continue
            coordinate = (k2.sqrt()/FORM_FACTOR_SPACING).clamp_max(table.shape[-1]-1.001)
            left = coordinate.long()
            form = (table[:, left]*(left+1-coordinate)+table[:, left+1]*(coordinate-left))[species[atoms]]
            cosine = torch.cos((positions[atoms]-self.counter_center[i])@waves.T)
            energy[i] = energy[i]-2*FIELD_CONSTANT*self.counter_charge[i]/self.volume[i]*(
                torch.exp(-.5*sigma**2*k2)*(form*cosine).sum(0)).sum()
        self.counter_energy = energy+.5*self.slab_factor*self.counter_moment.square()-self.field*self.counter_moment
        self.counter_potential = -self.coulomb*self.counter_density+self.ramp(-self.slab_factor*self.counter_moment)

    def vacuum_level(self, state, num_sources):
        """The field-free vacuum potential above the slab.

        Below the dipole correction plane the potential of the charges is flat and, in the zero-mean origin of
        periodic DFT, equals -(K/2V) times the second moment of the planar charge density measured from the
        plane. The point multipoles, the counter charge sheet and the local amplitudes have that moment in
        closed form, so the level is exact and independent of where in the vacuum it would be read. The applied
        field's sawtooth, zero at the slab centre, is left out: the level is that of the charges alone.
        """
        L, V, sigma = self.length, self.volume, self.width
        u = (self.zn+.5*L[:, None]).remainder(L[:, None])*self.present
        width, length = torch.full_like(u, sigma), L[:, None]
        mass, first, second, edge = window_moments(u, width, length)
        m2 = (state[..., 0]*(second-sigma**2*mass)*self.present).sum(-1)
        if self.max_l > 0:
            # The planar density of the moment lm is width^l R_lm(normal) (-d/dz)^l of the Gaussian; by parts its
            # second moment on the window is (-1)^l times end terms and, for l <= 2, the lower moments.
            scales = sigma**self.normal.new_tensor(component_orders(self.max_l))
            planar = state[..., :self.num_moments]*(solid_harmonics(self.max_l, self.normal)*scales)[:, None, :]
            start, end = window_derivatives(u, width, length, self.max_l-1) if self.max_l > 1 else (None, None)
            for l in range(1, self.max_l+1):
                if l == 1:
                    integral = length.square()*edge-2*first
                elif l == 2:
                    integral = length.square()*end[1]-2*length*end[0]+2*mass
                else:
                    integral = length.square()*end[l-1]-2*length*end[l-2]+2*(end[l-3]-start[l-3])
                m2 = m2+((-1)**l*integral*planar[..., l*l:(l+1)**2].sum(-1)*self.present).sum(-1)
        u_cc = (((self.counter_center-self.origin)*self.normal).sum(-1)+.5*L).remainder(L)
        m2 = m2+self.counter_charge*window_moments(u_cc, self.counter_width, L)[2]
        amplitudes = state[..., self.num_moments:self.num_moments+num_sources].sum((-1, -2))
        top = -FIELD_CONSTANT/(2*V)*m2-amplitudes*(2*math.pi)**1.5*sigma**3/V+self.vacuum_reference
        return top*self.slab

    def expand(self, value):
        output = value.new_zeros((len(self.counts), self.k2.shape[1], value.shape[-1]))
        return output.scatter_add(1, self.selected_indices[..., None].expand_as(value),
                                  value*self.selected_mask[..., None])

    def multipole_terms(self):
        """The Hessian of the multipoles and their drive by the counter charge and the uniform field of the slab,
        flattened as (component, atom). Both depend on the geometry alone and are made once."""
        if not hasattr(self, 'hessian'):
            self.hessian = moment_kernels(self)
            slope = self.field-self.slab_factor*self.counter_moment
            linear = (slope[:, None]*self.zn)[..., None]
            if self.max_l > 0:
                normal = (self.width*slope)[:, None]*self.normal
                linear = torch.cat((linear, normal[:, None, :].expand(-1, self.size, -1)), -1)
            drive = moment_projection(self.counter_coulomb, self)+F.pad(linear, (0, self.num_moments-linear.shape[-1]))
            self.drive = (drive*self.present[..., None]).transpose(1, 2).flatten(1)
        return self.hessian, self.drive


def coefficient_kernels(space, source_volume):
    """Fourier kernels of the state: Gaussian multipoles to max_l, then the local amplitudes to completion_max_l.

    The multipole lm of an atom is the density width^l R_lm(-grad) of its Gaussian charge, of modes
    (-i width)^l R_lm(k) times the Gaussian; the amplitude lm of a source width is the potential width^l
    R_lm(grad) of a Gaussian of that width, of modes (i width)^l R_lm(k) times it. R_lm are the solid harmonics.
    """
    sigma, count = space.width, space.num_moments
    scales = space.k.new_tensor([sigma**l for l in component_orders(max(space.max_l, space.completion_max_l))])
    shapes = space.harmonics[..., :len(scales)]*scales
    re, im = phase_factors(space.max_l, -1., space.k)
    multipoles = (space.smooth*space.poisson)[..., None]*shapes[..., :count]
    real, imag = [multipoles*re], [multipoles*im]
    re, im = phase_factors(space.completion_max_l, 1., space.k)
    for l in range(space.completion_max_l+1):
        block = slice(l*l, (l+1)**2)
        amplitudes = (space.radial*source_volume)[..., None]*shapes[..., None, block]
        real.append((amplitudes*re[block]).flatten(-2))
        imag.append((amplitudes*im[block]).flatten(-2))
    ivol = space.volume.reciprocal()[:, None, None]
    return torch.cat(real, -1)*ivol, torch.cat(imag, -1)*ivol


def coefficient_contributions(state, space):
    """Potential modes of the multipoles and of the local amplitudes, as (real, imag)."""
    cosine, sine = torch.bmm(space.co, state), torch.bmm(space.si, state)
    kr, ki = space.kernels
    real = kr*cosine+ki*sine
    imag = ki*cosine-kr*sine
    count = space.num_moments
    return (torch.stack((real[..., :count].sum(-1), imag[..., :count].sum(-1)), -1),
            torch.stack((real[..., count:].sum(-1), imag[..., count:].sum(-1)), -1))


def amplitude_spectrum(amplitudes, space):
    """Potential modes of the local amplitudes, as (real, imag)."""
    kr, ki = (kernel[..., space.num_moments:] for kernel in space.kernels)
    cosine, sine = torch.bmm(space.co, amplitudes), torch.bmm(space.si, amplitudes)
    return torch.stack(((kr*cosine+ki*sine).sum(-1), (ki*cosine-kr*sine).sum(-1)), -1)


def normal_dipole(state, space):
    """The dipole of the multipoles of each graph along the normal of its slab, from the dipole correction plane."""
    dipole = state[..., 0]*space.zn
    if space.max_l > 0:
        dipole = dipole+(state[..., 1:4]*space.width*space.normal[:, None]).sum(-1)
    return dipole.sum(-1)


def total_spectrum(state, space):
    """Potential modes of the state with its slab dipole field, and the fixed field."""
    cosine, sine = torch.bmm(space.co, state), torch.bmm(space.si, state)
    real, imag = space.kernels
    internal = torch.stack(((real*cosine+imag*sine).sum(-1), (imag*cosine-real*sine).sum(-1)), -1)
    return internal+normal_dipole(state, space)[:, None, None]*space.slab_ramp+space.fixed


def sample_spectrum(total, space, order, receivers=slice(None)):
    """R_lm(-grad) phi to order at every atom, averaged over a Gaussian receiver of each width.

    l = 0 is phi (V) and l = 1 is -grad phi (V/A); the result is (graphs, atoms, receivers, components).
    """
    count = (order+1)**2
    re, im = phase_factors(order, -1., space.k)
    shape = space.receiver[..., receivers, None]*space.harmonics[..., None, :count]
    a, b = shape*re, -shape*im
    vr, vi = total[..., 0, None, None], total[..., 1, None, None]
    sampled = (torch.bmm(space.co.transpose(1, 2), (vr*a+vi*b).flatten(-2))
               + torch.bmm(space.si.transpose(1, 2), (vr*b-vi*a).flatten(-2)))
    return sampled.reshape(*sampled.shape[:2], shape.shape[-2], count)


def moment_projection(total, space):
    """The energy per multipole coefficient of the potential modes total, width^l R_lm(grad) phi on the density."""
    scales = space.k.new_tensor([(-space.width)**l for l in component_orders(space.max_l)])
    return sample_spectrum(total, space, space.max_l, slice(0, 1))[..., 0, :]*scales


def moment_kernels(space):
    """The Hessian of the electrostatic energy in the multipole coefficients, rows and columns (component, atom).

    The modes (-i width)^l R_lm(k) exp(-ik.r) of the multipoles enter as a Gram matrix over the half space of
    the modes, counted twice since the sum is even in k; the slab field of the normal dipole is rank one.
    """
    half, count, size = space.half_count, space.num_moments, space.size
    re, im = phase_factors(space.max_l, -1., space.k)
    scales = space.k.new_tensor([space.width**l for l in component_orders(space.max_l)])
    shape = (space.harmonics[:, :half, :count]*scales)[..., None]
    co, si = space.co[:, :half, None, :], space.si[:, :half, None, :]
    modes = torch.cat((shape*(re[:, None]*co+im[:, None]*si), shape*(im[:, None]*co-re[:, None]*si)), 1).flatten(2)
    weight = (2*space.half_mask*space.smooth[:, :half]*space.receiver[:, :half, 0]*-space.poisson[:, :half]
              / space.volume[:, None])
    matrix = modes.transpose(1, 2)@(torch.cat((weight, weight), 1)[..., None]*modes)
    dipole = [space.zn[:, None, :]]
    if space.max_l > 0:
        dipole.append((space.width*space.normal)[..., None].expand(-1, -1, size))
    dipole = F.pad(torch.cat(dipole, 1), (0, 0, 0, count-min(count, 4))).flatten(1)
    matrix = matrix+space.slab_factor[:, None, None]*dipole[..., None]*dipole[:, None, :]
    present = space.present[:, None, :].expand(-1, count, -1).flatten(1)
    return matrix*present[..., None]*present[:, None, :]


def screen_moments(kernel, total, response, d0, size):
    """Eliminate the moments of order l >= 1, linear in their fields: the screened kernel and drive of the charges.

    The moments are h = d0 + R (t - K_hq q - K_hh h), with R the dd_dE of their order and t their drive. With
    h = d0 + D y and D = R^1/2, y solves S y = D (t - K_hh d0 - K_hq q), S = 1 + D K_hh D positive definite. One
    solve S [Y, z] = [D K_hq, D (t - K_hh d0)] gives the charges the kernel K_qq - (D K_hq)^T Y and the drive
    t_q - K_qh d0 - (D K_hq)^T z, and the moments d0 + D (z - Y q). The LU of torch.linalg.solve stays on
    cuSOLVER and cuBLAS for every batch: batched triangular solves, also inside the derivative of a Cholesky
    factor, go to MAGMA for large matrices, which fails on some GPUs.
    """
    k_qq, t_q = kernel[:, :size, :size], total[:, :size]
    if kernel.shape[-1] == size:
        return k_qq, t_q, None
    k_qh, k_hq, k_hh = kernel[:, :size, size:], kernel[:, size:, :size], kernel[:, size:, size:]
    # The atoms padding a graph have no response, and no derivative of its root.
    present = response > 0
    root = torch.where(present, response, 1.).sqrt()*present
    system = root[..., None]*k_hh*root[:, None, :]+torch.eye(k_hh.shape[-1], dtype=kernel.dtype, device=kernel.device)
    coupling = root[..., None]*k_hq
    drive = root*(total[:, size:]-(k_hh@d0[..., None]).squeeze(-1))
    solved = torch.linalg.solve(system, torch.cat((coupling, drive[..., None]), -1))
    y, z = solved[..., :size], solved[..., size]
    t_q = t_q-(k_qh@d0[..., None]).squeeze(-1)-(coupling.transpose(1, 2)@z[..., None]).squeeze(-1)
    return k_qq-coupling.transpose(1, 2)@y, t_q, (root, y, z)


def screened_moments(charges, d0, screening):
    """The moments of order l >= 1 at the charges, from the solve of screen_moments."""
    root, y, z = screening
    return d0+root*(z-(y@charges[..., None]).squeeze(-1))


def upstream_basis(moments, max_l):
    """The multipoles in the basis of MACE-SCF, the solid harmonics of (y, z, x)."""
    # On the CPU, so that no linear algebra of e3nn runs on the GPU.
    cycle = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=moments.dtype)
    rotation = torch.block_diag(*[o3.Irrep(l, 1).D_from_matrix(cycle) for l in range(max_l+1)])
    return moments@rotation.T.to(moments.device)


def solve_charges(effective, drive, fixed, slope, bias, target, mask):
    """The charges of the responses linearized as fixed + slope (bias + phi - mu), at total charge target."""
    operator = torch.eye(mask.shape[-1], dtype=effective.dtype, device=effective.device)+slope[..., None]*effective
    solved = torch.linalg.solve(operator, torch.stack((fixed+slope*(bias+drive), slope), -1))
    mu = ((solved[..., 0]*mask).sum(-1)-target)/(solved[..., 1]*mask).sum(-1)
    return (solved[..., 0]-mu[:, None]*solved[..., 1])*mask, mu


class PotentialResponse(ScaleShiftMACE):
    """A MACE whose atoms carry the response of MACE-VOLT, minimized at fixed Q, counter charge and field."""

    newton_steps, newton_tolerance = 60, 1.e-6

    def __init__(self, density_width=1.5, field_widths=(1.5, 3.), kspace_cutoff_factor=1., fermi_level_offset=0.,
                 fixed_point=False, short_range_potential=True, energy_range=6., resolution=.25,
                 band_edge_charge=.05, max_l=1, completion_max_l=None, **kwargs):
        super().__init__(keep_last_layer_irreps=True, **kwargs)
        irreps = o3.Irreps(str(self.products[0].linear.irreps_out))
        if {ir for _, ir in irreps} != {o3.Irrep('0e'), o3.Irrep('1o')}:
            raise ValueError('PotentialResponse needs scalar and vector features in every layer, e.g. 64x0e + 64x1o')
        self.feature_slices = [(sl.start, sl.stop, mul, ir.l) for (mul, ir), sl in zip(irreps, irreps.slices())]
        self.layout = 'mul_ir'
        self.density_width = float(density_width)
        # The order of the atomic multipoles (atomic_multipoles_max_l) and of the local amplitudes, by default the same.
        self.max_l = int(max_l)
        self.completion_max_l = self.max_l if completion_max_l is None else int(completion_max_l)
        # The local amplitudes: Gaussian potentials of the density width and twice it on every atom.
        sources = [self.density_width, 2*self.density_width] if short_range_potential else []
        receivers = [self.density_width]+sorted(set(float(w) for w in field_widths)-{self.density_width})
        self.register_buffer('source_widths', torch.tensor(sources))
        self.register_buffer('receiver_widths', torch.tensor(receivers))
        self.register_buffer('kspace_cutoff', torch.tensor(kspace_cutoff_factor*gto_basis_kspace_cutoff(
            [self.density_width]+receivers, max(self.max_l, self.completion_max_l, 1))))
        self.register_buffer('fermi_level_offset', torch.tensor(float(fermi_level_offset)))
        num_elements = len(self.atomic_numbers)
        self.register_buffer('vacuum_reference_integrals', torch.zeros(num_elements))
        self.register_buffer('neutral_atom_form_factors', torch.zeros(num_elements, len(FORM_FACTOR_GRID)))
        self.fixed_point = bool(fixed_point)
        # The hole and electron charge (e per cell) at which the band edges are read.
        self.band_edge_charge = float(band_edge_charge)
        layers = int(self.num_interactions)
        self.response = AtomicResponse(irreps.count('0e'), irreps.count('1o'), layers, num_elements, len(sources),
                                       len(receivers), fixed_point=self.fixed_point, energy_range=energy_range,
                                       resolution=resolution, max_l=self.max_l,
                                       completion_max_l=self.completion_max_l)

    def __setstate__(self, state):
        # Models saved before the renaming hold the atomic response as law.
        if 'law' in state.get('_modules', {}):
            state['_modules'] = type(state['_modules'])(('response' if k == 'law' else k, v)
                                                        for k, v in state['_modules'].items())
        super().__setstate__(state)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before the renaming hold the atomic response as law.
        for key in [k for k in state_dict if k.startswith(prefix+'law.')]:
            state_dict[prefix+'response.'+key[len(prefix)+4:]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __getstate__(self):
        # The replay head and its frames serve training only; a saved model is MACE-VOLT alone.
        state = self.__dict__.copy()
        state.pop('replay', None)
        state['_modules'] = type(state['_modules'])((k, v) for k, v in state['_modules'].items() if k != 'replay_head')
        return state

    def node_features(self, node_feats):
        """Scalars and Cartesian vectors of one layer, from either feature layout."""
        scalars, vectors = [], []
        for start, stop, mul, l in self.feature_slices:
            block = node_feats[:, start:stop]
            if l == 0:
                scalars.append(block)
            elif self.layout == 'ir_mul':
                vectors.append(block.reshape(-1, 3, mul).transpose(-1, -2))
            else:
                vectors.append(block.reshape(-1, mul, 3))
        return torch.cat(scalars, -1), torch.cat(vectors, -2)

    def backbone(self, data, ctx):
        """The local energies of every layer and the scalar and vector features of all layers."""
        node_attrs = data['node_attrs']
        if hasattr(self, 'foundation_element_map'):
            node_attrs = node_attrs@self.foundation_element_map
            numbers = self.foundation_atomic_numbers
        else:
            numbers = self.atomic_numbers
        num_atoms_arange = torch.arange(len(node_attrs), device=node_attrs.device)
        node_heads = torch.zeros_like(num_atoms_arange)
        node_feats = self.node_embedding(node_attrs)
        edge_attrs = self.spherical_harmonics(ctx.vectors)
        edge_feats, cutoff = self.radial_embedding(ctx.lengths, node_attrs, data['edge_index'], numbers)
        node_feats_list = []
        for i, (interaction, product) in enumerate(zip(self.interactions, self.products)):
            node_feats, sc = interaction(node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs,
                                         edge_feats=edge_feats, edge_index=data['edge_index'], cutoff=cutoff,
                                         first_layer=(i == 0))
            node_feats_list.append(product(node_feats=node_feats, sc=sc, node_attrs=node_attrs))
            node_feats = node_feats_list[-1]
        energies = [readout(node_feats_list[-1 if len(self.readouts) == 1 else i], node_heads)[num_atoms_arange, node_heads]
                    for i, readout in enumerate(self.readouts)]
        node_energy = self.scale_shift(torch.stack(energies).sum(0), node_heads)
        node_energy = node_energy+self.atomic_energies_fn(data['node_attrs'])[num_atoms_arange, node_heads]
        scalars, vectors = zip(*[self.node_features(feats) for feats in node_feats_list])
        return node_energy, torch.cat(scalars, -1), torch.cat(vectors, -2)

    def minimum(self, response, space, charge, start=None):
        """The exact minimum of the responses at total charge charge, from the charges and Fermi level start.

        Newton's method on the concave dual D(p, mu) = -p.Kp/2 - mu Q - sum_i A_i(b_i + v_i - (Kp)_i - mu),
        A_i the integral of the charge of atom i, whose maximum has p = q: each step solves the responses
        linearized at the current levels and a backtracking line search on D makes every step an ascent, so the
        solve converges for every response. One differentiable solve at the converged levels gives the exact
        derivatives. The moments of order l >= 1, linear in their fields, are eliminated first (screen_moments).
        """
        bias, dd_dE, coefficients, d0, amplitudes = response
        mask, size, count = space.present, space.size, space.num_moments
        kernel, drive = space.multipole_terms()
        # The potential of the local amplitudes on the multipoles; with the drive, the field of each moment.
        applied = moment_projection(amplitude_spectrum(amplitudes, space), space).transpose(1, 2).flatten(1)
        orders = torch.tensor(component_orders(space.max_l)[1:], dtype=torch.long, device=mask.device)-1
        dd_dE, d0 = dd_dE[..., orders].transpose(1, 2).flatten(1), d0.transpose(1, 2).flatten(1)
        effective, drive_q, screening = screen_moments(kernel, applied+drive, dd_dE, d0, size)
        drive_q = drive_q*mask
        base = (bias+drive_q)*mask

        def dual(p, mu, kp):
            delta = (base-kp-mu[:, None])*mask
            return -.5*(p*kp).sum(-1)-mu*charge-(self.response.energy(coefficients, delta)*mask).sum(-1)

        def linearized(delta):
            dq, n = self.response.charge(coefficients, delta)
            return solve_charges(effective, drive_q, (dq-n*delta)*mask, n*mask, bias, charge, mask)
        with torch.no_grad():
            # From the responses linearized at the Fermi level, unless a start is given.
            p, mu = linearized(torch.zeros_like(mask)) if start is None else start
            p, mu = p.detach().clone(), mu.detach().clone()
            kp = (effective@p[..., None]).squeeze(-1)
            for steps in range(1, self.newton_steps+1):
                delta = (base-kp-mu[:, None])*mask
                dq = self.response.charge(coefficients, delta)[0]*mask
                target, target_mu = linearized(delta)
                dp, dmu = target-p, target_mu-mu
                kdp = (effective@dp[..., None]).squeeze(-1)
                slope = ((dq-p)*kdp).sum(-1)+(dq.sum(-1)-charge)*dmu
                value, t = dual(p, mu, kp), torch.ones_like(mu)
                for _ in range(40):
                    trial = dual(p+t[:, None]*dp, mu+t*dmu, kp+t[:, None]*kdp)
                    accept = (trial >= value+1.e-4*t*slope) | (slope <= 1.e-12*(1.+value.abs()))
                    if bool(accept.all()):
                        break
                    t = torch.where(accept, t, .5*t)
                p, mu, kp = p+t[:, None]*dp, mu+t*dmu, kp+t[:, None]*kdp
                change = float((t[:, None]*(kdp+dmu[:, None])*mask).abs().max())
                if change < self.newton_tolerance:
                    break
        logging.debug(f'Newton solve of the levels: {steps} steps, last level change {change:.1e} V')
        if change > 1.e3*self.newton_tolerance:
            logging.warning(f'Newton solve of the levels stopped after {steps} steps with a level change of '
                            f'{change:.2e} V')
        delta = (base-kp-mu[:, None])*mask
        charges, mu = linearized(delta.detach())
        level = (bias+drive_q-(effective@charges[..., None]).squeeze(-1)-mu[:, None])*mask
        moments = charges if screening is None else torch.cat((charges, screened_moments(charges, d0, screening)), -1)
        field = applied+drive-(kernel@moments[..., None]).squeeze(-1)
        energy = ((charges*level-self.response.energy(coefficients, level)-bias*charges).sum(-1)
                  + .5*((moments[:, size:]-d0)*field[:, size:]).sum(-1)-(moments*applied).sum(-1))
        state = torch.cat((moments.reshape(len(mask), count, size).transpose(1, 2), amplitudes), -1)
        return state, energy, mu, (charges.detach(), mu.detach())

    def solve(self, data, space, positions, scalars, vectors, charge):
        """The minima of the energy: the state of the multipoles and amplitudes, the local energy and mu of each.

        A fixed point has two, the minimum with the response of the geometry and the final one with the response
        read there.
        """
        mask = space.present
        chemical, vector = self.response.features(space.pack(scalars), space.pack(vectors))
        response = self.response(chemical, vector, space.attrs, mask)
        state, energy, mu, start = self.minimum(response, space, charge)
        minima = [(state, energy, mu)]
        if self.fixed_point:
            transport = neighbour_transport(positions, data['edge_index'], data['shifts'], space.batch, space.slot,
                                            len(space.counts), space.size, float(self.r_max),
                                            float(self.receiver_widths.max()))
            sampled = sample_spectrum(total_spectrum(state, space), space, 1)
            potential, sampled = sampled[..., 0], sampled[..., 1:]
            levels = (response[0][..., None]+potential-mu[:, None, None])*mask[..., None]
            c = levels.shape[-1]
            messages = torch.bmm(transport, torch.cat((levels, sampled.flatten(-2)), -1))
            messages = messages.reshape(len(space.counts), 4, space.size, 4*c)
            radial, directional = messages[:, 0], messages[:, 1:]
            divergence = directional[..., c:].reshape(len(space.counts), 3, space.size, c, 3).diagonal(
                dim1=1, dim2=4).sum(-1)
            read = (torch.cat((levels, radial[..., :c], divergence), -1),
                    torch.cat((sampled, radial[..., c:].reshape_as(sampled), directional[..., :c].permute(0, 2, 3, 1)),
                              -2))
            response = self.response(chemical, vector, space.attrs, mask, state=read)
            minima.append(self.minimum(response, space, charge, start=start)[:3])
        return minima

    def forward(self, data, training=False, compute_force=True, compute_virials=False, compute_stress=False,
                band_edges=False, **kwargs):
        if compute_virials or compute_stress:
            raise NotImplementedError('PotentialResponse computes no stress')
        if data['positions'].is_cuda and torch.backends.cuda.preferred_linalg_library().name != 'Cusolver':
            torch.backends.cuda.preferred_linalg_library('cusolver')
        with torch.enable_grad():
            data = dict(data)
            charge = data['total_charge'].reshape(-1).clone().requires_grad_(True)
            counter = data['counter_charge'].reshape(-1).clone().requires_grad_(True)
            data['counter_charge'] = counter
            ctx = prepare_graph(data)
            positions = ctx.positions
            node_energy, scalars, vectors = self.backbone(data, ctx)
            space = FourierSpace(self, data, positions)
            minima = self.solve(data, space, positions, scalars, vectors, charge.to(space.volume))
            state, response_energy, _ = minima[-1]
            electrostatic, observed = self.observe(space, state, positions, charge)
            energy = scatter_sum(node_energy, data['batch'], dim=0, dim_size=len(space.counts))
            energy = energy+electrostatic+response_energy
            inputs = [charge, counter, positions] if compute_force else [charge, counter]
            gradients = torch.autograd.grad(energy.sum(), inputs, create_graph=training, allow_unused=True)
        forces = None
        if compute_force:
            forces = -gradients[2] if gradients[2] is not None else torch.zeros_like(positions)
        fermi_level = -gradients[0]
        # dE/dQ along Q_cc = -Q, the voltage between the electrode and the counter charge
        cell_voltage = -fermi_level-(gradients[1] if gradients[1] is not None else torch.zeros_like(fermi_level))
        output = {
            'energy': energy,
            'forces': forces,
            'virials': None,
            'stress': None,
            **observed,
            'fermi_level': fermi_level,
            'cell_voltage': cell_voltage,
            'k_vectors': space.wave,
            'k_vectors_mask': space.mask,
            'k_vectors_grid_shape': torch.tensor(space.shape, device=positions.device),
        }
        for key in ('fourier_density', 'fourier_potential'):
            if data.get(key) is not None and data[key].numel():
                output[key+'_dft'] = collate_fft(data[key], data[key+'_shape'], space.shape).to(positions)
                output[key+'_dft_mask'] = (space.modes[None].abs() <= torch.div(
                    data[key+'_shape'].reshape(-1, 3)-1, 2, rounding_mode='floor')[:, None]).all(-1)
        labelled = data.get('band_gap_weight')
        if band_edges or (labelled is not None and bool((labelled > 0).any())):
            # The band edges are Fermi levels of the same energy at a small hole and electron charge: the valence
            # band maximum at Q + delta and the conduction band minimum at Q - delta. With the response of the geometry
            # the multiplier of each minimum is -dE/dQ there exactly; a fixed point takes the multiplier of its
            # final minimum, with the response read at its own first minimum.
            delta = getattr(self, 'band_edge_charge', .05)
            edges = [self.solve(data, space, positions, scalars, vectors, charge.to(space.volume)+s*delta)[-1][2]
                     + self.fermi_level_offset+space.vacuum_reference for s in (1., -1.)]
            output.update(valence_band_maximum=edges[0], conduction_band_minimum=edges[1], band_gap=edges[1]-edges[0])
        if training and getattr(self, 'replay', None) is not None:
            output['replay'] = self.replay.loss(self.replay_head)
        return output

    def observe(self, space, state, positions, charge):
        """The electrostatic energy of a state, and its multipoles, dipole, vacuum potential and Fourier fields.

        The raw Fourier density and potential: the Gaussian smoothing of the charges and amplitudes is undone,
        the counter charge keeps its width and the boundary sawtooth is sharp.
        """
        sigma, count = self.density_width, space.num_moments
        moments = state[..., :count]*state.new_tensor([sigma**l for l in component_orders(space.max_l)])
        carrier, regular = (space.expand(x) for x in coefficient_contributions(state, space))
        total_charge = state[..., 0].sum(-1)
        coarse = carrier/torch.where(space.coulomb > 0, -space.coulomb, 1.)[..., None]
        coarse = coarse+torch.stack(((space.k2 == 0)*(total_charge/space.volume)[:, None], torch.zeros_like(space.k2)),
                                    -1)
        dipole = moments[..., :1]*(space.pack(positions)-space.origin[:, None])
        dipole = (dipole+moments[..., 1:4] if count > 1 else dipole).sum(1)
        normal_dipole = (dipole*space.normal).sum(-1)
        deformation = space.coulomb*torch.view_as_complex(coarse.contiguous())-torch.view_as_complex(
            regular.contiguous())
        counter_energy = (space.volume*(space.counter_density.conj()*deformation).real.sum(-1)
                          + space.slab_factor*normal_dipole*space.counter_moment+space.counter_energy)
        electrostatic = (.5*space.volume*(coarse.square().sum(-1)*space.coulomb).sum(-1)
                         + .5*space.slab_factor*normal_dipole.square()-(dipole*space.external).sum(-1)
                         + counter_energy-(self.fermi_level_offset+space.vacuum_reference)*charge)
        smooth = torch.where(space.mask, torch.exp(-.5*sigma**2*space.k2), 1.)
        boundary = space.field-space.slab_factor*(normal_dipole+space.counter_moment)
        potential = (torch.view_as_complex((carrier+regular).contiguous())/smooth-space.coulomb*space.counter_density
                     + space.ramp(boundary, width=0.))
        return electrostatic, {
            'density_coefficients': space.unpack(upstream_basis(moments, space.max_l)),
            'dipole': dipole,
            'vacuum_potential': space.vacuum_level(state, self.response.num_sources),
            'fourier_density': coarse/smooth[..., None]*space.ngrid,
            'fourier_potential': torch.view_as_real(potential)*space.ngrid,
        }


def collate_fft(values, shapes, grid):
    """The flattened Fourier labels of every graph on the common grid of the batch, as [..., (real, imag)]."""
    output, offset = [], 0
    for shape in shapes.reshape(-1, 3).tolist():
        size = math.prod(shape)
        modes = [torch.fft.fftfreq(n, d=1./n, device=values.device).round().long() for n in shape]
        value = values[offset:offset+size].reshape(shape)
        offset += size
        target = values.new_zeros(grid)
        index = [m % g for m, g in zip(modes, grid)]
        keep = [(m.abs() <= (g-1)//2) for m, g in zip(modes, grid)]
        target[torch.meshgrid(*[i[k] for i, k in zip(index, keep)], indexing='ij')] = value[torch.meshgrid(
            *[torch.nonzero(k)[:, 0] for k in keep], indexing='ij')]*(math.prod(grid)/size)
        output.append(target.reshape(-1))
    output = torch.stack(output)
    return torch.stack((output.real, output.imag), -1)


def fit_neutral_atoms(frames, knots=64, max_frames=32):
    """P_Z(k) of every element from the proto potentials of frames (cell, positions, numbers, proto modes).

    The proto potential is the potential of the superposed neutral atoms, a sum of spherical atoms. One
    linear fit gives P_Z(k) on hats in k^2; P_Z(0) is the integral of the atom's potential.
    """
    chosen, seen = [], {}
    for cell, positions, numbers, proto in frames:
        numbers = np.asarray(numbers, int)
        if len(chosen) < max_frames or any(seen.get(int(z), 0) < 4 for z in numbers):
            chosen.append((np.asarray(cell, float), np.asarray(positions, float), numbers, np.asarray(proto, float)))
            for z in set(numbers.tolist()):
                seen[z] = seen.get(z, 0)+1
    if not chosen:
        raise ValueError('No frame carries fourier_proto_potential')
    elements = sorted(seen)
    bandwidth = min(np.pi*(n-1)/np.linalg.norm(a) for cell, *_, proto in chosen for n, a in zip(proto.shape[1:], cell))
    size = len(elements)*knots
    gram, rhs, square = np.zeros((size, size)), np.zeros(size), 0.
    for cell, positions, numbers, proto in chosen:
        shape = proto.shape[1:]
        modes = np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(n)*n) for n in shape], indexing='ij'), -1).reshape(-1, 3)
        waves = 2*np.pi*modes@np.linalg.inv(cell).T
        k2 = np.square(waves).sum(-1)
        use = (k2 > 0) & (k2 <= bandwidth**2)
        values = (proto[0]+1j*proto[1]).reshape(-1)[use]/np.prod(shape)
        structure = np.stack([np.exp(-1j*waves[use]@positions[numbers == z].T).sum(-1) for z in elements], -1)
        radial = hats(k2[use], bandwidth, knots)
        design = (structure[:, :, None]*radial[:, None, :]).reshape(len(values), -1)/abs(np.linalg.det(cell))
        a, b = np.concatenate([design.real, design.imag]), np.concatenate([values.real, values.imag])
        gram += a.T@a
        rhs += a.T@b
        square += b@b
    curvature = np.diff(np.eye(knots), 2, axis=0)
    weight = 1.e-3*np.sqrt(np.diag(gram)).mean()
    solution = np.linalg.solve(gram+weight**2*np.kron(np.eye(len(elements)), curvature.T@curvature), rhs)
    residual = float(np.sqrt(max(square-2*solution@rhs+solution@gram@solution, 0.)/square))
    if residual > 1.e-2:
        raise ValueError(f'fourier_proto_potential does not match the atoms of its frames (residual {residual:.2g})')
    inside = FORM_FACTOR_GRID <= bandwidth
    radial = hats(FORM_FACTOR_GRID[inside]**2, bandwidth, knots)
    edge, integrals, rows = FORM_FACTOR_GRID[inside][-1], {}, {}
    for row, z in enumerate(elements):
        atom = radial@solution[row*knots:(row+1)*knots]
        integrals[z] = float(atom[0])
        rows[z] = np.concatenate([atom, atom[-1]*(edge/FORM_FACTOR_GRID[~inside])**2])/FIELD_CONSTANT
    logging.info(f'Neutral atoms of {elements} from {len(chosen)} proto potentials (residual {residual:.1e}), '
                 f'potential integrals {[round(integrals[z], 2) for z in elements]} eV A^3')
    return integrals, rows


def hats(k2, cutoff, knots):
    coordinate = np.clip(k2/cutoff**2, 0., 1.)*(knots-1)
    left = np.floor(coordinate).astype(int)
    fraction = coordinate-left
    basis = np.zeros((len(k2), knots))
    np.add.at(basis, (np.arange(len(k2)), left), 1-fraction)
    np.add.at(basis, (np.arange(len(k2)), np.minimum(left+1, knots-1)), fraction)
    return basis


@torch.no_grad()
def initialize_potential_response(model, args, train_loader, device):
    """The neutral atoms of the reference calculations and the feature units of the response, from training data."""
    key = next(iter(args.heads.values()))['info_keys'].get('fourier_proto_potential')
    if key is not None:
        frames = []
        for atoms in ase.io.iread(args.train_file, index=':'):
            if key in atoms.info:
                frames.append((atoms.cell.array, atoms.positions, atoms.numbers, atoms.info[key]))
            if len(frames) >= 64:
                break
        integrals, rows = fit_neutral_atoms(frames)
        for i, z in enumerate(model.atomic_numbers.tolist()):
            model.vacuum_reference_integrals[i] = -integrals.get(z, 0.)
            if z in rows:
                model.neutral_atom_form_factors[i] = torch.as_tensor(rows[z])
    scalar, vector, count = (torch.zeros_like(model.response.scalar_unit),
                             torch.zeros_like(model.response.vector_unit), 0.)
    for i, batch in enumerate(train_loader):
        if i*train_loader.batch_size >= 64:
            break
        data = batch.to(device).to_dict()
        with torch.enable_grad():
            ctx = prepare_graph(data)
            _, scalars, vectors = model.backbone(data, ctx)
        scalars, vectors = model.response.mix(scalars, vectors)
        scalar += torch.asinh(scalars).square().sum(0)
        vector += vectors.square().mean(-1).sum(0)
        count += len(scalars)
    model.response.scalar_unit.copy_((1.+scalar/count).sqrt())
    model.response.vector_unit.copy_((1.+vector/count).sqrt())


# The smearing widths (A) at which validation reports the Fourier errors: raw, the density width and the far field.
REPORTED_SMEARING_WIDTHS = {'fourier_density': (0., 1.5, 3.), 'fourier_potential': (0., 1.5)}


def spectral_errors(ref, pred, key, width=0.):
    """Parseval mean square of a Fourier prediction minus its label, both smeared by a Gaussian of width (A).

    Width zero compares the raw fields; potentials are compared modulo a constant.
    """
    weights = ref.weight*getattr(ref, key+'_weight')
    if pred.get(key+'_dft') is None:
        return weights*0.
    k2 = pred['k_vectors'].square().sum(-1)
    active = pred['k_vectors_mask'] & pred[key+'_dft_mask'] & (weights > 0)[:, None]
    if key == 'fourier_potential':
        active = active & (k2 > 0)
    difference = torch.where(active[..., None], (pred[key]-pred[key+'_dft'])*torch.exp(-.5*width**2*k2)[..., None], 0.)
    return weights*difference.square().sum((-1, -2))/pred['k_vectors_grid_shape'].prod().to(difference).square()


class FourierError:
    """Parseval mean square error of a Fourier field, summed over its smearing widths (A), zero for raw."""
    key, scale = '', 1.

    def __init__(self, smearing_widths):
        widths = ast.literal_eval(smearing_widths) if isinstance(smearing_widths, str) else smearing_widths
        self.smearing_widths = [float(w) for w in np.atleast_1d(widths)]
        if not self.smearing_widths or not all(0. <= w < math.inf for w in self.smearing_widths):
            raise ValueError(f'{self.key} smearing_widths must be widths >= 0 in A, not {smearing_widths!r}')

    def __call__(self, ref, pred):
        return torch.mean(sum(spectral_errors(ref, pred, self.key, w) for w in self.smearing_widths))/self.scale


class FourierDensityError(FourierError):
    # density, in (me/A^3)^2
    key, scale = 'fourier_density', 1.e-6

    def __init__(self, smearing_widths=(3.,)):
        super().__init__(smearing_widths)


class FourierPotentialError(FourierError):
    # deformation potential, in V^2
    key = 'fourier_potential'

    def __init__(self, smearing_widths=(0., 1.5)):
        super().__init__(smearing_widths)


def weighted_mean_squared_error_vacuum_potential(ref, pred):
    # vacuum_potential: [n_graphs, ]
    configs_weight = ref.weight*ref.vacuum_potential_weight
    return torch.mean(configs_weight*torch.square(ref['vacuum_potential']-pred['vacuum_potential']))


def weighted_mean_squared_error_band_gap(ref, pred):
    # band_gap: [n_graphs, ], predicted for batches with band gap labels only
    if pred.get('band_gap') is None:
        return ref.weight.new_zeros(())
    configs_weight = ref.weight*ref.band_gap_weight
    return torch.mean(configs_weight*torch.square(ref['band_gap']-pred['band_gap']))


def replay_loss(ref, pred):
    if 'replay' not in pred:
        return ref.weight.new_zeros(())
    return pred['replay']
