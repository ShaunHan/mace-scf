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


class LocalLaw(nn.Module):
    """The law of every atom: bias, density of states on an energy grid, dd_dE, d0 and local amplitudes.

    dd_dE times the density width squared is the dipole per field. The density of states is a sum of
    Gaussian bumps of width resolution on [-energy_range, energy_range] eV over a background of DOS_FLOOR,
    and starts at 1 e/V in the bands and 0.0025 e/V within 1 eV of the level. With fixed_point the law also
    reads the levels and fields of the atom and its neighbours; at zero levels and field that read is the law
    of the geometry.
    """

    def __init__(self, num_scalars, num_vectors, num_elements, num_sources, num_receivers, fixed_point=False,
                 energy_range=6., resolution=.5, width=64):
        super().__init__()
        self.vector_width = min(16, max(4, num_vectors))
        self.num_sources = num_sources
        self.chemical_scalar = nn.Linear(num_scalars, width)
        self.chemical_vector = nn.Linear(num_vectors, self.vector_width, bias=False)
        self.shared = nn.Sequential(nn.Linear(width+2*self.vector_width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.scalar_out = nn.Linear(width, 1+num_sources)
        self.vector_out = nn.Linear(width, 2*(1+num_sources)*self.vector_width)
        self.dd_dE_readout = nn.Linear(width, 1)
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

    def features(self, scalars, vectors):
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
        gate = self.vector_out(hidden).reshape(*hidden.shape[:-1], 1+self.num_sources, 2*self.vector_width)
        polar = torch.einsum('bnov,bnvc->bnoc', gate, torch.cat((vector, embedded), -2))/math.sqrt(2*self.vector_width)
        bias = (attrs@self.species_bias+scalars[..., 0])*mask
        dd_dE = positive(self.dd_dE_readout(hidden))[..., 0]*mask
        coefficients = F.softplus(self.dos_readout(hidden)+self.dos_offsets)*mask[..., None]
        amplitudes = torch.cat((attrs@self.species_source+scalars[..., 1:], polar[..., 1:, :].flatten(-2)), -1)
        return bias, dd_dE, coefficients, polar[..., 0, :]*mask[..., None], amplitudes*mask[..., None]

    def charge(self, coefficients, delta):
        """The charge of the law at the levels delta, and the density of states there."""
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


class Cell:
    """Fourier operators, counter charge and boundary terms of a batch, packed as (graphs, atoms)."""

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

        phase = torch.einsum('gkc,gnc->gkn', self.wave, self.pack(positions))
        co = phase.cos()*self.present[:, None, :]*active[..., None]
        si = phase.sin()*self.present[:, None, :]*active[..., None]
        smooth = torch.exp(-.5*self.width**2*self.k2)*active
        real = torch.cat((smooth[..., None], torch.zeros_like(self.wave)), -1)
        imag = torch.cat((torch.zeros_like(smooth[..., None]), -smooth[..., None]*self.wave), -1)
        radial = torch.exp(-.5*self.k2[..., None]*model.source_widths.to(positions).square())*active[..., None]
        receiver = torch.exp(-.5*self.k2[..., None]*model.receiver_widths.to(positions).square())*active[..., None]
        zn = ((self.pack(positions)-self.origin[:, None])*self.normal[:, None]).sum(-1)*self.present
        tensors = (co, si, self.wave, real, imag, radial, receiver, torch.view_as_real(self.ramp(-self.slab_factor)),
                   torch.view_as_real(self.ramp(self.field)), zn, self.normal, -self.coulomb, self.volume.reciprocal())
        # Only the nonzero modes inside the cutoff enter the moments and the sampled fields.
        count = int(active.sum(-1).max())
        indices = torch.arange(active.shape[1], device=positions.device).expand_as(active)
        indices = indices.masked_fill(~active, active.shape[1]).sort(-1).values[:, :count]
        self.selected_mask = indices < active.shape[1]
        self.selected_indices = indices.clamp_max(active.shape[1]-1)

        def selected(value):
            mask = self.selected_mask.reshape(*self.selected_mask.shape, *([1]*(value.ndim-2)))
            return value[rows[:, None], self.selected_indices]*mask
        self.tensors = tuple(selected(value) if i < 9 or i == 11 else value for i, value in enumerate(tensors))
        # The fixed field of the solve: the applied field and the counter charge.
        self.fixed = self.tensors[8]+selected(torch.view_as_real(self.counter_potential))
        self.counter_coulomb = selected(torch.view_as_real(-self.coulomb*self.counter_density))
        self.kernels = coefficient_kernels(self.tensors, self.width, (2*math.pi)**1.5*self.width**3)

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
        field's sawtooth, zero at the slab centre, is left out: the work function is that of the charges alone.
        """
        L, V, sigma = self.length, self.volume, self.width
        u = (self.tensors[9]+.5*L[:, None]).remainder(L[:, None])*self.present
        q = state[..., 0]*self.present
        d = (state[..., 1:4]*sigma*self.normal[:, None]).sum(-1)*self.present
        mass, first, second, edge = window_moments(u, torch.full_like(u, sigma), L[:, None])
        m2 = (q*(second-sigma**2*mass)+d*(2*first-L[:, None].square()*edge)).sum(-1)
        u_cc = (((self.counter_center-self.origin)*self.normal).sum(-1)+.5*L).remainder(L)
        m2 = m2+self.counter_charge*window_moments(u_cc, self.counter_width, L)[2]
        amplitudes = state[..., 4:4+num_sources].sum((-1, -2))
        top = -FIELD_CONSTANT/(2*V)*m2-amplitudes*(2*math.pi)**1.5*sigma**3/V+self.vacuum_reference
        return top*self.slab

    def expand(self, value):
        output = value.new_zeros((len(self.counts), self.k2.shape[1], value.shape[-1]))
        return output.scatter_add(1, self.selected_indices[..., None].expand_as(value),
                                  value*self.selected_mask[..., None])


def coefficient_kernels(tensors, width, source_volume):
    """Fourier kernels of the state: Gaussian charges and dipoles, then the local amplitudes."""
    _, _, wave, br, bi, radial, *_, poisson, ivol = tensors
    scales = br.new_tensor([1., width, width, width])
    shape_im = (radial[..., None]*wave[:, :, None, :]*(width*source_volume)).flatten(-2)
    real = torch.cat((br*poisson[..., None]*scales, radial*source_volume, torch.zeros_like(shape_im)), -1)
    imag = torch.cat((bi*poisson[..., None]*scales, torch.zeros_like(radial), shape_im), -1)
    return real*ivol[:, None, None], imag*ivol[:, None, None]


def coefficient_contributions(state, tensors, kernels):
    """Potential modes of the charges and dipoles and of the local amplitudes, as (real, imag)."""
    cosine, sine = torch.bmm(tensors[0], state), torch.bmm(tensors[1], state)
    kr, ki = kernels
    real = kr*cosine+ki*sine
    imag = ki*cosine-kr*sine
    return (torch.stack((real[..., :4].sum(-1), imag[..., :4].sum(-1)), -1),
            torch.stack((real[..., 4:].sum(-1), imag[..., 4:].sum(-1)), -1))


def total_spectrum(state, tensors, kernels, width, fixed):
    """Potential modes of the state with its slab dipole field, and the fixed field."""
    cosine, sine = torch.bmm(tensors[0], state), torch.bmm(tensors[1], state)
    real, imag = kernels
    internal = torch.stack(((real*cosine+imag*sine).sum(-1), (imag*cosine-real*sine).sum(-1)), -1)
    ramp, z, normal = tensors[7], tensors[9], tensors[10]
    dipole = (state[..., 0]*z+(state[..., 1:4]*width*normal[:, None]).sum(-1)).sum(-1)
    return internal+dipole[:, None, None]*ramp+fixed


def sample_spectrum(total, tensors):
    """Gaussian receiver averages of phi (V) and -grad phi (V/A) at every atom, one channel per width."""
    cosine, sine, wave, _, _, _, receiver, *_ = tensors
    re = total[..., 0, None]*receiver
    im = total[..., 1, None]*receiver
    re_k = (re[..., None]*wave[:, :, None]).flatten(-2)
    im_k = (im[..., None]*wave[:, :, None]).flatten(-2)
    sampled = (torch.bmm(cosine.transpose(1, 2), torch.cat((re, im_k), -1))
               + torch.bmm(sine.transpose(1, 2), torch.cat((-im, re_k), -1)))
    value, gradient = sampled.split((re.shape[-1], re_k.shape[-1]), -1)
    return value, gradient.reshape(*value.shape, 3)


def moment_kernels(tensors, kernels, mask, sigma, slab):
    """The Hessian of the electrostatic energy in the charges and dipoles, and its coupling to the amplitudes."""
    co, si, wave = tensors[:3]
    kr, ki = kernels
    source_re = (co[..., None]*kr[:, :, None, :]+si[..., None]*ki[:, :, None, :]).flatten(-2)
    source_im = (co[..., None]*ki[:, :, None, :]-si[..., None]*kr[:, :, None, :]).flatten(-2)
    receiver = tensors[6][..., 0, None, None]
    observer_re = torch.cat((co[..., None], -sigma*si[..., None]*wave[:, :, None, :]), -1)*receiver
    observer_im = torch.cat((-si[..., None], -sigma*co[..., None]*wave[:, :, None, :]), -1)*receiver
    matrix = -(observer_re.flatten(-2).transpose(1, 2)@source_re+observer_im.flatten(-2).transpose(1, 2)@source_im)
    matrix = matrix.reshape(*matrix.shape[:2], mask.shape[-1], kr.shape[-1])
    completion = -matrix[..., 4:].flatten(-2)
    # The slab field of the normal dipole is rank one.
    dipole = torch.cat((tensors[9][..., None], sigma*tensors[10][:, None, :].expand(-1, mask.shape[1], -1)),
                       -1).flatten(-2)
    matrix = matrix[..., :4].flatten(-2)+slab[:, None, None]*dipole[..., None]*dipole[:, None, :]
    present = mask[..., None].expand(-1, -1, 4).flatten(-2)
    return matrix*present[..., None]*present[:, None, :], completion


def screen_dipoles(kernel, completion, drive, dd_dE, d0, amplitudes, mask):
    """Eliminate the dipoles: the screened kernel of the charges, their drive and the dipoles as d = free - map q."""
    graphs, atoms = mask.shape
    index = torch.arange(4*atoms, device=mask.device).reshape(atoms, 4)
    charge_rows, dipole_rows = index[:, 0], index[:, 1:].reshape(-1)
    applied = (completion@amplitudes.flatten(1)[..., None]).squeeze(-1)
    total = applied+drive.flatten(1)
    k_qq, k_qd = kernel[:, charge_rows][:, :, charge_rows], kernel[:, charge_rows][:, :, dipole_rows]
    k_dq, k_dd = kernel[:, dipole_rows][:, :, charge_rows], kernel[:, dipole_rows][:, :, dipole_rows]
    response = (dd_dE[..., None].expand(-1, -1, 3)*mask[..., None]).flatten(1)
    operator = torch.eye(3*atoms, dtype=kernel.dtype, device=kernel.device)+response[..., None]*k_dd
    rhs = torch.cat((response[..., None]*k_dq, (d0.flatten(1)+response*total[:, dipole_rows])[..., None]), -1)
    solved = torch.linalg.solve(operator, rhs)
    free = solved[..., atoms]
    return (k_qq-k_qd@solved[..., :atoms], total[:, charge_rows]-(k_qd@free[..., None]).squeeze(-1),
            solved[..., :atoms], free, applied)


def solve_charges(effective, drive, fixed, slope, bias, target, mask):
    """The charges of the law linearized as fixed + slope (bias + phi - mu), at total charge target."""
    operator = torch.eye(mask.shape[-1], dtype=effective.dtype, device=effective.device)+slope[..., None]*effective
    solved = torch.linalg.solve(operator, torch.stack((fixed+slope*(bias+drive), slope), -1))
    mu = ((solved[..., 0]*mask).sum(-1)-target)/(solved[..., 1]*mask).sum(-1)
    return (solved[..., 0]-mu[:, None]*solved[..., 1])*mask, mu


class PotentialResponse(ScaleShiftMACE):
    """A MACE whose atoms carry the local law of MACE-VOLT, minimized at fixed Q, counter charge and field."""

    newton_steps, newton_tolerance = 60, 1.e-6

    def __init__(self, density_width=1.5, field_widths=(1.5, 3.), kspace_cutoff_factor=1., fermi_level_offset=0.,
                 fixed_point=False, short_range_potential=True, energy_range=6., resolution=.5, **kwargs):
        super().__init__(keep_last_layer_irreps=True, **kwargs)
        irreps = o3.Irreps(str(self.products[0].linear.irreps_out))
        if {ir for _, ir in irreps} != {o3.Irrep('0e'), o3.Irrep('1o')}:
            raise ValueError('PotentialResponse needs scalar and vector features in every layer, e.g. 64x0e + 64x1o')
        self.feature_slices = [(sl.start, sl.stop, mul, ir.l) for (mul, ir), sl in zip(irreps, irreps.slices())]
        self.layout = 'mul_ir'
        self.density_width = float(density_width)
        # The local amplitudes: Gaussian potentials of the density width and twice it on every atom.
        sources = [self.density_width, 2*self.density_width] if short_range_potential else []
        receivers = [self.density_width]+sorted(set(float(w) for w in field_widths)-{self.density_width})
        self.register_buffer('source_widths', torch.tensor(sources))
        self.register_buffer('receiver_widths', torch.tensor(receivers))
        self.register_buffer('kspace_cutoff', torch.tensor(kspace_cutoff_factor*gto_basis_kspace_cutoff(
            [self.density_width]+receivers, 1)))
        self.register_buffer('fermi_level_offset', torch.tensor(float(fermi_level_offset)))
        num_elements = len(self.atomic_numbers)
        self.register_buffer('vacuum_reference_integrals', torch.zeros(num_elements))
        self.register_buffer('neutral_atom_form_factors', torch.zeros(num_elements, len(FORM_FACTOR_GRID)))
        self.fixed_point = bool(fixed_point)
        layers = int(self.num_interactions)
        self.law = LocalLaw(layers*irreps.count('0e'), layers*irreps.count('1o'), num_elements, len(sources),
                            len(receivers), fixed_point=self.fixed_point, energy_range=energy_range,
                            resolution=resolution)

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

    def minimum(self, law, kernel, completion, drive, mask, charge, start=None):
        """The exact minimum of one law at total charge charge, from the charges and Fermi level start.

        Newton's method on the concave dual D(p, mu) = -p.Kp/2 - mu Q - sum_i A_i(b_i + v_i - (Kp)_i - mu),
        A_i the integral of the charge law, whose maximum has p = q: each step solves the law linearized at
        the current levels and a backtracking line search on D makes every step an ascent, so the solve
        converges for every law. One differentiable solve at the converged levels gives the exact derivatives.
        """
        bias, dd_dE, coefficients, d0, amplitudes = law
        effective, drive_q, dipole_map, free, applied = screen_dipoles(kernel, completion, drive, dd_dE, d0,
                                                                       amplitudes, mask)
        drive_q = drive_q*mask
        base = (bias+drive_q)*mask

        def dual(p, mu, kp):
            delta = (base-kp-mu[:, None])*mask
            return -.5*(p*kp).sum(-1)-mu*charge-(self.law.energy(coefficients, delta)*mask).sum(-1)

        def linearized(delta):
            dq, n = self.law.charge(coefficients, delta)
            return solve_charges(effective, drive_q, (dq-n*delta)*mask, n*mask, bias, charge, mask)
        with torch.no_grad():
            # From the law linearized at the Fermi level, unless a start is given.
            p, mu = linearized(torch.zeros_like(mask)) if start is None else start
            p, mu = p.detach().clone(), mu.detach().clone()
            kp = (effective@p[..., None]).squeeze(-1)
            for steps in range(1, self.newton_steps+1):
                delta = (base-kp-mu[:, None])*mask
                dq = self.law.charge(coefficients, delta)[0]*mask
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
        dipoles = free-(dipole_map@charges[..., None]).squeeze(-1)
        moments = torch.cat((charges[..., None], dipoles.reshape(*mask.shape, 3)), -1).flatten(1)
        total = (applied+drive.flatten(1)-(kernel@moments[..., None]).squeeze(-1)).reshape(*mask.shape, 4)*mask[..., None]
        state = torch.cat((moments.reshape(*mask.shape, 4), amplitudes), -1)
        induced = (dipoles.reshape(*mask.shape, 3)-d0)*mask[..., None]
        energy = ((charges*level-self.law.energy(coefficients, level)-bias*charges).sum(-1)
                  + .5*(induced*total[..., 1:4]).sum((-1, -2))-(moments*applied).sum(-1))
        return state, energy, mu, (charges.detach(), mu.detach())

    def solve(self, data, cell, positions, scalars, vectors, charge):
        """The state of the multipoles and amplitudes, the local energy of the law and the Fermi level."""
        mask, sigma = cell.present, self.density_width
        chemical, vector = self.law.features(cell.pack(scalars), cell.pack(vectors))
        law = self.law(chemical, vector, cell.attrs, mask)
        kernel, completion = moment_kernels(cell.tensors, cell.kernels, mask, sigma, cell.slab_factor)
        values, field = sample_spectrum(cell.counter_coulomb, cell.tensors)
        slope = cell.field-cell.slab_factor*cell.counter_moment
        values = values[..., 0]+slope[:, None]*cell.tensors[9]
        field = field[..., 0, :]-slope[:, None, None]*cell.normal[:, None, :]
        drive = torch.cat((values[..., None], -sigma*field), -1)*mask[..., None]
        state, energy, mu, start = self.minimum(law, kernel, completion, drive, mask, charge)
        if self.fixed_point:
            transport = neighbour_transport(positions, data['edge_index'], data['shifts'], cell.batch, cell.slot,
                                            len(cell.counts), cell.size, float(self.r_max),
                                            float(self.receiver_widths.max()))
            potential, sampled = sample_spectrum(total_spectrum(state, cell.tensors, cell.kernels, sigma, cell.fixed),
                                                 cell.tensors)
            levels = (law[0][..., None]+potential-mu[:, None, None])*mask[..., None]
            c = levels.shape[-1]
            messages = torch.bmm(transport, torch.cat((levels, sampled.flatten(-2)), -1))
            messages = messages.reshape(len(cell.counts), 4, cell.size, 4*c)
            radial, directional = messages[:, 0], messages[:, 1:]
            divergence = directional[..., c:].reshape(len(cell.counts), 3, cell.size, c, 3).diagonal(
                dim1=1, dim2=4).sum(-1)
            read = (torch.cat((levels, radial[..., :c], divergence), -1),
                    torch.cat((sampled, radial[..., c:].reshape_as(sampled), directional[..., :c].permute(0, 2, 3, 1)),
                              -2))
            law = self.law(chemical, vector, cell.attrs, mask, state=read)
            state, energy, mu, _ = self.minimum(law, kernel, completion, drive, mask, charge, start=start)
        return state, energy, mu

    def forward(self, data, training=False, compute_force=True, compute_virials=False, compute_stress=False,
                **kwargs):
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
            cell = Cell(self, data, positions)
            state, response_energy, _ = self.solve(data, cell, positions, scalars, vectors, charge.to(cell.volume))
            sigma = self.density_width
            moments = torch.cat((state[..., :1], state[..., 1:4]*sigma), -1)
            carrier, regular = (cell.expand(x) for x in coefficient_contributions(state, cell.tensors, cell.kernels))
            total_charge = state[..., 0].sum(-1)
            coarse = carrier/torch.where(cell.coulomb > 0, -cell.coulomb, 1.)[..., None]
            coarse = coarse+torch.stack(((cell.k2 == 0)*(total_charge/cell.volume)[:, None], torch.zeros_like(cell.k2)),
                                        -1)
            dipole = (moments[..., :1]*(cell.pack(positions)-cell.origin[:, None])+moments[..., 1:4]).sum(1)
            normal_dipole = (dipole*cell.normal).sum(-1)
            deformation = cell.coulomb*torch.view_as_complex(coarse.contiguous())-torch.view_as_complex(
                regular.contiguous())
            counter_energy = (cell.volume*(cell.counter_density.conj()*deformation).real.sum(-1)
                              + cell.slab_factor*normal_dipole*cell.counter_moment+cell.counter_energy)
            electrostatic = (.5*cell.volume*(coarse.square().sum(-1)*cell.coulomb).sum(-1)
                             + .5*cell.slab_factor*normal_dipole.square()-(dipole*cell.external).sum(-1)
                             + counter_energy-(self.fermi_level_offset+cell.vacuum_reference)*charge)
            energy = scatter_sum(node_energy, data['batch'], dim=0, dim_size=len(cell.counts))
            energy = energy+electrostatic+response_energy
            inputs = [charge, counter, positions] if compute_force else [charge, counter]
            gradients = torch.autograd.grad(energy.sum(), inputs, create_graph=training, allow_unused=True)
        forces = None
        if compute_force:
            forces = -gradients[2] if gradients[2] is not None else torch.zeros_like(positions)
        fermi_level = -gradients[0]
        # dE/dQ along Q_cc = -Q, the voltage between the electrode and the counter charge
        cell_voltage = -fermi_level-(gradients[1] if gradients[1] is not None else torch.zeros_like(fermi_level))
        workfunction = cell.vacuum_level(state, self.law.num_sources)-fermi_level
        # The raw Fourier density and potential: the Gaussian smoothing of the charges and amplitudes is undone,
        # the counter charge keeps its width and the boundary sawtooth is sharp.
        smooth = torch.where(cell.mask, torch.exp(-.5*sigma**2*cell.k2), 1.)
        boundary = cell.field-cell.slab_factor*(normal_dipole+cell.counter_moment)
        potential = (torch.view_as_complex((carrier+regular).contiguous())/smooth-cell.coulomb*cell.counter_density
                     + cell.ramp(boundary, width=0.))
        output = {
            'energy': energy,
            'forces': forces,
            'virials': None,
            'stress': None,
            'density_coefficients': cell.unpack(moments)[..., [0, 2, 3, 1]],
            'dipole': dipole,
            'fermi_level': fermi_level,
            'workfunction': workfunction,
            'cell_voltage': cell_voltage,
            'fourier_density': coarse/smooth[..., None]*cell.ngrid,
            'fourier_potential': torch.view_as_real(potential)*cell.ngrid,
            'k_vectors': cell.wave,
            'k_vectors_mask': cell.mask,
            'k_vectors_grid_shape': torch.tensor(cell.shape, device=positions.device),
        }
        for key in ('fourier_density', 'fourier_potential'):
            if data.get(key) is not None and data[key].numel():
                output[key+'_dft'] = collate_fft(data[key], data[key+'_shape'], cell.shape).to(positions)
                output[key+'_dft_mask'] = (cell.modes[None].abs() <= torch.div(
                    data[key+'_shape'].reshape(-1, 3)-1, 2, rounding_mode='floor')[:, None]).all(-1)
        if training and getattr(self, 'replay', None) is not None:
            output['replay'] = self.replay.loss(self.replay_head)
        return output


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
    """The neutral atoms of the reference calculations and the feature units of the law, from training data."""
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
    scalar, vector, count = torch.zeros_like(model.law.scalar_unit), torch.zeros_like(model.law.vector_unit), 0.
    for i, batch in enumerate(train_loader):
        if i*train_loader.batch_size >= 64:
            break
        data = batch.to(device).to_dict()
        with torch.enable_grad():
            ctx = prepare_graph(data)
            _, scalars, vectors = model.backbone(data, ctx)
        scalar += torch.asinh(scalars).square().sum(0)
        vector += vectors.square().mean(-1).sum(0)
        count += len(scalars)
    model.law.scalar_unit.copy_((1.+scalar/count).sqrt())
    model.law.vector_unit.copy_((1.+vector/count).sqrt())


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


def weighted_mean_squared_error_workfunction(ref, pred):
    # workfunction: [n_graphs, ]
    configs_weight = ref.weight*ref.workfunction_weight
    return torch.mean(configs_weight*torch.square(ref['workfunction']-pred['workfunction']))


def replay_loss(ref, pred):
    if 'replay' not in pred:
        return ref.weight.new_zeros(())
    return pred['replay']
