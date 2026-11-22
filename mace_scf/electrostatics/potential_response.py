import ast
import logging
import math

import ase.io
from ase.data import chemical_symbols
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
# The smallest softness (e/V per atom) of an atom at every level, so that every charge has one finite level however
# small the learned softness of the atoms of a cell with a gap becomes. It is no state: the density of states and the
# band edges do not see it.
DOS_FLOOR = 1.e-5
# The density of states of every atom at the start (e/V) of an element without density of states labels, the same at
# every energy. It is of the order of 1/U, U = 5.4 V/e the Coulomb energy of a charge of the density width 1.5 A,
# where the states and the Coulomb energy resist a charge alike: far fewer states at the level make an insulator, far
# more a metal. With labels the packets start where the states of DFT are (start_packets).
DOS_START = .1
# The energies (eV) to either side of the neutral level of an atom where its packets start: evenly without labels,
# and over the states of DFT that far from its Fermi level with them. The band edges of an insulator lie about half
# its gap from the levels of its atoms, and the levels move by volts with the potential, so the range holds band
# gaps up to about 15 eV at the start; the packets then go where the labels put them.
ENERGY_RANGE = 10.
# The states of an atom are PACKETS packets, each with the same share of its states. The network places every
# packet and gives it its width, so the states have no resolution of their own: they are as sharp or as broad as
# the data make them, a gap is empty, and a band edge is the edge of a packet. The occupation is that of zero
# temperature: a state below the Fermi level is filled, one above it empty. No smearing width enters anywhere.
PACKETS = 32
# The softness of the charge of an atom (e/V) between its states, at the start: the charge an atom gives or takes
# per volt of its level beyond what its states hold. Electrons in an insulator move between atoms without a state at
# the Fermi level, over the bonds, and the charges of a cell with a gap meet its total charge through it. Without it
# an atom whose level lies in a gap has no response at all, and labels cannot move its charge: in tests the water
# films of IrO2-water slabs with adsorbed O2 then lost the holes they hold in DFT, 3 eV off in work function, with or
# without the states of DFT at the start. With it and that start they kept them, or lost them for a few hundred
# steps and took them back, at each of three seeds.
SOFTNESS_START = 1.e-3
# The energies (eV) at which the curves of the states are compared with their labels and reported: the points of
# the comparison, not a resolution of the states, which are exact at every energy.
DOS_SAMPLING = .1
# Gaussian tails below exp(-TAIL) are dropped from the exact sums of the counter charge.
TAIL = 36.
# Form factors of the neutral atoms, P_Z(k)/K on a uniform grid in k (1/A).
FORM_FACTOR_SPACING, FORM_FACTOR_MAXIMUM = .01, 40.
FORM_FACTOR_GRID = np.arange(0., FORM_FACTOR_MAXIMUM+.5*FORM_FACTOR_SPACING, FORM_FACTOR_SPACING)


def positive(x):
    return RESPONSE_SCALE*(F.softplus(x+math.log(math.expm1(1.)))+1.e-6)


# A packet of states at x half widths from its center: the density (15/16) (1 - x^2)^2 on -1 < x < 1, the share of
# its states below x, and the integral of that share. The density and its slope vanish at the edges of the packet,
# so the charge of an atom has a continuous slope at every level, and nothing lies beyond the edges.
def packet_density(x):
    return .9375*(1.-x.clamp(-1., 1.).square()).square()


def packet_share(x):
    y = x.clamp(-1., 1.)
    return .5+.9375*y*(1.-y*y*(2./3.-.2*y*y))


def packet_integral(x):
    y = x.clamp(-1., 1.)
    y2 = y*y
    return .5*(y+1.)+.9375*(y2*(.5-y2*(1./6.-y2/30.))-11./30.)+F.relu(x-1.)


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


def vector_harmonics(vector, order):
    """The solid harmonics of order 1 to order of every vector channel, (..., channels, components); order 1 is the
    vector itself."""
    return vector if order <= 1 else torch.cat((vector, solid_harmonics(order, vector)[..., 4:]), -1)


class AtomicResponse(nn.Module):
    """The response of every atom: its bias (neutral level), its states as packets, its softness, dd_dE, d0 and
    local amplitudes.

    It reads a learned mixture of the scalars and vectors of all num_layers layers, as wide as one layer. The
    multipoles of the atom reach order max_l and its local amplitudes order completion_max_l. Every order l >= 1 has
    a permanent moment d0 and a response dd_dE, the moment per field projection of that order in units of the
    density width: for l = 1 dd_dE times the density width squared is the dipole per field. d0 and the amplitudes of
    order l >= 1 are gated sums of the order l solid harmonics of the vector channels.

    The states. The states of an atom are packets packets with the same share each of its states, each placed and
    given a width by the network: the density of states is their sum, with nothing between packets and nothing
    beyond them. They lie on the axis of the electrostatic potential at the atom: a state is at the Fermi level
    when the potential minus the Fermi level, the level of the atom, is at its energy. The bias of the atom is where
    on the axis it is neutral, at -bias, and its charge is the number of its states between there and its level,
    filled below the Fermi level and empty above it, plus its softness times that distance. The states and the
    potential place the bands, and the labels of the bands, the band gap and the density of states, train the
    states. The bias and the softness set the charge, and the Fermi level and the electrostatic labels train them.
    In a cell without states at the Fermi level the softness carries the charges: its mean of the neutral levels
    is the Fermi level.

    The start: the packets lie evenly on [-energy_range, energy_range] eV, together DOS_START at every energy, no
    gap and no band edge built in, from where the labels reach a metal and an insulator alike. A label moves the
    packets: a packet that a label of the states sends elsewhere keeps its states and follows the label back, and a
    gap opens where all of them have left.
    """

    def __init__(self, num_scalars, num_vectors, num_layers, num_elements, num_sources, energy_range=ENERGY_RANGE,
                 packets=PACKETS, width=64, max_l=1, completion_max_l=1):
        super().__init__()
        # The mixture maps every layer linearly and sums them, normalized like an e3nn linear map of one layer.
        self.mixer_scalar = nn.Parameter(torch.randn(num_layers*num_scalars, num_scalars))
        self.mixer_vector = nn.Parameter(torch.randn(num_layers*num_vectors, num_vectors))
        self.max_l, self.completion_max_l = int(max_l), int(completion_max_l)
        self.vector_width = min(16, max(4, num_vectors))
        self.num_sources = num_sources
        self.chemical_scalar = nn.Linear(num_scalars, width)
        self.chemical_vector = nn.Linear(num_vectors, self.vector_width, bias=False)
        self.shared = nn.Sequential(nn.Linear(width+self.vector_width, width), nn.SiLU(), nn.Linear(width, width),
                                    nn.SiLU())
        self.bias_readout = nn.Linear(width, 1)
        # The gates of the vector channels in d0, for every order.
        self.d0_readout = nn.Linear(width, self.max_l*self.vector_width)
        self.dd_dE_readout = nn.Linear(width, self.max_l)
        # The start of the packets of every element: evenly on the energy range, each as wide as their spacing to
        # either side, together DOS_START states per volt; initialize_potential_response puts them where the states
        # of DFT are.
        self.energy_range, spacing = float(energy_range), 2*energy_range/packets
        starts = torch.linspace(-energy_range+.5*spacing, energy_range-.5*spacing, packets)
        self.register_buffer('packet_starts', starts.repeat(num_elements, 1))
        self.register_buffer('packet_start_widths', torch.full((num_elements, packets), spacing))
        self.register_buffer('packet_start_states', torch.full((num_elements,), DOS_START*2*energy_range))
        # The centers, half widths and states of the packets: shifts of the centers, and the half widths and the
        # states of the atom as multiples of their starts.
        self.dos_readout = nn.Linear(width, 2*packets+1)
        self.softness_readout = nn.Linear(width, 1)
        self.species_bias = nn.Parameter(torch.zeros(num_elements))
        self.species_source = nn.Parameter(torch.zeros(num_elements, num_sources))
        # The amplitudes of the geometry: linear maps of the hidden features and of each order of the vector channels.
        self.local_source_scalar = nn.Linear(width, num_sources, bias=False)
        self.local_source_vectors = nn.ModuleList(nn.Linear(self.vector_width, num_sources, bias=False)
                                                  for _ in range(self.completion_max_l))
        for layer in (self.bias_readout, self.d0_readout, self.dd_dE_readout, self.dos_readout,
                      self.softness_readout, self.local_source_scalar, *self.local_source_vectors):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self.register_buffer('scalar_unit', torch.ones(num_scalars))
        self.register_buffer('vector_unit', torch.ones(num_vectors))

    def mix(self, scalars, vectors):
        """The mixture of the layers."""
        return (scalars@self.mixer_scalar/math.sqrt(self.mixer_scalar.shape[1]),
                torch.einsum('...kc,kd->...dc', vectors, self.mixer_vector)/math.sqrt(self.mixer_vector.shape[1]))

    def features(self, scalars, vectors):
        scalars, vectors = self.mix(scalars, vectors)
        chemical = self.chemical_scalar(torch.asinh(scalars)/self.scalar_unit)
        vector = vector_asinh(vectors/self.vector_unit[:, None])
        return chemical, self.chemical_vector(vector.transpose(-1, -2)).transpose(-1, -2)

    def forward(self, chemical, vector, attrs, mask):
        hidden = self.shared(torch.cat((chemical, torch.asinh(vector.square().sum(-1))), -1))
        bias = (attrs@self.species_bias+self.bias_readout(hidden)[..., 0])*mask
        dd_dE = positive(self.dd_dE_readout(hidden))*mask[..., None]
        count, one = self.packet_starts.shape[-1], math.log(math.expm1(1.))
        read = self.dos_readout(hidden)
        centers = attrs@self.packet_starts+read[..., :count]
        # The atoms padding a graph have no element and no states; their packets keep a width.
        widths = torch.where(mask[..., None] > 0, attrs@self.packet_start_widths, 1.)*F.softplus(
            read[..., count:2*count]+one)
        shares = ((attrs@self.packet_start_states)[..., None]/count*F.softplus(read[..., 2*count:]+one)
                  * mask[..., None]).expand_as(centers)
        packets = torch.stack((centers, widths, shares), -1)
        softness = SOFTNESS_START*F.softplus(self.softness_readout(hidden)[..., 0]+one)*mask
        harmonics = vector_harmonics(vector, max(self.max_l, self.completion_max_l))
        gates = self.d0_readout(hidden).reshape(*hidden.shape[:-1], self.max_l, vector.shape[-2])
        d0 = [torch.einsum('bnv,bnvc->bnc', gates[..., l-1, :], harmonics[..., l*l-1:(l+1)**2-1])
              for l in range(1, self.max_l+1)]
        d0 = torch.cat(d0, -1)/math.sqrt(vector.shape[-2]) if d0 else bias.new_zeros(*bias.shape, 0)
        amplitudes = [attrs@self.species_source+self.local_source_scalar(hidden)]
        for l, layer in enumerate(self.local_source_vectors, 1):
            amplitudes.append(layer(harmonics[..., l*l-1:(l+1)**2-1].transpose(-1, -2)).transpose(-1, -2).flatten(-2))
        return (bias, dd_dE, packets, d0*mask[..., None], torch.cat(amplitudes, -1)*mask[..., None], softness)

    def origin(self, bias):
        """Where every atom is neutral: its neutral level, -bias on the axis of the potential."""
        return -bias

    def charge(self, law, level, origin=None):
        """The charge of the atoms, their states from origin (zero if not given) to level and their softness, and
        the density of states and softness at level. law = (packets, softness)."""
        packets, softness = law
        centers, widths, shares = packets.unbind(-1)
        origin = torch.zeros_like(level) if origin is None else origin
        x, x0 = (level[..., None]-centers)/widths, (origin[..., None]-centers)/widths
        soft = softness+DOS_FLOOR
        dq = (shares*(packet_share(x)-packet_share(x0))).sum(-1)+soft*(level-origin)
        return dq, (shares*packet_density(x)/widths).sum(-1)+soft

    def energy(self, law, level, origin):
        """The integral of the charge over the level, from origin to level."""
        packets, softness = law
        centers, widths, shares = packets.unbind(-1)
        x, x0, span = (level[..., None]-centers)/widths, (origin[..., None]-centers)/widths, level-origin
        return ((shares*(widths*(packet_integral(x)-packet_integral(x0))-span[..., None]*packet_share(x0))).sum(-1)
                + .5*(softness+DOS_FLOOR)*span.square())

    def electrons(self, packets, level, shifts):
        """The electrons the states of every atom take up when the Fermi level rises by each of shifts (eV).

        At fixed potential the level of an atom falls by as much as the Fermi level rises, and the electrons it
        takes up are the states it passes. Returns (..., atoms, shifts).
        """
        centers, widths, shares = (part[..., None, :] for part in packets.unbind(-1))

        def below(energies):
            return (shares*packet_share((energies[..., None]-centers)/widths)).sum(-1)
        return below(level[..., None])-below(level[..., None]-shifts)

    def band_edges(self, packets, level, mask, signed=False):
        """How far the valence band maximum lies below the Fermi level and the conduction band minimum above it.

        The states of the atoms at their levels (level, the potential of each atom minus the Fermi level) are those
        of the ground state, its potential held fixed, as DFT band edges are eigenvalues of the ground-state
        Hamiltonian. A packet whose center lies on the valence side of the level of its atom belongs to the valence
        band, the others to the conduction band, and each edge is the nearest edge of the packets of its band in
        the cell. Both are at the Fermi level when a packet holds it. signed keeps how far a packet reaches past
        the level instead, as a negative distance, so that a cell whose gap is closed still shows which packets
        close it. Without a packet on a side the edge is read twice the energy range away. The derivatives reach
        the packets that make the edges; a level passed detached is held.
        """
        centers, widths, shares = packets.unbind(-1)
        lower, upper = centers-widths-level[..., None], level[..., None]-centers-widths
        present = (mask[..., None] > 0) & (shares > 0)
        valence = centers >= level[..., None]
        far = torch.full_like(lower, 2*ENERGY_RANGE)
        below = torch.where(present & valence, lower, far).flatten(1).amin(-1)
        above = torch.where(present & ~valence, upper, far).flatten(1).amin(-1)
        if signed:
            return below, above
        holds = (below < 0.) | (above < 0.)
        return torch.where(holds, 0., below), torch.where(holds, 0., above)


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
        # The vacuum level of the neutral atoms in the zero-mean origin of periodic DFT. In bulk it is the same
        # shift of every level, so it enters the Fermi level and band edges of bulk and slab cells alike.
        self.vacuum_reference = self.attrs.sum(1)@model.vacuum_reference_integrals/self.volume
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
        # The orders of the multipoles and of the local amplitudes.
        self.max_l, self.completion_max_l = model.max_l, model.completion_max_l
        self.num_moments = (self.max_l+1)**2
        self.k, k2 = selected(self.wave), selected(self.k2)
        phase = torch.einsum('gkc,gnc->gkn', self.k, self.pack(positions))
        self.co = phase.cos()*self.present[:, None, :]*self.selected_mask[..., None]
        self.si = phase.sin()*self.present[:, None, :]*self.selected_mask[..., None]
        self.smooth = torch.exp(-.5*self.width**2*k2)*self.selected_mask
        self.harmonics = solid_harmonics(max(self.max_l, self.completion_max_l, 1), self.k)
        inside = self.selected_mask[..., None]
        self.radial = torch.exp(-.5*k2[..., None]*model.source_widths.to(positions).square())*inside
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
        overlap = model.counter_charge_overlap
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


def moment_projection(total, space):
    """The energy per multipole coefficient of the potential modes total, width^l R_lm(grad) phi on the density.

    R_lm(-grad) phi is averaged over the Gaussian density of every atom; l = 0 is phi (V), l = 1 is -grad phi (V/A).
    """
    count = space.num_moments
    re, im = phase_factors(space.max_l, -1., space.k)
    shape = space.smooth[..., None]*space.harmonics[..., :count]
    a, b = shape*re, -shape*im
    vr, vi = total[..., 0, None], total[..., 1, None]
    sampled = torch.bmm(space.co.transpose(1, 2), vr*a+vi*b)+torch.bmm(space.si.transpose(1, 2), vr*b-vi*a)
    return sampled*space.k.new_tensor([(-space.width)**l for l in component_orders(space.max_l)])


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
    weight = (2*space.half_mask*space.smooth[:, :half].square()*-space.poisson[:, :half]
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


def reserve_solvers(device):
    """The cuSOLVER and cuBLAS handles of this thread and of the autograd thread of a GPU, made while it is empty.

    A thread makes its handles at its first solve and product, outside the caching allocator of torch, which may by
    then hold the whole GPU. In a finetune the first solve of screen_moments came with the foundation backbone and
    its forces in memory, and cusolverDnCreate failed (CUSOLVER_STATUS_INTERNAL_ERROR). A product and a solve of one
    512 x 512 matrix, with their derivatives, take the looped cuSOLVER routines of the solves of slabs in both threads.
    """
    if torch.device(device).type != 'cuda':
        return
    torch.backends.cuda.preferred_linalg_library('cusolver')
    matrix = torch.eye(512, dtype=torch.float64, device=device, requires_grad=True)
    torch.linalg.solve(matrix@matrix, torch.ones(512, 1, dtype=torch.float64, device=device)).sum().backward()


class PotentialResponse(ScaleShiftMACE):
    """A MACE whose atoms carry the response of MACE-VOLT, minimized at fixed Q, counter charge and field."""

    # The solve stops once it has converged; the most steps matter only for the hardest batches. Sharper packets of
    # states need more: on IrO2-water frames up to 44 steps with packets a hundred times narrower than trained, and in
    # an IrO2-water training a few batches per epoch stopped at 60 in the later epochs.
    newton_steps, newton_tolerance = 200, 1.e-6

    def __init__(self, density_width=1.5, kspace_cutoff_factor=1., fermi_level_offset=0.,
                 short_range_potential=True, energy_range=ENERGY_RANGE, packets=PACKETS, max_l=1,
                 completion_max_l=None, **kwargs):
        super().__init__(keep_last_layer_irreps=True, **kwargs)
        # The energy zeros of the elements are trained with the model: the energy of the response moves along
        # directions of the composition, such as the number of protons, while the response learns.
        self.atomic_energies_fn.atomic_energies = nn.Parameter(self.atomic_energies_fn.atomic_energies.clone())
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
        self.register_buffer('source_widths', torch.tensor(sources))
        self.register_buffer('kspace_cutoff', torch.tensor(kspace_cutoff_factor*gto_basis_kspace_cutoff(
            [self.density_width], max(self.max_l, self.completion_max_l, 1))))
        self.register_buffer('fermi_level_offset', torch.tensor(float(fermi_level_offset)))
        num_elements = len(self.atomic_numbers)
        # The potential integrals of the neutral atoms, the vacuum reference of periodic DFT (FourierSpace), from the
        # proto potentials of the training data, or fitted to its labels and trained as a parameter
        # (initialize_potential_response).
        self.register_buffer('vacuum_reference_integrals', torch.zeros(num_elements))
        self.register_buffer('neutral_atom_form_factors', torch.zeros(num_elements, len(FORM_FACTOR_GRID)))
        # A counter charge overlaps the neutral atoms; in constant potential MD it stands for ions and does not.
        self.counter_charge_overlap = True
        layers = int(self.num_interactions)
        self.response = AtomicResponse(irreps.count('0e'), irreps.count('1o'), layers, num_elements, len(sources),
                                       energy_range=energy_range, packets=int(packets), max_l=self.max_l,
                                       completion_max_l=self.completion_max_l)

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

    def minimum(self, response, space, charge):
        """The exact minimum of the responses at total charge charge.

        Newton's method on the concave dual D(p, mu) = -p.Kp/2 - mu Q - sum_i A_i(b_i + v_i - (Kp)_i - mu),
        A_i the integral of the charge of atom i, whose maximum has p = q: each step solves the responses
        linearized at the current levels and a backtracking line search on D makes every step an ascent, so the
        solve converges for every response. One differentiable solve at the converged levels gives the exact
        derivatives. The moments of order l >= 1, linear in their fields, are eliminated first (screen_moments).
        Returns the state, the local energy and the levels b_i + v_i - (Kp)_i - mu of the atoms.
        """
        bias, dd_dE, packets, d0, amplitudes, softness = response
        law = (packets, softness)
        mask, size, count = space.present, space.size, space.num_moments
        kernel, drive = space.multipole_terms()
        # The potential of the local amplitudes on the multipoles; with the drive, the field of each moment.
        applied = moment_projection(amplitude_spectrum(amplitudes, space), space).transpose(1, 2).flatten(1)
        orders = torch.tensor(component_orders(space.max_l)[1:], dtype=torch.long, device=mask.device)-1
        dd_dE, d0 = dd_dE[..., orders].transpose(1, 2).flatten(1), d0.transpose(1, 2).flatten(1)
        effective, drive_q, screening = screen_moments(kernel, applied+drive, dd_dE, d0, size)
        drive_q = drive_q*mask
        base = (bias+drive_q)*mask
        # delta is the level of an atom from its neutral level; on the axis of its states it is at delta + origin.
        origin = self.response.origin(bias)*mask

        def dual(p, mu, kp):
            delta = (base-kp-mu[:, None])*mask
            held = self.response.energy(law, delta+origin, origin)
            return -.5*(p*kp).sum(-1)-mu*charge-(held*mask).sum(-1)

        def linearized(delta):
            dq, n = self.response.charge(law, delta+origin, origin=origin)
            return solve_charges(effective, drive_q, (dq-n*delta)*mask, n*mask, bias, charge, mask)
        with torch.no_grad():
            # From the responses linearized at the Fermi level.
            p, mu = linearized(torch.zeros_like(mask))
            kp = (effective@p[..., None]).squeeze(-1)
            for steps in range(1, self.newton_steps+1):
                delta = (base-kp-mu[:, None])*mask
                dq = self.response.charge(law, delta+origin, origin=origin)[0]*mask
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
        if change > 1.e3*self.newton_tolerance:
            logging.warning(f'Newton solve of the levels stopped after {steps} steps with a level change of '
                            f'{change:.2e} V')
        delta = (base-kp-mu[:, None])*mask
        # The values are those of the solve and the derivatives those of the responses linearized at its levels.
        # Converged, the two agree to the tolerance. Unconverged, the linearized responses are one more full Newton
        # step from where the line search stopped, which put the Fermi level of single batches of an IrO2-water
        # training volts off.
        linear, linear_mu = linearized(delta.detach())
        charges, mu = p+(linear-linear.detach()), mu+(linear_mu-linear_mu.detach())
        level = (bias+drive_q-(effective@charges[..., None]).squeeze(-1)-mu[:, None])*mask
        moments = charges if screening is None else torch.cat((charges, screened_moments(charges, d0, screening)), -1)
        field = applied+drive-(kernel@moments[..., None]).squeeze(-1)
        energy = ((charges*level-self.response.energy(law, level+origin, origin)-bias*charges).sum(-1)
                  + .5*((moments[:, size:]-d0)*field[:, size:]).sum(-1)-(moments*applied).sum(-1))
        state = torch.cat((moments.reshape(len(mask), count, size).transpose(1, 2), amplitudes), -1)
        return state, energy, level

    def forward(self, data, training=False, compute_force=True, compute_virials=False, compute_stress=False,
                band_edges=False, dos=False, **kwargs):
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
            response = self.response(*self.response.features(space.pack(scalars), space.pack(vectors)), space.attrs,
                                     space.present)
            state, response_energy, level = self.minimum(response, space, charge.to(space.volume))
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
        # The band edges and the density of states are read where the states of the atoms are: at the potential of
        # each atom minus the Fermi level. Both are held: they have labels of their own, and a band label that
        # reaches them moves the charges of every frame. No bias enters: the bias of an atom is where it is neutral,
        # not where its bands are (AtomicResponse), and the labels of the bands train the packets. The same read
        # serves every cell.
        aligned = (level-response[0]).detach()
        labelled = data.get('band_gap_weight')
        labelled = labelled is not None and bool((labelled > 0).any())
        if band_edges or labelled:
            # The band edges are those of the ground state, as DFT band edges are eigenvalues of the ground-state
            # Hamiltonian: the nearest edges of the packets of the final minimum's response to the levels, the
            # potential held fixed. No charge is added to the cell, so no charging energy or screening of an added
            # charge enters the edges, in a slab or in bulk. The Fermi level is -dE/dQ in every cell, metal or not,
            # and the edges lie below and above it.
            if band_edges or not training:
                below, above = self.response.band_edges(response[2], aligned, space.present)
                output.update(valence_band_maximum=fermi_level-below, conduction_band_minimum=fermi_level+above,
                              band_gap=below+above, band_edge_distances=torch.stack((below, above), -1))
            if labelled:
                # For the band gap label the same edges are counted from where the Fermi level of the label lies
                # in the states of the model (it lies below that of the model by their difference, held; a frame
                # without a Fermi level label keeps the Fermi level of the model). The label then speaks of the
                # states alone: an insulator has none within half its gap of that level, a metal has states at
                # it. The Fermi level of the model enters neither, and its own label trains it. Counted from the
                # Fermi level of the model, a band label and a Fermi level off its label are traded against each
                # other: the bands of an insulator follow the wrong level, or a metal is told to empty the
                # states between the two levels.
                apart = torch.zeros_like(fermi_level)
                if data.get('fermi_level') is not None and data.get('fermi_level_weight') is not None:
                    apart = torch.where(data['fermi_level_weight'].reshape(-1) > 0.,
                                        (fermi_level-data['fermi_level'].reshape(-1).to(fermi_level)).detach(), apart)
                placed = self.response.band_edges(response[2], aligned+apart[:, None]*space.present, space.present,
                                                  signed=True)
                output['band_edge_distances_at_label'] = torch.stack(placed, -1)
        labelled = data.get('dos')
        if dos or (labelled is not None and labelled.numel() > 0):
            # The density of states of every atom, as the electrons it takes up when the Fermi level moves by the
            # energies at the potential of the ground state, over the energy range of its states. Like the band
            # edges it is read where the states are. The label is put on the same energies: the states of DFT as
            # they are, counted from the Fermi level of DFT to the same absolute energies, so that in a gap both
            # curves are flat wherever in it each Fermi level lies, and in a metal a Fermi level off the one of DFT
            # shows as the electrons between the two.
            steps = int(round(ENERGY_RANGE/DOS_SAMPLING))
            energies = torch.arange(-steps, steps+1, device=level.device, dtype=level.dtype)*DOS_SAMPLING
            curves = self.response.electrons(response[2], aligned, energies)
            output.update(atomic_electrons=space.unpack(curves), dos_energies=energies)
            if labelled is not None and labelled.numel() > 0:
                shift = fermi_level-data['fermi_level'].reshape(-1).to(fermi_level)
                output['atomic_electrons_dft'] = dos_label(data, energies, shift)
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


def dos_label(data, energies, shift):
    """The density of states label of every atom of the batch as electrons on the energies of the model: the states
    of DFT as they are, counted from the Fermi level of DFT to the energies from the model's Fermi level, which lies
    shift (per graph) above it. Each bin of the label holds its states evenly over its width, so the count is exact
    at the edges of the bins and linear between them. nan beyond the energies of the label, and in graphs without a
    label."""
    labels, start, first = [], 0, 0
    counts = (data['ptr'][1:]-data['ptr'][:-1]).tolist()
    for graph, (count, (atoms, bins)) in enumerate(zip(counts, data['dos_shape'].reshape(-1, 2).tolist())):
        if atoms == 0:
            labels.append(energies.new_full((count, len(energies)), float('nan')))
            continue
        states = data['dos'][start:start+atoms*bins].reshape(atoms, bins).to(energies)
        energy = data['dos_energy'][first:first+bins].to(energies)
        start, first = start+atoms*bins, first+bins
        step = energy[1]-energy[0]
        held = states*step
        below = torch.cat((held.new_zeros(atoms, 1), torch.cumsum(held, -1)), -1)

        def count_to(at):
            position = ((at-energy[0])/step+.5).clamp(0., bins)
            index = position.floor().long().clamp_max(bins-1)
            return below[:, index]+(position-index)*held[:, index]
        at = energies+shift[graph]
        inside = (at >= energy[0]-.5*step) & (at <= energy[-1]+.5*step)
        labels.append(torch.where(inside, count_to(at)-count_to(torch.zeros_like(at[:1])), float('nan')))
    return torch.cat(labels)


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
        if not use.any():
            # A small cell, such as one atom of bulk silicon, may have no mode below the common bandwidth.
            continue
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
def start_packets(response, dataset, numbers):
    """The packets of every element start where the states of DFT are: at the quantiles of the mean density of states
    of its atoms in the labels of the training data within the energy range of the Fermi level of DFT, each with its
    share of the states of that range and as wide as its distance to the nearer of its neighbours.

    Every atom then starts neutral where DFT fills the states of its element on average, and the packets of an
    element lie on the sides of the Fermi level where its states are, so that none has to cross a gap. An element
    without labels keeps the even start.
    """
    count, sums, covered, frames, half, step = response.packet_starts.shape[-1], {}, {}, 0, None, None
    for data in dataset:
        shape = getattr(data, 'dos_shape', None)
        if shape is None or int(shape[0]) == 0 or float(data.dos_weight) <= 0.:
            continue
        atoms, bins = (int(n) for n in shape)
        energy = data.dos_energy.double()
        step = float(energy[1]-energy[0])
        half = int(round(response.energy_range/step))
        offsets = torch.round(energy/step).long()
        keep = offsets.abs() <= half
        index = offsets[keep]+half
        held = data.dos.reshape(atoms, bins).double()[:, keep]*step
        elements = data.node_attrs.argmax(-1)
        for i in elements.unique().tolist():
            rows = held[elements == i]
            sums.setdefault(i, torch.zeros(2*half+1, dtype=torch.float64)).index_add_(0, index, rows.sum(0))
            covered.setdefault(i, torch.zeros(2*half+1, dtype=torch.float64)).index_add_(
                0, index, torch.full(index.shape, float(len(rows)), dtype=torch.float64))
        frames += 1
    if not sums:
        logging.info('Packets of the states start evenly: the training data have no density of states labels')
        return
    totals = {}
    edges = ((torch.arange(2*half+2)-half-.5)*step).numpy()
    for i, held in sums.items():
        mean = (held/covered[i].clamp_min(1.)).numpy()
        below = np.concatenate(([0.], np.cumsum(mean)))
        states = float(below[-1])
        if states <= 0.:
            continue
        # the electron energies of the quantiles; a state at electron energy E lies at -E on the axis of the potential
        quantiles = np.interp((np.arange(count)+.5)/count*states, below+1.e-12*np.arange(len(below)), edges)
        centers = torch.as_tensor(np.sort(-quantiles))
        spacing = centers.diff()
        widths = torch.minimum(torch.cat((spacing[:1], spacing)), torch.cat((spacing, spacing[-1:]))).clamp_min(step)
        response.packet_starts[i] = centers.to(response.packet_starts)
        response.packet_start_widths[i] = widths.to(response.packet_start_widths)
        response.packet_start_states[i] = states
        totals[numbers[i]] = states
    logging.info(f'Packets of the states of {sorted(totals)} start where the states of DFT are: the mean density of '
                 f'states of each element in {frames} training frames, '
                 f'{[round(totals[z], 2) for z in sorted(totals)]} states within {response.energy_range:g} eV of the '
                 f'Fermi level of DFT')


def initialize_potential_response(model, args, train_loader, device):
    """The neutral atoms of the reference calculations, the start of the packets of states and the feature units of
    the response, from training data.

    Periodic DFT measures its potentials and Fermi levels from the mean potential of the cell, which lies below the
    vacuum by the potential integrals of its neutral atoms per volume (FourierSpace.vacuum_reference), 6 to 21 V in
    dense bulk silica and silicon. The integrals of every element are fitted to the proto potentials
    (fourier_proto_potential) where the training data have them, with the potentials of the neutral atoms that a
    counter charge overlaps, which frames with counter charges need. Without them the integrals are fitted to the
    Fermi level and vacuum potential labels and train with the model: in the SiOx set the fit to the Fermi levels
    gives O 74 and Si 222 V A^3, against 76 and 236 from its proto potentials.
    """
    dataset = train_loader.dataset
    key = next(iter(args.heads.values()))['info_keys'].get('fourier_proto_potential')
    frames = []
    if key is not None:
        for atoms in ase.io.iread(args.train_file, index=':'):
            if key in atoms.info:
                frames.append((atoms.cell.array, atoms.positions, atoms.numbers, atoms.info[key]))
            if len(frames) >= 64:
                break
    if not frames and any(float(data.counter_charge) != 0. for data in dataset):
        raise ValueError('Frames with a counter charge need the neutral atoms of every element: give the proto '
                         'potentials of the training frames as fourier_proto_potential in info_keys')
    # The Fermi level and vacuum potential labels with the number densities of the elements of their cells.
    fermi, vacuum = [], []
    for data in dataset:
        density = (data.node_attrs.sum(0).double()/abs(float(torch.linalg.det(data.cell.reshape(3, 3).double())))).tolist()
        if float(data.fermi_level_weight) > 0.:
            fermi.append((density, float(data.fermi_level)))
        if float(data.vacuum_potential_weight) > 0.:
            vacuum.append((density, float(data.vacuum_potential)))
    # An offset set by hand is kept; the computed one is the mean Fermi level label.
    computed = bool(fermi) and abs(float(model.fermi_level_offset)-np.mean([label for _, label in fermi])) < 1.e-6
    with torch.no_grad():
        if frames:
            integrals, rows = fit_neutral_atoms(frames)
            for i, z in enumerate(model.atomic_numbers.tolist()):
                model.vacuum_reference_integrals[i] = -integrals.get(z, 0.)
                if z in rows:
                    model.neutral_atom_form_factors[i] = torch.as_tensor(rows[z])
            if computed:
                # The offset is the mean label less the mean vacuum reference of its cells.
                reference = float(np.mean([np.dot(density, model.vacuum_reference_integrals.tolist())
                                           for density, _ in fermi]))
                model.fermi_level_offset.fill_(float(np.mean([label for _, label in fermi]))-reference)
                logging.info(f'Fermi level offset {float(model.fermi_level_offset):.4f} V: the mean label '
                             f'{np.mean([label for _, label in fermi]):.4f} less the mean vacuum reference '
                             f'{reference:.4f} V')
        elif fermi or vacuum:
            offset = float(model.fermi_level_offset)
            a = np.array([[float(computed), *density] for density, _ in fermi]+[[0., *density] for density, _ in vacuum])
            b = np.array([label-(0. if computed else offset) for _, label in fermi]+[label for _, label in vacuum])
            solution = np.linalg.lstsq(a, b, rcond=None)[0]
            fitted = torch.as_tensor(solution[1:]).to(model.vacuum_reference_integrals)
            del model.vacuum_reference_integrals
            model.vacuum_reference_integrals = nn.Parameter(fitted)
            if computed:
                model.fermi_level_offset.fill_(float(solution[0]))
            logging.info(f'No proto potentials: potential integrals of the neutral atoms of '
                         f'{model.atomic_numbers.tolist()} fitted to {len(fermi)} Fermi level and {len(vacuum)} vacuum '
                         f'potential labels, {[round(float(x), 2) for x in solution[1:]]} V A^3, Fermi level offset '
                         f'{float(model.fermi_level_offset):.4f} V; they train with the model')
    start_packets(model.response, dataset, model.atomic_numbers.tolist())
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
        scalar += torch.asinh(scalars).square().sum(0).detach()
        vector += vectors.square().mean(-1).sum(0).detach()
        count += len(scalars)
    model.response.scalar_unit.copy_((1.+scalar/count).sqrt())
    model.response.vector_unit.copy_((1.+vector/count).sqrt())


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


def fourier_smearing_widths(loss_fn):
    """The smearing widths (A) of the Fourier losses of loss_fn by field, as the config sets them.

    Validation reports the error of each field at these widths. A field without a loss is not reported.
    """
    losses = getattr(loss_fn, 'loss_fns', {})
    return {key: list(losses[key].smearing_widths) for key in ('fourier_density', 'fourier_potential')
            if isinstance(losses.get(key), FourierError)}


def weighted_mean_squared_error_vacuum_potential(ref, pred):
    # vacuum_potential: [n_graphs, ]
    configs_weight = ref.weight*ref.vacuum_potential_weight
    return torch.mean(configs_weight*torch.square(ref['vacuum_potential']-pred['vacuum_potential']))


def weighted_mean_squared_error_band_gap(ref, pred):
    # band_gap: [n_graphs, ], predicted for batches with band gap labels only. The Fermi level label of a cell with
    # a gap is the middle of its gap (zero smearing), so the two labels place both band edges: half the band gap
    # label below and above the Fermi level label, and at it in a metal. The loss is the squared error of the gap
    # plus that of the difference of the two edge distances, twice the squared errors of the two edges, with the
    # distances counted from the Fermi level of the label in the states of the model (forward); a frame without a
    # Fermi level label has them counted from the Fermi level of the model. The distances are signed (band_edges):
    # where packets reach past the level of a label with a gap, they are pushed out of it. The label of a metal
    # says only that states reach its Fermi level, so there a packet past it reads zero, however far it reaches.
    if pred.get('band_edge_distances_at_label') is None:
        return ref.weight.new_zeros(())
    below, above = pred['band_edge_distances_at_label'].unbind(-1)
    metal = (ref['band_gap'] <= 0.) & ((below < 0.) | (above < 0.))
    below, above = torch.where(metal, 0., below), torch.where(metal, 0., above)
    configs_weight = ref.weight*ref.band_gap_weight
    return torch.mean(configs_weight*(torch.square(ref['band_gap']-below-above)+torch.square(above-below)))


def weighted_mean_squared_error_dos(ref, pred):
    # atomic_electrons: [n_atoms, n_energies], and its label on the same energies (nan where it has none), predicted
    # for batches with density of states labels only. Per frame the mean square over its atoms and the energies its
    # label holds, so that the weight means the same for every frame: the bands of DFT end 2 to 10 V from its Fermi
    # level, and the labels hold 30 to 95 % of the model's energies (on average half in IrO2-water, 70 % in SiOx).
    #
    # The curves are all of it, on purpose: a term that empties the gap energies of the labels directly, a
    # comparison on a logarithmic scale and hidden features of its own for the density of states each opened the
    # gaps of water films before their potentials settled.
    if pred.get('atomic_electrons_dft') is None:
        return ref.weight.new_zeros(())
    labelled = torch.isfinite(pred['atomic_electrons_dft'])
    error = torch.where(labelled, pred['atomic_electrons']-pred['atomic_electrons_dft'], 0.)
    graphs = ref.weight.shape[0]
    squares = scatter_sum(error.square().sum(-1), ref.batch, dim=0, dim_size=graphs)
    entries = scatter_sum(labelled.sum(-1).to(error), ref.batch, dim=0, dim_size=graphs)
    return torch.mean(ref.weight*ref.dos_weight*squares/entries.clamp_min(1.))


def replay_loss(ref, pred):
    if 'replay' not in pred:
        return ref.weight.new_zeros(())
    return pred['replay']


# The readout of each target, by the optimizer groups of get_param_options. The `lr` of a loss term is the learning
# rate of its target's readout, and the `lr` of the stage the rate of everything else: the backbone, the energy
# zeros, the neutral atoms, and every readout whose targets give no lr. The energy readouts give the energy and its
# derivatives, the readout of the density of states its curves and band edges, the rest of the response (its shared
# layers too) the charges, dipoles, levels and potentials of the solve, and the replay head the replay. Targets with
# one readout share its rate. `<group>_lr` in a stage still sets one group, above both.
READOUT_TARGETS = {
    'readouts': ('energy_per_atom', 'forces', 'stress'),
    'dos': ('dos', 'band_gap'),
    'response': ('dipole', 'dipole_per_atom', 'fermi_level', 'fermi_level_per_atom', 'vacuum_potential',
                 'fourier_density', 'fourier_potential'),
    'replay_head': ('replay',),
}


def readout_learning_rate(train_stage, group):
    """The learning rate of the optimizer group named `group` in a training stage."""
    targets = READOUT_TARGETS.get(group.removesuffix('_references'), ())
    rates = {float(train_stage['loss'][name]['lr']) for name in targets if 'lr' in train_stage['loss'].get(name, {})}
    if len(rates) > 1:
        raise ValueError(f"stage {train_stage['name']}: {', '.join(targets)} share one readout, so their loss terms "
                         "must give the same lr")
    return rates.pop() if rates else train_stage['lr']


def check_learning_rates(train_schedule, groups):
    """Stop before training at a loss term whose lr no optimizer group takes, or that another one contradicts."""
    read = {name for group in groups for name in READOUT_TARGETS.get(group.removesuffix('_references'), ())}
    for train_stage in train_schedule:
        for group in groups:
            readout_learning_rate(train_stage, group)
        for name, options in train_stage['loss'].items():
            if 'lr' in options and name not in read:
                raise ValueError(f"stage {train_stage['name']}: `{name}` has no readout of its own here to take an lr")


# The targets of the FullRMSE epoch line and error table (error_table), in their order: the name in the epoch line,
# the column of the table, the unit, the digits of the epoch line, the metric of evaluate, shown in thousandths of
# its unit (a Fourier field by its loss: its metrics at the smearing widths of the loss), and the loss terms of
# which one shows the target. Energy and forces are always shown.
FULL_RMSE = (
    ('E_per_atom', 'E / meV / atom', 'meV', 1, 'rmse_e_per_atom', ()),
    ('F', 'F / meV / A', 'meV / A', 1, 'rmse_f', ()),
    ('stress', 'stress / meV / A^3', 'meV / A^3', 1, 'rmse_stress', ('stress',)),
    ('virials_per_atom', 'virials / meV / atom', 'meV', 1, 'rmse_virials_per_atom', ('stress',)),
    ('Mu_per_atom', 'dipole / me A / atom', 'meA', 2, 'rmse_mu_per_atom', ('dipole', 'dipole_per_atom')),
    ('polarizability_per_atom', 'polarizability / me A^2 / V / atom', 'me A^2 / V', 2,
     'rmse_polarizability_per_atom', ('polarizability',)),
    ('DMA', 'DMA / me A^l', 'me A^l', 2, 'rmse_dma', ('atomic_multipoles',)),
    ('Q', 'Q / me', 'me', 2, 'rmse_total_charge', ('total_charge', 'total_charge_per_atom')),
    ('rho', 'rho {} A / me / A^3', 'me / A^3', 2, 'fourier_density', ('fourier_density',)),
    ('ESP', 'ESP {} A / mV', 'mV', 1, 'fourier_potential', ('fourier_potential',)),
    ('EF', 'EF / meV', 'meV', 1, 'rmse_fermi_level', ('fermi_level', 'fermi_level_per_atom')),
    ('WF', 'WF / meV', 'meV', 1, 'rmse_work_function', ('vacuum_potential',)),
    ('ESPvac', 'ESPvac / mV', 'mV', 1, 'rmse_vacuum_potential', ('vacuum_potential',)),
    ('Eg', 'Eg / meV', 'meV', 1, 'rmse_band_gap', ('band_gap',)),
    ('DOS', 'DOS / me', 'me', 1, 'rmse_dos', ('dos',)),
)


def full_rmse_targets(loss_fn):
    """The targets of FULL_RMSE that the loss of a stage trains: (name, column, unit, digits, metrics)."""
    terms, widths, shown = set(getattr(loss_fn, 'loss_fns', {})), fourier_smearing_widths(loss_fn), []
    for name, column, unit, digits, metric, needs in FULL_RMSE:
        if needs and not terms & set(needs):
            continue
        if metric in widths:
            label = '/'.join(f'{w:g}' for w in widths[metric])
            shown.append((f'{name}({label} A)', column.format(label), unit, digits,
                          [f'rmse_{metric}_{w:g}A' for w in widths[metric]]))
        elif '{}' not in column:
            shown.append((name, column, unit, digits, [metric]))
    return shown


def thousandths(metrics, key, digits):
    """A metric of evaluate in thousandths of its unit, nan where the data or the model have none."""
    value = metrics.get(key)
    return f'{float("nan") if value is None or isinstance(value, str) else 1e3*float(value):.{digits}f}'


def full_rmse_lines(valid_loss, metrics, epoch, loss_fn):
    """The FullRMSE epoch line, then for the debug log the frames of the largest errors: (logging level, text)."""
    values = ', '.join(f'RMSE_{name}={"/".join(thousandths(metrics, key, digits) for key in keys)} {unit}'
                       for name, _, unit, digits, keys in full_rmse_targets(loss_fn))
    lines = [(logging.INFO, f'Epoch {epoch}: loss={valid_loss:.4f}, {values}')]
    # A few frames can hold most of a squared error. Each is named by its energy label and composition.
    for name, key in (('the energy per atom', 'largest_energy'), ('the Fermi level', 'largest_fermi_level'),
                      ('the work function', 'largest_work_function'), ('the band gap', 'largest_band_gap')):
        if metrics.get(key):
            lines.append((logging.DEBUG, f'Largest validation errors of {name}, label - model (meV) and share of '
                                         f'the squared error: {metrics[key]}'))
    return lines


def full_rmse_columns(loss_fn):
    """The columns of the FullRMSE error table: the config type, the targets and the relative error of the forces."""
    columns = ['config_type']
    for name, column, *_ in full_rmse_targets(loss_fn):
        columns.append('RMSE '+column)
        if name == 'F':
            columns.append('relative F RMSE %')
    return columns


def full_rmse_row(config_type, metrics, loss_fn):
    """A row of the FullRMSE error table, in thousandths of the units."""
    row = [config_type]
    for name, _, _, _, keys in full_rmse_targets(loss_fn):
        row.append('/'.join(thousandths(metrics, key, 2) for key in keys))
        if name == 'F':
            relative = metrics.get('rel_rmse_f')
            row.append(f'{relative:.2f}' if isinstance(relative, float) else 'nan')
    return row


def frame_names(batch, numbers):
    """Every frame of a batch by its energy label and composition, such as '-134.5194 O10Si4'."""
    energies = batch.energy.detach().cpu().numpy() if batch.energy is not None else np.full(batch.num_graphs, np.nan)
    if numbers is None:
        return [f'{energy:.4f}' for energy in energies]
    counts = scatter_sum(batch.node_attrs, batch.batch, dim=0, dim_size=batch.num_graphs).cpu().numpy().round().astype(int)
    return [f'{energy:.4f} '+''.join(f'{chemical_symbols[int(z)]}{n}' for z, n in zip(numbers, row) if n)
            for energy, row in zip(energies, counts)]


def largest_errors(delta, names, count=5):
    """The frames with the largest errors, by energy label and composition: label - model in thousandths of the unit
    and the share of the squared error; then the RMSE of all the other frames."""
    delta = np.ravel(delta)
    worst = np.argsort(-np.abs(delta))[:count]
    rest = np.delete(delta, worst)
    text = ', '.join(f'{names[i]}: {1e3*delta[i]:+.0f} ({100*delta[i]**2/max((delta**2).sum(), 1e-300):.0f}%)'
                     for i in worst)
    if len(rest):
        text += f'; RMSE of the other {len(rest)} frames {1e3*np.sqrt(np.mean(rest**2)):.1f}'
    return text


class ResponseErrors:
    """The validation errors of the targets of MACE-VOLT over the batches of evaluate: the Fermi level, vacuum
    potential, work function, band gap and density of states of the frames with labels, the Fourier fields at the
    smearing widths of their losses, and the frames of the largest errors."""

    def __init__(self, model, loss_fn):
        self.numbers = getattr(model, 'atomic_numbers', None)
        self.smearing_widths = fourier_smearing_widths(loss_fn)
        self.deltas = {key: [] for key in ('energy', 'fermi_level', 'vacuum_potential', 'work_function', 'band_gap',
                                           'dos')}
        self.frames = {key: [] for key in ('energy', 'fermi_level', 'work_function', 'band_gap')}
        self.fourier = {}

    def add(self, batch, output):
        names = frame_names(batch, self.numbers)

        def keep(key, delta, observed):
            self.deltas[key].append(delta[observed])
            if key in self.frames:
                self.frames[key] += [name for name, kept in zip(names, observed.tolist()) if kept]
        if output.get('energy') is not None and batch.energy is not None:
            keep('energy', (batch.energy-output['energy'])/(batch.ptr[1:]-batch.ptr[:-1]),
                 torch.ones_like(batch.energy, dtype=torch.bool))
        if output.get('fermi_level') is not None and batch.fermi_level is not None:
            keep('fermi_level', batch.fermi_level-output['fermi_level'], batch.fermi_level_weight > 0.)
        if output.get('vacuum_potential') is not None:
            observed = batch.vacuum_potential_weight > 0.
            keep('vacuum_potential', batch.vacuum_potential-output['vacuum_potential'], observed)
            # the work function is the vacuum potential minus the Fermi level
            keep('work_function', batch.vacuum_potential-batch.fermi_level-output['vacuum_potential']
                 + output['fermi_level'], observed & (batch.fermi_level_weight > 0.))
        if output.get('band_gap') is not None:
            keep('band_gap', batch.band_gap-output['band_gap'], batch.band_gap_weight > 0.)
        if output.get('atomic_electrons_dft') is not None:
            # every atom and energy with a label
            keep('dos', output['atomic_electrons']-output['atomic_electrons_dft'],
                 torch.isfinite(output['atomic_electrons_dft']) & (batch.dos_weight > 0.)[batch.batch, None])
        for key, widths in self.smearing_widths.items():
            if output.get(key+'_dft') is not None:
                observed = getattr(batch, key+'_weight') > 0.
                for width in widths:
                    self.fourier.setdefault(f'rmse_{key}_{width:g}A', []).append(
                        spectral_errors(batch, output, key, width)[observed])

    def results(self):
        metrics = {}
        for key, deltas in self.deltas.items():
            delta = torch.cat(deltas).detach().cpu().numpy() if deltas else np.zeros(0)
            if not len(delta):
                continue
            if key != 'energy':
                metrics[f'mae_{key}'] = float(np.mean(np.abs(delta)))
                metrics[f'rmse_{key}'] = float(np.sqrt(np.mean(delta**2)))
                metrics[f'q95_{key}'] = float(np.percentile(np.abs(delta), 95.))
            if key in self.frames:
                metrics[f'largest_{key}'] = largest_errors(delta, self.frames[key])
        for name, errors in self.fourier.items():
            metrics[name] = float(np.sqrt(torch.cat(errors).detach().cpu().numpy().mean()))
        return metrics

