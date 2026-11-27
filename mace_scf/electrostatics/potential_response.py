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
# A background density of states (e/V per atom) at all energies, so that every charge has one finite level.
DOS_FLOOR = 1.e-5
# The density of states of every atom at the start (e/V), the same at every energy. It is of the order of 1/U, U = 5.4
# V/e the Coulomb energy of a charge of the density width 1.5 A, where the states and the Coulomb energy resist a
# charge alike: far fewer states at the level make an insulator, far more a metal. It is also about the mean of DFT's
# projected states over the energy range (H 0.02, Si 0.13, O 0.2, Ir 0.74 e/V).
DOS_START = .1
# The pull of the start of the states toward those of an average atom, as a fraction of the mean curvature of the
# least squares of start_from_cells. It decides only what the compositions of the training cells leave open.
START_PULL = 1.e-3
# The energies of the density of states of an atom reach ENERGY_RANGE (eV) to either side of its neutral level.
# The band edges of an insulator lie about half its gap from the levels of its atoms, and the levels move by
# volts with the potential, so the range holds band gaps up to about 15 eV. Beyond it an atom has DOS_FLOOR alone.
ENERGY_RANGE = 10.
# The spacing (eV) of the energies of the states of an atom, and the width of the Gaussian with which the bands are
# read. It is how finely band gaps are read: a band edge is where the states of a cell hold the band edge count, and
# DFT's own states, smoothed by this width, return the band gaps of DFT at that count to 69 meV on silicon and silica
# and to 24 meV on water. At 0.05 eV they return those of silicon and silica to 45 meV, with twice the energies per
# atom. How the states are occupied is a separate matter, the smearing of the reference calculations
# (AtomicResponse); without one, they are occupied with this Gaussian too.
RESOLUTION = 0.1
# The most electrons (e) that the states between a band edge and the Fermi level of a cell hold. A band edge is the
# energy of one state, and a state holds two electrons in the cell, its k-points together. Where it does not
# disperse, as in a molecular liquid or at a localized state, all of it lies at the edge: the edge is then where
# one electron is counted, half the state, at its energy, and a larger count reads the edge past the first state.
BAND_EDGE_ELECTRONS = 1.
# The band edge count per volume (e/A^3) of a model whose training data have no density of states label to take it
# from, between the counts measured on silicon and silica (6e-5) and on water (9e-5).
BAND_EDGE_DENSITY = 7.5e-5
# The density of states (e/V per atom) that forms no band. Band edges count only the states of each atom above it.
# Deep in a gap the softplus of the readouts never reaches zero, and no loss sees such sparse states: a residue of
# 3e-4 e/V on every atom narrows the band gap read in DFT's own states of silica by 1.2 eV, and of water by 0.2 eV,
# when every state counts, and by 20 meV above the background. DFT's own states read the band gaps as closely
# either way (silica 152 and water 75 meV above it, 142 and 86 meV with every state).
BAND_STATES = 1.e-3
# The most terms (cells x counts x atoms x energies) that one evaluation of the counted states holds at once.
COUNT_TERMS = 2**22
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


def fermi_step(x):
    return torch.sigmoid(x)


def fermi(x):
    return torch.sigmoid(x)*torch.sigmoid(-x)


class FermiRamp(torch.autograd.Function):
    """log(1 + exp(x)), the integral of the Fermi step, whose derivative is that step at every x and in every
    precision: the derivative that torch takes of the logarithm goes through exp(-x), which overflows in single
    precision below x = -88, and the second derivative, which the gradient of a Fermi level needs, was then nan."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.logaddexp(x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad*torch.sigmoid(x)


def fermi_ramp(x):
    return FermiRamp.apply(x)


# The occupation smearings of the reference calculations (ISMEAR of VASP: 0 gaussian, -1 fermi): the share of a state
# that is empty when the level of its atom lies x smearing widths above it, the density of that step and its integral.
OCCUPATIONS = {'gaussian': (gaussian_step, gaussian, gaussian_ramp), 'fermi': (fermi_step, fermi, fermi_ramp)}


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


def held_states(band, place, mask, shift, width):
    """The states (e) every cell passes when the levels of its atoms move by shift (cells, M), and its states (e/V)
    at the moved levels: band (cells, atoms, energies) the states of the atoms on their energies, of that width,
    place (cells, atoms, energies) the level of every atom from each of its energies (AtomicResponse.places), mask
    the atoms present. Returns two (cells, M)."""
    x0 = (place/width)[:, None]
    x = x0+shift[..., None, None]/width
    weights = band[:, None]*mask[:, None, :, None]
    return (weights*(gaussian_step(x)-gaussian_step(x0))).sum((-1, -2))*width, (weights*gaussian(x)).sum((-1, -2))


class HeldStates(torch.autograd.Function):
    """held_states' passed states, differentiable in band and place. The backward recomputes the steps and Gaussians
    in parts of `part` shifts, so the graph keeps no (cells, M, atoms, energies) tensor."""

    @staticmethod
    def forward(ctx, band, place, mask, shift, width, part):
        ctx.save_for_backward(band, place, mask, shift)
        ctx.width, ctx.part = width, part
        return torch.cat([held_states(band, place, mask, piece, width)[0] for piece in shift.split(part, -1)], -1)

    @staticmethod
    def backward(ctx, grad):
        band, place, mask, shift = ctx.saved_tensors
        width = ctx.width
        x0 = (place/width)[:, None]
        grad_band, grad_place = torch.zeros_like(band), torch.zeros_like(place)
        for piece, weight in zip(shift.split(ctx.part, -1), grad.split(ctx.part, -1)):
            x = x0+piece[..., None, None]/width
            scale = weight[..., None, None]*mask[:, None, :, None]
            grad_band = grad_band+width*(scale*(gaussian_step(x)-gaussian_step(x0))).sum(1)
            grad_place = grad_place+(scale*band[:, None]*(gaussian(x)-gaussian(x0))).sum(1)
        return grad_band, grad_place, None, None, None, None


class AtomicResponse(nn.Module):
    """The response of every atom: its bias (neutral level), density of states on an energy grid, band offset and
    band opening, dd_dE, d0 and local amplitudes.

    It reads a learned mixture of the scalars and vectors of all num_layers layers, as wide as one layer. The
    multipoles of the atom reach order max_l and its local amplitudes order completion_max_l. Every order l >= 1 has
    a permanent moment d0 and a response dd_dE, the moment per field projection of that order in units of the
    density width: for l = 1 dd_dE times the density width squared is the dipole per field. d0 and the amplitudes of
    order l >= 1 are gated sums of the order l solid harmonics of the vector channels. The states lie at energies
    smearing_width apart on [-energy_range, energy_range] eV, over a background of DOS_FLOOR.

    The occupation. A state is emptied as the level of its atom rises past it, by the occupation smearing of the
    reference calculations: `occupation` is 'gaussian' or 'fermi' (ISMEAR = 0 or -1 of VASP) and `occupation_width`
    its width in eV (SIGMA), by default the Gaussian of the spacing of the energies. The charge that an atom holds
    across a band edge is the tail of that function, and the two differ by orders of magnitude there: a state 0.5 eV
    from the level is emptied by exp(-5) = 7e-3 under a Fermi function of 0.1 eV and by 8e-13 under a Gaussian of
    0.1 eV. A water film next to an electrode holds its charge in such tails, so its work function is that of the
    reference calculations only if its states are occupied like theirs. The bands themselves, their edges and the
    density of states, are eigenvalues and not occupations: they are read with the Gaussian of the spacing whatever
    the occupation (count_energies).

    The energies of the states. The density of states of an atom is drawn on the axis of the electrostatic potential
    at the atom: its shape on the energy grid, and its band offset, how far up that axis the grid sits. A state is at
    the Fermi level when the potential minus the Fermi level, the level of the atom, is at its energy. The bias of
    the atom is where on the axis it is neutral, at -bias, and its charge is the number of its states between there
    and its level. Each quantity has one job, the same in a metal and in an insulator. The states, their band offset
    and the potential place the bands, and the labels of the bands, the band gap and the density of states, train the
    first two. The bias sets the charge, and the Fermi level and the electrostatic labels train it: no band moves with
    it. In a cell without states at the Fermi level the biases together set the Fermi level, the mean of the neutral
    levels weighted by the few states left in the gap.

    The band opening. The grid has a lower and an upper half, and the band opening of an atom moves them apart: the
    energies below the centre of its grid lie lower and those above it higher by half the opening, so the band offset
    is the middle of the atom's states and the opening how far its two halves stand apart. One number then opens,
    narrows or closes the gap of an atom, as a smooth function of its surroundings, and the coefficients keep
    describing the shape of the bands on either side. Without it a gap exists only where every coefficient inside
    it has been brought to zero, energy by energy and for each kind of surroundings again. It is zero at the start
    and of either sign: a negative opening lets the two halves overlap.

    The start of the density of states: DOS_START at every energy, no gap and no band edge built in, from where the
    labels reach a metal and an insulator alike. A density of states label takes the states out of the gap of an
    insulator and gives a metal its states at the level, and the band gap label places the band edges. Without a
    density of states label the labels can take states away, with a band gap or with a field the atom does not
    screen, but they do not give them back.
    """

    # A model saved before the occupation smearing was an option occupies its states with the Gaussian of their width.
    occupation, occupation_width = 'gaussian', None

    def __init__(self, num_scalars, num_vectors, num_layers, num_elements, num_sources, energy_range=ENERGY_RANGE,
                 smearing_width=RESOLUTION, width=64, max_l=1, completion_max_l=1, occupation='gaussian',
                 occupation_width=None):
        super().__init__()
        if occupation not in OCCUPATIONS:
            raise ValueError(f'electronic_smearing must be one of {sorted(OCCUPATIONS)}, not {occupation!r}')
        self.occupation = occupation
        self.occupation_width = None if occupation_width is None else float(occupation_width)
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
        # An even number of energies, none at the centre of the grid: the lower and the upper half of the states.
        half = int(round(energy_range/smearing_width))
        centers = (torch.arange(2*half)-half+.5)*(energy_range/half)
        self.dos_width = float(centers[1]-centers[0])
        self.register_buffer('dos_centers', centers)
        self.register_buffer('dos_sides', torch.sign(centers))
        self.register_buffer('dos_offsets', torch.full((num_elements, len(centers)), math.log(math.expm1(DOS_START))))
        self.dos_readout = nn.Linear(width, len(centers))
        self.species_bias = nn.Parameter(torch.zeros(num_elements))
        # The band offset and the band opening, each by element and as a linear map of the hidden features.
        self.species_band_offset = nn.Parameter(torch.zeros(num_elements))
        self.band_offset_readout = nn.Linear(width, 1, bias=False)
        self.species_band_opening = nn.Parameter(torch.zeros(num_elements))
        self.band_opening_readout = nn.Linear(width, 1, bias=False)
        self.species_source = nn.Parameter(torch.zeros(num_elements, num_sources))
        # The amplitudes of the geometry: linear maps of the hidden features and of each order of the vector channels.
        self.local_source_scalar = nn.Linear(width, num_sources, bias=False)
        self.local_source_vectors = nn.ModuleList(nn.Linear(self.vector_width, num_sources, bias=False)
                                                  for _ in range(self.completion_max_l))
        for layer in (self.bias_readout, self.d0_readout, self.dd_dE_readout, self.dos_readout,
                      self.band_offset_readout, self.band_opening_readout, self.local_source_scalar,
                      *self.local_source_vectors):
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
        coefficients = F.softplus(self.dos_readout(hidden)+attrs@self.dos_offsets)*mask[..., None]
        band_offset = (attrs@self.species_band_offset+self.band_offset_readout(hidden)[..., 0])*mask
        band_opening = (attrs@self.species_band_opening+self.band_opening_readout(hidden)[..., 0])*mask
        harmonics = vector_harmonics(vector, max(self.max_l, self.completion_max_l))
        gates = self.d0_readout(hidden).reshape(*hidden.shape[:-1], self.max_l, vector.shape[-2])
        d0 = [torch.einsum('bnv,bnvc->bnc', gates[..., l-1, :], harmonics[..., l*l-1:(l+1)**2-1])
              for l in range(1, self.max_l+1)]
        d0 = torch.cat(d0, -1)/math.sqrt(vector.shape[-2]) if d0 else bias.new_zeros(*bias.shape, 0)
        amplitudes = [attrs@self.species_source+self.local_source_scalar(hidden)]
        for l, layer in enumerate(self.local_source_vectors, 1):
            amplitudes.append(layer(harmonics[..., l*l-1:(l+1)**2-1].transpose(-1, -2)).transpose(-1, -2).flatten(-2))
        return (bias, dd_dE, coefficients, d0*mask[..., None], torch.cat(amplitudes, -1)*mask[..., None],
                band_offset, band_opening)

    def origin(self, bias, band_offset):
        """Where every atom is neutral on its energy grid: its neutral level, -bias on the axis of the potential,
        plus its band offset."""
        return band_offset-bias

    def places(self, level, opening):
        """The level of every atom from each of its energies (..., atoms, energies): the energies of the grid, the
        lower half lower and the upper half higher by half the band opening of the atom."""
        return level[..., None]-.5*opening[..., None]*self.dos_sides-self.dos_centers

    def charge(self, coefficients, level, opening, floor=DOS_FLOOR, origin=None):
        """The charge of the atoms, their states from origin (zero if not given) to level on the energy grid, and
        the density of states at level."""
        w, s = self.dos_width, self.occupation_width or self.dos_width
        step, density, _ = OCCUPATIONS[self.occupation]
        origin = torch.zeros_like(level) if origin is None else origin
        x, x0 = self.places(level, opening)/s, self.places(origin, opening)/s
        dq = (coefficients*(step(x)-step(x0))).sum(-1)*w+floor*(level-origin)
        return dq, (coefficients*density(x)).sum(-1)*(w/s)+floor

    def energy(self, coefficients, level, origin, opening):
        """The integral of the charge over the level, from origin to level."""
        w, s = self.dos_width, self.occupation_width or self.dos_width
        step, _, ramp = OCCUPATIONS[self.occupation]
        x, x0, span = self.places(level, opening)/s, self.places(origin, opening)/s, level-origin
        return (coefficients*(s*(ramp(x)-ramp(x0))-span[..., None]*step(x0))).sum(-1)*w+.5*DOS_FLOOR*span.square()

    def electrons(self, coefficients, level, opening, steps):
        """The electrons every atom takes up when the Fermi level rises by k energy spacings, k = -steps to steps.

        At fixed potential the level of an atom falls by as much as the Fermi level rises, and the charge it loses is
        the states it passes. Returns (..., atoms, 2 steps + 1).
        """
        w = self.dos_width
        x = self.places(level, opening)/w
        full = gaussian_step(x)
        shifts = torch.arange(-steps, steps+1, device=level.device, dtype=level.dtype)
        # a few shifts at a time: all of them at once hold (atoms, energies, shifts)
        taken = [(coefficients[..., None]*(full[..., None]-gaussian_step(x[..., None]-part))).sum(-2)
                 for part in shifts.split(16)]
        return torch.cat(taken, -1)*w+DOS_FLOOR*shifts*w

    def count_energies(self, coefficients, level, opening, mask, counts, floor, guess=None, tolerance=1.e-10):
        """The energies, from the Fermi level, at which the states of every cell hold counts (cells, M): electrons
        above the Fermi level for a positive count, holes below it for a negative one. Returns them with the sign of
        the counts.

        The states of the atoms at their levels (level, the bias and potential of each atom minus the Fermi level) are
        those of the ground state, its potential held fixed, as the eigenvalues of DFT are those of its ground-state
        Hamiltonian, so an energy is the Fermi level the cell would reach by taking up or giving away that many
        electrons with its potentials held. Every atom's states count above BAND_STATES only, so that the sparse
        residue a gap keeps does not move them. The count rises monotonically with the distance from the Fermi level,
        so each distance is the root of a monotone function, which Newton steps kept inside a bracket find, for all
        counts at once. They start from guess (cells, M), distances such as those of the label, where it is finite,
        and from the middle of the range elsewhere. One differentiable Newton step at the root (HeldStates) gives the
        derivative in the coefficients, the levels and the band openings; a level passed detached is held. The step
        divides by the states of the cell at the root, taken no smaller than floor (e/V per cell, (cells,)): an
        energy in a sparse stretch between bands moves by volts as the states there change by little, and its
        derivative is then that of one at the onset of a band. An energy lies within the energy range of the states:
        a cell whose states hold less there has it read at that range, with the derivative of a count reached there,
        which asks for the missing states. Extrapolated beyond the range by the step, such a read once came out at
        tens of eV.
        """
        band = F.relu(coefficients-BAND_STATES)
        w, top = self.dos_width, float(self.dos_centers[-1])
        place = self.places(level, opening)
        # No finer than the energies resolve in the precision of the model: in single precision a tolerance of 1e-10
        # eV is never met, and every read took all its steps.
        tolerance = max(tolerance, 16.*torch.finfo(level.dtype).eps*top)
        # The levels of the atoms rise as the Fermi level falls: holes are counted by raising them, electrons by
        # lowering them, both by the distance.
        direction, target = -torch.sign(counts), counts.abs()
        part = max(1, COUNT_TERMS//max(1, place.numel()))
        guess = torch.full_like(target, float('nan')) if guess is None else guess.detach().to(target)
        distances, slopes, short = [], [], []
        with torch.no_grad():
            fixed, start = band.detach(), place.detach()
            for sign, count, first in zip(direction.split(part, -1), target.split(part, -1), guess.split(part, -1)):
                def held(distance):
                    passed, states = held_states(fixed, start, mask, sign*distance, w)
                    return sign*passed-count, states
                low, high = torch.zeros_like(count), torch.full_like(count, top)
                reached = held(high)[0] >= 0.
                short.append(~reached)
                first = torch.where(torch.isfinite(first), first.clamp(0., top), .5*high)
                distance = torch.where(reached, first, high)
                for _ in range(200):
                    value, slope = held(distance)
                    low, high = torch.where(value < 0., distance, low), torch.where(value < 0., high, distance)
                    step = distance-value/slope
                    # A step onto an end of the bracket is the root reached: rejected, it would leave the root for
                    # the middle of the bracket and bisect back to it, 30 steps of the 40 a read took.
                    inside = (step >= low) & (step <= high)
                    step = torch.where(reached, torch.where(inside, step, .5*(low+high)), distance)
                    if float((step-distance).abs().max()) < tolerance:
                        distance = step
                        break
                    distance = step
                distances.append(distance)
                slopes.append(torch.maximum(held(distance)[1], floor[:, None]))
            distance, slope, short = torch.cat(distances, -1), torch.cat(slopes, -1), torch.cat(short, -1)
        value = direction*HeldStates.apply(band, place, mask, direction*distance, w, part)-target
        # at the range the step keeps the read where it is: only its derivative asks for the missing states
        value = value-torch.where(short, value.detach(), 0.)
        return -direction*(distance-value/slope)

    def band_edges(self, coefficients, level, opening, mask, count, floor=None, tolerance=1.e-10):
        """How far the valence band maximum lies below the Fermi level and the conduction band minimum above it: where
        the states of each cell hold count holes below it and count electrons above it (count_energies). The floor of
        the derivatives is by default count over the width of the states, which an edge on the onset of a band holds
        at least."""
        floor = count/self.dos_width if floor is None else floor
        energies = self.count_energies(coefficients, level, opening, mask, torch.stack((-count, count), -1), floor,
                                       tolerance=tolerance)
        return -energies[:, 0], energies[:, 1]


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

    # The solve stops once it has converged; the most steps matter only for the hardest batches.
    newton_steps, newton_tolerance = 200, 1.e-6

    def __init__(self, density_width=1.5, kspace_cutoff_factor=1., fermi_level_offset=0.,
                 short_range_potential=True, energy_range=ENERGY_RANGE, smearing_width=RESOLUTION,
                 band_edge_density=None, max_l=1, completion_max_l=None, electronic_smearing='gaussian',
                 electronic_smearing_width=None, **kwargs):
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
        # The holes and electrons per volume of the cell (e/A^3) that the ground-state states between each band edge
        # and the Fermi level hold, up to BAND_EDGE_ELECTRONS per cell. The states near a band edge grow in number
        # with the volume, and DFT samples a cell on a mesh as fine as the cell is small, so one state holds
        # electrons in proportion to the volume: a supercell has the band edges of its cell. The states are smeared
        # by their width, so the count sets how far in the tail of the first state its edge is read; a metal reads a
        # gap of twice the count over its density of states.
        # Left unset, it is taken from the density of states labels of the training data
        # (initialize_potential_response): the count per volume at which the states of DFT, smeared like those of
        # the model, return the band gaps of DFT.
        self.band_edge_density = None if band_edge_density is None else float(band_edge_density)
        # electronic_smearing and electronic_smearing_width: the occupation smearing of the reference calculations,
        # with which the states are occupied (AtomicResponse).
        layers = int(self.num_interactions)
        self.response = AtomicResponse(irreps.count('0e'), irreps.count('1o'), layers, num_elements, len(sources),
                                       energy_range=energy_range, smearing_width=float(smearing_width),
                                       max_l=self.max_l, completion_max_l=self.completion_max_l,
                                       occupation=electronic_smearing, occupation_width=electronic_smearing_width)

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
        bias, dd_dE, coefficients, d0, amplitudes, band_offset, opening = response
        mask, size, count = space.present, space.size, space.num_moments
        kernel, drive = space.multipole_terms()
        # The potential of the local amplitudes on the multipoles; with the drive, the field of each moment.
        applied = moment_projection(amplitude_spectrum(amplitudes, space), space).transpose(1, 2).flatten(1)
        orders = torch.tensor(component_orders(space.max_l)[1:], dtype=torch.long, device=mask.device)-1
        dd_dE, d0 = dd_dE[..., orders].transpose(1, 2).flatten(1), d0.transpose(1, 2).flatten(1)
        effective, drive_q, screening = screen_moments(kernel, applied+drive, dd_dE, d0, size)
        drive_q = drive_q*mask
        base = (bias+drive_q)*mask
        # delta is the level of an atom from its neutral level; on its energy grid it is at delta + origin.
        origin = self.response.origin(bias, band_offset)*mask

        def dual(p, mu, kp):
            delta = (base-kp-mu[:, None])*mask
            held = self.response.energy(coefficients, delta+origin, origin, opening)
            return -.5*(p*kp).sum(-1)-mu*charge-(held*mask).sum(-1)

        def linearized(delta):
            dq, n = self.response.charge(coefficients, delta+origin, opening, origin=origin)
            return solve_charges(effective, drive_q, (dq-n*delta)*mask, n*mask, bias, charge, mask)
        with torch.no_grad():
            # From the responses linearized at the Fermi level.
            p, mu = linearized(torch.zeros_like(mask))
            kp = (effective@p[..., None]).squeeze(-1)
            for steps in range(1, self.newton_steps+1):
                delta = (base-kp-mu[:, None])*mask
                dq = self.response.charge(coefficients, delta+origin, opening, origin=origin)[0]*mask
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
        held = self.response.energy(coefficients, level+origin, origin, opening)
        energy = ((charges*level-held-bias*charges).sum(-1)
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
        # each atom minus the Fermi level, on its energy grid. Both are held: they have labels of their own, and a
        # band label that reaches them moves the charges of every frame. No bias enters: the bias of an atom is
        # where it is neutral, not where its bands are (AtomicResponse). Its band offset and its band opening do,
        # and the labels of the bands train them with the states. The same read serves every cell.
        aligned = (level-response[0]).detach()+response[5]
        labelled = data.get('band_gap_weight')
        labelled = labelled is not None and bool((labelled > 0).any())
        spectrum = data.get('dos')
        spectrum = spectrum is not None and spectrum.numel() > 0
        if band_edges or labelled or spectrum:
            # The band edges are those of the ground state, as DFT band edges are eigenvalues of the ground-state
            # Hamiltonian: the states of the final minimum's response at its levels, the potential held fixed and
            # each atom's states counted above BAND_STATES, hold band_edge_density holes per volume of the cell
            # between the valence band maximum and the Fermi level, and as many electrons between it and the
            # conduction band minimum, and no more than BAND_EDGE_ELECTRONS in the cell. Per volume, the edges of a
            # supercell are those of its cell. Without the bound, a large cell would count more than its first state
            # holds and read its edges past that state. No charge is added to the cell, so no charging energy or
            # screening of an added charge enters the edges, in a slab or in bulk. The Fermi level is -dE/dQ in
            # every cell, metal or not, and the edges lie below and above it.
            count = ((self.band_edge_density or BAND_EDGE_DENSITY)*space.volume).clamp_max(BAND_EDGE_ELECTRONS)
            if band_edges or not training:
                below, above = self.response.band_edges(response[2], aligned, response[6], space.present, count)
                output.update(valence_band_maximum=fermi_level-below, conduction_band_minimum=fermi_level+above,
                              band_gap=below+above, band_edge_distances=torch.stack((below, above), -1))
            if labelled or spectrum:
                # For the band labels the same edges are counted from where the Fermi level of the label lies in
                # the states of the model (it lies below that of the model by their difference, held; a frame
                # without a Fermi level label keeps the Fermi level of the model). The labels then speak of the
                # states alone: an insulator has none within half its gap of that level, a metal has states at
                # it. The Fermi level of the model enters neither, and its own label trains it. Counted from the
                # Fermi level of the model, a band label and a Fermi level off its label are traded against each
                # other: the bands of an insulator follow the wrong level, or a metal is told to empty the
                # states between the two levels.
                known = torch.zeros_like(fermi_level, dtype=torch.bool)
                apart = torch.zeros_like(fermi_level)
                if data.get('fermi_level') is not None and data.get('fermi_level_weight') is not None:
                    known = data['fermi_level_weight'].reshape(-1) > 0.
                    apart = torch.where(known, (fermi_level-data['fermi_level'].reshape(-1).to(fermi_level)).detach(),
                                        apart)
                placed = aligned+apart[:, None]*space.present
                below, above = self.response.band_edges(response[2], placed, response[6], space.present, count)
                output['band_edge_distances_at_label'] = torch.stack((below, above), -1)
                if spectrum:
                    # The density of states label is the total density of states of the cell, the eigenvalues of
                    # DFT counted from its Fermi level. It is compared as it is counted: at every energy E of the
                    # states of the model, the number of states per atom between the Fermi level and E, electrons
                    # above it and holes below it (negative). That is the charge per atom the cell would take up
                    # if its Fermi level moved to E with its potentials held, what the charges of the model are
                    # made of. The states of the model are counted like the band edges, from the Fermi level of
                    # the label, and those of DFT are smoothed like them (label_electrons). A cell without a Fermi
                    # level label has no origin for its label and no density of states loss.
                    w = self.response.dos_width
                    steps = len(self.response.dos_centers)//2
                    grid = torch.arange(-steps, steps+1, device=level.device, dtype=level.dtype)*w
                    place = self.response.places(placed, response[6])
                    atoms = space.present.sum(-1, keepdim=True)
                    # the levels of the atoms fall as the Fermi level rises: the electrons taken up are the states
                    # passed, with their sign
                    passed = HeldStates.apply(response[2], place, space.present,
                                              -grid.expand(len(count), -1).contiguous(), w,
                                              max(1, COUNT_TERMS//max(1, place.numel())))
                    label, held = label_electrons(data, grid, w)
                    output.update(dos_electrons=-passed/atoms, dos_electrons_dft=label.to(level)/atoms,
                                  dos_electrons_mask=held & known[:, None])
        if dos:
            # The density of states of every atom, as the electrons it takes up when the Fermi level moves by the
            # energies at the potential of the ground state, over the energy range of its states, and of the cell.
            steps = len(self.response.dos_centers)//2
            energies = torch.arange(-steps, steps+1, device=level.device, dtype=level.dtype)*self.response.dos_width
            curves = self.response.electrons(response[2], aligned, response[6], steps)
            output.update(atomic_electrons=space.unpack(curves), dos_energies=energies,
                          cell_electrons=(curves*space.present[..., None]).sum(1))
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


def label_electrons(data, grid, width):
    """The density of states labels as they are counted: for every cell the states of its label between its Fermi
    level and each energy of grid (eV, from that Fermi level), electrons above it and holes below it (negative), and
    where the label holds them.

    The label is the total density of states of the cell on its own energies. Every bin is smoothed by the Gaussian
    step of width with which the states of the model are read, so the count is a smooth function of the energy. It is
    held at the energies at least two widths inside those of the label, where the smoothing misses no state. A cell
    without a label, or whose label does not hold its Fermi level, holds none. Returns two (cells, energies).
    """
    shapes = data['dos_shape'].reshape(-1, 2).tolist()
    energies = grid.double()
    counts = torch.zeros((len(shapes), len(grid)), dtype=torch.float64, device=grid.device)
    held = torch.zeros((len(shapes), len(grid)), dtype=torch.bool, device=grid.device)
    start = first = 0
    for graph, (rows, bins) in enumerate(shapes):
        if rows == 0:
            continue
        states = data['dos'][start:start+rows*bins].reshape(rows, bins).to(counts).sum(0)
        energy = data['dos_energy'][first:first+bins].to(counts)
        start, first = start+rows*bins, first+bins
        low, high = float(energy[0])+2.*width, float(energy[-1])-2.*width
        if not low < 0. < high:
            continue
        number = states*(energy[1]-energy[0])
        counts[graph] = (number*(gaussian_step((energies[:, None]-energy)/width)-gaussian_step(-energy/width))).sum(-1)
        held[graph] = (energies >= low) & (energies <= high)
    return counts, held


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
def start_from_cells(response, dataset, numbers):
    """The states of every element start at its share of the states of the training cells.

    At every energy of the grid the shares are the non-negative least squares solution of the density of states
    labels per atom of the training cells, smoothed by the bins of the model and read from the Fermi level of the
    label (a state at electron energy E lies at -E on the axis of the potential), over the fractions of the elements
    in each cell, the way the energy zeros of the elements are fitted to total energies. Only the energies at least
    two widths inside those of a label use it. A weak pull toward the states of an average atom, the mean label per
    atom, START_PULL of the mean curvature of the data, decides what the compositions leave open: in a data set whose
    cells have nearly one composition, water at O:H = 1:2 or a metal slab under a water film of nearly constant size,
    the least squares alone give the states of the cell to one element, and with the pull every atom starts with
    those of an average atom. Directions the compositions do determine keep their solution. The shares are floored
    at BAND_STATES, the background that counts for the charges and not for the bands, and set the offsets of the
    coefficients, so that at the start every atom of an element holds its share.
    """
    from scipy.optimize import nnls
    w, centers = response.dos_width, response.dos_centers.double().cpu()
    rows, targets, masks = [], [], []
    for data in dataset:
        shape = getattr(data, 'dos_shape', None)
        if shape is None or int(shape[0]) == 0 or float(data.dos_weight) <= 0.:
            continue
        lines, bins = (int(n) for n in shape)
        energy = data.dos_energy.double()
        cell = data.dos.reshape(lines, bins).double().sum(0)
        step = float(energy[1]-energy[0])
        # a state at electron energy E lies at -E on the axis of the potential
        at = -centers
        density = (cell@(step*gaussian((at[:, None]-energy[None, :])/w)/w).T).numpy()
        count = data.node_attrs.double().sum(0).numpy()
        rows.append(count/count.sum())
        targets.append(density/count.sum())
        masks.append(((at > energy[0]+2.*w) & (at < energy[-1]-2.*w)).numpy())
    if not rows:
        logging.info(f'The states start with {DOS_START:g} e/V at every energy: the training data have no density of '
                     f'states labels')
        return
    rows, targets, masks = np.array(rows), np.array(targets), np.array(masks)
    present = rows.sum(0) > 0
    shares = np.full(response.dos_offsets.shape, DOS_START)
    for j in range(len(centers)):
        use = masks[:, j]
        if not use.any():
            continue
        a, b = rows[use][:, present], targets[use, j]
        pull = math.sqrt(START_PULL*np.trace(a.T@a)/a.shape[1])
        shares[present, j] = nnls(np.vstack((a, pull*np.eye(a.shape[1]))),
                                  np.concatenate((b, np.full(a.shape[1], pull*b.mean()))))[0]
    shares = np.maximum(shares, BAND_STATES)
    with torch.no_grad():
        response.dos_offsets.copy_(torch.log(torch.expm1(torch.as_tensor(shares))).to(response.dos_offsets))
    shown = ', '.join(f'{numbers[i]}: {float(shares[i].sum()*w):.2f} ({float(shares[i][len(centers)//2]):.3f} at the '
                      f'Fermi level)' for i in range(len(shares)) if present[i])
    logging.info(f'The states of the elements start at their shares of the states of {len(rows)} training cells, the '
                 f'non-negative least squares over their compositions at every energy with a pull of {START_PULL:g} '
                 f'toward an average atom; states within the energy range: {shown}')


def band_edge_readings(dataset, width):
    """What reads the band gap of DFT from its own states, per gapped frame with a density of states label: the
    volume of its cell, its band gap label, and the holes and the electrons that its states, smoothed by width like
    the states of the model and counted from the Fermi level of the label as the density of states loss counts them
    (label_electrons), hold up to each distance from the Fermi level.

    Frames whose edges do not both lie well inside the energies of the label are left out.
    """
    frames = []
    for data in dataset:
        shape = getattr(data, 'dos_shape', None)
        if shape is None or int(shape[0]) == 0 or float(data.band_gap_weight) <= 0. or float(data.band_gap) <= 0.:
            continue
        rows, bins = (int(n) for n in shape)
        states, energy = data.dos.reshape(rows, bins).double().sum(0), data.dos_energy.double()
        filled = torch.nonzero(states > 0.).reshape(-1)
        below, above = filled[energy[filled] < 0.], filled[energy[filled] > 0.]
        if (len(below) == 0 or len(above) == 0 or float(energy[above[0]]) > float(energy[-1])-1.
                or float(energy[below[-1]]) < float(energy[0])+1.):
            continue
        distance = torch.arange(0., float(min(energy[-1], -energy[0]))-2.*width, .01, dtype=torch.float64)
        held = states*(energy[1]-energy[0])
        # where no state lies the counts are flat; a vanishing ramp keeps their inverses single valued
        ramp = 1.e-12*torch.arange(len(distance), dtype=torch.float64)
        electrons, holes = ((sign*(held*(gaussian_step((sign*distance[:, None]-energy)/width)
                                         - gaussian_step(-energy/width))).sum(-1)+ramp).numpy() for sign in (1., -1.))
        volume = abs(float(torch.linalg.det(data.cell.reshape(3, 3).double())))
        frames.append((volume, float(data.band_gap), holes, electrons, distance.numpy()))
    return frames


def initialize_potential_response(model, args, train_loader, device):
    """The neutral atoms of the reference calculations, the band edge count and the feature units of the response,
    from training data.

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
    start_from_cells(model.response, dataset, model.atomic_numbers.tolist())
    if model.band_edge_density is None:
        frames = band_edge_readings(train_loader.dataset, model.response.dos_width)
        if frames:
            def reading(density):
                """The band gap errors of DFT's own states read at the count density x volume of every frame."""
                return np.array([np.interp(count, holes, distance)+np.interp(count, electrons, distance)-gap
                                 for volume, gap, holes, electrons, distance in frames
                                 for count in [min(density*volume, BAND_EDGE_ELECTRONS)]])
            densities = np.exp(np.linspace(np.log(1.e-7), np.log(1.e-2), 461))
            model.band_edge_density = float(min(densities, key=lambda density: np.mean(reading(density)**2)))
            errors = reading(model.band_edge_density)
            logging.info(f'Band edge count {model.band_edge_density:.3g} electrons per A^3 of the cell: the count with '
                         f'which the states of DFT return the band gaps of {len(frames)} gapped training frames most '
                         f'closely, to {1e3*np.sqrt(np.mean(errors**2)):.0f} meV RMSE (mean {1e3*errors.mean():+.0f} '
                         f'meV, largest {1e3*np.abs(errors).max():.0f} meV): the band gap error of a model whose '
                         f'density of states is exact')
        else:
            model.band_edge_density = BAND_EDGE_DENSITY
            logging.info(f"Band edge count {model.band_edge_density:.2e} electrons per A^3, the default: no gapped "
                         f"training frame has a density of states label that holds both its band edges")
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
    # Fermi level label has them counted from the Fermi level of the model.
    if pred.get('band_edge_distances_at_label') is None:
        return ref.weight.new_zeros(())
    below, above = pred['band_edge_distances_at_label'].unbind(-1)
    configs_weight = ref.weight*ref.band_gap_weight
    return torch.mean(configs_weight*(torch.square(ref['band_gap']-below-above)+torch.square(above-below)))


def dos_mean_squares(pred):
    """The density of states loss of every cell (e^2 per atom^2): the mean squared error of the number of states per
    atom between the Fermi level and E, over the energies E that its label holds; and whether the cell has a label."""
    held = pred['dos_electrons_mask']
    error = torch.where(held, pred['dos_electrons']-pred['dos_electrons_dft'], 0.)
    number = held.sum(-1)
    return error.square().sum(-1)/number.clamp_min(1), number > 0


def weighted_mean_squared_error_dos(ref, pred):
    # dos_electrons: [n_graphs, n_energies], the states per atom of every cell between the Fermi level and each
    # energy of the states of the model (electrons above the Fermi level, holes below it), its label on the same
    # energies and where the label holds them, predicted for batches with density of states labels only. The density
    # of states is compared as a distribution: the number of states up to each energy, the charge per atom the cell
    # would take up if its Fermi level moved there with its potentials held. A band that lies too high or too low by
    # d costs its states times d at every energy it should have covered, so the loss falls as the bands move into
    # place; compared as densities, two bands that do not overlap would not see each other. Every energy weighs the
    # same, and the count is per atom, so a supercell weighs as its cell. The band edges are single energies, which
    # a count over all energies places only loosely: the band gap term trains them on the eigenvalues themselves.
    # Which atom holds which state no label says: the states of the cell are compared, and the electrostatic labels
    # and the Fermi level place them on the atoms.
    if pred.get('dos_electrons') is None:
        return ref.weight.new_zeros(())
    return torch.mean(ref.weight*ref.dos_weight*dos_mean_squares(pred)[0])


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
    ('Eg', 'Eg / meV', 'meV', 1, 'rmse_band_gap', ('band_gap', 'dos')),
    ('DOS', 'DOS / me / atom', 'me', 1, 'rmse_dos', ('dos',)),
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
        if output.get('dos_electrons') is not None:
            # every cell with a label: the root of its density of states loss, the error of its states per atom
            # counted from the Fermi level, over the energies of its label
            squares, observed = dos_mean_squares(output)
            keep('dos', squares.sqrt(), observed & (batch.dos_weight > 0.))
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

