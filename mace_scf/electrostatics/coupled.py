"""Coupled density/potential response of MACE-VOLT, its screened updates and observations.

Names follow MACE-SCF, which calls the charge response to the Fermi level dq_dmu and
its normalized form the Fukui functions. dq_dphi is the charge per volt of electron
potential, dd_dE the dipole per V/A of electric field, fermi_dos the local density of
states at the Fermi level, fukui its normalized form, which places the charge deficit,
and bias the field independent potential that the charges respond to besides phi, the
bias of a MACE-SCF biased linear potential embedding.

CoupledResponse charges follow the local law q_i = dq_dphi_i (bias_i + phi_i - mu) with
dq_dphi_i = fermi_dos_i of the geometry and one Fermi level mu per cell, which the Fukui
functions fix to the total charge: the charge goes where the states at the Fermi level
are, and no part of the cell is held at a charge of its own. The biases are read once,
at the field of the exact linear solution. Dipoles respond linearly to the field, and
two local potential amplitudes complete the short-range field as a nonlinear function
of the evolving deformation field. Every update solves the affine electrostatic block
exactly, so MACE-SCF linear mixing of the updates only resolves the nonlinear amplitudes.
The network reads the cell at its net charge Q + Q_cc without the counter charge. The
counter charge then enters the cell at exact linear response with every network output
held, so along Q_cc = -Q the energy is a convex quadratic and the cell voltage rises
linearly with Q.
Potentials are electron referenced (in V, equal to eV).
"""
import logging
import math

import torch
from e3nn import o3
from graph_longrange.utils import FIELD_CONSTANT
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .potential import FORM_FACTOR_GRID, CounterCharge, attach_labels, reciprocal_grid

RESPONSE_SCALE = .02
# Parameters renamed to the MACE-SCF naming, former name: name.
RENAMED = {'species_level': 'species_bias', 'dipole_hardness': 'dd_dE_readout', 'common_level': 'fermi_level_readout'}


def _positive(x):
    """A positive response from an unconstrained readout: RESPONSE_SCALE at x = 0."""
    return RESPONSE_SCALE*(F.softplus(x+math.log(math.expm1(1.)))+1.e-6)


def angular_transport(positions, edge_index, shifts, batch, slot, graphs, size, cutoff, width):
    """l=0 and l=1 transport on the MACE graph, no learned attention.

    Channel 0 is T_ij=w_ij/(1+sum_j w_ij): periodic images add and the extra 1
    bounds the row sum below 1. Channels 1:4 hold T_ij*r_ij/width, with r_ij from
    atom i to neighbor image j; solid harmonics stay smooth at coincident edges
    and the smooth cutoff supplies the second derivatives of force training.
    """
    sender, receiver = edge_index
    vector = positions[sender]-positions[receiver]-shifts.to(positions)
    square = vector.square().sum(-1)
    weights = torch.exp(-.5*square/width**2)*(1.-square/cutoff**2).clamp_min(0.).pow(3)
    index = (batch[receiver]*size+slot[receiver])*size+slot[sender]
    values = torch.cat((weights[:, None], weights[:, None]*vector/width), -1)
    raw = positions.new_zeros((graphs*size*size, 4)).index_add(0, index, values).reshape(graphs, size, size, 4)
    return raw/(1.+raw[..., 0].sum(-1, keepdim=True))[..., None]


def vector_asinh(vector):
    """Invertible, rotation-equivariant counterpart of scalar asinh for descriptors.

    The Taylor branch avoids a sqrt(0) derivative in force-loss differentiation.
    """
    squared = vector.square().sum(-1, keepdim=True)
    threshold = math.sqrt(torch.finfo(vector.dtype).eps)
    norm = squared.clamp_min(threshold).sqrt()
    small = squared.clamp_max(threshold)
    series = 1.-small/6.+3.*small.square()/40.
    return vector*torch.where(squared < threshold, series, torch.asinh(norm)/norm)


def sample_spectrum(total, geometry):
    """Gaussian receiver averages of phi (V) and -grad(phi) (V/A) at every atom.

    phi is the electron potential energy per elementary charge. Each receiver
    width is a separate channel of the same Fourier projection.
    """
    cosine, sine, wave, _, _, _, receiver, *_ = geometry
    re = total[..., 0, None]*receiver
    im = total[..., 1, None]*receiver
    re_k = (re[..., None]*wave[:, :, None]).flatten(-2)
    im_k = (im[..., None]*wave[:, :, None]).flatten(-2)
    sampled = (torch.bmm(cosine.transpose(1, 2), torch.cat((re, im_k), -1))
               + torch.bmm(sine.transpose(1, 2), torch.cat((-im, re_k), -1)))
    value, gradient = sampled.split((re.shape[-1], re_k.shape[-1]), -1)
    return value, gradient.reshape(*value.shape, 3)


class CoupledResponse(nn.Module):
    """Shared density/potential response with positive local responses.

    dq_dphi is the bare charge response of the charge law; the fixed-Q projection
    and electrostatic coupling give its full response. It belongs to the geometry.
    Biases are read once per solve, at the field of the exact linear solution, and held
    during the updates. The charge response to the evolving field therefore stays
    positive: a bias that followed it could cancel the response, an anti-screening that
    drives the fixed point to instability. dd_dE times density_width**2 is the dipole per
    electric field. The local potential amplitudes are Gaussians of
    width sigma and 2 sigma, sigma being potential_amplitudes_smearing_width,
    which is also the smoothing width of the smoothed potential observation.
    Receivers average the field over the density width and the field feature widths.
    Every update solves the screened law exactly and only the local amplitudes iterate.
    """
    coupled = True
    # The scf_tolerance of the last training stage; models saved without it take the calculator's.
    deployment_tolerance = None

    def __setstate__(self, state):
        # The nonlinear multipole update was removed: a model trained with it has no linear law.
        nonlinear = state.pop('atomic_multipole_update', 'linear') == 'nonlinear'
        if state.pop('include_nonlinear_multipole_update', nonlinear):
            raise ValueError('This model was trained with the removed nonlinear multipole update; train it again')
        # Models saved before the MACE-SCF naming keep their parameters.
        for group in ('_parameters', '_modules'):
            if any(name in state.get(group, ()) for name in RENAMED):
                state[group] = type(state[group])((RENAMED.get(k, k), v) for k, v in state[group].items())
        super().__setstate__(state)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints saved before the MACE-SCF naming keep their parameters.
        for old, new in RENAMED.items():
            for key in [k for k in state_dict if k == prefix+old or k.startswith(prefix+old+'.')]:
                state_dict[prefix+new+key[len(prefix+old):]] = state_dict.pop(key)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def __init__(self, node_feats_irreps, charges_irreps, num_elements, density_width=1.5,
                 field_widths=(1.5, 3.), potential_amplitudes_smearing_width=1.5, width=64, **kwargs):
        super().__init__()
        irreps = o3.Irreps(node_feats_irreps)
        if o3.Irreps(charges_irreps).dim != 4:
            raise ValueError('CoupledResponse uses monopoles and dipoles (atomic_multipoles_max_l=1)')
        sigma = float(potential_amplitudes_smearing_width)
        sources = [sigma, 2*sigma]
        receivers = [float(density_width)]+sorted(set(float(w) for w in field_widths)-{float(density_width)})
        if any(not math.isfinite(w) or w <= 0 for w in sources+receivers) or sigma > density_width:
            raise ValueError('Gaussian widths must be finite and positive, with potential_amplitudes_smearing_width '
                             'at most atomic_multipoles_smearing_width')
        self.scalar_slices = [(s.start, s.stop) for (_, ir), s in zip(irreps, irreps.slices()) if ir == o3.Irrep('0e')]
        self.vector_slices = [(s.start, s.stop) for (_, ir), s in zip(irreps, irreps.slices()) if ir == o3.Irrep('1o')]
        scalar_channels = sum(b-a for a, b in self.scalar_slices)
        vector_channels = sum((b-a)//3 for a, b in self.vector_slices)
        self.source_channels, self.field_channels = len(sources), len(receivers)
        self.density_width = float(density_width)
        self.potential_amplitudes_smearing_width = sigma
        self.density_reference_width = max(receivers)
        self.vector_width = min(16, max(4, vector_channels))
        self.vector_basis_size = 2*self.vector_width
        self.chemical_scalar = nn.Linear(scalar_channels, width)
        self.chemical_vector = (nn.Linear(vector_channels, self.vector_width, bias=False)
                                if vector_channels else None)
        self.shared = nn.Sequential(nn.Linear(width+2*self.vector_width, width), nn.SiLU(),
                                    nn.Linear(width, width), nn.SiLU())
        self.vector_out = nn.Linear(width, (1+self.source_channels)*self.vector_basis_size)
        self.species_bias = nn.Parameter(torch.zeros(num_elements))
        self.scalar_out = nn.Linear(width, 2+self.source_channels)
        self.dd_dE_readout = nn.Linear(width, 1)
        self.state_scalar = nn.Linear(self.field_channels, width, bias=False)
        self.state_vector = nn.Linear(self.field_channels, self.vector_width, bias=False)
        self.neighbor_scalar = nn.Linear(self.field_channels, width, bias=False)
        self.neighbor_vector = nn.Linear(self.field_channels, self.vector_width, bias=False)
        self.neighbor_divergence = nn.Linear(self.field_channels, width, bias=False)
        self.neighbor_gradient = nn.Linear(self.field_channels, self.vector_width, bias=False)
        # A uniform bias is a null direction of the fixed-Q charge law. Its own
        # projection of the same hidden features is the Fermi level readout of the
        # charging energy.
        self.fermi_level_readout = nn.Linear(width, 1, bias=False)
        self.local_source_scalar = nn.Linear(width, self.source_channels, bias=False)
        self.local_source_vector = nn.Linear(self.vector_width, self.source_channels, bias=False)
        self.species_source = nn.Parameter(torch.zeros(num_elements, self.source_channels))
        for layer in (self.scalar_out, self.vector_out, self.dd_dE_readout, self.fermi_level_readout,
                      self.local_source_scalar, self.local_source_vector, self.neighbor_scalar,
                      self.neighbor_vector, self.neighbor_divergence, self.neighbor_gradient):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        self.register_buffer('geometry_scalar_unit', torch.ones(scalar_channels))
        self.register_buffer('geometry_vector_unit', torch.ones(vector_channels))
        self.register_buffer('source_widths', torch.tensor(sources))
        self.register_buffer('receiver_widths', torch.tensor(receivers))
        # The neutral atoms of the reference calculations (potential.install_neutral_atoms).
        self.register_buffer('vacuum_reference_integrals', torch.zeros(num_elements))
        self.register_buffer('neutral_atom_form_factors', torch.zeros(num_elements, len(FORM_FACTOR_GRID)))
        self.register_buffer('deployment_steps', torch.tensor(16))
        self.register_buffer('deployment_mixing', torch.tensor(.5))

    def farfield_filter(self, k2):
        """Extra smoothing of the density to the far-field reference width."""
        return torch.exp(-.5*(self.density_reference_width**2-self.density_width**2)*k2)

    def geometry_features(self, scalars, vectors):
        chemical = self.chemical_scalar(torch.asinh(scalars)/self.geometry_scalar_unit)
        if self.chemical_vector is None:
            return chemical, vectors.new_zeros((*vectors.shape[:-2], self.vector_width, 3))
        scaled = vector_asinh(vectors/self.geometry_vector_unit[:, None])
        return chemical, self.chemical_vector(scaled.transpose(-1, -2)).transpose(-1, -2)

    def state_embedding(self, chemical, vector, v, e, transport):
        """Embed the sampled field and its l<=1 neighbor differences on the MACE graph.

        The state enters only through the field it creates. transport stacks the
        four neighbor operators with their row sums removed from the diagonal.
        """
        c = v.shape[-1]
        messages = torch.bmm(transport, torch.cat((v, e.flatten(-2)), -1)).reshape(v.shape[0], 4, v.shape[1], 4*c)
        radial, directional = messages[:, 0], messages[:, 1:]
        divergence = directional[..., c:].reshape(v.shape[0], 3, v.shape[1], c, 3).diagonal(dim1=1, dim2=4).sum(-1)
        scalar = torch.cat((v, radial[..., :c], divergence), -1)
        polar = torch.cat((e, radial[..., c:].reshape_as(e), directional[..., :c].permute(0, 2, 3, 1)), -2)
        scalar_weight = torch.cat((self.state_scalar.weight, self.neighbor_scalar.weight,
                                   self.neighbor_divergence.weight), -1)
        vector_weight = torch.cat((self.state_vector.weight, self.neighbor_vector.weight,
                                   self.neighbor_gradient.weight), -1)
        return (chemical+F.linear(torch.asinh(scalar), scalar_weight),
                vector+F.linear(vector_asinh(polar).transpose(-1, -2), vector_weight).transpose(-1, -2))

    def _invariants(self, chemical, vector, potential, field, mask, reference, target, transport):
        """Centered biased potentials bias+phi plus Q/sum(fermi_dos): the charge law with its Fermi level eliminated.

        A common shift of the field cancels; all reference derivatives stay live.
        """
        bias, fermi_dos = reference.unbind(-1)
        dos = fermi_dos*mask
        total = dos.sum(-1)
        biased = bias[..., None]+potential
        mean = (dos*biased[..., 0]).sum(-1)/total
        v = biased-mean[:, None, None]+(target/total)[:, None, None]
        scalar, embedded = self.state_embedding(chemical, vector, v, field, transport)
        return torch.cat((scalar, torch.asinh(embedded.square().sum(-1)),
                          torch.asinh((vector*embedded).sum(-1))), -1), embedded

    def _constitutive_outputs(self, invariants, vector, embedded, include_dipole=True):
        hidden = self.shared(invariants)
        # The geometry dipole is read once in prepare_reference; updates read only the amplitudes.
        start = 0 if include_dipole else self.vector_basis_size
        gate = F.linear(hidden, self.vector_out.weight[start:], self.vector_out.bias[start:])
        gate = gate.reshape(*hidden.shape[:-1], self.source_channels+int(include_dipole), self.vector_basis_size)
        polar = torch.einsum('bnov,bnvc->bnoc', gate, torch.cat((vector, embedded), -2))/math.sqrt(self.vector_basis_size)
        return self.scalar_out(hidden), polar, hidden

    def geometry_reference_features(self, chemical, vector):
        """The shared hidden features at zero field, where both geometric invariants coincide."""
        norm = torch.asinh(vector.square().sum(-1))
        first = self.shared[0]
        width, channels = chemical.shape[-1], norm.shape[-1]
        weight = torch.cat((first.weight[:, :width],
                            first.weight[:, width:width+channels]+first.weight[:, width+channels:]), -1)
        hidden = F.linear(torch.cat((chemical, norm), -1), weight, first.bias)
        return self.shared[3](self.shared[2](self.shared[1](hidden)))

    def prepare_reference(self, chemical, vector, attrs, p0, mask):
        """Geometry-only coefficients, differentiable in positions and parameters.

        Returns the local amplitudes, the geometry law [dd_dE, bias, fermi_dos] and the
        species biases.
        """
        hidden = self.geometry_reference_features(chemical, vector)
        initial = self.scalar_out(hidden)
        species_bias = attrs@self.species_bias
        local = torch.cat((attrs@self.species_source+self.local_source_scalar(hidden),
                           self.local_source_vector(vector.transpose(-1, -2)).transpose(-1, -2).flatten(-2)),
                          -1)*mask[..., None]
        law = torch.cat((_positive(self.dd_dE_readout(hidden)),
                         species_bias[..., None]+initial[..., :1]+p0[..., :1], _positive(initial[..., 1:2])), -1)
        return local, law, species_bias

    def reference_offsets(self, chemical, vector, local, law, mask, target, transport):
        """The reference [state offsets, dd_dE, bias, fermi_dos] of the geometry law at total charge target.

        At zero field the amplitudes equal the local readout; the field response
        enters as their difference. The geometry dipole is read here once.
        """
        zeros = local.new_zeros((*mask.shape, self.field_channels))
        invariants, embedded = self._invariants(chemical, vector, zeros, zeros[..., None].expand(-1, -1, -1, 3),
                                                mask, law[..., -2:], target, transport)
        scalar, polar, _ = self._constitutive_outputs(invariants, vector, embedded)
        raw = torch.cat((scalar[..., 2:], polar[..., 1:, :].flatten(-2)), -1)
        offset = torch.cat((torch.zeros_like(local[..., :1]), polar[..., 0, :], local-raw), -1)*mask[..., None]
        return torch.cat((offset, law), -1)

    def biases(self, chemical, vector, potential, field, mask, reference, target, transport, species_bias, p0):
        """The biases of the charge law, read from the environment at one sampled field."""
        invariants, _ = self._invariants(chemical, vector, potential, field, mask, reference[..., -2:], target,
                                         transport)
        return species_bias+self.scalar_out(self.shared(invariants))[..., 0]+p0[..., 0]

    def forward(self, chemical, vector, potential, field, mask, target, reference, transport):
        """Local potential amplitudes and hidden features at the sampled field."""
        invariants, embedded = self._invariants(chemical, vector, potential, field, mask, reference[..., -2:],
                                                target, transport)
        scalars, polar, hidden = self._constitutive_outputs(invariants, vector, embedded, include_dipole=False)
        amplitudes = (torch.cat((scalars[..., 2:], polar.flatten(-2)), -1)+reference[..., 4:-3])*mask[..., None]
        return amplitudes, hidden


class CoupledGeometry:
    """Per-forward Fourier operators on the support shared by response, observers and screening."""

    def __init__(self, model, data, positions):
        r = self.response = model.field_dependent_charges_map
        self.cell = data['cell'].reshape(-1, 3, 3)
        self.volume = torch.linalg.det(self.cell).abs()
        pbc = data['pbc'].reshape(-1, 3).bool()
        if bool((pbc.sum(-1) < 2).any()):
            raise NotImplementedError('MACE-VOLT supports periodic bulk and two-periodic slabs')
        self.slab = pbc.sum(-1) == 2
        self.axes = (~pbc).long().argmax(-1)
        rows = torch.arange(len(self.cell), device=positions.device)
        normal = torch.linalg.cross(self.cell[rows, (self.axes+1) % 3], self.cell[rows, (self.axes+2) % 3])
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
        self.width = r.density_width
        active = self.mask & (self.k2 > 0)
        self.coulomb = torch.where(active, FIELD_CONSTANT/self.k2.clamp_min(1.e-30), 0.)
        self.slab_factor = FIELD_CONSTANT/self.volume*self.slab
        self.external = data.get('external_field', positions.new_zeros((len(self.counts), 3))).reshape(-1, 3)
        field = (self.external*self.normal).sum(-1)
        if bool(((self.external-field[:, None]*self.normal).abs().amax(-1) > 1.e-10).any()
                or ((field != 0) & ~self.slab).any()):
            raise ValueError('A homogeneous field must be normal to a slab')
        branch = data.get('dipole_correction_fraction', positions.new_full((len(self.counts),), .5)).reshape(-1)
        self.branch = torch.where(torch.isfinite(branch), branch, .5).remainder(1.)
        # Centre of the continuous slab branch: normal moments use the zero-mean
        # sawtooth coordinate about it, the origin of the plug-in and of the ramp.
        self.branch_center = (self.branch+.5).remainder(1.)*self.slab
        self.origin = self.branch_center[:, None]*self.cell[rows, self.axes]*self.slab[:, None]
        vacuum = data.get('vacuum_fraction', positions.new_full((len(self.counts),), .5)).reshape(-1)
        self.zfrac = torch.where(torch.isfinite(vacuum), vacuum, .5).remainder(1.)
        self.planar = (self.modes[None]*pbc[:, None]).abs().sum(-1) == 0
        self.axial = self.planar & active & self.slab[:, None]
        self.order = self.modes[None].expand(len(self.counts), -1, -1).gather(
            -1, self.axes[:, None, None].expand(-1, len(self.modes), 1)).squeeze(-1)
        self.attrs = self.pack(data['node_attrs'])
        # The vacuum level in the zero-mean origin of periodic DFT, from the neutral atoms.
        self.vacuum_reference = self.attrs.sum(1)@r.vacuum_reference_integrals/self.volume*self.slab
        counter = CounterCharge(self, data, positions)
        self.counter_density, self.counter_energy, self.counter_moment = counter.density, counter.energy, counter.moment

        phase = torch.einsum('gkc,gnc->gkn', self.wave, self.pack(positions))
        co = phase.cos()*self.present[:, None, :]*active[..., None]
        si = phase.sin()*self.present[:, None, :]*active[..., None]
        gaussian = torch.exp(-.5*self.width**2*self.k2)*active
        real = torch.cat((gaussian[..., None], torch.zeros_like(self.wave)), -1)
        imag = torch.cat((torch.zeros_like(gaussian[..., None]), -gaussian[..., None]*self.wave), -1)
        radial = torch.exp(-.5*self.k2[..., None]*r.source_widths.to(positions).square())*active[..., None]
        receiver = torch.exp(-.5*self.k2[..., None]*r.receiver_widths.to(positions).square())*active[..., None]
        zn = ((self.pack(positions)-self.origin[:, None])*self.normal[:, None]).sum(-1)*self.present
        tensors = (co, si, self.wave, real, imag, radial, receiver, torch.view_as_real(self.ramp(-self.slab_factor)),
                   torch.view_as_real(self.ramp(field)), zn, self.normal, -self.coulomb, self.volume.reciprocal())
        # Pack the retained nonzero modes: padded FFT corners and k=0 never enter
        # the moment operator or a field sample. Observers expand them back.
        count = int(active.sum(-1).max())
        if count == 0:
            raise ValueError('The Fourier cutoff contains no nonzero cell modes')
        indices = torch.arange(active.shape[1], device=positions.device).expand_as(active)
        indices = indices.masked_fill(~active, active.shape[1]).sort(-1).values[:, :count]
        self.selected_mask = indices < active.shape[1]
        self.selected_indices = indices.clamp_max(active.shape[1]-1)

        def selected(value):
            mask = self.selected_mask.reshape(*self.selected_mask.shape, *([1]*(value.ndim-2)))
            return value[rows[:, None], self.selected_indices]*mask
        self.tensors = tuple(selected(value) if i < 9 or i == 11 else value for i, value in enumerate(tensors))
        # The fixed fields of the neutral interface (applied) and of the solve (applied and counter charge).
        self.fixed = (self.tensors[8], self.tensors[8]+selected(torch.view_as_real(counter.potential)))
        self.kernels = coefficient_kernels(self.tensors, self.width, (2*math.pi)**1.5*self.width**3,
                                           r.potential_amplitudes_smearing_width)

    def pack(self, values):
        result = values.new_zeros((len(self.counts), self.size, *values.shape[1:]))
        result[self.batch, self.slot] = values
        return result

    def unpack(self, values):
        return values[self.batch, self.slot]

    def ramp(self, field, width=None):
        """Fourier coefficients of the centred sawtooth of a normal field, smoothed to width."""
        safe = torch.where(self.axial, self.order, torch.ones_like(self.order))
        phase = -2*math.pi*safe*self.branch[:, None]
        amplitude = field[:, None]*self.length[:, None]/(2*math.pi*safe)
        width = self.width if width is None else width
        amplitude = amplitude*torch.exp(-.5*width**2*self.k2)*self.axial
        return torch.complex(-amplitude*phase.sin(), amplitude*phase.cos())

    def plane(self, spectrum):
        """Plane average at the vacuum fraction."""
        phase = 2*math.pi*self.order*self.zfrac[:, None]
        return (spectrum*torch.complex(phase.cos(), phase.sin())*self.axial).real.sum(-1)

    def ramp_plane(self, field):
        """The exact sawtooth at the vacuum plane, midpoint at its jump, free of Fourier ringing."""
        distance = (self.zfrac-self.branch).remainder(1.)
        return field*self.length*torch.where(distance == 0, 0., distance-.5)*self.slab

    def expand_spectrum(self, value):
        """Restore the retained modes to the shared FFT grid."""
        output = value.new_zeros((len(self.counts), self.k2.shape[1], value.shape[-1]))
        return output.scatter_add(1, self.selected_indices[..., None].expand_as(value),
                                  value*self.selected_mask[..., None])


def coefficient_kernels(geometry, width, source_volume, base_width):
    """Fourier kernels of the state: Gaussian moment carriers and local potential amplitudes."""
    _, _, wave, br, bi, radial, *_, poisson, ivol = geometry
    scales = br.new_tensor([1., width, width, width])
    shape_im = (radial[..., None]*wave[:, :, None, :]*(base_width*source_volume)).flatten(-2)
    real = torch.cat((br*poisson[..., None]*scales, radial*source_volume, torch.zeros_like(shape_im)), -1)
    imag = torch.cat((bi*poisson[..., None]*scales, torch.zeros_like(radial), shape_im), -1)
    return real*ivol[:, None, None], imag*ivol[:, None, None]


def coefficient_contributions(state, geometry, kernels):
    """Moment-carrier and local-amplitude potentials on the retained modes, as (real, imag)."""
    cosine, sine = torch.bmm(geometry[0], state), torch.bmm(geometry[1], state)
    kr, ki = kernels
    real = kr*cosine+ki*sine
    imag = ki*cosine-kr*sine
    return (torch.stack((real[..., :4].sum(-1), imag[..., :4].sum(-1)), -1),
            torch.stack((real[..., 4:].sum(-1), imag[..., 4:].sum(-1)), -1))


def total_spectrum(state, geometry, kernels, width, fixed):
    """The recurrent field: state, normal-dipole boundary and the fixed applied and counter-charge field."""
    cosine, sine = torch.bmm(geometry[0], state), torch.bmm(geometry[1], state)
    real, imag = kernels
    internal = torch.stack(((real*cosine+imag*sine).sum(-1), (imag*cosine-real*sine).sum(-1)), -1)
    ramp, z, normal = geometry[7], geometry[9], geometry[10]
    dipole = (state[..., 0]*z+(state[..., 1:4]*width*normal[:, None]).sum(-1)).sum(-1)
    return internal+dipole[:, None, None]*ramp+fixed


class SCFNumericalError(RuntimeError):
    """The trajectory became nonfinite or ran away; nothing was clipped or substituted."""


def ensure_cuda_linalg(device):
    """Select cuSOLVER for CUDA solves: PyTorch may pick MAGMA for large batches, which crashed."""
    if torch.device(device).type == 'cuda' and torch.backends.cuda.preferred_linalg_library().name != 'Cusolver':
        torch.backends.cuda.preferred_linalg_library('cusolver')
        logging.info('SCF CUDA linear algebra: cuSOLVER selected (process-wide)')


def moment_responses(law, mask):
    """The responses of the four moment coordinates of every atom: dq_dphi = fermi_dos and dd_dE."""
    return (torch.cat((law[..., -1:], law[..., -3:-2].expand(-1, -1, 3)), -1)*mask[..., None]).flatten(-2)


def moment_kernels(geometry, kernels, mask, sigma):
    """Linear response of the sampled moment drive to moments and local amplitudes."""
    co, si, wave = geometry[:3]
    kr, ki = kernels
    source_re = (co[..., None]*kr[:, :, None, :]+si[..., None]*ki[:, :, None, :]).flatten(-2)
    source_im = (co[..., None]*ki[:, :, None, :]-si[..., None]*kr[:, :, None, :]).flatten(-2)
    receiver = geometry[6][..., 0, None, None]
    observer_re = torch.cat((co[..., None], -sigma*si[..., None]*wave[:, :, None, :]), -1)*receiver
    observer_im = torch.cat((-si[..., None], -sigma*co[..., None]*wave[:, :, None, :]), -1)*receiver
    matrix = -(observer_re.flatten(-2).transpose(1, 2)@source_re+observer_im.flatten(-2).transpose(1, 2)@source_im)
    matrix = matrix.reshape(*matrix.shape[:2], mask.shape[-1], kr.shape[-1])
    completion = -matrix[..., 4:].flatten(-2)
    # The slab boundary field of the normal dipole is rank one.
    unit_v, unit_e = sample_spectrum(geometry[7], geometry)
    observed = torch.cat((unit_v[..., :1], -sigma*unit_e[..., 0, :]), -1).flatten(-2)
    dipole = torch.cat((geometry[9][..., None], sigma*geometry[10][:, None, :].expand(-1, mask.shape[1], -1)),
                       -1).flatten(-2)
    matrix = matrix[..., :4].flatten(-2)-observed[..., None]*dipole[:, None, :]
    present = mask[..., None].expand(-1, -1, 4).flatten(-2)
    return matrix*present[..., None]*present[:, None, :], completion


def screen_factors(response, kernel, mask):
    """Inverse of the bordered screened moment operator at fixed total charge, and its scaling.

    In coordinates scaled by root = sqrt(R/sum R), R the moment responses, the operator
    is I + root K root sum R, bordered by the total-charge constraint; dipoles are
    unconstrained. The live inverse keeps every coordinate, response and higher derivative.
    """
    present = mask[..., None].expand(-1, -1, 4).flatten(-2)
    total = response.sum(-1, keepdim=True)
    root = torch.sqrt(torch.where(present.bool(), response/total, torch.ones_like(response)))*present
    matrix = kernel*root[..., None]*root[:, None, :]*total[..., None]
    matrix = matrix+torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    border = root*torch.cat((mask[..., None], torch.zeros_like(mask[..., None].expand(-1, -1, 3))), -1).flatten(-2)
    bordered = torch.cat((torch.cat((matrix, border[..., None]), -1),
                          torch.cat((border, torch.zeros_like(root[:, :1])), -1)[:, None, :]), -2)
    ensure_cuda_linalg(bordered.device)
    inverse, info = torch.linalg.inv_ex(bordered)
    if not bool(torch.isfinite(inverse).all() & (info == 0).all()):
        raise SCFNumericalError('Singular or nonfinite screened moment operator')
    return inverse, root


def screen_action(completion, response, mask, factors):
    """The screened moment update as an affine map of the local amplitudes, built once per geometry.

    In old+(I-L)^-1 (F(old)-old) the linear moment response L cancels exactly.
    Returns the action on the moment drive and on the amplitudes.
    """
    inverse, root = factors
    present = mask[..., None].expand(-1, -1, 4).flatten(-2)
    divisor = torch.where(present.bool(), root, torch.ones_like(root))
    action = root[..., None]*inverse[:, :-1, :-1]/divisor[:, None, :]*present[..., None]*present[:, None, :]
    return action, action@(response[..., None]*completion)


def screen_offset(action, factors, bias, reference, fixed, geometry, p0, target, mask, sigma):
    """The part of the screened update that the amplitudes leave fixed.

    The biases, the fixed field, the geometry dipoles and the total charge.
    """
    inverse, root = factors
    values, field = sample_spectrum(fixed, geometry)
    base = torch.cat(((reference[..., -1]*(bias+values[..., 0]))[..., None],
                      p0[..., 1:4]/sigma+reference[..., 1:4]-reference[..., -3:-2]*sigma*field[..., 0, :]), -1)
    return (action@(base*mask[..., None]).flatten(1)[..., None]).squeeze(-1)+root*inverse[:, :-1, -1]*target[:, None]


def apply_screen_action(amplitudes, mask, target, sources, offset):
    value = ((sources@amplitudes.flatten(1)[..., None]).squeeze(-1)+offset).reshape(*mask.shape, 4)*mask[..., None]
    charge = value[..., 0]+mask*((target-value[..., 0].sum(-1))/mask.sum(-1))[:, None]
    return torch.cat((charge[..., None], value[..., 1:4], amplitudes), -1)


def mixing_steps(update, initial, steps, mixing, checkpointed=False, tolerance=None):
    """MACE-SCF linear mixing z <- z + mixing (F(z) - z), differentiated through every update.

    Every update already contains the exact screened response of the multipoles, so the
    mixing only resolves the local amplitudes. checkpointed recomputes blocks of updates in
    backward. With a positive tolerance, steps is a maximum: the updates stop once the
    largest change of the state in one update, over the whole batch, falls below it,
    checked after every update (every block when checkpointed). Autograd follows the
    updates made, so every derivative stays exact for the computed state. Returns the
    state and the number of updates made.
    """
    if not bool(torch.isfinite(initial).all()):
        raise SCFNumericalError('SCF initial state is not finite')
    # A precision-scaled runaway budget, not a physical bound.
    limit = initial.detach().abs().amax().clamp_min(1.)/math.sqrt(torch.finfo(initial.dtype).eps)
    axes = tuple(range(1, initial.ndim))

    def check(peaks, first):
        invalid = (~torch.isfinite(peaks)) | (peaks > limit)
        if bool(invalid.any()):
            step, graph = invalid.nonzero()[0].tolist()
            raise SCFNumericalError(f'SCF runaway at step {first+step}, graph {graph}: max|state| '
                                    f'{float(peaks[step, graph]):.6e} exceeds {float(limit):.6e}')

    stop = tolerance is not None and tolerance > 0
    block = max(1, math.isqrt(int(steps))) if checkpointed else 1 if stop else int(steps)
    state, taken = initial, 0
    for start in range(0, int(steps), block):
        def advance(value, count=min(block, int(steps)-start), first=start+1):
            peaks = []
            for _ in range(count):
                proposed = update(value)
                peaks.append(proposed.detach().abs().amax(axes))
                residual = proposed-value
                value = value+mixing*residual
            check(torch.stack(peaks), first)
            return value, residual
        state, residual = (checkpoint(advance, state, use_reentrant=False, preserve_rng_state=False)
                           if checkpointed else advance(state))
        taken = min(start+block, int(steps))
        if stop and float(residual.detach().abs().amax()) < tolerance:
            break
    return state, taken


def prepare_coupled(model, data, local):
    """One live geometry, inverse and screened action per batch.

    Returns the geometry and solve(target), which gives the initial state, the update
    map and the counter charge shift at total charge target with the counter charge.
    """
    r = model.field_dependent_charges_map
    g = CoupledGeometry(model, data, local.positions)
    counter = data.get('counter_charge')
    counter = (torch.zeros_like(g.volume) if counter is None else counter.reshape(-1).to(g.volume))
    features = g.pack(local.all_layer_feats)
    scalar = torch.cat([features[..., a:b] for a, b in r.scalar_slices], -1)
    vector = (torch.cat([features[..., a:b].reshape(*features.shape[:2], -1, 3) for a, b in r.vector_slices], -2)
              if r.vector_slices else features.new_zeros((*features.shape[:2], 0, 3)))
    p0 = g.pack(local.field_independent_charge_density)
    if not hasattr(model, 'foundation_element_map'):
        vector, p0 = vector[..., [2, 0, 1]], p0[..., [0, 3, 1, 2]]
    chemical, vector = r.geometry_features(scalar, vector)
    mask = g.present
    operator = angular_transport(local.positions, data['edge_index'], data['shifts'], g.batch, g.slot,
                                 len(g.counts), g.size, float(model.r_max),
                                 float(r.receiver_widths.max())).permute(0, 3, 1, 2)
    transport = (operator-torch.diag_embed(operator.sum(-1))).flatten(1, 2)
    source, geometry_law, species_bias = r.prepare_reference(chemical, vector, g.attrs, p0, mask)
    kernel, completion = moment_kernels(g.tensors, g.kernels, mask, r.density_width)
    response = moment_responses(geometry_law, mask)
    factors = screen_factors(response, kernel, mask)
    action, sources = screen_action(completion, response, mask, factors)
    fukui = geometry_law[..., -1]*mask
    fukui = fukui/fukui.sum(-1, keepdim=True)

    def solve(target):
        """The updates of the cell at its net charge, and the counter charge at exact linear response.

        The network reads the cell at its net charge Q + Q_cc without the counter
        charge, with the biases it reads at the field of the exact linear solution of the
        geometry law. Moving charge between the counter charge and the cell then acts at
        exact linear response with every network output held, so along Q_cc = -Q the
        energy is a convex quadratic and the cell voltage rises linearly with Q.
        """
        net = target+counter
        reference = r.reference_offsets(chemical, vector, source, geometry_law, mask, net, transport)

        def offset_at(bias):
            def offset(total, applied):
                return screen_offset(action, factors, bias, reference, applied, g.tensors, p0, total, mask,
                                     r.density_width)
            return offset
        linear = apply_screen_action(source, mask, net, sources, offset_at(reference[..., -2])(net, g.fixed[0]))
        potential, field = sample_spectrum(total_spectrum(linear, g.tensors, g.kernels, r.density_width, g.fixed[0]),
                                           g.tensors)
        offset_of = offset_at(r.biases(chemical, vector, potential, field, mask, reference, net, transport,
                                       species_bias, p0))
        offset = offset_of(net, g.fixed[0])

        def update(z, observe=False):
            potential, field = sample_spectrum(total_spectrum(z, g.tensors, g.kernels, r.density_width, g.fixed[0]),
                                               g.tensors)
            amplitudes, hidden = r(chemical, vector, potential, field, mask, net, reference, transport)
            proposal = apply_screen_action(amplitudes, mask, net, sources, offset)
            if not observe:
                return proposal
            # The Fermi level readout of the charging energy, from the final hidden features.
            fermi = r.fermi_level_readout((fukui[..., None]*hidden).sum(1)).squeeze(-1)+(fukui*species_bias).sum(-1)
            return proposal, potential, (geometry_law[..., -1]*mask).sum(-1), fermi
        # The counter charge at exact linear response, the same at any amplitudes; zero without one.
        empty = torch.zeros_like(source)
        shift = (apply_screen_action(empty, mask, target, sources, offset_of(target, g.fixed[1]))
                 - apply_screen_action(empty, mask, net, sources, offset))
        # The seed: the exact linear solution at the baseline amplitudes.
        return apply_screen_action(source, mask, net, sources, offset), update, shift
    return g, solve


def evaluate_coupled(model, data, steps=50, training=False, compute_force=True, compute_stress=False,
                     mode='unroll_scf', mixing=None, stability=False, tolerance=None):
    """One electronic trajectory, one scalar energy and its derivatives.

    The energy is the DFT free energy at fixed total charge Q, counter charge
    Q_cc and external field. Forces are -dE/dR, EF = -dE/dQ at fixed Q_cc and
    the cell voltage dE/dQ - dE/dQ_cc is the derivative along Q_cc = -Q.
    WF = vacuum potential - EF. The network reads the cell at its net charge Q + Q_cc without the counter
    charge. Moving charge between the counter charge and the electrode then acts at exact
    linear response with every network output held, so along Q_cc = -Q the energy is a
    convex quadratic and the cell voltage rises linearly with Q. The cell has one Fermi
    level, so its charge goes where the states at the Fermi level are. With stability,
    one more update probes the gain
    of the update map along a random direction at the final state, for the MACE-SCF
    stability loss (outputs perturbing_vector and perturbed_density). The updates use
    MACE-SCF linear mixing. With a positive tolerance, steps is the maximum number of
    updates, and the solve stops once the largest change of the state in one update falls
    below tolerance (scf_steps).
    """
    r = model.field_dependent_charges_map
    mixing = float(r.deployment_mixing) if mixing is None else float(mixing)
    if mode not in ('unroll_scf', 'shortcut_scf') or int(steps) < 1 or not 0 < mixing <= 1:
        raise ValueError('Invalid coupled SCF mode, step count or mixing')
    checkpointed, outer_grad = training and mode == 'shortcut_scf', torch.is_grad_enabled()
    with torch.enable_grad():
        data = dict(data)
        # Leaf copies give partial derivatives at fixed other charge, even when Q_cc = -Q.
        charge = data['total_charge'].clone().requires_grad_(True)
        counter = data.get('counter_charge', torch.zeros_like(charge)).clone().requires_grad_(True)
        data['total_charge'], data['counter_charge'] = charge, counter
        positions = data['positions']
        if compute_force:
            positions.requires_grad_(True)
        strain = None
        if compute_stress:
            strain = positions.new_zeros((len(data['ptr'])-1, 3, 3), requires_grad=True)
            symmetric = .5*(strain+strain.transpose(-1, -2))
            data['positions'] = positions+torch.einsum('ni,nij->nj', positions, symmetric[data['batch']])
            cell = data['cell'].reshape(-1, 3, 3)
            data['cell'] = cell+cell@symmetric
            edge_batch = data['batch'][data['edge_index'][0]]
            data['shifts'] = data['shifts']+torch.einsum('ni,nij->nj', data['shifts'], symmetric[edge_batch])
        local = model.local_part(data, compute_force=compute_force)
        g, solve = prepare_coupled(model, data, local)
        q = charge.reshape(-1).to(g.volume)
        initial, update, shift = solve(q)
        state, taken = mixing_steps(update, initial, steps, mixing, checkpointed, tolerance)
        proposal, potential, total_response, fermi = update(state, observe=True)
        residual = ((proposal-state)*g.present[..., None]).abs().amax((1, 2))
        probe = None
        if stability:
            # Finite difference along a random direction; a gain above one marks a map that swings.
            direction = torch.randn_like(state)*g.present[..., None]
            direction = direction*(1.e-4*state.detach().square().mean().sqrt().clamp_min(1.))
            probe = (direction, update(state+direction)-proposal)
        # The cell with its counter charge: the converged updates plus the exact response to the counter charge.
        state = state+shift
        potential, _ = sample_spectrum(total_spectrum(state, g.tensors, g.kernels, r.density_width, g.fixed[1]),
                                       g.tensors)

        moments = torch.cat((state[..., :1], state[..., 1:4]*r.density_width), -1)
        density_coefficients = g.unpack(moments)[..., [0, 2, 3, 1]]
        carrier, regular = (g.expand_spectrum(x) for x in coefficient_contributions(state, g.tensors, g.kernels))
        total_charge = state[..., 0].sum(-1)
        coarse = carrier/torch.where(g.coulomb > 0, -g.coulomb, 1.)[..., None]
        coarse = coarse+torch.stack(((g.k2 == 0)*(total_charge/g.volume)[:, None], torch.zeros_like(g.k2)), -1)
        dipole = (moments[..., :1]*(g.pack(local.positions)-g.origin[:, None])+moments[..., 1:4]).sum(1)
        normal_dipole = (dipole*g.normal).sum(-1)
        # The counter charge in the learned deformation field, the slab cross term and its own terms.
        deformation = g.coulomb*torch.view_as_complex(coarse.contiguous())-torch.view_as_complex(regular.contiguous())
        counter_energy = (g.volume*(g.counter_density.conj()*deformation).real.sum(-1)
                          + g.slab_factor*normal_dipole*g.counter_moment+g.counter_energy)
        electrostatic = (.5*g.volume*(coarse.square().sum(-1)*g.coulomb).sum(-1)+.5*g.slab_factor*normal_dipole.square()
                         - (dipole*g.external).sum(-1)+counter_energy-model.fermi_level_offset*total_charge
                         - g.vacuum_reference*q)
        # Charging work at the Fermi level readout, with the total response sum(fermi_dos).
        electron = .5*q.square()/total_response-q*fermi
        energy = local.energies.sum(-1)+electrostatic+electron
        inputs = [charge, counter]+([positions] if compute_force else [])+([strain] if compute_stress else [])
        derivatives = torch.autograd.grad(energy.sum(), inputs, create_graph=training, allow_unused=True,
                                          retain_graph=training or (outer_grad and not compute_force))
        charge_potential = derivatives[0].reshape(-1)
        counter_potential = (torch.zeros_like(counter) if derivatives[1] is None else derivatives[1]).reshape(-1)
        forces = stress = None
        if compute_force:
            forces = torch.zeros_like(positions) if derivatives[2] is None else -derivatives[2]
        if compute_stress:
            stress = torch.zeros_like(strain) if derivatives[-1] is None else derivatives[-1]/g.volume[:, None, None]

    # Raw fields undo the Gaussian smoothing of the learned ones, smooth = raw exp(-sigma^2 k^2/2):
    # sigma is the density width for the density and the amplitude width for the potential.
    # Labels are attached at the resolution of each prediction. The vacuum level reads the
    # smooth potential and the exact sawtooth: a Gaussian filter leaves a flat vacuum unchanged,
    # while the truncated raw series rings there, by 85 mV on average for DFT fields.
    sharpen = torch.exp(.5*r.density_width**2*g.k2)[..., None]
    smoothing = torch.exp(-.5*r.potential_amplitudes_smearing_width**2*g.k2)
    learned = torch.view_as_complex((carrier+regular).contiguous())
    boundary = (g.external*g.normal).sum(-1)-g.slab_factor*(normal_dipole+g.counter_moment)
    raw = learned/torch.where(g.mask, smoothing, 1.)-g.coulomb*g.counter_density
    prescribed = g.ramp(boundary, width=0.)-g.coulomb*g.counter_density
    fermi_level = -charge_potential
    vacuum = g.plane(learned-g.coulomb*g.counter_density*smoothing)+g.ramp_plane(boundary)+g.vacuum_reference
    farfield = r.farfield_filter(g.k2)[..., None]
    output = {'energy': energy, 'forces': forces, 'stress': stress,
              'virials': -stress*g.volume[:, None, None] if stress is not None else None,
              'density_coefficients': density_coefficients, 'dipole': dipole, 'total_charge': total_charge,
              'fermi_level': fermi_level, 'vacuum_potential': vacuum, 'work_function': vacuum-fermi_level,
              'potential_reference': g.vacuum_reference, 'charge_potential': charge_potential,
              'counter_charge_potential': counter_potential, 'cell_voltage': charge_potential-counter_potential,
              'fourier_density': coarse*sharpen*g.ngrid, 'smooth_fourier_density': coarse*g.ngrid,
              'fourier_farfield_density': coarse*farfield*g.ngrid,
              'fourier_potential': torch.view_as_real(raw+g.ramp(boundary, width=0.))*g.ngrid,
              'smooth_fourier_potential': torch.view_as_real(learned+prescribed*smoothing)*g.ngrid,
              'k_vectors': g.wave, 'k_vectors_mask': g.mask,
              'k_vectors_grid_shape': torch.tensor(g.shape, device=positions.device),
              'scf_residual': residual, 'scf_steps': positions.new_full((len(fermi_level),), float(taken)),
              'electrostatic_energy': electrostatic, 'electron_energy': electron, 'external_field': g.external,
              'esps': None, 'esps_dft': None, 'charges_history': density_coefficients[..., None],
              'electrostatic_features': g.unpack(potential)}
    # WF - eV is the potential step from the counter charge to the outer vacuum.
    output['she_offset'] = output['work_function']-output['cell_voltage']
    if probe is not None:
        output['perturbing_vector'], output['perturbed_density'] = (g.unpack(x) for x in probe)
    attach_labels(data, g, output)
    if 'fourier_density_dft' in output:
        output['smooth_fourier_density_dft'] = output['fourier_density_dft']/sharpen
        output['fourier_farfield_density_dft'] = output['smooth_fourier_density_dft']*farfield
    if 'fourier_potential_dft' in output:
        output['smooth_fourier_potential_dft'] = output['fourier_potential_dft']*smoothing[..., None]
    with torch.no_grad():
        invalid = [key for key in ('energy', 'forces', 'stress', 'fermi_level', 'vacuum_potential', 'cell_voltage')
                   if output[key] is not None and not bool(torch.isfinite(output[key]).all())]
    if invalid:
        raise SCFNumericalError(f'Nonfinite coupled SCF observations: {invalid}')
    return output


@torch.no_grad()
def initialize_coupled(model, loader, device):
    """Numerical units of the geometry descriptors, from the training set only."""
    from mace_scf.utils.foundation import calibration_loader
    r = model.field_dependent_charges_map
    scalar, vector, count = torch.zeros_like(r.geometry_scalar_unit), torch.zeros_like(r.geometry_vector_unit), 0
    for batch in calibration_loader(loader):
        data = batch.to(device).to_dict()
        features = model.local_part(data, compute_force=False).all_layer_feats
        counts = (data['ptr'][1:]-data['ptr'][:-1]).to(features)
        weights = data['weight'].reshape(-1).to(features)
        atom_weight = (weights/counts)[data['batch']][:, None]
        scalar += (torch.asinh(torch.cat([features[:, a:b] for a, b in r.scalar_slices], -1)).square()
                   * atom_weight).sum(0)
        if r.vector_slices:
            vector += (torch.cat([features[:, a:b].reshape(len(features), -1, 3) for a, b in r.vector_slices],
                                 -2).square().mean(-1)*atom_weight).sum(0)
        count += float(weights.sum())
    r.geometry_scalar_unit.copy_((1.+scalar/max(count, 1.e-30)).sqrt())
    r.geometry_vector_unit.copy_((1.+vector/max(count, 1.e-30)).sqrt())
    logging.info('Coupled response: %d local potential channels, Gaussian receiver widths %s',
                 r.source_channels, r.receiver_widths.tolist())
