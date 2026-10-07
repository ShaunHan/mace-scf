"""Coupled density/potential response, screened solves and physical observers.

One nonlinear constitutive map sees the evolving total field. Coarse charge
moments and the regular effective potential remain distinct observations.
Finite trajectories and converged-root differentiation are explicit modes.
"""
from dataclasses import dataclass
from typing import Callable, Tuple
import logging
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint
from e3nn import o3

from .potential import SpectralGeometry, initialize_response, attach_observations

def local_transport(positions, edge_index, shifts, batch, slot, graphs, max_nodes,
                    cutoff, width):
    """T_ij=w_ij/(1+sum_j w_ij), with positive C2-cutoff Gaussian weights.

    Repeated periodic images contribute by addition. No nearest-image assumption,
    new edges, cross-graph mixing, or learnable attention weights are introduced.
    ``T @ u - T.sum(-1)*u`` is the local difference used by the response. The
    extra 1 fixes the isolated-atom limit and bounds the row sum strictly below 1.
    """
    if cutoff <= 0 or width <= 0:
        raise ValueError('current transport cutoff and width must be positive')
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError('current edge_index must have shape [2, edges]')
    sender, receiver = edge_index
    if shifts is None:
        if sender.numel():
            raise ValueError('current transport requires the Cartesian image shifts of the MACE graph')
        shifts = positions.new_zeros((0, 3))
    if shifts.shape != (sender.numel(), 3):
        raise ValueError('current Cartesian edge shifts must have shape [edges, 3]')
    if sender.numel() and bool((batch[sender] != batch[receiver]).any()):
        raise ValueError('current electronic transport cannot connect different graphs')
    vector = positions[receiver] - positions[sender] + shifts.to(positions)
    square = vector.square().sum(-1)
    # Use r^2, not sqrt(r^2), so zero-length periodic self edges have finite
    # derivatives too. (1-r^2/R^2)^3_+ is C2 at the cutoff for force training.
    envelope = (1. - square / float(cutoff)**2).clamp_min(0.).pow(3)
    weights = torch.exp(-.5 * square / float(width)**2) * envelope
    indices = ((batch[receiver] * max_nodes + slot[receiver]) * max_nodes + slot[sender])
    matrix = positions.new_zeros(graphs * max_nodes * max_nodes).index_add(0, indices, weights)
    matrix = matrix.reshape(graphs, max_nodes, max_nodes)
    return matrix / (1. + matrix.sum(-1, keepdim=True))

def transport_difference(matrix, values):
    """Neighbor-minus-center differences, for either scalar or vector channels."""
    original = values.shape
    flat = values.flatten(2)
    return (torch.bmm(matrix, flat) - matrix.sum(-1, keepdim=True)*flat).reshape(original)

def angular_transport(positions, edge_index, shifts, batch, slot, graphs, max_nodes,
                      cutoff, width):
    """l=0 and l=1 transport on the SAME existing graph, no learned attention.

    Channel 0 is exactly local_transport. Channels 1:4 contain T_ij*r_ij/width,
    with r_ij pointing from receiving atom i to neighbor/image j. Solid l=1
    harmonics avoid the non-differentiable unit direction at a coincident edge.
    The same smooth cutoff supplies second derivatives for force training.
    """
    scalar = local_transport(positions, edge_index, shifts, batch, slot, graphs,
                             max_nodes, cutoff, width)
    sender, receiver = edge_index
    shifts = positions.new_zeros((0, 3)) if shifts is None else shifts.to(positions)
    vector = positions[sender]-positions[receiver]-shifts
    square = vector.square().sum(-1)
    weights = torch.exp(-.5*square/float(width)**2)*(1.-square/float(cutoff)**2).clamp_min(0.).pow(3)
    index = (batch[receiver]*max_nodes+slot[receiver])*max_nodes+slot[sender]
    raw = positions.new_zeros((graphs*max_nodes*max_nodes, 3)).index_add(
        0, index, weights[:, None]*vector/float(width)).reshape(graphs,max_nodes,max_nodes,3)
    denom = positions.new_zeros(graphs*max_nodes*max_nodes).index_add(0,index,weights)
    denom = 1.+denom.reshape(graphs,max_nodes,max_nodes).sum(-1,keepdim=True)
    return torch.cat((scalar[..., None],raw/denom[..., None]),-1)

def angular_differences(operator, scalar_values, vector_values):
    """(l=1 x vector)->scalar and (l=1 x scalar)->vector neighbor differences."""
    directional = operator[..., 1:]
    row = directional.sum(2)
    div = torch.einsum('bijr,bjcr->bic',directional,vector_values)
    div = div-(row[:,:,None]*vector_values).sum(-1)
    grad = torch.einsum('bijr,bjc->bicr',directional,scalar_values)
    grad = grad-row[:,:,None]*scalar_values[...,None]
    return div,grad

def vector_asinh(vector):
    """Invertible, rotation-equivariant counterpart of scalar asinh.

    Preserve the linear small-field response and compress large *descriptors*,
    not physical fields, moments or SCF iterates. The Taylor branch avoids a
    sqrt(0) derivative in force/force-loss differentiation. Its switch is set
    by floating-point precision, not a model or training hyperparameter.
    """
    squared = vector.square().sum(-1, keepdim=True)
    threshold = math.sqrt(torch.finfo(vector.dtype).eps)
    norm = squared.clamp_min(threshold).sqrt()
    ratio = torch.asinh(norm) / norm
    small = squared.clamp_max(threshold)
    series = 1. - small / 6. + 3. * small.square() / 40.
    return vector * torch.where(squared < threshold, series, ratio)

def sample_spectrum(total, geometry):
    """Gaussian-projected v and -grad(v), both in electron-energy convention.

    -grad(v) is an electron-energy gradient, NOT the electrical field -grad(phi).
    This explicit distinction avoids a hidden electron/positive-charge sign flip.
    """
    cosine, sine, wave, _, _, _, receiver, *_ = geometry
    re = total[..., 0, None] * receiver
    im = total[..., 1, None] * receiver
    re_k = (re[..., None] * wave[:, :, None]).flatten(-2)
    im_k = (im[..., None] * wave[:, :, None]).flatten(-2)
    # Two contractions share the phase transpose and its backward graph for
    # scalar and vector observations. This is the same Fourier projection.
    sampled = (torch.bmm(cosine.transpose(1, 2), torch.cat((re, im_k), -1))
               + torch.bmm(sine.transpose(1, 2), torch.cat((-im, re_k), -1)))
    value, gradient = sampled.split((re.shape[-1], re_k.shape[-1]), -1)
    return value, gradient.reshape(*value.shape, 3)

def charge_closure(reference_charge, levels, softness, target, mask):
    """Exact local quadratic KKT solution with a neutral chemical reference.

    min_q sum_i (q_i-q0_i)^2/(2*s_i) - eps_i*(q_i-q0_i), sum_i q_i=Q.
    q0 is projected to neutrality, not independently fitted to the Fermi label.
    This is a conditional response law, not a claim of total-energy stationarity.
    """
    count = mask.sum(-1).clamp_min(1.)
    q0 = (reference_charge - (reference_charge * mask).sum(-1)[:, None] / count[:, None]) * mask
    s = softness * mask
    response = s.sum(-1)
    # Positive softness and at least one atom make this denominator nonzero.
    mu = ((s * levels).sum(-1) - target) / response
    q = (q0 + s * (levels - mu[:, None])) * mask
    # Remove only arithmetic roundoff, not a separate constitutive correction.
    q = q + mask * ((target - q.sum(-1)) / count)[:, None]
    return q, mu, s / response[:, None], response, q0

class EmptyProjection(nn.Module):
    """A zero-dimensional observation has no trainable parameters."""
    bias = None
    def __init__(self, input_size):
        super().__init__()
        self.register_buffer("weight", torch.empty((0,input_size)))
    def forward(self,x):
        return x[..., :0]

class AtomicPotentialResponse(nn.Module):
    electrochemical_reference = True

    def __init__(self, scalar_channels, vector_channels, num_elements,
                 source_channels=2, field_channels=3, density_width=1.5, width=64,
                 separate_chemical_level=True, include_local_energy=False):
        super().__init__()
        if min(scalar_channels,num_elements,field_channels,width)<1 or min(source_channels,vector_channels)<0:
            raise ValueError('Invalid scalar/vector/radial response dimensions')
        if not math.isfinite(density_width) or density_width<=0:
            raise ValueError('density_width must be finite and positive')
        # Record the current constitutive law in checkpoints, without GPU
        # scalar branches inside the recurrent map. Older laws are not run.
        for name in ('screened_reference_initialization', 'regular_field_coordinates',
                     'geometry_softness', 'regular_geometry_coordinates',
                     'coarse_density_energy', 'geometry_charge_is_level', 'linear_induced_dipoles'):
            self.register_buffer(name, torch.tensor(True))
        self.source_channels=int(source_channels)
        self.field_channels=int(field_channels)
        self.density_width=float(density_width)
        self.vector_width=min(16,max(4,int(vector_channels)))
        self.vector_basis_size=2*self.vector_width
        self.chemical_scalar=nn.Linear(int(scalar_channels),width)
        self.chemical_vector=(nn.Linear(int(vector_channels),self.vector_width,bias=False)
                              if vector_channels else None)
        self.shared=nn.Sequential(nn.Linear(width+2*self.vector_width,width),nn.SiLU(),
                                  nn.Linear(width,width),nn.SiLU())
        self.vector_out=nn.Linear(width,(1+source_channels)*self.vector_basis_size)
        self.species_level=nn.Parameter(torch.zeros(num_elements))
        self.scalar_out=nn.Linear(width,2+source_channels)
        self.dipole_hardness=nn.Linear(width,1)
        nn.init.zeros_(self.dipole_hardness.weight)
        nn.init.zeros_(self.dipole_hardness.bias)
        self.energy_readout=(nn.Sequential(nn.Linear(width+2*self.vector_width,128),nn.SiLU(),
                                         nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1,bias=False))
                             if include_local_energy else None)
        self.state_scalar=nn.Linear(field_channels,width,bias=False)
        self.state_vector=nn.Linear(field_channels,self.vector_width,bias=False)
        self.neighbor_scalar=nn.Linear(field_channels,width,bias=False)
        self.neighbor_vector=nn.Linear(field_channels,self.vector_width,bias=False)
        self.neighbor_divergence=nn.Linear(field_channels,width,bias=False)
        self.neighbor_gradient=nn.Linear(field_channels,self.vector_width,bias=False)
        # The uniform chemical level is a null direction of fixed-Q charge
        # equilibration. Give that identifiable EF coordinate its own linear
        # projection of the SAME geometry/total-field hidden representation.
        # There is no extra field, vacuum/WF head, detached feature or SCF state.
        self.common_level = (nn.Linear(width, 1, bias=False)
                             if separate_chemical_level else None)
        if self.common_level is not None:
            nn.init.zeros_(self.common_level.weight)
        self.local_source_scalar=(nn.Linear(width,source_channels,bias=False) if source_channels else EmptyProjection(width))
        self.local_source_vector=(nn.Linear(self.vector_width,source_channels,bias=False) if source_channels else EmptyProjection(self.vector_width))
        self.species_source=nn.Parameter(torch.zeros(num_elements,source_channels))
        self.register_buffer('shared_reference_readout',torch.tensor(True))
        self.register_buffer('softness_scale',torch.tensor(.02))
        self.register_buffer('scalar_field_scale',torch.ones(field_channels))
        self.register_buffer('vector_field_scale',torch.ones(field_channels))
        self.register_buffer('field_scales_initialized',torch.tensor(False))
        self.register_buffer('geometry_scalar_unit',torch.ones(int(scalar_channels)))
        self.register_buffer('geometry_vector_unit',torch.ones(int(vector_channels)))
        self.register_buffer('geometry_units_initialized',torch.tensor(False))
        zero_layers = [self.scalar_out,self.vector_out,self.local_source_scalar,self.local_source_vector,
                       self.neighbor_scalar,self.neighbor_vector,self.neighbor_divergence,self.neighbor_gradient]
        if self.energy_readout is not None:
            zero_layers.append(self.energy_readout[-1])
        for layer in zero_layers:
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:nn.init.zeros_(layer.bias)
        # Local moment deviations enter only the terminal energy functional.
        # Zero weights reproduce the old energy/force predictor at construction,
        # without consuming random draws or changing the electronic update map.
        self.energy_charge_embedding = nn.Parameter(torch.zeros(width)) if include_local_energy else None
        self.energy_dipole_embedding = nn.Parameter(torch.zeros(self.vector_width)) if include_local_energy else None


    def geometry_features(self,scalars,vectors):
        if scalars.shape[-1]!=self.geometry_scalar_unit.numel() or vectors.shape[-2]!=self.geometry_vector_unit.numel():
            raise ValueError('Geometry irreps and stored numerical units disagree')
        chemical=self.chemical_scalar(torch.asinh(scalars)/self.geometry_scalar_unit)
        scaled_vector = vectors/self.geometry_vector_unit[:,None]
        scaled_vector = vector_asinh(scaled_vector)
        vector=(vectors.new_zeros((*vectors.shape[:-2],self.vector_width,3)) if self.chemical_vector is None
                else self.chemical_vector(scaled_vector.transpose(-1,-2)).transpose(-1,-2))
        return chemical,vector

    def state_embedding(self,state,chemical,vector,v,e,neighborhood=None):
        # State coefficients do NOT form an extra neural input. Their physical
        # information enters solely through the common projected v and -grad(v).
        if neighborhood is None or neighborhood.ndim!=4 or neighborhood.shape[-1]!=4:
            raise ValueError('The common response requires its periodic local transport operator')
        # The full receiver coordinates carry the electrochemical drive;
        # local radial contrasts and angular/neighbor variation remain present.
        # Scalar and vector channels use one linear transport and one adjoint.
        transported=transport_difference(neighborhood[...,0],torch.cat((v,e.flatten(-2)),-1))
        ds,dv=transported.split((v.shape[-1],e.shape[-2]*3),-1)
        dv=dv.reshape_as(e)
        div,grad=angular_differences(neighborhood,v,e)
        e,dv,grad=vector_asinh(e),vector_asinh(dv),vector_asinh(grad)
        scalar=chemical+self.state_scalar(torch.asinh(v))
        embedded=vector+self.state_vector(e.transpose(-1,-2)).transpose(-1,-2)
        scalar=scalar+self.neighbor_scalar(torch.asinh(ds))+self.neighbor_divergence(torch.asinh(div))
        embedded=embedded+self.neighbor_vector(dv.transpose(-1,-2)).transpose(-1,-2)
        embedded=embedded+self.neighbor_gradient(grad.transpose(-1,-2)).transpose(-1,-2)
        return scalar,embedded

    def potential_coordinates(self, potential, mask, electronic_reference, target):
        """Descriptors relative to an analytic reference charge multiplier.

        This mu0 is a descriptor reference, not the final predicted Fermi level.
        Geometry-only chi0,s0 define q0+s0*(chi0+v0-mu0), whose sum is Q.
        Keeping all tensors attached makes derivatives include this alignment.
        """
        if electronic_reference.shape != (*mask.shape, 2):
            raise ValueError('Missing geometry-only electronic reference [chi0,s0]')
        chi0, softness0 = electronic_reference.unbind(-1)
        soft = softness0 * mask
        total = soft.sum(-1)
        mu0 = ((soft*(chi0+potential[...,0])).sum(-1)-target)/total
        return (chi0[...,None]+potential-mu0[:,None,None])/self.scalar_field_scale

    def _invariants(self,state,chemical,vector,potential,field,mask,neighborhood,
                    electronic_reference,target):
        v=self.potential_coordinates(potential,mask,electronic_reference,target)
        e=field/self.vector_field_scale[:,None]
        scalar,embedded=self.state_embedding(state,chemical,vector,v,e,neighborhood)
        inv=torch.cat((scalar,torch.asinh(embedded.square().sum(-1)),
                       torch.asinh((vector*embedded).sum(-1))),-1)
        return inv,embedded

    def energy_invariants(self, invariants, embedded, vector, state, reference_charge, p0):
        """Equivariant local moment embedding for the nonvariational energy.

        Charges and scaled dipoles are in elementary-charge units. The reference
        has neutral geometry-predicted charges and the geometry-predicted dipole.
        No inferred atomic state is used as a training label or detached here.
        """
        delta_q = state[..., 0] - reference_charge
        delta_d = state[..., 1:4] - p0[..., 1:4] / self.density_width
        scalar = (invariants[..., :self.energy_charge_embedding.numel()]
                  + delta_q[..., None] * self.energy_charge_embedding)
        polar = embedded + delta_d[..., None, :] * self.energy_dipole_embedding[:, None]
        return torch.cat((scalar, torch.asinh(polar.square().sum(-1)),
                          torch.asinh((vector * polar).sum(-1))), -1)

    def _constitutive_outputs(self,invariants,vector,embedded,return_hidden=False,include_dipole=True):
        u1 = self.shared[0](invariants)
        u2 = self.shared[2](self.shared[1](u1))
        hidden = self.shared[3](u2)
        scalars=self.scalar_out(hidden)
        channels=self.source_channels+int(include_dipole)
        if channels:
            basis=torch.cat((vector,embedded),-2)
            # The geometry dipole is prepared once in reference_offset. Only
            # local-potential vector amplitudes use this readout in recurrence;
            # density-only models therefore need no recurrent vector readout.
            start=0 if include_dipole else self.vector_basis_size
            gate=F.linear(hidden,self.vector_out.weight[start:],self.vector_out.bias[start:])
            gate=gate.reshape(*hidden.shape[:-1],channels,self.vector_basis_size)
            polar=torch.einsum('bnov,bnvc->bnoc',gate,basis)/math.sqrt(self.vector_basis_size)
        else:
            polar=hidden.new_zeros((*hidden.shape[:-1],0,3))
        result = (scalars, polar)
        if return_hidden:
            result += (hidden,)
        return result

    def chemical_levels(self, raw_level, hidden, attrs, softness, mask):
        """Split local contrasts and the uniform level in the same KKT law.

        chi_i = raw_i - <raw>_s + <chi_common>_s,
        mu = <chi_common>_s + <v>_s - Q/sum(s).
        A common-readout change shifts mu without changing any charge. Field
        and geometry derivatives through the shared features remain attached.
        This conditional multiplier is not asserted to equal dE/dN_e.
        """
        common_readout = self.common_level
        if common_readout is None:
            return raw_level
        weights = softness * mask
        weights = weights / weights.sum(-1, keepdim=True)
        reference = attrs @ self.species_level + common_readout(hidden).squeeze(-1)
        return raw_level + (weights * (reference - raw_level)).sum(-1, keepdim=True)

    def geometry_reference_features(self, chemical, vector):
        """The shared local electronic representation with no applied field.

        This geometry-only hidden vector already defines the reference levels
        and softness. Reusing it for scalar potential amplitudes avoids forcing
        structural potential variation through the induced-response difference.
        It is atom-local, has no label/graph pooling and adds no layer or weight.
        """
        norm = torch.asinh(vector.square().sum(-1))
        return self.shared(torch.cat((chemical, norm, norm), -1))

    def local_potential_coordinates(self, chemical, vector, attrs, mask, *, hidden=None):
        hidden = self.geometry_reference_features(chemical, vector) if hidden is None else hidden
        scalar = attrs @ self.species_source + self.local_source_scalar(hidden)
        polar = self.local_source_vector(vector.transpose(-1, -2)).transpose(-1, -2)
        return torch.cat((scalar, polar.flatten(-2)), -1) * mask[..., None]

    def prepare_reference(self,state,chemical,vector,attrs,p0,mask,target,neighborhood):
        # Geometry-only coefficients. Everything remains differentiable with
        # respect to positions and parameters; no learned field is supplied here.
        hidden = self.geometry_reference_features(chemical, vector)
        initial_scalar = self.scalar_out(hidden)
        chi0 = attrs@self.species_level + initial_scalar[...,0]
        chi0 = chi0 + p0[..., 0]
        softness0 = self.softness_scale*(F.softplus(initial_scalar[...,1]+math.log(math.expm1(1.)))+1.e-6)
        electronic_reference = torch.stack((chi0,softness0),-1)
        zeros=state.new_zeros((*state.shape[:2],self.field_channels))
        inv,embedded=self._invariants(state,chemical,vector,zeros,
            zeros[...,None].expand(-1,-1,-1,3),mask,neighborhood,electronic_reference,target)
        scalar,polar=self._constitutive_outputs(inv,vector,embedded)
        local=self.local_potential_coordinates(chemical,vector,attrs,mask,hidden=hidden)
        raw=torch.cat((scalar[...,2:],polar[...,1:,:].flatten(-2)),-1)
        offset=torch.cat((torch.zeros_like(state[...,:1]),polar[...,0,:],local-raw),-1)*mask[...,None]
        dipole_soft=self.softness_scale*(F.softplus(self.dipole_hardness(hidden)+math.log(math.expm1(1.)))+1.e-6)
        return local,torch.cat((offset,dipole_soft,electronic_reference),-1)


    def forward(self,state,chemical,vector,attrs,potential,field,p0,mask,target,
                neighborhood=None,*,include_energy=True,reference_offset=None,
                observe_chemical_level=True):
        extra=3
        if reference_offset is None or reference_offset.shape!=(*state.shape[:-1],state.shape[-1]+extra):
            raise ValueError('Missing/mismatched differentiable geometry/electrochemical reference')
        offset,electronic_reference=reference_offset[...,:-extra],reference_offset[...,-2:]
        inv,embedded=self._invariants(state,chemical,vector,potential,field,mask,neighborhood,
                                     electronic_reference,target)
        computed = self._constitutive_outputs(inv, vector, embedded, return_hidden=True,include_dipole=False)
        scalars, polar = computed[:2]
        chemical_level=attrs@self.species_level+scalars[...,0]
        chemical_level = chemical_level + p0[..., 0]
        softness=electronic_reference[...,1]
        levels=chemical_level+potential[...,0]
        reference_charge = torch.zeros_like(p0[..., 0])
        q,mu,f,response,neutral=charge_closure(reference_charge,levels,softness,target,mask)
        # The common chemical level is exactly a fixed-Q null coordinate.
        # Keep it out of the charge calculation instead of adding/subtracting
        # a large common value through every atom and every recurrent step.
        if observe_chemical_level and self.common_level is not None:
            shifted = self.chemical_levels(chemical_level, computed[-1], attrs, softness, mask)
            mu = mu+(f*(shifted-chemical_level)).sum(-1)
            chemical_level = shifted
            levels = chemical_level+potential[...,0]
        induced=-reference_offset[...,-3:-2]*self.density_width*field[...,0,:]
        proposal=torch.cat((q[...,None],p0[...,1:4]/self.density_width+induced,
                            scalars[...,2:],polar.flatten(-2)),-1)*mask[...,None]
        proposal=(proposal+offset)*mask[...,None]
        if include_energy and self.energy_readout is not None:
            zeros=torch.zeros_like(potential)
            refinv,_=self._invariants(state,chemical,vector,zeros,torch.zeros_like(field),mask,neighborhood,
                                      electronic_reference,target)
            energy_inv = self.energy_invariants(inv, embedded, vector, state, neutral, p0)
            energy=(self.energy_readout(energy_inv)-self.energy_readout(refinv)).squeeze(-1)*mask
        else:energy=state.new_zeros(mask.shape)
        return proposal,mu,f,response,levels,chemical_level,energy,neutral

def coefficient_kernels(geometry,density_width,source_volume,base_width):
    """Geometry-only kernels prepared once; no K x atom x basis tensor."""
    co,si,wave,br,bi,radial,receiver,proto,ramp,applied,zn,normal,poisson,ivol=geometry
    scales=br.new_tensor([1.,density_width,density_width,density_width])
    tail_re=br*poisson[...,None]*scales
    tail_im=bi*poisson[...,None]*scales
    shape_re=radial*source_volume
    shape_im=(radial[...,None]*wave[:,:,None,:]*(base_width*source_volume)).flatten(-2)
    real=torch.cat((tail_re,shape_re,torch.zeros_like(shape_im)),-1)*ivol[:,None,None]
    imag=torch.cat((tail_im,torch.zeros_like(shape_re),shape_im),-1)*ivol[:,None,None]
    return real,imag

def coefficient_contributions(state,geometry,kernels):
    """The actual moment-carrier and regular contributions, without subtraction."""
    co,si=geometry[:2]
    kr,ki=kernels
    if state.shape[-1]!=kr.shape[-1] or kr.shape!=ki.shape:
        raise ValueError('Atomic potential coefficient/basis dimensions disagree')
    cosine_coeff=torch.bmm(co,state)
    sine_coeff=torch.bmm(si,state)
    real=kr*cosine_coeff+ki*sine_coeff
    imag=ki*cosine_coeff-kr*sine_coeff
    carrier=torch.stack((real[...,:4].sum(-1),imag[...,:4].sum(-1)),-1)
    shape=torch.stack((real[...,4:].sum(-1),imag[...,4:].sum(-1)),-1)
    return carrier,shape

def evaluate_coefficients(state,geometry,kernels,density_width):
    """Two node-to-Fourier matrix products replace six separate contractions.

    Returns the observer tuple, whose pieces are diagnostic contributions of
    ONE basis. No carrier is reconstructed by subtracting a regular field from
    the potential derived back from its density.
    """
    carrier,shape=coefficient_contributions(state,geometry,kernels)
    proto,ramp,applied,zn,normal,poisson=geometry[7:13]
    internal=carrier+shape
    source=torch.where(poisson[...,None]!=0.,internal/torch.where(poisson[...,None]!=0.,poisson[...,None],1.),0.)
    dipole_normal=(state[...,0]*zn+(state[...,1:4]*density_width*normal[:,None]).sum(-1)).sum(-1)
    boundary=dipole_normal[:,None,None]*ramp
    carrier_with_boundary=carrier+boundary+applied
    total=carrier_with_boundary+shape+proto
    return total,carrier_with_boundary,shape,source


def total_spectrum(state, geometry, kernels, density_width):
    """Recurrent field only; component observers are assembled once at the end."""
    cosine, sine = (torch.bmm(value, state) for value in geometry[:2])
    real, imag = kernels
    internal = torch.stack(((real*cosine+imag*sine).sum(-1),
                            (imag*cosine-real*sine).sum(-1)), -1)
    proto, ramp, applied, z, normal = geometry[7:12]
    dipole = (state[..., 0]*z+(state[..., 1:4]*density_width*normal[:, None]).sum(-1)).sum(-1)
    return internal+dipole[:, None, None]*ramp+applied+proto


def ensure_cuda_linalg(device):
    """Select cuSOLVER before any CUDA solve, including backward solves.

    PyTorch's default heuristics can choose MAGMA for large batched matrices.
    The v365 failure reports include a fatal MAGMA pointer-array error. Select
    the supported cuSOLVER/cuBLAS route explicitly and keep that process-wide
    preference for later autograd/audit calls. Never retry after a CUDA fault.
    CPU and other device backends are unchanged.
    """
    if torch.device(device).type != 'cuda':
        return
    preferred = torch.backends.cuda.preferred_linalg_library
    if preferred().name != 'Cusolver':
        preferred('cusolver')
        logging.info('SCF CUDA linear algebra: cuSOLVER/cuBLAS selected '
                     '(process-wide); exact cached matrix-solve derivatives; '
                     'torch=%s CUDA=%s', torch.__version__, torch.version.cuda)

def factor_linear_system(matrix):
    """Return checked, non-differentiated workspace for a live square matrix."""
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2] or matrix.shape[-1] == 0:
        raise ValueError('Linear operator must contain nonempty square matrices')
    ensure_cuda_linalg(matrix.device)
    with torch.no_grad():
        if not bool(torch.isfinite(matrix).all()):
            raise RuntimeError('Nonfinite constrained linear operator before LU factorization')
        lu, pivots, info = torch.linalg.lu_factor_ex(matrix)
        n = matrix.shape[-1]
        # Invalid pivots must never reach a native permutation/indexing kernel.
        valid = ((info == 0).all() & torch.isfinite(lu).all()
                 & (pivots >= 1).all() & (pivots <= n).all())
        if not bool(valid):
            raise RuntimeError('Constrained LU factorization failed its status/finite/pivot '
                               'checks; no shift or fallback applied')
    return lu, pivots

class _CachedLinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, lu, pivots, right, adjoint):
        ensure_cuda_linalg(matrix.device)
        solved = torch.linalg.lu_solve(lu, pivots, right, adjoint=adjoint)
        operator = matrix.mH if adjoint else matrix
        defect = operator @ solved - right
        scale = operator.abs() @ solved.abs() + right.abs()
        allowance = (64 * matrix.shape[-1] * torch.finfo(matrix.dtype).eps) * scale.clamp_min(1.)
        if not bool(torch.isfinite(solved).all() & (defect.abs() <= allowance).all()):
            raise RuntimeError('Constrained linear solve failed its finite/backward-error '
                               'check; no diagonal shift or solver substitution applied')
        ctx.adjoint = adjoint
        # Saving the output (not a detached internal copy) makes second and
        # higher derivatives include the solution's dependence on A and B.
        ctx.save_for_backward(matrix, lu, pivots, solved)
        return solved

    @staticmethod
    def backward(ctx, grad_solution):
        matrix, lu, pivots, solved = ctx.saved_tensors
        dual = _CachedLinearSolve.apply(matrix, lu, pivots, grad_solution, not ctx.adjoint)
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            grad_matrix = -(solved @ dual.mH if ctx.adjoint else dual @ solved.mH)
        return grad_matrix, None, None, dual if ctx.needs_input_grad[3] else None, None

def solve_factored_system(matrix, lu, pivots, right):
    """Solve using factors from THIS matrix/evaluation; no batch broadcasting.

    Keep matrix attached. Factors are reusable workspace and carry no independent
    derivative; all operator dependence goes through the exact solve identity.
    """
    if (matrix.shape != lu.shape or pivots.shape != matrix.shape[:-1]
            or right.ndim != matrix.ndim or right.shape[:-1] != matrix.shape[:-1]
            or pivots.dtype != torch.int32):
        raise ValueError('Incompatible matrix, LU, pivot or right-hand-side shapes/dtype')
    if (matrix.device != lu.device or matrix.device != pivots.device or matrix.device != right.device
            or matrix.dtype != lu.dtype or matrix.dtype != right.dtype):
        raise ValueError('Linear solve inputs must share device and floating dtype')
    return _CachedLinearSolve.apply(matrix, lu.detach(), pivots, right, False)

def prepare_charge_factorization(softness, kernel, mask, constraint=None):
    """Factor the constrained geometry-only charge operator once per forward.

    The matrix and scaling retain parameter/coordinate derivatives. Numerical
    LU workspace is detached; solve_factored_system supplies the exact matrix
    derivative, including double backward. Nothing is cached across calls.
    LU handles the existing generally nonsymmetric Fourier slab operator;
    no symmetrization, diagonal shift, or alternate boundary convention is used.
    The optional constraint selects charge coordinates in a joint moment solve;
    dipoles then remain unconstrained. The default constrains every coordinate.
    """
    present = mask.bool()
    if kernel.shape != (*mask.shape, mask.shape[-1]) or softness.shape != mask.shape:
        raise ValueError('Charge kernel/softness dimensions disagree with node mask')
    if not bool(torch.isfinite(kernel).all() & torch.isfinite(softness).all()) or bool((softness[present] <= 0).any()):
        raise RuntimeError('Charge kernel/softness must be finite, with positive atomic softness')
    s = softness * mask
    total = s.sum(-1, keepdim=True)
    root = torch.sqrt(torch.where(present, s/total, torch.ones_like(s))) * mask
    n = mask.shape[-1]
    matrix = (kernel * root[..., None] * root[:, None, :]) * total[..., None]
    matrix = matrix + torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    constraint = mask if constraint is None else constraint
    if constraint.shape != mask.shape or not bool(torch.isfinite(constraint).all()) or bool((constraint.square().sum(-1)==0).any()):
        raise ValueError('Each graph needs a finite, nonzero constraint with the coordinate-mask shape')
    border = root * constraint
    upper = torch.cat((matrix, border[..., None]), -1)
    lower = torch.cat((border, torch.zeros_like(root[:, :1])), -1)[:, None, :]
    bordered = torch.cat((upper, lower), -2)
    lu, pivots = factor_linear_system(bordered)
    return bordered, root, lu, pivots

def moment_kernel(geometry, kernels, mask, sigma):
    co,si,wave=geometry[:3]
    kr,ki=(k[..., :4] for k in kernels)
    source_re=(co[...,None]*kr[:,:,None,:]+si[...,None]*ki[:,:,None,:]).flatten(-2)
    source_im=(co[...,None]*ki[:,:,None,:]-si[...,None]*kr[:,:,None,:]).flatten(-2)
    receiver=geometry[6][...,0,None,None]
    observer_re=torch.cat((co[...,None],-sigma*si[...,None]*wave[:,:,None,:]),-1)*receiver
    observer_im=torch.cat((-si[...,None],-sigma*co[...,None]*wave[:,:,None,:]),-1)*receiver
    matrix=-(observer_re.flatten(-2).transpose(1,2)@source_re
             +observer_im.flatten(-2).transpose(1,2)@source_im)
    unit_v,unit_e=sample_spectrum(geometry[8],geometry)
    observed=torch.cat((unit_v[...,:1],-sigma*unit_e[...,0,:]),-1).flatten(-2)
    dipole=torch.cat((geometry[10][...,None],sigma*geometry[11][:,None,:].expand(-1,mask.shape[1],-1)),-1).flatten(-2)
    matrix=matrix-observed[...,None]*dipole[:,None,:]
    present=mask[...,None].expand(-1,-1,4).flatten(-2)
    return matrix*present[...,None]*present[:,None,:]

def moment_coordinates(reference, mask):
    # Last entries remain [dipole softness, chi0, charge softness].
    softness=torch.cat((reference[...,-1:],reference[...,-3:-2].expand(-1,-1,3)),-1)
    present=mask[...,None].expand_as(softness)
    constraint=torch.cat((mask[...,None],torch.zeros_like(softness[...,1:])), -1)
    return softness.flatten(-2),present.flatten(-2),constraint.flatten(-2)

def factor_moments(reference, kernel, mask):
    softness,present,constraint=moment_coordinates(reference,mask)
    return prepare_charge_factorization(softness,kernel,present,constraint)

def screen_moments(current, proposal, reference, kernel, mask, factors, *, materialized=False):
    """Apply the already validated geometry's constrained moment solve."""
    matrix, root, lu, pivots = factors
    present = mask[..., None].expand(-1, -1, 4).flatten(-2)
    old, raw = current[..., :4].flatten(-2), proposal[..., :4].flatten(-2)
    residual = (raw-old)*present
    rhs = residual/torch.where(present.bool(), root, torch.ones_like(root))
    charge_residual = residual.reshape(*mask.shape, 4)[..., 0].sum(-1, keepdim=True)
    right = torch.cat((rhs, charge_residual), -1)[..., None]
    solved = matrix @ right if materialized else solve_factored_system(matrix, lu, pivots, right)
    value = ((old+root*solved[:, :-1, 0])*present).reshape(*mask.shape, 4)
    # Remove charge roundoff only; the raw law already enforces fixed Q.
    charge = value[..., 0]+mask*((proposal[..., 0]-value[..., 0]).sum(-1)/mask.sum(-1))[:, None]
    return torch.cat((charge[..., None], value[..., 1:4], proposal[..., 4:]), -1)


@dataclass(frozen=True)
class RootOptions:
    max_steps: int = 100
    tolerance: float = 1.e-7
    mixing: float = .5
    history: int = 5
    linear_tolerance: float = 1.e-9
    linear_max_steps: int = 160
    linear_restart: int = 24
    fixed_steps: bool = False

    def __post_init__(self):
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps < 1:
            raise ValueError("Coupled SCF step count must be a positive integer")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0 or not 0 < self.mixing <= 1:
            raise ValueError("Invalid coupled fixed-point solver options")
        if not isinstance(self.fixed_steps, bool):
            raise TypeError("fixed_steps must be bool")
        if min(self.linear_max_steps, self.linear_restart) < 1 or self.linear_tolerance <= 0:
            raise ValueError("Invalid coupled adjoint solver options")

class SCFConvergenceError(RuntimeError):
    """A valid strict root iteration exhausted its convergence budget."""

class SCFNumericalError(RuntimeError):
    """The selected trajectory became nonfinite or catastrophically amplified.

    This is an error, not a clipped state, accepted earlier iterate, skipped batch,
    altered objective or solver-mode substitution.
    """

def _trajectory_scale(initial):
    # A precision-scaled runaway budget in the defined state coordinates, not
    # a physical bound or a theorem that smaller trajectories are stable. Large
    # state amplification precedes high-degree energy/force-loss contractions.
    # This fail-fast convention never changes an accepted finite trajectory.
    x=initial.detach()
    if not bool(torch.isfinite(x).all()):
        raise SCFNumericalError('SCF initial state is not finite')
    peak=x.abs().amax() if x.numel() else x.new_zeros(())
    return peak.clamp_min(1.) / math.sqrt(torch.finfo(x.dtype).eps)

def _check_trajectory(state,limit,step,kind):
    with torch.no_grad():
        peak=state.detach().abs().amax() if state.numel() else state.new_zeros(())
        # A single scalar decision covers NaN/Inf and amplitude. Avoid separate
        # device synchronizations for finiteness and magnitude on every step.
        if bool((~torch.isfinite(peak)) | (peak>limit)):
            flat=state.detach().abs().reshape(-1)
            index=int(torch.nan_to_num(flat,nan=float('inf'),posinf=float('inf')).argmax()) if flat.numel() else 0
            per_graph=state[0].numel() if state.ndim>=2 and len(state) else max(1,state.numel())
            graph=index//per_graph if state.ndim>=2 else 0
            detail = ""
            if state.ndim == 3 and state.shape[-1] >= 4 and (state.shape[-1]-4) % 4 == 0:
                c = (state.shape[-1]-4)//4
                sample = state[graph].detach()
                blocks = {"q": (0, 1), "d_scaled": (1, 4),
                          "scalar_potential": (4, 4+c), "vector_potential": (4+c, state.shape[-1])}
                peaks = {name: float(sample[:, a:b].abs().max()) for name, (a, b) in blocks.items() if b > a}
                detail = f" Failing-graph block maxima={peaks}."
            raise SCFNumericalError(f'{kind} SCF numerical runaway at step={step}, graph={graph}: '
                f'max|state|={float(peak):.6e}, fixed initial-scale precision budget={float(limit):.6e}. '
                'Finite-step unrolling is not a convergence solver. No state was clipped or substituted; '
                'check the requested update count and the nonlinear response. Do not relax spectrum parity.' + detail)

def _stopping_residual(function, state, args, proposal):
    physical = getattr(function, "convergence_residual", None)
    return proposal-state if physical is None else physical(state,*args)

def _norm(x):
    return torch.linalg.vector_norm(x.reshape(-1))

def solve_root(function: Callable, initial: torch.Tensor, args: Tuple[torch.Tensor, ...],
               options: RootOptions):
    """Safeguarded Anderson iteration.  The residual is F(x)-x, NOT a mixed step.

    Call in no_grad for implicit solving.  This routine does not detach inputs
    itself and is not used as a surrogate derivative for implicit force training.
    Nonconvergence is an error, never a silent rollback to another physical state.
    """
    x = initial.clone()
    limit = _trajectory_scale(initial)
    xs, fs = [], []
    previous_x = previous_f = None
    previous_error = float("inf")
    rejected = 0
    for step in range(options.max_steps + 1):
        _check_trajectory(x,limit,step,'strict-root iterate')
        f = function(x, *args)
        _check_trajectory(f,limit,step,'strict-root proposal')
        residual = f - x
        stopping = _stopping_residual(function, x, args, f)
        error = float(stopping.detach().abs().max()) if x.numel() else 0.
        if not torch.isfinite(residual).all() or not torch.isfinite(stopping).all():
            raise RuntimeError(f"Coupled SCF produced a nonfinite state at step {step}")
        if error <= options.tolerance:
            return x, {"iterations": step, "residual": error, "rejected": rejected}
        if step == options.max_steps:
            break
        # Reject an extrapolation that substantially worsens the fixed-point
        # residual.  A fresh damped step uses the *same* map and charge law.
        if previous_x is not None and error > 1.5 * previous_error and len(xs) > 1:
            x = previous_x + options.mixing * (previous_f - previous_x)
            xs, fs = [], []
            previous_x = previous_f = None
            rejected += 1
            continue
        previous_x, previous_f, previous_error = x, f, error
        xs.append(x); fs.append(f)
        if len(xs) > max(1, options.history):
            xs.pop(0); fs.pop(0)
        if len(xs) < 2 or options.history < 2:
            x = x + options.mixing * residual
            continue
        X = torch.stack([z.reshape(-1) for z in xs])
        F = torch.stack([z.reshape(-1) for z in fs])
        R = F - X
        gram = R @ R.T
        scale = gram.diagonal().mean().clamp_min(torch.finfo(x.dtype).tiny)
        gram = gram + (1.e-4 * scale) * torch.eye(len(xs), dtype=x.dtype, device=x.device)
        weights = torch.linalg.solve(gram, torch.ones(len(xs), dtype=x.dtype, device=x.device))
        denom = weights.sum()
        if not torch.isfinite(weights).all() or denom.abs() <= torch.finfo(x.dtype).eps:
            x = x + options.mixing * residual
            xs, fs = [], []
        else:
            weights = weights / denom
            x = ((1 - options.mixing) * (weights @ X) + options.mixing * (weights @ F)).reshape_as(x)
    raise SCFConvergenceError(
        f"Coupled SCF did not converge in {options.max_steps} steps: "
        f"max |F(x)-x|={error:.3e}, tolerance={options.tolerance:.3e}. "
        "Increase the SCF budget or reduce mixing; no state/backend was substituted.")

def gmres(operator: Callable, rhs: torch.Tensor, options: RootOptions):
    """Restarted matrix-free GMRES with a verified true residual.

    The Krylov vectors are constants during an implicit solve.  Derivatives of
    the solution are supplied by _LinearSolve, including its double backward.
    """
    shape = rhs.shape
    b = rhs.reshape(-1)
    x = torch.zeros_like(b)
    bnorm = _norm(b)
    if float(bnorm) == 0.:
        return x.reshape(shape)
    tol = max(options.linear_tolerance, 20. * torch.finfo(b.dtype).eps)
    target = tol * bnorm
    total = 0
    def matvec(v):
        return operator(v.reshape(shape)).reshape(-1)
    while total < options.linear_max_steps:
        r = b - matvec(x)
        beta = _norm(r)
        if beta <= target:
            return x.reshape(shape)
        length = min(options.linear_restart, options.linear_max_steps - total, b.numel())
        vectors = [r / beta]
        H = b.new_zeros((length + 1, length))
        # Small least-squares problems use normal QR/lstsq, never normal equations.
        for j in range(length):
            v = matvec(vectors[j])
            for _ in range(2):  # reorthogonalization matters near a solved root
                for k in range(j + 1):
                    coeff = torch.dot(vectors[k], v)
                    H[k, j] += coeff
                    v = v - coeff * vectors[k]
            hnext = _norm(v)
            H[j + 1, j] = hnext
            total += 1
            beta_e1 = b.new_zeros(j + 2); beta_e1[0] = beta
            small = H[:j + 2, :j + 1]
            # 'gels' is also supported on CUDA and Arnoldi columns have full rank
            # until happy breakdown.  QR handles that breakdown without a zero
            # trailing row being promoted to a column.
            coeffs = torch.linalg.lstsq(small, beta_e1, driver="gels").solution
            estimate = _norm(beta_e1 - small @ coeffs)
            candidate = x + torch.stack(vectors[:j + 1], dim=1) @ coeffs
            breakdown = float(hnext) <= 10. * torch.finfo(b.dtype).eps
            if estimate <= target or breakdown or j == length - 1:
                true_error = _norm(b - matvec(candidate))
                if true_error <= target:
                    return candidate.reshape(shape)
                if breakdown or j == length - 1:
                    x = candidate
                    break
            vectors.append(v / hnext)
    relative = float(_norm(b - matvec(x)) / bnorm)
    raise RuntimeError(f"Coupled SCF adjoint did not converge: relative residual={relative:.3e} "
                       f"after {total} Krylov steps (required {tol:.3e})")

def _linear_operator(function, transpose, state, args):
    """A = I-dF/dx or its transpose at frozen state/inputs."""
    with torch.enable_grad():
        z = state.detach().requires_grad_(True)
        f = function(z, *(a.detach() for a in args))
        if not f.requires_grad:
            return lambda v: v
        if transpose:
            def operator(v):
                with torch.enable_grad():
                    jt = torch.autograd.grad(f, z, v, retain_graph=True, allow_unused=True)[0]
                return v if jt is None else v - jt
        else:
            # Reverse-over-reverse JVP works for operations without forward-AD
            # kernels.  It does not materialize a Jacobian or a Hessian.
            seed = torch.zeros_like(f, requires_grad=True)
            jt_seed = torch.autograd.grad(f, z, seed, create_graph=True, retain_graph=True,
                                          allow_unused=True)[0]
            def operator(v):
                if jt_seed is None or not jt_seed.requires_grad:
                    return v
                with torch.enable_grad():
                    jv = torch.autograd.grad(jt_seed, seed, v, retain_graph=True,
                                             allow_unused=True)[0]
                return v if jv is None else v - jv
    return operator

def _proxies(values):
    # An intermediate proxy makes the following autograd.grad a PARTIAL
    # derivative, while preserving original-input dependence for double backward.
    # Leaf constants also need a proxy for the local Jacobian, but no derivative
    # is returned to a non-differentiable original input.
    return [x + 0. if x.requires_grad else x.detach() for x in values]

class _LinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, options, transpose, rhs, state, *args):
        solution = gmres(_linear_operator(function, transpose, state, args), rhs, options)
        ctx.function, ctx.options, ctx.transpose = function, options, transpose
        ctx.save_for_backward(solution, rhs, state, *args)
        return solution

    @staticmethod
    def backward(ctx, grad_solution):
        solution, rhs, state, *args = ctx.saved_tensors
        higher = torch.is_grad_enabled()
        with torch.enable_grad():
            adjoint = _LinearSolve.apply(ctx.function, ctx.options, not ctx.transpose,
                                        grad_solution, state, *args)
            z, *xs = _proxies([state, *args])
            f = ctx.function(z, *xs)
            left, right = (solution, adjoint) if ctx.transpose else (adjoint, solution)
            jt_left = (torch.autograd.grad(f, z, left, create_graph=True,
                                         retain_graph=True, allow_unused=True)[0]
                       if f.requires_grad and z.requires_grad else None)
            if jt_left is None or not jt_left.requires_grad:
                derivatives = [None] * (1 + len(args))
            else:
                variables = [z, *xs]
                chosen = [i for i, value in enumerate(variables) if value.requires_grad]
                calculated = torch.autograd.grad((jt_left * right).sum(),
                    [variables[i] for i in chosen], create_graph=higher, allow_unused=True)
                derivatives = [None] * len(variables)
                for i, derivative in zip(chosen, calculated): derivatives[i] = derivative
        derivatives = tuple(d if original.requires_grad else None
                            for d, original in zip(derivatives, [state, *args]))
        return (None, None, None, adjoint, *derivatives)

class _ImplicitRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, options, initial, *args):
        if options.fixed_steps:
            raise ValueError("implicit_root cannot differentiate an unconverged finite trajectory; "
                             "use unroll_fixed_steps explicitly")
        state, stats = solve_root(function, initial, args, options)
        # The root belongs to the physical constitutive equation, not the
        # nonlinear preconditioning path. Differentiate that equation directly.
        # The forward solver has already checked its raw residual.
        ctx.function, ctx.options = getattr(function, "raw", function), options
        ctx.save_for_backward(state, *args)
        # Statistics are not model inputs, never part of a training target.
        info = initial.new_tensor([stats["iterations"], stats["residual"], stats["rejected"]])
        ctx.mark_non_differentiable(info)
        return state, info

    @staticmethod
    def backward(ctx, grad_state, grad_info):
        state, *args = ctx.saved_tensors
        higher = torch.is_grad_enabled()
        with torch.enable_grad():
            z, *xs = _proxies([state, *args])
            f = ctx.function(z, *xs)
            adjoint = _LinearSolve.apply(ctx.function, ctx.options, True, grad_state,
                                        state, *args)
            chosen = [i for i, value in enumerate(xs) if value.requires_grad]
            calculated = (torch.autograd.grad(f, [xs[i] for i in chosen], adjoint,
                                            create_graph=higher, allow_unused=True)
                          if chosen and f.requires_grad else [None] * len(chosen))
            derivatives = [None] * len(xs)
            for i, derivative in zip(chosen, calculated): derivatives[i] = derivative
        derivatives = tuple(d if a.requires_grad else None for d, a in zip(derivatives, args))
        return (None, None, None, *derivatives)

def implicit_root(function, initial, *args, options=None):
    """Return a verified root and [iterations, residual, rejected extrapolations].

    Every differentiable dependency MUST be passed in args. The starting guess
    is not an implicit variable. A failed root or verified adjoint solve is an
    error, not an approximate gradient, restored state, or backend substitution.
    """
    options = options or RootOptions()
    if options.fixed_steps:
        raise ValueError('implicit_root requires a converged root, not fixed_steps=True')
    return _ImplicitRoot.apply(function, options, initial, *args)


class CoupledResponse(AtomicPotentialResponse):
    """Shared nonlinear response with the v366 charge/chemical-level split."""
    coupled = True
    spectral = True

    def __init__(self, node_feats_irreps, charges_irreps, num_elements,
                 potential_widths=(), density_width=1.5, field_widths=(1.5, 3.),
                 include_local_energy=False, **kwargs):
        irreps = o3.Irreps(node_feats_irreps)
        if o3.Irreps(charges_irreps).dim != 4:
            raise ValueError("CoupledResponse uses monopoles and dipoles (atomic_multipoles_max_l=1)")
        source_widths = sorted(set(float(w) for w in potential_widths))
        widths = [float(density_width), 0.]+sorted(set(float(w) for w in field_widths)-{float(density_width)})
        if any(not math.isfinite(w) or w <= 0 for w in source_widths+widths[2:]):
            raise ValueError("Gaussian widths must be finite and positive")
        scalar_slices, vector_slices = [], []
        for (mul, ir), sl in zip(irreps, irreps.slices()):
            if ir == o3.Irrep('0e'):
                scalar_slices.append((sl.start, sl.stop))
            elif ir == o3.Irrep('1o'):
                vector_slices.append((sl.start, sl.stop))
        scalar_channels = sum(b-a for a,b in scalar_slices)
        vector_channels = sum((b-a)//3 for a,b in vector_slices)
        super().__init__(scalar_channels, vector_channels, num_elements,
                         source_channels=len(source_widths), field_channels=len(widths),
                         density_width=density_width, separate_chemical_level=bool(source_widths),
                         include_local_energy=include_local_energy)
        self.scalar_slices, self.vector_slices = scalar_slices, vector_slices
        self.coarse_dim = 4
        # SpectralGeometry prepares the moment carrier. The regular-potential
        # basis below retains v366's normalization and radial-difference basis.
        self.potential_widths = ()
        channels=len(source_widths)
        self.state_irreps = o3.Irreps(f'0e+1o+{channels}x0e+{channels}x1o').simplify()
        self.register_buffer('source_widths', torch.tensor(source_widths))
        self.register_buffer('receiver_widths', torch.tensor(widths))
        self.register_buffer('feature_norms', torch.ones(irreps.dim))
        self.register_buffer('proto_coefficients', torch.zeros(num_elements, 64))
        self.register_buffer('proto_fitted', torch.tensor(False))
        self.register_buffer('spectral_cutoff', torch.tensor(1.))
        self.register_buffer('deployment_steps', torch.tensor(50))
        self.register_buffer('deployment_mixing', torch.tensor(.5))
        self.deployment_mode = 'unroll_scf'

    def get_extra_state(self):
        return {'deployment_mode': self.deployment_mode}

    def set_extra_state(self, state):
        mode = state.get('deployment_mode', 'unroll_scf')
        if mode not in ('unroll_scf', 'implicit'):
            raise ValueError(f'Unknown saved solver mode {mode}')
        self.deployment_mode = mode

class CoupledGeometry(SpectralGeometry):
    """Same Fourier support for the response, observers and moment screening."""
    def __init__(self, model, data, positions):
        super().__init__(model, data, positions)
        r = self.response
        active = self.mask & (self.k2 > 0)
        co, si = self.phase.real*active[..., None], -self.phase.imag*active[..., None]
        basis = self.basis[..., [0, 3, 1, 2]]
        widths = r.source_widths.to(positions)
        gaussian = torch.exp(-.5*self.k2[..., None]*widths.square())
        radial = torch.cat((gaussian[..., :1], gaussian[..., 1:]-gaussian[..., :1]), -1) if len(widths) else gaussian
        receiver = torch.exp(-.5*self.k2[..., None]*r.receiver_widths.to(positions).square())
        proto = self.proto if len(widths) else torch.zeros_like(self.proto)
        self.proto = proto
        ramp = self.ramp(-self.slab_factor)
        zn = (self.pack(positions)*self.normal[:, None]).sum(-1)*self.present
        self.tensors = (co, si, self.wave, basis.real*active[..., None], basis.imag*active[..., None],
                        radial*active[..., None], receiver*active[..., None], torch.view_as_real(proto),
                        torch.view_as_real(ramp), torch.view_as_real(self.applied+self.counter),
                        zn, self.normal, -self.coulomb, self.volume.reciprocal())
        # Match v366's physical-support packing. Padded FFT corners and k=0
        # never enter the dense moment operator or any recurrent field sample.
        # Observers expand back to the common FFT grid without changing modes.
        count = int(active.sum(-1).max())
        if count == 0:
            raise ValueError('The Fourier cutoff contains no nonzero cell modes')
        indices = torch.arange(active.shape[1],device=positions.device).expand_as(active)
        indices = indices.masked_fill(~active,active.shape[1]).sort(-1).values[:,:count]
        self.selected_mask = indices<active.shape[1]
        self.selected_indices = indices.clamp_max(active.shape[1]-1)
        rows = torch.arange(len(self.counts),device=positions.device)[:,None]
        def selected(value):
            mask = self.selected_mask.reshape(*self.selected_mask.shape,*([1]*(value.ndim-2)))
            return value[rows,self.selected_indices]*mask
        self.tensors = tuple(selected(value) if i<10 or i==12 else value
                             for i,value in enumerate(self.tensors))
        base_width = float(widths[0]) if len(widths) else self.width
        self.kernels = coefficient_kernels(self.tensors, self.width, (2*math.pi)**1.5*self.width**3, base_width)

    def expand_spectrum(self, value):
        """Restore only the retained physical modes to the shared FFT grid."""
        output = value.new_zeros((len(self.counts),self.k2.shape[1],value.shape[-1]))
        return output.scatter_add(1,self.selected_indices[...,None].expand_as(value),
                                  value*self.selected_mask[...,None])

def prepare_coupled(model, data, local, constant_charge=True, geometry=None, functional=False):
    """Prepare one live geometry/linear factorization, reused by every update."""
    r = model.field_dependent_charges_map
    g = CoupledGeometry(model, data, local.positions) if geometry is None else geometry
    features = g.pack(local.all_layer_feats)
    scalar = torch.cat([features[..., a:b] for a,b in r.scalar_slices], -1)
    vector = (torch.cat([features[..., a:b].reshape(*features.shape[:2], -1, 3)
                        for a,b in r.vector_slices], -2) if r.vector_slices else
              features.new_zeros((*features.shape[:2], 0, 3)))
    p0 = g.pack(local.field_independent_charge_density)
    if not hasattr(model, 'foundation_element_map'):
        vector = vector[..., [2, 0, 1]]
        p0 = p0[..., [0, 3, 1, 2]]
    chemical, vector = r.geometry_features(scalar, vector)
    mask = g.present
    target = data['total_charge'].reshape(-1).to(p0)
    state = p0.new_zeros((*p0.shape[:2], 4+4*r.source_channels))
    state = torch.cat(((target/mask.sum(-1))[:, None, None]*mask[..., None],
                       p0[..., 1:4]/r.density_width, state[..., 4:]), -1)
    transport = angular_transport(local.positions, data['edge_index'], data.get('shifts'),
        g.batch, g.slot, len(g.counts), g.size, float(model.r_max), float(r.receiver_widths.max()))
    source, reference = r.prepare_reference(state, chemical, vector, g.attrs, p0, mask, target, transport)
    state = torch.cat((state[..., :1], state[..., 1:4]+reference[..., 1:4], source), -1)
    kernel = moment_kernel(g.tensors, g.kernels, mask, r.density_width)
    factors = factor_moments(reference, kernel, mask)
    if not functional:
        # The same constrained matrix acts at EVERY finite update. Materialize
        # its action once, with the checked LU and exact inverse derivative.
        # This aggregates operator gradients before the expensive solve rather
        # than repeating LU solves/checks in every force-loss backward branch.
        # All coordinate, softness and higher derivatives remain attached.
        matrix, root, lu, pivots = factors
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device).expand_as(matrix)
        inverse = solve_factored_system(matrix, lu, pivots, identity)
        factors = (inverse, root, lu, pivots)
    parameters = tuple(r.parameters())
    names = tuple(name for name,_ in r.named_parameters())
    # All geometry, reference and parameter dependencies are explicit for the
    # converged-root adjoint, including the live linear operator behind LU.
    fixed = (chemical, vector, g.attrs, p0, mask, target, transport, reference,
             kernel, *factors, *g.tensors, *g.kernels)
    count = len(fixed)

    def evaluate(z, *args, screened=True, include_energy=False, observe=True):
        chem, vec, attrs, initial, present, charge, neighbors, ref, moment = args[:9]
        factor = args[9:13]
        geometry, kernels = args[13:27], args[27:29]
        spectral = (evaluate_coefficients(z*present[..., None], geometry, kernels, r.density_width)
                    if observe else (total_spectrum(z*present[..., None], geometry, kernels, r.density_width),))
        potential, field = sample_spectrum(spectral[0], geometry)
        inputs = (z, chem, vec, attrs, potential, field, initial, present, charge, neighbors)
        options = {'include_energy': include_energy, 'reference_offset': ref,
                   'observe_chemical_level': observe}
        result = (functional_call(r, dict(zip(names, args[count:])), inputs, options, strict=False)
                  if functional else r(*inputs, **options))
        proposal = screen_moments(z, result[0], ref, moment, present, factor,
                                  materialized=not functional) if screened else result[0]
        return (proposal, *result[1:]), (*spectral, potential, field)

    def update(z, *args):
        return evaluate(z, *args, observe=False)[0][0]

    update.raw = lambda z, *args: evaluate(z, *args, screened=False, observe=False)[0][0]
    update.convergence_residual = lambda z, *args: evaluate(z, *args, screened=False, observe=False)[0][0]-z
    # Exact screened geometry-reference seed from v366. The nonlinear response
    # then updates density AND potential; this is not an inference-time fit.
    spectra = evaluate_coefficients(state, g.tensors, g.kernels, r.density_width)
    values, gradient = sample_spectrum(spectra[0], g.tensors)
    chi, soft = reference[..., -2:].unbind(-1)
    q = charge_closure(torch.zeros_like(p0[..., 0]), chi+values[..., 0], soft, target, mask)[0]
    dipole = state[..., 1:4]-reference[..., -3:-2]*r.density_width*gradient[..., 0, :]
    proposal = torch.cat((q[..., None], dipole, state[..., 4:]), -1)
    state = screen_moments(state, proposal, reference, kernel, mask, factors, materialized=not functional)
    return g, state, (*fixed, *parameters), update, evaluate

def evaluate_coupled(model, data, steps=50, training=False, compute_force=True,
                     constant_charge=True, compute_stress=False, mode=None, mixing=None,
                     tolerance=1.e-7):
    """Return the selected finite trajectory or a verified constitutive root."""
    with torch.set_grad_enabled(training or compute_force or compute_stress or torch.is_grad_enabled()):
        r = model.field_dependent_charges_map
        mode = r.deployment_mode if mode is None else mode
        mixing = float(r.deployment_mixing) if mixing is None else float(mixing)
        if mode not in ('unroll_scf', 'shortcut_scf', 'implicit') or int(steps) < 1 or not 0 < mixing <= 1:
            raise ValueError('Invalid coupled SCF mode, iteration budget or mixing')
        if not constant_charge:
            raise NotImplementedError('The restored v366 response uses fixed total charge; voltage MD evolves that charge. Native and variational updates retain constant-Fermi evaluation.')
        data = dict(data)
        positions = data['positions']
        if compute_force:
            positions.requires_grad_(True)
        strain = None
        if compute_stress:
            strain = positions.new_zeros((len(data['ptr'])-1, 3, 3), requires_grad=True)
            symmetric = .5*(strain+strain.transpose(-1,-2))
            data['positions'] = positions+torch.einsum('ni,nij->nj', positions, symmetric[data['batch']])
            cell = data['cell'].reshape(-1,3,3)
            data['cell'] = cell+cell@symmetric
            edge_batch = data['batch'][data['edge_index'][0]]
            data['shifts'] = data['shifts']+torch.einsum('ni,nij->nj',data['shifts'],symmetric[edge_batch])
        local = model.local_part(data, compute_force=compute_force)
        g, initial, args, update, evaluate = prepare_coupled(model, data, local, functional=mode == 'implicit')
        if mode == 'implicit':
            state, info = implicit_root(update, initial, *args,
                options=RootOptions(max_steps=int(steps), mixing=mixing, tolerance=tolerance,
                                    linear_tolerance=min(1.e-9,tolerance*.01)))
        else:
            state = initial
            limit = _trajectory_scale(initial)
            # Unroll stores the actual trajectory. Shortcut checkpoints short
            # blocks, with the same complete first/second derivatives. Do not
            # force recomputation onto every unroll force-training evaluation.
            checkpointed = training and mode == 'shortcut_scf'
            block_size = max(1, math.isqrt(int(steps))) if checkpointed else int(steps)
            for start in range(0, int(steps), block_size):
                length = min(block_size, int(steps)-start)
                def advance(value, *fixed, count=length, offset=start):
                    for iteration in range(count):
                        proposal = update(value, *fixed)
                        _check_trajectory(proposal, limit, offset+iteration+1, 'finite-step proposal')
                        value = value+mixing*(proposal-value)
                    return value
                state = (checkpoint(advance, state, *args, use_reentrant=False, preserve_rng_state=False)
                         if checkpointed else advance(state, *args))
            info = state.new_tensor([steps, 0., 0.])
        response, spectral = evaluate(state, *args, screened=False, include_energy=True)
        proposal, mu = response[:2]
        total, lr, sr, effective = (g.expand_spectrum(value) for value in spectral[:4])
        potential, field = spectral[-2:]
        residual = ((proposal-state)*g.present[..., None]).abs().amax((1,2))
        if not bool(torch.isfinite(state).all() & torch.isfinite(residual).all()):
            raise RuntimeError('Nonfinite coupled electronic state')
        state = state*g.present[..., None]
        moments = torch.cat((state[..., :1], state[..., 1:4]*r.density_width), -1)
        density_coefficients = g.unpack(moments)[..., [0, 2, 3, 1]]
        carrier, regular = coefficient_contributions(state, g.tensors, g.kernels)
        carrier, regular = g.expand_spectrum(carrier), g.expand_spectrum(regular)
        safe = torch.where(g.coulomb>0, -g.coulomb, 1.)
        coarse = carrier/safe[..., None]
        charge = state[..., 0].sum(-1)
        zero = g.k2 == 0
        coarse = coarse+torch.stack((zero*(charge/g.volume)[:,None], torch.zeros_like(g.k2)), -1)
        dipole = (moments[..., :1]*g.pack(local.positions)+moments[..., 1:4]).sum(1)
        normal_dipole = (dipole*g.normal).sum(-1)
        periodic_energy = .5*g.volume*(coarse.square().sum(-1)*g.coulomb).sum(-1)
        boundary_energy = .5*g.slab_factor*normal_dipole.square()
        counter_energy = -g.volume*(torch.view_as_complex(coarse.contiguous()).conj()*g.counter).real.sum(-1)+g.counter_energy
        electronic = periodic_energy+boundary_energy-(dipole*g.external).sum(-1)+counter_energy
        electronic = electronic-model.fermi_level_offset*charge
        if not getattr(model.coulomb_energy, 'include_self_interaction', True):
            self_terms = model.coulomb_energy.self_interaction_terms(density_coefficients)
            electronic = electronic-.5*g.pack((density_coefficients*self_terms).sum(-1)).sum(-1)
        local_electronic = response[6].sum(-1)
        energy = local.energies.sum(-1)+electronic+local_electronic
        forces = stress = None
        if compute_force or compute_stress:
            inputs = ([positions] if compute_force else [])+([strain] if compute_stress else [])
            deriv = torch.autograd.grad(energy.sum(), inputs, create_graph=training, retain_graph=training, allow_unused=True)
            if compute_force:
                forces = -deriv[0] if deriv[0] is not None else torch.zeros_like(positions)
            if compute_stress:
                stress = deriv[-1]/g.volume[:,None,None] if deriv[-1] is not None else torch.zeros_like(strain)
        mu = mu+model.fermi_level_offset
        vacuum = g.plane(torch.view_as_complex(total.contiguous()))
        smoothing = torch.exp(-.5*(float(r.receiver_widths.max())**2-r.density_width**2)*g.k2)
        output = {'energy':energy, 'forces':forces, 'stress':stress,
            'virials': -stress*g.volume[:,None,None] if stress is not None else None,
            'density_coefficients':density_coefficients, 'dipole':dipole, 'total_charge':charge,
            'fermi_level':mu, 'vacuum_potential':vacuum, 'workfunction':vacuum-mu,
            'fourier_density':coarse*g.ngrid, 'fourier_farfield_density':coarse*smoothing[...,None]*g.ngrid,
            'fourier_effective_density':effective*g.ngrid,
            'fourier_potential':(lr+sr)*g.ngrid, 'fourier_total_potential':total*g.ngrid,
            'fourier_local_potential':regular*g.ngrid, 'fourier_carrier_potential':carrier*g.ngrid,
            'k_vectors_mask':g.mask, 'k_vectors':g.wave, 'planar_mode_mask':g.axial,
            'k_vectors_grid_shape':torch.tensor(g.shape, device=positions.device),
            'scf_residual':residual, 'scf_steps':positions.new_full((len(mu),),float(info[0])),
            'electrostatic_energy':electronic, 'electron_energy':local_electronic,
            'external_field':g.external, 'esps':None, 'esps_dft':None,
            'charges_history':density_coefficients[...,None],
            'potential_coefficients':g.unpack(state[...,4:]),
            'electrostatic_features':g.unpack(potential),
            'chemical_level':g.unpack(response[5]), 'node_field':g.unpack(field)}
        attach_observations(data, g, output)
        if 'fourier_density_dft' in output:
            output['fourier_farfield_density_dft'] = output['fourier_density_dft']*smoothing[...,None]
            output['fourier_farfield_density_dft_mask'] = output['fourier_density_dft_mask']
        return output

@torch.no_grad()
def initialize_coupled(model, loader, device, options=None, loss_config=None):
    """Training-only proto/feature conditioning; never a validation correction."""
    from mace_scf.utils.foundation import calibration_loader
    initialize_response(model, loader, device)
    r = model.field_dependent_charges_map
    ss, vv, count = torch.zeros_like(r.geometry_scalar_unit), torch.zeros_like(r.geometry_vector_unit), 0
    for batch in calibration_loader(loader):
        data = batch.to(device).to_dict()
        features = model.local_part(data, compute_force=False).all_layer_feats
        scalars = torch.cat([features[:,a:b] for a,b in r.scalar_slices], -1)
        vectors = torch.cat([features[:,a:b].reshape(len(features),-1,3) for a,b in r.vector_slices], -2)
        counts = (data['ptr'][1:]-data['ptr'][:-1]).to(features)
        weights = data['weight'].reshape(-1).to(features)
        atom_weight = (weights/counts)[data['batch']]
        ss += (torch.asinh(scalars).square()*atom_weight[:,None]).sum(0)
        vv += (vectors.square().mean(-1)*atom_weight[:,None]).sum(0)
        count += float(weights.sum())
    r.geometry_scalar_unit.copy_((1.+ss/max(count,1.e-30)).sqrt())
    r.geometry_vector_unit.copy_((1.+vv/max(count,1.e-30)).sqrt())
    r.geometry_units_initialized.fill_(True)
    if r.source_channels and options is not None and loss_config is not None:
        initialize_reference(model, calibration_loader(loader), device, options, loss_config)
    # Numerical field coordinates only: the measured spectra/states are unchanged.
    ps = torch.zeros_like(r.scalar_field_scale)
    fs = torch.zeros_like(r.vector_field_scale)
    count = 0
    for batch in calibration_loader(loader):
        data = batch.to(device).to_dict()
        local = model.local_part(data, compute_force=False)
        g, seed, args, _, evaluate = prepare_coupled(model, data, local)
        _, spectral = evaluate(seed, *args)
        values, gradients = spectral[-2:]
        weights = data['weight'].reshape(-1)
        mask = (g.present*weights[:,None]/g.present.sum(-1,keepdim=True))[...,None]
        drive = r.potential_coordinates(values, g.present, args[7][...,-2:], data['total_charge'].reshape(-1))*r.scalar_field_scale
        ps += (drive.square()*mask).sum((0,1))
        fs += (gradients.square().mean(-1)*mask).sum((0,1))
        count += float(weights.sum())
    r.scalar_field_scale.copy_((1.+ps/max(count,1.e-30)).sqrt())
    r.vector_field_scale.copy_((1.+fs/max(count,1.e-30)).sqrt())
    r.field_scales_initialized.fill_(True)
    logging.info('Coupled total-field response: screened moment seed, %d local potential channels, Gaussian receiver widths %s',
                 r.source_channels, r.receiver_widths.tolist())

@torch.no_grad()
def initialize_reference(model, loader, device, options, loss_config):
    """Restore v366's train-only cold species/field fit and objective guard.

    Only existing species levels and scalar reference amplitudes are fitted.
    A cold affine trajectory has a closed form, verified against its raw
    constitutive equation. A proposal is accepted only on the training loss,
    including force labels. No validation correction or extra head is fitted.
    """
    from .potential import _target_fft, target_mode_mask
    from .loss import WeightedLoss, vacuum_reference_weights
    r = model.field_dependent_charges_map
    nlevels = r.species_level.numel()
    original = torch.cat((r.species_level, r.species_source.flatten())).clone()
    size = original.numel()
    zero = torch.zeros_like(original)
    def put(value):
        r.species_level.copy_(value[:nlevels])
        r.species_source.copy_(value[nlevels:].reshape_as(r.species_source))
    def weight(name):
        value = loss_config.get(name, {})
        return float(value.get('weight', 0.)) if isinstance(value,dict) else float(value)
    potential_options = loss_config.get('fourier_potential', {})
    vacuum_weight = float(potential_options.get('vacuum_weight', 0.)) if isinstance(potential_options, dict) else 0.
    scales = original.new_tensor([weight('fourier_potential'), weight('fermi_level'), weight('fourier_potential')*vacuum_weight])
    grams = original.new_zeros((3,size,size)); right = original.new_zeros((3,size)); counts = original.new_zeros(3)
    modes = model.training
    flags = [p.requires_grad for p in model.parameters()]
    try:
        model.eval()
        put(zero)
        for batch in loader:
            batch = batch.to(device); data = batch.to_dict()
            local = model.local_part(data, compute_force=False)
            geom = CoupledGeometry(model, data, local.positions)
            target = _target_fft(data,'fourier_potential','fourier_potential_shape',geom.shape,local.positions.dtype)
            active = geom.mask & (geom.k2>0)
            if target is not None:
                active &= target_mode_mask(data['fourier_potential_shape'], geom.modes)
                target = torch.where(active[...,None],target/geom.ngrid,0.)
                pw = batch.weight*batch.fourier_potential_weight
            else:
                target = local.positions.new_zeros((*geom.k2.shape,2)); pw = batch.weight*0.
            vacuum, vw = vacuum_reference_weights(batch, batch.total_charge)
            labels = (target.flatten(1), batch.fermi_level[:,None], vacuum[:,None])
            availability = (pw, batch.weight*batch.fermi_level_weight, vw)
            def predict(coeff):
                put(coeff)
                g, initial, args, update, observe = prepare_coupled(model,data,local,geometry=geom)
                root = update(initial,*args)
                state = root+(1.-options.scf.mixing_parameter)**options.scf.num_scf_steps*(initial-root)
                if bool((update.convergence_residual(root,*args).abs().amax()>1.e-8)):
                    raise RuntimeError('Cold reference model is not affine; initialization must precede training')
                result, fields = observe(state,*args,screened=False)
                spectrum = torch.where(active[...,None],geom.expand_spectrum(fields[1]+fields[2]),0.)
                return (spectrum.flatten(1),(result[1]+model.fermi_level_offset)[:,None],
                        geom.plane(torch.view_as_complex(geom.expand_spectrum(fields[0]).contiguous()))[:,None])
            baseline = predict(zero)
            columns = [[],[],[]]
            for i in range(size):
                coordinate=zero.clone(); coordinate[i]=1.
                for j,value in enumerate(predict(coordinate)):
                    columns[j].append(value-baseline[j])
            put(zero)
            for j,(label,w) in enumerate(zip(labels,availability)):
                if not bool(scales[j]>0): continue
                design = torch.stack(columns[j],-1)
                residual = torch.where(w[:,None]>0,label-baseline[j],0.)
                grams[j] += torch.einsum('bkn,bkm,b->nm',design,design,w)
                right[j] += torch.einsum('bkn,bk,b->n',design,residual,w)
                counts[j] += w.sum()
        scales = torch.where(counts>0,scales/counts.clamp_min(1.e-30),0.)
        matrix = (grams*scales[:,None,None]).sum(0)
        rhs = (right*scales[:,None]).sum(0)
        units = matrix.diagonal().clamp_min(0.).sqrt().clamp_min(1.e-12)
        normalized = matrix/units[:,None]/units[None,:]
        values, vectors = torch.linalg.eigh(.5*(normalized+normalized.T))
        keep = values>values.max().clamp_min(1.e-30)*1.e-10
        proposal = ((vectors[:,keep]@((vectors[:,keep].T@(rhs/units))/values[keep]))/units
                    if bool(keep.any()) else original)
        objective = WeightedLoss(loss_config)
        model.requires_grad_(False)
        best, best_score, scores = original.clone(), float('inf'), []
        for fraction in (0.,1.,.5,.25):
            candidate = original+fraction*(proposal-original); put(candidate)
            total, count = 0., 0
            for batch in loader:
                batch=batch.to(device)
                output=evaluate_coupled(model,batch.to_dict(),steps=options.scf.num_scf_steps,
                    mode=options.mode,compute_force=weight('forces')>0,mixing=options.scf.mixing_parameter)
                value=float(objective(batch,output)); total+=value*batch.num_graphs;count+=batch.num_graphs
            score=total/max(count,1)
            if not math.isfinite(score): raise FloatingPointError('Nonfinite training objective in cold reference fit')
            scores.append({'fraction':fraction,'loss':score})
            if score<best_score:best,best_score=candidate.clone(),score
        put(best)
        logging.info('v366 training-only cold reference fit: rank=%d/%d; full-objective candidates=%s; no held-out calibration',int(keep.sum()),size,scores)
    except Exception:
        put(original)
        raise
    finally:
        for parameter, flag in zip(model.parameters(),flags): parameter.requires_grad_(flag)
        model.train(modes)
