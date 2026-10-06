"""Nonlinear total-field response and observations ported from v366.

One field drives charge, dipole and regular local-potential updates. The
conditional charge multiplier is an EF prediction, not a claim of global
energy stationarity. Forces differentiate the returned finite-step energy.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3

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
    value = torch.bmm(cosine.transpose(1, 2), re) - torch.bmm(sine.transpose(1, 2), im)
    re_k = (re[..., None] * wave[:, :, None]).flatten(-2)
    im_k = (im[..., None] * wave[:, :, None]).flatten(-2)
    gradient = torch.bmm(sine.transpose(1, 2), re_k) + torch.bmm(cosine.transpose(1, 2), im_k)
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
        self.preconditioner_damping = 0.1
        # Fresh fits start from the self-consistent geometry-only charge law.
        # Full-module checkpoints without this marker keep their old trajectory.
        self.register_buffer('screened_reference_initialization', torch.tensor(True))
        # v363 hotfix: an unscreened E * gate(log|E|) vector feedback path
        # can amplify a modest optimizer update on every SCF iteration.
        # Compress field descriptors consistently with the scalar coordinates.
        # Full-module checkpoints lacking this marker retain their old map.
        self.register_buffer('regular_field_coordinates', torch.tensor(True))
        # A local quadratic hardness is a material/geometry coefficient. Keep
        # it fixed during the electronic iteration; chi, dipoles and the local
        # potential still respond nonlinearly to the COMPLETE current field.
        # This removes a second feedback loop through the inverse charge solve.
        # Buffers distinguish fresh models from full-module legacy checkpoints.
        self.register_buffer('geometry_softness', torch.tensor(True))
        self.register_buffer('regular_geometry_coordinates', torch.tensor(True))
        self.register_buffer('coarse_density_energy', torch.tensor(True))
        # The scalar geometry readout is an electronegativity (eV), not a
        # second free charge added around the screened charge solve. q0 and
        # chi are redundant: q0+s*chi = s*(chi+q0/s). Predict the latter
        # driving level directly, so its Jacobian includes the susceptibility.
        # Old full-module checkpoints without this flag retain their units.
        self.register_buffer('geometry_charge_is_level', torch.tensor(True))
        self.register_buffer('linear_induced_dipoles', torch.tensor(True))
        self.source_channels=int(source_channels)
        self.field_channels=int(field_channels)
        self.density_width=float(density_width)
        self.vector_width=min(16,max(4,int(vector_channels)))
        self.vector_basis_size=2*self.vector_width
        # Fresh v362 fits use a direct construction. Saved modules retain their
        # tensors; reproducing obsolete initialization draws is not model physics.
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
        if bool(getattr(self, 'regular_geometry_coordinates', False)):
            # The old unbounded geometry-vector route bypassed the field
            # coordinate regularization and multiplied every induced dipole.
            # Apply the same smooth equivariant coordinate map, without
            # clipping physical multipoles or altering the MACE energy path.
            scaled_vector = vector_asinh(scaled_vector)
        vector=(vectors.new_zeros((*vectors.shape[:-2],self.vector_width,3)) if self.chemical_vector is None
                else self.chemical_vector(scaled_vector.transpose(-1,-2)).transpose(-1,-2))
        return chemical,vector

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # A pre-v365 state dictionary contains charge-unit geometry weights.
        # Preserve their interpretation when loading into a new constructor.
        key = prefix + 'geometry_charge_is_level'
        if key not in state_dict and hasattr(self, 'geometry_charge_is_level'):
            state_dict[key] = self.geometry_charge_is_level.new_tensor(False)
        key=prefix+'linear_induced_dipoles'
        if key not in state_dict and hasattr(self,'linear_induced_dipoles'):
            state_dict[key]=self.linear_induced_dipoles.new_tensor(False)
            for name,value in self.dipole_hardness.state_dict().items():
                state_dict[prefix+'dipole_hardness.'+name]=value
            self.dipole_hardness.requires_grad_(False)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def state_embedding(self,state,chemical,vector,v,e,neighborhood=None):
        # State coefficients do NOT form an extra neural input. Their physical
        # information enters solely through the common projected v and -grad(v).
        if neighborhood is None or neighborhood.ndim!=4 or neighborhood.shape[-1]!=4:
            raise ValueError('The common response requires its periodic local transport operator')
        # The full receiver coordinates carry the electrochemical drive;
        # local radial contrasts and angular/neighbor variation remain present.
        ds=transport_difference(neighborhood[...,0],v)
        dv=transport_difference(neighborhood[...,0],e)
        div,grad=angular_differences(neighborhood,v,e)
        if bool(getattr(self, 'regular_field_coordinates', False)):
            # Include the neighbor potential-gradient channel: compressing E
            # alone leaves another unbounded vector route through grad(v).
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

    def _constitutive_outputs(self,invariants,vector,embedded,return_intermediates=False,
                              return_hidden=False):
        u1 = self.shared[0](invariants)
        u2 = self.shared[2](self.shared[1](u1))
        hidden = self.shared[3](u2)
        scalars=self.scalar_out(hidden)
        basis=torch.cat((vector,embedded),-2)
        gate=self.vector_out(hidden).reshape(*hidden.shape[:-1],1+self.source_channels,self.vector_basis_size)
        polar=torch.einsum('bnov,bnvc->bnoc',gate,basis)/math.sqrt(self.vector_basis_size)
        result = (scalars, polar)
        if return_intermediates:
            result += ((u1, u2, gate),)
        if return_hidden:
            result += (hidden,)
        return result

    def chemical_levels(self, raw_level, hidden, attrs, softness, mask):
        """Split local contrasts and the uniform level in the same KKT law.

        chi_i = raw_i - <raw>_s + <chi_common>_s,
        mu = <chi_common>_s + <v>_s - Q/sum(s).
        A common-readout change shifts mu without changing any charge. Field
        and geometry derivatives through the shared features remain attached.
        Old full-module checkpoints, which lack common_level, keep their law.
        This conditional multiplier is not asserted to equal dE/dN_e.
        """
        common_readout = getattr(self, 'common_level', None)
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
        if not hasattr(self, 'shared_reference_readout'):
            raise ValueError('The saved response uses a different local reference readout. '
                             'Use its original source or start a fresh fit; weights are not reinterpreted.')
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
        if bool(getattr(self, 'geometry_charge_is_level', False)):
            chi0 = chi0 + p0[..., 0]
        softness0 = self.softness_scale*(F.softplus(initial_scalar[...,1]+math.log(math.expm1(1.)))+1.e-6)
        electronic_reference = torch.stack((chi0,softness0),-1)
        zeros=state.new_zeros((*state.shape[:2],self.field_channels))
        inv,embedded=self._invariants(state,chemical,vector,zeros,
            zeros[...,None].expand(-1,-1,-1,3),mask,neighborhood,electronic_reference,target)
        scalar,polar=self._constitutive_outputs(inv,vector,embedded)
        local=self.local_potential_coordinates(chemical,vector,attrs,mask,hidden=hidden)
        raw=torch.cat((scalar[...,2:],polar[...,1:,:].flatten(-2)),-1)
        offset=torch.cat((torch.zeros_like(state[...,:1]),-polar[...,0,:],local-raw),-1)*mask[...,None]
        if bool(getattr(self,'linear_induced_dipoles',False)):
            # The first vector readout defines the geometry reference dipole;
            # induced polarization follows a positive quadratic local cost.
            offset=torch.cat((offset[...,:1],polar[...,0,:],offset[...,4:]),-1)*mask[...,None]
            dipole_soft=self.softness_scale*(F.softplus(self.dipole_hardness(hidden)+math.log(math.expm1(1.)))+1.e-6)
            return local,torch.cat((offset,dipole_soft,electronic_reference),-1)
        return local,torch.cat((offset,electronic_reference),-1)

    def forward(self,state,chemical,vector,attrs,potential,field,p0,mask,target,
                neighborhood=None,*,include_energy=True,reference_offset=None):
        linear_dipoles=bool(getattr(self,'linear_induced_dipoles',False))
        extra=3 if linear_dipoles else 2
        if reference_offset is None or reference_offset.shape!=(*state.shape[:-1],state.shape[-1]+extra):
            raise ValueError('Missing/mismatched differentiable geometry/electrochemical reference')
        offset,electronic_reference=reference_offset[...,:-extra],reference_offset[...,-2:]
        inv,embedded=self._invariants(state,chemical,vector,potential,field,mask,neighborhood,
                                     electronic_reference,target)
        computed = self._constitutive_outputs(inv, vector, embedded, return_hidden=True)
        scalars, polar = computed[:2]
        chemical_level=attrs@self.species_level+scalars[...,0]
        if bool(getattr(self, 'geometry_charge_is_level', False)):
            chemical_level = chemical_level + p0[..., 0]
        softness=(electronic_reference[...,1] if bool(getattr(self, 'geometry_softness', False)) else
                  self.softness_scale*(F.softplus(scalars[...,1]+math.log(math.expm1(1.)))+1.e-6))
        chemical_level = self.chemical_levels(chemical_level, computed[-1], attrs, softness, mask)
        levels=chemical_level+potential[...,0]
        reference_charge = (torch.zeros_like(p0[..., 0]) if
            bool(getattr(self, 'geometry_charge_is_level', False)) else p0[..., 0])
        q,mu,f,response,neutral=charge_closure(reference_charge,levels,softness,target,mask)
        induced=(-reference_offset[...,-3:-2]*self.density_width*field[...,0,:]
                 if linear_dipoles else polar[...,0,:])
        proposal=torch.cat((q[...,None],p0[...,1:4]/self.density_width+induced,
                            scalars[...,2:],polar[...,1:,:].flatten(-2)),-1)*mask[...,None]
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

