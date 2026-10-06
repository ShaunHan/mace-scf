"""v366 nonlinear density/potential response on the released MACE backbone.

Charge and dipole screening removes the affine electrostatic stiffness. The
remaining local chemical and potential responses see the complete evolving
field on every step. Density labels observe a coarse moment carrier; potential
labels observe that carrier plus a regular local potential and frozen proto
reference. No reference label enters the electronic update.
"""
import logging
import math

import torch
from e3nn import o3
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint

from .potential import SpectralGeometry, initialize_response, attach_observations
from .coupled_response import (
    AtomicPotentialResponse, angular_transport, charge_closure,
    coefficient_kernels, coefficient_contributions, evaluate_coefficients,
    sample_spectrum,
)
from .coupled_solver import (
    RootOptions, implicit_root, moment_kernel, factor_moments, screen_moments,
    _trajectory_scale, _check_trajectory,
)


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


def prepare_coupled(model, data, local, constant_charge=True, geometry=None):
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
    parameters = tuple(r.parameters())
    names = tuple(name for name,_ in r.named_parameters())
    # All geometry, reference and parameter dependencies are explicit for the
    # converged-root adjoint, including the live linear operator behind LU.
    fixed = (chemical, vector, g.attrs, p0, mask, target, transport, reference,
             kernel, *factors, *g.tensors, *g.kernels)
    count = len(fixed)

    def evaluate(z, *args, screened=True, include_energy=False):
        chem, vec, attrs, initial, present, charge, neighbors, ref, moment = args[:9]
        factor = args[9:13]
        geometry, kernels = args[13:27], args[27:29]
        spectral = evaluate_coefficients(z*present[..., None], geometry, kernels, r.density_width)
        potential, field = sample_spectrum(spectral[0], geometry)
        result = functional_call(r, dict(zip(names, args[count:])),
            (z, chem, vec, attrs, potential, field, initial, present, charge, neighbors),
            {'include_energy': include_energy, 'reference_offset': ref}, strict=False)
        proposal = screen_moments(z, result[0], ref, moment, present, factor) if screened else result[0]
        return (proposal, *result[1:]), (*spectral, potential, field)

    def update(z, *args):
        return evaluate(z, *args)[0][0]

    update.raw = lambda z, *args: evaluate(z, *args, screened=False)[0][0]
    update.convergence_residual = lambda z, *args: update.raw(z, *args)-z
    # Exact screened geometry-reference seed from v366. The nonlinear response
    # then updates density AND potential; this is not an inference-time fit.
    spectra = evaluate_coefficients(state, g.tensors, g.kernels, r.density_width)
    values, gradient = sample_spectrum(spectra[0], g.tensors)
    chi, soft = reference[..., -2:].unbind(-1)
    q = charge_closure(torch.zeros_like(p0[..., 0]), chi+values[..., 0], soft, target, mask)[0]
    dipole = state[..., 1:4]-reference[..., -3:-2]*r.density_width*gradient[..., 0, :]
    proposal = torch.cat((q[..., None], dipole, state[..., 4:]), -1)
    state = screen_moments(state, proposal, reference, kernel, mask, factors)
    return g, state, (*fixed, *parameters), update, evaluate


@torch.enable_grad()
def evaluate_coupled(model, data, steps=50, training=False, compute_force=True,
                     constant_charge=True, compute_stress=False, mode=None, mixing=None,
                     tolerance=1.e-7):
    """Return the selected finite trajectory or a verified constitutive root."""
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
    g, initial, args, update, evaluate = prepare_coupled(model, data, local)
    if mode == 'implicit':
        state, info = implicit_root(update, initial, *args,
            options=RootOptions(max_steps=int(steps), mixing=mixing, tolerance=tolerance,
                                linear_tolerance=min(1.e-9,tolerance*.01)))
    else:
        state = initial
        limit = _trajectory_scale(initial)
        for iteration in range(int(steps)):
            proposal = (checkpoint(update, state, *args, use_reentrant=False, preserve_rng_state=False)
                        if training and (mode == 'shortcut_scf' or steps >= 12) else update(state, *args))
            _check_trajectory(proposal, limit, iteration+1, 'finite-step proposal')
            state = state+mixing*(proposal-state)
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
        'k_vectors_mask':g.mask, 'k_vectors':g.wave,
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
    logging.info('Restored v366 nonlinear total-field response: screened moment seed, %d local potential channels, receiver widths %s', r.source_channels, r.receiver_widths.tolist())


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
