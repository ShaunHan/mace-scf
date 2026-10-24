"""A nonlinear charge expansion with an algebraically shared reservoir energy.

The network predicts coefficients, never energies at quadrature charges.  A
cubic source expansion in the two independent charges gives a degree-six
electrostatic energy.  The vacuum work is an exact coefficient contraction of
that SAME energy and ESP, with its neutral value fixed to the DFT energy.
There is no independently fitted WF or reservoir-work head.

This finite charge basis is a model approximation, not a Taylor expansion of
an old checkpoint. Train it on DFT energies, forces, EF and field observations.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .coupled import CoupledResponse, CoupledGeometry, prepare_coupled, unroll_steps
from .potential import attach_observations


# Total-degree cubic basis. Q and Q_cc are independent for DFT derivatives.
POWERS = tuple((i, n-i) for n in range(4) for i in range(n, -1, -1))


def complete_counter_profile(g, data):
    """Full Gaussian-sheet self energy and dipole, independent of ML cutoff.

    The convergent Fourier sum also covers wide sheets and periodic wraparound;
    no isolated-sheet L/12 expansion or arbitrary clearance is used.
    """
    from .potential import FIELD_CONSTANT
    axis = data.get('counter_charge_axis', torch.full_like(g.axes, -1)).reshape(-1)
    sigma = data.get('counter_charge_width', torch.zeros_like(g.volume)).reshape(-1)
    sheet = g.slab & (axis == g.axes) & (sigma > 0)
    g.counter_vacuum_remainder = torch.zeros_like(g.volume)
    if bool(sheet.any()):
        width = torch.where(sheet, sigma, g.length)
        maximum = int(torch.ceil(8*g.length/(2*math.pi*width)).max().detach().cpu())
        n = torch.arange(1, maximum+1, dtype=g.volume.dtype, device=g.volume.device)
        wave = 2*math.pi*n[None]/g.length[:, None]
        gaussian = torch.exp(-.5*(width[:, None]*wave).square())
        fraction = data['counter_charge_fraction'].reshape(-1)
        moment = (-g.length[:, None]/(math.pi*n[None])*gaussian*
                  torch.sin(2*math.pi*n[None]*(fraction-g.branch)[:, None])).sum(-1)
        exact = FIELD_CONSTANT/g.volume*(gaussian.square()/wave.square()).sum(-1)
        retained = .5*g.volume*(g.coulomb*g.counter_density_unit.abs().square()).sum(-1)
        potential = -2*FIELD_CONSTANT/g.volume*(gaussian*torch.cos(
            2*math.pi*n[None]*(g.zfrac-fraction)[:, None])/wave.square()).sum(-1)
        retained_potential = g.plane(-g.coulomb*g.counter_density_unit)
        g.counter_vacuum_remainder = torch.where(sheet, potential-retained_potential, 0.)
        g.counter_moment_unit = torch.where(sheet, moment, g.counter_moment_unit)
        g.counter_self_remainder = torch.where(sheet, exact-retained, 0.)
    g.counter_unit = -g.coulomb*g.counter_density_unit+g.ramp(-g.slab_factor*g.counter_moment_unit)
    proto_reference = (-(g.attrs.sum(1)@g.response.proto_coefficients[:, 0])/g.volume*g.slab
                       if g.neutral_reference and bool(g.response.proto_fitted) else 0.)
    g.counter_gauge = g.vacuum_reference-proto_reference
    if g.molecular is not None:
        # Use the moments of the SAME desmeared effective Poisson source.
        r = g.response
        mapping = F.pad(g.molecular_source, (0, 0, 0, 4*r.source_channels))
        decoder = getattr(r, 'raw_decoder', None)
        variance = decoder.variances[1] if decoder is not None and bool(decoder.enabled[1]) else 0.
        mapping[..., 0, 4] = mapping[..., 0, 4]-3*variance*g.present
        for i in range(r.source_channels):
            if r.source_basis != 'difference' or i == 0:
                mapping[..., 4+i, 4] = 6*(2*math.pi)**1.5*r.density_width**3/FIELD_CONSTANT*g.present
        g.energy_molecular_source = mapping


def powers(q, c):
    return torch.stack([q**i*c**j for i, j in POWERS], -1)


def path_coefficients(coeff, derivative=False):
    """Restrict to c=-q, optionally AFTER a partial derivative in q."""
    result = []
    for n in range(3 if derivative else 4):
        indices = [(p, i*(-1)**j if derivative else (-1)**j)
                   for p, (i, j) in enumerate(POWERS)
                   if i+j-int(derivative) == n and (not derivative or i)]
        result.append(sum(coeff[..., p]*sign for p, sign in indices))
    return torch.stack(result, -1)


def product(a, b):
    """Polynomial multiplication on the last axis, retaining autograd."""
    return torch.stack([sum(a[..., i]*b[..., n-i]
                           for i in range(max(0, n-b.shape[-1]+1), min(n+1, a.shape[-1])))
                        for n in range(a.shape[-1]+b.shape[-1]-1)], -1)


def polynomial(coeff, x):
    result = coeff[..., -1]
    for p in range(coeff.shape[-1]-2, -1, -1):
        result = result*x+coeff[..., p]
    return result


class NonlinearEnergyResponse(CoupledResponse):
    """One neutral field trajectory supplies nonlinear charge coefficients.

    The trajectory is a geometry encoder, not an assumed variational minimum.
    All its coordinate derivatives are retained. Charge dependence is explicit
    in the polynomial coefficients, so neither stationarity nor numerical
    integration over charge is needed for consistent energies and forces.
    """
    shared_nonlinear_energy = True

    def get_extra_state(self):
        state = super().get_extra_state()
        state.pop('vacuum_reservoir_trained', None)
        state['energy_functional'] = 'nonlinear_charge_coefficients'
        return state

    def set_extra_state(self, state):
        if state.get('energy_functional') != 'nonlinear_charge_coefficients':
            raise ValueError('This charge functional requires retraining; an old response checkpoint is not interchangeable')
        super().set_extra_state(state)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        width = self.scalar_out.in_features
        # Counter-field descriptors specify the physical sheet, even at Q_cc=0.
        self.charge_features = nn.Sequential(nn.Linear(width+2, width), nn.SiLU())
        channels = self.source_channels+1
        self.charge_scalars = nn.Linear(width, 9*channels)
        self.charge_vectors = nn.Linear(width, 9*channels*(2*self.vector_width+1))
        self.charge_energy = nn.Linear(width, 9, bias=False)
        for layer in (self.charge_scalars, self.charge_vectors, self.charge_energy):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def coefficients(self, g, state, fixed, response, spectral):
        chemical, vector, _, p0, mask, _, neighbors, reference = fixed[:8]
        potential, field = spectral[-2:]
        invariants, embedded = self._invariants(state, chemical, vector, potential, field,
            mask, neighbors, reference[..., -2:], mask.new_zeros(mask.shape[0]))
        hidden = self.shared(invariants)
        from .coupled import sample_spectrum
        cc_value, cc_field = sample_spectrum(torch.view_as_real(g.counter_unit),
            (g.phase.real, -g.phase.imag, g.wave, None, None, None,
             torch.exp(-.5*g.k2[..., None]*self.receiver_widths.square())))
        # Scale by atom count for an intensive descriptor on repeated cells.
        count = mask.sum(-1)
        cc_value = cc_value[..., 0]*count[:, None]
        cc_field = cc_field[..., 0, :]*count[:, None, None]
        descriptor = torch.cat((torch.asinh(cc_value[..., None]),
            torch.asinh(cc_field.square().sum(-1, keepdim=True))), -1)
        # At Q_cc=0 neither the state nor partial_Q E can depend on the
        # position/width of an absent sheet. Only coefficients with j>0 see it.
        hidden = self.charge_features(torch.cat((hidden[..., None, :].expand(-1, -1, 2, -1),
            torch.stack((torch.zeros_like(descriptor), descriptor), -2)), -1))
        kind = torch.tensor([int(j > 0) for i, j in POWERS[1:]], device=state.device)
        index = torch.arange(9, device=state.device)
        exists = g.counter_density_unit.abs().sum(-1) > 0
        active = torch.where(kind[None].bool(), exists[:, None], True)
        channels = self.source_channels+1
        scalar = self.charge_scalars(hidden).reshape(*mask.shape, 2, 9, channels)
        scalar = scalar[:, :, kind, index]*active[:, None, :, None]
        basis = torch.cat((vector, embedded, cc_field[..., None, :]), -2)
        gates = self.charge_vectors(hidden).reshape(*mask.shape, 2, 9, channels, basis.shape[-2])
        gates = gates[:, :, kind, index]*active[:, None, :, None, None]
        basis_mask = F.pad(kind.to(state)[:, None], (basis.shape[-2]-1, 0), value=1.)
        gates = gates*basis_mask[None, None, :, None]
        polar = torch.einsum('bnpcv,bnvr->bnpcr', gates, basis)/math.sqrt(basis.shape[-2])
        changes = torch.cat((scalar[..., :1], polar[..., 0, :], scalar[..., 1:],
                             polar[..., 1:, :].flatten(-2)), -1)*mask[..., None, None]
        # Each higher coefficient carries zero total charge; the Q coefficient
        # alone carries scale electrons. This also holds for padded batches.
        monopoles = changes[..., 0]
        monopoles = monopoles-(monopoles.sum(1)/count[:, None])[:, None]*mask[..., None]
        scale = count*self.softness_scale
        linear = response[2]*scale[:, None]
        monopoles = monopoles+F.pad(linear[..., None], (0, 8))
        changes = torch.cat((monopoles[..., None], changes[..., 1:]), -1)
        sources = torch.cat((state[..., None, :], changes), -2)
        # Chemical charging energy and its EF derivative share the same head.
        chem = (self.charge_energy(hidden)[:, :, kind, index]*mask[..., None]).sum(1)*active
        common = (response[2]*response[5]).sum(-1)+g.vacuum_reference
        chem = chem+F.pad((-scale*common)[:, None], (0, 8))
        chem = chem+F.pad((.5*scale.square()/response[3])[:, None], (2, 6))
        chem = torch.cat((response[6].sum(-1)[:, None], chem), -1)
        return sources, chem, scale


def source_spectra(g, states):
    """Render all source coefficients together; no extra model evaluations."""
    r = g.response
    decoder = getattr(r, 'raw_decoder', None)
    variances = [0., 0.]
    if decoder is not None:
        if not hasattr(decoder, 'variances'):
            raise ValueError('Shared energy requires the fixed Gaussian observation map')
        for j in range(2):
            if bool(decoder.enabled[j]):
                if not bool(decoder.fitted[j]):
                    raise ValueError('Initialize Gaussian observation widths before evaluation')
                variances[j] = decoder.variances[j]
    coefficients = torch.einsum('bkn,bnpd->bkpd', g.phase, states.to(g.phase.dtype))
    moment = coefficients[..., 0]-1j*r.density_width*(
        coefficients[..., 1:4]*g.wave[..., None, :]).sum(-1)
    active = g.mask*(g.k2 > 0)
    rho = moment*(g.mask*torch.exp(-.5*(r.density_width**2-variances[0])*g.k2))[..., None]/g.volume[:, None, None]
    carrier = -g.coulomb[..., None]*moment*torch.exp(
        -.5*(r.density_width**2-variances[1])*g.k2)[..., None]/g.volume[:, None, None]
    local = torch.zeros_like(carrier)
    widths = r.source_widths
    n = len(widths)
    if n:
        radial = torch.exp(-.5*(widths.square()-variances[1])*g.k2[..., None])
        if r.source_basis == 'difference':
            radial = torch.cat((radial[..., :1], radial[..., 1:]-radial[..., :1]), -1)
        amplitudes = coefficients[..., 4:4+n]+1j*widths[0]*(
            coefficients[..., 4+n:].reshape(*coefficients.shape[:3], n, 3)*g.wave[..., None, None, :]).sum(-1)
        local = (amplitudes*radial[..., None, :]).sum(-1)*(2*math.pi)**1.5*r.density_width**3/g.volume[:, None, None]
    carrier, local = carrier*active[..., None], local*active[..., None]
    effective = (carrier+local)/torch.where(g.coulomb > 0, -g.coulomb, 1.)[..., None]
    # Poisson leaves k=0 free; the total-charge constraint fixes the source.
    charge = states[..., 0].sum(1)
    effective = effective+(g.k2 == 0)[..., None]*charge[:, None]/g.volume[:, None, None]
    moments = torch.cat((states[..., :1], states[..., 1:4]*r.density_width), -1)
    dipole = (moments[..., :1]*g.pack(g.positions)[:, :, None]+moments[..., 1:4]).sum(1)
    return rho, carrier, local, effective, dipole


def contract(coeff, basis):
    return (coeff*basis.reshape(basis.shape[0], *([1]*(coeff.ndim-2)), -1)).sum(-1)


def evaluate_nonlinear(model, data, steps=50, training=False, compute_force=True,
                       constant_charge=True, compute_stress=False, mode=None,
                       mixing=None, tolerance=1.e-7, reference_conditioning=False,
                       reservoir=False):
    """DFT observables, or vacuum-referenced mechanics for SHE-controlled MD."""
    if not constant_charge or reference_conditioning not in (False, 'none'):
        raise ValueError('The shared energy takes fixed Q and Q_cc; MD evolves Q separately')
    r = model.field_dependent_charges_map
    if not getattr(model.coulomb_energy, 'include_self_interaction', True):
        raise ValueError('Shared energy includes the full electrostatic source self energy')
    mode = mode or r.deployment_mode
    mixing = float(r.deployment_mixing) if mixing is None else float(mixing)
    if mode not in ('unroll_scf', 'shortcut_scf') or int(steps) < 1 or not 0 < mixing <= 1:
        raise ValueError('Shared energy uses a finite differentiable field trajectory')
    outer_grad = torch.is_grad_enabled()
    with torch.enable_grad():
        data = dict(data)
        q = data['total_charge'].clone().requires_grad_(True)
        c = data.get('counter_charge', torch.zeros_like(q)).clone().requires_grad_(True)
        data['total_charge'], data['counter_charge'] = q, c
        if bool(data.get('vacuum_reservoir', torch.zeros_like(q, dtype=torch.bool)).any()):
            raise ValueError('Remove obsolete vacuum_reservoir tags; supply the actual counter-charge profile')
        positions = data['positions']
        if compute_force:
            positions.requires_grad_(True)
        strain = None
        if compute_stress:
            from .molecular import enabled as molecular_enabled
            if bool(molecular_enabled(data, q.reshape(-1)).any()):
                raise NotImplementedError('The molecular image correction supports fixed-cell energies and forces, not general stress')
            strain = positions.new_zeros((len(q), 3, 3), requires_grad=True)
            symmetric = .5*(strain+strain.transpose(-1, -2))
            data['positions'] = positions+torch.einsum('ni,nij->nj', positions, symmetric[data['batch']])
            data['cell'] = data['cell'].reshape(-1, 3, 3)@(torch.eye(3, device=q.device)+symmetric)
            data['shifts'] = data['shifts']+torch.einsum('ni,nij->nj', data['shifts'], symmetric[data['batch'][data['edge_index'][0]]])
        # Neutral electronic features are shared by the WHOLE charge function.
        neutral = dict(data, total_charge=torch.zeros_like(q), counter_charge=torch.zeros_like(c))
        local = model.local_part(neutral, compute_force=compute_force)
        g = CoupledGeometry(model, neutral, local.positions)
        g.positions = local.positions
        if g.molecular is not None and bool((c.reshape(-1)[g.molecular.active] != 0).any()):
            raise ValueError('Isolated-molecule correction requires zero counter charge')
        profile = g.counter_density_unit.abs().sum(-1) > 0
        if bool(((c.reshape(-1) != 0) & ~profile).any()):
            raise ValueError('Nonzero counter charge needs its physical spatial profile')
        g, initial, args, update, evaluate = prepare_coupled(model, neutral, local, geometry=g)
        state = unroll_steps(update, initial, args, int(steps), mixing, checkpointed=training and mode == 'shortcut_scf')
        response, spectral = evaluate(state, *args, screened=False, include_energy=True)
        sources, chem, scale = r.coefficients(g, state, args, response, spectral)
        rho, carrier, regular, effective, dipoles = source_spectra(g, sources)
        normal = (dipoles*g.normal[:, None]).sum(-1)
        basis = powers(q.reshape(-1)/scale, c.reshape(-1)/scale)
        actual = contract(sources.transpose(-1, -2), basis)
        source = contract(effective, basis)
        dipole = contract(dipoles.transpose(-1, -2), basis)
        d = contract(normal, basis)
        cc = c.reshape(-1)
        total_source = source+cc[:, None]*g.counter_density_unit
        total_d = d+cc*g.counter_moment_unit
        # All resolved ESP sources, including the local completion, enter the
        # SAME electrostatic quadratic form. The remaining chemistry is learned.
        es = .5*g.volume*(g.coulomb*total_source.abs().square()).sum(-1)
        es = es+.5*g.slab_factor*total_d.square()-(dipole*g.external).sum(-1)
        es = es-cc*g.volume*(g.counter_density_unit.conj()*g.applied).real.sum(-1)
        es = es-g.volume*(total_source.conj()*g.observer_proto).real.sum(-1)
        es = es-g.counter_gauge*cc+cc.square()*g.counter_self_remainder
        molecular_moments = None
        if g.molecular is not None:
            from .molecular import moment_sum
            molecular_moments = moment_sum(actual, g.energy_molecular_source)
            es = es+g.molecular.energy(molecular_moments)
        energy = local.energies.sum(-1)+contract(chem, basis)+es
        eq, ec = torch.autograd.grad(energy.sum(), (q, c), create_graph=True, retain_graph=True)
        eq, ec = eq.reshape(-1), ec.reshape(-1)
        field = (g.external*g.normal).sum(-1)-g.slab_factor*total_d
        boundary = g.ramp(field, width=0.)
        raw_carrier, raw_local = contract(carrier, basis), contract(regular, basis)
        raw = raw_carrier+raw_local-g.coulomb*cc[:, None]*g.counter_density_unit+boundary
        if molecular_moments is not None:
            raw = raw+g.molecular.spectrum(molecular_moments)
        vacuum = (g.plane(raw)+g.ramp_plane(field)-g.plane(boundary)+g.vacuum_reference
                  +cc*g.counter_vacuum_remainder)
        wf = vacuum+eq
        mechanical = energy
        if reservoir:
            if not torch.allclose(cc, -q.reshape(-1), atol=1.e-8, rtol=0):
                raise ValueError('SHE charge transfer requires Q_cc=-Q')
            if not bool((g.slab & profile).all()):
                raise ValueError('SHE control needs slab geometry and a counter-charge profile, including at Q=0')
            if any(key not in data or not bool(torch.isfinite(data[key]).all())
                   for key in ('vacuum_fraction', 'dipole_correction_fraction')):
                raise ValueError('SHE control requires explicit vacuum and dipole-cut planes in the same DFT reference')
            # Exact coefficients of partial_Q E(R,Q,Q_cc)|Q_cc=-Q.
            path = path_coefficients(effective)
            partial = path_coefficients(effective, True)/scale[:, None, None]
            cc_path = F.pad((-scale[:, None]*g.counter_density_unit)[..., None], (1, 2))
            total_path = path+cc_path
            dp = path_coefficients(normal)-F.pad((scale*g.counter_moment_unit)[:, None], (1, 2))
            dq = path_coefficients(normal, True)/scale[:, None]
            e_coeff = g.volume[:, None]*(g.coulomb[..., None]*product(total_path.conj(), partial).real).sum(1)
            e_coeff = e_coeff+g.slab_factor[:, None]*product(dp, dq)
            e_coeff = e_coeff-F.pad(g.volume[:, None]*(partial.conj()*g.observer_proto[..., None]).real.sum(1), (0, 3))
            dipole_q = path_coefficients(dipoles.transpose(-1, -2), True)/scale[:, None, None]
            e_coeff = e_coeff-F.pad((dipole_q*g.external[..., None]).sum(1), (0, 3))
            e_coeff = e_coeff+F.pad(path_coefficients(chem, True)/scale[:, None], (0, 3))
            field_coeff = -g.slab_factor[:, None]*dp+F.pad((g.external*g.normal).sum(-1)[:, None], (0, 3))
            vacuum_coeff = []
            for n in range(4):
                # ESP is linear in sources. Only inexpensive observation
                # contractions are repeated; there is one electronic trajectory.
                spectrum = -g.coulomb*total_path[..., n]
                vacuum_coeff.append(g.plane(spectrum)+g.ramp_plane(field_coeff[:, n]))
            vacuum_coeff = torch.stack(vacuum_coeff, -1)+F.pad(g.vacuum_reference[:, None], (0, 3))
            vacuum_coeff = vacuum_coeff-F.pad((scale*g.counter_vacuum_remainder)[:, None], (1, 2))
            wf_coeff = e_coeff+F.pad(vacuum_coeff, (0, 2))
            # This is coefficient algebra, not charge-space quadrature. The
            # primitive has zero neutral work and no independent parameters.
            denom = torch.arange(1, 7, device=q.device, dtype=q.dtype)
            neutral_energy = local.energies.sum(-1)+chem[:, 0]
            neutral_energy = neutral_energy+.5*g.volume*(g.coulomb*effective[..., 0].abs().square()).sum(-1)
            neutral_energy = neutral_energy+.5*g.slab_factor*normal[:, 0].square()-(dipoles[:, 0]*g.external).sum(-1)
            neutral_energy = neutral_energy-g.volume*(effective[..., 0].conj()*g.observer_proto).real.sum(-1)
            mechanical = neutral_energy+q.reshape(-1)*polynomial(wf_coeff/denom, q.reshape(-1)/scale)
        inputs = ([positions] if compute_force else [])+([strain] if compute_stress else [])
        derivatives = torch.autograd.grad(mechanical.sum(), inputs, create_graph=training,
            retain_graph=training or outer_grad, allow_unused=True) if inputs else []
        forces = (-derivatives[0] if derivatives[0] is not None else torch.zeros_like(positions)) if compute_force else None
        stress = derivatives[-1]/g.volume[:, None, None] if compute_stress and derivatives[-1] is not None else None
        raw_density = contract(rho, basis)
        decoder = getattr(r, 'raw_decoder', None)
        density_variance = decoder.variances[0] if decoder is not None and bool(decoder.enabled[0]) else 0.
        density_smooth = torch.exp(-.5*density_variance*g.k2)
        potential_smooth = torch.exp(-.5*r.potential_amplitudes_smearing_width**2*g.k2)
        def fourier(value):
            return torch.view_as_real(value.contiguous())*g.ngrid
        moments = torch.cat((actual[..., :1], actual[..., 1:4]*r.density_width), -1)
        result = dict(energy=mechanical, dft_energy=energy, forces=forces, stress=stress,
            virials=-stress*g.volume[:, None, None] if stress is not None else None,
            total_charge=actual[..., 0].sum(-1), dipole=dipole,
            density_coefficients=g.unpack(moments)[..., [0, 2, 3, 1]],
            fermi_level=-eq, vacuum_potential=vacuum, work_function=wf,
            charge_potential=eq, counter_charge_potential=ec, charging_potential=eq-ec,
            counter_reference_offset=wf-(eq-ec), reservoir_energy_correction=mechanical-energy,
            potential_reference=g.vacuum_reference-g.plane(g.observer_proto),
            fourier_density=fourier(raw_density), fourier_effective_density=fourier(source),
            fourier_farfield_density=fourier(raw_density*r.farfield_filter(g.k2)),
            fourier_potential=fourier(raw), fourier_total_potential=fourier(raw+g.observer_proto),
            fourier_carrier_potential=fourier(raw_carrier), fourier_local_potential=fourier(raw_local),
            smooth_fourier_density=fourier(raw_density*density_smooth),
            smooth_fourier_farfield_density=fourier(raw_density*density_smooth*r.farfield_filter(g.k2)),
            smooth_fourier_potential=fourier(raw*potential_smooth),
            smooth_vacuum_potential=g.plane(raw*potential_smooth)+g.vacuum_reference,
            k_vectors_mask=g.mask, k_vectors=g.wave, planar_mode_mask=g.axial,
            k_vectors_grid_shape=torch.tensor(g.shape, device=q.device),
            scf_residual=((response[0]-state)*g.present[..., None]).abs().amax((1, 2)),
            scf_steps=q.new_full((len(q),), float(steps)), electrostatic_energy=es,
            electron_energy=contract(chem, basis), external_field=g.external,
            esps=None, esps_dft=None, potential_coefficients=g.unpack(actual[..., 4:]),
            electrostatic_features=g.unpack(spectral[-2]), node_field=g.unpack(spectral[-1]),
            chemical_level=g.unpack(response[5]))
        result['charges_history'] = result['density_coefficients'][..., None]
        attach_observations(data, g, result)
        if 'fourier_density_dft' in result:
            result['fourier_farfield_density_dft'] = result['fourier_density_dft']*r.farfield_filter(g.k2)[..., None]
            result['fourier_farfield_density_dft_mask'] = result['fourier_density_dft_mask']
        if reservoir:
            from .reservoir import SHE_ABSOLUTE_V
            result['md_charge_potential'] = wf-SHE_ABSOLUTE_V
            result['electrode_potential'] = result['md_charge_potential']
        for key in ('energy', 'forces', 'fermi_level', 'vacuum_potential'):
            value = result[key]
            if value is not None and not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f'Nonfinite shared-energy {key}')
        return result
