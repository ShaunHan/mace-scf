"""Learn a bounded, isotropic reconstruction of the retained raw DFT modes.

The recurrent state and Gaussian receivers stay smooth. Two parameters per
field learn how much of its Gaussian attenuation to undo; raw and smooth
observations both train the live SCF. Analytic boundary fields are kept separate.
"""
import math
import logging

import torch
from .molecular import moment_sum


class SpectralDecoder(torch.nn.Module):
    def __init__(self, density=True, potential=True):
        super().__init__()
        self.coefficients = torch.nn.Parameter(torch.tensor([[-4., 0.], [-4., 0.]]))
        self.register_buffer('enabled', torch.tensor([density, potential], dtype=torch.bool))
        # Restore the direct Gaussian-SCF auxiliary loss used by the working
        # raw-plus-smeared baseline. Existing checkpoints keep their convention.
        self.register_buffer('consistent_smearing', torch.tensor(False))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Old checkpoints retain their saved objective/observer convention.
        state_dict = dict(state_dict)
        state_dict.setdefault(prefix+'consistent_smearing', self.enabled.new_tensor(False))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    def fraction(self, k2, width, index):
        t = .5*width**2*k2
        a, b = self.coefficients[index]
        return torch.sigmoid(a+b*t/(1+t))

    def forward(self, geometry, state, output):
        g, r = geometry, geometry.response
        # Analytic evaluation combines attenuation and reconstruction in one
        # nonpositive exponent: no overflowing inverse times an underflowed FFT.
        coefficients = torch.bmm(g.phase, state.to(g.phase.dtype))
        moment = coefficients[..., 0]-1j*r.density_width*(
            coefficients[..., 1:4]*g.wave).sum(-1)
        sigma = r.density_width
        if bool(self.enabled[0]):
            fraction = self.fraction(g.k2, sigma, 0)
            rho = moment*torch.exp(-.5*sigma**2*g.k2*(1-fraction))/g.volume[:, None]*g.mask
            output['smooth_fourier_density'] = output['fourier_density']
            output['smooth_fourier_farfield_density'] = output['fourier_farfield_density']
            output['fourier_density'] = torch.view_as_real(rho)*g.ngrid
            output['fourier_farfield_density'] = output['fourier_density']*r.farfield_filter(g.k2)[..., None]
        if bool(self.enabled[1]):
            # Limit the common inverse to the narrowest represented source.
            # This also supports independently chosen density/potential widths.
            widths = r.source_widths.to(g.k2)
            width = min(sigma, r.potential_amplitudes_smearing_width,
                        float(widths.min()) if len(widths) else sigma)
            variance = width**2*self.fraction(g.k2, width, 1)
            carrier = -g.coulomb*moment*torch.exp(-.5*(sigma**2-variance)*g.k2)/g.volume[:, None]
            local = torch.zeros_like(carrier)
            if len(widths):
                radial = torch.exp(-.5*(widths.square()-variance[..., None])*g.k2[..., None])
                if getattr(r, 'source_basis', 'difference') == 'difference':
                    radial = torch.cat((radial[..., :1], radial[..., 1:]-radial[..., :1]), -1)
                n = len(widths)
                amplitudes = coefficients[..., 4:4+n]+1j*widths[0]*(
                    coefficients[..., 4+n:].reshape(*g.k2.shape, n, 3)*g.wave[..., None, :]).sum(-1)
                local = (radial*amplitudes).sum(-1)*(2*math.pi)**1.5*sigma**3/g.volume[:, None]
                local = local*g.mask*(g.k2 > 0)
            moments = torch.cat((state[..., :1], state[..., 1:4]*sigma), -1)
            normal_dipole = (moments[..., 0]*g.tensors[10]
                +(moments[..., 1:4]*g.normal[:, None]).sum(-1)).sum(-1)
            boundary_field = ((g.external*g.normal).sum(-1)
                -g.slab_factor*(normal_dipole+g.counter_moment))
            boundary = g.ramp(boundary_field, width=0.)
            known = boundary
            # A counter-charge's width describes the physical charge, not label
            # smearing. Preserve it; do not deconvolve it with the learned gain.
            known = known-g.coulomb*g.counter_density
            if g.molecular is not None:
                known = known+g.molecular.spectrum(
                    moment_sum(state, g.molecular_source)+g.molecular.counter)
            raw = torch.view_as_real(carrier+local+known)*g.ngrid
            smooth = output['fourier_carrier_potential']+output['fourier_local_potential']
            smooth = smooth+torch.view_as_real(known)*g.ngrid*torch.exp(
                -.5*r.potential_amplitudes_smearing_width**2*g.k2)[..., None]
            old = output['fourier_potential']
            output['smooth_fourier_potential'] = smooth
            output['smooth_vacuum_potential'] = output['vacuum_potential']
            output['vacuum_potential'] = output['vacuum_potential']+g.plane(
                torch.view_as_complex(((raw-old)/g.ngrid).contiguous()))
            if getattr(r, 'raw_vacuum_boundary', False):
                boundary_correction = g.ramp_plane(boundary_field)-g.plane(boundary)
                output['vacuum_potential'] = output['vacuum_potential']+boundary_correction
                output['vacuum_boundary_correction'] = boundary_correction
            output['fourier_total_potential'] = output['fourier_total_potential']+raw-old
            output['fourier_potential'] = raw
            for key, value in (('fourier_carrier_potential', carrier), ('fourier_local_potential', local)):
                output['smooth_'+key] = output[key]
                output[key] = torch.view_as_real(value)*g.ngrid
            if bool(getattr(self, 'consistent_smearing', False)):
                # One observed ESP at two resolutions. The recurrent Gaussian
                # coefficients remain a latent basis; partial reconstruction
                # does not make that basis the Gaussian convolution of raw ESP.
                gaussian = torch.exp(-.5*r.potential_amplitudes_smearing_width**2*g.k2)[..., None]
                for key in ('fourier_carrier_potential', 'fourier_local_potential'):
                    output['latent_'+key] = output['smooth_'+key]
                    output['smooth_'+key] = output[key]*gaussian
                smooth = raw*gaussian
                output['smooth_fourier_potential'] = smooth
                output['smooth_vacuum_potential'] = g.plane(
                    torch.view_as_complex((smooth/g.ngrid).contiguous()))+g.vacuum_reference
            if 'fourier_effective_density' in output:
                output['smooth_fourier_effective_density'] = output['fourier_effective_density']
                output['fourier_effective_density'] = torch.view_as_real(
                    (carrier+local)/torch.where(g.coulomb > 0, -g.coulomb, 1.))*g.ngrid
                if bool(getattr(self, 'consistent_smearing', False)):
                    output['smooth_fourier_effective_density'] = output['fourier_effective_density']*gaussian
        return output


def field_loss_options(options, *, density=True):
    """Resolve target resolution independently of optional raw reconstruction.

    Keep the old density alias and explicit apply_smearing convention readable
    so saved objectives retain their values when a checkpoint is restored.
    """
    options = options if isinstance(options, dict) else {}
    reference = options.get('reference', 'smeared')
    if density and reference == 'density':
        reference = 'smeared'
    allowed = ('raw', 'smeared', 'farfield') if density else ('raw', 'smeared')
    if reference not in allowed:
        raise ValueError(f'Fourier {"density" if density else "potential"} reference must be one of {allowed}')
    reconstruct = options.get('desmearing', 'apply_smearing' not in options)
    smear = options.get('apply_smearing', not reconstruct)
    if not isinstance(reconstruct, bool) or not isinstance(smear, bool):
        raise TypeError('desmearing and legacy apply_smearing must be booleans')
    reconstruct = reconstruct and not options.get('apply_smearing', False)
    if reference == 'raw':
        reconstruct = smear = False
    return reference, reconstruct, smear


def desmearing_enabled(options, *, density=True):
    return field_loss_options(options, density=density)[1]


def target_smearing_enabled(options):
    """Without reconstruction, supervise the Gaussian representation at its width.

    An explicit legacy apply_smearing value retains the preceding objective,
    including old checkpoints that fitted finite-width sources to raw labels.
    """
    return field_loss_options(options, density=False)[2]


def configure_desmearing(response, loss_config):
    response.raw_vacuum_boundary = True
    if not hasattr(response, 'potential_amplitudes_smearing_width'):
        response.potential_amplitudes_smearing_width = (float(response.source_widths[0])
            if len(response.source_widths) else response.density_width)
    enabled = [key in loss_config and desmearing_enabled(loss_config[key], density=key == 'fourier_density')
               for key in ('fourier_density', 'fourier_potential')]
    if any(enabled):
        if getattr(response, 'raw_decoder', None) is None:
            response.raw_decoder = SpectralDecoder(*enabled).to(response.source_widths)
        else:
            response.raw_decoder.enabled.copy_(torch.tensor(enabled, device=response.source_widths.device))
    else:
        response.raw_decoder = None


@torch.no_grad()
def calibrate_desmearing(model, loader, device, steps=50, mixing=.5):
    """Initialize four decoder parameters on training fields; keep neural weights.

    One smooth SCF evaluation per supplied frame is cached. The subsequent tiny
    regression does not rerun SCF. Joint training then updates the complete model.
    """
    from .coupled import evaluate_coupled
    r = model.field_dependent_charges_map
    decoder = getattr(r, 'raw_decoder', None)
    if decoder is None:
        return
    records = [[], []]
    training = model.training
    model.eval()
    try:
        for batch in loader:
            batch = batch.to(device)
            out = evaluate_coupled(model, batch.to_dict(), steps=steps, mixing=mixing, compute_force=False)
            k2 = out['k_vectors'].square().sum(-1)
            for i, key in enumerate(('fourier_density', 'fourier_potential')):
                if not bool(decoder.enabled[i]) or key+'_dft' not in out:
                    continue
                weight = batch.weight*getattr(batch, key+'_weight')
                mask = out['k_vectors_mask'] & out[key+'_dft_mask'] & (weight > 0)[:, None]
                prediction, target = out['smooth_'+key], out[key+'_dft']
                if i == 1:
                    mask &= k2 > 0
                    prefix = 'latent_' if 'latent_fourier_carrier_potential' in out else 'smooth_'
                    prediction = out[prefix+'fourier_carrier_potential']+out[prefix+'fourier_local_potential']
                    known = out[key]-out['fourier_carrier_potential']-out['fourier_local_potential']
                    target = target-known
                w = weight[:, None].expand_as(k2)/out['k_vectors_grid_shape'].prod().square()
                records[i].append(tuple(x[mask].cpu().double() for x in (prediction, target, k2, w)))
        fitted = SpectralDecoder().double()
        fitted.coefficients.copy_(decoder.coefficients.cpu())
        values = [tuple(torch.cat([row[j] for row in rows]) for j in range(4)) if rows else None
                  for rows in records]
        widths = [r.density_width, min(r.density_width, r.potential_amplitudes_smearing_width,
                    float(r.source_widths.min()) if len(r.source_widths) else r.density_width)]
        with torch.enable_grad():
            optimizer = torch.optim.LBFGS(fitted.parameters(), lr=.5, max_iter=160,
                                         tolerance_grad=1e-10, line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad()
                loss = fitted.coefficients.sum()*0.
                for i, record in enumerate(values):
                    if record is None:
                        continue
                    prediction, target, k2, weights = record
                    log_gain = .5*widths[i]**2*k2*fitted.fraction(k2, widths[i], i)
                    # Fixed cached coefficients: log magnitudes avoid 0*inf at
                    # very high k; production evaluates analytic kernels directly.
                    log_value = torch.where(prediction != 0,
                        prediction.abs().log()+log_gain[:, None], -torch.inf)
                    error = prediction.sign()*log_value.exp()-target
                    loss = loss+(10000. if i == 0 else 100.)*(error.square().sum(-1)*weights).sum()
                loss.backward()
                return loss
            optimizer.step(closure)
        decoder.coefficients.copy_(fitted.coefficients.to(decoder.coefficients))
        logging.info('Training-only spectral decoder initialization: %s', decoder.coefficients.tolist())
    finally:
        model.train(training)
