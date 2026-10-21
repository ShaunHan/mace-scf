"""Fit a Gaussian inverse once on training DFT pairs, independently of SCF.

The recurrent state stays smooth. `reference` selects the training loss;
`desmearing` selects the exported field resolution. The fitted map has no
optimizer parameters. SpectralDecoder is retained only to read old models.
"""
import math
import logging

import torch
from .molecular import moment_sum


class SpectralDecoder(torch.nn.Module):
    """Legacy jointly trained decoder; new runs use FrozenSpectralDecoder."""
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
        # Combine attenuation and reconstruction inside analytic kernels:
        # no overflowing inverse times an underflowed FFT.
        coefficients = torch.bmm(g.phase, state.to(g.phase.dtype))
        moment = coefficients[..., 0]-1j*r.density_width*(
            coefficients[..., 1:4]*g.wave).sum(-1)
        sigma = r.density_width
        if bool(self.enabled[0]) and 'fourier_density' in output:
            fraction = self.fraction(g.k2, sigma, 0)
            rho = moment*torch.exp(-.5*sigma**2*g.k2*(1-fraction))/g.volume[:, None]*g.mask
            output['smooth_fourier_density'] = output['fourier_density']
            output['smooth_fourier_farfield_density'] = output['fourier_farfield_density']
            output['fourier_density'] = torch.view_as_real(rho)*g.ngrid
            output['fourier_farfield_density'] = output['fourier_density']*r.farfield_filter(g.k2)[..., None]
        if bool(self.enabled[1]):
            # The legacy fraction limits its inverse to this source width.
            # Frozen maps instead return the fitted observation variance,
            # including independently chosen density/potential widths.
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


class FrozenSpectralDecoder(SpectralDecoder):
    """DFT-fitted inverse exp(variance*k^2/2), evaluated in analytic kernels.

    Combining exponents before evaluation avoids multiplying an underflowed
    smoothed spectrum by an overflowing inverse. No new modes are invented.
    """
    def __init__(self, density=True, potential=True):
        torch.nn.Module.__init__(self)
        self.register_buffer('enabled', torch.tensor([density, potential], dtype=torch.bool))
        self.register_buffer('variances', torch.zeros(2))
        self.register_buffer('fitted', torch.zeros(2, dtype=torch.bool))
        self.register_buffer('observations', torch.zeros(2, dtype=torch.long))
        self.register_buffer('consistent_smearing', torch.tensor(False))

    def fraction(self, k2, width, index):
        if not bool(self.fitted[index]):
            raise RuntimeError('Fit the desmearing map on training DFT fields before evaluation')
        return self.variances[index].to(k2)/width**2

    def _load_from_state_dict(self, *args, **kwargs):
        # Frozen buffers must all be present; never fill a missing map by
        # interpreting the four legacy live-decoder parameters as a DFT fit.
        torch.nn.Module._load_from_state_dict(self, *args, **kwargs)


def observe_smoothed_fields(g, state, output):
    """Give the loss the same smoothed field whether desmearing is on or off.

    Known counter-charge/boundary terms are filtered as observations only;
    their physical width, SCF drive and energy are never changed.
    """
    r = g.response
    width = r.potential_amplitudes_smearing_width
    gaussian = torch.exp(-.5*width**2*g.k2)
    moments = torch.cat((state[..., :1], state[..., 1:4]*r.density_width), -1)
    dipole = (moments[..., 0]*g.tensors[10]
              +(moments[..., 1:4]*g.normal[:, None]).sum(-1)).sum(-1)
    field = (g.external*g.normal).sum(-1)-g.slab_factor*(dipole+g.counter_moment)
    known = g.ramp(field, width=0.)-g.coulomb*g.counter_density
    if g.molecular is not None:
        known = known+g.molecular.spectrum(moment_sum(state, g.molecular_source)+g.molecular.counter)
    old = output['fourier_potential']
    smooth = output['fourier_carrier_potential']+output['fourier_local_potential']
    smooth = smooth+torch.view_as_real(known*gaussian)*g.ngrid
    decoder = getattr(r, 'raw_decoder', None)
    reconstruct = decoder is not None and bool(decoder.enabled[1])
    # Proto form factors are also once-fitted observations. Render their
    # saved resolution consistently with the requested output field.
    proto = (output['fourier_total_potential']-old)/g.ngrid
    saved_width = float(getattr(r, 'proto_smearing_width', 0.) or 0.)
    variance = saved_width**2-(0. if reconstruct else width**2)
    gain = torch.exp(torch.where(g.mask, .5*variance*g.k2, 0.))[..., None]
    proto_change = proto*(gain-1)*g.ngrid
    output['fourier_total_potential'] = output['fourier_total_potential']+smooth-old+proto_change
    output['potential_reference'] = output['potential_reference']-g.plane(
        torch.view_as_complex((proto_change/g.ngrid).contiguous()))
    output['vacuum_potential'] = output['vacuum_potential']+g.plane(
        torch.view_as_complex(((smooth-old)/g.ngrid).contiguous()))
    output['fourier_potential'] = smooth
    for key in ('fourier_density', 'fourier_farfield_density', 'fourier_potential', 'vacuum_potential'):
        if key in output:
            output['smooth_'+key] = output[key]
    return output


def field_loss_options(options, *, density=True):
    """Reference alone selects the loss; desmearing only selects inference."""
    options = options if isinstance(options, dict) else {}
    reference = options.get('reference', 'smeared' if options.get('apply_smearing', True) else 'raw')
    if density and reference == 'density':
        reference = 'smeared'
    allowed = ('raw', 'smeared', 'farfield') if density else ('raw', 'smeared')
    if reference not in allowed:
        raise ValueError(f'Fourier {"density" if density else "potential"} reference must be one of {allowed}')
    reconstruct = options.get('desmearing', False)
    if not isinstance(reconstruct, bool):
        raise TypeError('desmearing must be a boolean')
    return reference, reconstruct, reference != 'raw'


def desmearing_enabled(options, *, density=True):
    return field_loss_options(options, density=density)[1]


def target_smearing_enabled(options):
    """Loss target resolution, independent of whether an inverse is fitted."""
    return field_loss_options(options, density=False)[2]


def configure_desmearing(response, loss_config):
    response.raw_vacuum_boundary = True
    response.fixed_dft_observations = True
    if not hasattr(response, 'potential_amplitudes_smearing_width'):
        response.potential_amplitudes_smearing_width = (float(response.source_widths[0])
            if len(response.source_widths) else response.density_width)
    enabled = [key in loss_config and desmearing_enabled(loss_config[key], density=key == 'fourier_density')
               for key in ('fourier_density', 'fourier_potential')]
    if any(enabled):
        if not isinstance(getattr(response, 'raw_decoder', None), FrozenSpectralDecoder):
            response.raw_decoder = FrozenSpectralDecoder(*enabled).to(response.source_widths)
        else:
            response.raw_decoder.enabled.copy_(torch.tensor(enabled, device=response.source_widths.device))
    else:
        response.raw_decoder = None


def restore_desmearing(response, saved):
    """Reuse a saved DFT map before initialization, without reading DFT again."""
    decoder = getattr(response, 'raw_decoder', None)
    prefix = 'field_dependent_charges_map.raw_decoder.'
    if decoder is None or prefix+'variances' not in saved:
        return
    enabled = decoder.enabled.clone()
    fields = {key[len(prefix):]: value for key, value in saved.items() if key.startswith(prefix)}
    widths = decoder.variances.new_tensor([response.density_width, response.potential_amplitudes_smearing_width])
    fitted = fields['fitted'].to(widths.device)
    if not torch.allclose(fields['variances'].to(widths)[fitted], widths.square()[fitted], rtol=1.e-8, atol=1.e-12):
        raise ValueError('Saved desmearing widths differ from the requested widths; keep the checkpoint configuration to resume')
    decoder.load_state_dict(fields, strict=True)
    decoder.enabled.copy_(enabled)
    logging.info('Reusing the checkpoint DFT desmearing map; already fitted fields are not fitted again')


@torch.no_grad()
def calibrate_desmearing(model, loader, device):
    """One closed-form fit on training DFT pairs; no model predictions or SCF.

    Stored labels are unsmoothed. Construct each paired smoothed label with
    its declared source width and fit log(|DFT|/|smoothed DFT|)=variance*k^2/2.
    For exact Gaussian pairs this recovers that Gaussian's variance, rather
    than learning to compensate for errors in a particular model checkpoint.
    A fitted buffer is reused on resume and at every later stage/inference.
    """
    from .potential import reciprocal_grid, _target_fft, target_mode_mask
    from mace.tools import torch_geometric
    r = model.field_dependent_charges_map
    decoder = getattr(r, 'raw_decoder', None)
    if decoder is None:
        return
    if not isinstance(decoder, FrozenSpectralDecoder):
        raise TypeError('Configure a frozen DFT decoder before fitting')
    if bool((decoder.fitted | ~decoder.enabled).all()):
        return
    widths = [r.density_width, r.potential_amplitudes_smearing_width]
    sums = torch.zeros((2, 2), device=device, dtype=torch.float64)
    counts = torch.zeros(2, device=device, dtype=torch.long)
    fitting = torch_geometric.dataloader.DataLoader(loader.dataset,
        batch_size=getattr(loader, 'batch_size', 1) or 1, shuffle=False, drop_last=False,
        generator=torch.Generator().manual_seed(0))
    for batch in fitting:
        data = batch.to(device).to_dict()
        _, k2, mask, shape, modes = reciprocal_grid(data['cell'].reshape(-1, 3, 3).double(), model.kspace_cutoff)
        for i, key in enumerate(('fourier_density', 'fourier_potential')):
            if not bool(decoder.enabled[i] & ~decoder.fitted[i]):
                continue
            target = _target_fft(data, key, key+'_shape', shape, torch.float64)
            if target is None:
                continue
            weight = data['weight']*data[key+'_weight']
            if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
                raise ValueError('DFT desmearing weights must be finite and nonnegative')
            use = mask & (k2 > 0) & target_mode_mask(data[key+'_shape'], modes) & (weight > 0)[:, None]
            amplitude = torch.linalg.vector_norm(target, dim=-1)/math.prod(shape)
            smooth = amplitude*torch.exp(-.5*widths[i]**2*k2)
            if bool((use & ~torch.isfinite(amplitude)).any()):
                raise ValueError('Nonfinite training DFT field: '+key)
            use &= (smooth > torch.finfo(torch.float64).tiny) & (amplitude > 0)
            x = .5*k2[use]
            y = amplitude[use].log()-smooth[use].log()
            w = (weight[:, None]*amplitude.square())[use]
            sums[i, 0] += (w*x*y).sum()
            sums[i, 1] += (w*x*x).sum()
            counts[i] += use.sum()
    for i in range(2):
        if not bool(decoder.enabled[i] & ~decoder.fitted[i]):
            continue
        if not bool(sums[i, 1] > 0):
            raise ValueError('Enabled desmearing requires nonzero training DFT Fourier observations')
        variance = sums[i, 0]/sums[i, 1]
        if not bool(torch.isfinite(variance)) or not torch.isclose(variance, variance.new_tensor(widths[i]**2), rtol=1.e-8, atol=1.e-12):
            raise ValueError('DFT pairs do not obey the declared Gaussian smoothing convention')
        decoder.variances[i].copy_(variance.clamp(0., widths[i]**2))
        decoder.observations[i].copy_(counts[i])
        decoder.fitted[i] = True
    logging.info('Frozen DFT desmearing: variances=%s Angstrom^2, observed modes=%s; '
                 'training labels only, no SCF evaluations, no optimizer parameters',
                 decoder.variances.tolist(), decoder.observations.tolist())
