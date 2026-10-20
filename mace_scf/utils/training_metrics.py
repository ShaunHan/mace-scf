"""Epoch RMSEs from predictions already used for optimizer updates.

These are online training diagnostics: parameters and the training SCF budget
can differ from the EMA/deployment predictor used for validation. No model call,
reference calibration, or backward pass is performed here.
"""

import torch

from mace_scf.electrostatics.loss import spectral_errors, vacuum_reference_weights


RMSE_KEYS = ("rmse_e_per_atom", "rmse_f", "rmse_mu_per_atom",
             "rmse_rho", "rmse_esp", "rmse_wf_abs",
             "rmse_rho_smooth", "rmse_rho_farfield_smooth", "rmse_esp_smooth")


def smooth_field_errors(ref, pred, loss_fn=None):
    """Smooth latent fields versus internally smoothed labels, in base units.

    Raw input Fourier labels are assumed. The extra far-field density filter
    is identified separately; none of these diagnostics changes the loss.
    """
    density=getattr(loss_fn, 'loss_fns', {}).get('fourier_density')
    if density is not None:
        pred=density.reference_fields(pred)
    for name, key, label in (
            ('rho_smooth', 'fourier_density', 'fourier_density'),
            ('rho_farfield_smooth', 'fourier_farfield_density', 'fourier_density'),
            ('esp_smooth', 'fourier_potential', 'fourier_potential')):
        width = 'density_smearing_width' if label == 'fourier_density' else 'potential_smearing_width'
        if key not in pred or key+'_dft' not in pred or width not in pred:
            continue
        value = pred.get('smooth_'+key, pred[key])
        function = getattr(loss_fn, 'loss_fns', {}).get(label)
        direct_raw = (function is not None and not function.desmearing
                      and not function.apply_smearing)
        if direct_raw:
            # A direct raw fit predicts the unsmeared observable. For its
            # smeared diagnostic filter both prediction and label once.
            value = pred[key]*torch.exp(-.5*pred[width]**2*pred['k_vectors'].square().sum(-1))[...,None]
        latent = dict(pred, **{key: value})
        yield 'rmse_'+name, spectral_errors(ref, latent, key, True), ref.weight*getattr(ref, label+'_weight')


@torch.no_grad()
def training_error_sums(ref, pred, loss_fn):
    """Return squared-error sums and denominators with validation's masks/units."""
    sums = torch.zeros((len(RMSE_KEYS), 2), dtype=torch.float64,
                       device=ref.positions.device)

    def add(index, square, weights):
        weights = torch.broadcast_to(weights, square.shape).double()
        observed = weights > 0
        square = torch.where(observed, square, 0.).double()
        weights = torch.where(observed, weights, 0.)
        sums[index, 0] = (square * weights).sum()
        sums[index, 1] = weights.sum()

    counts = ref.ptr[1:] - ref.ptr[:-1]
    if pred.get("energy") is not None and getattr(ref, "energy", None) is not None:
        error = (pred["energy"] - ref.energy) / counts
        add(0, error.double().square(), ref.weight * ref.energy_weight > 0)
    if pred.get("forces") is not None and getattr(ref, "forces", None) is not None:
        mask = (ref.weight * ref.forces_weight)[ref.batch, None] > 0
        add(1, (pred["forces"] - ref.forces).double().square(), mask)
    if pred.get("dipole") is not None and getattr(ref, "dipole", None) is not None:
        error = (pred["dipole"] - ref.dipole).reshape(-1, 3) / counts[:, None]
        add(2, error.double().square(), ref.dipole_weight.reshape(-1, 3) > 0)

    for index, key in ((3, "fourier_density"), (4, "fourier_potential")):
        if pred.get(key) is not None and pred.get(key + "_dft") is not None:
            weights = ref.weight * getattr(ref, key + "_weight")
            # Keep the reported quantity independent of the chosen loss filter.
            add(index, spectral_errors(ref, pred, key), weights)
    for key, error, weights in smooth_field_errors(ref, pred, loss_fn):
        add(RMSE_KEYS.index(key), error, weights)
    if (pred.get("vacuum_potential") is not None and pred.get("fermi_level") is not None
            and getattr(ref, "vacuum_potential", None) is not None
            and getattr(ref, "fermi_level", None) is not None):
        target, weights = vacuum_reference_weights(ref, pred["vacuum_potential"])
        weights = weights * (ref.fermi_level_weight > 0)
        error = (pred["vacuum_potential"] - pred["fermi_level"]) - (target - ref.fermi_level)
        add(5, error.double().square(), weights)
    return sums


def training_error_metrics(sums):
    """Pool sums before the square root; never average minibatch RMSEs."""
    return {key: (square / count) ** 0.5
            for key, (square, count) in zip(RMSE_KEYS, sums.detach().cpu().tolist())
            if count > 0}
