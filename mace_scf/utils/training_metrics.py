"""Epoch RMSEs from predictions already used for optimizer updates.

These are online training diagnostics: parameters and the training SCF budget
can differ from the EMA/deployment predictor used for validation. No model call
or backward pass is performed here.
"""

import torch

from mace_scf.electrostatics.loss import WeightedWorkFunction, spectral_errors


FIELD_ERRORS = (("rmse_rho", "fourier_density"), ("rmse_rho_farfield", "fourier_farfield_density"),
                ("rmse_esp", "fourier_potential"), ("rmse_esp_smooth", "smooth_fourier_potential"))
RMSE_KEYS = ("rmse_e_per_atom", "rmse_f", "rmse_mu_per_atom", *[key for key, _ in FIELD_ERRORS], "rmse_wf_abs")


@torch.no_grad()
def training_error_sums(ref, pred):
    """Return squared-error sums and denominators with validation's masks/units."""
    sums = torch.zeros((len(RMSE_KEYS), 2), dtype=torch.float64,
                       device=ref.positions.device)

    def add(key, square, weights):
        weights = torch.broadcast_to(weights, square.shape).double()
        observed = weights > 0
        weights = torch.where(observed, weights, 0.)
        sums[RMSE_KEYS.index(key)] = torch.stack(((torch.where(observed, square, 0.).double()*weights).sum(),
                                                   weights.sum()))

    counts = ref.ptr[1:] - ref.ptr[:-1]
    if pred.get("energy") is not None and getattr(ref, "energy", None) is not None:
        error = (pred["energy"] - ref.energy) / counts
        add("rmse_e_per_atom", error.double().square(), ref.weight * ref.energy_weight > 0)
    if pred.get("forces") is not None and getattr(ref, "forces", None) is not None:
        mask = (ref.weight * ref.forces_weight)[ref.batch, None] > 0
        add("rmse_f", (pred["forces"] - ref.forces).double().square(), mask)
    if pred.get("dipole") is not None and getattr(ref, "dipole", None) is not None:
        error = (pred["dipole"] - ref.dipole).reshape(-1, 3) / counts[:, None]
        add("rmse_mu_per_atom", error.double().square(), ref.dipole_weight.reshape(-1, 3) > 0)
    for key, field in FIELD_ERRORS:
        if field + "_dft" in pred:
            label = "fourier_potential" if field.endswith("potential") else "fourier_density"
            add(key, spectral_errors(ref, pred, field), ref.weight * getattr(ref, label + "_weight"))
    if pred.get("work_function") is not None and getattr(ref, "vacuum_potential", None) is not None:
        error, weights = WeightedWorkFunction().errors(ref, pred)
        add("rmse_wf_abs", error.double().square(), weights)
    return sums


def training_error_metrics(sums):
    """Pool sums before the square root; never average minibatch RMSEs."""
    return {key: (square / count) ** 0.5
            for key, (square, count) in zip(RMSE_KEYS, sums.detach().cpu().tolist())
            if count > 0}
