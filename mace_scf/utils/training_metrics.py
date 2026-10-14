"""Epoch RMSEs from predictions already used for optimizer updates.

These are online training diagnostics: parameters and the training SCF budget
can differ from the EMA/deployment predictor used for validation. No model call,
reference calibration, or backward pass is performed here.
"""

import torch

from mace_scf.electrostatics.loss import spectral_errors, vacuum_reference_weights


RMSE_KEYS = ("rmse_e_per_atom", "rmse_f", "rmse_mu_per_atom",
             "rmse_rho", "rmse_esp", "rmse_phi_abs")


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

    functions = getattr(loss_fn, "loss_fns", {})
    density_reference = getattr(functions.get("fourier_density"), "reference", "density")
    for index, key in ((3, "fourier_density"), (4, "fourier_potential")):
        observer = ("fourier_farfield_density" if key == "fourier_density"
                    and density_reference == "farfield" else key)
        if pred.get(observer) is not None and pred.get(observer + "_dft") is not None:
            weights = ref.weight * getattr(ref, key + "_weight")
            add(index, spectral_errors(ref, pred, observer), weights)
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
