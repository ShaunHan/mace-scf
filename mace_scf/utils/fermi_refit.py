"""Training-only ridge fit of the existing common chemical-level readout."""
import numpy as np
import torch

from mace_scf.electrostatics.coupled import evaluate_coupled


DEFAULT_ALPHAS = (1.e-6, 1.e-5, 1.e-4, 1.e-3, .01, .1, 1., 10., 100., 1000.)


def _ridge(features, target, weight, alpha):
    mean = np.average(features, axis=0, weights=weight)
    scale = np.sqrt(np.average((features-mean)**2, axis=0, weights=weight)).clip(1.e-8)
    target_mean = np.average(target, weights=weight)
    z = (features-mean)/scale
    gram = z.T@(z*weight[:, None])/weight.sum()
    right = z.T@((target-target_mean)*weight)/weight.sum()
    coefficient = np.linalg.solve(gram+alpha*np.eye(features.shape[1]), right)/scale
    return coefficient, target_mean-mean@coefficient


def fit_fermi_arrays(features, fermi_error, weight, previous_weight,
                     seed=20261020, folds=5, alphas=DEFAULT_ALPHAS):
    """Select ridge strength using training EF labels; no vacuum/WF input."""
    features, fermi_error, weight, previous_weight = (
        np.asarray(value, dtype=np.float64)
        for value in (features, fermi_error, weight, previous_weight))
    if (features.ndim != 2 or fermi_error.shape != (len(features),)
            or weight.shape != (len(features),)
            or previous_weight.shape != (features.shape[1],)):
        raise ValueError('Inconsistent Fermi-refit array shapes')
    if not np.isfinite(weight).all() or (weight < 0).any():
        raise ValueError('Observation weights must be finite and nonnegative')
    use = weight > 0
    x, error, w = features[use], fermi_error[use], weight[use]
    if (len(x) < folds or folds < 2 or not np.isfinite(x).all()
            or not np.isfinite(error).all() or not np.isfinite(previous_weight).all()):
        raise ValueError('Need finite observed EF examples and at least one per fold')
    if not alphas or not all(np.isfinite(a) and a > 0 for a in alphas):
        raise ValueError('Ridge strengths must be positive and finite')
    # The rest of the converged model stays fixed. Remove its old common-level
    # contribution to obtain targets for the replacement affine readout.
    target = x@previous_weight-error
    partitions = np.array_split(np.random.default_rng(seed).permutation(len(x)), folds)
    history = []
    for alpha in alphas:
        prediction = np.zeros(len(x))
        for held_out in partitions:
            keep = np.ones(len(x), dtype=bool)
            keep[held_out] = False
            coefficient, intercept = _ridge(x[keep], target[keep], w[keep], alpha)
            prediction[held_out] = x[held_out]@coefficient+intercept
        history.append(dict(alpha=float(alpha), cv_EF_RMSE_meV=float(
            1000*np.sqrt(np.average((prediction-target)**2, weights=w)))))
    chosen = min(history, key=lambda row: row['cv_EF_RMSE_meV'])
    coefficient, intercept = _ridge(x, target, w, chosen['alpha'])
    error_after = error+x@(coefficient-previous_weight)+intercept
    report = dict(
        observed_training_frames=len(x), seed=seed, folds=folds,
        history=history, chosen=chosen,
        training_EF_RMSE_before_meV=float(1000*np.sqrt(np.average(error**2, weights=w))),
        training_EF_RMSE_after_meV=float(1000*np.sqrt(np.average(error_after**2, weights=w))),
        objective='Fermi-level labels only; no WF or vacuum labels',
        validation_fitting=False,
        limitation='Frozen features already saw training data. These folds screen the readout, not the full model.',
    )
    return coefficient, float(intercept), report


def refit_fermi_readout(model, loader, device, seed=20261020):
    """Mutate only common_level.weight and scalar_reference[0], after fitting."""
    response = model.field_dependent_charges_map
    head = getattr(response, 'common_level', None)
    if not getattr(response, 'coupled', False) or not isinstance(head, torch.nn.Linear):
        raise ValueError('A coupled model with the existing linear common-level readout is required')
    if head.out_features != 1 or head.bias is not None:
        raise ValueError('Unexpected common-level readout structure')
    old_weight = head.weight.detach().cpu().double().numpy().reshape(-1).copy()
    captured, features, errors, weights = [], [], [], []
    handle = head.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().cpu().double().numpy()))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                batch = batch.to(device)
                output = evaluate_coupled(
                    model, batch.to_dict(), steps=int(response.deployment_steps),
                    mixing=float(response.deployment_mixing), compute_force=False,
                    reference_conditioning=False)
                if len(captured) != 1:
                    raise ValueError('Expected one terminal common-level observation per batch')
                features.append(captured[0])
                errors.append((output['fermi_level']-batch.fermi_level).reshape(-1).cpu().double().numpy())
                weights.append((batch.weight*batch.fermi_level_weight).reshape(-1).cpu().double().numpy())
    finally:
        handle.remove()
        model.train(was_training)
    if not features:
        raise ValueError('Training loader is empty')
    coefficient, intercept, report = fit_fermi_arrays(
        np.concatenate(features), np.concatenate(errors), np.concatenate(weights),
        old_weight, seed=seed)
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(coefficient, device=head.weight.device,
                                         dtype=head.weight.dtype).reshape_as(head.weight))
        response.scalar_reference[0].add_(intercept)
    report['SCF_steps'] = int(response.deployment_steps)
    report['changed_parameters'] = ['field_dependent_charges_map.common_level.weight']
    report['changed_buffers'] = ['field_dependent_charges_map.scalar_reference[0]']
    return report
