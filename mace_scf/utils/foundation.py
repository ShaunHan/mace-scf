"""Exact adoption and training-only conditioning of a MACE-POLAR backbone."""
from copy import deepcopy
import logging

import torch
from e3nn import o3
from mace.tools import torch_geometric
from mace.tools.scatter import scatter_sum


def load_foundation(args):
    """Read a local model, preserving its realized tensor-product basis."""
    path = getattr(args, "foundation_model", None)
    if not path:
        return None
    model = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("foundation_model must be a full MACE-POLAR .model file")
    for name in ("joint_embedding", "embedding_readout", "pair_repulsion_fn"):
        if getattr(model, name, None) is not None:
            raise ValueError(f"Foundation module {name} needs an explicit transfer adapter; it cannot be silently omitted")
    if len(getattr(model, "heads", ["Default"])) != 1:
        raise ValueError("Select one foundation head before transfer")
    if abs(float(model.r_max)-float(args.r_max)) > 1.e-8:
        raise ValueError("r_max must match the foundation before constructing neighbours")
    irreps = [o3.Irreps(str(p.linear.irreps_out)) for p in model.products]
    if any(rep != irreps[0] for rep in irreps):
        raise ValueError("This native layer mixer requires equal realized irreps in each foundation layer")
    args.hidden_irreps = str(irreps[0])
    args.num_interactions = len(model.interactions)
    return model.to(dtype=torch.get_default_dtype())


def install_foundation(model, foundation):
    """Adopt modules rather than reinterpret flattened tensor-product weights."""
    source = foundation.atomic_numbers.tolist()
    target = model.atomic_numbers.tolist()
    missing = set(target)-set(source)
    if missing:
        raise ValueError(f"Foundation is missing elements {sorted(missing)}")
    mapping = torch.zeros(len(target), len(source), dtype=torch.get_default_dtype())
    for i, z in enumerate(target):
        mapping[i, source.index(z)] = 1.
    model.register_buffer("foundation_element_map", mapping)
    model.register_buffer("foundation_atomic_numbers", foundation.atomic_numbers.clone())
    for name in ("node_embedding", "radial_embedding", "spherical_harmonics", "interactions", "products", "readouts", "layer_feature_mixer"):
        if hasattr(foundation, name):
            setattr(model, name, deepcopy(getattr(foundation, name)))
    for name in ("node_embedding", "interactions", "products", "readouts", "layer_feature_mixer"):
        getattr(model, name).requires_grad_(True)
    model.radial_embedding.requires_grad_(False)
    scale_shift = getattr(foundation, "scale_shift", None)
    scale = float(scale_shift.scale.reshape(-1)[0]) if scale_shift is not None else 1.
    shift = float(scale_shift.shift.reshape(-1)[0]) if scale_shift is not None else 0.
    model.register_buffer("foundation_energy_scale", torch.tensor(scale))
    model.register_buffer("foundation_energy_shift", torch.tensor(shift))
    logging.info("Foundation: exact module adoption, %d species retained, native neighbour normalization preserved", len(source))


def calibration_loader(loader, maximum=64):
    """Deterministic, evenly spaced training panel; validation is never read."""
    n = len(loader.dataset)
    indices = torch.linspace(0, n-1, min(n, maximum)).round().long().unique().tolist()
    return torch_geometric.dataloader.DataLoader([loader.dataset[i] for i in indices], batch_size=1, shuffle=False)


def condition_energy(model, loader, device):
    """Calibrate the deployed total-energy reference without changing forces."""
    from mace_scf.electrostatics.potential import evaluate_variational
    features, residuals, weights, mu_errors, mu_weights, charges = [], [], [], [], [], []
    states = [p.requires_grad for p in model.parameters()]
    try:
        model.requires_grad_(False)
        for batch in calibration_loader(loader):
            batch = batch.to(device)
            with torch.no_grad():
                output = evaluate_variational(model, batch.to_dict(), steps=50, compute_force=False)
            counts = scatter_sum(batch.node_attrs, batch.batch, dim=0)
            n = counts.sum(-1)
            features.append(counts/n[:, None])
            residuals.append((batch.energy-output["energy"])/n)
            weights.append(batch.weight*batch.energy_weight)
            mu_errors.append(batch.fermi_level-output["fermi_level"])
            mu_weights.append(batch.weight*batch.fermi_level_weight)
            charges.append(output["total_charge"]/n)
        a, b, w = torch.cat(features).double(), torch.cat(residuals).double(), torch.cat(weights).double()
        me, mw = torch.cat(mu_errors).double(), torch.cat(mu_weights).double()
        valid_mu = (mw>0)&torch.isfinite(me)
        if bool(valid_mu.any()):
            shift = (me[valid_mu]*mw[valid_mu]).sum()/mw[valid_mu].sum()
            with torch.no_grad():
                model.fermi_level_offset.add_(shift.to(model.fermi_level_offset))
            b = b+shift*torch.cat(charges).double()
            logging.info("Training-only electronic gauge initialization: offset correction %.6g eV", float(shift))
        use = (w>0)&torch.isfinite(b)
        if bool(use.any()):
            a, b, w = a[use], b[use], w[use].sqrt()
            # SVD acts on the design matrix, not its squared-condition normal equations.
            delta = torch.linalg.lstsq((a*w[:, None]).cpu(), (b*w)[:, None].cpu(),
                                      rcond=1.e-4, driver="gelsd").solution[:, 0].to(a)
            with torch.no_grad():
                e0 = model.atomic_energies_fn.atomic_energies
                e0.add_(delta.to(e0).reshape_as(e0))
            logging.info("Training-only total-energy reference: %.6g -> %.6g eV/atom on %d graphs", float(b.square().mean().sqrt()), float((b-a@delta).square().mean().sqrt()), len(b))
    finally:
        for p, flag in zip(model.parameters(), states):
            p.requires_grad_(flag)
