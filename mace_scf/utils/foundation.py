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
    from mace_scf.electrostatics.potential import evaluate_electronic
    features, residuals, weights, mu_errors, mu_weights, charges = [], [], [], [], [], []
    states = [p.requires_grad for p in model.parameters()]
    try:
        model.requires_grad_(False)
        for batch in calibration_loader(loader):
            batch = batch.to(device)
            with torch.no_grad():
                output = evaluate_electronic(model, batch.to_dict(), steps=50, compute_force=False)
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
            logging.info("Training-only EF reference initialization: offset correction %.6g eV", float(shift))
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


def condition_readouts(model, loader, device, fit_forces=True):
    """Restore v366's fixed readout units and local-force initialization.

    Fit a single nonnegative local energy scale on training forces when a
    foundation is transferred. Fold that scale into its output weights before
    optimizer creation. RMS input units are rotation invariant and fixed; an
    exact inverse change of energy-readout weights preserves its initial E/F.
    Density readouts are zero at this point, so their initial function is also
    preserved. No backbone tensor-product weights or validation labels change.
    """
    if not hasattr(model, 'readout_feature_units'):
        return
    irreps = [o3.Irreps(str(p.linear.irreps_out)) for p in model.products]
    sums = [model.readout_feature_units.new_zeros(len(rep)) for rep in irreps]
    count = model.readout_feature_units.new_zeros(())
    pp, py, yy, force_count = [count.clone() for _ in range(4)]
    atom_weights = None
    hooks = []
    flags = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    def collect(index):
        def hook(module, inputs, output):
            value = output.detach()
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite readout features before conditioning')
            for j,sl in enumerate(irreps[index].slices()):
                sums[index][j] += (value[:,sl].square().mean(-1)*atom_weights).sum()
        return hook
    try:
        model.eval(); model.requires_grad_(False)
        for index, module in enumerate(model.products):
            hooks.append(module.register_forward_hook(collect(index)))
        for batch in calibration_loader(loader):
            data=batch.to(device).to_dict()
            sizes=(data['ptr'][1:]-data['ptr'][:-1]).to(data['positions'])
            atom_weights=(data['weight'].reshape(-1)/sizes)[data['batch']]
            force_weight=data['weight']*data['forces_weight']
            compute_force=fit_forces and hasattr(model,'foundation_element_map') and bool((force_weight>0).any())
            with torch.set_grad_enabled(compute_force):
                local=model.local_part(data,compute_force=compute_force)
                if compute_force:
                    force=-torch.autograd.grad(local.energies.sum(),data['positions'])[0]
                    weights=force_weight[data['batch']][:,None].expand_as(force)
                    use=weights>0
                    p,y,w=force.detach()[use],data['forces'][use],weights[use]
                    if not bool(torch.isfinite(p).all() & torch.isfinite(y).all()):
                        raise FloatingPointError('Nonfinite observed local forces before conditioning')
                    pp+=(w*p.square()).sum();py+=(w*p*y).sum();yy+=(w*y.square()).sum();force_count+=w.sum()
            count+=data['weight'].sum()
        with torch.no_grad():
            if force_count>0 and pp>0:
                scale=(py/pp).clamp_min(0.)
                seen=set()
                for readout in model.readouts:
                    output=getattr(readout,'linear_2',getattr(readout,'linear',None))
                    if output is None:
                        raise TypeError('Cannot identify the final energy readout for conditioning')
                    for parameter in output.parameters():
                        if id(parameter) not in seen:
                            parameter.mul_(scale);seen.add(id(parameter))
                logging.info('Training-only foundation force scale: %.6g; local F RMSE %.6g -> %.6g eV/A; folded into readout weights',
                    float(scale),float(((pp-2*py+yy)/force_count).clamp_min(0).sqrt()),
                    float(((scale.square()*pp-2*scale*py+yy)/force_count).clamp_min(0).sqrt()))
            # A shared energy head must use one unit set across all layers.
            if len(model.readouts)==1:
                shared=sum(sums)/len(sums);sums=[shared for _ in sums]
            units=[(1.+value/count.clamp_min(1.e-30)).sqrt() for value in sums]
            seen=set()
            for index,(rep,unit) in enumerate(zip(irreps,units)):
                readout=model.readouts[0 if len(model.readouts)==1 else index]
                first=getattr(readout,'linear_1',getattr(readout,'linear',None))
                if first is None or not hasattr(first,'weight_views'):
                    raise TypeError('Energy readout needs an equivariant linear input map for exact unit folding')
                if id(first) not in seen:
                    last=getattr(readout,'linear_2',first)
                    zero_output=all(not bool(parameter.any()) for parameter in last.parameters())
                    # A rejected pretrained energy head is already identically
                    # zero. Retain bounded hidden coordinates, as in v366;
                    # recreating its old large hidden activations would defeat
                    # conditioning on the first nonzero output-weight update.
                    if not zero_output:
                        by_irrep={ir: value for (_,ir),value in zip(rep,unit)}
                        for _,instruction,weight in first.weight_views(yield_instruction=True):
                            if instruction.i_in>=0:
                                weight.mul_(by_irrep[first.irreps_in[instruction.i_in].ir])
                    seen.add(id(first))
                expanded=torch.cat([value.expand(mul*ir.dim) for (mul,ir),value in zip(rep,unit)])
                model.readout_feature_units[index].copy_(expanded)
            logging.info('Fixed training-geometry readout units by layer/irrep: %s; initial energy function preserved by inverse weight transformation',
                         [unit.tolist() for unit in units])
    finally:
        for hook in hooks:hook.remove()
        for p,flag in zip(model.parameters(),flags):p.requires_grad_(flag)
        model.train(was_training)


def set_foundation_stage(model, frozen):
    """Train new heads before unfreezing the transferred feature extractor."""
    if not isinstance(frozen,bool):
        raise TypeError('freeze_foundation_backbone must be bool')
    if not hasattr(model,'foundation_element_map'):
        if frozen:raise ValueError('freeze_foundation_backbone requires foundation_model')
        return
    for name in ('node_embedding','interactions','products'):
        getattr(model,name).requires_grad_(not frozen)
    logging.info('Foundation feature extractor frozen=%s; energy/electronic readouts remain trainable',frozen)
