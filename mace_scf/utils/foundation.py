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
    model.use_reduced_cg = bool(getattr(foundation, 'use_reduced_cg', False))
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


class FusedConvolution(torch.nn.Module):
    """Serializable CuEq tensor product and scatter, with the same CPU map.

    Older MACE releases attach a local Python function to the CuEq module.
    A normal module keeps full-model exports reloadable on those releases too.
    """
    def __init__(self, unfused, dtype):
        super().__init__()
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        descriptor = cue.descriptors.channelwise_tensor_product(
            unfused.irreps_in1, unfused.irreps_in2, unfused.irreps_out)
        polynomial = descriptor.flatten_coefficient_modes().squeeze_modes().polynomial
        self.fused = cuet.SegmentedPolynomial(polynomial, math_dtype=dtype, method='uniform_1d')
        if self.fused.method != 'uniform_1d':
            raise RuntimeError('CuEq fused CUDA convolution is unavailable; check the CUDA ops installation.')
        self.weight_numel = polynomial.operands[0].size
        self.unfused = unfused

    def forward(self, node_feats, edge_attrs, weights, edge_index):
        sender, receiver = edge_index
        if node_feats.device.type == 'cuda':
            return self.fused([weights, node_feats, edge_attrs], {1:sender},
                              {0:node_feats}, {0:receiver})[0]
        messages = self.unfused(node_feats[sender], edge_attrs, weights)
        return scatter_sum(messages, receiver, dim=0, dim_size=len(node_feats))


def accelerate_backbone(model, device):
    """Convert the realized backbone with MACE's CuEq weight transformation.

    Conversion follows conditioning and precedes optimizer/EMA creation. The
    electronic law and readouts stay in their physical/e3nn coordinates; the
    equivariant message-passing blocks use ir_mul and fused CUDA convolution.
    The upstream symmetric-contraction projection preserves the realized CG
    basis, including the older MACE-POLAR basis. No flattened-weight guessing.
    """
    from mace.modules import EquivariantProductBasisBlock
    from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig
    from mace.cli.convert_e3nn_cueq import transfer_weights
    if not CUET_AVAILABLE:
        raise ImportError('enable_cueq=True requires cuequivariance and cuequivariance-torch; install the matching CUDA ops package on GPU hosts.')
    if torch.device(device).type == 'cuda':
        try:
            import cuequivariance_ops_torch  # noqa: F401
        except ImportError as exc:
            raise ImportError('enable_cueq=True on CUDA requires cuequivariance-ops-torch matching torch.version.cuda; refusing a silent unaccelerated run.') from exc
    if getattr(model, 'backbone_layout', 'mul_ir') == 'ir_mul':
        return
    dtype = next(model.parameters()).dtype
    original_dtype = torch.get_default_dtype()
    source = torch.nn.Module()
    source.interactions, source.products = model.interactions, model.products
    target = torch.nn.Module()
    target.interactions, target.products = torch.nn.ModuleList(), torch.nn.ModuleList()
    cuda_fusion = torch.device(device).type == 'cuda'
    # The serializable adapter below avoids the old MACE bound-method export.
    config = CuEquivarianceConfig(enabled=True, layout='ir_mul', group='O3_e3nn',
                                  optimize_all=True, conv_fusion=False)
    reduced = bool(getattr(model, 'use_reduced_cg', False))
    try:
        torch.set_default_dtype(dtype)
        # Backend conversion must not alter later shuffling or initialization.
        with torch.random.fork_rng(devices=[]):
            for interaction, product in zip(source.interactions, source.products):
                names = ('node_attrs_irreps', 'node_feats_irreps', 'edge_attrs_irreps',
                         'edge_feats_irreps', 'target_irreps', 'hidden_irreps', 'radial_MLP')
                options = {name: getattr(interaction, name) for name in names}
                options.update(avg_num_neighbors=float(interaction.avg_num_neighbors),
                               edge_irreps=getattr(interaction, 'edge_irreps', None), cueq_config=config)
                target.interactions.append(type(interaction)(**options))
                contraction = product.symmetric_contractions
                correlation = len(contraction.contractions[0].weights)+1
                target.products.append(EquivariantProductBasisBlock(
                    node_feats_irreps=contraction.irreps_in,
                    target_irreps=o3.Irreps(str(product.linear.irreps_out)),
                    correlation=correlation, use_sc=product.use_sc,
                    num_elements=interaction.node_attrs_irreps.dim,
                    use_agnostic_product=getattr(product, 'use_agnostic_product', False),
                    use_reduced_cg=reduced, cueq_config=config))
            target.to(device=device, dtype=dtype)
            transfer_weights(source, target, o3.Irreps(str(source.products[0].linear.irreps_out)).lmax,
                             correlation, len(source.products), reduced, True)
            if cuda_fusion:
                for interaction in target.interactions:
                    interaction.conv_tp = FusedConvolution(interaction.conv_tp, dtype).to(device)
                    interaction.conv_fusion = True
    finally:
        torch.set_default_dtype(original_dtype)
    for old, new in zip(source.interactions, target.interactions):
        new.requires_grad_(any(p.requires_grad for p in old.parameters()))
    for old, new in zip(source.products, target.products):
        new.requires_grad_(any(p.requires_grad for p in old.parameters()))
    # One common realized feature irrep is required by the layer mixer.
    irreps = o3.Irreps(str(source.products[0].linear.irreps_out))
    indices=[]
    for (mul, ir), sl in zip(irreps, irreps.slices()):
        indices.append(torch.arange(sl.start, sl.stop).reshape(ir.dim,mul).T.flatten())
    model.register_buffer('backbone_to_e3nn', torch.cat(indices).to(device), persistent=False)
    model.interactions, model.products = target.interactions, target.products
    model.backbone_layout = 'ir_mul'
    model.train(model.training)
    logging.info('CuEquivariance ACTIVE: %d interaction/product layers, O3_e3nn, ir_mul, fused CUDA convolution=%s; electronic response/readouts retain their original coordinates',
                 len(model.interactions), cuda_fusion)


def calibration_loader(loader, maximum=64):
    """Deterministic, evenly spaced training panel; validation is never read."""
    n = len(loader.dataset)
    indices = torch.linspace(0, n-1, min(n, maximum)).round().long().unique().tolist() if maximum is not None else range(n)
    return torch_geometric.dataloader.DataLoader([loader.dataset[i] for i in indices],
        batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(0))


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
            logging.debug("Training-only EF reference initialization: offset correction %.6g eV", float(shift))
        use = (w>0)&torch.isfinite(b)
        if bool(use.any()):
            a, b, w = a[use], b[use], w[use].sqrt()
            # SVD acts on the design matrix, not its squared-condition normal equations.
            delta = torch.linalg.lstsq((a*w[:, None]).cpu(), (b*w)[:, None].cpu(),
                                      rcond=1.e-4, driver="gelsd").solution[:, 0].to(a)
            with torch.no_grad():
                e0 = model.atomic_energies_fn.atomic_energies
                e0.add_(delta.to(e0).reshape_as(e0))
            logging.debug("Training-only total-energy reference: %.6g -> %.6g eV/atom on %d graphs", float(b.square().mean().sqrt()), float((b-a@delta).square().mean().sqrt()), len(b))
    finally:
        for p, flag in zip(model.parameters(), states):
            p.requires_grad_(flag)


def condition_readouts(model, loader, device, fit_forces=True):
    """Condition transferred readouts in fixed, rotation-invariant feature units.

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
                logging.debug('Training-only foundation force scale: %.6g; local F RMSE %.6g -> %.6g eV/A; folded into readout weights',
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
            logging.debug('Fixed training-geometry readout units by layer/irrep: %s; initial energy function preserved by inverse weight transformation',
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
