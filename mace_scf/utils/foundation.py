import ast
import inspect
import logging
from copy import deepcopy

import ase.io
import mace.modules
import numpy as np
import torch
from e3nn import o3
from graph_longrange.energy import GTOElectrostaticEnergy
from graph_longrange.features import GTOElectrostaticFeatures
from mace import data
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.scatter import scatter_sum

from mace_scf.electrostatics.potential_response import PotentialResponse, initialize_potential_response
from mace_scf.electrostatics.loss import WeightedLoss


def build_potential_response(args, model_config, train_loader, atomic_energies):
    foundation = None
    if args.foundation_model:
        foundation = torch.load(args.foundation_model, map_location='cpu', weights_only=False)
        foundation = foundation.to(dtype=torch.get_default_dtype())
        if abs(float(foundation.r_max)-float(args.r_max)) > 1.e-8:
            raise ValueError('r_max must be the r_max of the foundation model')
        if len(getattr(foundation, 'heads', ['Default'])) != 1:
            raise ValueError('Select one head of the foundation model first')
        model_config.update(hidden_irreps=o3.Irreps(str(foundation.products[0].linear.irreps_out)),
                            num_interactions=len(foundation.interactions))
        mean, std = 0., 1.
    else:
        mean, std = mace.modules.scaling_classes[args.scaling](train_loader, atomic_energies)
    model = PotentialResponse(
        **model_config,
        interaction_cls_first=mace.modules.interaction_classes[args.interaction_first],
        atomic_inter_scale=std,
        atomic_inter_shift=mean,
        density_width=args.atomic_multipoles_smearing_width,
        field_widths=ast.literal_eval(args.field_feature_widths),
        kspace_cutoff_factor=args.kspace_cutoff_factor,
        fermi_level_offset=args.fermi_level_offset,
        **args.potential_response_config,
    ).to(args.device)
    if foundation is not None:
        adopt_foundation(model, foundation)
    initialize_potential_response(model, args, train_loader, args.device)
    condition_energy(model, train_loader, args.device)
    if args.enable_cueq:
        accelerate_backbone(model, args.device)
    if args.foundation_model and args.multiheads_finetuning:
        install_replay_head(model, foundation, args)
    return model


def adopt_foundation(model, foundation):
    """The backbone and energy readouts of the foundation, its element basis kept by a one-hot map."""
    source = foundation.atomic_numbers.tolist()
    target = model.atomic_numbers.tolist()
    if set(target)-set(source):
        raise ValueError(f'The foundation model has no elements {sorted(set(target)-set(source))}')
    mapping = torch.zeros(len(target), len(source))
    for i, z in enumerate(target):
        mapping[i, source.index(z)] = 1.
    device = model.atomic_numbers.device
    model.register_buffer('foundation_element_map', mapping.to(device))
    model.register_buffer('foundation_atomic_numbers', foundation.atomic_numbers.clone().to(device))
    model.use_reduced_cg = bool(getattr(foundation, 'use_reduced_cg', False))
    for name in ('node_embedding', 'radial_embedding', 'spherical_harmonics', 'interactions', 'products',
                 'readouts', 'scale_shift'):
        setattr(model, name, deepcopy(getattr(foundation, name)).to(device))
    model.radial_embedding.requires_grad_(False)
    logging.info(f'Foundation backbone and energy readouts adopted, {len(source)} elements')


@torch.no_grad()
def condition_energy(model, loader, device):
    """Energy zeros by element, so that the local energies start at the training energies."""
    counts, residuals = [], []
    for i, batch in enumerate(loader):
        if i*loader.batch_size >= 64:
            break
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_force=False)
        n = scatter_sum(batch.node_attrs, batch.batch, dim=0)
        counts.append(n)
        residuals.append(batch.energy-output['energy'].detach())
    a, b = torch.cat(counts), torch.cat(residuals)
    delta = torch.linalg.lstsq(a.cpu(), b[:, None].cpu(), rcond=1.e-6).solution[:, 0]
    model.atomic_energies_fn.atomic_energies.add_(delta.to(device).reshape_as(model.atomic_energies_fn.atomic_energies))
    logging.info(f'Energy zeros shifted by {delta.tolist()} eV')


class FusedConvolution(torch.nn.Module):
    """The CuEq tensor product and scatter of an interaction as one CUDA kernel, the unfused map on the CPU."""

    def __init__(self, unfused, dtype):
        super().__init__()
        import cuequivariance as cue
        import cuequivariance_torch as cuet
        descriptor = cue.descriptors.channelwise_tensor_product(unfused.irreps_in1, unfused.irreps_in2,
                                                                unfused.irreps_out)
        polynomial = descriptor.flatten_coefficient_modes().squeeze_modes().polynomial
        self.fused = cuet.SegmentedPolynomial(polynomial, math_dtype=dtype, method='uniform_1d')
        self.weight_numel = polynomial.operands[0].size
        self.unfused = unfused

    def forward(self, node_feats, edge_attrs, weights, edge_index):
        sender, receiver = edge_index
        if node_feats.device.type == 'cuda':
            return self.fused([weights, node_feats, edge_attrs], {1: sender}, {0: node_feats}, {0: receiver})[0]
        messages = self.unfused(node_feats[sender], edge_attrs, weights)
        return scatter_sum(messages, receiver, dim=0, dim_size=len(node_feats))


def accelerate_backbone(model, device):
    """Interactions and products in the CuEq ir_mul layout, with MACE's own weight transfer."""
    if model.layout == 'ir_mul':
        return
    from mace.cli.convert_e3nn_cueq import transfer_weights
    from mace.modules import EquivariantProductBasisBlock
    from mace.modules.wrapper_ops import CuEquivarianceConfig
    if torch.device(device).type == 'cuda':
        import cuequivariance_ops_torch  # noqa: F401, without it CuEq runs slower than e3nn
    dtype = next(model.parameters()).dtype
    config = CuEquivarianceConfig(enabled=True, layout='ir_mul', group='O3_e3nn', optimize_all=True,
                                  conv_fusion=False)
    reduced = bool(getattr(model, 'use_reduced_cg', False))
    source, target = torch.nn.Module(), torch.nn.Module()
    source.interactions, source.products = model.interactions, model.products
    target.interactions, target.products = torch.nn.ModuleList(), torch.nn.ModuleList()
    default = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.random.fork_rng(devices=[]):
            for interaction, product in zip(source.interactions, source.products):
                options = {name: getattr(interaction, name) for name in (
                    'node_attrs_irreps', 'node_feats_irreps', 'edge_attrs_irreps', 'edge_feats_irreps',
                    'target_irreps', 'hidden_irreps', 'radial_MLP')}
                target.interactions.append(type(interaction)(
                    **options, avg_num_neighbors=float(interaction.avg_num_neighbors),
                    edge_irreps=getattr(interaction, 'edge_irreps', None), cueq_config=config))
                contraction = product.symmetric_contractions
                correlation = len(contraction.contractions[0].weights)+1
                target.products.append(EquivariantProductBasisBlock(
                    node_feats_irreps=contraction.irreps_in, target_irreps=o3.Irreps(str(product.linear.irreps_out)),
                    correlation=correlation, use_sc=product.use_sc, num_elements=interaction.node_attrs_irreps.dim,
                    use_agnostic_product=getattr(product, 'use_agnostic_product', False), use_reduced_cg=reduced,
                    cueq_config=config))
            target.to(device=device, dtype=dtype)
            transfer_weights(source, target, o3.Irreps(str(source.products[0].linear.irreps_out)).lmax,
                             correlation, len(source.products), reduced, True)
            if torch.device(device).type == 'cuda':
                for interaction in target.interactions:
                    interaction.conv_tp = FusedConvolution(interaction.conv_tp, dtype).to(device)
                    interaction.conv_fusion = True
    finally:
        torch.set_default_dtype(default)
    for old, new in zip(list(source.interactions)+list(source.products), list(target.interactions)+list(target.products)):
        new.requires_grad_(any(p.requires_grad for p in old.parameters()))
    model.interactions, model.products = target.interactions, target.products
    model.layout = 'ir_mul'
    logging.info('CuEquivariance backbone: ir_mul layout, fused CUDA convolution')


class ReplayInteraction(torch.nn.Module):
    """A shared CuEq interaction fed with the e3nn features of the replay head."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        irreps = o3.Irreps(str(block.node_feats_irreps))
        indices = [torch.arange(sl.start, sl.stop).reshape(ir.dim, mul).T.flatten()
                   for (mul, ir), sl in zip(irreps, irreps.slices())]
        self.register_buffer('to_cueq', torch.argsort(torch.cat(indices)))

    def forward(self, node_feats, **kwargs):
        return self.block(node_feats=node_feats[..., self.to_cueq], **kwargs)


class ReplayProduct(torch.nn.Module):
    """A shared CuEq product returning e3nn features to the replay head."""

    def __init__(self, block):
        super().__init__()
        self.block = block
        irreps = o3.Irreps(str(block.linear.irreps_out))
        indices = [torch.arange(sl.start, sl.stop).reshape(ir.dim, mul).T.flatten()
                   for (mul, ir), sl in zip(irreps, irreps.slices())]
        self.register_buffer('to_e3nn', torch.cat(indices))

    def forward(self, **kwargs):
        return self.block(**kwargs)[..., self.to_e3nn]


class ReplayFeatures(GTOElectrostaticFeatures):
    """MACE-POLAR's evaluator keyword on the installed graph_longrange."""

    def precompute_geometry(self, force_pbc_evaluator=False, **kwargs):
        self.set_pbc_handling('mixed_periodic' if force_pbc_evaluator else 'auto')
        return super().precompute_geometry(**kwargs)

    def forward_dynamic(self, cache, source_feats, pbc=None):
        return super().forward_dynamic(cache=cache, source_feats=source_feats.squeeze(-2)
                                       if source_feats.ndim == 3 else source_feats)


class ReplayEnergy(GTOElectrostaticEnergy):
    def forward(self, force_pbc_evaluator=False, **kwargs):
        self.set_pbc_handling('mixed_periodic' if force_pbc_evaluator else 'auto')
        return super().forward(**kwargs)


def compatible_electrostatics(head):
    """MACE-POLAR electrostatics rebuilt for the installed graph_longrange; they hold no learned parameters."""
    if 'force_pbc_evaluator' in inspect.signature(head.electric_potential_descriptor.precompute_geometry).parameters:
        return
    common = dict(density_max_l=head.atomic_multipoles_max_l, density_smearing_width=head.atomic_multipoles_smearing_width,
                  kspace_cutoff=float(head.kspace_cutoff), pbc_handling='auto')
    parameter = next(head.parameters())
    head.electric_potential_descriptor = ReplayFeatures(
        **common, feature_max_l=head.field_feature_max_l, feature_smearing_widths=head.field_feature_widths,
        include_self_interaction=head.field_si,
        quadrupole_feature_corrections=getattr(head, 'quadrupole_feature_corrections', False),
        integral_normalization='receiver').to(parameter)
    head.coulomb_energy = ReplayEnergy(**common, include_self_interaction=head.include_electrostatic_self_interaction).to(parameter)


class Replay:
    """Foundation energies and forces of the replay frames, one batch per training step of the replay head."""

    def __init__(self, graphs, batch_size, energy_weight, forces_weight, device):
        self.loader = torch_geometric.dataloader.DataLoader(graphs, batch_size=batch_size, shuffle=True, drop_last=True)
        self.batches = iter(self.loader)
        self.loss_fn = WeightedLoss({'energy_per_atom': {'weight': energy_weight}, 'forces': {'weight': forces_weight}})
        self.device = device

    def loss(self, head):
        try:
            batch = next(self.batches)
        except StopIteration:
            self.batches = iter(self.loader)
            batch = next(self.batches)
        batch = batch.to(self.device)
        output = head(batch.to_dict(), training=True, compute_force=True)
        return self.loss_fn(ref=batch, pred=output)


def install_replay_head(model, foundation, args):
    """The MACE-POLAR foundation as a second head on the shared backbone.

    Its labels are its own energies and forces on the frames of pt_train_file, computed once before training,
    so the replay keeps the foundation where the MACE-VOLT data says nothing.
    """
    if foundation is None or not hasattr(foundation, 'field_dependent_charges_maps'):
        raise ValueError('multiheads_finetuning needs a MACE-POLAR foundation_model')
    head = deepcopy(foundation).to(args.device)
    compatible_electrostatics(head)
    keys = data.KeySpecification(info_keys={'total_charge': 'charge', 'total_spin': 'spin',
                                            'external_field': 'external_field'}, arrays_keys={})
    table = AtomicNumberTable(head.atomic_numbers.tolist())
    graphs = [torch_geometric.data.Data.from_dict(dict(data.AtomicData.from_config(
        data.config_from_atoms(atoms, key_specification=keys), z_table=table, cutoff=float(head.r_max),
        heads=head.heads).to_dict(), num_nodes=len(atoms))) for atoms in ase.io.iread(args.pt_train_file, index=':')]
    if args.num_samples_pt and len(graphs) > args.num_samples_pt:
        chosen = np.random.default_rng(args.seed).choice(len(graphs), args.num_samples_pt, replace=False)
        graphs = [graphs[i] for i in chosen]
    head.eval()
    for graph in graphs:
        batch = torch_geometric.Batch.from_data_list([graph]).to(args.device)
        output = head(batch.to_dict(), training=False, compute_force=True)
        graph.energy = output['energy'].detach().cpu().reshape(-1)[0]
        graph.forces = output['forces'].detach().cpu()
        graph.energy_weight = torch.tensor(1., dtype=graph.forces.dtype)
        graph.forces_weight = torch.tensor(1., dtype=graph.forces.dtype)
    head.train()
    for name in ('node_embedding', 'radial_embedding', 'spherical_harmonics'):
        setattr(head, name, getattr(model, name))
    if model.layout == 'ir_mul':
        head.interactions = torch.nn.ModuleList(ReplayInteraction(x) for x in model.interactions)
        head.products = torch.nn.ModuleList(ReplayProduct(x) for x in model.products)
    else:
        head.interactions, head.products = model.interactions, model.products
    model.replay_head = head.to(args.device)
    model.replay = Replay(graphs, args.batch_size, args.energy_weight, args.forces_weight, args.device)
    logging.info(f'Replay head on the shared backbone, {len(graphs)} replay frames')
