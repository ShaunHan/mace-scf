"""MACE-POLAR replay head sharing the MACE-VOLT geometric backbone.

The two heads retain their own energy references and electronic models.
Replay supplies energy/force gradients only; every target label remains on
the MACE-VOLT head. One optimizer update uses the sum of the two losses.
"""
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path

import ase.io
import numpy as np
import torch
from e3nn import o3
from mace import data
from mace.tools import AtomicNumberTable, torch_geometric
from graph_longrange.features import GTOElectrostaticFeatures
from graph_longrange.energy import GTOElectrostaticEnergy

from mace_scf.electrostatics.loss import WeightedLoss


class ReplayFeatures(GTOElectrostaticFeatures):
    """Bridge POLAR's evaluator keyword to the explicit-boundary API."""
    def precompute_geometry(self, force_pbc_evaluator=False, **kwargs):
        self.set_pbc_handling('mixed_periodic' if force_pbc_evaluator else 'auto')
        return super().precompute_geometry(**kwargs)

    def forward_dynamic(self, cache, source_feats, pbc=None):
        if source_feats.ndim == 3:
            source_feats = source_feats.squeeze(-2)
        return super().forward_dynamic(cache=cache, source_feats=source_feats)


class ReplayEnergy(GTOElectrostaticEnergy):
    def forward(self, force_pbc_evaluator=False, **kwargs):
        self.set_pbc_handling('mixed_periodic' if force_pbc_evaluator else 'auto')
        return super().forward(**kwargs)


def compatible_electrostatics(head):
    """Rebuild analytic operators for the installed graph_longrange API.

    Native POLAR uses real-space electrostatics for fully aperiodic batches.
    The newer library expresses that behavior as pbc_handling='auto'. These
    operators contain no learned parameters; widths and self terms are kept.
    """
    if 'force_pbc_evaluator' in inspect.signature(head.electric_potential_descriptor.precompute_geometry).parameters:
        return
    common = dict(density_max_l=head.atomic_multipoles_max_l,
        density_smearing_width=head.atomic_multipoles_smearing_width,
        kspace_cutoff=float(head.kspace_cutoff), pbc_handling='auto')
    parameter = next(head.parameters())
    head.electric_potential_descriptor = ReplayFeatures(**common,
        feature_max_l=head.field_feature_max_l, feature_smearing_widths=head.field_feature_widths,
        include_self_interaction=head.field_si,
        quadrupole_feature_corrections=getattr(head, 'quadrupole_feature_corrections', False),
        integral_normalization='receiver').to(parameter)
    head.coulomb_energy = ReplayEnergy(**common,
        include_self_interaction=head.include_electrostatic_self_interaction).to(parameter)


class ReplayInteraction(torch.nn.Module):
    """Feed native replay features to a shared CuEq interaction."""
    def __init__(self, block):
        super().__init__()
        self.block = block
        rep = o3.Irreps(str(block.node_feats_irreps))
        indices = [torch.arange(sl.start, sl.stop).reshape(ir.dim, mul).T.flatten()
                   for (mul, ir), sl in zip(rep, rep.slices())]
        self.register_buffer('to_cueq', torch.argsort(torch.cat(indices)))

    def forward(self, node_feats, **kwargs):
        return self.block(node_feats=node_feats[..., self.to_cueq], **kwargs)


class ReplayProduct(torch.nn.Module):
    """Return shared CuEq product features in the replay head's native basis."""
    def __init__(self, block, to_e3nn):
        super().__init__()
        self.block = block
        self.register_buffer('to_e3nn', to_e3nn.clone())

    def forward(self, **kwargs):
        return self.block(**kwargs)[..., self.to_e3nn]


def install_replay_head(model, foundation):
    if not hasattr(foundation, 'field_dependent_charges_maps'):
        raise TypeError('Multihead replay requires a MACE-POLAR foundation model')
    head = deepcopy(foundation).requires_grad_(True)
    for name in ('node_embedding', 'radial_embedding', 'spherical_harmonics'):
        setattr(head, name, getattr(model, name))
    if getattr(model, 'backbone_layout', 'mul_ir') == 'ir_mul':
        head.interactions = torch.nn.ModuleList(ReplayInteraction(x) for x in model.interactions)
        head.products = torch.nn.ModuleList(ReplayProduct(x, model.backbone_to_e3nn) for x in model.products)
    else:
        head.interactions, head.products = model.interactions, model.products
    model.replay_head = head.to(next(model.parameters()).device)
    compatible_electrostatics(model.replay_head)
    logging.info('Replay: shared geometric backbone; independent MACE-POLAR and MACE-VOLT energy/electronic heads')


def deployment_model(model):
    """Export only MACE-VOLT; resumable checkpoints retain both heads."""
    if not hasattr(model, 'replay_head'):
        return model
    head = model._modules.pop('replay_head')
    try:
        return deepcopy(model)
    finally:
        model.replay_head = head


def _fingerprint(args, keys):
    digest = hashlib.sha256(json.dumps(dict(keys=keys, dtype=args.default_dtype,
        seed=args.seed, valid_fraction=args.valid_fraction, num_samples=args.num_samples_pt,
        pseudolabel=args.pseudolabel_replay), sort_keys=True).encode())
    for path in (args.foundation_model, args.pt_train_file, args.pt_valid_file):
        if path:
            with open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
    return digest.hexdigest()


def _read_graphs(path, keys, head):
    spec = data.KeySpecification(info_keys=keys['info_keys'], arrays_keys=keys['arrays_keys'])
    table = AtomicNumberTable(head.atomic_numbers.tolist())
    # Plain Data supports reconstruction for memory-bounded batch splitting;
    # upstream AtomicData's constructor requires all tensor arguments.
    return [torch_geometric.data.Data.from_dict(dict(data.AtomicData.from_config(
                data.config_from_atoms(atoms, key_specification=spec),
                z_table=table, cutoff=float(head.r_max), heads=head.heads).to_dict(), num_nodes=len(atoms)))
            for atoms in ase.io.iread(path, index=':')]


def pseudolabel(graphs, head, device):
    """Evaluate the unchanged foundation once, before any training update."""
    compatible_electrostatics(head)
    flags = [p.requires_grad for p in head.parameters()]
    was_training = head.training
    try:
        head.eval().requires_grad_(False)
        for graph in graphs:
            batch = torch_geometric.Batch.from_data_list([graph]).to(device)
            with torch.enable_grad():
                output = head(batch.to_dict(), training=False, compute_force=True)
            graph.energy = output['energy'].detach().cpu().reshape_as(graph.energy)
            graph.forces = output['forces'].detach().cpu()
            if not bool(torch.isfinite(graph.energy).all() & torch.isfinite(graph.forces).all()):
                raise FloatingPointError('Nonfinite foundation replay labels')
            graph.energy_weight = torch.ones_like(graph.energy_weight)
            graph.forces_weight = torch.ones_like(graph.forces_weight)
    finally:
        for parameter, flag in zip(head.parameters(), flags):
            parameter.requires_grad_(flag)
        head.train(was_training)


def prepare_replay(args, model, tag):
    if not args.multiheads_finetuning:
        return None
    keys = {'info_keys': {'energy':'energy', 'total_charge':'charge', 'total_spin':'spin',
                         'external_field':'external_field'}, 'arrays_keys': {'forces':'forces'}}
    for name in keys:
        keys[name].update(args.replay_head.get(name, {}))
    fingerprint = _fingerprint(args, keys)
    cache = Path(args.results_dir)/(tag+'_replay.pt')
    saved = torch.load(cache, map_location='cpu', weights_only=False) if cache.exists() else {}
    if saved.get('fingerprint') == fingerprint:
        train, valid = saved['train'], saved['valid']
    else:
        train = _read_graphs(args.pt_train_file, keys, model.replay_head)
        if args.pt_valid_file:
            valid = _read_graphs(args.pt_valid_file, keys, model.replay_head)
        else:
            order = np.random.default_rng(args.seed).permutation(len(train))
            n = max(1, round(args.valid_fraction*len(train)))
            valid, train = [train[i] for i in order[:n]], [train[i] for i in order[n:]]
        if args.num_samples_pt and len(train) > args.num_samples_pt:
            ids = np.random.default_rng(args.seed).choice(len(train), args.num_samples_pt, replace=False)
            train = [train[i] for i in ids]
        if not train or not valid:
            raise ValueError('Replay needs nonempty training and validation sets')
        if args.pseudolabel_replay:
            logging.info('Generating and caching foundation E/F replay labels: %d train, %d valid', len(train), len(valid))
            pseudolabel(train+valid, model.replay_head, args.device)
        elif any(not bool(g.energy_weight > 0 and g.forces_weight > 0) for g in train+valid):
            raise ValueError('Replay E/F labels are missing; specify pt_head keys or pseudolabel_replay=True')
        cache.parent.mkdir(parents=True, exist_ok=True)
        pending = cache.with_suffix('.tmp')
        torch.save(dict(fingerprint=fingerprint, train=train, valid=valid), pending)
        os.replace(pending, cache)
    logging.info('Replay dataset: %d training, %d validation configurations; E/F-only losses', len(train), len(valid))
    return ReplayTraining(train, valid, args.batch_size, args.energy_weight, args.forces_weight,
                          args.weight_pt_head)


class ReplayTraining:
    def __init__(self, train, valid, batch_size, energy_weight=1., forces_weight=100., weight=1.):
        self.train_loader = torch_geometric.dataloader.DataLoader(train, batch_size=batch_size, shuffle=True)
        self.valid_loader = torch_geometric.dataloader.DataLoader(valid, batch_size=1, shuffle=False,
            generator=torch.Generator().manual_seed(0))
        self.loss_fn = WeightedLoss({'energy_per_atom':energy_weight, 'forces':forces_weight})
        self.weight = weight

    def backward(self, model, batch, parameters, device, limit):
        from copy import copy
        normalizers = {k:v.to(device) for k,v in self.loss_fn.normalizers(batch).items()}
        total = 0.
        if batch.num_graphs <= limit:
            pieces = (copy(batch),)
        else:
            graphs = batch.to_data_list()
            pieces = (torch_geometric.Batch.from_data_list(graphs[start:start+limit])
                      for start in range(0,len(graphs),limit))
        for piece in pieces:
            piece = piece.to(device)
            output = model.replay_head(piece.to_dict(), training=True, compute_force=True)
            loss = self.weight*self.loss_fn(ref=piece, pred=output, normalizers=normalizers)
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError('Nonfinite replay loss; no optimizer/EMA update performed')
            torch.autograd.backward(loss, inputs=parameters)
            total = total+loss.detach()
        return total

    def evaluate(self, model, device, ema=None, training_set=False):
        loader = self.valid_loader
        if training_set:
            loader = torch_geometric.dataloader.DataLoader(self.train_loader.dataset,
                batch_size=1, shuffle=False, generator=torch.Generator().manual_seed(0))
        flags = [p.requires_grad for p in model.parameters()]
        was_training = model.training
        energy, force, reference = [], [], []
        try:
            model.eval().requires_grad_(False)
            with ema.average_parameters() if ema is not None else nullcontext():
                for batch in loader:
                    batch = batch.to(device)
                    output = model.replay_head(batch.to_dict(), training=False, compute_force=True)
                    n = batch.ptr[1:]-batch.ptr[:-1]
                    energy.append(((output['energy']-batch.energy)/n).detach())
                    force.append((output['forces']-batch.forces).detach())
                    reference.append(batch.forces.detach())
            e, f, ref = torch.cat(energy), torch.cat(force), torch.cat(reference)
            return dict(rmse_e_per_atom=float(e.square().mean().sqrt()),
                rmse_f=float(f.square().mean().sqrt()),
                rel_rmse_f=float(100*f.square().mean().sqrt()/(ref.square().mean().sqrt()+1.e-9)))
        finally:
            for parameter, flag in zip(model.parameters(), flags):
                parameter.requires_grad_(flag)
            model.train(was_training)
