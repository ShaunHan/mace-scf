import logging
import os
import random
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from mace.tools.checkpoint import CheckpointHandler as NativeCheckpointHandler, CheckpointState
from mace.tools.torch_tools import tensor_dict_to_device, to_numpy
from mace.tools.utils import (
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)
from mace.tools.scatter import scatter_sum
from mace_scf.electrostatics.coupled import SCFNumericalError
from mace_scf.electrostatics.loss import (
    WeightedFermiLevel, WeightedVacuumPotential, WeightedWorkFunction, spectral_errors,
)
from mace_scf.utils.training_metrics import FIELD_ERRORS, RMSE_KEYS, training_error_metrics, training_error_sums


class CheckpointHandler(NativeCheckpointHandler):
    """Separate resumable raw parameters from deployable EMA parameters."""
    def __init__(self, *args, ema=None, deployment=False, **kwargs):
        super().__init__(*args, **kwargs)
        from pathlib import Path
        self.ema, self.deployment = ema, deployment
        self.progress_path = Path(kwargs["directory"])/"resume"/(kwargs["tag"]+".pt")
        self.best_loss = float("inf")
        self.selection_metric = 'loss'
        self.best_score = float("inf")
        self.best_epoch = None

    def _mark_best(self, epoch):
        path = self.progress_path.with_suffix('.best')
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix('.best.tmp')
        pending.write_text(str(epoch), encoding='utf-8')
        os.replace(pending, path)

    def _best_path(self, swa=False):
        from pathlib import Path
        marker = self.progress_path.with_suffix('.best')
        if not marker.exists():
            return None
        epoch = int(marker.read_text(encoding='utf-8'))
        return Path(self.io.directory)/self.io._get_checkpoint_filename(epoch, self.io.swa_start)

    def _checkpoint(self, state):
        value = self.builder.create_checkpoint(state)
        value["ema"] = None if self.ema is None else self.ema.state_dict()
        value["best_loss"] = self.best_loss
        value["selection_metric"] = self.selection_metric
        value["best_score"] = self.best_loss if self.selection_metric == 'loss' else self.best_score
        value["best_epoch"] = self.best_epoch
        value["python_rng"] = random.getstate()
        value["numpy_rng"] = np.random.get_state()
        value["torch_rng"] = torch.random.get_rng_state()
        value["cuda_rng"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        value['backbone_layout'] = getattr(state.model, 'backbone_layout', 'mul_ir')
        return value

    def save(self, state, epochs, keep_last=False):
        from pathlib import Path
        directory = Path(self.io.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/self.io._get_checkpoint_filename(epochs, self.io.swa_start)
        temporary = path.with_suffix('.tmp')
        torch.save(self._checkpoint(state), temporary)
        os.replace(temporary, path)
        if self.best_epoch == epochs:
            # Retaining every validation checkpoint must not make export pick
            # the last (possibly worse) epoch instead of the validated best.
            self._mark_best(epochs)
        # Commit the new checkpoint before removing its predecessor.
        if not self.io.keep and self.io.old_path and not keep_last:
            previous = Path(self.io.old_path)
            if previous.resolve().parent != directory.resolve():
                raise ValueError('Previous checkpoint is outside the checkpoint directory')
            if previous != path and previous.exists():
                previous.unlink()
        self.io.old_path = str(path)

    def save_progress(self, state, epoch):
        value = self._checkpoint(state)
        value["epoch"] = epoch
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.progress_path.with_suffix(".tmp")
        torch.save(value, temporary)
        os.replace(temporary, self.progress_path)

    def load_latest(self, state, swa=False, device=None, strict=True):
        if not self.deployment and self.progress_path.exists():
            logging.info('Loading resumable state: %s', self.progress_path)
            value = torch.load(self.progress_path, map_location=device, weights_only=False)
            epoch = value["epoch"]
        else:
            path = self._best_path(swa) if self.deployment else self.io._get_latest_checkpoint_path(swa=swa)
            if path is None:
                return None
            logging.info("Loading checkpoint: %s", path)
            value = torch.load(path, map_location=device, weights_only=False)
            epoch = self.io._parse_checkpoint_path(path).epochs
        if value.get('backbone_layout', 'mul_ir') != getattr(state.model, 'backbone_layout', 'mul_ir'):
            raise ValueError('Checkpoint and requested backbone use different tensor-product layouts. '
                             'Keep the original enable_cueq setting to resume.')
        self.builder.load_checkpoint(state=state, checkpoint=value, strict=strict)
        self.best_loss = value.get("best_loss", float("inf"))
        self.selection_metric = value.get('selection_metric', 'loss')
        self.best_score = value.get('best_score', self.best_loss)
        self.best_epoch = value.get("best_epoch")
        if self.ema is not None and value.get("ema") is not None:
            self.ema.load_state_dict(value["ema"])
        if self.deployment:
            if value.get("ema") is not None:
                shadow = ExponentialMovingAverage(state.model.parameters(), decay=0.)
                shadow.load_state_dict(value["ema"])
                shadow.copy_to()
        elif value.get("torch_rng") is not None:
            torch.random.set_rng_state(value["torch_rng"].cpu())
            random.setstate(value["python_rng"])
            np.random.set_state(value["numpy_rng"])
            if torch.cuda.is_available() and value.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([x.cpu() for x in value["cuda_rng"]])
        return epoch


def train(model, model_eval_wrapper, loss_fn, train_loader, valid_loader,
          optimizer, lr_scheduler, start_epoch, end_epoch, patience,
          checkpoint_handler, logger, eval_interval, device, log_errors,
          rank=0, save_all_checkpoints=False, train_sampler=None, ema=None,
          distributed_model=None, max_grad_norm=10., log_wandb=False,
          test_loaders=None, debug_log_grad_summary=False,
          debug_grad_log_frequency=None, wandb_watch="off", replay=None,
          checkpoint_metric='loss'):
    """Train each stage without resetting Adam moments or the EMA trajectory."""
    if checkpoint_metric not in ('loss', 'rmse_wf_abs'):
        raise ValueError('checkpoint_metric must be loss or rmse_wf_abs')
    best_score = (float("inf") if checkpoint_handler.selection_metric != checkpoint_metric else
                  checkpoint_handler.best_loss if checkpoint_metric == 'loss' else checkpoint_handler.best_score)
    checkpoint_handler.selection_metric = checkpoint_metric
    stalled = 0
    best_epoch = None
    for epoch in range(start_epoch, end_epoch+1):
        epoch_start = time.perf_counter()
        cuda_device = torch.device(device).type == 'cuda'
        if cuda_device:
            torch.cuda.reset_peak_memory_stats(device)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(optimizer, "train"):
            optimizer.train()
        norms, clipped, skipped = [], 0, 0
        replay_iterator = iter(replay.train_loader) if replay is not None else None
        epoch_errors = torch.zeros((len(RMSE_KEYS), 2), dtype=torch.float64, device=device)
        for step, batch in enumerate(train_loader):
            replay_batch = None
            if replay is not None:
                replay_batch = next(replay_iterator, None)
                if replay_batch is None:
                    replay_iterator = iter(replay.train_loader)
                    replay_batch = next(replay_iterator)
            _, metrics = take_step(model if distributed_model is None else distributed_model,
                                   model_eval_wrapper, loss_fn, batch, optimizer, ema, max_grad_norm, device,
                                   debug_log_grad_summary, debug_grad_log_frequency, step,
                                   replay=replay, replay_batch=replay_batch)
            if metrics["skipped"]:
                skipped += 1
                continue
            epoch_errors += metrics["_rmse_sums"]
            norms.append(metrics["grad_norm_before_clip"])
            clipped += int(metrics["grad_clip_applied"])
        if skipped:
            logging.warning('Epoch %d: skipped %d of %d updates with a nonfinite gradient or SCF state',
                            epoch, skipped, skipped+len(norms))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(epoch_errors)
        if epoch % eval_interval == 0 or epoch == end_epoch:
            if cuda_device:
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter()-epoch_start
            peak_mib = torch.cuda.max_memory_allocated(device)/2**20 if cuda_device else 0.
            validation_start = time.perf_counter()
            if hasattr(optimizer, "eval"):
                optimizer.eval()
            valid_loss, metrics = evaluate(model, model_eval_wrapper, loss_fn, ema, valid_loader, device)
            score = float(valid_loss if checkpoint_metric == 'loss' else metrics[checkpoint_metric])
            valid_err_log(valid_loss, metrics, logger, log_errors, epoch)
            if replay is not None:
                replay_metrics = replay.evaluate(model, device, ema=ema)
                logging.info('Replay validation epoch %d: RMSE_E/atom=%.3f meV, RMSE_F=%.3f meV/A, relative F RMSE=%.2f%%',
                    epoch, 1000*replay_metrics['rmse_e_per_atom'], 1000*replay_metrics['rmse_f'], replay_metrics['rel_rmse_f'])
                if logger is not None:
                    logger.log({**replay_metrics, 'mode':'eval', 'split':'replay_valid', 'epoch':epoch,
                                'metric_context':'replay_checkpoint_evaluation'})
            logging.debug("Optimizer epoch %d: lr=%s, mean gradient norm=%.5g, clipped=%d/%d", epoch,
                          [g["lr"] for g in optimizer.param_groups], float(np.mean(norms)) if norms else 0.,
                          clipped, len(norms))
            logging.debug('Epoch timing: train=%.3fs, validation=%.3fs, updates=%d, train/update=%.5fs, '
                          'peak allocated=%.1f MiB, backbone=%s', train_seconds, time.perf_counter()-validation_start,
                          len(norms), train_seconds/max(1, len(norms)), peak_mib,
                          'CuEq' if getattr(model, 'backbone_layout', 'mul_ir') == 'ir_mul' else 'e3nn')
            if log_wandb:
                import wandb
                wandb.log({"epoch":epoch, **{"valid_"+k:v for k,v in metrics.items() if isinstance(v,(int,float))}})
            improved = score < best_score
            if improved:
                best_score, best_epoch, stalled = score, epoch, 0
                checkpoint_handler.best_epoch = epoch
            else:
                stalled += eval_interval
            checkpoint_handler.best_loss = min(checkpoint_handler.best_loss, valid_loss)
            checkpoint_handler.best_score = best_score
            logging.info('Checkpoint selection epoch %d: %s=%.8g, best=%.8g, improved=%s',
                         epoch, checkpoint_metric, score, best_score, improved)
            if improved or save_all_checkpoints:
                checkpoint_handler.save(CheckpointState(model, optimizer, lr_scheduler), epoch,
                                        keep_last=save_all_checkpoints)
            lr_scheduler.step(metrics=score)
        if rank == 0 and logger is not None:
            online = training_error_metrics(epoch_errors)
            scf = getattr(getattr(model_eval_wrapper, 'training_options', None), 'scf', None)
            logger.log({**online, 'mode':'train', 'split':'training', 'epoch':epoch,
                        'metric_context':'online_training', 'parameter_state':'optimizer',
                        'scf_steps':getattr(scf, 'num_scf_steps', None)})
            logging.debug('Training epoch %d, online RMSE (base units): %s', epoch, online)
        checkpoint_handler.save_progress(CheckpointState(model, optimizer, lr_scheduler), epoch)
        if stalled >= patience:
            logging.info("Stage early stopping after %d epochs without improvement", stalled)
            break
    return best_epoch


def take_step(model, model_eval_wrapper, loss_fn, batch, optimizer, ema,
              max_grad_norm, device, debug_log_grad_summary=False,
              debug_grad_log_frequency=None, opt_step=0, replay=None, replay_batch=None):
    """One optimizer update of the model parameters; a nonfinite gradient skips it."""
    start = time.time()
    model.train()
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)
    batch = batch.to(device)
    try:
        output = model_eval_wrapper(model, batch.to_dict(), training=True)
        loss = loss_fn(pred=output, ref=batch)
        torch.autograd.backward(loss, inputs=parameters)
    except SCFNumericalError as error:
        logging.warning('Skipped an update: %s', error)
        optimizer.zero_grad(set_to_none=True)
        return None, {"skipped": True}
    metrics = {"_rmse_sums": training_error_sums(batch, output)}
    loss = loss.detach()
    del output
    if replay is not None:
        loss = loss+replay.backward(model, replay_batch, parameters, device)
    frequency = debug_grad_log_frequency
    if (debug_log_grad_summary and logging.getLogger().isEnabledFor(logging.DEBUG)
            and (opt_step == 0 or (frequency is not None and frequency > 0 and opt_step % frequency == 0))):
        logging.debug('Optimizer gradient ownership before clipping: %s', _gradient_diagnostics(optimizer))
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, float('inf') if max_grad_norm is None else max_grad_norm))
    if not np.isfinite(norm) or not bool(torch.isfinite(loss)):
        optimizer.zero_grad(set_to_none=True)
        return loss, {"skipped": True}
    optimizer.step()
    if ema is not None:
        ema.update()
    metrics.update({"skipped": False, "loss": float(loss), "grad_norm_before_clip": norm,
                    "grad_clip_applied": max_grad_norm is not None and norm > max_grad_norm,
                    "graphs": int(batch.num_graphs), "time": time.time()-start})
    return loss, metrics


def _gradient_diagnostics(optimizer):
    """Measure existing gradients by owner, without another backward pass."""
    report = []
    for group in optimizer.param_groups:
        parameters = [p for p in group['params'] if p.requires_grad]
        present = [p for p in parameters if p.grad is not None]
        count = sum(p.numel() for p in present)
        squares = [torch.linalg.vector_norm(p.grad.detach()).square() for p in present]
        squared_norm = float(torch.stack(squares).sum()) if squares else 0.
        report.append({'group':group.get('name', str(len(report))), 'lr':float(group['lr']),
            'trainable_values':sum(p.numel() for p in parameters), 'gradient_values':count,
            'norm':squared_norm**.5, 'rms':(squared_norm/max(count,1))**.5})
    total = sum(row['norm']**2 for row in report)
    for row in report:
        row['squared_norm_fraction'] = row['norm']**2/max(total,1.e-30)
    return report


def evaluate(model, model_eval_wrapper, loss_fn, ema, data_loader, device, split='validation'):
    """Restore trainability and mode even when a validation batch fails."""
    flags = [p.requires_grad for p in model.parameters()]
    was_training = model.training
    try:
        model.eval()
        model.requires_grad_(False)
        for parameter in model.parameters():
            parameter.grad = None
        # One EMA swap per split, not two full-model copies per minibatch.
        with ema.average_parameters() if ema is not None else nullcontext():
            return _evaluate(model, model_eval_wrapper, loss_fn, None, data_loader, device, split=split)
    finally:
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        model.train(was_training)


def _evaluate(
    model: torch.nn.Module,
    model_eval_wrapper,
    loss_fn: torch.nn.Module,
    ema: Optional[ExponentialMovingAverage],
    data_loader: DataLoader,
    device: torch.device,
    split: str = 'validation',
) -> Tuple[float, Dict[str, Any]]:
    E_computed = False
    delta_es_list = []
    delta_es_per_atom_list = []
    delta_fs_list = []
    Fs_computed = False
    fs_list = []
    stress_computed = False
    delta_stress_list = []
    delta_stress_per_atom_list = []
    virials_computed = False
    delta_virials_list = []
    delta_virials_per_atom_list = []
    Mus_computed = False
    delta_mus_list = []
    delta_mus_per_atom_list = []
    mus_list = []
    dmas_computed = False
    delta_dmas_list = []
    dmas_list = []
    delta_esps_list = []
    esps_list = []
    polarizability_computed = False
    delta_polarizability_list = []
    delta_polarizability_per_atom_list = []
    total_charge_computed = False
    delta_total_charge_list = []

    # Pooled numerators and denominators: one exact mean over the split per term.
    objective = {name: [0., 0.] for name, weight in loss_fn.loss_weights.items() if weight}
    scalars, fields, residuals, wf_rows = {}, {}, [], []
    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        output = model_eval_wrapper(model, batch.to_dict(), training=False, ema=ema)
        if hasattr(model, "batch_positions"):
            del model.batch_positions
        # avoid memory leaks
        output = tensor_dict_to_device({key: value.detach() if isinstance(value, torch.Tensor) else value
                                        for key, value in output.items()}, device=torch.device("cpu"))
        batch = batch.cpu()
        denominators = loss_fn.denominators(batch)
        for name, pair in objective.items():
            pair[0] += float(loss_fn.loss_fns[name](batch, output))*float(denominators[name])
            pair[1] += float(denominators[name])
        for function in (WeightedFermiLevel(), WeightedVacuumPotential(), WeightedWorkFunction()):
            if output.get(function.key) is not None and getattr(batch, 'vacuum_potential', None) is not None:
                error, weight = function.errors(batch, output)
                scalars[function.key] = scalars.get(function.key, 0.)+torch.stack(
                    (weight.sum(), (weight*error).sum(), (weight*error.square()).sum())).double()
                if function.key == 'work_function':
                    wf_rows.append(_dipole_step_rows(batch, output, error, weight))
        for key, field in FIELD_ERRORS:
            if field+'_dft' in output:
                label = 'fourier_potential' if field.endswith('potential') else 'fourier_density'
                weight = batch.weight*getattr(batch, label+'_weight')
                fields[key] = fields.get(key, 0.)+torch.stack(
                    ((spectral_errors(batch, output, field)*weight).sum(), weight.sum())).double()
        if output.get("scf_residual") is not None:
            residuals.append(output["scf_residual"].max())

        if output.get("energy") is not None and batch.energy is not None:
            observed = batch.weight*batch.energy_weight > 0
            if bool(observed.any()):
                E_computed = True
                delta_es_list.append((batch.energy - output["energy"])[observed])
                delta_es_per_atom_list.append(
                    ((batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1]))[observed]
                )
        if output.get("forces") is not None and batch.forces is not None:
            observed = (batch.weight*batch.forces_weight)[batch.batch] > 0
            if bool(observed.any()):
                Fs_computed = True
                delta_fs_list.append((batch.forces - output["forces"])[observed])
                fs_list.append(batch.forces[observed])
        if output.get("stress") is not None and batch.stress is not None:
            stress_computed = True
            delta_stress_list.append(batch.stress - output["stress"])
            delta_stress_per_atom_list.append(
                (batch.stress - output["stress"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("virials") is not None and batch.virials is not None:
            virials_computed = True
            delta_virials_list.append(batch.virials - output["virials"])
            delta_virials_per_atom_list.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("density_coefficients") is not None and batch.total_charge is not None:
            total_charge_computed = True
            total_charge = scatter_sum(
                src=output["density_coefficients"][:,0], index=batch.batch, dim=-1
            )
            delta_total_charge_list.append(batch.total_charge - total_charge)
        if output.get("dipole") is not None and batch.dipole is not None:
            dipole_components_to_include = batch.dipole_weight.view(-1, 3) > 0.0
            if torch.any(dipole_components_to_include):
                dipole_differences = (batch.dipole - output["dipole"])
                num_atoms = (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1)
                num_atoms = num_atoms.repeat(1, 3)

                delta_mus_list.append(dipole_differences[dipole_components_to_include])
                delta_mus_per_atom_list.append(
                    dipole_differences[dipole_components_to_include] / num_atoms[dipole_components_to_include]
                )
                mus_list.append(batch.dipole[dipole_components_to_include]) # mus list is len(observations) not len(structures)
        if (
            output.get("density_coefficients") is not None
            and batch.density_coefficients is not None
        ):
            observed = batch.density_coefficients_weight[batch.batch]>0
            if bool(observed.any()):
                dmas_computed = True
                delta_dmas_list.append((batch.density_coefficients-output["density_coefficients"])[observed])
                dmas_list.append(batch.density_coefficients[observed])

        if (
            output.get("electrostatic_potentials") is not None
            and batch.electrostatic_potentials is not None
        ):
            esps_computed = True
            delta_esps_list.append(
                batch.electrostatic_potentials - output["electrostatic_potentials"]
            )
            esps_list.append(batch.electrostatic_potentials)
        else:
            esps_computed= False
        if output.get("polarizability") is not None and batch.polarizability is not None:
            polars_to_include = batch.polarizability_weight > 0.0
            if torch.any(polars_to_include):
                polarizability_computed = True
                delta_polarizability_list.append(batch.polarizability[polars_to_include] - output["polarizability"][polars_to_include])
                delta_polarizability_per_atom_list.append(
                    (batch.polarizability - output["polarizability"])[polars_to_include]
                    / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)[polars_to_include]
                )

    Mus_computed = len(delta_mus_list) > 0

    avg_loss = sum(loss_fn.loss_weights[name]*total/count for name, (total, count) in objective.items() if count > 0)
    aux = {
        "loss": avg_loss,
        "weighted_loss_terms": {name: loss_fn.loss_weights[name]*total/count
                                for name, (total, count) in objective.items() if count > 0},
    }
    if E_computed:
        delta_es = to_numpy(torch.cat(delta_es_list, dim=0))
        delta_es_per_atom = to_numpy(torch.cat(delta_es_per_atom_list, dim=0))
        aux["mae_e"] = compute_mae(delta_es)
        aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
        aux["rmse_e"] = compute_rmse(delta_es)
        aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
        aux["q95_e"] = compute_q95(delta_es)
        offset = np.mean(delta_es_per_atom)
        reduced = delta_es_per_atom - offset
        aux["offset_e_per_atom"] = offset
        aux["rmse_spread_e_per_atom"] = compute_rmse(reduced)
        aux["mae_spread_e_per_atom"] = compute_mae(reduced)
    if Fs_computed:
        delta_fs = to_numpy(torch.cat(delta_fs_list, dim=0))
        fs = to_numpy(torch.cat(fs_list, dim=0))
        aux["mae_f"] = compute_mae(delta_fs)
        aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
        aux["rmse_f"] = compute_rmse(delta_fs)
        aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
        aux["q95_f"] = compute_q95(delta_fs)
    if stress_computed:
        delta_stress = to_numpy(torch.cat(delta_stress_list, dim=0))
        delta_stress_per_atom = to_numpy(torch.cat(delta_stress_per_atom_list, dim=0))
        aux["mae_stress"] = compute_mae(delta_stress)
        aux["rmse_stress"] = compute_rmse(delta_stress)
        aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
        aux["q95_stress"] = compute_q95(delta_stress)
    if virials_computed:
        delta_virials = to_numpy(torch.cat(delta_virials_list, dim=0))
        delta_virials_per_atom = to_numpy(torch.cat(delta_virials_per_atom_list, dim=0))
        aux["mae_virials"] = compute_mae(delta_virials)
        aux["rmse_virials"] = compute_rmse(delta_virials)
        aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
        aux["q95_virials"] = compute_q95(delta_virials)
    if Mus_computed:
        delta_mus = to_numpy(torch.cat(delta_mus_list, dim=0))
        delta_mus_per_atom = to_numpy(torch.cat(delta_mus_per_atom_list, dim=0))
        mus = to_numpy(torch.cat(mus_list, dim=0))
        aux["mae_mu"] = compute_mae(delta_mus)
        aux["mae_mu_per_atom"] = compute_mae(delta_mus_per_atom)
        aux["rel_mae_mu"] = compute_rel_mae(delta_mus, mus)
        aux["rmse_mu"] = compute_rmse(delta_mus)
        aux["rmse_mu_per_atom"] = compute_rmse(delta_mus_per_atom)
        aux["rel_rmse_mu"] = compute_rel_rmse(delta_mus, mus)
        aux["q95_mu"] = compute_q95(delta_mus)
    if dmas_computed:
        delta_dmas = to_numpy(torch.cat(delta_dmas_list, dim=0))
        dmas = to_numpy(torch.cat(dmas_list, dim=0))
        aux["mae_dma"] = compute_mae(delta_dmas)
        aux["rel_mae_dma"] = compute_rel_mae(delta_dmas, dmas)
        aux["rmse_dma"] = compute_rmse(delta_dmas)
        aux["rel_rmse_dma"] = compute_rel_rmse(delta_dmas, dmas)
        aux["q95_dma"] = compute_q95(delta_dmas)
        if delta_dmas.shape[0] > 0:
            aux['rmse_charges'] = compute_rmse(delta_dmas[:,0:1])
        if delta_dmas.shape[1] > 1:
            aux['rmse_local_dipoles'] = compute_rmse(delta_dmas[:,1:4])
    if esps_computed:
        delta_esps = to_numpy(torch.cat(delta_esps_list, dim=0))
        esps = to_numpy(torch.cat(esps_list, dim=0))
        aux["mae_esp"] = compute_mae(delta_esps)
        aux["rel_mae_esp"] = compute_rel_mae(delta_esps, esps)
        aux["rmse_esp"] = compute_rmse(delta_esps)
        aux["rel_rmse_esp"] = compute_rel_rmse(delta_esps, esps)
        aux["q95_esp"] = compute_q95(delta_esps)
    if polarizability_computed:
        delta_polarizability = to_numpy(torch.cat(delta_polarizability_list, dim=0))
        delta_polarizability_per_atom = to_numpy(torch.cat(delta_polarizability_per_atom_list, dim=0))
        aux["mae_polarizability"] = compute_mae(delta_polarizability)
        aux["rmse_polarizability"] = compute_rmse(delta_polarizability)
        aux["rmse_polarizability_per_atom"] = compute_rmse(delta_polarizability_per_atom)
        aux["q95_polarizability"] = compute_q95(delta_polarizability)
    if total_charge_computed:
        delta_total_charge = to_numpy(torch.cat(delta_total_charge_list, dim=0))
        aux["mae_total_charge"] = compute_mae(delta_total_charge)
        aux["rmse_total_charge"] = compute_rmse(delta_total_charge)
        aux["q95_total_charge"] = compute_q95(delta_total_charge)

    aux["time"] = time.time() - start_time
    for key, (square, weight) in fields.items():
        if weight > 0:
            aux[key] = float((square/weight).sqrt())
    for key, (weight, first, second) in scalars.items():
        if weight > 0:
            name = {'vacuum_potential': 'esp_vac', 'work_function': 'wf_abs'}.get(key, key)
            aux['rmse_'+name] = float((second/weight).sqrt())
            aux[name+'_bias'] = float(first/weight)
    if residuals:
        aux["scf_residual_max"] = float(torch.stack(residuals).max())
    rows = [row for chunk in wf_rows for row in chunk]
    if rows:
        _voltage_diagnostics(rows, split)
    return avg_loss, aux


def _dipole_step_rows(batch, output, error, weight):
    """Per-slab WF error and the potential step of its normal-dipole error, 4 pi K d_n/A."""
    from graph_longrange.utils import FIELD_CONSTANT
    cell = batch.cell.reshape(-1, 3, 3)
    pbc = batch.pbc.reshape(-1, 3).bool()
    axis = (~pbc).long().argmax(-1)
    rows = torch.arange(batch.num_graphs)
    normal = torch.linalg.cross(cell[rows, (axis+1) % 3], cell[rows, (axis+2) % 3])
    area = torch.linalg.vector_norm(normal, dim=-1)
    step = torch.full_like(error, float('nan'))
    if output.get('dipole') is not None and batch.dipole is not None:
        observed = (batch.dipole_weight.reshape(-1, 3) > 0).any(-1) & (pbc.sum(-1) == 2)
        delta = output['dipole'].reshape(-1, 3)-batch.dipole.reshape(-1, 3)
        step = torch.where(observed, FIELD_CONSTANT*(delta*normal/area[:, None]).sum(-1)/area, step)
    return [(float(weight[i]), float(error[i]), float(step[i])) for i in torch.nonzero(weight > 0).flatten().tolist()]


def _voltage_diagnostics(rows, split):
    """Correlation of the centered WF error with the dipole-step error; no correction is applied."""
    weight, error, step = (np.array(column) for column in zip(*rows))
    use = np.isfinite(step)
    if np.count_nonzero(use) > 1:
        w, x, y = weight[use], step[use], error[use]
        x, y = x-np.average(x, weights=w), y-np.average(y, weights=w)
        xx, yy, xy = (np.average(value, weights=w) for value in (x*x, y*y, x*y))
        logging.debug('%s polarization diagnostic: centered dipole-step RMSE=%.6g mV, correlation with WF '
                      'residual=%.6g', split, 1000*np.sqrt(xx), xy/max(np.sqrt(xx*yy), 1.e-30))


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch,
):
    if log_errors == "ElectrostaticRMSE" or "rmse_wf_abs" in eval_metrics or "rmse_rho" in eval_metrics:
        fields = ((('rmse_e_per_atom',), 'RMSE_E_per_atom', 'meV'), (('rmse_f',), 'RMSE_F', 'meV/A'),
                  (('rmse_mu_per_atom',), 'RMSE_dip_per_atom', 'meA'),
                  (('rmse_rho', 'rmse_rho_smooth', 'rmse_rho_farfield_smooth'), 'RMSE_rho(raw/smeared/farfield)',
                   'me/A^3'), (('rmse_fermi_level',), 'RMSE_EF', 'meV'),
                  (('rmse_esp', 'rmse_esp_smooth'), 'RMSE_ESP(raw/smeared)', 'mV'),
                  (('rmse_esp_vac',), 'RMSE_ESPvac', 'mV'), (('rmse_wf_abs',), 'RMSE_WF', 'meV'))
        pieces = [f"{label}={'/'.join(f'{1000*eval_metrics[key]:.4f}' for key in keys)} {unit}"
                  for keys, label, unit in fields if all(key in eval_metrics for key in keys)]
        logging.info("Epoch %d: loss=%.6g, %s", epoch, valid_loss, ', '.join(pieces))
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)

    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_stress_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_stress = eval_metrics["rmse_stress_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_stress_per_atom={error_stress:.1f} meV / A^3"
        )
    elif (
        log_errors == "PerAtomRMSEstressvirials"
        and eval_metrics["rmse_virials_per_atom"] is not None
    ):
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_virials = eval_metrics["rmse_virials_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_virials_per_atom={error_virials:.1f} meV"
        )
    elif log_errors == "TotalRMSE":
        error_e = eval_metrics["rmse_e"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "PerAtomMAE":
        error_e = eval_metrics["mae_e_per_atom"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "TotalMAE":
        error_e = eval_metrics["mae_e"] * 1e3
        error_f = eval_metrics["mae_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_dip_per_atom={error_mu:.2f} meA"
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_dip_per_atom={error_mu:.2f} meA"
        )
    elif log_errors == "DensityCoefficientsRMSE":
        error_dma = eval_metrics["rmse_dma"] * 1e3
        rel_error_dma = eval_metrics["rel_rmse_dma"]
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_DMA={error_dma:.1f} me, rel_RMSE_DMA={rel_error_dma:.2f} %"
        )
    elif log_errors == "DensityEnergyRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_dma = eval_metrics["rmse_dma"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_DMA={error_dma:.1f} me"
        )
    elif log_errors == "DensityDipoleRMSE":
        error_dma = eval_metrics["rmse_dma"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_DMA={error_dma:.1f} me, RMSE_dip_per_atom={error_mu:.6f} meA"
        )
    elif log_errors == "EnergyDensityDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_dma = eval_metrics["rmse_dma"] * 1e3
        if not "rmse_mu_per_atom" in eval_metrics:
            error_mu = "NO DIPOLES FOUND VALID SET WHEN LOGGING"
        else:
            error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
            error_mu = f"{error_mu:.2f}"
        if not "rmse_polarizability_per_atom" in eval_metrics:
            error_polarizability = "no polarizability found"
        else:
            error_polarizability = eval_metrics["rmse_polarizability_per_atom"] * 1e3
            error_polarizability = f"{error_polarizability:.2f}"
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_DMA={error_dma:.1f} me, RMSE_dip_per_atom={error_mu} meA, RMSE_polarizability_per_atom={error_polarizability} me A^2 / V"
        )
    elif log_errors == "EnergyDipolePotentialsRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        if not "rmse_mu_per_atom" in eval_metrics:
            error_mu = "NO DIPOLES FOUND VALID SET WHEN LOGGING"
        else:
            error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
            error_mu = f"{error_mu:.2f}"
        error_esp = eval_metrics["rmse_esp"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_dip_per_atom={error_mu} meA, RMSE_ESP={error_esp:.1f} mV"
        )
