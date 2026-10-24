import random
import logging
import time
from copy import copy, deepcopy
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
import os
from mace_scf.utils.training_metrics import (
    RMSE_KEYS, training_error_sums, training_error_metrics, smooth_field_errors,
)
from mace.tools.scatter import scatter_sum
from mace_scf.electrostatics.loss import (
    vacuum_observation_weight, vacuum_reference_weights, WeightedScalarLoss,
    WeightedFermiLevel, WeightedVacuumPotential, WeightedCellVoltage,
)
from mace_scf.electrostatics.coupled import SCFNumericalError


class _TrainingGrowthError(FloatingPointError):
    """Finite outlier requiring comparison on the same batch before rollback."""


def _migrate_desmearing_checkpoint(value, state):
    """Retain learned weights/Adam/EMA; replace only the obsolete live decoder.

    The new model has already fitted its frozen map from training DFT labels.
    The legacy coefficient tensor was one optimizer/EMA entry, not four entries.
    Validate its ordering before removing that entry from the saved state.
    """
    from mace_scf.electrostatics.desmearing import FrozenSpectralDecoder
    prefix = 'field_dependent_charges_map.raw_decoder.'
    obsolete = prefix+'coefficients'
    response = getattr(state.model, 'field_dependent_charges_map', None)
    decoder = getattr(response, 'raw_decoder', None)
    if not getattr(response, 'fixed_dft_observations', False):
        return value, False
    if decoder is not None and (not isinstance(decoder, FrozenSpectralDecoder)
            or not bool((decoder.fitted | ~decoder.enabled).all())):
        raise ValueError('Fit the frozen DFT map before migrating an old checkpoint')
    expected = (torch.zeros(2, dtype=torch.bool) if decoder is None else decoder.enabled.cpu())
    previous = value['model'].get(prefix+'enabled', torch.zeros(2, dtype=torch.bool)).cpu()
    old_observer = not value['model'].get('field_dependent_charges_map._extra_state', {}).get('fixed_dft_observations', False)
    if obsolete not in value['model'] and not old_observer and torch.equal(previous, expected):
        return value, False
    parameters = dict(state.model.named_parameters())
    old_names = [key for key in value['model'] if key in parameters or key == obsolete]
    if [name for name in old_names if name != obsolete] != list(parameters):
        raise ValueError('Cannot safely identify legacy decoder ordering in checkpoint EMA')
    value = deepcopy(value)
    ema = value.get('ema')
    if obsolete in old_names and ema is not None:
        for key in ('shadow_params', 'collected_params'):
            if ema.get(key) is not None:
                if len(ema[key]) != len(old_names):
                    raise ValueError('Legacy EMA parameter count differs from model')
                ema[key].pop(old_names.index(obsolete))
    if obsolete in old_names:
        group = next(g for g in value['optimizer']['param_groups'] if g.get('name') == 'electronic_references')
        references = [name for name in old_names if name.startswith('field_dependent_charges_map.')
                      and not (name.endswith('weight') and value['model'][name].ndim >= 2)]
        if len(group['params']) != len(references):
            raise ValueError('Legacy electronic-reference optimizer ordering is ambiguous')
        removed = group['params'].pop(references.index(obsolete))
        value['optimizer']['state'].pop(removed, None)
    requested = state.optimizer.param_groups
    if len(requested) != len(value['optimizer']['param_groups']):
        raise ValueError('Optimizer groups changed beyond the frozen decoder conversion')
    for saved, current in zip(value['optimizer']['param_groups'], requested):
        if saved.get('name') != current.get('name') or len(saved['params']) != len(current['params']):
            raise ValueError('Optimizer ownership differs beyond the frozen decoder conversion')
        for index, parameter in zip(saved['params'], current['params']):
            moment = value['optimizer']['state'].get(index, {}).get('exp_avg')
            if moment is not None and moment.shape != parameter.shape:
                raise ValueError('Saved Adam moment shape differs during decoder conversion')
    value['model'] = {key: tensor for key, tensor in value['model'].items() if not key.startswith(prefix)}
    if decoder is not None:
        value['model'].update({prefix+key: tensor.detach().clone() for key, tensor in decoder.state_dict().items()})
    value['model']['field_dependent_charges_map._extra_state']['fixed_dft_observations'] = True
    value['model']['field_dependent_charges_map._extra_state']['raw_vacuum_boundary'] = True
    value.pop('training_guard', None)
    logging.info('Applied the requested frozen-DFT output mapping; retained other model, Adam and EMA values. '
                 'The changed objective/observer will receive a new validation baseline.')
    return value, True


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
        self.loss_definition = None
        self.loaded_epoch = None

    def _mark_best(self, epoch):
        path = self.progress_path.with_suffix('.best')
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_suffix('.best.tmp')
        pending.write_text(str(epoch), encoding='utf-8')
        os.replace(pending, path)

    def _best_path(self, swa=False):
        from pathlib import Path
        directory = Path(self.io.directory)
        marker = self.progress_path.with_suffix('.best')
        if marker.exists():
            epoch = int(marker.read_text(encoding='utf-8'))
            return directory/self.io._get_checkpoint_filename(epoch, self.io.swa_start)
        # One-time recovery for existing runs. Older archives recorded the
        # running best loss: its first occurrence is the improving epoch.
        # mmap reads metadata without allocating archived optimizer tensors.
        records = [self.io._parse_checkpoint_path(p) for p in self.io._list_file_paths()]
        records = sorted((r for r in records if r and r.tag == self.io.tag and r.swa == swa),
                         key=lambda r: r.epochs)
        if records:
            latest = torch.load(records[-1].path, map_location='cpu', weights_only=False, mmap=True)
            if 'best_epoch' in latest:
                best = latest['best_epoch']
                if best is None:
                    return None
                self._mark_best(best)
                return directory/self.io._get_checkpoint_filename(best, self.io.swa_start)
        best, loss = None, float('inf')
        for record in records:
            value = torch.load(record.path, map_location='cpu', weights_only=False, mmap=True)
            epoch = value.get('best_epoch', record.epochs)
            if epoch is not None and value.get('best_loss', float('inf')) < loss:
                best, loss = epoch, value['best_loss']
        if best is None:
            return None
        self._mark_best(best)
        return directory/self.io._get_checkpoint_filename(best, self.io.swa_start)

    def _checkpoint(self,state):
        value = self.builder.create_checkpoint(state)
        value["ema"] = None if self.ema is None else self.ema.state_dict()
        value["best_loss"] = self.best_loss
        value["selection_metric"] = self.selection_metric
        value["best_score"] = self.best_loss if self.selection_metric == 'loss' else self.best_score
        value["best_epoch"] = self.best_epoch
        value["loss_definition"] = self.loss_definition
        value["python_rng"] = random.getstate()
        value["numpy_rng"] = np.random.get_state()
        value["torch_rng"] = torch.random.get_rng_state()
        value["cuda_rng"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        value['backbone_layout'] = getattr(state.model, 'backbone_layout', 'mul_ir')
        value['training_guard'] = getattr(state.optimizer, '_training_guard', None)
        value['guard_definition'] = getattr(state.optimizer, '_guard_definition', None)
        return value

    def save(self,state,epochs,keep_last=False):
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

    def save_progress(self,state,epoch):
        value=self._checkpoint(state)
        value["epoch"]=epoch
        self.progress_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.progress_path.with_suffix(".tmp")
        torch.save(value,temporary)
        os.replace(temporary,self.progress_path)

    def load_latest(self,state,swa=False,device=None,strict=True):
        if not self.deployment and self.progress_path.exists():
            logging.info('Loading resumable state: %s', self.progress_path)
            value=torch.load(self.progress_path,map_location=device,weights_only=False)
            epoch=value["epoch"]
        else:
            path = self._best_path(swa) if self.deployment else self.io._get_latest_checkpoint_path(swa=swa)
            if path is None:
                return None
            logging.info("Loading checkpoint: %s", path)
            value=torch.load(path,map_location=device,weights_only=False)
            epoch=self.io._parse_checkpoint_path(path).epochs
        layout = getattr(state.model, 'backbone_layout', 'mul_ir')
        if value.get('backbone_layout', 'mul_ir') != layout:
            raise ValueError('Checkpoint and requested backbone use different tensor-product layouts. '
                             'Keep the original enable_cueq setting to resume, or start a fresh CuEq run in a new checkpoint directory; Adam moments cannot be relabeled.')
        if not self.deployment:
            mismatched = [name for name, parameter in state.model.named_parameters()
                          if name not in value['model'] or value['model'][name].shape != parameter.shape]
            if mismatched:
                raise ValueError('Checkpoint parameter shapes differ from the requested model: '
                                 + ', '.join(mismatched[:6])
                                 + '. Resume with the original architecture, or start a fresh run. '
                                   'Resizing learned receivers after optimizer construction would leave stale Adam/EMA owners.')
        value, self._desmearing_migrated = _migrate_desmearing_checkpoint(value, state)
        self.builder.load_checkpoint(state=state,checkpoint=value,strict=strict)
        if value.get('training_guard') is not None:
            state.optimizer._training_guard = value['training_guard']
            state.optimizer._guard_definition = value.get('guard_definition')
        self.best_loss=value.get("best_loss",float("inf"))
        self.selection_metric=value.get('selection_metric', 'loss')
        if self.selection_metric == 'rmse_phi_abs':  # Existing saved checkpoints.
            self.selection_metric = 'rmse_wf_abs'
        self.best_score=value.get('best_score', self.best_loss)
        self.best_epoch=value.get("best_epoch")
        self.loss_definition=value.get('loss_definition')
        self.loaded_epoch=epoch
        if 'best_epoch' not in value and np.isfinite(self.best_loss):
            best_path = self._best_path(swa)
            if best_path is not None:
                self.best_epoch = self.io._parse_checkpoint_path(best_path).epochs
        if self.ema is not None and value.get("ema") is not None:
            self.ema.load_state_dict(value["ema"])
        if self.deployment:
            if value.get("ema") is not None:
                from torch_ema import ExponentialMovingAverage
                shadow=ExponentialMovingAverage(state.model.parameters(),decay=0.)
                shadow.load_state_dict(value["ema"])
                shadow.copy_to()
        elif value.get("torch_rng") is not None:
            torch.random.set_rng_state(value["torch_rng"].cpu())
            if "python_rng" in value:
                random.setstate(value["python_rng"])
            if "numpy_rng" in value:
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
    _resume_loss_baseline(model, model_eval_wrapper, loss_fn, train_loader, valid_loader,
                          optimizer, lr_scheduler, checkpoint_handler, ema, device,
                          checkpoint_metric=checkpoint_metric)
    best = getattr(checkpoint_handler, "best_loss", float("inf"))
    best_score = best if checkpoint_metric == 'loss' else getattr(checkpoint_handler, "best_score", float("inf"))
    stalled = 0
    best_epoch = None
    coupled = getattr(getattr(model, 'field_dependent_charges_map', None), 'coupled', False)
    definition = getattr(loss_fn, 'definition', None)
    if coupled and getattr(optimizer, '_guard_definition', None) != definition:
        from mace_scf.electrostatics.loss import WeightedLoss
        saved_guard = getattr(optimizer, '_guard_definition', None)
        saved_guard = WeightedLoss.from_checkpoint(saved_guard) if saved_guard else None
        if saved_guard is None or saved_guard.definition != definition:
            optimizer._training_guard = None
        optimizer._guard_definition = deepcopy(definition)
    for epoch in range(start_epoch, end_epoch+1):
        epoch_start = time.perf_counter()
        cuda_device = torch.device(device).type == 'cuda'
        if cuda_device:
            torch.cuda.reset_peak_memory_stats(device)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if hasattr(optimizer, "train"):
            optimizer.train()
        snapshot = _training_snapshot(model, optimizer, ema, train_loader, device) if coupled else None
        for attempt in range(5):
            norms, retained, clipped, graphs_seen = [], [], 0, 0
            replay_iterator = iter(replay.train_loader) if replay is not None else None
            epoch_errors = torch.zeros((len(RMSE_KEYS), 2), dtype=torch.float64, device=device)
            try:
                for step, batch in enumerate(train_loader):
                    replay_batch = None
                    if replay is not None:
                        try:
                            replay_batch = next(replay_iterator)
                        except StopIteration:
                            replay_iterator = iter(replay.train_loader)
                            replay_batch = next(replay_iterator)
                    _, metrics = _guarded_take_step(snapshot, model if distributed_model is None else distributed_model,
                        model_eval_wrapper, loss_fn, batch, optimizer, ema, max_grad_norm, device,
                        debug_log_grad_summary, debug_grad_log_frequency, step,
                        replay=replay, replay_batch=replay_batch)
                    if "_rmse_sums" in metrics:
                        epoch_errors += metrics.pop("_rmse_sums")
                    norms.append(metrics["grad_norm_before_clip"])
                    retained.append(metrics["gradient_retained"])
                    graphs_seen += metrics["graphs"]
                    clipped += int(metrics["grad_clip_applied"])
                break
            except (FloatingPointError, SCFNumericalError) as error:
                if snapshot is None:
                    raise
                _restore_training_snapshot(snapshot, model, optimizer, ema, train_loader, device)
                if attempt == 4:
                    raise FloatingPointError(f'Epoch {epoch} remained numerically unstable after four learning-rate reductions. '
                        'Restored the complete pre-epoch state; no failed epoch was saved.') from error
                for group in optimizer.param_groups:
                    group['lr'] *= .5**(attempt+1)
                logging.warning('Numerical training failure in epoch %d: %s. Restored model, Adam, EMA and RNG; '
                    'replaying the complete epoch at lr=%s (retry %d/4). No configuration was skipped.',
                    epoch, error, [g['lr'] for g in optimizer.param_groups], attempt+1)
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
            try:
                valid_loss, metrics = evaluate(model, model_eval_wrapper, loss_fn, ema, valid_loader, device)
                score = checkpoint_score(checkpoint_metric, valid_loss, metrics)
                if coupled and (not np.isfinite(valid_loss) or
                        (np.isfinite(best) and valid_loss > 100*max(best, 1.e-12))):
                    raise FloatingPointError(f'Validation loss exploded to {valid_loss:.6g} '
                        f'from the stage best {best:.6g}; no failed checkpoint was committed.')
            except Exception:
                if snapshot is not None:
                    _restore_training_snapshot(snapshot, model, optimizer, ema, train_loader, device)
                    logging.exception('Validation failed in epoch %d. Restored the complete pre-epoch state; '
                                      'best and resumable checkpoints are unchanged.', epoch)
                elif save_all_checkpoints:
                    checkpoint_handler.save(CheckpointState(model,optimizer,lr_scheduler),epoch,keep_last=True)
                    logging.exception('Validation failed; retained epoch %d for diagnosis. The best checkpoint is unchanged.',epoch)
                raise
            valid_err_log(valid_loss, metrics, logger, log_errors, epoch)
            if replay is not None:
                replay_metrics = replay.evaluate(model, device, ema=ema)
                logging.info('Replay validation epoch %d: RMSE_E/atom=%.3f meV, RMSE_F=%.3f meV/A, relative F RMSE=%.2f%%',
                    epoch, 1000*replay_metrics['rmse_e_per_atom'], 1000*replay_metrics['rmse_f'], replay_metrics['rel_rmse_f'])
                if logger is not None:
                    logger.log({**replay_metrics, 'mode':'eval', 'split':'replay_valid', 'epoch':epoch,
                                'metric_context':'replay_checkpoint_evaluation'})
            logging.debug("Optimizer epoch %d: lr=%s, mean gradient norm=%.5g, clipped=%d/%d", epoch,
                         [g["lr"] for g in optimizer.param_groups], float(np.mean(norms)), clipped, len(norms))
            logging.debug('Epoch timing: train=%.3fs, validation=%.3fs, updates=%d, train/update=%.5fs, peak allocated=%.1f MiB, backbone=%s',
                         train_seconds, time.perf_counter()-validation_start, len(norms),
                         train_seconds/max(1,len(norms)), peak_mib,
                         'CuEq' if getattr(model,'backbone_layout','mul_ir') == 'ir_mul' else 'e3nn')
            logging.debug('Optimization budget: graphs=%d/%d, updates=%d, mean retained gradient=%.5g, min retained gradient=%.5g',
                         graphs_seen, len(train_loader.dataset), len(norms), float(np.mean(retained)), min(retained,default=1.))
            if hasattr(optimizer, '_device_batch_size'):
                logging.debug('Batching: optimizer graphs=%s, device graphs<=%d; one clipping/Adam/EMA update per optimizer batch',
                             train_loader.batch_size, optimizer._device_batch_size)
            if log_wandb:
                import wandb
                wandb.log({"epoch":epoch, **{"valid_"+k:v for k,v in metrics.items() if isinstance(v,(int,float))}})
            residual = metrics.get('scf_residual_max', 0.)
            tolerance = getattr(getattr(model_eval_wrapper, 'scf_options', None), 'scf_tolerance', float('inf'))
            finite_budget = (getattr(getattr(model, 'field_dependent_charges_map', None), 'spectral', False)
                             and model_eval_wrapper.mode in ('unroll_scf', 'shortcut_scf'))
            deployable = np.isfinite(residual) and (finite_budget or residual <= tolerance)
            if finite_budget and residual > tolerance:
                logging.debug('50-step finite-budget validation residual %.5g exceeds equilibrium tolerance %.5g; '
                             'metrics describe the exported finite trajectory, not a converged root', residual, tolerance)
            improved = score < best_score and deployable
            if not deployable:
                logging.warning('50-step validation residual %.5g exceeds %.5g; keeping the previous deployable best checkpoint', residual, tolerance)
            if improved:
                best_score, best_epoch, stalled = score, epoch, 0
                checkpoint_handler.best_epoch = epoch
            else:
                stalled += eval_interval
            if deployable:
                best = min(best, valid_loss)
            checkpoint_handler.best_loss = best
            checkpoint_handler.best_score = best_score
            logging.info('Checkpoint selection epoch %d: %s=%.8g, best=%.8g, improved=%s',
                         epoch, checkpoint_metric, score, best_score, improved)
            if improved or save_all_checkpoints:
                checkpoint_handler.save(CheckpointState(model,optimizer,lr_scheduler),epoch,keep_last=save_all_checkpoints)
            lr_scheduler.step(metrics=score)
        if rank == 0 and logger is not None:
            online = training_error_metrics(epoch_errors)
            options = getattr(model_eval_wrapper, 'training_options', None)
            scf = getattr(options, 'scf', None)
            logger.log({**online, 'mode':'train', 'split':'training', 'epoch':epoch,
                        'metric_context':'online_training', 'parameter_state':'optimizer',
                        'scf_steps':getattr(scf, 'num_scf_steps', None)})
            logging.debug('Training epoch %d, online RMSE (base units): %s', epoch, online)
        if hasattr(checkpoint_handler, "save_progress"):
            checkpoint_handler.save_progress(CheckpointState(model,optimizer,lr_scheduler),epoch)
        del snapshot
        if stalled >= patience:
            logging.info("Stage early stopping after %d epochs without improvement", stalled)
            break
    return best_epoch


def checkpoint_score(metric, loss, metrics):
    """A missing/nonfinite WF metric must never win checkpoint selection."""
    if metric not in ('loss', 'rmse_wf_abs'):
        raise ValueError('checkpoint_metric must be loss or rmse_wf_abs')
    if metric != 'loss' and metric not in metrics:
        raise ValueError('WF checkpoint selection requires paired vacuum/EF validation labels')
    value = float(loss if metric == 'loss' else metrics[metric])
    if not np.isfinite(value):
        raise FloatingPointError(f'Nonfinite checkpoint metric {metric}')
    return value


def _resume_loss_baseline(model, wrapper, loss_fn, train_loader, valid_loader,
                          optimizer, scheduler, handler, ema, device, checkpoint_metric='loss'):
    """Compare resumed checkpoints on the current objective's scale."""
    from copy import deepcopy
    from mace_scf.electrostatics.loss import WeightedLoss
    definition = getattr(loss_fn, 'definition', None)
    if definition is None:
        return
    epoch = getattr(handler, 'loaded_epoch', None)
    old_definition = getattr(handler, 'loss_definition', None)
    saved_objective = WeightedLoss.from_checkpoint(old_definition) if old_definition else None
    selection_changed = getattr(handler, 'selection_metric', 'loss') != checkpoint_metric
    changed = ((None if saved_objective is None else saved_objective.definition) != definition
               or selection_changed)
    response = getattr(model, 'field_dependent_charges_map', None)
    decoder = getattr(response, 'raw_decoder', None)
    upgrade = (decoder is not None and bool(decoder.enabled[1])
               and not getattr(response, 'raw_vacuum_boundary', False))
    changed = changed or upgrade
    migrated = getattr(handler, '_desmearing_migrated', False)
    changed = changed or migrated
    old_best = getattr(handler, 'best_loss', float('inf'))
    verify = (not migrated and epoch is not None and saved_objective is not None and np.isfinite(old_best)
              and not getattr(handler, '_resume_verified', False))
    if epoch is None or (not changed and not verify):
        if upgrade:
            response.raw_vacuum_boundary = True
        handler.loss_definition = deepcopy(definition)
        handler.selection_metric = checkpoint_metric
        return
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None
    generator = getattr(valid_loader, 'generator', None)
    loader_rng = generator.get_state() if generator is not None else None
    try:
        if hasattr(optimizer, 'eval'):
            optimizer.eval()
        if verify:
            # Compare on the SAVED objective before accepting a changed recipe.
            # A finite collapse must not replace its good historical baseline.
            value, _ = evaluate(model, wrapper, saved_objective, ema, valid_loader, device)
            if not np.isfinite(value) or value > 100*max(old_best, 1.e-12):
                raise FloatingPointError(f'Resumed epoch {epoch} has loss {value:.6g} on its saved objective '
                    f'(recorded best {old_best:.6g}). Restore a stable earlier checkpoint; '
                    'optimizer, EMA and best-checkpoint selection were not advanced.')
        if changed:
            if upgrade:
                response.raw_vacuum_boundary = True
            value, metrics = evaluate(model, wrapper, loss_fn, ema, valid_loader, device)
            score = checkpoint_score(checkpoint_metric, value, metrics)
            if not np.isfinite(value) or not np.isfinite(metrics.get('scf_residual_max', 0.)):
                raise FloatingPointError('Nonfinite resumed validation baseline; checkpoint selection unchanged')
    except Exception:
        if upgrade:
            response.raw_vacuum_boundary = False
        raise
    finally:
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.random.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, device)
        if loader_rng is not None:
            generator.set_state(loader_rng)
    handler._resume_verified = True
    if not changed:
        handler.loss_definition = deepcopy(definition)
        return
    handler.loss_definition = deepcopy(definition)
    handler.best_loss, handler.best_epoch = value, epoch
    handler.selection_metric, handler.best_score = checkpoint_metric, score
    handler.save(CheckpointState(model, optimizer, scheduler), epoch, keep_last=True)
    logging.info('Re-evaluated resumed epoch %d with the current objective and observer: loss=%.6g. '
                 'The objective, observer or selection metric changed; optimizer and EMA state are retained. '
                 'Selection: %s=%.8g.', epoch, value, checkpoint_metric, score)


def _guarded_take_step(snapshot, model, model_eval_wrapper, loss_fn, batch,
                       optimizer, ema, max_grad_norm, device, *args, **kwargs):
    """Check rare finite outliers against this batch at the start of the epoch.

    An average over other batches is a trigger for this check, not evidence of
    divergence. The probe does no Adam/EMA update and retains every label.
    """
    arguments = (model, model_eval_wrapper, loss_fn, batch, optimizer, ema,
                 max_grad_norm, device, *args)
    if snapshot is None:
        return take_step(*arguments, **kwargs)
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
           torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None)

    def restore_rng():
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.random.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state(rng[3], device)

    try:
        return take_step(*arguments, **kwargs)
    except _TrainingGrowthError as error:
        reason = str(error)
    # Leave the exception scope to release the failed forward/backward graph
    # before loading another parameter state.
    target = getattr(model, 'module', model)
    current = deepcopy(target.state_dict())
    flags = [p.requires_grad for p in model.parameters()]
    try:
        target.load_state_dict(snapshot['model'])
        restore_rng()
        _, reference = take_step(*arguments, **kwargs, _update=False)
    finally:
        target.load_state_dict(current)
        for parameter, flag in zip(model.parameters(), flags):
            parameter.requires_grad_(flag)
        optimizer.zero_grad(set_to_none=True)
        restore_rng()
    del current
    baseline = {'loss': reference['loss'], 'norm': reference['grad_norm_before_clip']}
    logging.warning('%s. Checking the same batch against its pre-epoch loss=%.6g, '
                    'gradient norm=%.6g; no optimizer/EMA update was made by the probe.',
                    reason, baseline['loss'], baseline['norm'])
    # Require growth against both scales. An almost-zero reference derivative
    # must not turn an ordinary gradient into an apparent explosion.
    history = getattr(optimizer, '_training_guard', None) or {}
    baseline = {key: max(value, history.get(key, 0.)) for key, value in baseline.items()}
    return take_step(*arguments, **kwargs, _growth_reference=baseline)


def take_step(model, model_eval_wrapper, loss_fn, batch, optimizer, ema,
              max_grad_norm, device, debug_log_grad_summary=False,
              debug_grad_log_frequency=None, opt_step=0, replay=None, replay_batch=None,
              _growth_reference=None, _update=True):
    """One atomic optimizer update, differentiating model parameters only."""
    start = time.time()
    model.train()
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    # One optimizer batch can use several device batches. Split only after an
    # actual allocation OOM, and remember the successful graph count. The
    # optimizer, clipping and EMA still update exactly once for the whole batch.
    coupled=getattr(getattr(model,'field_dependent_charges_map',None),'coupled',False)
    adaptive=(coupled and torch.device(device).type=='cuda'
              and getattr(loss_fn,'supports_graph_accumulation',False))
    graphs=None
    logical=batch.num_graphs
    capacity=getattr(optimizer,'_device_batch_size',logical) if adaptive else logical
    limit=min(logical,capacity)
    normalizers=({key:value.to(device) for key,value in loss_fn.normalizers(batch).items()}
                 if adaptive else None)

    def pieces():
        nonlocal graphs
        from mace.tools.torch_geometric import Batch
        if not adaptive or limit >= logical:
            # Data.to mutates its container: keep the original CPU batch for
            # a possible OOM retry, without unbatching/rebuilding every graph.
            yield copy(batch)
        else:
            if graphs is None:
                graphs=batch.to_data_list()
            for i in range(0,logical,limit):
                yield Batch.from_data_list(graphs[i:i+limit])

    def backward_batch():
        total=None
        metric_sums=None
        for piece in pieces():
            piece=piece.to(device)
            output=model_eval_wrapper(model,piece.to_dict(),training=True)
            loss=loss_fn(pred=output,ref=piece,**({'normalizers':normalizers} if adaptive else {}))
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError('Nonfinite loss; no optimizer/EMA update performed')
            history = (_growth_reference if _growth_reference is not None
                       else getattr(optimizer, '_training_guard', None))
            partial = loss.detach() if total is None else total+loss.detach()
            if coupled and _update and history is not None and history.get('count', 8) >= 8:
                if float(partial) > 100*max(history['loss'], 1.e-12):
                    raise _TrainingGrowthError('Finite training loss outlier before backward/Adam: '
                        f'{float(partial):.6g} versus reference {history["loss"]:.6g}')
            torch.autograd.backward(loss,inputs=parameters)
            total=loss.detach() if total is None else total+loss.detach()
            observed = training_error_sums(piece, output, loss_fn)
            metric_sums = observed if metric_sums is None else metric_sums+observed
            del output,loss,piece
        if replay is not None:
            total = total+replay.backward(model, replay_batch, parameters, device, limit)
        return total, metric_sums

    def attempt():
        # Returning out of the exception scope releases the failed autograd
        # frames before empty_cache; do not retain the OOM traceback or retry
        # illegal-access/device-assert/linear-algebra/numerical failures.
        try:
            return backward_batch()
        except torch.cuda.OutOfMemoryError:
            if not adaptive or limit==1:
                raise
            return None

    rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state(device)) if adaptive else None
    while True:
        optimizer.zero_grad(set_to_none=True)
        result=attempt()
        if result is not None:
            loss, metric_sums = result
            break
        optimizer.zero_grad(set_to_none=True)
        import gc
        gc.collect();torch.cuda.empty_cache()
        torch.random.set_rng_state(rng[0]);torch.cuda.set_rng_state(rng[1],device)
        limit=max(1,limit//2)
        capacity=limit
        logging.warning('Allocation OOM: retrying the unchanged optimizer batch with at most %d graphs per device batch; no optimizer/EMA update occurred',limit)
    if adaptive:optimizer._device_batch_size=capacity
    frequency = debug_grad_log_frequency
    if (debug_log_grad_summary and logging.getLogger().isEnabledFor(logging.DEBUG)
            and (opt_step == 0 or (frequency is not None and frequency > 0 and opt_step % frequency == 0))):
        logging.debug('Optimizer gradient ownership before clipping: %s', _gradient_diagnostics(optimizer))
    try:
        norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf') if max_grad_norm is None else max_grad_norm,
                                             error_if_nonfinite=True)
    except RuntimeError as error:
        if 'non-finite' not in str(error):
            raise
        raise FloatingPointError('Nonfinite gradient; no optimizer/EMA update performed') from error
    norm_value = float(norm)
    if coupled and _update:
        _check_training_growth(optimizer, float(loss.detach()), norm_value, _growth_reference)
    metrics = {"loss": float(loss.detach()), "grad_norm_before_clip": norm_value,
               "grad_clip_applied": max_grad_norm is not None and norm_value>max_grad_norm,
               "gradient_retained": min(1., max_grad_norm/(norm_value+1.e-6)) if max_grad_norm is not None else 1.,
               "graphs": int(batch.num_graphs),
               "device_batch_size":limit, "_rmse_sums":metric_sums}
    if _update:
        optimizer.step()
        if ema is not None:
            ema.update()
    metrics["time"] = time.time()-start
    return loss.detach(), metrics


def _check_training_growth(optimizer, loss, norm, reference=None):
    """Flag finite growth for comparison with the same pre-epoch batch.

    A running scale over different batches only triggers an additional probe.
    Confirmed same-batch growth rejects the update before Adam/EMA.
    """
    previous = getattr(optimizer, '_training_guard', None)
    baseline = reference if reference is not None else previous
    if baseline is not None and baseline.get('count', 8) >= 8:
        if loss > 100*max(baseline['loss'], 1.e-12) or norm > 100*max(baseline['norm'], 1.e-12):
            raise _TrainingGrowthError('Finite training growth before Adam/EMA: '
                f'loss={loss:.6g} (reference {baseline["loss"]:.6g}), '
                f'gradient norm={norm:.6g} (reference {baseline["norm"]:.6g})')
    if previous is None:
        optimizer._training_guard = {'loss':loss, 'norm':norm, 'count':1}
    else:
        rate = max(.01, 1./(previous['count']+1))
        optimizer._training_guard = {'loss':previous['loss']+rate*(loss-previous['loss']),
            'norm':previous['norm']+rate*(norm-previous['norm']), 'count':previous['count']+1}


def _cpu_training_copy(value):
    if torch.is_tensor(value):
        return value.detach().to(device='cpu', copy=True)
    if isinstance(value, dict):
        result=copy(value)
        result.update({key:_cpu_training_copy(item) for key,item in value.items()})
        return result
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_training_copy(item) for item in value)
    return deepcopy(value)


def _training_snapshot(model, optimizer, ema, loader, device):
    """Keep rollback parameters/Adam/EMA on CPU, not resident on the GPU."""
    generator = getattr(loader, 'generator', None)
    return {'model':_cpu_training_copy(model.state_dict()), 'optimizer':_cpu_training_copy(optimizer.state_dict()),
        'ema':None if ema is None else _cpu_training_copy(ema.state_dict()),
        'guard':deepcopy(getattr(optimizer, '_training_guard', None)),
        'rng':torch.random.get_rng_state(), 'python':random.getstate(), 'numpy':np.random.get_state(),
        'cuda':torch.cuda.get_rng_state(device) if torch.device(device).type=='cuda' else None,
        'loader':generator.get_state() if generator is not None else None}


def _restore_training_snapshot(snapshot, model, optimizer, ema, loader, device):
    model.load_state_dict(snapshot['model'])
    # Optimizer loading can alias same-device tensors. Retries must not mutate
    # the saved Adam moments that subsequent retries need to restore.
    optimizer.load_state_dict(deepcopy(snapshot['optimizer']))
    optimizer.zero_grad(set_to_none=True)
    optimizer._training_guard = deepcopy(snapshot['guard'])
    if ema is not None:
        ema.load_state_dict(snapshot['ema'])
    torch.random.set_rng_state(snapshot['rng'])
    random.setstate(snapshot['python']); np.random.set_state(snapshot['numpy'])
    if snapshot['cuda'] is not None:
        torch.cuda.set_rng_state(snapshot['cuda'], device)
    if snapshot['loader'] is not None:
        loader.generator.set_state(snapshot['loader'])


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


def _evaluation_batches(model, wrapper, data_loader, device, ema):
    """Evaluate all graphs, reducing only the device batch on allocation OOM."""
    from mace.tools.torch_geometric import Batch
    adaptive=(torch.device(device).type=='cuda' and
              getattr(getattr(model,'field_dependent_charges_map',None),'coupled',False))

    def attempt(piece):
        try:
            piece=piece.to(device)
            output=wrapper(model,piece.to_dict(),training=False,ema=ema)
            return piece, output
        except torch.cuda.OutOfMemoryError:
            if not adaptive or piece.num_graphs==1:
                raise
            if hasattr(model,'batch_positions'):
                del model.batch_positions
            return None

    capacity=getattr(wrapper,'_evaluation_device_batch_size',None)
    for batch in data_loader:
        if not adaptive:
            yield attempt(batch)
            continue
        if capacity is None:capacity=batch.num_graphs
        if capacity >= batch.num_graphs:
            result=attempt(copy(batch))
            if result is not None:
                yield result
                del result
                wrapper._evaluation_device_batch_size=capacity
                continue
            import gc
            gc.collect();torch.cuda.empty_cache()
            capacity=max(1,batch.num_graphs//2)
            logging.warning('Validation allocation OOM: reducing device batch to %d graphs; every graph is still evaluated',capacity)
        graphs=batch.to_data_list()
        index=0
        while index<len(graphs):
            count=min(capacity,len(graphs)-index)
            result=attempt(Batch.from_data_list(graphs[index:index+count]))
            if result is None:
                import gc
                gc.collect();torch.cuda.empty_cache()
                capacity=max(1,count//2)
                logging.warning('Validation allocation OOM: reducing device batch to %d graphs; every graph is still evaluated',capacity)
                continue
            yield result
            del result
            index+=count
        wrapper._evaluation_device_batch_size=capacity


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
    delta_fermi_level_list = []

    voltage_errors = {"potential_difference": [], "vacuum_potential": [], "fourier_potential": [], "fourier_density": [], "fourier_total_potential": []}
    smooth_errors = {}
    scf_residuals = []
    objective_sums = {}
    scalar_objectives = {}
    scalar_metrics = {}
    wf_moments = []
    paired_scalar_moments = []
    voltage_rows = []
    debug_stability = logging.getLogger().isEnabledFor(logging.DEBUG)
    force_outliers, residual_outliers, graph_offset = [], [], 0

    start_time = time.time()
    for batch, output in _evaluation_batches(model,model_eval_wrapper,data_loader,device,ema):

        if hasattr(model, "batch_positions"):
            del model.batch_positions
        # avoid memory leaks
        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].detach()
        
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device("cpu"))
        for function in (WeightedFermiLevel(), WeightedVacuumPotential(), WeightedCellVoltage()):
            if output.get(function.key) is not None and getattr(batch, function.key, None) is not None:
                scalar_metrics[function.key] = scalar_metrics.get(function.key, 0.)+function.moments(batch, output).double()

        # Accumulate each objective once using its full-split denominator;
        # an additional per-batch WeightedLoss evaluation would be discarded.
        for key, function in loss_fn.loss_fns.items():
            if loss_fn.loss_weights[key] == 0:
                continue
            if isinstance(function, WeightedScalarLoss):
                scalar_objectives[key] = scalar_objectives.get(key, 0.)+function.moments(batch, output).double()
                continue
            if hasattr(function, 'statistics'):
                # Spatial and vacuum labels may live on different graphs.
                # Accumulate each numerator/denominator before normalization.
                for part, (numerator, denominator) in function.statistics(batch, output).items():
                    pair = objective_sums.setdefault((key,part), [0.,0.])
                    pair[0] += float(numerator)
                    pair[1] += float(denominator)
                continue
            if key in ("fourier_density", "fourier_potential", "fermi_level"):
                denominator = float((batch.weight*getattr(batch,key+"_weight")).sum())
            elif key == "vacuum_potential":
                denominator = float(vacuum_observation_weight(batch).sum())
            elif key in ("forces", "atomic_multipoles", "esps", "field_features"):
                denominator = float(len(batch.positions))
            else:
                denominator = float(batch.num_graphs)
            pair = objective_sums.setdefault((key,''), [0.,0.])
            pair[0] += float(function(batch,output))*denominator
            pair[1] += denominator

        from mace_scf.electrostatics.loss import spectral_errors
        for key in voltage_errors:
            if key == "potential_difference":
                if any(output.get(part) is None for part in ("vacuum_potential", "fermi_level")):
                    continue
            elif output.get(key) is None:
                continue
            if key.startswith("fourier"):
                weight = (batch.fourier_potential_weight*batch.fourier_proto_potential_weight*batch.weight
                          if key == "fourier_total_potential" else getattr(batch,key+"_weight")*batch.weight)
                if key+"_dft" not in output:
                    continue
                error = spectral_errors(batch,output,key)
            elif key == "potential_difference":
                target, weight = vacuum_reference_weights(batch, output['vacuum_potential'])
                weight = weight*(batch.fermi_level_weight>0)
                difference = (output['vacuum_potential']-output['fermi_level'])-(target-batch.fermi_level)
                use = weight>0
                if bool((use & ~torch.isfinite(difference)).any()):
                    raise FloatingPointError('Nonfinite observed potential difference error')
                wf_moments.append(torch.stack(((difference[use]*weight[use]).sum(),
                                               (difference[use].square()*weight[use]).sum(), weight[use].sum())))
                ef = (output['fermi_level']-batch.fermi_level)[use]
                vac = (output['vacuum_potential']-target)[use]
                paired_scalar_moments.append(torch.stack(((ef*weight[use]).sum(),
                    (vac*weight[use]).sum(),(ef*vac*weight[use]).sum(),weight[use].sum())))
                if debug_stability and bool(use.any()):
                    from graph_longrange.utils import FIELD_CONSTANT
                    cell = batch.cell.reshape(-1, 3, 3)
                    pbc = batch.pbc.reshape(-1, 3).bool()
                    axis = (~pbc).long().argmax(-1)
                    rows = torch.arange(batch.num_graphs)
                    normal = torch.linalg.cross(cell[rows, (axis+1)%3], cell[rows, (axis+2)%3])
                    area = torch.linalg.vector_norm(normal, dim=-1)
                    normal = normal/area[:, None]
                    step_error = torch.full_like(weight, float('nan'))
                    if output.get('dipole') is not None and batch.dipole is not None:
                        observed = ((batch.dipole_weight.reshape(-1, 3)>0) | (normal.abs()<1.e-12)).all(-1)
                        observed &= pbc.sum(-1) == 2
                        delta = output['dipole'].reshape(-1, 3)-batch.dipole.reshape(-1, 3)
                        step_error[observed] = (FIELD_CONSTANT*(delta*normal).sum(-1)/area)[observed]
                    voltage_rows.extend({'index':graph_offset+i, 'weight':float(weight[i]),
                        'wf_error':float(difference[i]), 'ef_error':float(output['fermi_level'][i]-batch.fermi_level[i]),
                        'vacuum_error':float(output['vacuum_potential'][i]-target[i]),
                        'dipole_step_error':float(step_error[i])} for i in torch.where(use)[0].tolist())
                error = difference.square()
            else:
                target, weight = vacuum_reference_weights(batch, output[key])
                error = (output[key]-target).square()
            use = weight>0
            if bool((use & ~torch.isfinite(error)).any()):
                raise FloatingPointError('Nonfinite observed '+key+' error')
            voltage_errors[key].append(torch.stack(((error[use]*weight[use]).sum(),weight[use].sum())))
        for key, error, weight in smooth_field_errors(batch, output, loss_fn):
            use = weight > 0
            pair = torch.stack(((error[use]*weight[use]).sum(), weight[use].sum()))
            smooth_errors[key] = smooth_errors.get(key, 0.)+pair
        if output.get("scf_residual") is not None:
            scf_residuals.append(output["scf_residual"].max())
            if debug_stability:
                # Reuse observations already on CPU. A small state residual
                # alone does not certify a well-conditioned force derivative.
                sums = torch.full((batch.num_graphs,), float('nan'))
                if output.get('forces') is not None and batch.forces is not None:
                    sums = scatter_sum((batch.forces-output['forces']).square().sum(-1),
                                       batch.batch, dim=0, dim_size=batch.num_graphs)
                    sums = sums.masked_fill(batch.weight*batch.forces_weight<=0, float('nan'))
                counts = batch.ptr[1:]-batch.ptr[:-1]
                rows = [{'index':graph_offset+i, 'SCF_residual':float(output['scf_residual'][i]),
                         'force_SSE':float(sums[i]), 'force_RMSE_meV_A':float(1000*(sums[i]/(3*counts[i])).sqrt())}
                        for i in range(batch.num_graphs)]
                residual_outliers = sorted(residual_outliers+rows, key=lambda x:x['SCF_residual'], reverse=True)[:5]
                force_outliers = sorted(force_outliers+[x for x in rows if np.isfinite(x['force_SSE'])],
                                        key=lambda x:x['force_SSE'], reverse=True)[:5]
        graph_offset += batch.num_graphs

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
        if output.get("fermi_level") is not None and batch.fermi_level is not None:
            use = batch.fermi_level_weight > 0
            if bool(use.any()):
                delta_fermi_level_list.append((batch.fermi_level-output["fermi_level"])[use])
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
    polars_computed = len(delta_polarizability_list) > 0

    weighted_terms = {key+('/'+part if part else ''):loss_fn.loss_weights[key]*values[0]/values[1]
                      for (key,part),values in objective_sums.items() if values[1]>0}
    weighted_terms.update({key:loss_fn.loss_weights[key]*float(loss_fn.loss_fns[key].from_moments(values))
                           for key,values in scalar_objectives.items() if values[0]>0})
    avg_loss = sum(weighted_terms.values())

    aux = {
        "loss": avg_loss,
        "weighted_loss_terms": weighted_terms,
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
    if delta_fermi_level_list:
        delta_fermi_level = to_numpy(torch.cat(delta_fermi_level_list, dim=0))
        aux["mae_fermi_level"] = compute_mae(delta_fermi_level)
        aux["rmse_fermi_level"] = compute_rmse(delta_fermi_level)
        aux["q95_fermi_level"] = compute_q95(delta_fermi_level)

    aux["time"] = time.time() - start_time

    for key, values in voltage_errors.items():
        if values:
            sums = torch.stack(values).sum(0)
            if sums[1]>0:
                name = {'fourier_density':'rho', 'fourier_potential':'esp',
                        'fourier_total_potential':'esp_with_proto',
                        'vacuum_potential':'esp_vac', 'potential_difference':'wf_abs'}[key]
                aux["rmse_"+name] = float((sums[0]/sums[1]).sqrt())
    if wf_moments:
        first, second, weight = torch.stack(wf_moments).sum(0)
        if weight>0:
            # One weighted offset over the WHOLE validation split, never a
            # per-batch correction or a shift installed in the deployed model.
            aux['wf_bias'] = float(first/weight)
            aux['rmse_wf_rel'] = float((second/weight-(first/weight).square()).clamp_min(0.).sqrt())
    for key, (square, weight) in smooth_errors.items():
        if weight > 0:
            aux[key] = float((square/weight).sqrt())
    for key, moments in scalar_metrics.items():
        weight, first, second = moments
        if weight > 0:
            name = {'fermi_level': 'fermi_level', 'vacuum_potential': 'esp_vac'}.get(key, key)
            aux['rmse_'+name] = float((second/weight).sqrt())
            aux['rmse_'+name+'_rel'] = float((second/weight-(first/weight).square()).clamp_min(0.).sqrt())
            aux[name+'_bias'] = float(first/weight)
    aux['esp_vacuum_enabled'] = bool(loss_fn.loss_weights.get('vacuum_potential', 0.))
    if scf_residuals:
        aux["scf_residual_max"] = float(torch.stack(scf_residuals).max())
        if debug_stability:
            logging.debug('%s stability outliers (zero-based evaluation order; force SSE in eV^2/A^2): '
                          'by force contribution=%s; by SCF residual=%s', split, force_outliers, residual_outliers)
    if 'rmse_wf_rel' in aux:
        ef, vac, product, weight = torch.stack(paired_scalar_moments).sum(0)
        logging.debug('Validation voltage: centered (EF/e)/vacuum covariance=%.6g V^2, '
                      'WF bias=%.6g eV, maximum SCF residual=%.6g',
                      float(product/weight-ef*vac/weight.square()),
                      aux.get('wf_bias',0.),aux.get('scf_residual_max',0.))
    if voltage_rows:
        _voltage_diagnostics(voltage_rows, split)
    return avg_loss, aux


def _voltage_diagnostics(rows, split):
    """Reuse evaluation residuals; no extra model call or fitted correction."""
    weight = np.array([row['weight'] for row in rows])
    error = np.array([row['wf_error'] for row in rows])
    error -= np.average(error, weights=weight)
    contribution = weight*error**2
    largest = np.argsort(-contribution)[:5]
    logging.debug('%s largest relative WF contributions (zero-based evaluation indices): %s', split,
        [{'index':rows[i]['index'], 'WF_error_meV':float(1000*error[i]),
          'relative_WF_SSE_fraction':float(contribution[i]/max(contribution.sum(),1.e-30)),
          'EF_error_meV':1000*rows[i]['ef_error'],
          'vacuum_error_mV':1000*rows[i]['vacuum_error']} for i in largest])
    dipole = np.array([row['dipole_step_error'] for row in rows])
    use = np.isfinite(dipole)
    if np.count_nonzero(use)>1:
        w = weight[use]
        x, y = dipole[use], error[use]
        x, y = x-np.average(x,weights=w), y-np.average(y,weights=w)
        xx, yy, xy = (np.average(value,weights=w) for value in (x*x,y*y,x*y))
        logging.debug('%s polarization diagnostic: centered dipole-step RMSE=%.6g mV, '
                      'correlation with WF residual=%.6g; 4*pi*kappa*delta(D_normal)/area, '
                      'observed slab components only, no prediction correction',
                      split,1000*np.sqrt(xx),xy/max(np.sqrt(xx*yy),1.e-30))


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch,
):
    if log_errors == "ElectrostaticRMSE" or "rmse_wf_abs" in eval_metrics or "rmse_rho" in eval_metrics:
        pieces = [f"{label}={1000*eval_metrics[key]:.4f} {unit}" for key,label,unit in
                  (('rmse_e_per_atom','RMSE_E_per_atom','meV'), ('rmse_f','RMSE_F','meV/A'),
                   ('rmse_mu_per_atom','RMSE_dip_per_atom','meA')) if key in eval_metrics]
        fmt = lambda key: f"{1000*eval_metrics[key]:.4f}" if key in eval_metrics else 'n/a'
        if 'rmse_rho' in eval_metrics:
            label = 'RMSE_rho(raw/smeared/farfield)' if 'rmse_rho_smooth' in eval_metrics else 'RMSE_rho'
            value = fmt('rmse_rho')
            if 'rmse_rho_smooth' in eval_metrics:
                value += '/'+fmt('rmse_rho_smooth')+'/'+fmt('rmse_rho_farfield_smooth')
            pieces.append(label+'='+value+' me/A^3')
        if 'rmse_fermi_level' in eval_metrics:
            pieces.append('RMSE_EF(abs/centered)='+fmt('rmse_fermi_level')+'/'+fmt('rmse_fermi_level_rel')+' meV')
        if 'rmse_esp' in eval_metrics:
            if 'rmse_esp_smooth' in eval_metrics:
                pieces.append('RMSE_ESP(raw/smeared)='+fmt('rmse_esp')+'/'+fmt('rmse_esp_smooth')+' mV')
            else:
                pieces.append('RMSE_ESP='+fmt('rmse_esp')+' mV')
        if 'rmse_esp_vac' in eval_metrics:
            pieces.append('RMSE_ESPvac(abs/centered)='+fmt('rmse_esp_vac')+'/'+fmt('rmse_esp_vac_rel')+' mV')
        if 'rmse_wf_abs' in eval_metrics:
            pieces.append('RMSE_WF(abs/centered)='+fmt('rmse_wf_abs')+'/'+fmt('rmse_wf_rel')+' meV')
        if 'rmse_cell_voltage' in eval_metrics:
            pieces.append('RMSE_V='+fmt('rmse_cell_voltage')+' mV')
        logging.info("Epoch %d: loss=%.6g, %s", epoch, valid_loss, ', '.join(pieces))
        logging.debug('50-step electronic residual maximum: %.5g', eval_metrics.get('scf_residual_max', float('nan')))
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
