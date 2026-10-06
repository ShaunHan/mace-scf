import random
import logging
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
import os
from mace.tools.scatter import scatter_sum
from mace_scf.electrostatics.loss import vacuum_observation_weight, vacuum_reference_weights


class CheckpointHandler(NativeCheckpointHandler):
    """Separate resumable raw parameters from deployable EMA parameters."""
    def __init__(self, *args, ema=None, deployment=False, **kwargs):
        super().__init__(*args, **kwargs)
        from pathlib import Path
        self.ema, self.deployment = ema, deployment
        self.progress_path = Path(kwargs["directory"])/"resume"/(kwargs["tag"]+".pt")
        self.best_loss = float("inf")

    def _checkpoint(self,state):
        value = self.builder.create_checkpoint(state)
        value["ema"] = None if self.ema is None else self.ema.state_dict()
        value["best_loss"] = self.best_loss
        value["python_rng"] = random.getstate()
        value["numpy_rng"] = np.random.get_state()
        value["torch_rng"] = torch.random.get_rng_state()
        value["cuda_rng"] = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        value['backbone_layout'] = getattr(state.model, 'backbone_layout', 'mul_ir')
        return value

    def save(self,state,epochs,keep_last=False):
        from pathlib import Path
        directory = Path(self.io.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/self.io._get_checkpoint_filename(epochs, self.io.swa_start)
        temporary = path.with_suffix('.tmp')
        torch.save(self._checkpoint(state), temporary)
        os.replace(temporary, path)
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
            value=torch.load(self.progress_path,map_location=device,weights_only=False)
            epoch=value["epoch"]
        else:
            path=self.io._get_latest_checkpoint_path(swa=swa)
            if path is None:
                return None
            logging.info("Loading checkpoint: %s", path)
            value=torch.load(path,map_location=device,weights_only=False)
            epoch=self.io._parse_checkpoint_path(path).epochs
        layout = getattr(state.model, 'backbone_layout', 'mul_ir')
        if value.get('backbone_layout', 'mul_ir') != layout:
            raise ValueError('Checkpoint and requested backbone use different tensor-product layouts. '
                             'Keep the original enable_cueq setting to resume, or start a fresh CuEq run in a new checkpoint directory; Adam moments cannot be relabeled.')
        self.builder.load_checkpoint(state=state,checkpoint=value,strict=strict)
        self.best_loss=value.get("best_loss",float("inf"))
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
          debug_grad_log_frequency=None, wandb_watch="off"):
    """Train each stage without resetting Adam moments or the EMA trajectory."""
    best = getattr(checkpoint_handler, "best_loss", float("inf"))
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
        norms, clipped = [], 0
        for step, batch in enumerate(train_loader):
            _, metrics = take_step(model if distributed_model is None else distributed_model,
                model_eval_wrapper, loss_fn, batch, optimizer, ema, max_grad_norm, device,
                debug_log_grad_summary, debug_grad_log_frequency, step)
            norms.append(metrics["grad_norm_before_clip"])
            clipped += int(metrics["grad_clip_applied"])
        if epoch % eval_interval == 0 or epoch == end_epoch:
            if cuda_device:
                torch.cuda.synchronize(device)
            train_seconds = time.perf_counter()-epoch_start
            peak_mib = torch.cuda.max_memory_allocated(device)/2**20 if cuda_device else 0.
            validation_start = time.perf_counter()
            if hasattr(optimizer, "eval"):
                optimizer.eval()
            valid_loss, metrics = evaluate(model, model_eval_wrapper, loss_fn, ema, valid_loader, device)
            valid_err_log(valid_loss, metrics, logger, log_errors, epoch)
            logging.info("Optimizer epoch %d: lr=%s, mean gradient norm=%.5g, clipped=%d/%d", epoch,
                         [g["lr"] for g in optimizer.param_groups], float(np.mean(norms)), clipped, len(norms))
            logging.info('Epoch timing: train=%.3fs, validation=%.3fs, updates=%d, train/update=%.5fs, peak allocated=%.1f MiB, backbone=%s',
                         train_seconds, time.perf_counter()-validation_start, len(norms),
                         train_seconds/max(1,len(norms)), peak_mib,
                         'CuEq' if getattr(model,'backbone_layout','mul_ir') == 'ir_mul' else 'e3nn')
            if log_wandb:
                import wandb
                wandb.log({"epoch":epoch, **{"valid_"+k:v for k,v in metrics.items() if isinstance(v,(int,float))}})
            residual = metrics.get('scf_residual_max', 0.)
            tolerance = getattr(getattr(model_eval_wrapper, 'scf_options', None), 'scf_tolerance', float('inf'))
            finite_budget = (getattr(getattr(model, 'field_dependent_charges_map', None), 'spectral', False)
                             and model_eval_wrapper.mode in ('unroll_scf', 'shortcut_scf'))
            deployable = np.isfinite(residual) and (finite_budget or residual <= tolerance)
            if finite_budget and residual > tolerance:
                logging.info('50-step finite-budget validation residual %.5g exceeds equilibrium tolerance %.5g; '
                             'metrics describe the exported finite trajectory, not a converged root', residual, tolerance)
            improved = valid_loss < best and deployable
            if not deployable:
                logging.warning('50-step validation residual %.5g exceeds %.5g; keeping the previous deployable best checkpoint', residual, tolerance)
            if improved:
                best, best_epoch, stalled = valid_loss, epoch, 0
            else:
                stalled += eval_interval
            checkpoint_handler.best_loss = best
            if improved or save_all_checkpoints:
                checkpoint_handler.save(CheckpointState(model,optimizer,lr_scheduler),epoch,keep_last=save_all_checkpoints)
            lr_scheduler.step(metrics=valid_loss)
            # Optional, read-only probes. Removing diagnostics.py is supported.
            if epoch % max(50, eval_interval) == 0:
                try:
                    from mace_scf.utils.diagnostics import audit_training
                except ModuleNotFoundError as exc:
                    if exc.name != "mace_scf.utils.diagnostics":
                        raise
                else:
                    audit_training(model, model_eval_wrapper, loss_fn, ema,
                                   train_loader, valid_loader, device, epoch)
        if hasattr(checkpoint_handler, "save_progress"):
            checkpoint_handler.save_progress(CheckpointState(model,optimizer,lr_scheduler),epoch)
        if stalled >= patience:
            logging.info("Stage early stopping after %d epochs without improvement", stalled)
            break
    return best_epoch


def take_step(model, model_eval_wrapper, loss_fn, batch, optimizer, ema,
              max_grad_norm, device, debug_log_grad_summary=False,
              debug_grad_log_frequency=None, opt_step=0):
    """One atomic optimizer update, differentiating model parameters only."""
    start = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    batch = batch.to(device)
    output = model_eval_wrapper(model, batch.to_dict(), training=True)
    loss = loss_fn(pred=output, ref=batch)
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError("Nonfinite loss; no optimizer/EMA update performed")
    parameters = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
    torch.autograd.backward(loss, inputs=parameters)
    norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf') if max_grad_norm is None else max_grad_norm,
                                         error_if_nonfinite=True)
    metrics = {"loss": float(loss.detach()), "grad_norm_before_clip": float(norm),
               "grad_clip_applied": max_grad_norm is not None and float(norm)>max_grad_norm}
    optimizer.step()
    if ema is not None:
        ema.update()
    metrics["time"] = time.time()-start
    return loss.detach(), metrics


def evaluate(model, model_eval_wrapper, loss_fn, ema, data_loader, device):
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
            return _evaluate(model, model_eval_wrapper, loss_fn, None, data_loader, device)
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
) -> Tuple[float, Dict[str, Any]]:
    num_configs = 0
    total_loss = 0.0
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
    batch = None  # for pylint

    voltage_errors = {"workfunction": [], "vacuum_potential": [], "fourier_potential": [], "fourier_density": [], "fourier_total_potential": []}
    scf_residuals = []
    objective_sums = {}
    wf_moments = []

    start_time = time.time()
    for batch in data_loader:
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model_eval_wrapper(
            model,
            batch_dict,
            training=False,
            ema=ema,
        )

        if hasattr(model, "batch_positions"):
            del model.batch_positions
        # avoid memory leaks
        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].detach()
        
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device("cpu"))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()*batch.num_graphs
        for key, function in loss_fn.loss_fns.items():
            if loss_fn.loss_weights[key] == 0:
                continue
            if hasattr(function, 'statistics'):
                # Spatial and vacuum labels may live on different graphs.
                # Accumulate each numerator/denominator before normalization.
                for part, (numerator, denominator) in function.statistics(batch, output).items():
                    pair = objective_sums.setdefault((key,part), [0.,0.])
                    pair[0] += float(numerator)
                    pair[1] += float(denominator)
                continue
            if key in ("fourier_density", "fourier_potential", "fermi_level", "workfunction"):
                denominator = float((batch.weight*getattr(batch,key+"_weight")).sum())
            elif key == "vacuum_potential":
                denominator = float(vacuum_observation_weight(batch).sum())
            elif key == "forces":
                denominator = float(batch.forces.numel())
            else:
                denominator = float(batch.num_graphs)
            pair = objective_sums.setdefault((key,''), [0.,0.])
            pair[0] += float(function(batch,output))*denominator
            pair[1] += denominator
        num_configs += batch.num_graphs

        from mace_scf.electrostatics.loss import spectral_errors
        for key in voltage_errors:
            if key not in output:
                continue
            if key.startswith("fourier"):
                weight = (batch.fourier_potential_weight*batch.fourier_proto_potential_weight*batch.weight
                          if key == "fourier_total_potential" else getattr(batch,key+"_weight")*batch.weight)
                if key+"_dft" not in output:
                    continue
                density_reference = getattr(loss_fn.loss_fns.get('fourier_density'), 'reference', 'density')
                observed_key = ('fourier_farfield_density' if key == 'fourier_density' and density_reference == 'farfield' else key)
                error = spectral_errors(batch,output,observed_key)
            elif key == "workfunction":
                weight = batch.workfunction_weight*batch.weight
                weight = weight*(batch.pbc.reshape(-1,3).sum(-1)==2)
                difference = output[key]-batch.workfunction
                use = weight>0
                if bool((use & ~torch.isfinite(difference)).any()):
                    raise FloatingPointError('Nonfinite observed workfunction error')
                wf_moments.append(torch.stack(((difference[use]*weight[use]).sum(),
                                               (difference[use].square()*weight[use]).sum(), weight[use].sum())))
                error = difference.square()
            else:
                target, weight = vacuum_reference_weights(batch, output[key])
                error = (output[key]-target).square()
            use = weight>0
            if bool((use & ~torch.isfinite(error)).any()):
                raise FloatingPointError('Nonfinite observed '+key+' error')
            voltage_errors[key].append(torch.stack(((error[use]*weight[use]).sum(),weight[use].sum())))
        if output.get("scf_residual") is not None:
            scf_residuals.append(output["scf_residual"].max())

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

    avg_loss = sum(loss_fn.loss_weights[key]*values[0]/values[1]
                   for (key,_),values in objective_sums.items() if values[1]>0)

    aux = {
        "loss": avg_loss,
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
                        'vacuum_potential':'esp_vac', 'workfunction':'wf_abs'}[key]
                aux["rmse_"+name] = float((sums[0]/sums[1]).sqrt())
    if wf_moments:
        first, second, weight = torch.stack(wf_moments).sum(0)
        if weight>0:
            # One weighted offset over the WHOLE validation split, never a
            # per-batch correction or a shift installed in the deployed model.
            aux['wf_bias'] = float(first/weight)
            aux['rmse_wf_rel'] = float((second/weight-(first/weight).square()).clamp_min(0.).sqrt())
    aux['esp_vacuum_enabled'] = bool(getattr(loss_fn.loss_fns.get('fourier_potential'), 'vacuum_weight', 0.))
    if scf_residuals:
        aux["scf_residual_max"] = float(torch.stack(scf_residuals).max())
    return avg_loss, aux


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch,
):
    if "rmse_wf_abs" in eval_metrics or "rmse_rho" in eval_metrics:
        pieces = [f"{label}={1000*eval_metrics[key]:.4f} {unit}" for key,label,unit in
                  (('rmse_e_per_atom','RMSE_E_per_atom','meV'), ('rmse_f','RMSE_F','meV/A'),
                   ('rmse_mu_per_atom','RMSE_MU_per_atom','meA'), ('rmse_rho','RMSE_RHO','me/A^3'),
                   ('rmse_fermi_level','RMSE_EF','meV')) if key in eval_metrics]
        fmt = lambda key: f"{1000*eval_metrics[key]:.4f}" if key in eval_metrics else 'n/a'
        if 'rmse_esp' in eval_metrics:
            pieces.append(('RMSE_ESP(tot/vac)='+fmt('rmse_esp')+'/'+fmt('rmse_esp_vac')
                           if eval_metrics['esp_vacuum_enabled'] else 'RMSE_ESP='+fmt('rmse_esp'))+' mV')
        if 'rmse_wf_abs' in eval_metrics:
            pieces.append('RMSE_WF(abs/rel)='+fmt('rmse_wf_abs')+'/'+fmt('rmse_wf_rel')+' meV')
        logging.info("Epoch %d: loss=%.6g, %s", epoch, valid_loss, ', '.join(pieces))
        logging.info('50-step electronic residual maximum: %.5g', eval_metrics.get('scf_residual_max', float('nan')))
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
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.2f} mDebye"
        )
    elif log_errors == "EnergyDipoleRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu:.2f} mDebye"
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
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_DMA={error_dma:.1f} me, RMSE_MU_per_atom={error_mu:.6f} meA/atom"
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
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_DMA={error_dma:.1f} me, RMSE_Mu_per_atom={error_mu} meA, RMSE_polarizability_per_atom={error_polarizability} me A^2 / V"
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
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A, RMSE_Mu_per_atom={error_mu} meA, RMSE_ESP={error_esp:.1f} mV"
        )
