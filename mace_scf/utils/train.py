import dataclasses
import gc
import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from mace.tools import torch_geometric
from mace.tools.checkpoint import CheckpointHandler, CheckpointState
from mace.tools.torch_tools import tensor_dict_to_device, to_numpy
from mace.tools.utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)
import os
from ase.data import chemical_symbols
from mace.tools.scatter import scatter_sum
from mace_scf.electrostatics.potential_response import REPORTED_SMEARING_WIDTHS, spectral_errors


def _should_log_grad_summary(opt_step: int, frequency: Optional[int]) -> bool:
    return frequency is not None and frequency > 0 and opt_step % frequency == 0


def _scalar_param_grad_summary(model: torch.nn.Module) -> Dict[str, Any]:
    param_norm_sq = 0.0
    grad_norm_sq = 0.0
    grad_max_abs = 0.0
    nonfinite_grad_count = 0
    none_grad_param_count = 0

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "batch_positions":
                continue
            param_detached = param.detach()
            param_norm_sq += float(torch.sum(param_detached * param_detached).item())

            grad = param.grad
            if grad is None:
                none_grad_param_count += 1
                continue

            grad_detached = grad.detach()
            finite_mask = torch.isfinite(grad_detached)
            nonfinite_grad_count += int((~finite_mask).sum().item())
            if torch.any(finite_mask):
                finite_grad = grad_detached[finite_mask]
                grad_norm_sq += float(torch.sum(finite_grad * finite_grad).item())
                grad_max_abs = max(
                    grad_max_abs,
                    float(torch.max(torch.abs(finite_grad)).item()),
                )

    param_norm = param_norm_sq**0.5
    grad_norm = grad_norm_sq**0.5
    grad_param_norm_ratio = grad_norm / param_norm if param_norm > 0.0 else np.nan
    return {
        "grad/global_norm": grad_norm,
        "grad/global_max_abs": grad_max_abs,
        "grad/nonfinite_count": nonfinite_grad_count,
        "grad/none_param_count": none_grad_param_count,
        "param/global_norm": param_norm,
        "grad/param_norm_ratio": grad_param_norm_ratio,
    }


def _log_wandb_parameter_histograms(model: torch.nn.Module) -> None:
    import wandb

    histograms = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name == "batch_positions":
                continue
            histograms[f"parameters/{name}"] = wandb.Histogram(
                param.detach().cpu().flatten()
            )
    if histograms:
        wandb.log(histograms)


def train(
    model: torch.nn.Module,
    model_eval_wrapper,
    loss_fn: torch.nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
    start_epoch: int,
    end_epoch: int,
    patience: int,
    checkpoint_handler: CheckpointHandler,
    logger: MetricsLogger,
    eval_interval: int,
    device: torch.device,
    log_errors: str,
    rank: int,
    save_all_checkpoints: bool = False,
    train_sampler: Optional[DistributedSampler] = None,
    ema: Optional[ExponentialMovingAverage] = None,
    distributed_model: Optional[DistributedDataParallel] = None,
    max_grad_norm: Optional[float] = 10.0,
    log_wandb: bool = False,
    test_loaders: Optional[dict] = None,
    debug_log_grad_summary: bool = False,
    debug_grad_log_frequency: Optional[int] = None,
    wandb_watch: str = "off",
):
    lowest_loss = np.inf
    valid_loss = np.inf
    patience_counter = 0
    keep_last = False

    if max_grad_norm is not None:
        logging.info(f"Using gradient clipping with tolerance={max_grad_norm:.3f}")

    model_to_train = model if distributed_model is None else distributed_model
    epoch = start_epoch
    opt_step = 0
    while epoch <= end_epoch:
        if epoch > start_epoch:
            lr_scheduler.step(metrics=valid_loss)

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.train()

        for batch in train_loader:
            _, opt_metrics = take_step(
                model=model_to_train,
                model_eval_wrapper=model_eval_wrapper,
                loss_fn=loss_fn,
                batch=batch,
                optimizer=optimizer,
                ema=ema,
                max_grad_norm=max_grad_norm,
                device=device,
                debug_log_grad_summary=debug_log_grad_summary,
                debug_grad_log_frequency=debug_grad_log_frequency,
                opt_step=opt_step,
            )
            if rank == 0:
                opt_metrics["mode"] = "opt"
                opt_metrics["epoch"] = epoch
                opt_metrics["opt_step"] = opt_step
                logger.log(opt_metrics)
                if log_wandb and debug_log_grad_summary and _should_log_grad_summary(
                    opt_step, debug_grad_log_frequency
                ):
                    import wandb

                    wandb.log(opt_metrics)
                if (
                    log_wandb
                    and wandb_watch in ("parameters", "all")
                    and _should_log_grad_summary(opt_step, debug_grad_log_frequency)
                ):
                    # W&B's native parameter watcher is a top-level forward hook.
                    # FixedPoint training calls model submethods directly, so log
                    # parameter histograms explicitly instead of relying on forward().
                    _log_wandb_parameter_histograms(model_to_train)
            opt_step += 1
        if train_sampler is not None:
            torch.distributed.barrier()

        # Validate
        if "ScheduleFree" in type(optimizer).__name__:
            optimizer.eval()
        if epoch % eval_interval == 0:
            valid_loss, eval_metrics = evaluate(
                model=model,
                model_eval_wrapper=model_eval_wrapper,
                loss_fn=loss_fn,
                ema=ema,
                data_loader=valid_loader,
                device=device,
            )
            eval_metrics["mode"] = "eval"
            eval_metrics["epoch"] = epoch
            logger.log(eval_metrics)
            if test_loaders is not None:
                for name, loader in test_loaders.items():
                    _, test_eval_metrics = evaluate(
                        model=model,
                        model_eval_wrapper=model_eval_wrapper,
                        loss_fn=loss_fn,
                        ema=ema,
                        data_loader=loader,
                        device=device,
                    )
                    test_eval_metrics["epoch"] = epoch
                    test_eval_metrics["mode"] = "eval_test"
                    test_eval_metrics["test_name"] = name
                    logger.log(test_eval_metrics)
            
            if rank == 0:
                valid_err_log(
                    valid_loss,
                    eval_metrics,
                    logger,
                    log_errors,
                    epoch,
                )

                if log_wandb:
                    import wandb
                    wandb_log_dict = {
                        "epoch": epoch,
                        "valid_loss": valid_loss,
                        "valid_rmse_e_per_atom": eval_metrics["rmse_e_per_atom"],
                        "valid_rmse_f": eval_metrics["rmse_f"],
                        "valid_mae_f": eval_metrics["mae_f"],
                    }
                    if "rmse_dma" in eval_metrics:
                        wandb_log_dict["valid_rmse_dma"] = eval_metrics["rmse_dma"]
                        wandb_log_dict["valid_mae_dma"] = eval_metrics["mae_dma"]
                    if "rmse_mu_per_atom" in eval_metrics:
                        wandb_log_dict["valid_rmse_mu_per_atom"] = eval_metrics["rmse_mu_per_atom"]
                        wandb_log_dict["valid_mae_mu_per_atom"] = eval_metrics["mae_mu_per_atom"]
                    if "mae_total_charge" in eval_metrics:
                        wandb_log_dict["valid_mae_total_charge"] = eval_metrics["mae_total_charge"]
                        wandb_log_dict["valid_rmse_total_charge"] = eval_metrics["rmse_total_charge"]
                    if "mae_fermi_level" in eval_metrics:
                        wandb_log_dict["valid_mae_fermi_level"] = eval_metrics["mae_fermi_level"]
                        wandb_log_dict["valid_rmse_fermi_level"] = eval_metrics["rmse_fermi_level"]
                    
                    wandb.log(wandb_log_dict)
                if valid_loss >= lowest_loss:
                    if save_all_checkpoints:
                        if ema is not None:
                            with ema.average_parameters():
                                checkpoint_handler.save(
                                    state=CheckpointState(model, optimizer, lr_scheduler),
                                    epochs=epoch,
                                    keep_last=True,
                                )
                        else:
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=True,
                            )
                    patience_counter += 1
                    if patience_counter >= patience:
                        logging.info(
                            f"Stopping optimization after {patience_counter} epochs without improvement"
                        )
                        break
                else:
                    lowest_loss = valid_loss
                    patience_counter = 0
                    if ema is not None:
                        with ema.average_parameters():
                            checkpoint_handler.save(
                                state=CheckpointState(model, optimizer, lr_scheduler),
                                epochs=epoch,
                                keep_last=keep_last,
                            )
                            keep_last = False or save_all_checkpoints
                    else:
                        checkpoint_handler.save(
                            state=CheckpointState(model, optimizer, lr_scheduler),
                            epochs=epoch,
                            keep_last=keep_last,
                        )
                        keep_last = False or save_all_checkpoints
        if train_sampler is not None:
            torch.distributed.barrier()
        epoch += 1

    logging.info("Training complete")


def get_attribute(obj, attr_name):
    """ parse a string to access an attribute of an object """
    parts = attr_name.split('.')
    try:
        for part in parts:
            if '[' in part and ']' in part:
                key, index = part.split('[')
                index = int(index[:-1])
                obj = getattr(obj, key)[index]
            else:
                obj = getattr(obj, part)
        if not( type(obj) == torch.nn.Parameter):
            raise AttributeError(f"model.{attr_name} is not a parameter")
    except AttributeError as e:
        raise ValueError(f"gradient debugging: weight {attr_name} was not found") from e
    return obj


def take_step(
    model: torch.nn.Module,
    model_eval_wrapper,
    loss_fn: torch.nn.Module,
    batch: torch_geometric.batch.Batch,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    max_grad_norm: Optional[float],
    device: torch.device,
    debug_log_grad_summary: bool = False,
    debug_grad_log_frequency: Optional[int] = None,
    opt_step: int = 0,
) -> Tuple[float, Dict[str, Any]]:
    start_time = time.time()
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    batch_dict = batch.to_dict()

    # do not set ema when training
    for attempt in range(2):
        try:
            output = model_eval_wrapper(
                model,
                batch_dict,
                training=True,
            )
            loss = loss_fn(pred=output, ref=batch)
            loss.backward()
            break
        except RuntimeError as error:
            # The cuEq kernels of the backbone take their work space from the GPU itself, next to PyTorch's pool. A
            # step that finds no memory is repeated once with the blocks that PyTorch keeps cached given back.
            if attempt or "out of memory" not in str(error).lower():
                raise
            logging.warning(f"Out of GPU memory in training step {opt_step}; repeating it with PyTorch's cache released")
        output = loss = None
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    del output
    loss_dict = {
        "loss": to_numpy(loss),
        "time": time.time() - start_time,
    }

    if debug_log_grad_summary and _should_log_grad_summary(opt_step, debug_grad_log_frequency):
        loss_dict.update(_scalar_param_grad_summary(model))

    if "DEBUG_IMPLICIT_GRADIENTS" in os.environ:
        the_weight = get_attribute(model, os.environ["DEBUG_IMPLICIT_GRADIENTS"])

        the_loss = loss.clone().detach()
        the_gradient = the_weight.grad.clone().detach()

        # re-evaluate with different values
        delta = 1e-3
        initial_value = the_weight.clone().detach()
        initial_value[0] += delta
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)

        output = model_eval_wrapper(
            model,
            batch_dict,
            training=True,
        )
        loss_ = loss_fn(pred=output, ref=batch)

        # compute deltas and reset weight
        fd_gradient = (loss_ - the_loss)/delta
        diff_gradient = the_gradient[0]
        error = fd_gradient.detach() - diff_gradient
        frac_error = error / fd_gradient.detach()

        initial_value[0] -= delta
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)
        del output
        del loss_
        del diff_gradient
        del error
        del frac_error

        # re-evaluate with different values
        initial_value = the_weight.clone().detach()
        initial_value[0] += delta*2
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)

        output = model_eval_wrapper(
            model,
            batch_dict,
            training=True,
        )
        loss_ = loss_fn(pred=output, ref=batch)

        # compute deltas and reset weight
        fd_gradient1 = 0.5 * (loss_ - the_loss) / delta
        diff_gradient = the_gradient[0]
        error = fd_gradient1.detach() - diff_gradient
        frac_error = error / fd_gradient.detach()
        estimate_of_noise = fd_gradient1.detach() - fd_gradient.detach()
        logging.info(f"true g(2)={fd_gradient1.item():1.4g}+-{abs(estimate_of_noise):1.3g}, diff_gradient={diff_gradient.item():1.10g}, actual error={error.item()}, fractional={frac_error.item()}")

        initial_value[0] -= delta*2
        the_weight.requires_grad_(False)
        the_weight.copy_(initial_value)
        the_weight.requires_grad_(True)
        del output
        del loss_
        del fd_gradient
        del diff_gradient
        del error
        del frac_error
        del the_gradient
    
    if max_grad_norm is not None:
        grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        )
        grad_norm_before_clip_value = float(grad_norm_before_clip.detach().cpu().item())
        loss_dict["grad_norm_before_clip"] = grad_norm_before_clip_value
        loss_dict["grad_clip_applied"] = grad_norm_before_clip_value > max_grad_norm
    
    if hasattr(model, "batch_positions"):
        del model.batch_positions

    optimizer.step()
    if ema is not None:
        ema.update()
    loss_dict["time"] = time.time() - start_time
    return loss, loss_dict


def evaluate(
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
    esps_computed = False
    delta_esps_list = []
    esps_list = []
    polarizability_computed = False
    delta_polarizability_list = []
    delta_polarizability_per_atom_list = []
    total_charge_computed = False
    delta_total_charge_list = []
    fermi_level_computed = False
    delta_fermi_level_list = []
    delta_work_function_list = []
    delta_vacuum_potential_list = []
    delta_band_gap_list = []
    dos_error_list = []
    # For the debug log: the frames behind every error, the band gap labels, and the states inside the gaps.
    numbers = getattr(model, "atomic_numbers", None)
    frames = {"E": [], "EF": [], "WF": [], "Eg": [], "states": []}
    band_gap_labels, gap_states = [], []
    fourier_errors = {}
    batch = None  # for pylint

    for name, param in model.named_parameters():
        if name == 'batch_positions':
            continue
        param.requires_grad_(False)
        param.grad = None

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
        for name, param in model.named_parameters():
            param.requires_grad_(False)
            param.grad = None

        # avoid memory leaks
        for key in output:
            if isinstance(output[key], torch.Tensor):
                output[key] = output[key].detach()
        
        batch = batch.cpu()
        output = tensor_dict_to_device(output, device=torch.device("cpu"))

        loss = loss_fn(pred=output, ref=batch)
        total_loss += to_numpy(loss).item()
        num_configs += batch.num_graphs
        names = frame_names(batch, numbers)

        if output.get("energy") is not None and batch.energy is not None:
            E_computed = True
            delta_es_list.append(batch.energy - output["energy"])
            delta_es_per_atom_list.append(
                (batch.energy - output["energy"]) / (batch.ptr[1:] - batch.ptr[:-1])
            )
            frames["E"] += names
        if output.get("forces") is not None and batch.forces is not None:
            Fs_computed = True
            delta_fs_list.append(batch.forces - output["forces"])
            fs_list.append(batch.forces)
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
            fermi_level_computed = True
            observed = batch.fermi_level_weight > 0.0
            delta_fermi_level_list.append((batch.fermi_level - output["fermi_level"])[observed])
            frames["EF"] += [name for name, keep in zip(names, observed.tolist()) if keep]
        if output.get("vacuum_potential") is not None:
            observed = batch.vacuum_potential_weight > 0.0
            delta_vacuum_potential_list.append((batch.vacuum_potential - output["vacuum_potential"])[observed])
            # the work function is the vacuum potential minus the Fermi level
            observed = observed & (batch.fermi_level_weight > 0.0)
            delta_work_function_list.append((batch.vacuum_potential - batch.fermi_level
                                             - output["vacuum_potential"] + output["fermi_level"])[observed])
            frames["WF"] += [name for name, keep in zip(names, observed.tolist()) if keep]
        if output.get("band_gap") is not None:
            observed = batch.band_gap_weight > 0.0
            delta_band_gap_list.append((batch.band_gap - output["band_gap"])[observed])
            frames["Eg"] += [name for name, keep in zip(names, observed.tolist()) if keep]
            band_gap_labels.append(batch.band_gap[observed])
        if output.get("atomic_electrons_dft") is not None:
            observed = torch.isfinite(output["atomic_electrons_dft"]) & (batch.dos_weight > 0.0)[batch.batch].unsqueeze(-1)
            dos_error_list.append((output["atomic_electrons"] - output["atomic_electrons_dft"])[observed])
            chosen, held, held_dft = states_in_gaps(batch, output)
            if bool(chosen.any()):
                gap_states.append((batch.node_attrs.argmax(-1)[chosen], held[chosen], held_dft[chosen]))
                in_cell = scatter_sum(torch.where(chosen, held, 0.0), batch.batch, dim=0, dim_size=batch.num_graphs)
                listed = scatter_sum(chosen.to(held), batch.batch, dim=0, dim_size=batch.num_graphs) > 0
                frames["states"] += [(float(value), name) for name, value, keep
                                     in zip(names, in_cell.tolist(), listed.tolist()) if keep]
        for key, widths in REPORTED_SMEARING_WIDTHS.items():
            if output.get(key + "_dft") is not None:
                observed = getattr(batch, key + "_weight") > 0.0
                for width in widths:
                    fourier_errors.setdefault(f"rmse_{key}_{width:g}A", []).append(
                        spectral_errors(batch, output, key, width)[observed])
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
            # only the atoms of frames with multipole labels
            observed = (batch.density_coefficients_weight > 0.0)[batch.batch]
            if torch.any(observed):
                dmas_computed = True
                delta_dmas_list.append(
                    (batch.density_coefficients - output["density_coefficients"])[observed]
                )
                dmas_list.append(batch.density_coefficients[observed])

        if (
            output.get("electrostatic_potentials") is not None
            and batch.electrostatic_potentials is not None
        ):
            observed = (batch.electrostatic_potentials_weight > 0.0)[batch.batch]
            if torch.any(observed):
                esps_computed = True
                delta_esps_list.append(
                    (batch.electrostatic_potentials - output["electrostatic_potentials"])[observed]
                )
                esps_list.append(batch.electrostatic_potentials[observed])
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

    avg_loss = total_loss / len(data_loader)

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
    if E_computed:
        aux["largest_energy"] = largest_errors(delta_es_per_atom, frames["E"])
    if fermi_level_computed:
        delta_fermi_level = to_numpy(torch.cat(delta_fermi_level_list, dim=0))
        aux["largest_fermi_level"] = largest_errors(delta_fermi_level, frames["EF"])
        aux["mae_fermi_level"] = compute_mae(delta_fermi_level)
        aux["rmse_fermi_level"] = compute_rmse(delta_fermi_level)
        aux["q95_fermi_level"] = compute_q95(delta_fermi_level)
    if delta_work_function_list:
        delta_work_function = to_numpy(torch.cat(delta_work_function_list, dim=0))
        aux["mae_work_function"] = compute_mae(delta_work_function)
        aux["rmse_work_function"] = compute_rmse(delta_work_function)
        aux["largest_work_function"] = largest_errors(delta_work_function, frames["WF"])
    delta_vacuum_potential = to_numpy(torch.cat(delta_vacuum_potential_list)) if delta_vacuum_potential_list else []
    if len(delta_vacuum_potential):
        aux["rmse_vacuum_potential"] = compute_rmse(delta_vacuum_potential)
    if delta_band_gap_list:
        delta_band_gap = to_numpy(torch.cat(delta_band_gap_list, dim=0))
        aux["mae_band_gap"] = compute_mae(delta_band_gap)
        aux["rmse_band_gap"] = compute_rmse(delta_band_gap)
        aux["largest_band_gap"] = largest_errors(delta_band_gap, frames["Eg"])
        aux["band_gap_reading"] = band_gap_reading(delta_band_gap, to_numpy(torch.cat(band_gap_labels, dim=0)))
    if gap_states:
        elements, held, held_dft = (to_numpy(torch.cat(parts)) for parts in zip(*gap_states))
        by_element = ", ".join(
            f"{chemical_symbols[int(numbers[index])] if numbers is not None else index} "
            f"{held[elements == index].mean():.4f} (DFT {held_dft[elements == index].mean():.4f})"
            for index in np.unique(elements))
        most = ", ".join(f"{name}: {value:.3f}" for value, name in sorted(frames["states"], reverse=True)[:5])
        aux["states_in_gaps"] = f"{by_element}; the most in a cell (electrons): {most}"
    if dos_error_list and sum(len(errors) for errors in dos_error_list):
        # over every atom and energy with a label
        aux["rmse_dos"] = compute_rmse(to_numpy(torch.cat(dos_error_list)))
    for name, errors in fourier_errors.items():
        aux[name] = float(np.sqrt(to_numpy(torch.cat(errors)).mean()))

    aux["time"] = time.time() - start_time

    for name, param in model.named_parameters():
        param.requires_grad = True

    return avg_loss, aux


def frame_names(batch, numbers):
    """Every frame of a batch by its energy label and composition, such as '-134.5194 O10Si4'."""
    energies = to_numpy(batch.energy) if batch.energy is not None else np.full(batch.num_graphs, np.nan)
    if numbers is None:
        return [f"{energy:.4f}" for energy in energies]
    counts = to_numpy(scatter_sum(batch.node_attrs, batch.batch, dim=0, dim_size=batch.num_graphs)).round().astype(int)
    return [f"{energy:.4f} " + "".join(f"{chemical_symbols[int(z)]}{n}" for z, n in zip(numbers, row) if n)
            for energy, row in zip(energies, counts)]


def largest_errors(delta, names, count=5):
    """The frames with the largest errors, by energy label and composition: label - model in thousandths of the unit
    and the share of the squared error; then the RMSE of all the other frames."""
    delta = np.ravel(delta)
    worst = np.argsort(-np.abs(delta))[:count]
    rest = np.delete(delta, worst)
    text = ", ".join(f"{names[i]}: {1e3 * delta[i]:+.0f} ({100 * delta[i]**2 / max((delta**2).sum(), 1e-300):.0f}%)"
                     for i in worst)
    if len(rest):
        text += f"; RMSE of the other {len(rest)} frames {1e3 * np.sqrt(np.mean(rest**2)):.1f}"
    return text


def band_gap_reading(delta, labels, gapped=0.5, closed=0.1):
    """How many band gaps the model reads closed or opened (delta = label - model, eV), and its mean error on the
    frames with a gap."""
    delta, labels = np.ravel(delta), np.ravel(labels)
    model, wide, narrow = labels - delta, labels > gapped, labels < closed
    text = (f"{int((wide & (model < closed)).sum())} of {int(wide.sum())} frames with a gap above {gapped} eV are read "
            f"below {closed} eV, {int((narrow & (model > gapped)).sum())} of {int(narrow.sum())} frames with a gap below "
            f"{closed} eV are read above {gapped} eV")
    if wide.any():
        text += f"; mean label - model of the first {1e3 * delta[wide].mean():+.0f} meV"
    return text


def states_in_gaps(batch, output):
    """Per atom of the frames with a band gap, a Fermi level and a density of states label: the electrons that the
    states of the model and of DFT hold inside the band gap of the label, one width of the states in from either
    edge. The Fermi level label of such a frame is the middle of its gap. Returns the atoms it holds for, and the
    two numbers of electrons."""
    energies = output["dos_energies"]
    width = energies[1] - energies[0]
    gap = batch.band_gap[batch.batch]
    middle = (batch.fermi_level - output["fermi_level"])[batch.batch]
    chosen = ((batch.band_gap_weight > 0.0) & (batch.fermi_level_weight > 0.0) & (batch.band_gap > 4.0 * width))[batch.batch]

    def at(curves, energy):
        position = ((energy - energies[0]) / width).clamp(0.0, len(energies) - 1.000001)
        index = position.floor().long()
        weight = position - index
        return (curves.gather(1, index[:, None])[:, 0] * (1.0 - weight)
                + curves.gather(1, index[:, None] + 1)[:, 0] * weight)

    lower, upper = middle - 0.5 * gap + width, middle + 0.5 * gap - width
    held = at(output["atomic_electrons"], upper) - at(output["atomic_electrons"], lower)
    held_dft = at(output["atomic_electrons_dft"], upper) - at(output["atomic_electrons_dft"], lower)
    return chosen & torch.isfinite(held_dft), held, held_dft


# The targets of the FullRMSE epoch line and error table: their metric of evaluate, shown in thousandths of its unit,
# and the digits shown; the Fourier fields follow REPORTED_SMEARING_WIDTHS.
FULL_RMSE = {
    "E": ("rmse_e_per_atom", 1),
    "F": ("rmse_f", 1),
    "stress": ("rmse_stress", 1),
    "virials": ("rmse_virials_per_atom", 1),
    "Mu": ("rmse_mu_per_atom", 2),
    "polarizability": ("rmse_polarizability_per_atom", 2),
    "DMA": ("rmse_dma", 2),
    "Q": ("rmse_total_charge", 2),
    "EF": ("rmse_fermi_level", 1),
    "WF": ("rmse_work_function", 1),
    "ESPvac": ("rmse_vacuum_potential", 1),
    "Eg": ("rmse_band_gap", 1),
    "DOS": ("rmse_dos", 1),
}


def full_rmse(metrics, key):
    """A metric of evaluate in thousandths of its unit, nan where the data or the model have none."""
    value = metrics.get(key)
    return float("nan") if value is None or isinstance(value, str) else 1e3 * float(value)


def valid_err_log(
    valid_loss,
    eval_metrics,
    logger,
    log_errors,
    epoch,
):
    eval_metrics["mode"] = "eval"
    eval_metrics["epoch"] = epoch
    logger.log(eval_metrics)

    if log_errors == "PerAtomRMSE":
        error_e = eval_metrics["rmse_e_per_atom"] * 1e3
        error_f = eval_metrics["rmse_f"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
        )
    elif log_errors == "FullRMSE":
        # every target, nan where the data or the model have none
        errors = {name: f"{full_rmse(eval_metrics, key):.{digits}f}" for name, (key, digits) in FULL_RMSE.items()}
        errors.update({key: "/".join(f"{full_rmse(eval_metrics, f'rmse_{key}_{w:g}A'):.{2 if key == 'fourier_density' else 1}f}"
                                     for w in widths) for key, widths in REPORTED_SMEARING_WIDTHS.items()})
        widths_rho, widths_esp = ("/".join(f"{w:g}" for w in REPORTED_SMEARING_WIDTHS[key]) for key in REPORTED_SMEARING_WIDTHS)
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_E_per_atom={errors['E']} meV, RMSE_F={errors['F']} meV / A, "
            f"RMSE_stress={errors['stress']} meV / A^3, RMSE_virials_per_atom={errors['virials']} meV, "
            f"RMSE_Mu_per_atom={errors['Mu']} meA, RMSE_polarizability_per_atom={errors['polarizability']} me A^2 / V, "
            f"RMSE_DMA={errors['DMA']} me A^l, RMSE_Q={errors['Q']} me, "
            f"RMSE_rho({widths_rho} A)={errors['fourier_density']} me / A^3, "
            f"RMSE_ESP({widths_esp} A)={errors['fourier_potential']} mV, RMSE_EF={errors['EF']} meV, "
            f"RMSE_WF={errors['WF']} meV, RMSE_ESPvac={errors['ESPvac']} mV, RMSE_Eg={errors['Eg']} meV, "
            f"RMSE_DOS={errors['DOS']} me"
        )
        # A few frames can hold most of a squared error. Each is named by its energy label and composition.
        for name, key in (("the energy per atom", "largest_energy"), ("the Fermi level", "largest_fermi_level"),
                          ("the work function", "largest_work_function"), ("the band gap", "largest_band_gap")):
            if eval_metrics.get(key):
                logging.debug(f"Largest validation errors of {name}, label - model (meV) and share of the squared "
                              f"error: {eval_metrics[key]}")
        if eval_metrics.get("band_gap_reading"):
            logging.debug(f"Band gaps of the validation frames: {eval_metrics['band_gap_reading']}")
        if eval_metrics.get("states_in_gaps"):
            logging.debug("Electrons per atom that the states of the model hold inside the band gaps of the labels (one "
                          f"state width in from either edge), by element: {eval_metrics['states_in_gaps']}")
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
    elif log_errors == "DipoleRMSE":
        error_mu = eval_metrics["rmse_mu_per_atom"] * 1e3
        logging.info(
            f"Epoch {epoch}: loss={valid_loss:.4f}, RMSE_MU_per_atom={error_mu:.6f} meA/atom"
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
