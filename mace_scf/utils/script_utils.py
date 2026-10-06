from mace.modules.models import (
    ScaleShiftMACE,
    MACE,
)
import dataclasses
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from prettytable import PrettyTable
from torch_ema import ExponentialMovingAverage

from mace.tools import AtomicNumberTable, torch_geometric
import mace.tools.scripts_utils
from mace.tools.tables_utils import custom_key  as mace_table_custom_key

# update
from mace import data
from ..data import ExtAtomicData
from .train import evaluate



@dataclasses.dataclass
class SubsetCollection:
    train: data.Configurations
    valid: data.Configurations
    tests: List[Tuple[str, data.Configurations]]


NEW_TABLE_TYPES = [
    "ElectrostaticRMSE",
    "DensityCoefficientsRMSE", 
    "DensityEnergyRMSE", 
    "PerAtomRMSE",
    "DipoleRMSE",
    "DensityDipoleRMSE",
    "EnergyDensityDipoleRMSE",
    "EnergyDipolePotentialsRMSE"
]


def create_error_table(
    table_type: str,
    all_data_loaders: dict,
    model: torch.nn.Module,
    model_eval_wrapper,
    loss_fn: torch.nn.Module,
    log_wandb: bool,
    device: str,
    distributed: bool = False,
) -> PrettyTable:
    assert table_type in NEW_TABLE_TYPES

    if log_wandb:
        import wandb
    table = PrettyTable()

    if table_type == "ElectrostaticRMSE":
        vacuum_enabled = bool(getattr(loss_fn.loss_fns.get('fourier_potential'), 'vacuum_weight', 0.))
        esp_title = 'rmse_esp(tot/vac) mV' if vacuum_enabled else 'rmse_esp mV'
        table.field_names = ["split", "rmse_E meV/atom", "rmse_F meV/A", "rmse_dip meA/atom", "rmse_rho me/A^3", "rmse_EF meV", esp_title, "rmse_wf(abs/rel) meV"]
    elif table_type == "DensityCoefficientsRMSE":
        table.field_names = [
            "config_type", 
            "RMSE DMA / e A^l", 
            "rel DMA %",
            "RMSE qs",
            "RMSE dipoles"
        ]
    elif table_type == "DensityEnergyRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
        ]
    elif table_type == "PerAtomRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
        ]
    elif table_type == "DensityDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "RMSE DMA / e A^l", 
            "rel DMA %",
        ]
    elif table_type == "EnergyDensityDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE DMA / e A^l",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "polarizability / me A^2 / V",
        ]
    elif table_type == "EnergyDipolePotentialsRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE dipole / eA / atom",
            "relative dipole RMSE %",
            "RMSE ESP / mV",
            "relative ESP RMSE %",
        ]
    # add new tables here...

    for name in sorted(all_data_loaders, key=mace_table_custom_key):
        data_loader = all_data_loaders[name]
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            model_eval_wrapper=model_eval_wrapper,
            ema=None,
            loss_fn=loss_fn,
            data_loader=data_loader,
            device=device,
        )
        if distributed:
            torch.distributed.barrier()
        del data_loader
        torch.cuda.empty_cache()
        
        if hasattr(model, "batch_positions"):
            del model.batch_positions
        if log_wandb:
            wandb_log_dict = {
                name
                + "_final_rmse_e_per_atom": metrics["rmse_e_per_atom"]
                * 1e3,  # meV / atom
                name + "_final_rmse_f": metrics["rmse_f"] * 1e3,  # meV / A
                name + "_final_rel_rmse_f": metrics["rel_rmse_f"],
            }
            wandb.log(wandb_log_dict)

        # catch missing metrics
        all_metric_name = [
            "rmse_e_per_atom",
            "rmse_f",
            "rel_rmse_f",
            "rmse_dma",
            "rel_rmse_dma",
            "rmse_charges",
            "rmse_local_dipoles",
            "rmse_mu_per_atom",
            "rel_rmse_mu",
            "rmse_esp",
            "rel_rmse_esp",
            "rmse_polarizability_per_atom",
            "rmse_fermi_level", "rmse_esp_vac", "rmse_wf_abs", "rmse_wf_rel", "rmse_rho",
        ]
        for metric_name in all_metric_name:
            if metric_name not in metrics:
                metrics[metric_name] = "not found"
                continue
            if not ("rel" in metric_name) or metric_name == 'rmse_wf_rel':
                metrics[metric_name] = f"{1000 * metrics[metric_name]:.2f}"
            else:
                metrics[metric_name] = f"{metrics[metric_name]:.2f}"
        
        # add new tables here...
        if table_type == "ElectrostaticRMSE":
            table.add_row([name]+[metrics[key] for key in
                ("rmse_e_per_atom", "rmse_f", "rmse_mu_per_atom", "rmse_rho", "rmse_fermi_level")]
                +[metrics['rmse_esp']+'/'+metrics['rmse_esp_vac'] if vacuum_enabled else metrics['rmse_esp'],
                  metrics['rmse_wf_abs']+'/'+metrics['rmse_wf_rel']])
        elif table_type == "DensityCoefficientsRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_dma'],
                    metrics['rel_rmse_dma'],
                    metrics['rmse_charges'],
                    metrics['rmse_local_dipoles'],
                ]
            )
        elif table_type == "DensityEnergyRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                    metrics['rmse_dma'],
                ]
            )
        elif table_type == "PerAtomRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                ]
            )
        elif table_type == "DipoleRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                ]
            )
        elif table_type == "DensityDipoleRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                    metrics['rmse_dma'],
                    metrics['rel_rmse_dma'],
                ]
            )
        elif table_type == "EnergyDensityDipoleRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                    metrics['rmse_dma'],
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                    metrics['rmse_polarizability_per_atom'],
                ]
            )
        elif table_type == "EnergyDipolePotentialsRMSE":
            table.add_row(
                [
                    name,
                    metrics['rmse_e_per_atom'],
                    metrics['rmse_f'],
                    metrics['rel_rmse_f'],
                    metrics['rmse_mu_per_atom'],
                    metrics['rel_rmse_mu'],
                    metrics['rmse_esp'],
                    metrics['rel_rmse_esp'],
                ]
            )
        # add new tables here...

    if table_type == 'ElectrostaticRMSE' and table.rows:
        for index,name in reversed(list(enumerate(table.field_names))):
            if index and all(row[index] in ('not found','not found/not found') for row in table.rows):
                table.del_column(name)
    return table
