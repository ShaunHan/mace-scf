#!/usr/bin/env python3
"""Plot MACE-VOLT scratch / MACE-POLAR fine-tuning learning curves.

Edit RUNS below, then run: python plot_learning_curves.py
Requires only numpy and matplotlib. Input files contain one JSON object per line.

The supplied *_train.txt contains validation records (mode="eval"). Its name
does not imply training-set RMSEs. The debug log's final error tables provide
training RMSEs at selected checkpoints, used only in the value annotations.
Isolated training checkpoints are never plotted as points or joined into curves.
Actual mode/split="train" epoch records in the JSON log produce training curves.
The new logging patch marks these as online training measurements: parameters
change during the epoch, and validation can use EMA and a different SCF budget.
Its sibling *_final.txt supplies unrounded checkpoint errors when available.
Minibatch losses and scalar-reference calibration errors are not substituted.

JSON values use base units; the human-readable training log multiplies each
metric below by 1000. The plot does the same. Here "me" means 0.001 elementary
charges. rmse_rho and rmse_esp already have real-space normalization (Parseval)
for the retained Fourier modes. The potential metric is uniform-offset aligned.
No FFT, volume factor, or extra smoothing is applied.
The work-function panel uses absolute RMSE, not the offset-removed relative RMSE.
Each panel reports the last recorded value and its epoch for each curve.
Missing measurements are explicitly labeled "not logged". Table values are
rounded by the training logger. A saved checkpoint can precede the last epoch.
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
import numpy as np


# (label, validation/curve line style, JSON results log, optional debug log)
# Paths are relative to the directory from which you run this script.
RUNS = [
    ("Scratch", "--", Path("fit_0_run-7777_train.txt"),
     Path("fit_0_run-7777_debug.log")),
    ("MACE-POLAR fine-tuning", "-",
     Path("finetune/results/fit_0_run-7777_train.txt"),
     Path("finetune/logs/fit_0_run-7777_debug.log")),
]
OUTPUT = Path("mace_volt_learning_curves")  # Saves .png and .svg.
Y_SCALE = "log"  # Set to "linear" if preferred. Epoch is always linear.

COLORS = {"Training": "#D55E00", "Validation": "#0072B2"}
# The newer rmse_phi_abs in V is numerically equivalent to rmse_wf_abs in eV
# for one elementary charge, so both convert to the requested meV panel by 1000.
PANELS = [
    ("Energy", "RMSE (meV/atom)", ("rmse_e_per_atom",), 1000.0),
    ("Forces", "RMSE (meV/Å)", ("rmse_f",), 1000.0),
    ("Dipole", "RMSE (me Å/atom)", ("rmse_mu_per_atom",), 1000.0),
    ("Deformation charge density", "RMSE (me/Å³)", ("rmse_rho",), 1000.0),
    ("Deformation electrostatic potential", "RMSE (mV)", ("rmse_esp",), 1000.0),
    ("Work function", "RMSE (meV)", ("rmse_wf_abs", "rmse_phi_abs"), 1000.0),
]


def read_metrics(path):
    """Keep the last record at each epoch, e.g. after restarting training."""
    records = {split: {} for split in COLORS}
    if path is None:
        return records
    path = Path(path).expanduser()
    if not path.is_file():
        warnings.warn(f"Missing log: {path}; its curves will be omitted.", stacklevel=2)
        return records
    aliases = {"train": "Training", "training": "Training",
               "eval": "Validation", "valid": "Validation",
               "val": "Validation", "validation": "Validation"}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            tag = row.get("split", row.get("mode", "eval"))
            split = aliases.get(str(tag).lower())
            if split is not None and row.get("epoch") is not None:
                records[split][float(row["epoch"])] = row
    return records


def read_training_checkpoints(path):
    """Read only final-table training rows; their printed units are milli-units."""
    if path is None or not Path(path).expanduser().is_file():
        return {}
    names = {"RMSE_E_per_atom": "rmse_e_per_atom", "RMSE_F": "rmse_f",
             "RMSE_dip_per_atom": "rmse_mu_per_atom", "RMSE_rho": "rmse_rho",
             "RMSE_ESP": "rmse_esp", "RMSE_WF(abs/rel)": "rmse_wf_abs",
             "RMSE_Phi(abs/rel)": "rmse_phi_abs"}
    records, columns, epoch, in_table = {}, [], None, False
    for line in Path(path).expanduser().read_text(encoding="utf-8-sig").splitlines():
        loaded = re.search(r"Loaded model from epoch\s+(\d+)", line)
        if loaded:
            epoch, in_table = int(loaded[1]), False
        if "Final error table for stage " in line:
            in_table, columns = True, []
        if not in_table or not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells[0] == "split":
            columns = [names.get(cell.split()[0]) for cell in cells]
        elif cells[0] == "train" and columns and epoch is not None:
            row = {"epoch": epoch, "_checkpoint": True}
            for key, value in zip(columns, cells):
                absolute = value.split("/")[0]
                if key and absolute != "not found":
                    # Absolute error is the first value in abs/rel columns.
                    row[key] = float(absolute) / 1000.0
            records[float(epoch)] = row
    return records


def read_run(log_path, debug_path):
    data = read_metrics(log_path)
    checkpoints = read_training_checkpoints(debug_path)
    path = Path(log_path).expanduser()
    final_path = path.with_name(path.name.replace("_train.txt", "_final.txt"))
    if final_path != path and final_path.is_file():
        checkpoints.update(read_metrics(final_path)["Training"])
    for epoch, row in list(data["Training"].items()):
        if row.get("metric_context") == "checkpoint_evaluation":
            checkpoints[epoch] = data["Training"].pop(epoch)
    data["Checkpoint"] = checkpoints
    return data


def curve(records, keys, factor):
    points = []
    for epoch, row in sorted(records.items()):
        value = next((row[key] for key in keys if row.get(key) is not None), np.nan)
        points.append((epoch, float(value) * factor))
    if not points:
        return np.array([]), np.array([])
    epochs, errors = np.asarray(points).T
    # Missing/invalid records make gaps; never invent or interpolate errors.
    errors[~np.isfinite(errors) | (errors < 0)] = np.nan
    return epochs, errors


def annotate_final(ax, summary):
    """Final values use the axis units; parentheses give the recorded epoch."""
    rows = [TextArea("Latest RMSE (epoch)",
                     textprops={"size": 8.3, "weight": "bold"})]
    for label, values in summary.items():
        cells = [TextArea(label.replace(" fine-tuning", "") + ":",
                          textprops={"size": 8.3})]
        for split, color in COLORS.items():
            item = values[split]
            tag = "T online" if split == "Training" and values["online"] else split[:1]
            if split == "Training" and item is None and values["Checkpoint"] is not None:
                item, tag = values["Checkpoint"], "T checkpoint"
            value = "not logged" if item is None else f"{item[0]:.4g} ({item[1]:g})"
            cells.append(TextArea(f"{tag} {value}",
                                  textprops={"size": 8.3, "color": color}))
        rows.append(HPacker(children=cells, align="baseline", pad=0, sep=7))
        if values["Training"] is not None and values["Checkpoint"] is not None:
            value, epoch = values["Checkpoint"]
            rows.append(TextArea(f"  T checkpoint {value:.4g} ({epoch:g})",
                                 textprops={"size": 8.3, "color": COLORS["Training"]}))
    box = AnchoredOffsetbox(
        loc="upper right", child=VPacker(children=rows, align="left", pad=0, sep=4),
        pad=0.4, borderpad=0.6, frameon=True,
    )
    box.patch.set(facecolor="white", edgecolor="0.85", alpha=0.95, linewidth=0.6)
    ax.add_artist(box)


def plot_learning_curves(runs=RUNS, output=OUTPUT):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
        "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "xtick.direction": "out", "ytick.direction": "out",
        "svg.fonttype": "none",  # Keep labels editable as text in Inkscape.
        "mathtext.fontset": "dejavusans", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True)
    handles = []
    summaries = [{} for _ in PANELS]
    for label, style, log_path, debug_path in runs:
        data = read_run(log_path, debug_path)
        online = any(row.get("metric_context") == "online_training"
                     for row in data["Training"].values())
        for panel, summary in zip(PANELS, summaries):
            epochs, errors = curve(data["Checkpoint"], panel[2], panel[3])
            finite = np.flatnonzero(np.isfinite(errors))
            summary[label] = {"Training": None, "Validation": None, "online": online,
                              "Checkpoint": (errors[finite[-1]], epochs[finite[-1]])
                              if len(finite) else None}
        groups = [("Training", COLORS["Training"], "online training" if online else "training"),
                  ("Validation", COLORS["Validation"], "validation")]
        for split, color, description in groups:
            missing, plotted = [], False
            for ax, panel, summary in zip(axes.flat, PANELS, summaries):
                title, units, keys, factor = panel
                epochs, errors = curve(data[split], keys, factor)
                finite = np.flatnonzero(np.isfinite(errors))
                if not len(finite):
                    missing.append(title)
                    continue
                last = finite[-1]
                summary[label][split] = (errors[last], epochs[last])
                if len(finite) < 2:
                    continue  # An isolated measurement cannot define a curve.
                if Y_SCALE == "log":
                    errors[errors == 0] = np.nan
                ax.plot(epochs, errors, color=color, linestyle=style, linewidth=1.8)
                plotted = True
            if plotted:
                handles.append(Line2D([], [], color=color, linestyle=style, linewidth=1.8,
                                      label=f"{label} · {description}"))
            if missing and (split == "Validation" or data[split]):
                warnings.warn(f"{label}: no {split.lower()} RMSE for "
                              + ", ".join(missing) + ".", stacklevel=2)
    if not handles:
        plt.close(fig)
        raise ValueError("No RMSE curves found. Set the input paths in RUNS.")

    for letter, ax, panel, summary in zip("abcdef", axes.flat, PANELS, summaries):
        title, units, _, _ = panel
        ax.set_title(title, pad=12)
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=13, va="bottom")
        ax.set_ylabel(units)
        ax.set_yscale(Y_SCALE)
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.margins(x=0.015, y=0.09)
        annotate_final(ax, summary)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("MACE-VOLT learning curves", fontsize=15, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=2, frameon=False, handlelength=3.0, columnspacing=2.2)
    legend_rows = (len(handles) + 1) // 2
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.035 * legend_rows), h_pad=2.7, w_pad=2.4)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = output.with_suffix("." + extension)
        fig.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.16)
        print(f"Saved {destination.resolve()}")
    plt.close(fig)
    return fig, axes


if __name__ == "__main__":
    plot_learning_curves()
