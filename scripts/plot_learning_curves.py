#!/usr/bin/env python3
"""Plot MACE-VOLT scratch / MACE-POLAR fine-tuning learning curves.

Edit RUNS below, then run: python plot_learning_curves.py
Requires only numpy and matplotlib. Input files contain one JSON object per line.

The supplied *_train.txt contains validation records (mode="eval"). Its name
does not imply training-set RMSEs. Training curves need actual epoch-level RMSE
records: mode/split="train" in the same log, or a separate training-evaluation
log supplied as the fourth entry in RUNS. Minibatch losses are not substituted.

rmse_rho and rmse_esp already have real-space normalization (Parseval): e/Ang^3
and V, respectively, for the retained Fourier modes. The potential metric is
uniform-offset aligned. No FFT, volume factor, or extra smoothing is applied.
The work-function panel uses absolute RMSE, not the offset-removed relative RMSE.
"""

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


# (label, line style, results log, optional separate TRAINING-set RMSE log)
# Paths are relative to the directory from which you run this script.
RUNS = [
    ("Scratch", "--", Path("fit_0_run-7777_train.txt"), None),
    ("MACE-POLAR fine-tuning", "-",
     Path("finetune/results/fit_0_run-7777_train.txt"), None),
]
OUTPUT = Path("mace_volt_learning_curves")  # Saves .png and .svg.
Y_SCALE = "log"  # Set to "linear" if preferred. Epoch is always linear.

COLORS = {"Training": "#0072B2", "Validation": "#D55E00"}
# The newer rmse_phi_abs in V is numerically equivalent to rmse_wf_abs in eV
# for one elementary charge, so both convert to the requested meV panel by 1000.
PANELS = [
    ("Energy", "RMSE (eV/atom)", ("rmse_e_per_atom",), 1.0),
    ("Forces", "RMSE (eV/Å)", ("rmse_f",), 1.0),
    ("Dipole", "RMSE (e Å/atom)", ("rmse_mu_per_atom",), 1.0),
    ("Deformation charge density", "RMSE (e/Å³)", ("rmse_rho",), 1.0),
    ("Deformation electrostatic potential", "RMSE (V)", ("rmse_esp",), 1.0),
    ("Work function", "RMSE (meV)", ("rmse_wf_abs", "rmse_phi_abs"), 1000.0),
]


def read_metrics(path, force_split=None):
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
            split = force_split or aliases.get(str(tag).lower())
            if split is not None and "epoch" in row:
                records[split][float(row["epoch"])] = row
    return records


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
    if Y_SCALE == "log":
        errors[errors == 0] = np.nan
    return epochs, errors


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
    for label, style, log_path, training_path in runs:
        data = read_metrics(log_path)
        data["Training"].update(read_metrics(training_path, "Training")["Training"])
        for split, color in COLORS.items():
            missing, plotted = [], False
            for ax, (title, units, keys, factor) in zip(axes.flat, PANELS):
                epochs, errors = curve(data[split], keys, factor)
                if not np.isfinite(errors).any():
                    missing.append(title)
                    continue
                ax.plot(epochs, errors, color=color, linestyle=style, linewidth=1.8)
                plotted = True
            if plotted:
                handles.append(Line2D([], [], color=color, linestyle=style,
                                      linewidth=1.8, label=f"{label} · {split.lower()}"))
            if missing:
                warnings.warn(f"{label}: no {split.lower()} RMSE for "
                              + ", ".join(missing) + ".", stacklevel=2)
    if not handles:
        plt.close(fig)
        raise ValueError("No RMSE curves found. Set the input paths in RUNS.")

    for letter, ax, (title, units, _, _) in zip("abcdef", axes.flat, PANELS):
        ax.set_title(title, pad=12)
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=13, va="bottom")
        ax.set_ylabel(units)
        ax.set_yscale(Y_SCALE)
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.margins(x=0.015, y=0.09)
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("MACE-VOLT learning curves", fontsize=15, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=2, frameon=False, handlelength=3.0, columnspacing=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.7, w_pad=2.4)

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
