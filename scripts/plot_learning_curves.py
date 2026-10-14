#!/usr/bin/env python3
"""Plot six RMSE learning curves from JSON-lines results .txt files.

Edit RUNS, then run this script. Requires numpy and matplotlib.
Values are converted from base units to milli-units. Density and potential
already use real-space Parseval normalization for the retained Fourier modes.
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# (label, line style, results file)
RUNS = [
    ("Scratch", "--", Path("fit_0_run-7777_train.txt")),
    ("MACE-POLAR fine-tuning", "-", Path("finetune/results/fit_0_run-7777_train.txt")),
]
OUTPUT = Path("mace_volt_learning_curves")
Y_SCALE = "log"  # Or "linear".
COLORS = {"Training": "#D55E00", "Validation": "#0072B2"}
PANELS = [
    ("Energy", "meV/atom", ("rmse_e_per_atom",)),
    ("Forces", "meV/Å", ("rmse_f",)),
    ("Dipole", "me Å/atom", ("rmse_mu_per_atom",)),
    ("Deformation charge density", "me/Å³", ("rmse_rho",)),
    ("Deformation electrostatic potential", "mV", ("rmse_esp",)),
    ("Work function", "meV", ("rmse_wf_abs", "rmse_phi_abs")),
]


def read_metrics(path):
    """Read epoch metrics; the last record wins after a restart."""
    records = {split: {} for split in COLORS}
    path = Path(path).expanduser()
    if not path.is_file():
        warnings.warn(f"Missing results file: {path}", stacklevel=2)
        return records
    aliases = {"train": "Training", "training": "Training", "eval": "Validation",
               "valid": "Validation", "val": "Validation", "validation": "Validation"}
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("metric_context") == "checkpoint_evaluation":
                continue
            split = aliases.get(str(row.get("split", row.get("mode", "eval"))).lower())
            if split is not None and row.get("epoch") is not None:
                records[split][float(row["epoch"])] = row
    return records


def curve(records, keys):
    epochs = np.array(sorted(records))
    errors = np.array([next((records[e][k] for k in keys if records[e].get(k) is not None),
                           np.nan) for e in epochs], dtype=float) * 1000
    errors[~np.isfinite(errors) | (errors < 0)] = np.nan
    return epochs, errors


def plot_learning_curves(runs=RUNS, output=OUTPUT):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 11.5, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.4), sharex=True)
    handles, summaries = [], [[] for _ in PANELS]
    for label, style, path in runs:
        data = read_metrics(path)
        for split, color in COLORS.items():
            online = split == "Training" and any(
                row.get("metric_context") == "online_training" for row in data[split].values())
            description = "online training" if online else split.lower()
            plotted = False
            for ax, (_, _, keys), summary in zip(axes.flat, PANELS, summaries):
                epochs, errors = curve(data[split], keys)
                finite = np.flatnonzero(np.isfinite(errors))
                if len(finite) < 2:
                    continue  # Never plot an isolated measurement.
                last = finite[-1]
                tag = "T online" if online else split[0]
                short = label.replace(" fine-tuning", "")
                summary.append(f"{short} {tag}: {errors[last]:.4g} ({epochs[last]:g})")
                if Y_SCALE == "log":
                    errors[errors == 0] = np.nan
                ax.plot(epochs, errors, color=color, linestyle=style, linewidth=1.8)
                plotted = True
            if plotted:
                handles.append(Line2D([], [], color=color, linestyle=style, linewidth=1.8,
                                      label=f"{label} · {description}"))
    if not handles:
        plt.close(fig)
        raise ValueError("No RMSE curves found. Set the results paths in RUNS.")

    for letter, ax, (title, unit, _), summary in zip("abcdef", axes.flat, PANELS, summaries):
        ax.set_title(title, pad=12)
        ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=13, va="bottom")
        ax.set_ylabel(f"RMSE ({unit})")
        ax.set_yscale(Y_SCALE)
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.6)
        ax.margins(x=0.015, y=0.09)
        if summary:
            ax.text(0.98, 0.97, "Latest RMSE (epoch)\n" + "\n".join(summary),
                    transform=ax.transAxes, ha="right", va="top", fontsize=8.3,
                    linespacing=1.5, bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.95))
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    fig.suptitle("MACE-VOLT learning curves", fontsize=15, y=0.995)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=2, frameon=False, handlelength=3.0, columnspacing=2.2)
    fig.tight_layout(rect=(0, 0, 1, 0.97 - 0.035 * ((len(handles) + 1) // 2)),
                     h_pad=2.7, w_pad=2.4)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return fig, axes


if __name__ == "__main__":
    plot_learning_curves()
