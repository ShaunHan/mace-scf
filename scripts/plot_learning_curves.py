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
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter
import numpy as np

# (label, line style, results file)
RUNS = [
    ("Scratch training", "--", Path("fit_0_run-7777_train.txt")),
    ("MACE-POLAR finetuning", "-", Path("finetune/results/fit_0_run-7777_train.txt")),
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


def format_ticks(ax):
    """Label intermediate log ticks, including ranges below one decade."""
    lo, hi = ax.get_ylim()
    if Y_SCALE == "log":
        decades = np.log10(hi / lo)
        locator = (MaxNLocator(nbins=5, min_n_ticks=4) if decades < 0.5 else
                   LogLocator(base=10, subs=(1,) if decades > 3 else (1, 2, 3, 5, 7),
                              numticks=100))
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=100))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # Keep text clear of the horizontal borders shared with the next row.
    inner = np.exp(np.log([lo, hi]) + np.array([1, -1]) * 0.025 * np.log(hi / lo)) if Y_SCALE == "log" else (
        np.array([lo, hi]) + np.array([1, -1]) * 0.025 * (hi - lo))
    ticks = ax.get_yticks()
    ax.set_yticks(ticks[(ticks > inner[0]) & (ticks < inner[1])])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_learning_curves(runs=RUNS, output=OUTPUT):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10, "axes.linewidth": 0.85,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.bottom": True, "axes.spines.left": True, "axes.grid": False,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.8), sharex=True,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    handles, summaries, epoch_limits = [], [[] for _ in PANELS], []
    has_training = False
    for label, style, path in runs:
        data = read_metrics(path)
        run_plotted = False
        for split, color in COLORS.items():
            for ax, (_, _, keys), summary in zip(axes.flat, PANELS, summaries):
                epochs, errors = curve(data[split], keys)
                finite = np.flatnonzero(np.isfinite(errors))
                if len(finite) < 2:
                    continue
                last = finite[-1]
                if split == "Validation":
                    summary.append(Line2D([], [], color=color, linestyle=style, linewidth=1.7,
                        label=f"{label}: {errors[last]:.4g} ({epochs[last]:g})"))
                if Y_SCALE == "log":
                    errors[errors == 0] = np.nan
                ax.plot(epochs, errors, color=color, linestyle=style, linewidth=1.7)
                epoch_limits.extend((epochs[finite[0]], epochs[last]))
                run_plotted = True
                has_training |= split == "Training"
        if run_plotted:
            handles.append(Line2D([], [], color="0.15", linestyle=style, linewidth=1.8, label=label))
    if not handles:
        plt.close(fig)
        raise ValueError("No RMSE curves found. Set the results paths in RUNS.")

    start, end = min(epoch_limits), max(epoch_limits)
    span = max(end - start, 1)
    # Keep a clear strip for y tick labels inside each shared border.
    axes[0, 0].set_xlim(start - 0.17 * span, end + 0.04 * span)
    ticks = MaxNLocator(nbins=6, integer=True).tick_values(start, end)
    axes[0, 0].set_xticks(ticks[(ticks >= start) & (ticks <= end)])
    axes[0, 0].xaxis.set_minor_locator(AutoMinorLocator(2))
    for letter, ax, (title, unit, _), summary in zip("abcdef", axes.flat, PANELS, summaries):
        ax.text(0.15, 0.965, letter, transform=ax.transAxes, fontweight="bold",
                fontsize=11, va="top")
        ax.text(0.52, 0.965, title, transform=ax.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold")
        ax.text(0.52, 0.905, f"RMSE ({unit})", transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="0.3")
        ax.set_yscale(Y_SCALE)
        ax.margins(y=0.16)
        format_ticks(ax)
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=8.5)
        ax.tick_params(which="major", length=4, width=0.8)
        ax.tick_params(which="minor", length=2, width=0.5)
        ax.tick_params(axis="y", which="major", pad=-6)
        for tick in ax.get_yticklabels():
            tick.set_horizontalalignment("left")
        if summary:
            box = ax.legend(handles=summary, title="Latest validation RMSE (epoch)",
                            loc="upper right", bbox_to_anchor=(0.985, 0.835),
                            fontsize=8, title_fontsize=8.5, handlelength=2.8,
                            borderaxespad=0, borderpad=0.5, labelspacing=0.5,
                            frameon=True, facecolor="white", edgecolor="none", framealpha=0.94)
            box.get_title().set_fontweight("normal")
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    if has_training:
        handles.extend(Line2D([], [], color=color, linewidth=1.8, label=split)
                       for split, color in COLORS.items())
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(handles), frameon=False, handlelength=3.0, columnspacing=2.0)
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.08, top=0.93, wspace=0, hspace=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output.with_suffix(suffix), dpi=300, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    return fig, axes


if __name__ == "__main__":
    plot_learning_curves()
