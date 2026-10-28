#!/usr/bin/env python3
"""Animate a saved MACE-VOLT umbrella run; no model or GPU is required.

Examples:
  python plot_umbrella_gif.py umbrella_8 --index 8
  python plot_umbrella_gif.py umbrella_20 --index 20 --frames 50 --dpi 600
  python plot_umbrella_gif.py umbrella_8 --index 8 --start-step 34000 --step-interval 100

Requires: numpy, matplotlib, ase, pillow.
Read run.json, md.csv and md.traj from the SAME run directory.
The GIF is rendered at FIGSIZE * DPI pixels; GIF has no physical DPI metadata.
Only the display is repeated/reimaged. Charges and energies refer to the
original simulation cell. The energy trace is not a free-energy profile.
The movie ends BEFORE the first release observation; the unrestrained tail
is excluded from both the atomic frames and every plotted trace.
O1 and O2 label the two oxygen IDs in run.json, in that order.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, to_rgba
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
from matplotlib.ticker import MaxNLocator
from ase.io.trajectory import Trajectory
from ase.utils import rotate
from ase.visualize.plot import plot_atoms
from PIL import Image, GifImagePlugin

# --------------------------- settings to edit --------------------------- #
CONFIG_INDEX = 8
N_FRAMES = 50
DPI = 600
FPS = 6
FIGSIZE = (10.8, 7.6)                # 6480 x 4560 pixels at 600 dpi
FONT_PATH = "/mnt/nfs/source/arial.ttf"
ROTATION = "auto"                   # Fixed side view that exposes the short O-O pair
REPEAT = (2, 2, 1)
BACKGROUND_ATOM_ALPHA = 0.70
REFERENCE_SHIFT_V = 4.44            # U vs SHE = calibrated saved voltage - 4.44
ENERGY_MODE = "bias_removed"        # "logged" also available
PAIR_COLORS = ("#D71920", "#E8392F")
ELEMENT_COLORS = {"Ir": "#527A99", "O": "#D87870", "H": "#E9EDF0"}
# ------------------------------------------------------------------------ #


def configure_style(font_path):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    path = Path(font_path).expanduser()
    if not path.is_file():
        try:
            path = Path(font_manager.findfont("Arial", fallback_to_default=False))
        except ValueError:
            warnings.warn(f"Arial not found at {font_path}; using DejaVu Sans.")
            path = Path(font_manager.findfont("DejaVu Sans"))
    font_manager.fontManager.addfont(str(path))
    prop = font_manager.FontProperties(fname=str(path))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [prop.get_name()]
    plt.rcParams["text.usetex"] = False
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "axes.linewidth": .65,
        "axes.edgecolor": "#BCC6D1", "axes.labelcolor": "#334155",
        "xtick.color": "#526173", "ytick.color": "#526173",
        "text.color": "#192C40", "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.unicode_minus": True,
    })
    return str(path)


def load_run(folder, index, reference_shift, energy_mode):
    protocol = json.loads((folder / "run.json").read_text())
    if int(protocol["source_index"]) != index:
        raise ValueError(f"--index {index} does not match run.json source_index.")
    if protocol.get("method") != "distance_umbrella":
        raise ValueError("This script expects a saved distance_umbrella run.")
    with (folder / "md.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("md.csv is empty.")
    release_start = next((i for i, row in enumerate(rows)
                          if row["phase"].strip().lower() == "release"), len(rows))
    excluded = len(rows) - release_start
    rows = rows[:release_start]
    if not rows:
        raise ValueError("No restrained umbrella observations precede the release.")
    if excluded:
        print(f"Excluded {excluded} release observations; "
              f"movie ends at step {rows[-1]['step']} "
              f"({float(rows[-1]['time_fs'])/1000:.3f} ps).")
    def column(name):
        return np.array([float(row[name]) for row in rows])
    steps = column("step").astype(np.int64)
    if np.any(np.diff(steps) <= 0):
        raise ValueError("CSV steps must be unique and strictly increasing.")
    # Read the controller's calibrated voltage, not an uncalibrated ESP-EF.
    voltage_key = next((key for key in (
        "electrode_potential_V", "voltage_V", "WF_over_e_V", "Phi_V")
        if key in rows[0]), None)
    if voltage_key is None:
        raise ValueError("No calibrated controller-voltage column in md.csv.")
    traces = np.array([
        column(voltage_key) - reference_shift,
        column("total_charge_e"),
        column("temperature_K"),
        column("distance_A"),
        column("energy_eV") - (column("bias_eV") if energy_mode == "bias_removed" else 0.),
    ])
    time_ps = column("time_fs") / 1000.
    if not np.isfinite(traces).all() or not np.isfinite(time_ps).all():
        raise ValueError("Nonfinite observations in md.csv; inspect the run first.")
    if np.any(np.diff(time_ps) <= 0):
        raise ValueError("CSV time must be strictly increasing.")
    return protocol, rows, steps, time_ps, traces


def select_rows(steps, start, end, frames, interval):
    candidates = np.flatnonzero((steps >= start) & (steps <= end))
    if not len(candidates):
        raise ValueError("The requested step range contains no saved observations.")
    if interval is not None:
        chosen = [candidates[0]]
        for row in candidates[1:]:
            if steps[row] - steps[chosen[-1]] >= interval:
                chosen.append(row)
        if chosen[-1] != candidates[-1]:
            chosen.append(candidates[-1])
        return np.asarray(chosen), candidates
    count = min(frames, len(candidates))
    selected = (np.array([candidates[-1]]) if count == 1 else
                candidates[np.rint(np.linspace(0, len(candidates)-1, count)).astype(int)])
    return selected, candidates


def load_matched_frames(path, wanted_steps, pair, reference_distances):
    """Join by recorded MD step; never assume CSV row == trajectory index."""
    wanted = set(map(int, wanted_steps))
    found = {}
    with Trajectory(str(path)) as trajectory:
        for frame in trajectory:
            step = int(frame.info["step"])
            if step not in wanted:
                continue
            if step in found:
                raise ValueError(f"Duplicate trajectory step {step}.")
            if any(frame[i].symbol != "O" for i in pair):
                raise ValueError(f"The selected atom IDs {pair} are not both oxygens.")
            found[step] = frame
    missing = sorted(wanted - found.keys())
    if missing:
        raise ValueError(f"Trajectory is missing requested CSV steps: {missing[:8]}")
    frames = [found[int(step)] for step in wanted_steps]
    errors = [abs(frame.get_distance(*pair, mic=True) - distance)
              for frame, distance in zip(frames, reference_distances)]
    if max(errors) > 1.e-5:
        raise ValueError("CSV and trajectory O-O distances disagree; check run provenance.")
    return frames, max(errors)


def prepare_display(frames, pair, rotation):
    """Fixed display translation, minimum-image pair, one highlighted replica."""
    first = frames[0]
    if not np.array_equal(first.pbc, [True, True, False]):
        raise ValueError("This view expects an xy-periodic slab with open z.")
    matrix = rotate(rotation)
    pair_vector = first.get_distance(*pair, mic=True, vector=True)
    midpoint = first.positions[pair[0]] + .5 * pair_vector
    fractional = midpoint @ np.linalg.inv(first.cell.array)
    shift = (.5 - fractional[0]) * first.cell[0] + (.5 - fractional[1]) * first.cell[1]
    displays = []
    for frame in frames:
        if not np.allclose(frame.cell, first.cell):
            raise ValueError("A fixed-cell NVT trajectory is required for this camera.")
        atoms = frame.copy()
        atoms.calc = None
        atoms.set_constraint()
        atoms.info = {}
        atoms.positions += shift
        scaled = atoms.get_scaled_positions(wrap=False)
        scaled[:, :2] %= 1.
        atoms.set_scaled_positions(scaled)
        atoms.positions[pair[1]] = (atoms.positions[pair[0]]
                                   + atoms.get_distance(*pair, mic=True, vector=True))
        displays.append(atoms.repeat(REPEAT))
    # Choose the periodic copy nearest the viewer; only these two are colored.
    copies = list(itertools.product(*(range(n) for n in REPEAT)))
    centers = [displays[0].positions[np.asarray(pair) + len(first)*k].mean(0)
               for k in range(len(copies))]
    chosen = int(np.argmax(np.asarray(centers) @ matrix[:, 2]))
    highlighted = np.asarray(pair) + len(first)*chosen
    z = np.concatenate([atoms.positions[:, 2] for atoms in displays])
    cell = displays[0].cell.array
    origin = np.array([0., 0., .5*(z.min()+z.max()-cell[2, 2])])
    bits = np.array(list(itertools.product((0, 1), repeat=3)))
    corners = (origin + bits @ cell) @ matrix
    clouds = np.concatenate([atoms.positions @ matrix for atoms in displays] + [corners])
    lo, hi = clouds[:, :2].min(0)-1.4, clouds[:, :2].max(0)+1.4
    bbox = np.r_[lo, hi]
    corners[:, :2] -= lo
    edges = [(i, j) for i in range(8) for j in range(i+1, 8)
             if np.sum(np.abs(bits[i]-bits[j])) == 1]
    return displays, highlighted, matrix, bbox, corners, edges


def draw_structure(ax, atoms, highlighted, pair, rotation, matrix, bbox, corners, edges):
    ax.clear()
    colors = np.array([to_rgba(ELEMENT_COLORS.get(symbol, "#8996A6"),
                              BACKGROUND_ATOM_ALPHA)
                       for symbol in atoms.get_chemical_symbols()])
    radii = np.array([{"Ir": .82, "O": .46, "H": .25}.get(symbol, .55)
                      for symbol in atoms.get_chemical_symbols()])
    for k, atom_id in enumerate(highlighted):
        colors[atom_id] = to_rgba(PAIR_COLORS[k])
        radii[atom_id] = .64
    # Projection and bounding box are shared with ASE. The faint full cell
    # outline is drawn explicitly to avoid hundreds of tiny dashed-cell patches.
    for face in ([0, 1, 3, 2], [0, 1, 5, 4]):
        ax.add_patch(Polygon(corners[face, :2], facecolor="#F2F6FA",
                             edgecolor="none", alpha=.55, zorder=-3))
    for i, j in edges:
        ax.plot(corners[[i, j], 0], corners[[i, j], 1], color="#94A4B5",
                lw=.65, alpha=.65, linestyle=(0, (4, 3)), zorder=-2)
    plot_atoms(atoms, ax=ax, rotation=rotation, radii=radii,
               colors=colors, show_unit_cell=0, bbox=bbox)
    for patch in ax.patches:
        if isinstance(patch, Circle):
            patch.set_edgecolor((.28, .36, .43, .40))
            patch.set_linewidth(.34)
    xy = (atoms.positions[highlighted] @ matrix)[:, :2] - bbox[:2]
    # Annotation rings deliberately remain visible through the water layer.
    ax.plot(xy[:, 0], xy[:, 1], color="white", lw=3.2, zorder=7)
    ax.plot(xy[:, 0], xy[:, 1], color="#44556A", lw=1.1, zorder=8)
    for k, point in enumerate(xy):
        ax.add_patch(Circle(point, .64, facecolor=PAIR_COLORS[k], edgecolor="white",
                            lw=1.25, zorder=9))
        offset = (-28, 21) if k == 0 else (28, -21)
        ax.annotate(f"O{k+1}", point, xytext=offset, textcoords="offset points",
                    ha="center", va="center", fontsize=9, weight="bold",
                    color=PAIR_COLORS[k], zorder=12,
                    bbox=dict(boxstyle="round,pad=.22", fc="white", ec="none", alpha=.94),
                    arrowprops=dict(arrowstyle="-", color=PAIR_COLORS[k], lw=.8))
    ax.set_xlim(0, bbox[2]-bbox[0])
    ax.set_ylim(0, bbox[3]-bbox[1])
    ax.set_aspect("equal")
    ax.set_axis_off()


def report_run(protocol, rows, traces):
    sample = np.array([r["phase"] == "sample" for r in rows])
    settings = protocol["settings"]
    print(f"Config {protocol['source_index']}: O pair {protocol['oxygen_pair']}; "
          f"{len(protocol['centers_A'])} windows; final step {rows[-1]['step']}.")
    if sample.any():
        print(f"Production U vs SHE = {traces[0, sample].mean():.3f} +/- "
              f"{traces[0, sample].std():.3f} V (SD); "
              f"T = {traces[2, sample].mean():.1f} K.")
    print(f"O-O: initial {traces[3, 0]:.3f}, minimum {traces[3].min():.3f}, "
          f"final {traces[3, -1]:.3f} A.")
    dt = settings["TIMESTEP_FS"]
    print(f"Per window: {settings['EQUIL_STEPS']*dt/1000:.2f} ps equilibration + "
          f"{(settings['WINDOW_STEPS']-settings['EQUIL_STEPS'])*dt/1000:.2f} ps sampling.")
    print("An activation free energy requires a converged, reweighted PMF; "
          "do not read a barrier from this potential-energy trace.")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", nargs="?", type=Path)
    parser.add_argument("--index", type=int, default=CONFIG_INDEX)
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    parser.add_argument("--step-interval", type=int, help="Minimum MD-step spacing; overrides --frames.")
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--rotation", default=ROTATION,
                        help="auto, or an ASE angle string, e.g. --rotation=-35z,-73x,0y")
    parser.add_argument("--font", default=FONT_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--energy", choices=("bias_removed", "logged"), default=ENERGY_MODE)
    parser.add_argument("--preview-only", action="store_true", help="Save the last selected frame as PNG only.")
    args = parser.parse_args()
    if (args.frames < 1 or args.dpi < 1 or args.fps <= 0
            or (args.step_interval is not None and args.step_interval < 1)):
        parser.error("frames, dpi, fps and step-interval must be positive.")
    folder = (args.folder or Path(f"umbrella_{args.index}")).expanduser().resolve()
    output = (args.output or folder / f"umbrella_{args.index}_evolution.gif").resolve()
    if output.suffix.lower() != ".gif":
        parser.error("--output must end in .gif.")
    output.parent.mkdir(parents=True, exist_ok=True)
    font = configure_style(args.font)
    protocol, rows, steps, time_ps, traces = load_run(
        folder, args.index, REFERENCE_SHIFT_V, args.energy)
    end = int(steps[-1]) if args.end_step is None else args.end_step
    selected, scope = select_rows(steps, args.start_step, end, args.frames, args.step_interval)
    pair = protocol["oxygen_pair"]
    frames, error = load_matched_frames(folder / "md.traj", steps[selected], pair, traces[3, selected])
    if args.rotation == "auto":
        shortest = frames[int(np.argmin(traces[3, selected]))]
        vector = shortest.get_distance(*pair, mic=True, vector=True)
        # A fixed camera keeps the bond's in-plane direction horizontal, making
        # both oxygens visible when the pair is closest. It never rotates in MD.
        azimuth = (-np.degrees(np.arctan2(vector[1], vector[0])) + 90) % 180 - 90
        args.rotation = f"{azimuth:.4f}z,-73x,0y"
    displays, highlighted, matrix, bbox, corners, edges = prepare_display(frames, pair, args.rotation)
    report_run(protocol, rows, traces)
    print(f"Matched {len(selected)} frames by MD step; maximum O-O mismatch {error:.2g} A.")
    print(f"Font: {font}")
    print(f"Fixed ASE viewing rotation: {args.rotation}")

    fig = plt.figure(figsize=FIGSIZE, dpi=args.dpi)
    grid = fig.add_gridspec(5, 2, width_ratios=(1.0, 1.18), left=.035, right=.965,
                           bottom=.080, top=.885, wspace=.25, hspace=.57)
    structure_ax = fig.add_subplot(grid[:, 0])
    axes = [fig.add_subplot(grid[i, 1]) for i in range(5)]
    target = float(protocol["settings"]["TARGET_POTENTIAL"]) - REFERENCE_SHIFT_V
    fig.text(.5, .972, "O-O coupling at the electrified IrO$_2$-water interface "
             f"at {protocol['settings']['TEMPERATURE_K']:g}K and {target:g} V vs. SHE",
             fontsize=13, weight="bold", ha="center", va="top")
    timestamp = fig.text(.245, .910, "", ha="center", fontsize=9, color="#617085")

    headings = ["Electrode potential", "Electrode charge", "Temperature",
                "O-O distance", "Potential energy"]
    labels = ["V vs SHE", r"$q/e$", "K", "Å", "eV"]
    palette = ["#3978B3", "#8270B5", "#D18C39", "#198F83", "#617895"]
    lines, dots, cursors, numbers = [], [], [], []
    t0, t1 = time_ps[scope[0]], time_ps[scope[-1]]
    if t1 == t0:
        t1 = t0 + .001
    for i, ax in enumerate(axes):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E9EEF3", lw=.55)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5, width=.6, pad=3)
        ax.set_xlim(t0, t1)
        lo, hi = traces[i, scope].min(), traces[i, scope].max()
        if i == 0:
            lo, hi = min(lo, target), max(hi, target)
        if i == 2:
            lo, hi = min(lo, 300.), max(hi, 300.)
        padding = max((hi-lo)*.16, [0.12, .015, 8., .08, .1][i])
        ax.set_ylim(lo-padding, hi+padding)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_ylabel(labels[i], labelpad=8)
        ax.text(0, 1.09, headings[i], transform=ax.transAxes, weight="bold", fontsize=9)
        numbers.append(ax.text(1, 1.09, "", transform=ax.transAxes, ha="right",
                               weight="bold", fontsize=9, color=palette[i]))
        lines.append(ax.plot([], [], color=palette[i], lw=.85, alpha=.95)[0])
        dots.append(ax.plot([], [], "o", ms=4, color=palette[i], mec="white", mew=.65, zorder=5)[0])
        cursors.append(ax.axvline(t0, color="#93A4B6", lw=.55, alpha=.5, zorder=0))
        if i < 4:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Time (ps)", labelpad=5)
    axes[0].axhline(target, color="#B8794E", linestyle=(0, (4, 3)), lw=.8)
    axes[2].axhline(300., color="#7C8C9F", linestyle=(0, (4, 3)), lw=.8)
    lo, hi = traces[0, scope].min(), traces[0, scope].max()
    voltage_norm = TwoSlopeNorm(vmin=min(lo, target-.1), vcenter=target, vmax=max(hi, target+.1))

    def render(frame_index):
        row_index = selected[frame_index]
        current = time_ps[row_index]
        draw_structure(structure_ax, displays[frame_index], highlighted, pair,
                       args.rotation, matrix, bbox, corners, edges)
        history = scope[scope <= row_index]
        for i in range(5):
            lines[i].set_data(time_ps[history], traces[i, history])
            dots[i].set_data([current], [traces[i, row_index]])
            cursors[i].set_xdata([current, current])
            value = traces[i, row_index]
            numbers[i].set_text(f"{value:.0f} K" if i == 2 else
                               f"{value:+.3f} e" if i == 1 else
                               f"{value:.2f} Å" if i == 3 else
                               f"{value:.2f} eV" if i == 4 else f"{value:.2f} V")
        dots[0].set_color(matplotlib.colormaps["coolwarm"](voltage_norm(traces[0, row_index])))
        timestamp.set_text(f"{current:.3f} ps")
        fig.canvas.draw()

    preview = output.with_suffix(".png")
    if args.preview_only:
        render(len(selected)-1)
        fig.savefig(preview, dpi=args.dpi)
        print(f"Saved preview: {preview}")
        plt.close(fig)
        return

    # Stream full GIF frames: unlike PillowWriter, do not retain 50 large RGB
    # canvases in RAM. Reuse one palette for stable colors throughout the GIF.
    duration = max(10, int(round(1000/args.fps/10))*10)
    temporary = output.with_suffix(".gif.part")
    palette_image = None
    with temporary.open("wb") as stream:
        for k in range(len(selected)):
            render(k)
            rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
            if palette_image is None:
                # Reserve palette colors from the whole trace view while the
                # animation itself still starts at the first requested frame.
                render(len(selected)-1)
                full = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
                palette_image = full.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
                full.close()
            frame_image = rgb.quantize(palette=palette_image, dither=Image.Dither.NONE)
            rgb.close()
            if k == 0:
                header, _ = GifImagePlugin.getheader(frame_image, info={"loop": 0, "optimize": False})
                for block in header:
                    stream.write(block)
            for block in GifImagePlugin.getdata(
                    frame_image, duration=duration + (1200 if k == len(selected)-1 else 0),
                    disposal=1, optimize=False):
                stream.write(block)
            frame_image.close()
            if k == 0 or (k+1) % 5 == 0 or k == len(selected)-1:
                print(f"Rendered {k+1}/{len(selected)} frames at {args.dpi} dpi.", flush=True)
        stream.write(b";")
    temporary.replace(output)
    fig.savefig(preview, dpi=args.dpi)
    plt.close(fig)
    if palette_image is not None:
        palette_image.close()
    with Image.open(output) as check:
        if check.n_frames != len(selected):
            raise RuntimeError("GIF frame count differs from requested saved frames.")
        print(f"Saved {check.n_frames} frames, {check.width} x {check.height} pixels: {output}")
    print(f"Final-frame PNG: {preview}")


if __name__ == "__main__":
    main()
