"""Collect training labels directly from VASP outputs. Edit the settings below."""
from pathlib import Path
import json
import multiprocessing as mp
import re

import h5py
import numpy as np
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Locpot, Outcar


BASE_PATH = Path("/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft_plugin/")
PROTO_BASE_PATH = None  # Proto LOCPOTs, only for old runs without their own LOCPOT_proto.
CHUNK_SIZE = 5
N_PROCS = 64
REUSE_HDF = True
K_CUTOFF = 18.0  # HDF bandwidth in inverse Angstrom.
FOURIER_CUTOFF_FACTOR = 1.25  # XYZ bandwidth; must cover the model's k-space cutoff.
DENSITY_SIGMAS = [1.5]
POTENTIAL_SIGMAS = [1.5]  # [] disables potential collection.
FMAX_TOL = 10.0  # eV/A: largest atomic force of a kept frame.
VALID_FRACTION = 0.2
SPLIT_SEED = 7777
SKIP_IDS = [1, 7, 10, 46, 66, 97, 101, 115, 194, 218, 228, 266, 288, 296, 308, 318, 342, 353, 375, 391, 420, 439, 506, 555, 566, 755]  # Original VASP IDs: batch start + output index.
DOS_STEP = 0.05  # eV: DOS_STEP of mace_scf/data/new_atomic_data.py.
DOS_RANGE = 10.0  # eV: the total DOS is kept at most this far from the Fermi level.
HDF_NAME = "fourier_data.hdf"

OUT_XYZ = BASE_PATH / "combined_VASP_shifted.xyz"
OUT_TRAIN_XYZ = BASE_PATH / "combined_VASP_shifted_train.xyz"
OUT_VALID_XYZ = BASE_PATH / "combined_VASP_shifted_valid.xyz"


def get_output_dirs():
    folders = sorted((p for p in BASE_PATH.glob("vasp_*") if p.is_dir()),
                     key=lambda p: int(p.name.split("_")[1]))
    return [folder / f"output_{i}" for folder in folders for i in range(CHUNK_SIZE)
            if int(folder.name.split("_")[1]) + i not in SKIP_IDS
            and (folder / f"output_{i}").is_dir()]


def read_grid(path, atoms, reader):
    grid = reader.from_file(str(path))
    structure = grid.structure
    delta = structure.frac_coords - atoms.get_scaled_positions(wrap=False)
    if ([str(site.specie) for site in structure] != atoms.get_chemical_symbols()
            or not np.allclose(structure.lattice.matrix, atoms.cell, rtol=0, atol=1e-4)
            or not np.allclose(delta - np.rint(delta), 0, rtol=0, atol=1e-5)):
        raise ValueError(f"Structure/grid mismatch: {path}")
    values = np.asarray(grid.data["total"], dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f"Invalid volumetric data: {path}")
    return values


def hermitian(coefficients):
    partner = np.ix_(*[(-np.arange(n)) % n for n in coefficients.shape])
    return 0.5 * (coefficients + coefficients[partner].conj())


def truncate_fft(coefficients, shape):
    if np.any(np.asarray(shape) > coefficients.shape):
        raise ValueError(f"Requested FFT shape {tuple(shape)} exceeds {coefficients.shape}")
    axes = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in shape]
    indices = np.ix_(*[k % n for k, n in zip(axes, coefficients.shape)])
    return hermitian(coefficients[indices] * np.prod(shape) / coefficients.size)


def dos_label(kpoints, parameters, fermi):
    """The total DOS of the cell in states/eV at fermi + j * DOS_STEP, j = -n..n, from the eigenvalues; or None.

    Nothing is projected. VASP has all the states of every k-point up to the lowest of their highest bands (NBANDS),
    so n is the largest whose bins lie below it, and at most DOS_RANGE / DOS_STEP.
    """
    if not kpoints:
        return None
    spinor = parameters.get("lnoncollinear", False) or parameters.get("lsorbit", False)
    capacity = 1.0 if spinor or parameters.get("ispin", 1) == 2 else 2.0
    top = min(float(np.max(k.eps_n)) for k in kpoints) - fermi
    n = min(round(DOS_RANGE / DOS_STEP), int(np.floor(top / DOS_STEP - 0.5)))
    if n <= 0:
        return None
    channel = {}
    for k in kpoints:
        channel[k.s] = channel.get(k.s, 0.0) + k.weight
    dos = np.zeros(2 * n + 1)
    for k in kpoints:
        # Each state adds its k-point weight, normalized in its spin channel, times the electrons it holds.
        bins = np.rint((np.asarray(k.eps_n) - fermi) / DOS_STEP).astype(int)
        inside = np.abs(bins) <= n
        np.add.at(dos, bins[inside] + n, k.weight / channel[k.s] * capacity / DOS_STEP)
    return dos


def fourier_data(outdir, atoms, parameters, axis, plugin=False):
    """Cache unsmoothed Fourier grids; widths and vacuum sampling remain editable."""
    names = ["fourier_density"] if DENSITY_SIGMAS else []
    if POTENTIAL_SIGMAS:
        names += ["fourier_potential", "fourier_proto_potential"]
        if axis is not None:
            names += ["planar_potential"]
    if plugin:
        names += ["planar_electron_density"]
    if not names:
        return {}
    path = outdir / HDF_NAME
    if REUSE_HDF and path.is_file():
        with h5py.File(path, "r") as handle:
            if handle.attrs["k_cutoff"] != K_CUTOFF or not set(names) <= set(handle):
                raise ValueError(f"Cache settings differ in {path}; set REUSE_HDF=False")
            return {name: handle[name][...] for name in names}

    shape = 2 * np.ceil(K_CUTOFF * atoms.cell.lengths() / (2 * np.pi)).astype(int) + 1
    data = {}
    if DENSITY_SIGMAS:
        density = read_grid(outdir / "AECCAR1", atoms, Chgcar)
        final_density = read_grid(outdir / "AECCAR2", atoms, Chgcar)
        if density.shape != final_density.shape:
            raise ValueError(f"AECCAR grid mismatch: {outdir}")
        density -= final_density
        density /= atoms.get_volume()  # Signed deformation charge, in e/Angstrom^3.
        data["fourier_density"] = truncate_fft(np.fft.fftn(density), shape)
        del density, final_density
    if plugin:
        # Only plug-in runs need this grid, to recover their physical dipole.
        data["planar_electron_density"] = read_grid(outdir / "CHGCAR", atoms, Chgcar).mean(
            axis=tuple(i for i in range(3) if i != axis)) / atoms.get_volume()
    if POTENTIAL_SIGMAS:
        if not parameters["lvhar"]:
            raise ValueError(f"Potential collection needs LVHAR=True: {outdir}")
        potential = read_grid(outdir / "LOCPOT", atoms, Locpot)
        # The in-run proto is VASP's first ICHARG=2 potential, the neutral atoms for every charge state.
        proto_path = outdir / "LOCPOT_proto"
        if not proto_path.is_file():
            if plugin:
                raise ValueError(f"Charged plug-in run without its in-run LOCPOT_proto: {outdir}")
            proto_path = PROTO_BASE_PATH / outdir.relative_to(BASE_PATH) / "LOCPOT"
        proto = read_grid(proto_path, atoms, Locpot)
        if potential.shape != proto.shape:
            raise ValueError(f"SCF/proto potential grid mismatch: {outdir}")
        data["fourier_proto_potential"] = truncate_fft(np.fft.fftn(proto), shape)
        if axis is not None:
            data["planar_potential"] = potential.mean(axis=tuple(i for i in range(3) if i != axis))
        potential -= proto  # Neither is shifted: VASP writes both with zero mean, the origin of its Fermi level.
        data["fourier_potential"] = truncate_fft(np.fft.fftn(potential), shape)
    temporary = path.with_suffix(".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["k_cutoff"] = K_CUTOFF
        for name, values in data.items():
            handle.create_dataset(name, data=values)
    temporary.replace(path)
    return data


def add_fourier_targets(atoms, coefficients, sigmas, prefix, charge=None):
    """Compact, unsmoothed XYZ targets; higher-frequency data stay in the HDF."""
    # The widths set the bandwidth only: FOURIER_CUTOFF_FACTOR times the model's k-space cutoff
    # 0.75 * 3.0 * (max_l + 1)**0.3 / sigma (graph_longrange gto_basis_kspace_cutoff, max_l = 1).
    sigma = float(min(sigmas))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Fourier widths must be positive and finite")
    cutoff = FOURIER_CUTOFF_FACTOR * 0.75 * 3.0 * 2**0.3 / sigma
    shape = 2 * np.ceil(cutoff * atoms.cell.lengths() / (2 * np.pi)).astype(int) + 1
    if cutoff > K_CUTOFF or np.any(shape > coefficients.shape):
        raise ValueError("XYZ Fourier bandwidth exceeds the HDF/VASP grid; increase its resolution")
    coefficients = truncate_fft(coefficients, shape)
    target = np.empty((2, *coefficients.shape), dtype=np.float32)
    target[0], target[1] = coefficients.real, coefficients.imag
    if charge is not None:
        target[:, 0, 0, 0] = (charge * coefficients.size / atoms.get_volume(), 0.0)
    atoms.info[f"{prefix}_fftn_shifted"] = target


def vacuum_potential(atoms, planar, axis, correction, field):
    """The vacuum potential just below the dipole correction plane, and its spread over the vacuum.

    As LVACPOTAV, it averages the planar potential over the vacuum planes at least 3 A above the outermost atom
    whose field, the applied field removed, is below 0.1 eV/A. The applied field is carried to the plane, so that
    V(z) + field * (distance to the plane) is the same at every vacuum plane.
    """
    normal = np.linalg.inv(atoms.cell.array)[:, axis]
    length = 1.0 / np.linalg.norm(normal)
    below = (correction - np.arange(len(planar)) / len(planar)) % 1.0 * length
    top = np.min((correction - atoms.get_scaled_positions(wrap=False)[:, axis]) % 1.0) * length
    level = planar + field * below
    gradient = (np.roll(level, -1) - np.roll(level, 1)) / (2 * length / len(planar))
    vacuum = (below > length / len(planar)) & (below < top - 3.0) & (np.abs(gradient) < 0.1)
    if not vacuum.any():
        raise ValueError("No field-free vacuum above the slab; enlarge the vacuum")
    return float(level[vacuum].mean()), float(np.ptp(level[vacuum]))


def converged_electronic_state(outdir, text, plugin, area):
    """Why the electronic state of a run is not converged, or None.

    VASP stops at EDIFF on one small energy change, which a stalled density can pass by chance: the density residual
    rms(c) of the last three steps must be below 1e-2 too, and in a plug-in run the dipole of its last callbacks may
    move the vacuum potential by at most 0.05 V.
    """
    if "EDIFF is reached" not in text.rsplit("Iteration", 1)[-1]:
        return "EDIFF not reached"
    residuals = []
    for line in (outdir / "OSZICAR").read_text().splitlines():
        numbers = line.split()
        if len(numbers) == 8 and numbers[0][-1] == ":":
            residuals.append(float(numbers[7]))
    if residuals and max(residuals[-3:]) > 1e-2:
        return f"density residual rms(c) {max(residuals[-3:]):.1e}"
    if plugin and (outdir / "plugin_scf.csv").is_file():
        dipoles = np.loadtxt(outdir / "plugin_scf.csv", delimiter=",", skiprows=1, ndmin=2)[-6:, 1]
        drift = np.ptp(dipoles) / (0.005526349406 * area)
        if drift > 0.05:
            return f"plug-in dipole drifts by {np.ptp(dipoles):.1e} eA, {drift:.3f} V of vacuum potential"
    return None


def plugin_run(outdir, meta):
    """Whether a counter-charge plug-in acted in this run, as scripts/dft_runs.py records in dft_metadata.json."""
    if "plugin" in meta:
        return bool(meta["plugin"])
    if (outdir / "plugin_observables.npz").is_file():
        raise ValueError(f"Plug-in run without dft_metadata.json: {outdir}")
    return False


def plugin_forces(outdir, forces, normal):
    """The forces of a plug-in run, -dE/dR of its energy: VASP's forces plus the net force of the counter charge
    along the slab normal, which VASP removes. The two sets may differ only by a uniform force."""
    with np.load(outdir / "plugin_observables.npz") as observables:
        if "expected_forces" not in observables:
            raise ValueError(f"Plug-in run without its forces: {outdir}")
        full = np.asarray(observables["expected_forces"], dtype=float)
    if full.shape != forces.shape or not np.isfinite(full).all():
        raise ValueError(f"Invalid plug-in forces: {outdir}")
    difference = full - forces
    if np.abs(difference - difference.mean(0)).max() > 1e-3:
        raise ValueError(f"Plug-in and VASP forces differ by more than a net force: {outdir}")
    return forces + (difference.mean(0) @ normal) * normal


def band_edges(kpoints):
    """The valence band maximum and conduction band minimum at zero temperature, or None.

    In every spin channel the electrons of the run fill its lowest bands at every k point, so that the smearing does
    not decide which states are filled. None when they fill no whole number of bands (a charged metal) or when the
    filled bands reach above the empty ones (a metal).
    """
    if not kpoints:
        return None
    full = max(float(np.max(k.f_n)) for k in kpoints)
    maximum, minimum = -np.inf, np.inf
    for spin in sorted({k.s for k in kpoints}):
        channel = [k for k in kpoints if k.s == spin]
        bands = sum(k.weight * float(np.sum(k.f_n)) for k in channel) / (sum(k.weight for k in channel) * full)
        filled = int(round(bands))
        if abs(bands - filled) > 0.01 or not 0 < filled < min(len(k.eps_n) for k in channel):
            return None
        levels = [np.sort(np.asarray(k.eps_n)) for k in channel]
        maximum = max(maximum, max(float(e[filled - 1]) for e in levels))
        minimum = min(minimum, min(float(e[filled]) for e in levels))
    return (maximum, minimum) if minimum > maximum else None


def collect_job(outdir):
    path = outdir / "dft_metadata.json"
    meta = json.loads(path.read_text()) if path.is_file() else {}
    plugin = plugin_run(outdir, meta)
    atoms = read(str(outdir / "vasprun.xml"), format="vasp-xml", index=-1)
    parameters = atoms.calc.parameters
    kpoints = atoms.calc.kpts
    forces = atoms.get_forces(apply_constraint=False)
    energy = atoms.get_potential_energy(force_consistent=True)
    if not np.isfinite(forces).all() or not np.isfinite(energy):
        raise ValueError(f"Nonfinite energy/forces: {outdir}")
    if plugin:
        normal = np.linalg.inv(atoms.cell.array)[:, meta["slab_axis"]]
        forces = plugin_forces(outdir, forces, normal / np.linalg.norm(normal))
    if np.linalg.norm(forces, axis=1).max() > FMAX_TOL:
        print(f"Skipping {outdir}: force exceeds FMAX_TOL")
        return None

    outcar = Outcar(str(outdir / "OUTCAR"))
    outcar.read_pseudo_zval()
    charge = sum(outcar.zval_dict[s] for s in atoms.get_chemical_symbols()) - parameters["nelect"]
    fermi = float(outcar.efermi)
    if not np.isfinite(charge) or not np.isfinite(fermi):
        raise ValueError(f"Nonfinite charge/Fermi level: {outdir}")
    text = (outdir / "OUTCAR").read_text(errors="replace")
    axis = meta["slab_axis"] if plugin else (int(parameters["idipol"]) - 1 if parameters["ldipol"] else 2)
    area = np.linalg.norm(np.cross(*np.delete(atoms.cell.array, axis, axis=0)))
    reason = converged_electronic_state(outdir, text, plugin, area)
    if reason is not None:
        print(f"Skipping {outdir}: {reason}")
        return None

    counter = -charge if plugin else 0.0
    slab = plugin or bool(parameters["ldipol"])
    axis = meta["slab_axis"] if plugin else (int(parameters["idipol"]) - 1 if slab else None)
    external_field = np.zeros(3)
    dipole = np.zeros(3)
    if slab:
        if axis not in (0, 1, 2):
            raise ValueError(f"Slab collection needs IDIPOL=1, 2 or 3: {outdir}")
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        if plugin:
            external_field = np.asarray(meta["external_field"], dtype=float)
            correction = float(meta[f"dipole_correction_{'xyz'[axis]}frac"]) % 1.0
            center = (correction + 0.5) % 1.0
        else:
            external_field = -float(parameters["efield"]) * normal
            dipole = -np.asarray(atoms.get_dipole_moment(), dtype=float).reshape(3)
            if np.all(np.asarray(parameters["dipol"]) == -100):
                raise ValueError(f"Specify DIPOL in VASP to locate its correction plane: {outdir}")
            center = float(parameters["dipol"][axis]) % 1.0
            correction = (center + 0.5) % 1.0
        if not np.isfinite(dipole).all():
            raise ValueError(f"Nonfinite dipole: {outdir}")
        fractional = atoms.get_scaled_positions(wrap=False)[:, axis]
        atoms.positions -= np.floor(fractional - center + 0.5)[:, None] * atoms.cell[axis]
        atoms.pbc[axis] = False

    atoms.calc = None
    atoms.set_constraint()
    atoms.arrays["vasp_forces"] = forces
    # The Fermi level and band gap come from the eigenvalues at zero temperature, so that no smearing changes them.
    # A neutral cell with a gap above 0.1 eV has its Fermi level in the middle of the gap (VASP's EFERMI = MIDGAP);
    # a metal or a charged cell keeps VASP's Fermi level and has no gap label. vasp_dos is counted from this level.
    edges = band_edges(kpoints)
    gapped = abs(charge) < 1e-6 and edges is not None and edges[1] - edges[0] > 0.1
    atoms.info = {
        "config_type": "slab" if slab else "bulk",
        "vasp_free_energy": float(energy),
        "vasp_efermi": 0.5 * (edges[0] + edges[1]) if gapped else fermi,
        "vasp_band_gap": edges[1] - edges[0] if gapped else 0.0,
        "total_charge": float(charge),
        "counter_charge": float(counter),
        "external_field": external_field,
        "vasp_dipole": dipole,
        "config_dipole_weight": np.eye(3)[axis] if slab else np.zeros(3),
        "config_band_gap_weight": float(abs(charge) < 1e-6),
    }
    if slab:
        atoms.info[f"dipole_correction_{'xyz'[axis]}frac"] = correction
    places = [key for key in ("counter_charge_center", "counter_charge_xfrac", "counter_charge_yfrac",
                              "counter_charge_zfrac") if key in meta]
    if places:
        # The Gaussian counter charge as placed, also at Q = 0.
        atoms.info["counter_charge_width"] = float(meta["counter_charge_width"])
        for key in places:
            atoms.info[key] = np.asarray(meta[key], dtype=float) if key.endswith("center") else float(meta[key])
    if plugin:
        # The plug-in implements forces, but not its stress derivative.
        atoms.info["config_stress_weight"] = 0.
        atoms.info["config_virials_weight"] = 0.
    data = fourier_data(outdir, atoms, parameters, axis, plugin)
    dos = dos_label(kpoints, parameters, atoms.info["vasp_efermi"])
    if dos is None:
        print(f"Note {outdir}: no vasp_dos; vasprun.xml has no eigenvalues, or its bands end at the Fermi level")
    else:
        atoms.info["vasp_dos"] = np.round(dos, 3)  # One row for the cell, in states/eV.
    if plugin:
        planar = data["planar_electron_density"]
        length = float(atoms.cell[axis] @ normal)
        fractions = np.arange(len(planar)) / len(planar)
        z = ((fractions - center + 0.5) % 1.0 - 0.5) * length
        z[np.isclose(fractions, correction, rtol=0, atol=1e-12)] = 0.0
        ion_z = ((atoms.get_scaled_positions(wrap=False)[:, axis] - center + 0.5) % 1.0 - 0.5) * length
        valences = [outcar.zval_dict[s] for s in atoms.get_chemical_symbols()]
        moment = np.dot(valences, ion_z) - atoms.get_volume() * np.mean(planar * z)
        atoms.info["vasp_dipole"] = float(moment) * normal
    if DENSITY_SIGMAS:
        add_fourier_targets(atoms, data["fourier_density"], DENSITY_SIGMAS, "vasp_rho", charge)
    if POTENTIAL_SIGMAS:
        if slab:
            level, spread = vacuum_potential(atoms, data["planar_potential"], axis, correction,
                                             float(external_field @ normal))
            if spread > 0.2:
                print(f"Skipping {outdir}: the vacuum potential varies by {spread:.3f} V")
                return None
            if spread > 0.05:
                print(f"Note {outdir}: the vacuum potential varies by {spread:.3f} V; kept at its average")
            vasp = re.findall(r"vacuum level on the upper side and lower side of the slab\s+([-+.\d]+)", text)
            if vasp and not external_field.any() and abs(float(vasp[-1]) - level) > 0.02:
                print(f"Warning {outdir}: vacuum level {level:.3f} V, VASP's LVACPOTAV {float(vasp[-1]):.3f} V")
            atoms.info["vasp_vacuum_potential"] = level
        add_fourier_targets(atoms, data["fourier_potential"], POTENTIAL_SIGMAS, "vasp_phi")
        add_fourier_targets(atoms, data["fourier_proto_potential"], POTENTIAL_SIGMAS, "vasp_proto_phi")
    print(f"Collected {outdir}")
    return atoms, outdir.parent.name, outdir


def same_results(atoms, other):
    """Whether two jobs gave the same energy, Fermi level and vacuum level (to 1e-4) and forces (to 1e-3 eV/A)."""
    labels = ("vasp_free_energy", "vasp_efermi", "vasp_vacuum_potential")
    return (all(abs(atoms.info[key] - other.info[key]) <= 1e-4 for key in labels
                if key in atoms.info and key in other.info)
            and np.abs(atoms.arrays["vasp_forces"] - other.arrays["vasp_forces"]).max() <= 1e-3)


def drop_repeats(records):
    """Drops a job that repeats an earlier one: the same atoms, cell, positions and inputs, and the same results, as
    when the zero-field point of a structure is part of several field series. Jobs with the same inputs but other
    results are all kept and reported."""
    inputs = ("total_charge", "counter_charge", "external_field", "counter_charge_width", "counter_charge_center",
              "counter_charge_xfrac", "counter_charge_yfrac", "counter_charge_zfrac", "dipole_correction_xfrac",
              "dipole_correction_yfrac", "dipole_correction_zfrac")
    kept, earlier = [], {}
    for atoms, group, outdir in records:
        names = tuple(key for key in inputs if key in atoms.info)
        values = (atoms.cell.array, atoms.positions, *(atoms.info[key] for key in names))
        key = (tuple(atoms.get_chemical_symbols()), tuple(atoms.pbc), names,
               *((np.round(np.asarray(value, dtype=float), 6) + 0.0).tobytes() for value in values))
        jobs = earlier.setdefault(key, [])
        match = next((path for other, path in jobs if same_results(atoms, other)), None)
        if match is not None:
            print(f"Skipping {outdir}: repeats {match}")
            continue
        if jobs:
            print(f"Note {outdir}: same inputs as {jobs[0][1]} but other results; both are kept")
        jobs.append((atoms, outdir))
        kept.append((atoms, group))
    return kept


def split_configs(records):
    """Keep each VASP batch in one split, without storing group metadata in XYZ, and VALID_FRACTION of the frames in
    the validation split."""
    sizes, groups = {}, {}
    for atoms, group in records:
        sizes[group] = sizes.get(group, 0) + 1
        groups.setdefault(group, set()).add(atoms.info["config_type"])
    stratum = {group: tuple(sorted(types)) for group, types in groups.items()}
    strata = {}
    for group, key in stratum.items():
        strata.setdefault(key, []).append(group)
    rng = np.random.default_rng(SPLIT_SEED)
    valid_groups = set()
    for groups in strata.values():
        groups = np.asarray(sorted(groups), dtype=object)
        rng.shuffle(groups)
        if len(groups) > 1:
            count = min(max(round(VALID_FRACTION * len(groups)), 1), len(groups) - 1)
            valid_groups.update(groups[:count])
    # Batches differ in size, so whole batches change sides, in one shuffled order, until the frames match; a batch of
    # every stratum stays on each side, and a last frame or two is settled by exchanging two batches of one stratum.
    inside = {key: sum(group in valid_groups for group in groups) for key, groups in strata.items()}
    missing = round(VALID_FRACTION * len(records)) - sum(sizes[group] for group in valid_groups)
    order = np.asarray(sorted(sizes), dtype=object)
    rng.shuffle(order)
    for group in order:
        key, size = stratum[group], sizes[group]
        if size <= missing and group not in valid_groups and len(strata[key]) - inside[key] > 1:
            valid_groups.add(group)
            inside[key] += 1
            missing -= size
        elif size <= -missing and group in valid_groups and inside[key] > 1:
            valid_groups.discard(group)
            inside[key] -= 1
            missing += size
    if missing:
        pair = next(((a, b) for a in order if a not in valid_groups for b in order
                     if b in valid_groups and stratum[a] == stratum[b] and sizes[a] - sizes[b] == missing), None)
        if pair is not None:
            valid_groups.add(pair[0])
            valid_groups.discard(pair[1])
    train = [atoms for atoms, group in records if group not in valid_groups]
    valid = [atoms for atoms, group in records if group in valid_groups]
    if not train or not valid:
        raise ValueError("A group-disjoint split needs at least two VASP batches per stratum")
    return train, valid


def collect():
    paths = get_output_dirs()
    if N_PROCS is None or N_PROCS <= 1:
        records = [collect_job(path) for path in paths]
    else:
        with mp.Pool(processes=N_PROCS) as pool:
            records = list(pool.imap(collect_job, paths))
    records = [record for record in records if record is not None]
    if not records:
        raise SystemExit("No configurations collected")
    collected = len(records)
    records = drop_repeats(records)
    print(f"{len(paths)} jobs: {len(paths) - collected} skipped by the checks above, "
          f"{collected - len(records)} repeats dropped, {len(records)} kept")
    train, valid = split_configs(records)
    for path, configs in ((OUT_XYZ, [a for a, _ in records]),
                          (OUT_TRAIN_XYZ, train), (OUT_VALID_XYZ, valid)):
        write(str(path), configs, format="extxyz", write_results=False)
        print(f"Wrote {len(configs)} configurations to {path}")


if __name__ == "__main__":
    collect()
