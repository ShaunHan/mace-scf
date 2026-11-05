from pathlib import Path
import json
import multiprocessing as mp
import re
import sys

import h5py
import numpy as np
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Locpot, Outcar


BASE_PATH = Path("/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft_plugin/")
PROTO_BASE_PATH = None
CHUNK_SIZE = 5
N_PROCS = 64
REUSE_HDF = True
K_CUTOFF = 18.0  # HDF bandwidth in inverse Angstrom.
FOURIER_CUTOFF_FACTOR = 1.25  # XYZ bandwidth; must cover the model's k-space cutoff.
DENSITY_SIGMAS = [1.5]
POTENTIAL_SIGMAS = [1.5]  # [] disables potential collection.
FMAX_TOL = 6.0
PLUGIN_FORCE_TOL = 1e-3  # eV/A: plug-in and VASP forces may differ only by a uniform net force.
VALID_FRACTION = 0.2
SPLIT_SEED = 7777
SKIP_IDS = [1, 7, 10, 46, 66, 97, 101, 115, 194, 218, 228, 266, 288, 296, 308, 318, 342, 353, 375, 391, 420, 439, 506, 555, 566, 755]  # Original VASP IDs: batch start + output index.
VACUUM_MARGIN = 3.0  # A: vacuum planes start this far above the outermost atom.
VACPOTFLAT = 0.1  # eV/A: largest planar field of a vacuum plane, applied field removed, as LVACPOTAV.
VACUUM_FLATNESS = 0.2  # V: a vacuum potential that varies more than this is not a converged density.
SCF_DENSITY_TOL = 1e-2  # Largest rms(c) of the last SCF steps; VASP stops at EDIFF on one small energy change.
DIPOLE_DRIFT_TOL = 0.05  # V: largest vacuum potential change from the plug-in dipole over its last callbacks.
PDOS_RANGE = 5.0  # eV: each atom's projected DOS is kept within EF +- PDOS_RANGE.
PDOS_STEP = 0.05  # eV: DOS histogram bins, centered on EF + j * PDOS_STEP.
BAND_GAP_MIN = 0.1  # eV: a neutral cell with a larger DFT gap is labelled by its gap too.
OCCUPATION_TOL = 0.01  # A state filled by more than this and less than 1 - this is in a band at the Fermi level.
FERMI_LEVEL_TOL = 0.1  # eV: largest difference of VASP's Fermi level of a gapped cell from its intrinsic level.
HDF_NAME = "fourier_data.hdf"

OUT_XYZ = BASE_PATH / "combined_VASP_shifted.xyz"
OUT_TRAIN_XYZ = BASE_PATH / "combined_VASP_shifted_train.xyz"
OUT_VALID_XYZ = BASE_PATH / "combined_VASP_shifted_valid.xyz"

COUNTER_CHARGE_PLACES = ("counter_charge_center", "counter_charge_xfrac", "counter_charge_yfrac",
                         "counter_charge_zfrac")


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


def add_electronic_diagnostics(handle, outdir, atoms, parameters, kpoints):
    """Small band arrays plus each atom's projected DOS around EF.

    Returns the DOS energies relative to EF and the per-atom DOS in states/eV, or None
    without projections in vaspout.h5.
    """
    if not kpoints:
        return None
    states = handle.create_group("states")
    spinor = parameters.get("lnoncollinear", False) or parameters.get("lsorbit", False)
    occupations = [np.asarray(k.f_n) * (0.5 if spinor else 1.0) for k in kpoints]
    energies = [np.asarray(k.eps_n) - atoms.info["vasp_efermi"] for k in kpoints]
    for name, values in {
        "energy_minus_fermi": energies, "occupation": occupations,
        "kpoint_weight": [k.weight for k in kpoints],
        "spin": [k.s for k in kpoints], "kpoint": [k.k for k in kpoints],
    }.items():
        states.create_dataset(name, data=values)
    states.attrs["projections_available"] = False
    path = outdir / "vaspout.h5"
    if not path.is_file():
        return None  # Do not rescan large PROCAR/XML files for optional projections.
    with h5py.File(path, "r") as source:
        if "results/projectors/par" not in source:
            return None
        projected = source["results/projectors/par"]
        if projected.shape[1] != len(atoms):
            raise ValueError(f"Projection atom count mismatch: {path}")
        half = round(PDOS_RANGE / PDOS_STEP)
        offsets = np.arange(-half, half + 1)
        pdos = np.zeros((len(atoms), len(offsets)))
        capacity = 1.0 if spinor or parameters.get("ispin", 1) == 2 else 2.0
        for row, k in enumerate(kpoints):
            # ASE numbers k-points from 1 and reads the XML table rounded to four decimal places.
            saved = source["results/electron_eigenvalues/eigenvalues"][k.s, k.k - 1]
            if not np.allclose(saved, k.eps_n, rtol=0, atol=5.1e-5):
                raise ValueError(f"Projection eigenvalues do not match this VASP run: {path}")
            energy = saved - atoms.info["vasp_efermi"]
            occupation = capacity * source["results/electron_eigenvalues/fermiweights"][k.s, k.k - 1]
            states["energy_minus_fermi"][row] = energy
            states["occupation"][row] = occupation
            # Histogram: each band adds its orbital-summed projections to the bin of its eigenvalue.
            bins = np.rint(energy / PDOS_STEP).astype(int)
            bands = np.flatnonzero(abs(bins) <= half)
            if len(bands):
                site = projected[k.s, :, :, k.k - 1, bands].sum(axis=1)
                pdos += k.weight * capacity / PDOS_STEP * site @ (bins[bands, None] == offsets)
    states.attrs.update(projections_available=True,
                        projection_units="PAW projected states per eV; not atomic charges")
    states.create_dataset("pdos_energy_minus_fermi", data=offsets * PDOS_STEP)
    states.create_dataset("pdos", data=pdos.astype(np.float32))
    # Geometry identifies electrode/solvent atoms without hard-coded species rules.
    for name, values in {"atomic_numbers": atoms.numbers, "positions": atoms.positions,
                         "cell": atoms.cell.array}.items():
        states.create_dataset(name, data=values)
    return states["pdos_energy_minus_fermi"][...], states["pdos"][...]


def fourier_data(outdir, atoms, parameters, axis, plugin=False, kpoints=()):
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
            data = {name: handle[name][...] for name in names}
            states = handle.get("states")
            if states is not None and states.attrs.get("projections_available", False):
                if "pdos" not in states:
                    raise ValueError(f"No projected DOS in {path}; set REUSE_HDF=False")
                data["pdos"] = states["pdos_energy_minus_fermi"][...], states["pdos"][...]
            return data

    shape = 2 * np.ceil(K_CUTOFF * atoms.cell.lengths() / (2 * np.pi)).astype(int) + 1
    data = {}
    if DENSITY_SIGMAS:
        density = read_grid(outdir / "AECCAR1", atoms, Chgcar)
        final_density = read_grid(outdir / "AECCAR2", atoms, Chgcar)
        if density.shape != final_density.shape:
            raise ValueError(f"AECCAR grid mismatch: {outdir}")
        if axis is not None:
            data["planar_valence_density"] = final_density.mean(
                axis=tuple(i for i in range(3) if i != axis)) / atoms.get_volume()
        density -= final_density
        density /= atoms.get_volume()  # Signed deformation charge, in e/Angstrom^3.
        data["fourier_density"] = truncate_fft(np.fft.fftn(density), shape)
        del density, final_density
    if plugin:
        # Only plugin runs need this extra grid, to recover their physical dipole.
        data["planar_electron_density"] = read_grid(outdir / "CHGCAR", atoms, Chgcar).mean(
            axis=tuple(i for i in range(3) if i != axis)) / atoms.get_volume()
    if POTENTIAL_SIGMAS:
        if not parameters["lvhar"]:
            raise ValueError(f"Potential collection needs LVHAR=True: {outdir}")
        potential = read_grid(outdir / "LOCPOT", atoms, Locpot)
        # The in-run proto is VASP's first ICHARG=2 density, the neutral-atom
        # superposition for every charge state. A separate proto run is only a
        # fallback for old neutral outputs; it would carry a charged run's NELECT.
        proto_path = outdir / "LOCPOT_proto"
        if not proto_path.is_file():
            if plugin:
                raise ValueError(f"Charged plug-in run without its in-run LOCPOT_proto: {outdir}")
            proto_path = PROTO_BASE_PATH / outdir.relative_to(BASE_PATH) / "LOCPOT"
        proto = read_grid(proto_path, atoms, Locpot)
        if potential.shape != proto.shape:
            raise ValueError(f"SCF/proto potential grid mismatch: {outdir}")
        # The neutral atoms: the vacuum origin and the potential a counter charge overlaps.
        data["fourier_proto_potential"] = truncate_fft(np.fft.fftn(proto), shape)
        if axis is not None:
            data["planar_potential"] = potential.mean(axis=tuple(i for i in range(3) if i != axis))
        # Neither potential is shifted: VASP writes both with zero mean, the origin of its Fermi level.
        potential -= proto
        data["fourier_potential"] = truncate_fft(np.fft.fftn(potential), shape)
    temporary = path.with_suffix(".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["k_cutoff"] = K_CUTOFF
        handle.attrs["nelect"] = parameters["nelect"]
        handle.attrs["fermi_energy"] = atoms.info["vasp_efermi"]
        handle.attrs["counter_charge"] = atoms.info["counter_charge"]
        handle.attrs["potential_reference"] = "LOCPOT minus proto LOCPOT"
        if axis is not None:
            handle.attrs["profile_axis"] = axis
        pdos = add_electronic_diagnostics(handle, outdir, atoms, parameters, kpoints)
        for name, values in data.items():
            handle.create_dataset(name, data=values)
    temporary.replace(path)
    if pdos is not None:
        data["pdos"] = pdos
    return data


def add_fourier_targets(atoms, coefficients, sigmas, prefix, charge=None):
    """Compact, unsmoothed XYZ targets; higher-frequency data stay in the HDF."""
    # The widths set the bandwidth only; they apply no Gaussian filter. The model's k-space
    # cutoff is 0.75 * 3.0 * (max_l + 1)**0.3 / sigma (graph_longrange gto_basis_kspace_cutoff,
    # max_l = 1), and the targets keep FOURIER_CUTOFF_FACTOR times it.
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
    """The vacuum potential above the slab, just below the dipole correction plane, and its spread.

    LVACPOTAV averages the planar potential over the field-free vacuum. The vacuum planes here lie between
    VACUUM_MARGIN above the outermost atom and the correction plane, where the field, the applied field
    removed, is below VACPOTFLAT. The applied field raises the potential by the field per A towards the plane,
    so V(z) + field * (distance to the plane) is the same at every vacuum plane, the value just below it. A
    density that has not settled leaves a uniform charge in the vacuum, which bends that average by the spread
    returned with it. Training subtracts the applied field's zero-mean sawtooth at the plane, field * L / 2,
    and fits the field-free vacuum potential.
    """
    normal = np.linalg.inv(atoms.cell.array)[:, axis]
    length = 1.0 / np.linalg.norm(normal)
    below = (correction - np.arange(len(planar)) / len(planar)) % 1.0 * length
    top = np.min((correction - atoms.get_scaled_positions(wrap=False)[:, axis]) % 1.0) * length
    level = planar + field * below
    gradient = (np.roll(level, -1) - np.roll(level, 1)) / (2 * length / len(planar))
    vacuum = (below > length / len(planar)) & (below < top - VACUUM_MARGIN) & (np.abs(gradient) < VACPOTFLAT)
    if not vacuum.any():
        raise ValueError("No field-free vacuum above the slab; enlarge the vacuum")
    return float(level[vacuum].mean()), float(np.ptp(level[vacuum]))


def converged_electronic_state(outdir, text, plugin, area):
    """Why the electronic state of a run is not converged, or None.

    VASP stops at EDIFF when one energy change is small, which a stalled density can pass by chance: the
    density residual rms(c) of the last steps and, in a plug-in run, the dipole of its last callbacks must
    have settled too.
    """
    if "EDIFF is reached" not in text.rsplit("Iteration", 1)[-1]:
        return "EDIFF not reached"
    residuals = []
    for line in (outdir / "OSZICAR").read_text().splitlines():
        numbers = line.split()
        if len(numbers) == 8 and numbers[0][-1] == ":":
            residuals.append(float(numbers[7]))
    if residuals and max(residuals[-3:]) > SCF_DENSITY_TOL:
        return f"density residual rms(c) {max(residuals[-3:]):.1e}"
    if plugin and (outdir / "plugin_scf.csv").is_file():
        dipoles = np.loadtxt(outdir / "plugin_scf.csv", delimiter=",", skiprows=1, ndmin=2)[-6:, 1]
        drift = np.ptp(dipoles) / (0.005526349406 * area)
        if drift > DIPOLE_DRIFT_TOL:
            return f"plug-in dipole drifts by {np.ptp(dipoles):.1e} eA, {drift:.3f} V of vacuum potential"
    return None


def plugin_run(outdir, meta):
    """Whether a counter-charge plug-in acted in this run.

    scripts/dft_runs.py records it in dft_metadata.json, together with any placed counter
    charge (Q = 0 included), and installs its proto-potential hook vasp_plugin.py in every
    run, so that file alone proves nothing. Without metadata, only the plug-in's own output
    marks a plug-in run, which then lacks the settings it needs.
    """
    if "plugin" in meta:
        return bool(meta["plugin"])
    if (outdir / "plugin_observables.npz").is_file():
        raise ValueError(f"Plug-in run without dft_metadata.json: {outdir}")
    return False


def plugin_forces(outdir, forces, normal):
    """The forces of a plug-in run, -dE/dR of its energy, in VASP's atom order.

    VASP removes the net force from the forces it writes. A planar counter charge exerts a real
    one along the slab normal, which the plug-in keeps in expected_forces; the two sets may differ
    only by a uniform force. Its part in the plane of the slab, which a planar counter charge
    cannot exert, is VASP's numerical drift and stays removed, as in every run.
    """
    with np.load(outdir / "plugin_observables.npz") as observables:
        if "expected_forces" not in observables:
            raise ValueError(f"Plug-in run without its forces: {outdir}")
        full = np.asarray(observables["expected_forces"], dtype=float)
    if full.shape != forces.shape or not np.isfinite(full).all():
        raise ValueError(f"Invalid plug-in forces: {outdir}")
    difference = full - forces
    if np.abs(difference - difference.mean(0)).max() > PLUGIN_FORCE_TOL:
        raise ValueError(f"Plug-in and VASP forces differ by more than a net force: {outdir}")
    return forces + (difference.mean(0) @ normal) * normal


def band_edges(kpoints, sigma):
    """The valence band maximum, conduction band minimum and intrinsic Fermi level over all k points and spins, or None.

    None also when a state is partly filled: the Fermi level then lies in a band, as in a metal or a half-filled
    level, which the half occupation of a filled state would count as empty. The fullest state sets the occupation
    of a filled one, so that ISPIN, spinors and ASE's doubled occupations of spin-degenerate runs need no special case.
    The intrinsic level is where the Fermi-Dirac tails of width sigma hold as many holes in the valence band as
    electrons in the conduction band: exp(2 mu / sigma) = sum_v w exp(e / sigma) / sum_c w exp(-e / sigma).
    """
    if not kpoints:
        return None
    full = max(float(np.max(k.f_n)) for k in kpoints)
    energies = np.concatenate([np.asarray(k.eps_n) for k in kpoints])
    weights = np.concatenate([np.full(len(k.eps_n), np.log(k.weight)) for k in kpoints])
    filling = np.concatenate([np.asarray(k.f_n) for k in kpoints]) / full
    occupied = filling > 0.5
    if np.abs(filling - occupied).max() > OCCUPATION_TOL or occupied.all() or not occupied.any():
        return None
    intrinsic = 0.5 * sigma * (np.logaddexp.reduce(weights[occupied] + energies[occupied] / sigma)
                               - np.logaddexp.reduce(weights[~occupied] - energies[~occupied] / sigma))
    return float(energies[occupied].max()), float(energies[~occupied].min()), float(intrinsic)


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
        # The counter charge exerts a net force along the slab normal, which VASP removed.
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
    # A neutral cell with a gap, bulk or slab such as a water film, is labelled by its gap, and by its Fermi level
    # where that is defined. With Fermi-Dirac occupations (ISMEAR = -1) it is the intrinsic level, where the few
    # holes of the valence band equal the few electrons of the conduction band, and VASP finds it unless a wide
    # gap leaves those tails below double precision. Other smearings, and such a gap, leave it wherever VASP's
    # search stops. A metal, or a charged cell with the Fermi level in a band, has a zero gap.
    edges = band_edges(kpoints, float(parameters["sigma"]))
    gapped = abs(charge) < 1e-6 and edges is not None and edges[1] - edges[0] > BAND_GAP_MIN
    intrinsic = gapped and parameters["ismear"] == -1 and abs(fermi - edges[2]) < FERMI_LEVEL_TOL
    atoms.info = {
        "config_type": "slab" if slab else "bulk",
        "vasp_free_energy": float(energy),
        "vasp_efermi": fermi,
        "vasp_band_gap": edges[1] - edges[0] if gapped else 0.0,
        "total_charge": float(charge),
        "counter_charge": float(counter),
        "external_field": external_field,
        "vasp_dipole": dipole,
        "config_dipole_weight": np.eye(3)[axis] if slab else np.zeros(3),
        "config_fermi_level_weight": float(intrinsic or not gapped),
        "config_band_gap_weight": float(gapped),
    }
    if gapped:
        # The band edges in the origin of vasp_efermi.
        atoms.info["vasp_valence_band_maximum"], atoms.info["vasp_conduction_band_minimum"] = edges[:2]
    if slab:
        atoms.info[f"dipole_correction_{'xyz'[axis]}frac"] = correction
    places = [key for key in COUNTER_CHARGE_PLACES if key in meta]
    if places:
        # The Gaussian counter charge as placed, also at Q = 0.
        atoms.info["counter_charge_width"] = float(meta["counter_charge_width"])
        for key in places:
            atoms.info[key] = np.asarray(meta[key], dtype=float) if key.endswith("center") else float(meta[key])
    if plugin:
        # The plug-in implements forces, but not its stress derivative.
        atoms.info["config_stress_weight"] = 0.
        atoms.info["config_virials_weight"] = 0.
    data = fourier_data(outdir, atoms, parameters, axis, plugin, kpoints)
    if "pdos" in data:
        # One row per atom, in states/eV rounded to 0.01. As a JSON array in the header this
        # takes about 6 characters per number; per-atom columns would take 17.
        energy, pdos = data["pdos"]
        atoms.info["vasp_pdos_energy_minus_fermi"] = np.round(energy, 3)
        atoms.info["vasp_pdos"] = np.round(np.asarray(pdos, dtype=float), 2)
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
            if spread > VACUUM_FLATNESS:
                print(f"Skipping {outdir}: the vacuum potential varies by {spread:.3f} V")
                return None
            if spread > 0.05:
                print(f"Note {outdir}: the vacuum potential varies by {spread:.3f} V; kept at its average")
            vasp = re.findall(r"vacuum level on the upper side and lower side of the slab\s+([-+.\d]+)", text)
            if vasp and not external_field.any() and abs(float(vasp[-1]) - level) > 0.02:
                print(f"Warning {outdir}: vacuum level {level:.3f} V, VASP's LVACPOTAV {float(vasp[-1]):.3f} V")
            atoms.info["vasp_vacuum_potential"] = level
        add_fourier_targets(atoms, data["fourier_potential"], POTENTIAL_SIGMAS, "vasp_phi")
        # The neutral atoms: unsmoothed LOCPOT_proto modes, as vasp_proto_phi_fftn_shifted.
        add_fourier_targets(atoms, data["fourier_proto_potential"], POTENTIAL_SIGMAS, "vasp_proto_phi")
    print(f"Collected {outdir}")
    return atoms, outdir.parent.name


def split_configs(records):
    """Keep each VASP batch in one split, without storing group metadata in XYZ."""
    groups = {}
    for atoms, group in records:
        groups.setdefault(group, set()).add(atoms.info["config_type"])
    strata = {}
    for group, types in groups.items():
        strata.setdefault(tuple(sorted(types)), []).append(group)
    rng = np.random.default_rng(SPLIT_SEED)
    valid_groups = set()
    for groups in strata.values():
        groups = np.asarray(sorted(groups), dtype=object)
        rng.shuffle(groups)
        if len(groups) > 1:
            count = min(max(round(VALID_FRACTION * len(groups)), 1), len(groups) - 1)
            valid_groups.update(groups[:count])
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
    train, valid = split_configs(records)
    for path, configs in ((OUT_XYZ, [a for a, _ in records]),
                          (OUT_TRAIN_XYZ, train), (OUT_VALID_XYZ, valid)):
        write(str(path), configs, format="extxyz", write_results=False)
        print(f"Wrote {len(configs)} configurations to {path}")


def relabel(paths):
    """vasp_vacuum_potential, the vacuum potential just below the plane, in the xyz files of earlier collections.

    It is vasp_workfunction plus the Fermi level, or vacuum_potential at vacuum_zfrac carried to the plane by the
    applied field. Writes NAME_vac.xyz next to each file.
    """
    pairs = re.compile(r'(?:^|\s)(\w+)=("[^"]*"|\S+)')
    for path in map(Path, paths):
        target = path.with_name(path.stem + "_vac" + path.suffix)
        with open(path, encoding="utf-8", newline="") as source, open(target, "w", encoding="utf-8", newline="") as out:
            while True:
                first = source.readline()
                if not first:
                    break
                if not first.strip():
                    continue
                header = source.readline()
                info = {key: value.strip('"') for key, value in pairs.findall(header)
                        if key in ("Lattice", "pbc", "external_field", "vasp_efermi", "vasp_workfunction",
                                   "vacuum_potential") or key.startswith(("vacuum_", "dipole_correction_"))}
                level = None
                if "vasp_workfunction" in info:
                    level = float(info["vasp_workfunction"]) + float(info["vasp_efermi"])
                elif "vacuum_potential" in info:
                    cell = np.array(info["Lattice"].split(), float).reshape(3, 3)
                    axis = [x == "T" for x in info["pbc"].split()].index(False)
                    normal = np.linalg.inv(cell)[:, axis]
                    normal /= np.linalg.norm(normal)
                    field = np.array(info["external_field"].split(), float) @ normal
                    plane = float(info[f"dipole_correction_{'xyz'[axis]}frac"])
                    below = (plane - float(info[f"vacuum_{'xyz'[axis]}frac"])) % 1.0 * float(cell[axis] @ normal)
                    level = float(info["vacuum_potential"]) + field * below
                header = re.sub(r'(?:^|\s)(vasp_workfunction|vacuum_potential|vacuum_[xyz]frac|'
                                r'electronic_converged)=("[^"]*"|\S+)', "", header)
                if level is not None:
                    header = header.rstrip("\r\n") + f" vasp_vacuum_potential={level:.8f}\n"
                out.write(first + header + "".join(source.readline() for _ in range(int(first))))
        print(f"Wrote {target}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        relabel(sys.argv[1:])
    else:
        collect()
