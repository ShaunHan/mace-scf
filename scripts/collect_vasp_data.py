from pathlib import Path
import json
import multiprocessing as mp

import h5py
import numpy as np
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Locpot, Outcar


BASE_PATH = Path("/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft/")
PROTO_BASE_PATH = BASE_PATH.with_name("dft_proto")
CHUNK_SIZE = 5
N_PROCS = 64
REUSE_HDF = False
K_CUTOFF = 18.0  # HDF bandwidth in inverse Angstrom.
FOURIER_CUTOFF_FACTOR = 1.25  # XYZ bandwidth; must cover the model's k-space cutoff.
DENSITY_SIGMAS = [1.5]
POTENTIAL_SIGMAS = [1.5]  # [] disables potential collection.
FMAX_TOL = 6.0
PLUGIN_FORCE_TOL = 1e-3  # eV/A: plug-in and VASP forces may differ only by a uniform net force.
VALID_FRACTION = 0.2
SPLIT_SEED = 7777
SKIP_IDS = [279, 1668]  # Original VASP IDs: batch start + output index.
VACUUM_PLANE = {}  # e.g. {"vacuum_zfrac": 0.75}; empty selects a vacuum plane.
PDOS_RANGE = 5.0  # eV: each atom's projected DOS is kept within EF +- PDOS_RANGE.
PDOS_STEP = 0.05  # eV: DOS histogram bins, centered on EF + j * PDOS_STEP.
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
        "energy_minus_EF": energies, "occupation": occupations,
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
            states["energy_minus_EF"][row] = energy
            states["occupation"][row] = occupation
            # Histogram: each band adds its orbital-summed projections to the bin of its eigenvalue.
            bins = np.rint(energy / PDOS_STEP).astype(int)
            bands = np.flatnonzero(abs(bins) <= half)
            if len(bands):
                site = projected[k.s, :, :, k.k - 1, bands].sum(axis=1)
                pdos += k.weight * capacity / PDOS_STEP * site @ (bins[bands, None] == offsets)
    states.attrs.update(projections_available=True,
                        projection_units="PAW projected states per eV; not atomic charges")
    states.create_dataset("pdos_energy_minus_EF", data=offsets * PDOS_STEP)
    states.create_dataset("pdos", data=pdos.astype(np.float32))
    # Geometry identifies electrode/solvent atoms without hard-coded species rules.
    for name, values in {"atomic_numbers": atoms.numbers, "positions": atoms.positions,
                         "cell": atoms.cell.array}.items():
        states.create_dataset(name, data=values)
    return states["pdos_energy_minus_EF"][...], states["pdos"][...]


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
                data["pdos"] = states["pdos_energy_minus_EF"][...], states["pdos"][...]
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
        # Neither potential is shifted: VASP writes both with zero mean, the origin of vacuum_potential.
        potential -= proto
        data["fourier_potential"] = truncate_fft(np.fft.fftn(potential), shape)
    temporary = path.with_suffix(".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["k_cutoff"] = K_CUTOFF
        handle.attrs["nelect"] = parameters["nelect"]
        handle.attrs["fermi_energy"] = atoms.info["vasp_efermi"]
        handle.attrs["counter_charge"] = atoms.info["counter_charge"]
        handle.attrs["electronic_converged"] = atoms.info["electronic_converged"]
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


def add_vacuum_target(atoms, planar, axis, correction):
    key = f"vacuum_{'xyz'[axis]}frac"
    if VACUUM_PLANE and set(VACUUM_PLANE) != {key}:
        raise ValueError(f"For this VASP IDIPOL, specify only {key}")
    if VACUUM_PLANE:
        fraction = float(VACUUM_PLANE[key])
    else:
        distances = (correction - atoms.get_scaled_positions()[:, axis]) % 1.0
        fraction = (correction - 0.5 * np.min(distances[distances > 1e-8])) % 1.0
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("Vacuum fraction must be in [0, 1)")
    coordinate = fraction * len(planar)
    index = int(np.floor(coordinate))
    weight = coordinate - index
    atoms.info["vacuum_potential"] = float(
        (1 - weight) * planar[index] + weight * planar[(index + 1) % len(planar)])
    atoms.info[key] = fraction
    atoms.info[f"dipole_correction_{'xyz'[axis]}frac"] = correction


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


def plugin_forces(outdir, forces):
    """Forces of a plug-in run with the net force of its counter charge, in VASP's atom order.

    VASP removes the net force from the forces it writes. A fixed counter charge exerts a
    real one, which the plug-in keeps as expected_forces; the two sets may differ only by it.
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
    return full


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
    # Without a plug-in there is no external source, and both force sets agree.
    forces_with_net_force = plugin_forces(outdir, forces) if plugin else forces
    if max(np.linalg.norm(f, axis=1).max() for f in (forces, forces_with_net_force)) > FMAX_TOL:
        print(f"Skipping {outdir}: force exceeds FMAX_TOL")
        return None

    outcar = Outcar(str(outdir / "OUTCAR"))
    outcar.read_pseudo_zval()
    charge = sum(outcar.zval_dict[s] for s in atoms.get_chemical_symbols()) - parameters["nelect"]
    fermi = float(outcar.efermi)
    if not np.isfinite(charge) or not np.isfinite(fermi):
        raise ValueError(f"Nonfinite charge/Fermi level: {outdir}")
    text = (outdir / "OUTCAR").read_text(errors="replace")
    converged = "EDIFF is reached" in text.rsplit("Iteration", 1)[-1]

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
    atoms.arrays["vasp_forces_with_net_force"] = forces_with_net_force
    atoms.info = {
        "config_type": "slab" if slab else "bulk",
        "vasp_free_energy": float(energy),
        "vasp_efermi": fermi,
        "total_charge": float(charge),
        "counter_charge": float(counter),
        "electronic_converged": converged,
        "external_field": external_field,
        "vasp_dipole": dipole,
        "config_dipole_weight": np.eye(3)[axis] if slab else np.zeros(3),
        "config_fermi_level_weight": float(slab),
        "config_vacuum_potential_weight": float(slab and bool(POTENTIAL_SIGMAS)),
    }
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
        atoms.info["vasp_pdos_energy_minus_EF"] = np.round(energy, 3)
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
            add_vacuum_target(atoms, data["planar_potential"], axis, correction)
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


if __name__ == "__main__":
    collect()
