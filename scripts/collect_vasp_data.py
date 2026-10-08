"""Collect training labels directly from VASP outputs. Edit the settings below."""
from pathlib import Path
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
K_CUTOFF = 18.0
FOURIER_CUTOFF_FACTOR = 1.25
DENSITY_SIGMAS = [1.5]
POTENTIAL_SIGMAS = [1.5]  # [] disables potential collection.
FMAX_TOL = 6.0
VALID_FRACTION = 0.2
SPLIT_SEED = 7777
SKIP_IDS = [279, 1668]  # Original VASP IDs: batch start + output index.
VACUUM_PLANE = {}  # e.g. {"vacuum_zfrac": 0.75}; empty selects a vacuum plane.
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


def fourier_data(outdir, atoms, parameters, axis):
    """Cache unsmoothed Fourier grids; widths and vacuum sampling remain editable."""
    names = ["fourier_density"] if DENSITY_SIGMAS else []
    if POTENTIAL_SIGMAS:
        names += ["fourier_potential", "proto_fourier_potential"]
        if axis is not None:
            names += ["planar_potential"]
    if not names:
        return {}
    path = outdir / HDF_NAME
    if REUSE_HDF and path.is_file():
        with h5py.File(path, "r") as handle:
            if handle.attrs["k_cutoff"] != K_CUTOFF or not set(names) <= set(handle):
                raise ValueError(f"Cache settings differ in {path}; set REUSE_HDF=False")
            return {name: handle[name][...] for name in names}

    shape = 2 * np.ceil(K_CUTOFF * atoms.cell.lengths() / (2 * np.pi)).astype(int)
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
    if POTENTIAL_SIGMAS:
        if not parameters["lvhar"]:
            raise ValueError(f"Potential collection needs LVHAR=True: {outdir}")
        potential = read_grid(outdir / "LOCPOT", atoms, Locpot)
        proto_dir = PROTO_BASE_PATH / outdir.relative_to(BASE_PATH)
        proto = read_grid(proto_dir / "LOCPOT", atoms, Locpot)
        if potential.shape != proto.shape:
            raise ValueError(f"SCF/proto potential grid mismatch: {outdir}")
        if axis is not None:
            data["planar_potential"] = potential.mean(axis=tuple(i for i in range(3) if i != axis))
        potential -= proto  # No mean subtraction; retain the potential zero mode.
        data["fourier_potential"] = truncate_fft(np.fft.fftn(potential), shape)
        data["proto_fourier_potential"] = truncate_fft(np.fft.fftn(proto), shape)
    temporary = path.with_suffix(".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["k_cutoff"] = K_CUTOFF
        for name, values in data.items():
            handle.create_dataset(name, data=values)
    temporary.replace(path)
    return data


def add_fourier_targets(atoms, coefficients, sigmas, prefix, charge=None):
    """Keep the existing Gaussian widths, FFT normalization and target names."""
    reciprocal = 2 * np.pi * np.linalg.inv(atoms.cell.array).T
    for sigma in sigmas:
        cutoff = FOURIER_CUTOFF_FACTOR * 0.75 * 3.0 * 2**0.3 / float(sigma)
        shape = 2 * np.ceil(cutoff * atoms.cell.lengths() / (2 * np.pi)).astype(int)
        target = truncate_fft(coefficients, shape)
        axes = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in shape]
        wavevectors = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1) @ reciprocal
        target *= np.exp(-0.5 * float(sigma)**2 * np.sum(wavevectors**2, axis=-1))
        if charge is not None:
            target[0, 0, 0] = charge * np.prod(shape) / atoms.get_volume()
        target = hermitian(target)
        atoms.info[f"{prefix}_fftn_shifted_{float(sigma)}"] = np.stack(
            [target.real, target.imag]).astype(np.float32)


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


def collect_job(outdir):
    atoms = read(str(outdir / "vasprun.xml"), format="vasp-xml", index=-1)
    parameters = atoms.calc.parameters
    forces = atoms.get_forces(apply_constraint=False)
    energy = atoms.get_potential_energy(force_consistent=True)
    if not np.isfinite(forces).all() or not np.isfinite(energy):
        raise ValueError(f"Nonfinite energy/forces: {outdir}")
    if np.linalg.norm(forces, axis=1).max() > FMAX_TOL:
        print(f"Skipping {outdir}: force exceeds FMAX_TOL")
        return None

    outcar = Outcar(str(outdir / "OUTCAR"))
    outcar.read_pseudo_zval()
    charge = sum(outcar.zval_dict[s] for s in atoms.get_chemical_symbols()) - outcar.nelect
    fermi = float(outcar.efermi)
    if not np.isfinite(charge) or not np.isfinite(fermi):
        raise ValueError(f"Nonfinite charge/Fermi level: {outdir}")

    slab = bool(parameters["ldipol"])
    axis = int(parameters["idipol"]) - 1 if slab else None
    external_field = np.zeros(3)
    dipole = np.zeros(3)
    if slab:
        if axis not in (0, 1, 2):
            raise ValueError(f"Slab collection needs IDIPOL=1, 2 or 3: {outdir}")
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        external_field = -float(parameters["efield"]) * normal
        dipole = -np.asarray(atoms.get_dipole_moment(), dtype=float).reshape(3)
        if not np.isfinite(dipole).all():
            raise ValueError(f"Nonfinite dipole: {outdir}")
        atoms.pbc[axis] = False

    atoms.calc = None
    atoms.set_constraint()
    atoms.arrays["vasp_forces"] = forces
    atoms.info = {
        "config_type": "slab" if slab else "bulk",
        "vasp_free_energy": float(energy),
        "vasp_efermi": fermi,
        "total_charge": float(charge),
        "external_field": external_field,
        "vasp_dipole": dipole,
        "config_dipole_weight": np.eye(3)[axis] if slab else np.zeros(3),
        "config_fermi_level_weight": float(slab),
        "config_vacuum_potential_weight": float(slab and bool(POTENTIAL_SIGMAS)),
    }
    data = fourier_data(outdir, atoms, parameters, axis)
    if DENSITY_SIGMAS:
        add_fourier_targets(atoms, data["fourier_density"], DENSITY_SIGMAS, "vasp_rho", charge)
    if POTENTIAL_SIGMAS:
        if slab:
            if np.all(np.asarray(parameters["dipol"]) == -100):
                raise ValueError(f"Specify DIPOL in VASP to locate its correction plane: {outdir}")
            correction = float((parameters["dipol"][axis] + 0.5) % 1.0)
            add_vacuum_target(atoms, data["planar_potential"], axis, correction)
        add_fourier_targets(atoms, data["fourier_potential"], POTENTIAL_SIGMAS, "vasp_phi")
        add_fourier_targets(atoms, data["proto_fourier_potential"], POTENTIAL_SIGMAS, "vasp_proto_phi")
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
