"""Collect training labels directly from VASP outputs. Edit the settings below."""
from pathlib import Path
import json
import multiprocessing as mp

import h5py
import numpy as np
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Locpot, Outcar


BASE_PATH = Path("/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft/")
N_PROCS = 64
REUSE_HDF = False
K_CUTOFF = 3.0  # Å^-1; retained bandwidth, independent of training smoothing.
COLLECT_DENSITY = True
COLLECT_POTENTIAL = True
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
    paths = []
    for folder in folders:
        outputs = sorted((p for p in folder.glob("output_*") if p.is_dir()),
                         key=lambda p: int(p.name.split("_")[1]))
        for outdir in outputs:
            meta = json.loads((outdir / "dft_metadata.json").read_text())
            if int(folder.name.split("_")[1]) + meta["source_index"] not in SKIP_IDS:
                paths.append(outdir)
    return paths


def read_grid(path, reader, atoms):
    """Refuse volumetric labels belonging to a different atomic structure."""
    grid = reader.from_file(str(path))
    structure = grid.structure
    delta = structure.frac_coords-atoms.get_scaled_positions(wrap=False)
    if ([str(site.specie) for site in structure] != atoms.get_chemical_symbols()
            or not np.allclose(structure.lattice.matrix, atoms.cell, rtol=0, atol=1.e-4)
            or not np.allclose(delta-np.rint(delta), 0., rtol=0, atol=1.e-5)):
        raise ValueError(f'Structure/grid mismatch: {path}')
    values = np.asarray(grid.data['total'], dtype=float)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError(f'Invalid volumetric data: {path}')
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
    names = ["fourier_density"] if COLLECT_DENSITY else []
    if COLLECT_POTENTIAL:
        names += ["fourier_potential", "proto_fourier_potential"]
        if axis is not None:
            names += ["planar_potential"]
    if not names:
        return {}
    path = outdir / HDF_NAME
    if REUSE_HDF and path.is_file():
        with h5py.File(path, "r") as handle:
            if (handle.attrs.get("schema") != "raw_same_run_v2"
                    or handle.attrs["k_cutoff"] != K_CUTOFF or not set(names) <= set(handle)):
                raise ValueError(f"Cache settings differ in {path}; set REUSE_HDF=False")
            return {name: handle[name][...] for name in names}

    shape = 2 * np.ceil(K_CUTOFF * atoms.cell.lengths() / (2 * np.pi)).astype(int)
    data = {}
    if COLLECT_DENSITY:
        density = read_grid(outdir / "AECCAR1", Chgcar, atoms)
        final_density = read_grid(outdir / "AECCAR2", Chgcar, atoms)
        if density.shape != final_density.shape:
            raise ValueError(f'AECCAR grid mismatch: {outdir}')
        density -= final_density
        density /= atoms.get_volume()  # Signed deformation charge, in e/Angstrom^3.
        data["fourier_density"] = truncate_fft(np.fft.fftn(density), shape)
        del density, final_density
    if COLLECT_POTENTIAL:
        if not parameters["lvhar"]:
            raise ValueError(f"Potential collection needs LVHAR=True: {outdir}")
        # Native LOCPOT includes the plugin; the old callback-exported
        # LOCPOT_total could capture a trial electronic state.
        potential = read_grid(outdir / 'LOCPOT', Locpot, atoms)
        proto = read_grid(outdir / "LOCPOT_proto", Locpot, atoms)
        if potential.shape != proto.shape:
            raise ValueError(f'SCF/proto potential grid mismatch: {outdir}')
        if axis is not None:
            data["planar_potential"] = potential.mean(axis=tuple(i for i in range(3) if i != axis))
        potential -= proto  # No mean subtraction; retain the potential zero mode.
        data["fourier_potential"] = truncate_fft(np.fft.fftn(potential), shape)
        data["proto_fourier_potential"] = truncate_fft(np.fft.fftn(proto), shape)
    temporary = path.with_suffix(".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["schema"] = "raw_same_run_v2"
        handle.attrs["k_cutoff"] = K_CUTOFF
        for name, values in data.items():
            handle.create_dataset(name, data=values)
    temporary.replace(path)
    return data


def add_fourier_targets(atoms, coefficients, prefix, charge=None):
    """Store unsmoothed coefficients in native FFT order at K_CUTOFF."""
    target = hermitian(coefficients)
    if charge is not None:
        target[0, 0, 0] = charge * target.size / atoms.get_volume()
    atoms.info[f"{prefix}_fftn_shifted"] = np.stack(
        [target.real, target.imag]).astype(np.float32)
    atoms.info["fourier_target_convention"] = "raw_retained_modes"
    atoms.info["fourier_cutoff_A_inverse"] = K_CUTOFF


def add_vacuum_target(atoms, planar, axis, correction):
    key = f"vacuum_{'xyz'[axis]}frac"
    if VACUUM_PLANE and set(VACUUM_PLANE) != {key}:
        raise ValueError(f'For this slab axis, specify only {key}')
    if VACUUM_PLANE:
        fraction = float(VACUUM_PLANE[key])
    elif key in atoms.info:
        fraction = float(atoms.info[key])
    else:
        distances = (correction - atoms.get_scaled_positions()[:, axis]) % 1.0
        fraction = (correction - 0.5 * np.min(distances[distances > 1e-8])) % 1.0
    if not np.isfinite(fraction):
        raise ValueError('Vacuum fraction must be finite')
    fraction %= 1.0
    coordinate = fraction * len(planar)
    index = int(np.floor(coordinate))
    weight = coordinate - index
    atoms.info["vacuum_potential"] = float(
        (1 - weight) * planar[index] + weight * planar[(index + 1) % len(planar)])
    atoms.info[key] = fraction
    atoms.info[f"dipole_correction_{'xyz'[axis]}frac"] = correction


def collect_job(outdir):
    metadata = json.loads((outdir / "dft_metadata.json").read_text())
    atoms = read(str(outdir / "vasprun.xml"), format="vasp-xml", index=-1)
    parameters = atoms.calc.parameters
    forces = atoms.get_forces(apply_constraint=False)
    if metadata.get('forces_file') == 'plugin_observables.npz':
        with np.load(outdir/'plugin_observables.npz') as obs:
            forces = obs['expected_forces'].copy()
    energy = atoms.get_potential_energy(force_consistent=True)
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all() or not np.isfinite(energy):
        raise ValueError(f'Invalid energy/forces: {outdir}')

    outcar = Outcar(str(outdir / "OUTCAR"))
    outcar.read_pseudo_zval()
    # OUTCAR's last electron-count report can belong to the neutral atomic
    # initialization. The executed NELECT input retains the electrode charge.
    charge = sum(outcar.zval_dict[s] for s in atoms.get_chemical_symbols()) - parameters['nelect']
    fermi = float(outcar.efermi)
    if not np.isfinite(charge) or not np.isfinite(fermi):
        raise ValueError(f'Nonfinite charge/Fermi level: {outdir}')
    counter = float(metadata["counter_charge"])

    # Charged slabs use a Python dipole correction with native LDIPOL=False.
    axis = metadata["slab_axis"]
    slab = axis is not None
    external_field = np.asarray(metadata["external_field"], dtype=float)
    dipole = np.zeros(3)
    if slab:
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        if metadata["plugin"]:
            with np.load(outdir / "plugin_observables.npz") as data:
                # Electrode + explicit solvent only; exclude the countercharge.
                dipole = float(data["electrode_dipole"]) * normal
        else:
            dipole = -np.asarray(atoms.get_dipole_moment(), dtype=float).reshape(3)
        correction = float(metadata[f"dipole_correction_{'xyz'[axis]}frac"])
        center = (correction + 0.5) % 1.0
        fractional = atoms.get_scaled_positions(wrap=False)[:, axis]
        atoms.positions -= np.floor(fractional - center + 0.5)[:, None] * atoms.cell[axis]
        atoms.pbc[axis] = False

    atoms.calc = None
    atoms.set_constraint()
    atoms.arrays["vasp_forces"] = forces
    if (external_field.shape != (3,) or not np.isfinite(external_field).all()
            or not np.isfinite(dipole).all() or not np.isfinite(counter)):
        raise ValueError(f'Nonfinite or malformed electrostatic controls/labels: {outdir}')
    atoms.info = {
        "config_type": "slab" if slab else "bulk",
        "vasp_free_energy": float(energy),
        "vasp_efermi": fermi,
        "total_charge": float(charge),
        "counter_charge": counter,
        "external_field": external_field,
        "vasp_dipole": dipole,
        "config_dipole_weight": np.eye(3)[axis] if slab else np.zeros(3),
        "config_fermi_level_weight": float(slab),
        "config_vacuum_potential_weight": float(slab and COLLECT_POTENTIAL),
        "config_stress_weight": 0.0,  # The fixed-cell plugin supplies no stress.
    }
    if metadata['schema'] == 'mace_volt_cdce_v1' and slab:
        # The old runner always used sheets, despite its Cartesian key name.
        center_cc = np.asarray(metadata['counter_charge_center'])
        atoms.info[f"counter_charge_{'xyz'[axis]}frac"] = float(
            (center_cc @ np.linalg.inv(atoms.cell.array))[axis] % 1.)
    else:
        for key in ('counter_charge_center', *[f'counter_charge_{a}frac' for a in 'xyz']):
            if key in metadata:
                atoms.info[key] = metadata[key]
    if any(key in atoms.info for key in ('counter_charge_center', *[f'counter_charge_{a}frac' for a in 'xyz'])):
        atoms.info['counter_charge_width'] = float(metadata['counter_charge_width'])
    if slab:
        atoms.info[f"dipole_correction_{'xyz'[axis]}frac"] = correction
        vacuum_key = f"vacuum_{'xyz'[axis]}frac"
        if vacuum_key in metadata:
            atoms.info[vacuum_key] = float(metadata[vacuum_key])
    data = fourier_data(outdir, atoms, parameters, axis)
    if COLLECT_DENSITY:
        add_fourier_targets(atoms, data["fourier_density"], "vasp_rho", charge)
    if COLLECT_POTENTIAL:
        if slab:
            add_vacuum_target(atoms, data["planar_potential"], axis, correction)
        add_fourier_targets(atoms, data["fourier_potential"], "vasp_phi")
        add_fourier_targets(atoms, data["proto_fourier_potential"], "vasp_proto_phi")
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
    if not records:
        raise SystemExit("No configurations collected")
    train, valid = split_configs(records)
    for path, configs in ((OUT_XYZ, [a for a, _ in records]),
                          (OUT_TRAIN_XYZ, train), (OUT_VALID_XYZ, valid)):
        write(str(path), configs, format="extxyz", write_results=False)
        print(f"Wrote {len(configs)} configurations to {path}")


if __name__ == "__main__":
    collect()
