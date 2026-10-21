"""Collect VASP labels and compact electronic diagnostics; edit settings below.

Each folder gets one fourier_data.hdf. Fourier fields are unsmoothed, in
NumPy FFT order: density is AECCAR1-AECCAR2 / volume; ESP is native LOCPOT.
The density zero mode is fixed to the net charge from sum(ZVAL)-NELECT.
The XYZ training ESP remains LOCPOT minus its protoatomic reference.
Dependencies: numpy, h5py, ase. Plugin marker contents are never read.
CHGCAR supplies the planar electron density and slab dipole. Near-EF site
weights come from native vaspout.h5, or PROCAR if HDF output is unavailable.
"""
from pathlib import Path
import multiprocessing as mp
import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from ase.io import read, write
from ase.io.vasp import read_vasp_configuration


BASE_PATH = Path('/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft/')
OUTPUT_GLOB = 'vasp_*/output_*'  # Use 'output_*' for the two supplied folders.
N_PROCS = 4  # Each worker reads full VASP grids; increase according to RAM.
K_CUTOFF = 3.0  # inverse Angstrom; no Gaussian smearing is applied.
COLLECT_DENSITY = True  # Requires AECCAR1 and AECCAR2 (LAECHG=True).
COLLECT_POTENTIAL = True  # Requires electrostatic LOCPOT (LVHAR=True).
STATE_WINDOW = 3.0  # eV on either side of EF; store site weights, not orbitals.
PROFILE_AXIS = 2  # Used for profiles when no slab direction is specified.
VACUUM_PLANE = {}  # Optional {'vacuum_zfrac': 0.75}; otherwise choose a plane.
PROTO_BASE_PATH = None  # Optional tree of ordinary proto-run LOCPOT files.
VALID_FRACTION = 0.2
SPLIT_SEED = 7777

# Explicit settings for the supplied Gaussian-sheet plugin runs. A marker's
# presence is checked; its contents are NEVER imported or read.
PLUGIN_FILES = ('vasp_plugin.py', 'plugin_observables.npz', 'plugin_scf.csv')
PLUGIN_SETTINGS = dict(slab_axis=2, counter_charge_position=10.0,
    counter_charge_width=1.0, dipole_correction_fraction=0.5,
    external_field=(0.0, 0.0, 0.0))
# position is distance along the cell normal in Angstrom; width is Gaussian sigma.
# Per-folder overrides, keyed relative to BASE_PATH, e.g.:
# RUN_SETTINGS = {'vasp_0_4/output_1': {'counter_charge_position': 12.0}}
# plugin=False selects ordinary VASP even if a stale marker is present.
RUN_SETTINGS = {}


def read_parameters(folder):
    """Stop before the large DOS/eigenvector sections of vasprun.xml."""
    parameters = {}
    with (folder / 'vasprun.xml').open('rb') as stream:
        for _, element in ET.iterparse(stream, events=('end',)):
            if element.tag in ('incar', 'parameters'):
                for item in element.iter():
                    if item.tag not in ('i', 'v') or item.get('name', '').lower() not in (
                            'nelect', 'lvhar', 'idipol', 'ldipol', 'dipol', 'efield',
                            'ispin', 'lnoncollinear', 'lsorbit', 'sigma', 'ismear'):
                        continue
                    kind = item.get('type', 'float')
                    text = (item.text or '').strip()
                    cast = {'logical': lambda x: x.strip('.').upper() == 'T',
                            'int': int, 'float': float, 'string': str}[kind]
                    parameters[item.get('name').lower()] = (
                        [cast(x) for x in text.split()] if item.tag == 'v' else cast(text))
            elif element.tag == 'atominfo':
                rows = element.findall("array[@name='atomtypes']/set/rc")
                valences = {row[1].text.strip(): float(row[3].text) for row in rows}
                return parameters, valences
    raise ValueError(f'Incomplete VASP parameters/atom types: {folder}')


def read_structure(stream, atoms):
    structure = read_vasp_configuration(stream)
    delta = structure.get_scaled_positions(False) - atoms.get_scaled_positions(False)
    if (structure.get_chemical_symbols() != atoms.get_chemical_symbols()
            or not np.allclose(structure.cell, atoms.cell, rtol=0, atol=1.e-4)
            or not np.allclose(delta - np.rint(delta), 0., rtol=0, atol=1.e-5)):
        raise ValueError(f'Structure/grid mismatch: {stream.name}')
    return structure


def read_grid(path, atoms):
    """Read just the scalar grid; skip augmentation and magnetization blocks."""
    with path.open() as stream:
        read_structure(stream, atoms)
        line = stream.readline()
        while line and not line.strip():
            line = stream.readline()
        shape = tuple(map(int, line.split()))
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError(f'Invalid grid dimensions: {path}')
        data = np.fromfile(stream, count=int(np.prod(shape)), sep=' ')
    if data.size != np.prod(shape) or not np.isfinite(data).all():
        raise ValueError(f'Incomplete/nonfinite grid: {path}')
    return data.reshape(shape, order='F')


def fourier(grid, cell):
    # Odd retained sizes keep +/-k pairs without a new Nyquist-plane ambiguity.
    extent = np.floor(K_CUTOFF * cell.lengths() / (2 * np.pi)).astype(int)
    shape = 2 * extent + 1
    if np.any(shape > grid.shape):
        raise ValueError('K_CUTOFF exceeds the available VASP grid resolution')
    modes = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in shape]
    indices = np.ix_(*[k % n for k, n in zip(modes, grid.shape)])
    return np.fft.fftn(grid)[indices] * np.prod(shape) / grid.size


def controls(folder, atoms, parameters, charge):
    custom = RUN_SETTINGS.get(folder.relative_to(BASE_PATH).as_posix(), {})
    plugin = custom.get('plugin', any((folder / name).is_file() for name in PLUGIN_FILES))
    counter = float(custom.get('counter_charge', -charge)) if plugin else 0.0
    if abs(counter) < 1.e-8:
        counter = 0.0
    settings = {**PLUGIN_SETTINGS, **custom}
    native_axis = int(parameters.get('idipol', 0)) - 1
    axis = settings['slab_axis'] if plugin else (native_axis if native_axis in (0, 1, 2) else None)
    axis = custom.get('slab_axis', axis)
    info = dict(counter_charge=counter, plugin_present=bool(plugin))
    correction = None
    field = np.zeros(3)
    if axis is not None:
        if axis not in (0, 1, 2):
            raise ValueError('slab_axis must be 0, 1, 2 or None')
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        length = atoms.cell[axis] @ normal
        if plugin:
            correction = float(settings['dipole_correction_fraction']) % 1.
            field = np.asarray(settings['external_field'], dtype=float)
        else:
            field = -float(parameters.get('efield', 0.)) * normal
            if parameters.get('ldipol', False):
                center = np.asarray(parameters.get('dipol', [-100.] * 3))
                if np.any(center == -100.):
                    print(f'{folder}: automatic DIPOL center unavailable; dipole/vacuum labels omitted.')
                else:
                    correction = (float(center[axis]) + 0.5) % 1.
        if plugin and abs(counter) > 1.e-8:
            key = f"counter_charge_{'xyz'[axis]}frac"
            info[key] = float(settings.get(key, settings['counter_charge_position'] / length)) % 1.
            info['counter_charge_width'] = float(settings['counter_charge_width'])
    elif abs(counter) > 1.e-8:
        # A localized 3D Gaussian is also supported when explicitly specified.
        info['counter_charge_center'] = np.asarray(settings['counter_charge_center'], dtype=float)
        info['counter_charge_width'] = float(settings['counter_charge_width'])
        field = np.asarray(settings['external_field'], dtype=float)
        if info['counter_charge_center'].shape != (3,) or not np.isfinite(info['counter_charge_center']).all():
            raise ValueError('Counter-charge center must be a finite Cartesian 3-vector')
    centers = [value for key, value in info.items() if key.endswith('frac')]
    if correction is not None:
        centers.append(correction)
    if field.shape != (3,) or not np.isfinite(field).all() or not np.isfinite([counter, *centers]).all():
        raise ValueError(f'Invalid electrostatic settings: {folder}')
    if 'counter_charge_width' in info and not (0 < info['counter_charge_width'] < np.inf):
        raise ValueError('Counter-charge width must be finite and positive')
    info['external_field'] = field
    return axis, correction, info


def scf_diagnostics(folder, text):
    final = text[text.rfind('Iteration'):]
    result = dict(electronic_converged='EDIFF is reached' in final)
    path = folder / 'OSZICAR'
    if path.is_file():
        # Keep only the final residuals, not a second copy of the SCF history.
        for line in path.read_text().splitlines():
            if re.match(r'\s*[A-Z]+:\s+\d+\s', line):
                tokens = line.split()
                if int(tokens[1]) == 1:
                    result.pop('scf_density_residual', None)
                result.update(scf_steps=int(tokens[1]), scf_energy_change=float(tokens[3]),
                              scf_band_energy_change=float(tokens[4]), scf_orbital_residual=float(tokens[6]))
                if len(tokens) > 7:
                    result['scf_density_residual'] = float(tokens[7])
    return result


def electronic_states(folder, fermi, parameters, natoms):
    """Retain only near-EF site projections; no DOS or wavefunction reread."""
    path = folder / 'vaspout.h5'
    rows, projections = [], []
    if path.is_file():
        with h5py.File(path) as handle:
            if all(k in handle for k in ('results/electron_eigenvalues', 'results/projectors/par')):
                eigen = handle['results/electron_eigenvalues']
                energy = eigen['eigenvalues'][...] - fermi
                occupancy = eigen['fermiweights'][...]
                capacity = 1 if (parameters.get('ispin', 1) == 2 or parameters.get('lnoncollinear', False)
                                 or parameters.get('lsorbit', False)) else 2
                weights = eigen['kpoints_symmetry_weight'][...]
                weights = weights / weights.sum()
                site = handle['results/projectors/par']
                if site.shape[1] != natoms:
                    raise ValueError(f'Projection atom count differs: {path}')
                for spin in range(len(energy)):
                    for k, weight in enumerate(weights):
                        bands = np.flatnonzero(abs(energy[spin, k]) <= STATE_WINDOW)
                        if len(bands):
                            projected = site[spin, :, :, k, bands].sum(axis=1).T
                            for band, atom_weights in zip(bands, projected):
                                rows.append((energy[spin, k, band], capacity * occupancy[spin, k, band],
                                             weight, spin, k + 1, band + 1))
                                projections.append(atom_weights)
                return rows, projections, True
    # PROCAR prints physical occupations (0..2 for nonmagnetic scalar states).
    path = folder / 'PROCAR'
    if not path.is_file():
        return rows, projections, False
    spin, previous_k, selected, first_table = 0, 0, False, False
    with path.open() as stream:
        for line in stream:
            if line.lstrip().startswith('k-point'):
                fields = line.split()
                k, weight = int(fields[1]), float(fields[-1])
                if k <= previous_k:
                    spin += 1
                previous_k = k
            elif line.lstrip().startswith('band '):
                fields = line.split()
                band, energy, occupancy = int(fields[1]), float(fields[4]) - fermi, float(fields[-1])
                selected, first_table = abs(energy) <= STATE_WINDOW, True
            elif selected and first_table and line.lstrip().startswith('ion '):
                values = [float(next(stream).split()[-1]) for _ in range(natoms)]
                rows.append((energy, occupancy, weight, spin, k, band))
                projections.append(values)
                first_table = False  # Noncollinear magnetization tables are not charge weights.
    return rows, projections, True


def collect_job(folder):
    parameters, valences = read_parameters(folder)
    atoms = read(str(folder / 'OUTCAR'), format='vasp-out', index=-1)
    forces = atoms.get_forces(apply_constraint=False)
    energy = atoms.get_potential_energy(force_consistent=True)
    # OUTCAR and grid headers round positions; CONTCAR preserves their precision.
    geometry = folder / 'CONTCAR'
    if geometry.is_file():
        with geometry.open() as stream:
            accurate = read_structure(stream, atoms)
        atoms.set_cell(accurate.cell)
        atoms.set_positions(accurate.positions, apply_constraint=False)
    text = (folder / 'OUTCAR').read_text(errors='replace')
    number = r'([-+\d.]+(?:[Ee][-+]?\d+)?)'
    values = re.findall(r'Fermi energy:\s*' + number, text) or re.findall(r'E-fermi\s*:\s*' + number, text)
    fermi = float(values[-1])
    charge = sum(valences[s] for s in atoms.get_chemical_symbols()) - float(parameters['nelect'])
    if not np.isfinite(forces).all() or not np.isfinite([energy, fermi, charge]).all():
        raise ValueError(f'Nonfinite VASP labels: {folder}')
    axis, correction, info = controls(folder, atoms, parameters, charge)
    profile_axis = PROFILE_AXIS if axis is None else axis
    normal = np.linalg.inv(atoms.cell.array)[:, profile_axis]
    length = 1. / np.linalg.norm(normal)
    volume = atoms.get_volume()
    average_axes = tuple(i for i in range(3) if i != profile_axis)
    data = dict(cell=atoms.cell.array.copy(), positions=atoms.positions.copy(), atomic_numbers=atoms.numbers)
    info.update(vasp_free_energy=float(energy), vasp_efermi=fermi, total_charge=float(charge),
                config_type='slab' if axis is not None else 'bulk',
                config_dipole_weight=np.zeros(3), config_stress_weight=0.,
                config_fermi_level_weight=float(axis is not None), config_vacuum_potential_weight=0.,
                **scf_diagnostics(folder, text))
    atoms.calc = None
    atoms.set_constraint()
    atoms.info = info
    atoms.arrays['vasp_forces'] = forces
    if axis is not None:
        atoms.pbc[axis] = False
    if COLLECT_DENSITY:
        initial = read_grid(folder / 'AECCAR1', atoms)
        density = read_grid(folder / 'AECCAR2', atoms)
        if initial.shape != density.shape:
            raise ValueError(f'AECCAR grid mismatch: {folder}')
        initial -= density
        initial /= volume
        data['fourier_density'] = fourier(initial, atoms.cell)
        data['fourier_density'][0, 0, 0] = charge * data['fourier_density'].size / volume
        del initial, density
    # CHGCAR is the density used for the electrostatics; AECCAR2 is an
    # all-electron reconstruction and must not supply the plugin dipole.
    if (folder / 'CHGCAR').is_file():
        density = read_grid(folder / 'CHGCAR', atoms).mean(axis=average_axes)
        data['planar_electron_density'] = density / volume
        info['electron_count_error'] = float(density.mean() - parameters['nelect'])
        if axis is not None and correction is not None:
            center = (correction + 0.5) % 1.
            fractions = np.arange(len(density)) / len(density)
            z = ((fractions - center + 0.5) % 1. - 0.5) * length
            z[np.isclose(fractions, correction, rtol=0, atol=1.e-12)] = 0.
            ion_z = ((atoms.get_scaled_positions()[:, axis] - center + 0.5) % 1. - 0.5) * length
            # Ionic valences minus final valence electrons: exclude prescribed countercharge.
            dipole = np.dot([valences[s] for s in atoms.get_chemical_symbols()], ion_z)
            dipole -= np.mean(density * z)
            info['vasp_dipole'] = float(dipole) * normal / np.linalg.norm(normal)
            info['config_dipole_weight'] = np.eye(3)[axis]
        del density
    if COLLECT_POTENTIAL:
        if not parameters.get('lvhar', False):
            raise ValueError(f'Electrostatic LOCPOT needs LVHAR=True: {folder}')
        potential = read_grid(folder / 'LOCPOT', atoms)
        data['planar_potential'] = potential.mean(axis=average_axes)
        data['fourier_potential'] = fourier(potential, atoms.cell)
        proto_path = folder / 'LOCPOT_proto'
        if not proto_path.is_file() and PROTO_BASE_PATH is not None:
            proto_path = Path(PROTO_BASE_PATH) / folder.relative_to(BASE_PATH) / 'LOCPOT'
        if proto_path.is_file():
            proto = read_grid(proto_path, atoms)
            if proto.shape != potential.shape:
                raise ValueError(f'ESP/proto grid mismatch: {folder}')
            data['proto_fourier_potential'] = fourier(proto, atoms.cell)
            del proto
        del potential
        if axis is not None and (correction is not None or VACUUM_PLANE):
            key = f"vacuum_{'xyz'[axis]}frac"
            if VACUUM_PLANE:
                fraction = float(VACUUM_PLANE[key]) % 1.
            else:
                distances = (correction - atoms.get_scaled_positions()[:, axis]) % 1.
                fraction = (correction - 0.5 * distances[distances > 1.e-8].min()) % 1.
            planar = data['planar_potential']
            index = fraction * len(planar)
            j, weight = int(index), index % 1.
            info['vacuum_potential'] = float((1 - weight) * planar[j] + weight * planar[(j + 1) % len(planar)])
            info['config_vacuum_potential_weight'] = 1.
            info[key] = fraction
    if correction is not None:
        info[f"dipole_correction_{'xyz'[axis]}frac"] = correction
        fractional = atoms.get_scaled_positions(False)[:, axis]
        atoms.positions -= np.floor(fractional - (correction + 0.5) % 1. + 0.5)[:, None] * atoms.cell[axis]
    # Preserve existing XYZ keys used by the supplied training configs.
    targets = {'vasp_rho': data.get('fourier_density')}
    if 'proto_fourier_potential' in data:
        targets.update(vasp_phi=data['fourier_potential'] - data['proto_fourier_potential'],
                       vasp_proto_phi=data['proto_fourier_potential'])
    elif COLLECT_POTENTIAL:
        print(f'{folder}: no proto reference; total ESP saved in HDF, deformation ESP training label omitted.')
    for prefix, coefficients in targets.items():
        if coefficients is not None:
            info[f'{prefix}_fftn_shifted'] = np.stack((coefficients.real, coefficients.imag)).astype(np.float32)
    rows, projections, available = electronic_states(folder, fermi, parameters, len(atoms))
    path = folder / 'fourier_data.hdf'
    temporary = folder / '.fourier_data.tmp'
    with h5py.File(temporary, 'w') as handle:
        handle.attrs.update(k_cutoff=K_CUTOFF, profile_axis=profile_axis, profile_length=length,
                            state_window_eV=STATE_WINDOW, projections_available=available,
                            esp_reference='LOCPOT', density_reference='AECCAR1-AECCAR2 / volume',
                            fft_normalization='numpy.fft.fftn on retained grid',
                            nelect=parameters['nelect'], sigma_eV=parameters.get('sigma', np.nan),
                            ismear=parameters.get('ismear', 0))
        for key, value in info.items():
            if '_fftn_' not in key:
                handle.attrs[key] = value
        for name, values in data.items():
            handle.create_dataset(name, data=values)
        if available:
            states = np.asarray(rows, dtype=float).reshape(-1, 6)
            group = handle.create_group('states')
            for i, name in enumerate(('energy_minus_EF', 'occupation', 'kpoint_weight', 'spin', 'kpoint', 'band')):
                group.create_dataset(name, data=states[:, i].astype(int if i >= 3 else float))
            group.create_dataset('atom_weights', data=np.asarray(projections, dtype=np.float32).reshape(-1, len(atoms)))
            group.attrs['indexing'] = 'spin: zero based; kpoint and band: one based; atoms: VASP order'
    temporary.replace(path)
    print(f'Collected {folder}: charge={charge:+.6f}, counter_charge={info["counter_charge"]:+.6f}, '
          f'SCF converged={info["electronic_converged"]}, near-EF states={len(rows)}', flush=True)
    return atoms, folder.parent.name


def collect():
    paths = sorted(p for p in BASE_PATH.glob(OUTPUT_GLOB) if (p / 'OUTCAR').is_file())
    if not paths:
        raise SystemExit('No VASP output folders found; check BASE_PATH and OUTPUT_GLOB')
    if N_PROCS <= 1:
        records = [collect_job(p) for p in paths]
    else:
        with mp.Pool(min(N_PROCS, len(paths))) as pool:
            records = list(pool.imap(collect_job, paths))
    write(BASE_PATH / 'combined_VASP_shifted.xyz', [a for a, _ in records], format='extxyz', write_results=False)
    # Keep whole batches together, stratified by their slab/bulk content.
    groups = {}
    for atoms, group in records:
        groups.setdefault(group, set()).add(atoms.info['config_type'])
    strata = {}
    for group, types in groups.items():
        strata.setdefault(tuple(sorted(types)), []).append(group)
    rng, valid = np.random.default_rng(SPLIT_SEED), set()
    for batch_names in strata.values():
        batch_names = sorted(batch_names)
        rng.shuffle(batch_names)
        if len(batch_names) > 1:
            nvalid = min(max(round(VALID_FRACTION * len(batch_names)), 1), len(batch_names) - 1)
            valid.update(batch_names[:nvalid])
    if not valid:
        print('Combined XYZ written; too few independent batches for a stratified split.')
        return
    for name, selected in (('train', False), ('valid', True)):
        write(BASE_PATH / f'combined_VASP_shifted_{name}.xyz',
              [a for a, group in records if (group in valid) == selected], format='extxyz', write_results=False)


if __name__ == '__main__':
    collect()
