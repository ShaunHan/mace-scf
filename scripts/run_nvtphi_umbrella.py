#!/usr/bin/env python3
import csv
import hashlib
import json
import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from itertools import combinations
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from ase import units
from ase.constraints import FixAtoms, FixConstraint, Hookean
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# ------------------------------ user settings ------------------------------ #
ATOMS_FILE = 'IrO2OH_110_mdinit_0_50_vasp_sp.xyz'
MODEL_FILE = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/ml_training/wf_training_186/fit_0_stage4_finetune_v383.model'
RUN_INDEXES = list(range(10, 20))
OXYGEN_PAIRS = {}  # Optional {frame_index: (oxygen_id, oxygen_id)}; ASE zero-based IDs.
DEVICE = 'cuda:0'
TARGET_POTENTIAL = 1.60 + 4.44  # V
VACUUM_POTENTIAL_KEY, FERMI_LEVEL_KEY = 'vacuum_potential', 'fermi_level'
COUNTER_SHEET_POINT, GAUSSIAN_WIDTH = [0., 0., 10.], 1.0  # Angstrom; point on the Gaussian sheet.
CAPACITANCE, TAUPHI_FS = 0.01, 10.0
Z_FIXED, Z_REPULSIVE, WALL_K = -11.0, 15.0, 10.0
PAIR_CUTOFF = 3.0  # Initial minimum-image distance, Angstrom.
TEMPERATURE_K, TIMESTEP_FS, FRICTION_PER_FS = 300., 0.5, 0.01
OO_MIN, WINDOW_SPACING, UMBRELLA_K = 1.20, 0.05, 20.0  # A, A, eV/A^2.
WINDOW_STEPS, EQUIL_STEPS = 10000, 2000  # Equilibrate separately in EACH fixed window.
RELEASE_STEPS = 10000  # Final 5 ps without the O-O restraint; excluded from the PMF.
LOGINTERVAL, SEED, SCF_STEPS = 10, 7777, 50
# -------------------------------------------------------------------------- #


def scalar(value):
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f'Expected one finite scalar, got {value}')
    return float(a.item())


def jsonable(value):
    return np.asarray(value).tolist()


def oxygen_pair(atoms, index):
    """Uniform random pair of mobile interfacial O atoms, without an H-count filter."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    ir = np.flatnonzero(symbols == 'Ir')
    if not len(ir):
        raise ValueError('An Ir electrode is required to locate the interface')
    top = atoms.positions[ir, 2].max()
    ids = np.flatnonzero((symbols == 'O') & (atoms.positions[:, 2] >= Z_FIXED)
                        & (atoms.positions[:, 2] > top-2.) & (atoms.positions[:, 2] < top+5.))
    distances = atoms.get_all_distances(mic=True)
    candidates = [p for p in combinations(ids.tolist(), 2)
                  if OO_MIN < distances[p] <= PAIR_CUTOFF]
    if index in OXYGEN_PAIRS:
        pair = tuple(sorted(OXYGEN_PAIRS[index]))
        if pair not in candidates:
            raise ValueError(f'Frame {index}: override is not an eligible interfacial O pair')
        return pair
    if not candidates:
        raise ValueError(f'Frame {index}: no mobile interfacial O pair within {PAIR_CUTOFF} A')
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index, 1]))
    return candidates[int(rng.integers(len(candidates)))]


def physical_constraints(atoms, fixed_ids):
    fixed = set(fixed_ids)
    return [FixAtoms(indices=fixed_ids)] + [
        Hookean(a1=i, a2=(0, 0, 1, -Z_REPULSIVE), k=WALL_K)
        for i, a in enumerate(atoms) if a.symbol == 'O' and i not in fixed]


def prepare_frame(source, index):
    atoms, info = source.copy(), source.info
    if not np.array_equal(atoms.pbc, [True, True, False]):
        raise ValueError('Expected an xy-periodic slab with open z')
    if not np.allclose(info['external_field'], 0.) or scalar(info['total_charge']) != 0.:
        raise ValueError('The initial DFT reference must be neutral and at zero external field')
    fermi_key = FERMI_LEVEL_KEY if FERMI_LEVEL_KEY in info else 'vasp_efermi'
    vacuum, fermi = scalar(info[VACUUM_POTENTIAL_KEY]), scalar(info[fermi_key])
    pair = oxygen_pair(atoms, index)
    atoms.info = dict(vacuum_zfrac=scalar(info['vacuum_zfrac']),
                      dipole_correction_zfrac=scalar(info['dipole_correction_zfrac']),
                      dft_vacuum_potential=vacuum, dft_fermi_level=fermi,
                      reference_potential=vacuum-fermi, source_index=index, oxygen_pair=list(pair),
                      external_field=np.zeros(3), total_charge=0., counter_charge=0.,
                      counter_charge_zfrac=float((np.asarray(COUNTER_SHEET_POINT) @ np.linalg.inv(atoms.cell.array))[2] % 1.),
                      counter_charge_width=GAUSSIAN_WIDTH)
    atoms.set_constraint(physical_constraints(atoms, np.flatnonzero(atoms.positions[:, 2] < Z_FIXED).tolist()))
    return atoms


class DistanceUmbrella(FixConstraint):
    """V = k/2 (d-center)^2; the center is fixed throughout each sampling window."""
    def __init__(self, dyn, protocol):
        self.dyn, self.protocol = dyn, protocol
        self.pair = dyn.atoms.info['oxygen_pair']

    def schedule(self):
        step = self.dyn.nsteps
        window, offset = divmod(step, WINDOW_STEPS)
        if window >= len(self.protocol['centers_A']):
            return window, 'release', None
        phase = 'equil' if offset < EQUIL_STEPS else 'sample'
        return window, phase, self.protocol['centers_A'][window]

    def evaluate(self, atoms):
        vector = atoms.get_distance(*self.pair, mic=True, vector=True)
        distance = np.linalg.norm(vector)
        if not np.isfinite(distance) or distance < 1e-8:
            raise ValueError('The O-O distance is invalid')
        center = self.schedule()[2]
        radial = 0. if center is None else -UMBRELLA_K*(distance-center)
        forces = np.zeros((len(atoms), 3))
        forces[self.pair[1]] = radial*vector/distance
        forces[self.pair[0]] = -forces[self.pair[1]]
        energy = 0. if center is None else .5*UMBRELLA_K*(distance-center)**2
        return forces, energy, float(distance), float(radial)

    def adjust_forces(self, atoms, forces):
        forces += self.evaluate(atoms)[0]

    def adjust_potential_energy(self, atoms):
        return self.evaluate(atoms)[1]

    def adjust_momenta(self, atoms, momenta):
        pass

    def get_removed_dof(self, atoms):
        return 0

    def __deepcopy__(self, memo):
        return self


def make_calculator(model_file, enable_cueq=True):
    import torch
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    from mace_scf.utils.foundation import accelerate_backbone
    if torch.device(DEVICE).type == 'cuda':
        torch.cuda.set_device(DEVICE)
        print(f'GPU: {torch.cuda.get_device_name(DEVICE)}', flush=True)
    calc = MACEFixedPointSCF(
        model_path=model_file, device=DEVICE, pbc_handling='slab', external_field_key='external_field',
        total_charge_key='total_charge', counter_charge_width=GAUSSIAN_WIDTH, scf_restart=False,
        scf_options=dict(num_scf_steps=SCF_STEPS, constant_charge=True, use_autograd_forces=True,
                         initial_density='local_guess', initial_fermi_level='zero'),
        ignore_nonconverged=True)
    if enable_cueq:
        accelerate_backbone(calc.model, DEVICE)
    calc.model.field_dependent_charges_map.deployment_mode = 'unroll_scf'
    calc.model.eval().requires_grad_(False)
    calc.enable_cueq = enable_cueq
    return calc


def settings():
    names = ('TARGET_POTENTIAL', 'VACUUM_POTENTIAL_KEY', 'FERMI_LEVEL_KEY', 'COUNTER_SHEET_POINT',
             'GAUSSIAN_WIDTH', 'CAPACITANCE', 'TAUPHI_FS', 'Z_REPULSIVE', 'WALL_K',
             'TEMPERATURE_K', 'TIMESTEP_FS', 'FRICTION_PER_FS', 'OO_MIN', 'WINDOW_SPACING',
             'UMBRELLA_K', 'WINDOW_STEPS', 'EQUIL_STEPS', 'RELEASE_STEPS', 'SCF_STEPS', 'LOGINTERVAL')
    return {k: globals()[k] for k in names}


def table(path):
    """Read the last complete row and column order before opening an append handle."""
    if not path.exists() or path.stat().st_size == 0:
        return None, None
    with path.open() as stream:
        reader = csv.DictReader(stream)
        fields, row = reader.fieldnames, None
        previous = -1
        for row in reader:
            if None in row or None in row.values() or int(row['step']) <= previous:
                raise ValueError(f'{path}: incomplete or nonmonotonic output; preserved for recovery')
            previous = int(row['step'])
    return fields, row


def simulate(atoms, model_file, max_steps=None):
    """max_steps is used by validation; normal runs complete the saved protocol."""
    import torch
    from mace_scf.md import NVTPhiLangevin
    from mace_scf.md.logger import NVTPhiMDLogger
    torch.set_num_threads(1)
    restart = atoms.info.pop('restart_state', None)
    saved_record = atoms.info.pop('md_record', None)
    start = int(atoms.info['step']) if restart else 0
    index, pair = atoms.info['source_index'], atoms.info['oxygen_pair']
    physical = atoms.constraints.copy()
    digest = hashlib.sha256(Path(model_file).read_bytes()).hexdigest()
    cfg_path = Path('run.json')
    if cfg_path.exists():
        protocol = json.loads(cfg_path.read_text())
        if protocol.get('method') != 'distance_umbrella' or restart is None:
            raise ValueError('Use a fresh folder: restart is supported only for new umbrella runs')
        previous = dict(protocol['settings'])
        if 'COUNTER_CHARGE_CENTER' in previous:
            previous['COUNTER_SHEET_POINT'] = previous.pop('COUNTER_CHARGE_CENTER')
        if previous != settings() or protocol['model_sha256'] != digest:
            raise ValueError('Saved umbrella settings/model differ; restore them before continuing')
        if restart['run_id'] != protocol['run_id']:
            raise ValueError('Trajectory and run.json belong to different runs')
    else:
        if restart:
            raise ValueError('Restart requires the original run.json')
        distance = atoms.get_distance(*pair, mic=True)
        n = max(2, int(np.ceil(max(0., distance-OO_MIN)/WINDOW_SPACING))+1)
        protocol = dict(method='distance_umbrella', source_index=index, oxygen_pair=pair,
                        centers_A=np.linspace(distance, OO_MIN, n).tolist(),
                        fixed_ids=physical[0].get_indices().tolist(), settings=settings(),
                        model_sha256=digest, model_file=str(model_file))
        protocol['run_id'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    rng = np.random.default_rng(SEED+index)
    if restart:
        rng.bit_generator.state = restart['rng_state']
    atoms.calc = make_calculator(model_file, enable_cueq=True)
    if not restart:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, force_temp=True, rng=rng)
    dyn = NVTPhiLangevin(atoms, timestep=TIMESTEP_FS*units.fs, temperature_K=TEMPERATURE_K,
                        friction=FRICTION_PER_FS/units.fs, target_potential=TARGET_POTENTIAL,
                        C0=CAPACITANCE, tauphi=TAUPHI_FS*units.fs, fixcm=False, rng=rng,
                        vacuum_potential_key=VACUUM_POTENTIAL_KEY, fermi_level_key=FERMI_LEVEL_KEY,
                        reference_potential=atoms.info['reference_potential'])
    dyn.nsteps = start
    if restart:
        dyn._reference_model_potential = restart['reference_model_potential']
    dyn._read_potential()
    if not cfg_path.exists():
        protocol.update(reference_model_potential=dyn._reference_model_potential,
                        initial_rng_state=rng.bit_generator.state, initial_state=dict(
                            numbers=atoms.numbers, positions=atoms.positions, momenta=atoms.get_momenta(),
                            cell=atoms.cell.array, pbc=atoms.pbc, info=dict(atoms.info, step=0)))
        temporary = cfg_path.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(protocol, stream, indent=2, default=jsonable)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cfg_path)
    bias = DistanceUmbrella(dyn, protocol)
    atoms.set_constraint([bias]+physical)
    end = len(protocol['centers_A'])*WINDOW_STEPS+RELEASE_STEPS
    print(f'Frame {index}: O pair {pair}; {len(protocol["centers_A"])} fixed windows '
          f'to {OO_MIN} A, then {RELEASE_STEPS} release steps; resume={start}. '
          f'CuEq={getattr(atoms.calc, "enable_cueq", True)}, SCF={SCF_STEPS}, no convergence gate.', flush=True)
    symbols = np.asarray(atoms.get_chemical_symbols())
    hydrogen, iridium = [np.flatnonzero(symbols == s) for s in ('H', 'Ir')]
    with ExitStack() as stack:
        fields, row = table(Path('md.csv'))
        last_csv = -1 if row is None else int(row['step'])
        log_path = Path('md.log')
        lines = log_path.read_text().splitlines() if log_path.exists() else []
        times = [float(line.split()[0]) for line in lines if line.strip() and not line.startswith('Time')]
        last_log = round(times[-1]*1000/TIMESTEP_FS) if times else -1
        if max(last_csv, last_log) > start:
            raise ValueError('Logs are ahead of the last saved frame; refusing to overwrite')
        stream = stack.enter_context(open('md.csv', 'a', newline='', buffering=1))
        logger = NVTPhiMDLogger(dyn, atoms, 'md.log', header=not lines, mode='a',
                               vacuum_potential_key=VACUUM_POTENTIAL_KEY, fermi_level_key=FERMI_LEVEL_KEY)
        stack.callback(logger.close)
        last_traj = -1
        if Path('md.traj').exists() and Path('md.traj').stat().st_size:
            with Trajectory('md.traj') as previous:
                if len(previous):
                    last_traj = int(previous[-1].info['step'])
        trajectory = stack.enter_context(Trajectory('md.traj', 'a'))

        def record():
            nonlocal last_csv, last_log, last_traj, fields
            energy = atoms.get_potential_energy()
            results = atoms.calc.results
            force, bias_energy, distance, radial = bias.evaluate(atoms)
            window, phase, center = bias.schedule()
            row = dict(step=dyn.nsteps, time_fs=dyn.get_time()/units.fs, window=window, phase=phase,
                       distance_A=distance, center_A='' if center is None else center,
                       k_eV_A2=0. if center is None else UMBRELLA_K, bias_eV=bias_energy,
                       radial_bias_eV_A=radial, energy_eV=energy, temperature_K=atoms.get_temperature(),
                       Phi_V=dyn._read_potential(), total_charge_e=scalar(atoms.info['total_charge']),
                       vacuum_potential_V=scalar(results[VACUUM_POTENTIAL_KEY]),
                       fermi_level_eV=scalar(results[FERMI_LEVEL_KEY]), scf_steps=results['num_scf_steps'],
                       scf_residual=scalar(results['scf_residual']))
            for j, o in enumerate(pair):
                h = atoms.get_distances(o, hydrogen, mic=True)
                ir = atoms.get_distances(o, iridium, mic=True)
                row.update({f'O{j+1}_H_neighbors': int(np.sum(h < 1.25)),
                            f'O{j+1}_Ir_neighbors': int(np.sum(ir < 2.60)),
                            f'O{j+1}_nearest_Ir_A': float(ir.min())})
            if dyn.nsteps > last_traj:
                frame = atoms.copy()
                frame.set_constraint(physical)
                frame.info.update(row)
                frame.info.update(md_record=row, restart_state=dict(run_id=protocol['run_id'],
                    rng_state=rng.bit_generator.state, reference_model_potential=dyn._reference_model_potential))
                trajectory.write(frame, energy=scalar(results['energy'])+bias_energy,
                                 forces=atoms.calc.get_forces(atoms)+force)
                last_traj = dyn.nsteps
            if dyn.nsteps > last_csv:
                if dyn.nsteps == start and saved_record:
                    row = saved_record
                if fields is not None and set(fields) != set(row):
                    raise ValueError('md.csv columns differ; original data is preserved')
                writer = csv.DictWriter(stream, fieldnames=fields or list(row))
                if fields is None:
                    fields = list(row)
                    writer.writeheader()
                writer.writerow(row)
                stream.flush()
                last_csv = dyn.nsteps
            if dyn.nsteps > last_log:
                logger()
                last_log = dyn.nsteps

        record()  # Repair a missing final CSV/log row if a trajectory write completed first.
        dyn.attach(record, interval=LOGINTERVAL)
        steps = max(0, end-start)
        dyn.run(steps if max_steps is None else min(steps, max_steps))
        record()  # Save the final complete step even between log intervals.
    print(f'Saved through step {dyn.nsteps}; run again to continue if incomplete.', flush=True)


def load_atoms(source, index):
    cfg_path = Path('run.json')
    if not cfg_path.exists():
        if any(Path(p).exists() for p in ('md.traj', 'md.csv', 'HILLS')):
            raise ValueError('Existing outputs have no umbrella metadata; use a fresh folder')
        if source is None:
            raise FileNotFoundError('New runs require ATOMS_FILE')
        return prepare_frame(source, index)
    cfg = json.loads(cfg_path.read_text())
    if cfg.get('method') != 'distance_umbrella':
        raise ValueError('Method not supported')
    if cfg['source_index'] != index:
        raise ValueError('Restart folder and frame index differ')
    atoms = None
    if Path('md.traj').exists() and Path('md.traj').stat().st_size:
        with Trajectory('md.traj') as trajectory:
            if len(trajectory):
                atoms = trajectory[-1]
    if atoms is None:
        from ase import Atoms
        state = cfg['initial_state']
        atoms = Atoms(numbers=state['numbers'], positions=state['positions'], momenta=state['momenta'],
                      cell=state['cell'], pbc=state['pbc'], info=state['info'])
        atoms.info['restart_state'] = dict(run_id=cfg['run_id'], rng_state=cfg['initial_rng_state'],
                                          reference_model_potential=cfg['reference_model_potential'])
    if 'restart_state' not in atoms.info or not atoms.has('momenta'):
        raise ValueError('The trajectory lacks restart state or momenta')
    # This script's earlier runs used a Gaussian sheet under the center key.
    if 'COUNTER_CHARGE_CENTER' in cfg['settings'] and 'counter_charge_center' in atoms.info:
        center = np.asarray(atoms.info.pop('counter_charge_center'))
        atoms.info['counter_charge_zfrac'] = float((center @ np.linalg.inv(atoms.cell.array))[2] % 1.)
    if index in OXYGEN_PAIRS and set(OXYGEN_PAIRS[index]) != set(atoms.info['oxygen_pair']):
        raise ValueError('Cannot change the oxygen pair during continuation')
    atoms.set_constraint(physical_constraints(atoms, cfg['fixed_ids']))
    return atoms


def run_one(source, index, model_file, directory):
    Path(directory).mkdir(exist_ok=True)
    os.chdir(directory)
    atoms = load_atoms(source, index)  # Reject old runs before opening their output files.
    with open('run.log', 'a', buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        print('\n--- NVT-Phi umbrella start/continuation ---', flush=True)
        try:
            simulate(atoms, model_file)
        except Exception:
            traceback.print_exc()
            raise
    return index


def main():
    indexes = list(RUN_INDEXES)
    if not indexes or len(indexes) != len(set(indexes)) or any(type(i) is not int or i < 0 for i in indexes):
        raise ValueError('RUN_INDEXES must contain unique nonnegative integer indexes')
    if not 0 <= EQUIL_STEPS < WINDOW_STEPS or RELEASE_STEPS < 0:
        raise ValueError('Require 0 <= EQUIL_STEPS < WINDOW_STEPS and RELEASE_STEPS >= 0')
    if min(OO_MIN, WINDOW_SPACING, UMBRELLA_K, LOGINTERVAL, TIMESTEP_FS, TEMPERATURE_K) <= 0:
        raise ValueError('Widths, stiffness, temperature and time intervals must be positive')
    model_file = str(Path(MODEL_FILE).resolve(strict=True))
    frames = read(ATOMS_FILE, index=':') if Path(ATOMS_FILE).exists() else None
    if frames is not None and max(indexes) >= len(frames):
        raise ValueError(f'Frame indexes must lie in 0..{len(frames)-1}')
    directories = [str(Path(f'umbrella_{i}').resolve()) for i in indexes]
    failures = []
    with ProcessPoolExecutor(max_workers=len(indexes), mp_context=mp.get_context('spawn')) as pool:
        jobs = {pool.submit(run_one, frames[i] if frames is not None else None, i, model_file, d): i
                for i, d in zip(indexes, directories)}
        for job in as_completed(jobs):
            i = jobs[job]
            try:
                job.result()
                print(f'umbrella_{i}: complete', flush=True)
            except Exception as error:
                failures.append(i)
                print(f'umbrella_{i}: failed ({error}); see run.log', flush=True)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')


if __name__ == '__main__':
    main()
