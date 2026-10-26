import csv
import hashlib
import json
import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from ase import units
from ase.constraints import FixAtoms, FixConstraint, Hookean
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# ------------------------------ user settings ------------------------------ #
ATOMS_FILE = 'IrO2OH_110_mdinit_0_51.xyz'
MODEL_FILE = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/ml_training/wf_training_209/models/fit_0_best_wf.model'
OUTPUT_ROOT = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/nvtphi_umbrella'
RUN_INDEXES = list(range(0, 10))                # A list of frame indexes.
OXYGEN_PAIRS = {}           # Optional {index: (O_id, O_id)}, zero-based ASE indexes.
MAX_WORKERS, REPLICAS = len(RUN_INDEXES), 1                    # Workers share one visible GPU.
DEVICE, HEAD, ENABLE_CUEQ, SCF_STEPS = 'cuda:0', None, True, 50
TARGET_U_SHE = 1.60                             # V vs SHE: <WF> = 4.44 eV + e U*.
SHE_OFFSET = None                               # <WF - eV> in V from an earlier run; None aligns online.
SHEET_HEIGHT, GAUSSIAN_WIDTH = 6.0, 1.0         # A above the highest Ir-bound O; sigma of the DFT data.
CAPACITANCE, TAUPHI_FS = 0.05, 100.0            # C0 (e/V, whole cell) sets only the charge relaxation.
FIXED_DEPTH, WALL_GAP, WALL_K = 6.5, 3.0, 10.0  # A, A, eV/A^2.
O2_BOND, PAIR_CUTOFF = 1.60, 3.00               # A: an existing O2; partner search in other frames.
R_MIN, R_MAX, WINDOW_SPACING = 1.20, 3.00, 0.10  # A: one grid of window centers.
UMBRELLA_K = 20.0                               # eV/A^2: thermal width 0.036 A at 300 K.
INITIAL_EQUIL_STEPS = 3000                      # 1.5 ps: equilibrates the added water, aligns V*.
ALIGNMENT_CONSTANT_FS = 1000.0                  # Memory of the online <WF - eV> average.
EQUIL_STEPS, SAMPLE_STEPS = 500, 1000           # 0.25 ps + 0.5 ps per window.
RETURN_SWEEP = False                            # Sweep back afterwards to check hysteresis.
TEMPERATURE_K, TIMESTEP_FS, FRICTION_PER_FS = 300.0, 0.5, 0.01
LOGINTERVAL, SEED = 20, 7777                    # 10 fs records; each record is a restart point.
MAX_TEMPERATURE_K, MAX_FORCE_EV_A = 1000.0, 100.0
CHECK_CHARGE_RESPONSE, CHARGE_PROBE_E = True, 0.002  # dE/dQ against finite differences before MD.
# -------------------------------------------------------------------------- #
PHYSICAL = ('HEAD SCF_STEPS TARGET_U_SHE SHE_OFFSET SHEET_HEIGHT GAUSSIAN_WIDTH CAPACITANCE TAUPHI_FS '
            'FIXED_DEPTH WALL_GAP WALL_K O2_BOND PAIR_CUTOFF R_MIN R_MAX WINDOW_SPACING UMBRELLA_K '
            'INITIAL_EQUIL_STEPS ALIGNMENT_CONSTANT_FS EQUIL_STEPS SAMPLE_STEPS RETURN_SWEEP TEMPERATURE_K '
            'TIMESTEP_FS FRICTION_PER_FS LOGINTERVAL SEED').split()


def scalar(value):
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f'Expected one finite scalar, got {value}')
    return float(a.item())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=lambda v: np.asarray(v).tolist()).encode()).hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, default=lambda v: np.asarray(v).tolist(), allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def joined(source):
    """A copy with the slab continuous along z, cut in the middle of its largest gap."""
    atoms = source.copy()
    c = atoms.cell[2, 2]
    f = atoms.get_scaled_positions()[:, 2]
    s = np.sort(f)
    gaps = np.diff(np.append(s, s[0] + 1.0))
    cut = s[np.argmax(gaps)] + 0.5 * gaps.max()
    z = ((f - cut) % 1.0 + cut) * c
    atoms.positions[:, 2] = z + c * np.rint(np.mean(atoms.positions[:, 2] - z) / c)
    return atoms


def environment(atoms, oxygen):
    symbols = np.asarray(atoms.get_chemical_symbols())
    di = atoms.get_distances(oxygen, np.flatnonzero(symbols == 'Ir'), mic=True)
    dh = atoms.get_distances(oxygen, np.flatnonzero(symbols == 'H'), mic=True)
    return dict(H_neighbors=int((dh < 1.25).sum()), Ir_neighbors=int((di < 2.60).sum()), nearest_Ir_A=float(di.min()))


def o2_pairs(atoms, mobile):
    d = atoms.get_all_distances(mic=True)
    return [(int(a), int(b)) for k, a in enumerate(mobile) for b in mobile[k + 1:] if d[a, b] < O2_BOND]


def select_pair(atoms, index, mobile):
    if index in OXYGEN_PAIRS:
        pair = sorted(int(i) for i in OXYGEN_PAIRS[index])
        if not set(pair) <= set(mobile) or len(set(pair)) != 2:
            raise ValueError(f'{index}: OXYGEN_PAIRS must name two different mobile O')
        return pair, 'override'
    bonded = o2_pairs(atoms, mobile)
    if len(bonded) > 1:
        raise ValueError(f'{index}: several O-O bonds {bonded}; choose one in OXYGEN_PAIRS')
    if bonded:
        return list(bonded[0]), 'O2'
    top = atoms.positions[np.asarray(atoms.get_chemical_symbols()) == 'Ir', 2].max()
    env = {o: environment(atoms, o) for o in mobile}
    adsorbed = [o for o in mobile if env[o]['Ir_neighbors'] == 1 and env[o]['H_neighbors'] <= 1
                and atoms.positions[o, 2] > top + 0.3]
    free = [o for o in mobile if env[o]['Ir_neighbors'] == 0]
    pairs = [(atoms.get_distance(a, b, mic=True), a, b) for a in adsorbed for b in free]
    pairs = [p for p in pairs if p[0] <= PAIR_CUTOFF]
    if not pairs:
        raise ValueError(f'{index}: no exposed adsorbate O with a free O within PAIR_CUTOFF; set OXYGEN_PAIRS')
    return sorted(int(o) for o in min(pairs)[1:]), 'adsorbate + free O'


def prepare(source, index):
    """The starting atoms with their run info, fixed atoms and wall height, from geometry only."""
    atoms = joined(source)
    if not np.array_equal(atoms.pbc, [True, True, False]) or not np.allclose(atoms.cell.array[:2, 2], 0.):
        raise ValueError(f'{index}: expected an xy-periodic slab, open along Cartesian z')
    symbols, z, c = np.asarray(atoms.get_chemical_symbols()), atoms.positions[:, 2], atoms.cell[2, 2]
    bottom, top = z.min(), z.max()
    if c - (top - bottom) < 12.0:
        raise ValueError(f'{index}: only {c - top + bottom:.1f} A of vacuum')
    ir = np.flatnonzero(symbols == 'Ir')
    oxygens = np.flatnonzero(symbols == 'O')
    surface = max(z[o] for o in oxygens if (atoms.get_distances(o, ir, mic=True) < 2.3).any())
    fixed = [int(i) for i in np.flatnonzero(z < bottom + FIXED_DEPTH)]
    mobile = [int(o) for o in oxygens if o not in set(fixed)]
    pair, kind = select_pair(atoms, index, mobile)
    dipole = top + 0.5 * (c - top + bottom)
    atoms.info = dict(source_index=index, oxygen_pair=pair, pair_kind=kind, total_charge=0.0, counter_charge=0.0,
                      counter_charge_zfrac=float((surface + SHEET_HEIGHT) / c % 1.0),
                      counter_charge_width=float(GAUSSIAN_WIDTH), dipole_correction_zfrac=float(dipole / c % 1.0),
                      vacuum_zfrac=float(0.5 * (top + dipole) / c % 1.0), external_field=np.zeros(3))
    atoms.calc = None
    atoms.set_constraint()
    return atoms, fixed, float(z[oxygens].max() + WALL_GAP)


def restraints(atoms, fixed, wall):
    oxygens = set(np.flatnonzero(np.asarray(atoms.get_chemical_symbols()) == 'O')) - set(fixed)
    return [FixAtoms(indices=fixed)] + [Hookean(a1=int(o), a2=(0., 0., 1., -wall), k=WALL_K) for o in sorted(oxygens)]


class DistanceUmbrella(FixConstraint):
    """Bias V = k/2 (d - center)^2 on the minimum-image distance d of two atoms."""
    def __init__(self, pair, center, stiffness):
        self.pair, self.center, self.k = list(pair), float(center), float(stiffness)

    def evaluate(self, atoms):
        vector = atoms.get_distance(*self.pair, mic=True, vector=True)
        distance = float(np.linalg.norm(vector))
        radial = -self.k * (distance - self.center)
        forces = np.zeros((len(atoms), 3))
        forces[self.pair[1]] = radial * vector / distance
        forces[self.pair[0]] = -forces[self.pair[1]]
        return forces, 0.5 * self.k * (distance - self.center) ** 2, distance, radial

    def adjust_forces(self, atoms, forces):
        forces += self.evaluate(atoms)[0]

    def adjust_potential_energy(self, atoms):
        return self.evaluate(atoms)[1]

    def adjust_momenta(self, atoms, momenta):
        pass

    def get_removed_dof(self, atoms):
        return 0


def make_stages(distance):
    """Initial stage at the starting distance, then windows away from it on the common grid."""
    grid = np.round(np.arange(R_MIN, R_MAX + 1e-9, WINDOW_SPACING), 6)
    if not R_MIN <= distance <= R_MAX:
        raise ValueError(f'Starting O-O distance {distance:.2f} A lies outside R_MIN..R_MAX')
    start, up = int(np.argmin(np.abs(grid - distance))), distance < 0.5 * (R_MIN + R_MAX)
    centers = grid[start:] if up else grid[start::-1]
    sweeps = [('cleavage' if up else 'formation', centers)]
    if RETURN_SWEEP:
        sweeps.append(('formation' if up else 'cleavage', centers[::-1]))
    stages = [dict(direction='initial', window=-1, phase='equil', center_A=float(distance), steps=INITIAL_EQUIL_STEPS)]
    for direction, sweep in sweeps:
        for window, center in enumerate(sweep):
            for phase, steps in (('equil', EQUIL_STEPS), ('sample', SAMPLE_STEPS)):
                stages.append(dict(direction=direction, window=window, phase=phase, center_A=float(center), steps=steps))
    return stages


def make_calculator(model_file, source):
    import torch
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    torch.set_num_threads(1)
    options = dict(constant_charge=True, use_autograd_forces=True, initial_density='local_guess',
                   initial_fermi_level='zero', **({} if SCF_STEPS is None else dict(num_scf_steps=int(SCF_STEPS))))
    calc = MACEFixedPointSCF(model_path=model_file, device=DEVICE, default_dtype='float64', pbc_handling='slab',
                             scf_options=options, scf_restart=False, ignore_nonconverged=True, use_compile=False,
                             counter_charge_width=GAUSSIAN_WIDTH, constant_potential=True, neutral_atoms=source)
    heads = list(getattr(calc.model, 'heads', [calc.head]))
    head = HEAD or (heads[0] if len(heads) == 1 else next((h for h in heads if h.lower() == 'default'), None))
    if head not in heads:
        raise ValueError(f'Set HEAD to the electrochemistry head; model heads are {heads}')
    calc.head = head
    if ENABLE_CUEQ:
        from mace_scf.utils.foundation import accelerate_backbone
        accelerate_backbone(calc.model, DEVICE)
    calc.model.eval().requires_grad_(False)
    return calc


def charge_probe(atoms):
    """Frozen geometry at Q and Q +- dQ: dE/dQ must equal the cell voltage, and dV/dQ > 0."""
    q, data = scalar(atoms.info['total_charge']), []
    try:
        for dq in (-CHARGE_PROBE_E, 0., CHARGE_PROBE_E):
            atoms.info.update(total_charge=q + dq, counter_charge=-q - dq)
            atoms.calc.reset()
            energy = scalar(atoms.get_potential_energy(apply_constraint=False))
            data.append([energy, scalar(atoms.calc.results['cell_voltage']), scalar(atoms.calc.results['work_function'])])
    finally:
        atoms.info.update(total_charge=q, counter_charge=-q)
        atoms.calc.reset()
    derivative, dv, dwf = (np.asarray(data[2]) - np.asarray(data[0])) / (2 * CHARGE_PROBE_E)
    result = dict(charge_e=q, cell_voltage_V=data[1][1], dE_dQ_fd_V=float(derivative), work_function_eV=data[1][2],
                  she_offset_V=data[1][2] - data[1][1], dV_dQ_V_per_e=float(dv), dWF_dQ_V_per_e=float(dwf))
    if abs(derivative - data[1][1]) > 0.005 or dv <= 0.:
        raise RuntimeError(f'Charge response fails: {result}')
    return result


def last_row(path):
    if not path.exists():
        return None, None
    with path.open(newline='') as f:
        reader, previous, row = csv.DictReader(f), -1, None
        for row in reader:
            if None in row or None in row.values() or int(row['step']) <= previous:
                raise ValueError(f'{path}: incomplete or nonmonotonic; preserve it and recover before resuming')
            previous = int(row['step'])
        return reader.fieldnames, row


def simulate(source, index, replica, model_file, model_sha, directory):
    from mace_scf.md import NVTPhiLangevin
    atoms, fixed, wall = prepare(source, index)
    pair = atoms.info['oxygen_pair']
    stages = make_stages(atoms.get_distance(*pair, mic=True))
    identity = dict(method='fixed_window_OO', settings={k: globals()[k] for k in PHYSICAL}, model_sha256=model_sha,
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), replica=replica,
                    initial=dict(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell.array,
                                 pbc=atoms.pbc, info=atoms.info, fixed_ids=fixed, wall_A=wall), stages=stages)
    run_id = digest(identity)
    config, traj_path, csv_path = (directory / name for name in ('run.json', 'md.traj', 'md.csv'))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index, replica]))
    state = None
    if config.exists():
        if json.loads(config.read_text()).get('run_id') != run_id:
            raise ValueError('Source, model or physical settings differ from this run; use a fresh output folder')
        if traj_path.exists():
            with Trajectory(traj_path) as previous:
                if len(previous):
                    atoms = previous[-1]
                    state = atoms.info.pop('restart_state')
                    rng.bit_generator.state = state['rng_state']
    elif traj_path.exists() or csv_path.exists():
        raise ValueError('Output exists without matching run metadata; refusing to overwrite')
    else:
        write_json(config, dict(identity, run_id=run_id, model_file=str(model_file),
                                pair_environments=[environment(atoms, o) for o in pair],
                                proton_ensemble='Fixed atom counts; pH is not imposed by this driver.'))
    physical = restraints(atoms, fixed, wall)
    atoms.set_constraint(physical)
    atoms.calc = make_calculator(model_file, source)
    if state is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, force_temp=True, rng=rng)
        if CHECK_CHARGE_RESPONSE:
            write_json(directory / 'charge_response.json', charge_probe(atoms))
    fields, saved = last_row(csv_path)
    last_step = -1 if saved is None else int(saved['step'])
    start = 0 if state is None else int(atoms.info['step'])
    if last_step > start or (state is None and last_step >= 0):
        raise ValueError('CSV is ahead of the restart trajectory; outputs have been preserved')
    stage_id, done = (0, 0) if state is None else (state['stage_id'], state['stage_done'])
    bias = DistanceUmbrella(pair, stages[stage_id]['center_A'], UMBRELLA_K)
    atoms.set_constraint([bias] + physical)
    with ExitStack() as stack:
        dyn = stack.enter_context(NVTPhiLangevin(
            atoms, timestep=TIMESTEP_FS * units.fs, temperature_K=TEMPERATURE_K, friction=FRICTION_PER_FS / units.fs,
            target_potential=TARGET_U_SHE, she_offset=SHE_OFFSET, alignment_time=INITIAL_EQUIL_STEPS * TIMESTEP_FS * units.fs,
            alignment_constant=ALIGNMENT_CONSTANT_FS * units.fs, C0=CAPACITANCE, tauphi=TAUPHI_FS * units.fs,
            fixcm=False, rng=rng, logfile=str(directory / 'md.log'), loginterval=LOGINTERVAL))
        dyn.nsteps = start
        if state is not None:
            dyn.restore_potentiostat_state(state['potentiostat'])
        trajectory = stack.enter_context(Trajectory(traj_path, 'a'))
        stream = stack.enter_context(csv_path.open('a', newline='', buffering=1))
        segment_start = dyn.nsteps - done

        def write_row(row):
            nonlocal fields, last_step
            if fields is not None and fields != list(row):
                raise ValueError('CSV columns differ; refusing to append')
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if fields is None:
                writer.writeheader()
                fields = list(row)
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
            last_step = int(row['step'])

        if state is not None and last_step < start:
            write_row(atoms.info['md_record'])  # The trajectory was written, the CSV row was not.

        def record():
            if dyn.nsteps <= last_step:
                return
            stage, r = stages[stage_id], atoms.calc.results
            energy = scalar(atoms.get_potential_energy(apply_constraint=False))
            _, umbrella, distance, radial = bias.evaluate(atoms)
            wall_energy = sum(c.adjust_potential_energy(atoms) for c in physical if isinstance(c, Hookean))
            forces, temperature = atoms.get_forces(md=True), scalar(atoms.get_temperature())
            if (not np.isfinite(forces).all() or temperature > MAX_TEMPERATURE_K
                    or np.linalg.norm(forces, axis=1).max() > MAX_FORCE_EV_A):
                raise RuntimeError('Nonfinite or excessive force or temperature; inspect the last saved state')
            row = dict(step=dyn.nsteps, time_fs=dyn.get_time() / units.fs, direction=stage['direction'],
                       window=stage['window'], phase=stage['phase'], phase_step=dyn.nsteps - segment_start,
                       distance_A=distance, center_A=bias.center, k_eV_A2=bias.k, bias_eV=umbrella,
                       radial_bias_eV_A=radial, energy_model_eV=energy, energy_wall_eV=wall_energy,
                       energy_total_eV=energy + wall_energy + umbrella, temperature_K=temperature,
                       total_charge_e=dyn.electrode_charge, cell_voltage_V=scalar(r['cell_voltage']),
                       target_cell_voltage_V=dyn.target_cell_voltage, wf_eV=scalar(r['work_function']),
                       U_SHE_V=scalar(r['electrode_potential_she']), she_offset_V=scalar(r['she_offset']),
                       aligned=int(dyn.aligned), omega_eV=energy - dyn.target_cell_voltage * dyn.electrode_charge,
                       vacuum_potential_V=scalar(r['vacuum_potential']), fermi_level_eV=scalar(r['fermi_level']),
                       scf_steps=int(r['num_scf_steps']), scf_residual=scalar(np.max(np.asarray(r['scf_residual']))))
            for j, oxygen in enumerate(pair, 1):
                row.update({f'O{j}_{key}': value for key, value in environment(atoms, oxygen).items()})
            frame = atoms.copy()
            frame.set_constraint([])  # Stored energy and forces include the restraints once.
            frame.info.update(step=dyn.nsteps, md_record=row, restart_state=dict(
                rng_state=rng.bit_generator.state, potentiostat=dyn.potentiostat_state(),
                stage_id=stage_id, stage_done=dyn.nsteps - segment_start))
            trajectory.write(frame, energy=energy + wall_energy + umbrella, forces=forces)
            write_row(row)

        record()
        dyn.attach(record, interval=LOGINTERVAL)
        while stage_id < len(stages):
            stage = stages[stage_id]
            bias.center = stage['center_A']  # Changed only between complete integration steps.
            segment_start = dyn.nsteps - done
            if stage['steps'] > done:
                print(f'{index}/{replica} {stage["direction"]} w{stage["window"]} {stage["phase"]}: '
                      f'center {bias.center:.2f} A, {stage["steps"] - done} steps', flush=True)
                dyn.run(stage['steps'] - done)
                record()
            stage_id, done = stage_id + 1, 0
    print(f'Complete at step {dyn.nsteps}', flush=True)


def run_one(source, index, replica, model_file, model_sha, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'running.lock'
    os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))  # One process per output folder.
    try:
        with open(directory / 'run.log', 'a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                simulate(source, index, replica, model_file, model_sha, directory)
            except Exception:
                traceback.print_exc()
                raise
    finally:
        lock.unlink(missing_ok=True)
    return index, replica


def prepared_model():
    """The model file to run and its hash; a model without neutral atoms fits them from its source frame."""
    model = Path(MODEL_FILE).resolve(strict=True)
    hasher = hashlib.sha256()
    with model.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            hasher.update(chunk)
    return model, hasher.hexdigest()


def main():
    if min(MAX_WORKERS, REPLICAS, INITIAL_EQUIL_STEPS, EQUIL_STEPS, SAMPLE_STEPS, LOGINTERVAL) <= 0 or min(
            R_MIN, WINDOW_SPACING, UMBRELLA_K, TEMPERATURE_K, TIMESTEP_FS, CAPACITANCE, TAUPHI_FS) <= 0 or R_MAX <= R_MIN:
        raise ValueError('Step counts and physical scales must be positive, with R_MAX > R_MIN')
    frames = read(ATOMS_FILE, index=':')
    with_o2 = []
    for index, source in enumerate(frames):
        atoms = joined(source)
        mobile = [int(o) for o in np.flatnonzero(atoms.numbers == 8)
                  if atoms.positions[o, 2] >= atoms.positions[:, 2].min() + FIXED_DEPTH]
        if o2_pairs(atoms, mobile):
            with_o2.append(index)
    print(f'Frames that already hold an O2: {with_o2}', flush=True)
    indexes = list(RUN_INDEXES)
    if not indexes or len(set(indexes)) != len(indexes) or not set(indexes) <= set(range(len(frames))):
        raise ValueError('RUN_INDEXES must list distinct frames of ATOMS_FILE')
    for index in indexes:
        atoms, _, wall = prepare(frames[index], index)
        pair, stages = atoms.info['oxygen_pair'], make_stages(atoms.get_distance(*atoms.info['oxygen_pair'], mic=True))
        print(f'{index}: {atoms.info["pair_kind"]} pair {pair} at {atoms.get_distance(*pair, mic=True):.2f} A, '
              f'{stages[-1]["direction"]} over {stages[-1]["window"] + 1} windows, '
              f'{sum(s["steps"] for s in stages) * TIMESTEP_FS / 1000:.1f} ps; sheet z/c '
              f'{atoms.info["counter_charge_zfrac"]:.3f}, wall {wall:.1f} A', flush=True)
    model, model_sha = prepared_model()
    failures = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(indexes) * REPLICAS),
                             mp_context=mp.get_context('spawn')) as pool:
        jobs = {pool.submit(run_one, frames[i], i, rep, str(model), model_sha,
                            (Path(OUTPUT_ROOT) / f'umbrella_{i}' / f'replica_{rep}').resolve()): (i, rep)
                for rep in range(REPLICAS) for i in indexes}
        for job in as_completed(jobs):
            try:
                job.result()
                print(f'{jobs[job]}: complete', flush=True)
            except Exception as error:
                failures.append(jobs[job])
                print(f'{jobs[job]}: failed: {error}; see run.log', flush=True)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')


if __name__ == '__main__':
    main()
