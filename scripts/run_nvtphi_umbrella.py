#!/usr/bin/env python3
"""NVTPhi umbrella sampling of one O-O distance per frame at constant electrode potential.

Each run has three stages, all under the O-O umbrella and the restraints:
  relax    Langevin MD at Q = 0 at the starting distance, averaging Delta = WF - eV
           (the step from the counter charge to the vacuum);
  align    thermopotentiostat at V* = 4.44 V + U* - <Delta>, with <Delta> still
           averaged online and then frozen, so that <U_SHE> = U*;
  windows  equilibrate, then sample, each umbrella center from the start onward.
The counter charge -Q is a Gaussian sheet SHEET_HEIGHT above the electrode surface
in the water. It stands for solvated ions and acts through electrostatics only.
Frames run in parallel processes on one GPU. Every record is a restart point, so
rerunning the script resumes each run from its last record.
"""
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
MODEL_FILE = 'fit_0_best_wf_final.model'
OUTPUT_ROOT = 'nvtphi_umbrella'
RUN_INDEXES = list(range(0, 10))                # Frames of ATOMS_FILE to run.
OXYGEN_PAIRS = {}                               # Optional {index: (O_id, O_id)}, zero-based.
MAX_WORKERS, REPLICAS = len(RUN_INDEXES), 1     # Workers share one visible GPU.
DEVICE, HEAD, ENABLE_CUEQ, SCF_STEPS = 'cuda:0', None, True, None  # None: the model's training steps.
TARGET_U_SHE = 1.60                             # V vs SHE: <WF> = 4.44 eV + e U*.
SHEET_HEIGHT, SHEET_WIDTH = 6.0, 1.0            # A above the electrode surface, Gaussian sigma.
CAPACITANCE, TAUPHI_FS = 0.05, 100.0            # C0 (e/V, whole cell) and tau set only the charge relaxation.
FIXED_DEPTH, WALL_GAP, WALL_K = 6.5, 3.0, 10.0  # A, A, eV/A^2.
O2_BOND, PAIR_CUTOFF = 1.60, 3.50               # A: an existing O2; adsorbate O to free O.
R_MIN, R_MAX, WINDOW_SPACING = 1.20, 3.50, 0.10  # A: one grid of umbrella centers.
UMBRELLA_K = 20.0                               # eV/A^2: thermal width 0.036 A at 300 K.
RELAX_STEPS, ALIGN_STEPS = 1000, 3000           # 0.5 ps at Q = 0, then 1.5 ps aligning <Delta>.
ALIGNMENT_CONSTANT_FS = 250.0                   # Memory of the online <Delta>.
EQUIL_STEPS, SAMPLE_STEPS = 500, 1000           # 0.25 ps + 0.5 ps per window.
TEMPERATURE_K, TIMESTEP_FS, FRICTION_PER_FS = 300.0, 0.5, 0.01
LOGINTERVAL, SEED = 20, 7777                    # 10 fs records; stage lengths are multiples of it.
MAX_CHARGE_E, MAX_TEMPERATURE_K, MAX_FORCE_EV_A = 1.0, 1000.0, 100.0
# -------------------------------------------------------------------------- #
PHYSICAL = ('HEAD SCF_STEPS TARGET_U_SHE SHEET_HEIGHT SHEET_WIDTH CAPACITANCE TAUPHI_FS FIXED_DEPTH WALL_GAP '
            'WALL_K O2_BOND PAIR_CUTOFF R_MIN R_MAX WINDOW_SPACING UMBRELLA_K RELAX_STEPS ALIGN_STEPS '
            'ALIGNMENT_CONSTANT_FS EQUIL_STEPS SAMPLE_STEPS TEMPERATURE_K TIMESTEP_FS FRICTION_PER_FS '
            'LOGINTERVAL SEED').split()


def plain(value):
    return np.asarray(value).tolist()


def sha256(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=1, default=plain, allow_nan=False))
    os.replace(temporary, path)


def continuous(source):
    """The slab open along z and continuous across its largest vacuum gap."""
    atoms = source.copy()
    if not np.allclose(atoms.cell.array[:2, 2], 0.):
        raise ValueError('Expected an xy-periodic slab, open along Cartesian z')
    atoms.pbc, atoms.calc = [True, True, False], None
    atoms.set_constraint()
    c, f = atoms.cell[2, 2], atoms.get_scaled_positions()[:, 2]
    s = np.sort(f)
    gaps = np.diff(np.append(s, s[0]+1.))
    cut = s[np.argmax(gaps)]+.5*gaps.max()
    z = ((f-cut) % 1.+cut)*c
    atoms.positions[:, 2] = z+c*np.rint(np.mean(atoms.positions[:, 2]-z)/c)
    return atoms


def environment(atoms, oxygen):
    di = atoms.get_distances(oxygen, np.flatnonzero(atoms.numbers == 77), mic=True)
    dh = atoms.get_distances(oxygen, np.flatnonzero(atoms.numbers == 1), mic=True)
    return dict(H_neighbors=int((dh < 1.25).sum()), Ir_neighbors=int((di < 2.6).sum()), nearest_Ir_A=float(di.min()))


def select_pair(atoms, index, mobile):
    """An existing O2, else the exposed adsorbate O with the nearest free O within PAIR_CUTOFF."""
    if index in OXYGEN_PAIRS:
        pair = sorted(int(i) for i in OXYGEN_PAIRS[index])
        if len(set(pair)) != 2 or not set(pair) <= set(mobile):
            raise ValueError(f'{index}: OXYGEN_PAIRS must name two different mobile O')
        return pair, 'chosen'
    d = atoms.get_all_distances(mic=True)
    bonded = [(a, b) for k, a in enumerate(mobile) for b in mobile[k+1:] if d[a, b] < O2_BOND]
    if len(bonded) > 1:
        raise ValueError(f'{index}: several O-O bonds {bonded}; choose one in OXYGEN_PAIRS')
    if bonded:
        return sorted(bonded[0]), 'O2'
    top = atoms.positions[atoms.numbers == 77, 2].max()
    env = {o: environment(atoms, o) for o in mobile}
    adsorbed = [o for o in mobile if env[o]['Ir_neighbors'] == 1 and env[o]['H_neighbors'] <= 1
                and atoms.positions[o, 2] > top+.3]
    free = [o for o in mobile if env[o]['Ir_neighbors'] == 0]
    pairs = [(d[a, b], a, b) for a in adsorbed for b in free if d[a, b] <= PAIR_CUTOFF]
    if not pairs:
        raise ValueError(f'{index}: no exposed adsorbate O has a free O within PAIR_CUTOFF; set OXYGEN_PAIRS')
    return sorted(int(o) for o in min(pairs)[1:]), 'adsorbate + free O'


def prepare(source, index):
    """Atoms at Q = 0 with the planes from geometry; fixed atoms, wall height and O-O pair."""
    from mace_scf.electrostatics.reservoir import slab_controls
    atoms = continuous(source)
    z = atoms.positions[:, 2]
    if atoms.cell[2, 2]-np.ptp(z) < 12.:
        raise ValueError(f'{index}: only {atoms.cell[2, 2]-np.ptp(z):.1f} A of vacuum')
    fixed = [int(i) for i in np.flatnonzero(z < z.min()+FIXED_DEPTH)]
    mobile = [int(o) for o in np.flatnonzero(atoms.numbers == 8) if o not in set(fixed)]
    pair, kind = select_pair(atoms, index, mobile)
    atoms.info = dict(total_charge=0., counter_charge=0.,
                      **slab_controls(atoms, sheet_height=SHEET_HEIGHT, sheet_width=SHEET_WIDTH))
    return atoms, fixed, mobile, float(z[mobile].max()+WALL_GAP), pair, kind


class DistanceUmbrella(FixConstraint):
    """Bias k/2 (d - center)^2 on the minimum-image distance d of two atoms."""
    def __init__(self, pair, center, stiffness):
        self.pair, self.center, self.k = list(pair), float(center), float(stiffness)

    def evaluate(self, atoms):
        vector = atoms.get_distance(*self.pair, mic=True, vector=True)
        distance = float(np.linalg.norm(vector))
        forces = np.zeros((len(atoms), 3))
        forces[self.pair[1]] = -self.k*(distance-self.center)*vector/distance
        forces[self.pair[0]] = -forces[self.pair[1]]
        return forces, .5*self.k*(distance-self.center)**2, distance

    def adjust_forces(self, atoms, forces):
        forces += self.evaluate(atoms)[0]

    def adjust_potential_energy(self, atoms):
        return self.evaluate(atoms)[1]

    def adjust_momenta(self, atoms, momenta):
        pass

    def get_removed_dof(self, atoms):
        return 0


def make_stages(distance):
    """Relax and align at the starting distance, then the windows away from it on one grid."""
    grid = np.round(np.arange(R_MIN, R_MAX+1e-9, WINDOW_SPACING), 6)
    if not R_MIN <= distance <= R_MAX:
        raise ValueError(f'Starting O-O distance {distance:.2f} A lies outside R_MIN..R_MAX')
    start = int(np.argmin(np.abs(grid-distance)))
    direction, centers = (('cleavage', grid[start:]) if distance < .5*(R_MIN+R_MAX) else
                          ('formation', grid[start::-1]))
    stages = [dict(stage='relax', window=-1, center_A=float(distance), steps=RELAX_STEPS),
              dict(stage='align', window=-1, center_A=float(distance), steps=ALIGN_STEPS)]
    for window, center in enumerate(centers):
        stages += [dict(stage=f'{direction} {phase}', window=window, center_A=float(center), steps=steps)
                   for phase, steps in (('equil', EQUIL_STEPS), ('sample', SAMPLE_STEPS))]
    if any(stage['steps'] % LOGINTERVAL for stage in stages):
        raise ValueError('Stage lengths must be multiples of LOGINTERVAL')
    return stages


def make_calculator(model_file):
    import torch
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    torch.set_num_threads(1)
    options = dict(constant_charge=True, use_autograd_forces=True, initial_density='local_guess',
                   initial_fermi_level='zero')
    if SCF_STEPS is not None:
        options['num_scf_steps'] = int(SCF_STEPS)
    calc = MACEFixedPointSCF(model_path=model_file, device=DEVICE, default_dtype='float64', pbc_handling='slab',
                             scf_options=options, scf_restart=False, ignore_nonconverged=True, use_compile=False,
                             counter_charge_height=SHEET_HEIGHT, counter_charge_width=SHEET_WIDTH,
                             constant_potential=True)
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


def charge_check(atoms, dq=2.e-3):
    """At Q = 0, dE/dQ by central difference must equal the cell voltage, and dV/dQ must be positive."""
    values = []
    for q in (-dq, 0., dq):
        atoms.info.update(total_charge=q, counter_charge=-q)
        atoms.calc.reset()
        energy = atoms.get_potential_energy(apply_constraint=False)
        values.append((energy, atoms.calc.results['cell_voltage'], atoms.calc.results['work_function']))
    atoms.info.update(total_charge=0., counter_charge=0.)
    atoms.calc.reset()
    (em, vm, wm), (_, v, w), (ep, vp, wp) = values
    result = dict(cell_voltage_V=v, dE_dQ_fd_V=(ep-em)/(2*dq), work_function_eV=w, delta_V=w-v,
                  dV_dQ_V_per_e=(vp-vm)/(2*dq), dWF_dQ_V_per_e=(wp-wm)/(2*dq))
    if abs(result['dE_dQ_fd_V']-v) > 5.e-3 or result['dV_dQ_V_per_e'] <= 0.:
        raise RuntimeError(f'Charge response fails: {result}')
    return result


def simulate(source, index, replica, model_file, model_sha, directory):
    from mace_scf.md import NVTPhiLangevin
    atoms, fixed, mobile, wall, pair, kind = prepare(source, index)
    stages = make_stages(atoms.get_distance(*pair, mic=True))
    identity = dict(settings={k: globals()[k] for k in PHYSICAL}, model_sha256=model_sha, replica=replica,
                    driver_sha256=sha256(__file__), pair=pair, fixed=fixed, wall_A=wall, stages=stages,
                    initial=dict(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell.array,
                                 info=atoms.info))
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, default=plain).encode()).hexdigest()
    config, traj_path, csv_path = (directory/name for name in ('run.json', 'md.traj', 'md.csv'))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index, replica]))
    state = None
    if config.exists():
        if json.loads(config.read_text()).get('run_id') != run_id:
            raise ValueError('Source, model or settings differ from this run; use a fresh output folder')
        if traj_path.exists():
            with Trajectory(traj_path) as previous:
                if len(previous):
                    atoms = previous[-1]
                    state = atoms.info.pop('restart')
    elif traj_path.exists() or csv_path.exists():
        raise ValueError('Output exists without run.json; refusing to overwrite')
    else:
        write_json(config, dict(identity, run_id=run_id, pair_kind=kind, model_file=str(model_file)))
    stage_id, done = (0, 0) if state is None else (state['stage'], state['done'])
    if state is not None:
        rng.bit_generator.state = state['rng']
        if done >= stages[stage_id]['steps']:
            stage_id, done = stage_id+1, 0
        if csv_path.exists():
            # The trajectory is the record of the run; drop CSV rows written after its last frame.
            with csv_path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                fields, kept = reader.fieldnames, [row for row in reader if int(row['step']) <= state['step']]
            with csv_path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(kept)
    bias = DistanceUmbrella(pair, stages[min(stage_id, len(stages)-1)]['center_A'], UMBRELLA_K)
    walls = [Hookean(a1=o, a2=(0., 0., 1., -wall), k=WALL_K) for o in mobile]
    atoms.set_constraint([bias, FixAtoms(indices=fixed)]+walls)
    atoms.calc = make_calculator(model_file)
    if state is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, force_temp=True, rng=rng)
        write_json(directory/'charge_response.json', charge_check(atoms))
    with ExitStack() as stack:
        dyn = stack.enter_context(NVTPhiLangevin(
            atoms, timestep=TIMESTEP_FS*units.fs, temperature_K=TEMPERATURE_K, friction=FRICTION_PER_FS/units.fs,
            target_potential=TARGET_U_SHE, alignment_time=(RELAX_STEPS+ALIGN_STEPS)*TIMESTEP_FS*units.fs,
            alignment_constant=ALIGNMENT_CONSTANT_FS*units.fs, C0=CAPACITANCE, tauphi=TAUPHI_FS*units.fs,
            fixcm=False, rng=rng, logfile=str(directory/'md.log'), loginterval=LOGINTERVAL))
        if state is not None:
            dyn.nsteps = state['step']
            dyn.restore_potentiostat_state(state['potentiostat'])
        trajectory = stack.enter_context(Trajectory(traj_path, 'a'))
        stream = stack.enter_context(csv_path.open('a', newline='', buffering=1))
        writer, last = None, -1 if state is None else state['step']
        start = dyn.nsteps-done

        def record():
            nonlocal writer, last
            if dyn.nsteps <= last:
                return  # ASE also observes step 0 of a fresh run.
            last = dyn.nsteps
            stage, r = stages[stage_id], atoms.calc.results
            energy = float(atoms.get_potential_energy(apply_constraint=False))
            forces, temperature, charge = atoms.get_forces(md=True), float(atoms.get_temperature()), dyn.electrode_charge
            if (not np.isfinite(forces).all() or temperature > MAX_TEMPERATURE_K or abs(charge) > MAX_CHARGE_E
                    or np.linalg.norm(forces, axis=1).max() > MAX_FORCE_EV_A):
                raise RuntimeError(f'Stopped at step {dyn.nsteps}: T = {temperature:.0f} K, Q = {charge:+.3f} e or '
                                   'a force is out of range; inspect the last records')
            _, umbrella, distance = bias.evaluate(atoms)
            row = dict(step=dyn.nsteps, time_fs=dyn.nsteps*TIMESTEP_FS, stage=stage['stage'], window=stage['window'],
                       center_A=bias.center, distance_A=distance, bias_eV=umbrella, energy_model_eV=energy,
                       temperature_K=temperature, charge_e=charge, cell_voltage_V=float(r['cell_voltage']),
                       target_cell_voltage_V=dyn.target_cell_voltage, wf_eV=float(r['work_function']),
                       U_SHE_V=float(r['electrode_potential_she']), delta_V=float(r['she_offset']),
                       mean_delta_V=dyn.mean_offset, aligned=int(dyn.aligned),
                       scf_residual=float(np.max(r['scf_residual'])))
            for j, oxygen in enumerate(pair, 1):
                row.update({f'O{j}_{key}': value for key, value in environment(atoms, oxygen).items()})
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if stream.tell() == 0:
                    writer.writeheader()
            writer.writerow(row)
            frame = atoms.copy()
            frame.set_constraint([])
            frame.info['restart'] = dict(step=dyn.nsteps, stage=stage_id, done=dyn.nsteps-start,
                                         rng=rng.bit_generator.state, potentiostat=dyn.potentiostat_state())
            trajectory.write(frame, energy=energy, forces=forces)

        if state is None:
            record()
        dyn.attach(record, interval=LOGINTERVAL)
        while stage_id < len(stages):
            stage = stages[stage_id]
            bias.center, dyn.frozen = stage['center_A'], stage['stage'] == 'relax'
            start = dyn.nsteps-done
            print(f'{index}/{replica} {stage["stage"]} w{stage["window"]}: center {bias.center:.2f} A, '
                  f'{stage["steps"]-done} steps', flush=True)
            dyn.run(stage['steps']-done)
            stage_id, done = stage_id+1, 0
    print(f'Complete at step {dyn.nsteps}', flush=True)


def run_one(source, index, replica, model_file, model_sha, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'running.lock'
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))  # One process per output folder.
    except FileExistsError:
        raise RuntimeError(f'{lock} exists: another run is active there, or one was killed; delete it to resume') from None
    try:
        with open(directory/'run.log', 'a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                simulate(source, index, replica, model_file, model_sha, directory)
            except Exception:
                traceback.print_exc()
                raise
    finally:
        lock.unlink(missing_ok=True)
    return index, replica


def main():
    frames = read(ATOMS_FILE, index=':')
    indexes = list(RUN_INDEXES)
    if not indexes or len(set(indexes)) != len(indexes) or not set(indexes) <= set(range(len(frames))):
        raise ValueError('RUN_INDEXES must list distinct frames of ATOMS_FILE')
    for index in indexes:
        atoms, _, _, wall, pair, kind = prepare(frames[index], index)
        stages = make_stages(atoms.get_distance(*pair, mic=True))
        print(f'{index}: {kind} pair {pair} at {atoms.get_distance(*pair, mic=True):.2f} A, {stages[-1]["stage"]} '
              f'over {stages[-1]["window"]+1} windows, {sum(s["steps"] for s in stages)*TIMESTEP_FS/1000:.1f} ps; '
              f'sheet z/c {atoms.info["counter_charge_zfrac"]:.3f}, wall {wall:.1f} A', flush=True)
    model = Path(MODEL_FILE).resolve(strict=True)
    model_sha, failures = sha256(model), []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(indexes)*REPLICAS),
                             mp_context=mp.get_context('spawn')) as pool:
        jobs = {pool.submit(run_one, frames[i], i, rep, str(model), model_sha,
                            (Path(OUTPUT_ROOT)/f'umbrella_{i}'/f'replica_{rep}').resolve()): (i, rep)
                for i in indexes for rep in range(REPLICAS)}
        for job in as_completed(jobs):
            try:
                job.result()
                print(f'{jobs[job]}: complete', flush=True)
            except Exception as error:
                failures.append(jobs[job])
                print(f'{jobs[job]}: failed: {error}', flush=True)
    if failures:
        raise SystemExit(f'Failed runs: {failures}; see their run.log')


if __name__ == '__main__':
    main()
