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
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
from ase import units
from ase.constraints import FixAtoms, FixConstraint, Hookean
from ase.io import iread
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# ------------------------------ user settings ------------------------------ #
ATOMS_FILE = 'IrO2OH_110_mdinit_0_51_vasp_sp.xyz'
MODEL_FILE = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/ml_training/wf_training_203/models/fit_0_best_wf.model'
RUN_INDEXES = list(range(0, 10))
OUTPUT_ROOT = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/nvtphi_umbrella'
MAX_WORKERS, REPLICAS = 10, 1                 # Two processes share one visible GPU.
DEVICE, HEAD, ENABLE_CUEQ = 'cuda:0', None, True
SCF_STEPS = 50                              # None uses the exported model's deployment budget.
OXYGEN_PAIRS = {}                           # Optional {index: (O_id, O_id)}, zero-based ASE IDs.
INITIAL_U_SHE = {}                          # Optional independently known initial potentials.
TARGET_U_SHE, SHE_ABSOLUTE_V = 1.60, 4.44    # Same explicit reference convention as the old run.
COUNTER_SHEET_POINT, GAUSSIAN_WIDTH = [0., 0., 10.], 1.0
CAPACITANCE, TAUPHI_FS = 0.01, 100.0         # e/V for the whole cell; fs. Not an areal capacitance.
Z_FIXED, Z_REPULSIVE, WALL_K = -11.0, 15.0, 10.0
PAIR_CUTOFF, OO_MIN = 3.0, 1.20
WINDOW_SPACING = 0.15        # Å
INITIAL_EQUIL_STEPS = 2000   # 1 ps
EQUIL_STEPS = 500           # 0.25 ps per window
SAMPLE_STEPS = 1000         # 0.50 ps per window
UMBRELLA_K, REACTANT_MARGIN = 20.0, 0.10     # eV/A^2; A. Center is not an enforced bond length.
REQUIRE_ADSORBATE = True                    # Require an exposed, singly Ir-bound O with <=1 H.
REVERSE_SWEEP = False                       # Enable a separate return sweep after inspecting the forward pilot.
TEMPERATURE_K, TIMESTEP_FS, FRICTION_PER_FS = 300., 0.5, 0.01
LOGINTERVAL, SEED = 20, 7777                # 10 fs output; saved states allow continuation.
MAX_TEMPERATURE_K, MAX_FORCE_EV_A = 1000., 100.
CHECK_CHARGE_RESPONSE, CHARGE_PROBE_E = True, 0.02  # Three inexpensive frozen-geometry calls.
# -------------------------------------------------------------------------- #


def scalar(value):
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f'Expected one finite scalar, got {value}')
    return float(a.item())


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    return np.asarray(value).tolist()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=jsonable).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as f:
        json.dump(value, f, indent=2, default=jsonable, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def settings():
    names = ('DEVICE HEAD ENABLE_CUEQ SCF_STEPS TARGET_U_SHE SHE_ABSOLUTE_V '
             'COUNTER_SHEET_POINT GAUSSIAN_WIDTH CAPACITANCE TAUPHI_FS Z_FIXED '
             'Z_REPULSIVE WALL_K PAIR_CUTOFF OO_MIN WINDOW_SPACING UMBRELLA_K '
             'REACTANT_MARGIN REQUIRE_ADSORBATE REVERSE_SWEEP TEMPERATURE_K '
             'TIMESTEP_FS FRICTION_PER_FS INITIAL_EQUIL_STEPS EQUIL_STEPS SAMPLE_STEPS '
             'LOGINTERVAL SEED MAX_TEMPERATURE_K MAX_FORCE_EV_A '
             'CHECK_CHARGE_RESPONSE CHARGE_PROBE_E').split()
    return {name: globals()[name] for name in names}


def physical_constraints(atoms, fixed):
    frozen = set(fixed)
    return [FixAtoms(indices=fixed)] + [
        Hookean(a1=i, a2=(0, 0, 1, -Z_REPULSIVE), k=WALL_K)
        for i, atom in enumerate(atoms) if atom.symbol == 'O' and i not in frozen]


def oxygen_environment(atoms, oxygen):
    symbols = np.asarray(atoms.get_chemical_symbols())
    ir, h = np.flatnonzero(symbols == 'Ir'), np.flatnonzero(symbols == 'H')
    di, dh = atoms.get_distances(oxygen, ir, mic=True), atoms.get_distances(oxygen, h, mic=True)
    n_ir, n_h = int(np.sum(di < 2.60)), int(np.sum(dh < 1.25))
    above = float(atoms.positions[oxygen, 2] - atoms.positions[ir, 2].max())
    # These are geometry-based selection labels, not a determination of chemical identity.
    kind = ('water' if n_h >= 2 else 'lattice' if n_ir >= 3 or above <= .3
            else 'adsorbate' if n_ir == 1 else 'surface_bridge' if n_ir == 2 else 'solution')
    return dict(kind=kind, H_neighbors=n_h, Ir_neighbors=n_ir,
                nearest_Ir_A=float(di.min()), height_above_Ir_A=above)


def select_pair(atoms, index, fixed):
    symbols = np.asarray(atoms.get_chemical_symbols())
    ir = np.flatnonzero(symbols == 'Ir')
    if not len(ir):
        raise ValueError('An Ir slab is required to locate this interface')
    top = atoms.positions[ir, 2].max()
    ids = [int(i) for i in np.flatnonzero((symbols == 'O') &
           (atoms.positions[:, 2] > top-2.) & (atoms.positions[:, 2] < top+5.)) if i not in fixed]
    environments = {i: oxygen_environment(atoms, i) for i in ids}
    eligible = [p for p in combinations(ids, 2) if OO_MIN < atoms.get_distance(*p, mic=True) <= PAIR_CUTOFF]
    if index in OXYGEN_PAIRS:
        pair = tuple(sorted(OXYGEN_PAIRS[index]))
        if pair not in eligible:
            raise ValueError(f'{index}: override must be two mobile interfacial oxygens within PAIR_CUTOFF')
    else:
        eligible = [p for p in eligible if not REQUIRE_ADSORBATE or
                    any(environments[o]['kind'] == 'adsorbate' for o in p)]
        if not eligible:
            raise ValueError(f'{index}: no eligible pair; inspect the interface or supply OXYGEN_PAIRS')
        rng = np.random.default_rng(np.random.SeedSequence([SEED, index, 1]))
        pair = eligible[int(rng.integers(len(eligible)))]
    return list(pair), [environments[o] for o in pair], len(eligible)


def prepare_frame(source, index):
    atoms, info = source.copy(), source.info
    if not np.array_equal(atoms.pbc, [True, True, False]) or not np.allclose(atoms.cell.array[:2, 2], 0.):
        raise ValueError('This driver expects an xy-periodic slab, open along Cartesian z')
    fixed = set(np.flatnonzero(atoms.positions[:, 2] < Z_FIXED).tolist())
    for constraint in atoms.constraints:
        if not isinstance(constraint, FixAtoms):
            raise ValueError('Input contains non-FixAtoms constraints; specify the physical restraints explicitly')
        fixed.update(map(int, constraint.get_indices()))
    pair, environments, candidates = select_pair(atoms, index, fixed)
    charge = scalar(info.get('total_charge', 0.))
    fraction = float((np.asarray(COUNTER_SHEET_POINT) @ np.linalg.inv(atoms.cell.array))[2] % 1.)
    if charge != 0.:
        if not np.isclose(scalar(info.get('counter_charge', np.nan)), -charge):
            raise ValueError('A charged starting reference must already have counter_charge = -total_charge')
        if ('counter_charge_center' in info or not np.isclose(scalar(info.get('counter_charge_zfrac', np.nan)), fraction)
                or not np.isclose(scalar(info.get('counter_charge_width', np.nan)), GAUSSIAN_WIDTH)):
            raise ValueError('The charged DFT reference and configured Gaussian sheet must have identical geometry')
    if index in INITIAL_U_SHE:
        reference = scalar(INITIAL_U_SHE[index])
    else:
        fermi = scalar(info['fermi_level'] if 'fermi_level' in info else info['vasp_efermi'])
        reference = scalar(info['vacuum_potential']) - fermi - SHE_ABSOLUTE_V
    # A fractional coordinate selects a sheet; counter_charge_center would select a 3D Gaussian.
    atoms.info = dict(source_index=index, oxygen_pair=pair, reference_potential=reference,
                      total_charge=charge, counter_charge=-charge,
                      external_field=np.asarray(info.get('external_field', np.zeros(3)), dtype=float),
                      counter_charge_zfrac=fraction, counter_charge_width=GAUSSIAN_WIDTH,
                      vacuum_zfrac=scalar(info['vacuum_zfrac']),
                      dipole_correction_zfrac=scalar(info['dipole_correction_zfrac']))
    atoms.calc = None
    fixed = sorted(fixed)
    atoms.set_constraint(physical_constraints(atoms, fixed))
    return atoms, fixed, environments, candidates


class DistanceUmbrella(FixConstraint):
    """Static V = k/2 (minimum-image distance - center)^2 during every MD call."""
    def __init__(self, pair, center, stiffness):
        self.pair, self.center, self.k = list(pair), float(center), float(stiffness)

    def evaluate(self, atoms):
        vector = atoms.get_distance(*self.pair, mic=True, vector=True)
        distance = float(np.linalg.norm(vector))
        if not np.isfinite(distance) or distance < 1e-8:
            raise ValueError('Invalid O-O distance')
        radial = -self.k * (distance-self.center)
        forces = np.zeros((len(atoms), 3))
        forces[self.pair[1]] = radial * vector / distance
        forces[self.pair[0]] = -forces[self.pair[1]]
        return forces, .5*self.k*(distance-self.center)**2, distance, radial

    def adjust_forces(self, atoms, forces):
        forces += self.evaluate(atoms)[0]

    def adjust_potential_energy(self, atoms):
        return self.evaluate(atoms)[1]

    def adjust_momenta(self, atoms, momenta):
        pass

    def get_removed_dof(self, atoms):
        return 0


def make_stages(distance):
    high = distance + REACTANT_MARGIN
    n = int(np.ceil((high-OO_MIN)/WINDOW_SPACING)) + 1
    centers = np.linspace(high, OO_MIN, n).tolist()
    stages = [dict(direction='initial', window=-1, phase='equil', center_A=distance,
                   steps=INITIAL_EQUIL_STEPS)]
    sweeps = [('forward', centers)] + ([('reverse', centers[::-1])] if REVERSE_SWEEP else [])
    for direction, sweep in sweeps:
        for window, center in enumerate(sweep):
            for phase, steps in [('equil', EQUIL_STEPS), ('sample', SAMPLE_STEPS)]:
                stages.append(dict(direction=direction, window=window, phase=phase,
                                   center_A=center, steps=steps))
    return stages


def make_calculator(model_file):
    import torch
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    torch.set_num_threads(1)
    options = dict(constant_charge=True, use_autograd_forces=True,
                   initial_density='local_guess', initial_fermi_level='zero')
    if SCF_STEPS is not None:
        options['num_scf_steps'] = int(SCF_STEPS)
    calc = MACEFixedPointSCF(model_path=model_file, device=DEVICE, default_dtype='float64',
                             pbc_handling='slab', scf_options=options, scf_restart=False,
                             ignore_nonconverged=True, use_compile=False, enable_cueq=ENABLE_CUEQ,
                             counter_charge_width=GAUSSIAN_WIDTH)
    heads = list(getattr(calc.model, 'heads', [calc.head]))
    head = HEAD or (heads[0] if len(heads) == 1 else next((h for h in heads if h.lower() == 'default'), None))
    if head not in heads:
        raise ValueError(f'Set HEAD explicitly to the electrochemistry head; model heads are {heads}')
    calc.head = head
    if ENABLE_CUEQ:
        from mace_scf.utils.foundation import accelerate_backbone
        accelerate_backbone(calc.model, DEVICE)
    calc.model.eval().requires_grad_(False)
    print(f'head={head}; SCF={calc.scf_options.num_scf_steps}; '
          f'deployment={getattr(calc.model.field_dependent_charges_map, "deployment_mode", "implicit")}', flush=True)
    return calc                         # Do not change the fitted decoder, mixing or solver mode.


def charge_probe(atoms):
    """A local diagnostic, not proof of energy/voltage conjugacy over a trajectory."""
    q = scalar(atoms.info['total_charge'])
    data = []
    try:
        for dq in (-CHARGE_PROBE_E, 0., CHARGE_PROBE_E):
            atoms.info.update(total_charge=q+dq, counter_charge=-q-dq)
            atoms.calc.reset()
            energy = atoms.get_potential_energy(apply_constraint=False)
            r = atoms.calc.results
            data.append([energy, scalar(r['vacuum_potential'])-scalar(r['fermi_level'])])
    finally:
        atoms.info.update(total_charge=q, counter_charge=-q)
        atoms.calc.reset()
    derivative, slope = (np.asarray(data[2])-np.asarray(data[0]))/(2*CHARGE_PROBE_E)
    return dict(charge_e=q, dE_dQ_eV_per_e=float(derivative), model_WF_eV=data[1][1],
                dWF_dQ_V_per_e=float(slope), E_curvature_eV_per_e2=float(
                    (data[2][0]+data[0][0]-2*data[1][0])/CHARGE_PROBE_E**2),
                note='Fixed geometry and counter_charge=-Q; reference/gauge and consistency across geometries still need validation.')


def last_csv(path):
    if not path.exists():
        return None, None
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        previous, row = -1, None
        for row in reader:
            if None in row or None in row.values() or int(row['step']) <= previous:
                raise ValueError('Incomplete/nonmonotonic CSV; preserve it and recover before resuming')
            previous = int(row['step'])
        return reader.fieldnames, row


def simulate(source, index, replica, model_file, model_sha, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'running.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f'{os.getpid()}\n'.encode()); os.close(fd)
    try:
        return _simulate(source, index, replica, model_file, model_sha, directory)
    finally:
        lock.unlink(missing_ok=True)


def _simulate(source, index, replica, model_file, model_sha, directory):
    from mace_scf.md import NVTPhiLangevin

    atoms, fixed, environments, candidates = prepare_frame(source, index)
    stages = make_stages(atoms.get_distance(*atoms.info['oxygen_pair'], mic=True))
    initial = dict(numbers=atoms.numbers, positions=atoms.positions, cell=atoms.cell.array,
                   pbc=atoms.pbc, info=atoms.info, fixed_ids=fixed)
    identity = dict(method='fixed_window_OO', settings=settings(), model_sha256=model_sha,
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    initial=initial, replica=replica, stages=stages)
    run_id = digest(identity)
    cfg_path, traj_path, csv_path = [directory/n for n in ('run.json', 'md.traj', 'md.csv')]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, index, replica, 2]))
    state = None
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get('run_id') != run_id:
            raise ValueError('Source/model/physical settings differ from this run; use a fresh output folder')
        if traj_path.exists():
            with Trajectory(traj_path) as previous:
                if len(previous):
                    atoms = previous[-1]
                    state = atoms.info.pop('restart_state')
                    if state['run_id'] != run_id:
                        raise ValueError('Trajectory belongs to a different run')
                    rng.bit_generator.state = state['rng_state']
    else:
        if any(p.exists() for p in (traj_path, csv_path)):
            raise ValueError('Output exists without matching run metadata; refusing to overwrite')
        cfg = dict(identity, run_id=run_id, model_file=str(model_file),
                   pair_environments=environments, eligible_pairs=candidates,
                   proton_ensemble='Fixed atom counts; pH is not imposed by this driver.')
        write_json(cfg_path, cfg)
    physical = physical_constraints(atoms, fixed)
    atoms.set_constraint(physical)
    atoms.calc = make_calculator(model_file)
    atoms.calc.reset()
    if state is None:
        MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, force_temp=True, rng=rng)
        if CHECK_CHARGE_RESPONSE:
            probe = charge_probe(atoms)
            write_json(directory/'charge_response.json', probe)
            if probe['dWF_dQ_V_per_e'] <= 0.:
                raise RuntimeError('Nonpositive local voltage response to electrode charge: inspect charge_response.json before MD')
    fields, saved_csv = last_csv(csv_path)
    last_step = -1 if saved_csv is None else int(saved_csv['step'])
    start = 0 if state is None else int(atoms.info['step'])
    if last_step > start or (state is None and last_step >= 0):
        raise ValueError('CSV is ahead of the restart trajectory; outputs have been preserved')
    stage_id = 0 if state is None else state['stage_id']
    done = 0 if state is None else state['stage_done']
    bias = DistanceUmbrella(atoms.info['oxygen_pair'], stages[stage_id]['center_A'], UMBRELLA_K)
    atoms.set_constraint([bias]+physical)
    with ExitStack() as stack:
        dyn = stack.enter_context(NVTPhiLangevin(
            atoms, timestep=TIMESTEP_FS*units.fs, temperature_K=TEMPERATURE_K,
            friction=FRICTION_PER_FS/units.fs, target_potential=TARGET_U_SHE,
            C0=CAPACITANCE, tauphi=TAUPHI_FS*units.fs, fixcm=False, rng=rng,
            logfile=str(directory / 'md.log'), loginterval=LOGINTERVAL,
            reference_potential=atoms.info['reference_potential']))
        dyn.nsteps = start
        if state is not None:
            dyn._reference_model_potential = state['model_anchor']
            dyn._read_potential()
        trajectory = stack.enter_context(Trajectory(traj_path, 'a'))
        stream = stack.enter_context(csv_path.open('a', newline='', buffering=1))

        def write_row(row):
            nonlocal fields, last_step
            if fields is not None and fields != list(row):
                raise ValueError('CSV columns differ; refusing to append')
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if fields is None:
                writer.writeheader(); fields = list(row)
            writer.writerow(row); stream.flush(); os.fsync(stream.fileno())
            last_step = int(row['step'])

        if state is not None and last_step < start:
            write_row(atoms.info['md_record'])  # Recover a completed trajectory write preceding a CSV write.
        segment_start = dyn.nsteps-done

        def record():
            if dyn.nsteps <= last_step:
                return
            energy = scalar(atoms.get_potential_energy(apply_constraint=False))
            r = atoms.calc.results
            _, umbrella, distance, radial = bias.evaluate(atoms)
            wall = sum(c.adjust_potential_energy(atoms) for c in physical if hasattr(c, 'adjust_potential_energy'))
            forces = atoms.get_forces(md=True)
            temperature = scalar(atoms.get_temperature())
            residual = scalar(np.max(np.asarray(r['scf_residual'])))
            if not np.isfinite(forces).all() or temperature > MAX_TEMPERATURE_K or np.max(np.linalg.norm(forces,axis=1)) > MAX_FORCE_EV_A:
                raise RuntimeError('Nonfinite/excessive force or temperature; inspect the last saved valid state')
            row = dict(step=dyn.nsteps, time_fs=dyn.get_time()/units.fs,
                       direction=stages[stage_id]['direction'], window=stages[stage_id]['window'],
                       phase=stages[stage_id]['phase'], phase_step=dyn.nsteps-segment_start,
                       distance_A=distance, center_A=bias.center, k_eV_A2=bias.k,
                       bias_eV=umbrella, radial_bias_eV_A=radial, energy_model_eV=energy,
                       energy_wall_eV=wall, energy_total_eV=energy+wall+umbrella,
                       temperature_K=temperature, voltage_V_SHE=dyn._read_potential(),
                       total_charge_e=scalar(atoms.info['total_charge']),
                       vacuum_potential_V=scalar(r['vacuum_potential']), fermi_level_eV=scalar(r['fermi_level']),
                       wf_eV=scalar(r['vacuum_potential'])-scalar(r['fermi_level']),
                       scf_steps=int(r['num_scf_steps']), scf_residual=residual)
            for j, o in enumerate(atoms.info['oxygen_pair'], 1):
                env = oxygen_environment(atoms, o)
                row.update({f'O{j}_{k}':env[k] for k in ('H_neighbors','Ir_neighbors','nearest_Ir_A')})
            frame = atoms.copy()
            frame.set_constraint([])  # Stored energy/forces include restraints exactly once.
            frame.info.update(step=dyn.nsteps, md_record=row, restart_state=dict(
                run_id=run_id, rng_state=rng.bit_generator.state, model_anchor=dyn._reference_model_potential,
                stage_id=stage_id, stage_done=dyn.nsteps-segment_start))
            trajectory.write(frame, energy=energy+wall+umbrella, forces=forces)
            write_row(row)

        record()
        dyn.attach(record, interval=LOGINTERVAL)
        while stage_id < len(stages):
            stage = stages[stage_id]
            bias.center = stage['center_A']       # Changed only BETWEEN complete integration steps.
            segment_start = dyn.nsteps-done
            remaining = stage['steps']-done
            if remaining > 0:
                print(f'{index}/{replica} {stage["direction"]} w{stage["window"]} '
                      f'{stage["phase"]}: center={bias.center:.3f} A, {remaining} steps', flush=True)
                dyn.run(remaining); record()
            stage_id += 1; done = 0
        print(f'Saved step {dyn.nsteps}; complete', flush=True)


def run_one(source, index, replica, model_file, model_sha, directory):
    Path(directory).mkdir(parents=True, exist_ok=True)
    with open(Path(directory)/'run.log', 'a', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            simulate(source, index, replica, model_file, model_sha, directory)
        except Exception:
            traceback.print_exc(); raise
    return index, replica


def main():
    if not RUN_INDEXES or any(not isinstance(i, int) or i < 0 for i in RUN_INDEXES):
        raise ValueError('RUN_INDEXES must contain nonnegative integers')
    if len(set(RUN_INDEXES)) != len(RUN_INDEXES):
        raise ValueError('RUN_INDEXES must not contain duplicates')
    if any(not isinstance(n, int) or n <= 0 for n in
           (MAX_WORKERS, REPLICAS, INITIAL_EQUIL_STEPS, EQUIL_STEPS, SAMPLE_STEPS, LOGINTERVAL)):
        raise ValueError('Worker, replica and step counts must be positive integers')
    if min(OO_MIN, WINDOW_SPACING, UMBRELLA_K, TEMPERATURE_K, TIMESTEP_FS,
           GAUSSIAN_WIDTH, CAPACITANCE, TAUPHI_FS, FRICTION_PER_FS, CHARGE_PROBE_E) <= 0:
        raise ValueError('Physical scales must be positive')
    wanted, frames = set(RUN_INDEXES), {}
    for index, atoms in enumerate(iread(ATOMS_FILE, index=':')):
        if index in wanted:
            frames[index] = atoms
        if index >= max(wanted):
            break
    if set(frames) != wanted:
        raise ValueError(f'Missing source indexes: {sorted(wanted-set(frames))}')
    for index in RUN_INDEXES:
        atoms, _, envs, n = prepare_frame(frames[index], index)
        stages = make_stages(atoms.get_distance(*atoms.info['oxygen_pair'], mic=True))
        print(f'{index}: pair {atoms.info["oxygen_pair"]}, {[e["kind"] for e in envs]}, '
              f'{n} eligible; {sum(s["steps"] for s in stages)*TIMESTEP_FS/1000:.1f} ps/replica', flush=True)
    model = Path(MODEL_FILE).resolve(strict=True)
    hasher = hashlib.sha256()
    with model.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''):
            hasher.update(chunk)
    model_sha = hasher.hexdigest()
    failures = []
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS,len(wanted)*REPLICAS),
                             mp_context=mp.get_context('spawn')) as pool:
        jobs = {pool.submit(run_one,frames[i],i,rep,str(model),model_sha,
                            str((Path(OUTPUT_ROOT)/f'umbrella_{i}'/f'replica_{rep}').resolve())):(i,rep)
                for rep in range(REPLICAS) for i in RUN_INDEXES}
        for job in as_completed(jobs):
            label = jobs[job]
            try:
                job.result(); print(f'{label}: complete', flush=True)
            except Exception as error:
                failures.append(label); print(f'{label}: failed: {error}; see run.log', flush=True)
    if failures:
        raise RuntimeError(f'Failed runs: {failures}')


if __name__ == '__main__':
    main()
