import weakref
from typing import IO, Optional, Union

import mace.data
import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.langevin import Langevin
from ase.parallel import DummyMPI, world
from ase.utils import IOContext
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools

import mace_scf.data

SHE_ABSOLUTE_V = 4.44  # Trasatti's absolute SHE potential at 298.15 K, IUPAC
D_BLOCK = frozenset([*range(21, 31), *range(39, 49), *range(57, 81), *range(89, 113)])


def slab_controls(atoms, sheet_height=6., sheet_width=1., cut=None):
    """The dipole correction plane and a counter charge sheet of a slab open along z, from its geometry.

    The plane lies in the middle of the largest empty gap along z unless cut is given. The electrode surface is
    the outermost d-block atom, or O bonded to one, on the side that holds the solvent; the sheet lies
    sheet_height beyond it.
    """
    z = atoms.get_scaled_positions(wrap=True)[:, 2]
    if cut is None:
        order = np.sort(z)
        gaps = np.diff(np.append(order, order[0]+1.))
        i = int(np.argmax(gaps))
        cut = (order[i]+.5*gaps[i]) % 1.
    length = atoms.cell[2, 2]
    height = ((z-cut) % 1.)*length
    numbers = atoms.get_atomic_numbers()
    electrode = np.isin(numbers, list(D_BLOCK))
    if not electrode.any():
        raise ValueError('No d-block electrode atoms to place the counter charge sheet')
    surface = list(np.flatnonzero(electrode))
    for o in np.flatnonzero(numbers == 8):
        if (atoms.get_distances(int(o), np.flatnonzero(electrode), mic=True) < 2.3).any():
            surface.append(int(o))
    top, bottom = height[surface].max(), height[surface].min()
    sheet = top+sheet_height if (height > top).sum() >= (height < bottom).sum() else bottom-sheet_height
    return {'dipole_correction_zfrac': float(cut), 'counter_charge_zfrac': float((cut+sheet/length) % 1.),
            'counter_charge_width': float(sheet_width)}


class MACEPotentialResponse(Calculator):
    """ASE calculator of a PotentialResponse model at the total charge, counter charge and field in atoms.info.

    With constant_potential the counter charge stands for solvated ions and acts through electrostatics only,
    and a slab without one gets a sheet counter_charge_height above its electrode surface on the first call,
    kept for the trajectory (slab_controls).
    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'dipole', 'charges', 'fermi_level',
                              'workfunction', 'cell_voltage', 'she_offset', 'electrode_potential_she']

    def __init__(self, model_path, device='cuda', default_dtype='float64', constant_potential=False,
                 counter_charge_height=6., counter_charge_width=1., enable_cueq=False, **kwargs):
        Calculator.__init__(self, **kwargs)
        torch_tools.set_default_dtype(default_dtype)
        self.device = torch_tools.init_device(device)
        self.model = torch.load(f=model_path, map_location=self.device, weights_only=False).to(self.device)
        if enable_cueq:
            from mace_scf.utils.foundation import accelerate_backbone
            accelerate_backbone(self.model, self.device)
        self.model.counter_charge_overlap = not constant_potential
        self.model.eval().requires_grad_(False)
        self.constant_potential = constant_potential
        self.counter_charge_height = float(counter_charge_height)
        self.counter_charge_width = float(counter_charge_width)
        self.z_table = AtomicNumberTable([int(z) for z in self.model.atomic_numbers])
        self.keyspec = mace.data.KeySpecification(info_keys={
            'total_charge': 'total_charge', 'counter_charge': 'counter_charge',
            'counter_charge_width': 'counter_charge_width', 'counter_charge_fraction': 'counter_charge_zfrac',
            'dipole_correction_fraction': 'dipole_correction_zfrac', 'external_field': 'external_field'},
            arrays_keys={})

    def check_state(self, atoms, tol=1e-15):
        changes = Calculator.check_state(self, atoms, tol)
        if self.atoms is not None and any(np.any(np.asarray(atoms.info.get(key)) != np.asarray(self.atoms.info.get(key)))
                                          for key in self.keyspec.info_keys.values()):
            changes.append('info')
        return changes

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        if self.constant_potential and atoms is not None and 'counter_charge_zfrac' not in atoms.info:
            atoms.info.update(slab_controls(atoms, self.counter_charge_height, self.counter_charge_width))
        Calculator.calculate(self, atoms, properties, system_changes)
        config = mace.data.config_from_atoms(self.atoms, key_specification=self.keyspec)
        data = mace_scf.data.ExtAtomicData.from_config(config, z_table=self.z_table, cutoff=float(self.model.r_max))
        batch = torch_geometric.batch.Batch.from_data_list([data]).to(self.device)
        output = self.model(batch.to_dict(), compute_force=True)
        output = {key: value.detach() if isinstance(value, torch.Tensor) else value for key, value in output.items()}
        workfunction, cell_voltage = float(output['workfunction'][0]), float(output['cell_voltage'][0])
        self.results = {
            'energy': float(output['energy'][0]),
            'free_energy': float(output['energy'][0]),
            'forces': output['forces'].cpu().numpy(),
            'dipole': output['dipole'][0].cpu().numpy(),
            'charges': output['density_coefficients'][:, 0].cpu().numpy(),
            'fermi_level': float(output['fermi_level'][0]),
            'workfunction': workfunction,
            'cell_voltage': cell_voltage,
            'she_offset': workfunction-cell_voltage,
            'electrode_potential_she': workfunction-SHE_ABSOLUTE_V,
        }
        for key in ('fourier_potential', 'k_vectors_grid_shape', 'k_vectors_mask'):
            self.results[key] = output[key].cpu().numpy()


class NVTPhiLangevin(Langevin):
    """Langevin dynamics of the nuclei with the thermopotentiostat on the electrode charge Q.

    Q and the counter charge -Q are extended coordinates of Omega(R, Q) = E(R, Q, -Q) - e V* Q (Deissenbeck et al.,
    PRL 126, 136803), V = dE/dQ the cell voltage. Before every step Q makes the exact Ornstein-Uhlenbeck move

        Q' = Q - C0 (V - V*) (1 - g) + sqrt(kT C0 (1 - g^2)) xi,    g = exp(-dt/tauphi),

    whose stationary weight is exp(-Omega/kT) for any C0, so each step needs one evaluation. The electrode
    potential is U_SHE = WF/e - 4.44 V, and WF - eV = Delta is the step from the counter charge to the vacuum.
    For a target U* the dynamics runs at V* = 4.44 V + U* - <Delta>/e: she_offset fixes <Delta>, otherwise it
    is averaged online for alignment_time and then frozen. With frozen = True the charge stays fixed while
    the average runs.
    """

    def __init__(self, atoms: Atoms, timestep: float, temperature_K: float, friction: float,
                 target_potential: Optional[float] = None, C0: float = .05, tauphi: float = 100.*units.fs,
                 cell_voltage: Optional[float] = None, she_offset: Optional[float] = None,
                 alignment_time: float = 1000.*units.fs, alignment_constant: float = 250.*units.fs,
                 trajectory: Optional[str] = None, logfile: Optional[Union[IO, str]] = None, loginterval: int = 1,
                 communicator=world, rng=None, append_trajectory: bool = False, fixcm: bool = True):
        if (cell_voltage is None) == (target_potential is None):
            raise ValueError('Give exactly one of target_potential (V vs SHE) or cell_voltage (V*)')
        if C0 <= 0. or tauphi <= 0.:
            raise ValueError('C0 and tauphi must be positive')
        self.kT = units.kB*float(temperature_K)
        self.target_potential = None if target_potential is None else float(target_potential)
        self.cell_voltage = None if cell_voltage is None else float(cell_voltage)
        self.capacitance, self.time_constant = float(C0), float(tauphi)
        self.alignment_time, self.alignment_constant = float(alignment_time), float(alignment_constant)
        self.aligned = cell_voltage is not None or she_offset is not None or alignment_time == 0.
        self.mean_offset = None if she_offset is None else float(she_offset)
        self.alignment_elapsed, self.frozen = 0., False
        Langevin.__init__(self, atoms, timestep, temperature_K=temperature_K, friction=friction, fixcm=fixcm,
                          trajectory=trajectory, logfile=None, loginterval=loginterval, rng=rng,
                          append_trajectory=append_trajectory)
        self.phi_comm = DummyMPI() if communicator is None else communicator
        self.phi_rng = np.random if rng is None else rng
        atoms.info.setdefault('counter_charge', -float(atoms.info.get('total_charge', 0.)))
        self.set_electrode_charge(self.electrode_charge)
        atoms.get_forces(md=True)
        if self.mean_offset is None and self.cell_voltage is None:
            self.mean_offset = self.observation('she_offset')
        if logfile:
            self.attach(self.closelater(NVTPhiMDLogger(self, atoms, logfile)), loginterval)

    @property
    def electrode_charge(self):
        return -float(np.asarray(self.atoms.info['counter_charge']).reshape(-1)[0])

    def set_electrode_charge(self, charge):
        charge = float(charge)
        info = self.atoms.info
        if (float(np.asarray(info.get('total_charge', np.nan)).reshape(-1)[0]) != charge
                or float(np.asarray(info.get('counter_charge', np.nan)).reshape(-1)[0]) != -charge):
            info.update(total_charge=charge, counter_charge=-charge)
            self.atoms.calc.reset()

    def observation(self, key):
        value = float(np.asarray(self.atoms.calc.results[key]).reshape(-1)[0])
        if not np.isfinite(value):
            raise FloatingPointError(f'Nonfinite {key}')
        return value

    @property
    def target_cell_voltage(self):
        if self.cell_voltage is not None:
            return self.cell_voltage
        return SHE_ABSOLUTE_V+self.target_potential-self.mean_offset

    def grand_potential(self):
        """Omega = E - e V* Q in eV, without restraints."""
        return float(self.atoms.get_potential_energy(apply_constraint=False))-self.target_cell_voltage*self.electrode_charge

    def update_charge(self):
        """One Ornstein-Uhlenbeck move of Q at fixed R, with the evaluation at (R, Q)."""
        voltage = self.observation('cell_voltage')
        if not self.aligned:
            weight = 1.-np.exp(-self.dt/self.alignment_constant)
            self.mean_offset += weight*(self.observation('she_offset')-self.mean_offset)
            self.alignment_elapsed += self.dt
            self.aligned = self.alignment_elapsed >= self.alignment_time
        if self.frozen:
            return
        decay = np.exp(-self.dt/self.time_constant)
        noise = np.asarray([self.phi_rng.standard_normal()], dtype=float)
        self.phi_comm.broadcast(noise, 0)
        charge = self.electrode_charge-self.capacitance*(voltage-self.target_cell_voltage)*(1.-decay)
        self.set_electrode_charge(charge+np.sqrt(self.kT*self.capacitance*(1.-decay*decay))*float(noise[0]))

    def step(self, forces=None):
        if forces is None:
            forces = self.atoms.get_forces(md=True)
        self.update_charge()
        return Langevin.step(self, forces)

    def todict(self):
        result = Langevin.todict(self)
        result.update(target_potential_V_SHE=self.target_potential, cell_voltage_V=self.cell_voltage,
                      she_offset_V=self.mean_offset, aligned=self.aligned, capacitance_e_per_V=self.capacitance,
                      time_constant_fs=self.time_constant/units.fs)
        return result

    def potentiostat_state(self):
        return dict(mean_offset=self.mean_offset, aligned=self.aligned, alignment_elapsed=self.alignment_elapsed)

    def restore_potentiostat_state(self, state):
        self.mean_offset = None if state['mean_offset'] is None else float(state['mean_offset'])
        self.aligned, self.alignment_elapsed = bool(state['aligned']), float(state['alignment_elapsed'])


class NVTPhiMDLogger(IOContext):
    """Energies, temperature, electrode charge, the cell voltage and its target, WF, U_SHE and Delta."""

    def __init__(self, dyn, atoms, logfile, header=True, mode='a', comm=world):
        self.dyn = weakref.proxy(dyn)
        self.atoms = atoms
        self.logfile = self.openfile(file=logfile, mode=mode, comm=comm)
        self.fmt = '%-10.4f %16.6f %16.6f %12.4f %8.1f %12.6f %10.5f %10.5f %10.5f %10.5f %10.5f\n'
        if header:
            columns = ('Time[ps]', 'Epot[eV]', 'Omega[eV]', 'Ekin[eV]', 'T[K]', 'Q[e]', 'V[V]', 'V*[V]', 'WF[eV]',
                       'U_SHE[V]', 'Delta[V]')
            self.logfile.write(' '.join(f'{c:>12s}' for c in columns)+'\n')

    def __del__(self):
        self.close()

    def __call__(self):
        results = self.atoms.calc.results
        self.logfile.write(self.fmt % (
            self.dyn.get_time()/(1000.*units.fs), self.atoms.get_potential_energy(apply_constraint=False),
            self.dyn.grand_potential(), self.atoms.get_kinetic_energy(), self.atoms.get_temperature(),
            self.dyn.electrode_charge, results['cell_voltage'], self.dyn.target_cell_voltage, results['workfunction'],
            results['electrode_potential_she'], results['she_offset']))
        self.logfile.flush()
