"""ASE voltage-controlled Langevin/Verlet dynamics and its trajectory logger."""
from __future__ import annotations

import weakref
from typing import IO, Any, Optional, Union

import numpy as np
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.md import MolecularDynamics
from ase.parallel import DummyMPI, world
from ase.utils import IOContext


class NVTPhiMixin:
    def _set_potentiostat_parameters(
        self,
        temperature_K: float,
        target_potential: float,
        capacitance: float,
        time_constant: float,
        counter_charge_key: str,
        total_charge_key: str,
        workfunction_key: str,
        use_relative_workfunction: bool,
        reference_potential: Optional[float],
    ) -> None:
        """Store metadata before ASE's constructor can call ``todict``."""
        if capacitance <= 0.0 or time_constant <= 0.0:
            raise ValueError("capacitance and time_constant must be positive")
        self._phi_kbt = units.kB * float(temperature_K)
        self.target_potential = float(target_potential)
        self.capacitance = float(capacitance)
        self.time_constant = float(time_constant)
        self.counter_charge_key = counter_charge_key
        self.total_charge_key = total_charge_key
        self.workfunction_key = workfunction_key
        self.use_relative_workfunction = bool(use_relative_workfunction)
        self.reference_potential = (
            None if reference_potential is None else float(reference_potential)
        )
        self._reference_model_workfunction = None
        self.current_potential = None

    def _init_potentiostat_state(self, communicator, rng) -> None:
        self._phi_comm = DummyMPI() if communicator is None else communicator
        self._phi_rng = np.random if rng is None else rng
        if self.counter_charge_key not in self.atoms.info:
            self.atoms.info[self.counter_charge_key] = 0.0
        self._sync_total_charge()
        if (
            self.atoms.calc is not None
            and self.workfunction_key in self.atoms.calc.results
        ):
            self._read_potential()

    def _counter_charge(self) -> float:
        value = self.atoms.info.get(self.counter_charge_key, 0.0)
        return float(np.asarray(value).reshape(-1)[0])

    def _set_counter_charge(self, value: float) -> None:
        value = float(np.asarray(value).reshape(-1)[0])
        old_counter = self.atoms.info.get(self.counter_charge_key)
        old_total = self.atoms.info.get(self.total_charge_key)
        changed = (
            old_counter is None
            or old_total is None
            or float(np.asarray(old_counter).reshape(-1)[0]) != value
            or float(np.asarray(old_total).reshape(-1)[0]) != -value
        )
        self.atoms.info[self.counter_charge_key] = value
        self.atoms.info[self.total_charge_key] = -value
        if changed and self.atoms.calc is not None:
            self.atoms.calc.reset()

    def _sync_total_charge(self) -> None:
        self._set_counter_charge(self._counter_charge())

    def _control_potential(self, workfunction: float) -> float:
        workfunction = float(np.asarray(workfunction).reshape(-1)[0])
        if not self.use_relative_workfunction:
            self.current_potential = workfunction
            return workfunction
        if self._reference_model_workfunction is None:
            self._reference_model_workfunction = workfunction
            if self.reference_potential is None:
                self.reference_potential = workfunction
        self.current_potential = self.reference_potential + (
            workfunction - self._reference_model_workfunction
        )
        return float(self.current_potential)

    def _read_potential(self) -> float:
        if self.atoms.calc is None:
            raise RuntimeError("NVTPhi dynamics requires an attached calculator")
        if self.workfunction_key not in self.atoms.calc.results:
            raise RuntimeError(
                f"Calculator did not return {self.workfunction_key!r}"
            )
        return self._control_potential(
            self.atoms.calc.results[self.workfunction_key]
        )

    def _normal(self) -> float:
        value = np.asarray([self._phi_rng.standard_normal()], dtype=float)
        self._phi_comm.broadcast(value, 0)
        return float(value[0])

    def _update_counter_charge(self) -> float:
        potential = self._read_potential()
        decay = np.exp(-self.dt / self.time_constant)
        noise = np.sqrt(
            self._phi_kbt * self.capacitance * (1.0 - decay * decay)
        )
        charge = self._counter_charge()
        charge += self.capacitance * (
            potential - self.target_potential
        ) * (1.0 - decay)
        charge -= noise * self._normal()
        self._set_counter_charge(charge)
        return charge

    def _potentiostat_dict(self) -> dict:
        return {
            "target_potential": self.target_potential,
            "capacitance": self.capacitance,
            "time_constant": self.time_constant,
            "counter_charge_key": self.counter_charge_key,
            "total_charge_key": self.total_charge_key,
            "workfunction_key": self.workfunction_key,
            "use_relative_workfunction": self.use_relative_workfunction,
            "reference_potential": self.reference_potential,
        }


class NVTPhiMDLogger(IOContext):
    def __init__(
        self,
        dyn: Any,
        atoms: Atoms,
        logfile: Union[IO, str],
        counter_charge_key: str = "counter_charge",
        workfunction_key: str = "workfunction",
        header: bool = True,
        peratom: bool = False,
        mode: str = "a",
        comm=world,
    ):
        self.dyn = weakref.proxy(dyn) if hasattr(dyn, "get_time") else None
        self.atoms = atoms
        self.counter_charge_key = counter_charge_key
        self.workfunction_key = workfunction_key
        self.peratom = peratom
        self.logfile = self.openfile(file=logfile, mode=mode, comm=comm)

        self.header = "%-9s " % "Time[ps]" if self.dyn is not None else ""
        self.fmt = "%-10.4f " if self.dyn is not None else ""
        scale = "/N" if peratom else ""
        self.header += "%12s %12s %12s %8s %16s %16s" % (
            f"Etot{scale}[eV]",
            f"Epot{scale}[eV]",
            f"Ekin{scale}[eV]",
            "T[K]",
            "q_electrode[e]",
            "Phi[V]",
        )
        natoms = atoms.get_global_number_of_atoms()
        if peratom or natoms <= 100:
            digits = 4
        elif natoms <= 1000:
            digits = 3
        elif natoms <= 10000:
            digits = 2
        else:
            digits = 1
        self.fmt += 3 * (f"%12.{digits}f ") + "%8.1f %16.8f %16.8f\n"
        if header:
            self.logfile.write(self.header + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        if self.peratom:
            natoms = self.atoms.get_global_number_of_atoms()
            epot /= natoms
            ekin /= natoms
        counter = self.atoms.info.get(self.counter_charge_key, np.nan)
        electrode_charge = -float(np.asarray(counter).reshape(-1)[0])
        potential = getattr(self.dyn, "current_potential", None)
        if potential is None and self.atoms.calc is not None:
            potential = self.atoms.calc.results.get(self.workfunction_key, np.nan)

        values = ()
        if self.dyn is not None:
            values += (self.dyn.get_time() / (1000.0 * units.fs),)
        values += (
            epot + ekin,
            epot,
            ekin,
            self.atoms.get_temperature(),
            electrode_charge,
            float(np.asarray(potential).reshape(-1)[0]),
        )
        self.logfile.write(self.fmt % values)
        self.logfile.flush()


class NVTPhiLangevin(NVTPhiMixin, Langevin):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        friction: float,
        target_potential: float,
        C0: float = 0.01,
        tauphi: float = 100.0 * units.fs,
        counter_charge_key: str = "counter_charge",
        total_charge_key: str = "total_charge",
        workfunction_key: str = "workfunction",
        use_relative_workfunction: bool = True,
        reference_potential: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        communicator=world,
        rng=None,
        append_trajectory: bool = False,
        fixcm: bool = True,
    ):
        self._set_potentiostat_parameters(
            temperature_K,
            target_potential,
            C0,
            tauphi,
            counter_charge_key,
            total_charge_key,
            workfunction_key,
            use_relative_workfunction,
            reference_potential,
        )
        Langevin.__init__(
            self,
            atoms,
            timestep,
            temperature_K=temperature_K,
            friction=friction,
            fixcm=fixcm,
            trajectory=trajectory,
            logfile=None,
            loginterval=loginterval,
            rng=rng,
            append_trajectory=append_trajectory,
        )
        self._init_potentiostat_state(communicator, rng)
        if logfile:
            logger = self.closelater(
                NVTPhiMDLogger(
                    self,
                    atoms,
                    logfile,
                    counter_charge_key,
                    workfunction_key,
                )
            )
            self.attach(logger, loginterval)

    def todict(self):
        result = Langevin.todict(self)
        result.update(self._potentiostat_dict())
        return result

    def step(self, forces=None):
        self._sync_total_charge()
        Langevin.step(self, forces)
        self.atoms.get_potential_energy()
        self._update_counter_charge()
        forces = self.atoms.get_forces(md=True)
        self._read_potential()
        return forces


class NVTPhiVelocityVerlet(NVTPhiMixin, MolecularDynamics):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        target_potential: float,
        C0: float = 0.01,
        tauphi: float = 100.0 * units.fs,
        counter_charge_key: str = "counter_charge",
        total_charge_key: str = "total_charge",
        workfunction_key: str = "workfunction",
        use_relative_workfunction: bool = True,
        reference_potential: Optional[float] = None,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        communicator=world,
        rng=None,
        append_trajectory: bool = False,
    ):
        self._set_potentiostat_parameters(
            temperature_K,
            target_potential,
            C0,
            tauphi,
            counter_charge_key,
            total_charge_key,
            workfunction_key,
            use_relative_workfunction,
            reference_potential,
        )
        MolecularDynamics.__init__(
            self,
            atoms,
            timestep,
            trajectory=trajectory,
            logfile=None,
            loginterval=loginterval,
            append_trajectory=append_trajectory,
        )
        self._init_potentiostat_state(communicator, rng)
        if logfile:
            logger = self.closelater(
                NVTPhiMDLogger(
                    self,
                    atoms,
                    logfile,
                    counter_charge_key,
                    workfunction_key,
                )
            )
            self.attach(logger, loginterval)

    def todict(self):
        result = MolecularDynamics.todict(self)
        result.update(self._potentiostat_dict())
        return result

    def step(self, forces=None):
        atoms = self.atoms
        self._sync_total_charge()
        if forces is None:
            forces = atoms.get_forces(md=True)
        momenta = atoms.get_momenta() + 0.5 * self.dt * forces
        masses = atoms.get_masses()[:, None]
        old_positions = atoms.get_positions()
        atoms.set_positions(old_positions + self.dt * momenta / masses)
        if atoms.constraints:
            momenta = (atoms.get_positions() - old_positions) * masses / self.dt
        atoms.set_momenta(momenta, apply_constraint=False)
        atoms.get_potential_energy()
        self._update_counter_charge()
        forces = atoms.get_forces(md=True)
        self._read_potential()
        atoms.set_momenta(
            atoms.get_momenta() + 0.5 * self.dt * forces,
            apply_constraint=True,
        )
        return forces
