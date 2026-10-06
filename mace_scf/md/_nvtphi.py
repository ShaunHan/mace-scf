from __future__ import annotations
from typing import Optional
import numpy as np
from ase import units
from ase.parallel import DummyMPI


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
