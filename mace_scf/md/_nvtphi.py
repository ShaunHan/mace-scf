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
        vacuum_potential_key: str,
        fermi_level_key: str,
        reference_potential: Optional[float],
    ) -> None:
        """Store metadata before ASE's constructor can call ``todict``."""
        if not np.isfinite([capacitance, time_constant]).all() or capacitance <= 0.0 or time_constant <= 0.0:
            raise ValueError("capacitance and time_constant must be finite and positive")
        if not np.isfinite(temperature_K) or temperature_K < 0.0:
            raise ValueError("temperature_K must be finite and nonnegative")
        if not np.isfinite(target_potential):
            raise ValueError("target_potential must be finite")
        if reference_potential is not None and not np.isfinite(reference_potential):
            raise ValueError("reference_potential must be finite when supplied")
        self._phi_kbt = units.kB * float(temperature_K)
        self.target_potential = float(target_potential)
        self.capacitance = float(capacitance)
        self.time_constant = float(time_constant)
        self.counter_charge_key = counter_charge_key
        self.total_charge_key = total_charge_key
        self.vacuum_potential_key = vacuum_potential_key
        self.fermi_level_key = fermi_level_key
        self.reference_potential = (
            None if reference_potential is None else float(reference_potential)
        )
        self._reference_model_potential = None
        self.current_potential = None

    def _init_potentiostat_state(self, communicator, rng) -> None:
        self._phi_comm = DummyMPI() if communicator is None else communicator
        self._phi_rng = np.random if rng is None else rng
        if self.counter_charge_key not in self.atoms.info:
            self.atoms.info[self.counter_charge_key] = 0.0
        self._sync_total_charge()
        if self.atoms.calc is None:
            raise RuntimeError("NVTPhi dynamics requires an attached calculator")
        # Capture the reference on the initial geometry, before the first MD
        # displacement, regardless of whether the calculator was evaluated.
        self.atoms.get_potential_energy()
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

    def _control_potential(self, model_potential: float) -> float:
        """Apply an optional, fixed initial-frame calibration.

        With no reference, the target uses the model's own scalar origin.
        Otherwise U(t) = reference_potential + U_E(t) - U_E(0),
        where U_E=dE(R,Q,-Q)/dQ. This adds a constant charge-linear
        term to the grand potential and leaves fixed-Q forces unchanged.
        A vacuum/SHE WF anchor alone does not establish the counter-electrode
        reference: this controls pair voltage, not vacuum WF at every frame.
        """
        if self._reference_model_potential is None:
            self._reference_model_potential = model_potential
        self.current_potential = model_potential
        if self.reference_potential is not None:
            self.current_potential = self.reference_potential + (
                model_potential - self._reference_model_potential
            )
        return float(self.current_potential)

    def _read_potential(self) -> float:
        if self.atoms.calc is None:
            raise RuntimeError("NVTPhi dynamics requires an attached calculator")
        value = self.atoms.calc.results.get('charging_potential')
        if value is None:
            raise RuntimeError('NVTPhi requires charging_potential=dE(R,Q,-Q)/dQ. '
                               'This model has an independent WF observer; retrain with the '
                               'energy-derived model before using voltage dynamics.')
        value = float(value)
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite energy-derived charging potential')
        return self._control_potential(value)

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
            "vacuum_potential_key": self.vacuum_potential_key,
            "fermi_level_key": self.fermi_level_key,
            "reference_potential": self.reference_potential,
        }
