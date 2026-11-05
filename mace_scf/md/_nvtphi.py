from __future__ import annotations

from typing import Optional

import numpy as np
from ase import units
from ase.parallel import DummyMPI

from mace_scf.calculators.potential_response import SHE_ABSOLUTE_V


class NVTPhiMixin:
    """Thermopotentiostat on the electrode charge Q, the energy-conjugate of the cell voltage.

    Q and the counter charge -Q are extended coordinates of Omega(R, Q) = E(R, Q, -Q) - e V* Q (Deissenbeck et
    al., PRL 126, 136803), with V = dE/dQ. At fixed R, before every step, Q makes the exact Ornstein-Uhlenbeck
    move Q' = Q - C0 (V - V*) (1 - g) + sqrt(kT C0 (1 - g^2)) xi, g = exp(-dt/tauphi), whose stationary weight is
    exp(-Omega/kT) for any C0. U_SHE = WF/e - 4.44 V and Delta = WF - eV, so a target U* runs at
    V* = 4.44 V + U* - <Delta>/e. Delta, the potential of the vacuum over the counter charge, is the drop across the
    water beyond the counter charge and its surface. It fluctuates with the water and only its mean enters V*,
    which stays constant while the run samples. <Delta> is given (she_offset), or it is the time average of Delta
    over alignment_time at constant potential, then frozen; alignment_time 0 takes Delta of the first frame. At
    fixed V* the average of Delta continues (sampled_offset): the run samples <U> = V* + <Delta>/e - 4.44 V
    (sampled_potential), and that <Delta> is the she_offset of the next run. With frozen = True the charge stays
    fixed and the averages wait.
    """

    def _set_potentiostat_parameters(
        self,
        temperature_K: float,
        target_potential: Optional[float],
        capacitance: float,
        time_constant: float,
        counter_charge_key: str = "counter_charge",
        total_charge_key: str = "total_charge",
        cell_voltage: Optional[float] = None,
        she_offset: Optional[float] = None,
        alignment_time: float = 1000.0 * units.fs,
    ) -> None:
        """Store controller settings before ASE's constructor can call todict.

        Either give cell_voltage (V*, absolute ensemble) or target_potential
        (U* vs SHE). With target_potential, she_offset fixes <Delta>; when it
        is None, <Delta> is the time average over alignment_time at constant
        potential, then frozen.
        """
        if not np.isfinite([capacitance, time_constant]).all() or capacitance <= 0.0 or time_constant <= 0.0:
            raise ValueError("capacitance and time_constant must be finite and positive")
        if not np.isfinite(temperature_K) or temperature_K < 0.0:
            raise ValueError("temperature_K must be finite and nonnegative")
        if (cell_voltage is None) == (target_potential is None):
            raise ValueError("Give exactly one of target_potential (V vs SHE) or cell_voltage (V*)")
        if cell_voltage is not None and she_offset is not None:
            raise ValueError("she_offset only converts a target_potential; cell_voltage is already absolute")
        for value in (target_potential, cell_voltage, she_offset):
            if value is not None and not np.isfinite(value):
                raise ValueError("Potential settings must be finite")
        if not alignment_time >= 0.0:
            raise ValueError("alignment_time must be nonnegative")
        self._phi_kbt = units.kB * float(temperature_K)
        self.target_potential = None if target_potential is None else float(target_potential)
        self.capacitance = float(capacitance)
        self.time_constant = float(time_constant)
        self.counter_charge_key = counter_charge_key
        self.total_charge_key = total_charge_key
        self.cell_voltage = None if cell_voltage is None else float(cell_voltage)
        self.alignment_time = float(alignment_time)
        # Delta is averaged until aligned; a given she_offset or cell_voltage is aligned from the start.
        self.aligned = cell_voltage is not None or she_offset is not None or alignment_time == 0.0
        self.mean_offset = None if she_offset is None else float(she_offset)
        self.averaged_time = 0.0
        self.sampled_offset, self.sampled_time = 0.0, 0.0
        self.frozen = False

    def _init_potentiostat_state(self, communicator, rng) -> None:
        self._phi_comm = DummyMPI() if communicator is None else communicator
        self._phi_rng = np.random if rng is None else rng
        if self.counter_charge_key not in self.atoms.info:
            self.atoms.info[self.counter_charge_key] = -float(self.atoms.info.get(self.total_charge_key, 0.0))
        self._set_counter_charge(self._counter_charge())
        if self.atoms.calc is None:
            raise RuntimeError("NVTPhi dynamics requires an attached calculator")
        self.atoms.get_forces(md=True)
        if self.mean_offset is None and self.cell_voltage is None:
            self.mean_offset = self._observation("she_offset")

    # ----------------------------------------------------------------- state
    def _counter_charge(self) -> float:
        value = self.atoms.info.get(self.counter_charge_key, 0.0)
        return float(np.asarray(value).reshape(-1)[0])

    def _set_counter_charge(self, value: float) -> None:
        value = float(np.asarray(value).reshape(-1)[0])
        old_counter = self.atoms.info.get(self.counter_charge_key)
        old_total = self.atoms.info.get(self.total_charge_key)
        changed = (old_counter is None or old_total is None
                   or float(np.asarray(old_counter).reshape(-1)[0]) != value
                   or float(np.asarray(old_total).reshape(-1)[0]) != -value)
        self.atoms.info[self.counter_charge_key] = value
        self.atoms.info[self.total_charge_key] = -value
        if changed and self.atoms.calc is not None:
            self.atoms.calc.reset()

    def _observation(self, key: str) -> float:
        results = self.atoms.calc.results
        if key not in results:
            raise RuntimeError(f"NVTPhi needs the calculator result {key!r}: use "
                               "MACEPotentialResponse(..., constant_potential=True)")
        value = float(np.asarray(results[key]).reshape(-1)[0])
        if not np.isfinite(value):
            raise FloatingPointError(f"Nonfinite {key}")
        return value

    @property
    def electrode_charge(self) -> float:
        return -self._counter_charge()

    @property
    def target_cell_voltage(self) -> float:
        """V* in volts: fixed, or 4.44 V + U* - <Delta> (frozen after alignment)."""
        if self.cell_voltage is not None:
            return self.cell_voltage
        return SHE_ABSOLUTE_V + self.target_potential - self.mean_offset

    @property
    def sampled_potential(self) -> Optional[float]:
        """<U> vs SHE in volts of the steps at fixed V*, V* + <Delta> - 4.44 V; None before the first."""
        if not self.sampled_time:
            return None
        return self.target_cell_voltage + self.sampled_offset - SHE_ABSOLUTE_V

    def grand_potential(self) -> float:
        """Omega = E - e V* Q in eV, for the current state (restraints excluded)."""
        energy = self.atoms.get_potential_energy(apply_constraint=False)
        return float(energy) - self.target_cell_voltage * self.electrode_charge

    # ------------------------------------------------------------- controller
    def _normal(self) -> float:
        value = np.asarray([self._phi_rng.standard_normal()], dtype=float)
        self._phi_comm.broadcast(value, 0)
        return float(value[0])

    def _update_charge(self) -> None:
        """One OU step of Q at fixed R, with the cached evaluation at (R, Q)."""
        voltage = self._observation("cell_voltage")
        if self.frozen:
            return
        offset = self._observation("she_offset")
        if not self.aligned:
            # The time average of Delta over alignment_time at constant potential, then frozen in V*.
            self.averaged_time += self.dt
            self.mean_offset += self.dt / self.averaged_time * (offset - self.mean_offset)
            self.aligned = self.averaged_time >= self.alignment_time
        else:
            # At fixed V*, the average of Delta that gives the potential the run samples.
            self.sampled_time += self.dt
            self.sampled_offset += self.dt / self.sampled_time * (offset - self.sampled_offset)
        decay = np.exp(-self.dt / self.time_constant)
        charge = self.electrode_charge
        charge -= self.capacitance * (voltage - self.target_cell_voltage) * (1.0 - decay)
        charge += np.sqrt(self._phi_kbt * self.capacitance * (1.0 - decay * decay)) * self._normal()
        self._set_counter_charge(-charge)

    def _potentiostat_dict(self) -> dict:
        return {
            "target_potential_V_SHE": self.target_potential,
            "cell_voltage_V": self.cell_voltage,
            "target_cell_voltage_V": self.target_cell_voltage,
            "she_offset_V": self.mean_offset,
            "aligned": self.aligned,
            "averaged_time_fs": self.averaged_time / units.fs,
            "alignment_time_fs": self.alignment_time / units.fs,
            "sampled_time_fs": self.sampled_time / units.fs,
            "capacitance_e_per_V": self.capacitance,
            "time_constant_fs": self.time_constant / units.fs,
            "counter_charge_key": self.counter_charge_key,
            "total_charge_key": self.total_charge_key,
        }

    def potentiostat_state(self) -> dict:
        """Restart record of the controller (the averages of Delta, their times and the flag)."""
        return dict(mean_offset=self.mean_offset, aligned=self.aligned, averaged_time=self.averaged_time,
                    sampled_offset=self.sampled_offset, sampled_time=self.sampled_time)

    def restore_potentiostat_state(self, state: dict) -> None:
        self.mean_offset = None if state["mean_offset"] is None else float(state["mean_offset"])
        self.aligned = bool(state["aligned"])
        self.averaged_time = float(state.get("averaged_time", 0.0))
        self.sampled_offset = float(state.get("sampled_offset", 0.0))
        self.sampled_time = float(state.get("sampled_time", 0.0))
