from __future__ import annotations
from typing import IO, Optional, Union
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.parallel import world
from ._nvtphi import NVTPhiMixin
from .logger import NVTPhiMDLogger


class NVTPhiLangevin(NVTPhiMixin, Langevin):
    """Langevin nuclei plus the thermopotentiostat charge; one evaluation per step."""

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature_K: float,
        friction: float,
        target_potential: Optional[float] = None,
        C0: float = 0.05,
        tauphi: float = 100.0 * units.fs,
        cell_voltage: Optional[float] = None,
        she_offset: Optional[float] = None,
        alignment_time: float = 1000.0 * units.fs,
        alignment_constant: float = 250.0 * units.fs,
        counter_charge_key: str = "counter_charge",
        total_charge_key: str = "total_charge",
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        communicator=world,
        rng=None,
        append_trajectory: bool = False,
        fixcm: bool = True,
    ):
        self._set_potentiostat_parameters(
            temperature_K, target_potential, C0, tauphi, counter_charge_key, total_charge_key,
            cell_voltage=cell_voltage, she_offset=she_offset,
            alignment_time=alignment_time, alignment_constant=alignment_constant)
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
            logger = self.closelater(NVTPhiMDLogger(self, atoms, logfile))
            self.attach(logger, loginterval)

    def todict(self):
        result = Langevin.todict(self)
        result.update(self._potentiostat_dict())
        return result

    def step(self, forces=None):
        # Forces and voltage of the cached evaluation at (R_n, Q_n).
        if forces is None:
            forces = self.atoms.get_forces(md=True)
        self._update_charge()
        # Half kick with F(R_n, Q_n), drift, ONE evaluation at (R_n+1, Q_n+1).
        return Langevin.step(self, forces)
