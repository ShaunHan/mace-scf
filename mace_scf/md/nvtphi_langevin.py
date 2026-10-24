from __future__ import annotations
from typing import IO, Optional, Union
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.parallel import world
from ._nvtphi import NVTPhiMixin
from .logger import NVTPhiMDLogger


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
        vacuum_potential_key: str = "vacuum_potential",
        fermi_level_key: str = "fermi_level",
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
            vacuum_potential_key,
            fermi_level_key,
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
                    vacuum_potential_key=vacuum_potential_key,
                    fermi_level_key=fermi_level_key,
                )
            )
            self.attach(logger, loginterval)

    def todict(self):
        result = Langevin.todict(self)
        result.update(self._potentiostat_dict())
        return result

    def step(self, forces=None):
        self._sync_total_charge()
        if forces is None:
            forces = self.atoms.get_forces(md=True)
        # Explicit charge drift/noise uses the old state's voltage. Reuse its
        # forces for the first kick, then evaluate the new R and Q together.
        self._update_counter_charge()
        forces = Langevin.step(self, forces)
        self._read_potential()
        return forces
