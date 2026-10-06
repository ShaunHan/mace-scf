from __future__ import annotations
from typing import IO, Optional, Union
from ase import Atoms, units
from ase.md.md import MolecularDynamics
from ase.parallel import world
from ._nvtphi import NVTPhiMixin
from .logger import NVTPhiMDLogger


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
