from __future__ import annotations
import weakref
from typing import IO, Any, Union
import numpy as np
from ase import Atoms, units
from ase.parallel import world
from ase.utils import IOContext


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
