from __future__ import annotations
import weakref
from typing import IO, Any, Union
import numpy as np
from ase import Atoms, units
from ase.parallel import world
from ase.utils import IOContext


class NVTPhiMDLogger(IOContext):
    """Energies, temperature, electrode charge and the three voltages.

    V is the controlled cell voltage, V* its target, WF the solution-side work
    function (U_SHE = WF/e - 4.44 V) and Delta = WF - eV. Omega = E - e V* Q.
    No column triggers an additional electronic evaluation.
    """

    def __init__(self, dyn: Any, atoms: Atoms, logfile: Union[IO, str], header: bool = True,
                 mode: str = "a", comm=world):
        self.dyn = weakref.proxy(dyn)
        self.atoms = atoms
        self.logfile = self.openfile(file=logfile, mode=mode, comm=comm)
        columns = ("Time[ps]", "Epot[eV]", "Omega[eV]", "Ekin[eV]", "T[K]", "Q[e]",
                   "V[V]", "V*[V]", "WF[eV]", "U_SHE[V]", "Delta[V]")
        self.fmt = "%-10.4f %16.6f %16.6f %12.4f %8.1f %12.6f %10.5f %10.5f %10.5f %10.5f %10.5f\n"
        if header:
            self.logfile.write(" ".join(f"{c:>12s}" for c in columns) + "\n")

    def __del__(self):
        self.close()

    def __call__(self):
        results = self.atoms.calc.results
        epot = self.atoms.get_potential_energy(apply_constraint=False)
        voltage = float(results["cell_voltage"])
        wf = float(results["work_function"])
        self.logfile.write(self.fmt % (
            self.dyn.get_time() / (1000.0 * units.fs), epot, self.dyn.grand_potential(),
            self.atoms.get_kinetic_energy(), self.atoms.get_temperature(), self.dyn.electrode_charge,
            voltage, self.dyn.target_cell_voltage, wf, wf - 4.44, float(results["she_offset"])))
        self.logfile.flush()
