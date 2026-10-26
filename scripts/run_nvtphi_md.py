#!/usr/bin/env python3
"""Constant-potential (NVT-Phi) MD with a Gaussian counter charge in the solution.

Hamiltonian Omega(R, Q) = E(R, Q, -Q) - e V* Q. Forces are the MACE-VOLT DFT
forces; the electrode charge follows the thermopotentiostat on the cell
voltage V = dE/dQ (one electronic evaluation per step). U_SHE = <WF>/e - 4.44 V
with WF on the solution side; V* = 4.44 V + U* - <Delta> is aligned during the
first ALIGNMENT_FS and then frozen (Delta = WF - eV contains no Fermi level).
Only the geometry is needed: the calculator places the sheet, the dipole cut and
the vacuum plane once, unless the starting frame gives them.
"""
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

# ------------------------------ user settings ------------------------------ #
ATOMS_FILE = "IrO2_water_interface.traj"
MODEL_FILE = "models/fit_0_best_wf.model"
TARGET_U_SHE = 1.60           # V vs SHE: imposes <WF> = 4.44 eV + 1.60 eV.
SHE_OFFSET = None             # <Delta> in V from a previous run; None aligns it online.
ALIGNMENT_FS = 1000.0         # Alignment period of <Delta>, then V* is frozen.
SHEET_HEIGHT = 6.0            # Angstrom above the electrode surface, inside the water.
SHEET_WIDTH = 1.0             # Gaussian sigma in Angstrom; the width of the DFT plug-in data.

TEMPERATURE_K = 300.0
TIMESTEP_FS = 0.5
FRICTION_PER_FS = 0.01
TAUPHI_FS = 100.0
CAPACITANCE = 0.05            # C0 in e/V for the whole cell; sets the charge relaxation rate only.
STEPS = 10000
SEED = 7777
SCF_STEPS = 50

DEVICE = "cuda"
TRAJECTORY = "nvtphi.traj"
LOGFILE = "nvtphi.log"
LOGINTERVAL = 10
# -------------------------------------------------------------------------- #

PLANE_KEYS = ("counter_charge_zfrac", "counter_charge_width", "dipole_correction_zfrac", "vacuum_zfrac")


def prepare_frame(source):
    """The slab open along z, its charge and any planes the frame already fixes."""
    atoms = source.copy()
    if not np.allclose(atoms.cell.array[:2, 2], 0.):
        raise ValueError("This driver expects an xy-periodic slab, open along Cartesian z")
    atoms.pbc = [True, True, False]
    charge = float(source.info.get("total_charge", 0.))
    atoms.info = {"total_charge": charge, "counter_charge": -charge, "external_field": np.zeros(3),
                  **{key: float(source.info[key]) for key in PLANE_KEYS if source.info.get(key) is not None}}
    atoms.calc = None
    return atoms


def report_geometry(atoms):
    """Heights above the dipole cut of the atoms, the sheet and the vacuum plane."""
    length, cut = atoms.cell[2, 2], atoms.info["dipole_correction_zfrac"]

    def height(fraction):
        return ((fraction - cut) % 1.) * length

    z = height(atoms.get_scaled_positions(wrap=True)[:, 2])
    vacuum = height(atoms.info["vacuum_zfrac"])
    print(f"Heights above the dipole cut (L = {length:.2f} A): atoms {z.min():.2f}..{z.max():.2f}, "
          f"sheet {height(atoms.info['counter_charge_zfrac']):.2f} (sigma {atoms.info['counter_charge_width']:.2f}), "
          f"vacuum plane {vacuum:.2f}")
    if np.min(np.abs(z - vacuum)) < 3.0:
        print("WARNING: the vacuum plane is within 3 A of an atom; WF needs a field-free vacuum")


def main() -> None:
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    from mace_scf.md import NVTPhiLangevin

    for path in (ATOMS_FILE, MODEL_FILE):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    rng = np.random.default_rng(SEED)
    atoms = prepare_frame(read(ATOMS_FILE, index=-1))
    atoms.calc = MACEFixedPointSCF(model_path=MODEL_FILE, device=DEVICE, default_dtype="float64",
                                   constant_potential=True, scf_options={"num_scf_steps": SCF_STEPS},
                                   ignore_nonconverged=True, counter_charge_height=SHEET_HEIGHT,
                                   counter_charge_width=SHEET_WIDTH)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = NVTPhiLangevin(atoms=atoms, timestep=TIMESTEP_FS * units.fs, temperature_K=TEMPERATURE_K,
                         friction=FRICTION_PER_FS / units.fs, target_potential=TARGET_U_SHE,
                         she_offset=SHE_OFFSET, alignment_time=ALIGNMENT_FS * units.fs,
                         C0=CAPACITANCE, tauphi=TAUPHI_FS * units.fs, trajectory=TRAJECTORY,
                         logfile=LOGFILE, loginterval=LOGINTERVAL, rng=rng)
    report_geometry(atoms)
    r = atoms.calc.results
    print(f"Initial: E={r['energy']:.6f} eV, V={r['cell_voltage']:.4f} V, WF={r['work_function']:.4f} eV, "
          f"Delta={r['she_offset']:.4f} V, V*={dyn.target_cell_voltage:.4f} V")

    def status() -> None:
        r = atoms.calc.results
        print(f"step={dyn.nsteps:8d} T={atoms.get_temperature():7.1f} K Q={dyn.electrode_charge:+.5f} e "
              f"V={r['cell_voltage']:.4f} V*={dyn.target_cell_voltage:.4f} WF={r['work_function']:.4f} "
              f"U_SHE={r['work_function'] - 4.44:+.4f} Delta={r['she_offset']:.4f}"
              f"{'' if dyn.aligned else ' (aligning)'}", flush=True)

    dyn.attach(status, interval=LOGINTERVAL)
    dyn.run(STEPS)
    print(f"Final aligned <Delta> = {dyn.mean_offset:.5f} V; reuse it as SHE_OFFSET to skip alignment.")


if __name__ == "__main__":
    main()
