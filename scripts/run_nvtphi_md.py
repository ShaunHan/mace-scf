#!/usr/bin/env python3
"""Constant-potential (NVTPhi) MD of an electrode slab in water at U* vs SHE.

  relax       Langevin MD at Q = 0, averaging Delta = WF - eV (counter charge to vacuum);
  align       thermopotentiostat at V* = 4.44 V + U* - <Delta>, <Delta> averaged online
              and then frozen, so that <U_SHE> = U*;
  production  the same dynamics with V* fixed.
A known SHE_OFFSET skips the alignment. atoms.info needs at most total_charge =
counter_charge = 0: the calculator places the counter-charge sheet, the dipole cut
and the vacuum plane from the geometry. The sheet stands for solvated ions and acts
through electrostatics only.
"""
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation

# ------------------------------ user settings ------------------------------ #
ATOMS_FILE = "IrO2_water_interface.traj"
MODEL_FILE = "fit_0_best_wf_final.model"
TARGET_U_SHE = 1.60           # V vs SHE: imposes <WF> = 4.44 eV + 1.60 eV.
SHE_OFFSET = None             # <Delta> in V from an earlier run; None aligns it online.
SHEET_HEIGHT = 6.0            # Angstrom above the electrode surface, inside the water.
SHEET_WIDTH = 1.0             # Gaussian sigma in Angstrom.
RELAX_FS = 500.0              # At Q = 0 first.
ALIGN_FS = 1500.0             # Then aligning <Delta> at constant potential.
ALIGNMENT_CONSTANT_FS = 250.0 # Memory of the online <Delta>.

TEMPERATURE_K = 300.0
TIMESTEP_FS = 0.5
FRICTION_PER_FS = 0.01
TAUPHI_FS = 100.0
CAPACITANCE = 0.05            # C0 in e/V for the whole cell; sets the charge relaxation rate only.
STEPS = 10000                 # Production steps after relaxation and alignment.
MAX_CHARGE_E = 1.0            # Stop if |Q| exceeds it: the model has left its range.
SEED = 7777
SCF_STEPS = None              # Updates per solve; None runs the model's last training stage.

DEVICE = "cuda"
TRAJECTORY = "nvtphi.traj"
LOGFILE = "nvtphi.log"
LOGINTERVAL = 10
# -------------------------------------------------------------------------- #

PLANE_KEYS = ("counter_charge_zfrac", "counter_charge_width", "dipole_correction_zfrac", "vacuum_zfrac")


def prepare_frame(source):
    """The slab open along z at Q = 0, keeping any planes the frame already fixes."""
    atoms = source.copy()
    if not np.allclose(atoms.cell.array[:2, 2], 0.):
        raise ValueError("This driver expects an xy-periodic slab, open along Cartesian z")
    atoms.pbc, atoms.calc = [True, True, False], None
    atoms.info = {"total_charge": 0., "counter_charge": 0.,
                  **{key: float(source.info[key]) for key in PLANE_KEYS if source.info.get(key) is not None}}
    return atoms


def main() -> None:
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    from mace_scf.md import NVTPhiLangevin

    for path in (ATOMS_FILE, MODEL_FILE):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    rng = np.random.default_rng(SEED)
    atoms = prepare_frame(read(ATOMS_FILE, index=-1))
    atoms.calc = MACEFixedPointSCF(model_path=MODEL_FILE, device=DEVICE, default_dtype="float64",
                                   constant_potential=True, scf_options={} if SCF_STEPS is None else {"num_scf_steps": SCF_STEPS},
                                   ignore_nonconverged=True, counter_charge_height=SHEET_HEIGHT,
                                   counter_charge_width=SHEET_WIDTH)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)
    relax, align = int(round(RELAX_FS/TIMESTEP_FS)), 0 if SHE_OFFSET is not None else int(round(ALIGN_FS/TIMESTEP_FS))
    dyn = NVTPhiLangevin(atoms=atoms, timestep=TIMESTEP_FS*units.fs, temperature_K=TEMPERATURE_K,
                         friction=FRICTION_PER_FS/units.fs, target_potential=TARGET_U_SHE, she_offset=SHE_OFFSET,
                         alignment_time=(relax+align)*TIMESTEP_FS*units.fs,
                         alignment_constant=ALIGNMENT_CONSTANT_FS*units.fs, C0=CAPACITANCE,
                         tauphi=TAUPHI_FS*units.fs, trajectory=TRAJECTORY, logfile=LOGFILE,
                         loginterval=LOGINTERVAL, rng=rng)
    length, cut = atoms.cell[2, 2], atoms.info["dipole_correction_zfrac"]
    z = ((atoms.get_scaled_positions(wrap=True)[:, 2]-cut) % 1.)*length
    print(f"Heights above the dipole cut (L = {length:.2f} A): atoms {z.min():.2f}..{z.max():.2f}, sheet "
          f"{((atoms.info['counter_charge_zfrac']-cut) % 1.)*length:.2f}, vacuum plane "
          f"{((atoms.info['vacuum_zfrac']-cut) % 1.)*length:.2f}")

    def status() -> None:
        r, q = atoms.calc.results, dyn.electrode_charge
        print(f"step={dyn.nsteps:7d} T={atoms.get_temperature():6.1f} K Q={q:+.4f} e V={r['cell_voltage']:.4f} "
              f"V*={dyn.target_cell_voltage:.4f} WF={r['work_function']:.4f} U={r['electrode_potential_she']:+.4f} "
              f"<Delta>={dyn.mean_offset:.4f}{'' if dyn.aligned else ' (aligning)'}", flush=True)
        if abs(q) > MAX_CHARGE_E:
            raise RuntimeError(f"|Q| = {abs(q):.3f} e exceeds MAX_CHARGE_E: the model has left its range")

    dyn.attach(status, interval=LOGINTERVAL)
    dyn.frozen = True
    dyn.run(relax)
    dyn.frozen = False
    dyn.run(align+STEPS)
    print(f"<Delta> = {dyn.mean_offset:.5f} V; reuse it as SHE_OFFSET to skip the alignment.")


if __name__ == "__main__":
    main()
