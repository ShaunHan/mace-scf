#!/usr/bin/env python3
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

DEVICE = "cuda"
TRAJECTORY = "nvtphi.traj"
LOGFILE = "nvtphi.log"
LOGINTERVAL = 10
# -------------------------------------------------------------------------- #

def prepare_frame(source):
    """The slab at Q = 0, keeping any planes the frame already fixes along its non-periodic axis."""
    from mace_scf.calculators.potential_response import slab_axis
    atoms = source.copy()
    letter = "xyz"[slab_axis(atoms)[0]]
    atoms.calc = None
    keys = (f"counter_charge_{letter}frac", "counter_charge_width", f"dipole_correction_{letter}frac")
    atoms.info = {"total_charge": 0., "counter_charge": 0.,
                  **{key: float(source.info[key]) for key in keys if source.info.get(key) is not None}}
    return atoms


def main() -> None:
    from mace_scf.calculators.potential_response import MACEPotentialResponse, slab_axis
    from mace_scf.md import NVTPhiLangevin

    for path in (ATOMS_FILE, MODEL_FILE):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    rng = np.random.default_rng(SEED)
    atoms = prepare_frame(read(ATOMS_FILE, index=-1))
    atoms.calc = MACEPotentialResponse(model_path=MODEL_FILE, device=DEVICE, default_dtype="float64",
                                       constant_potential=True, counter_charge_height=SHEET_HEIGHT,
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
    axis, _, length = slab_axis(atoms)
    letter = "xyz"[axis]
    cut, sheet = atoms.info[f"dipole_correction_{letter}frac"], atoms.info[f"counter_charge_{letter}frac"]
    z = ((atoms.get_scaled_positions(wrap=True)[:, axis]-cut) % 1.)*length
    print(f"Heights above the dipole cut (L = {length:.2f} A): atoms {z.min():.2f}..{z.max():.2f}, sheet "
          f"{((sheet-cut) % 1.)*length:.2f}")

    def status() -> None:
        r, q = atoms.calc.results, dyn.electrode_charge
        print(f"step={dyn.nsteps:7d} T={atoms.get_temperature():6.1f} K Q={q:+.4f} e V={r['cell_voltage']:.4f} "
              f"V*={dyn.target_cell_voltage:.4f} WF={r['workfunction']:.4f} U={r['electrode_potential_she']:+.4f} "
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
