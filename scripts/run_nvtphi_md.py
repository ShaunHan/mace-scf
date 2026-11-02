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
# The dipole correction plane and the counter charge sheet, as fractions along the slab's non-periodic axis;
# None takes them from the frame.
DIPOLE_CORRECTION_FRACTION = None
COUNTER_CHARGE_FRACTION = None
COUNTER_CHARGE_WIDTH = 1.0    # Gaussian sigma of the sheet in Angstrom.
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

def plane(source, name, value):
    """A plane of the slab as a fraction along its non-periodic axis: the setting, else the frame's name or the
    collector's key of that axis."""
    letter = "xyz"[int(np.flatnonzero(~source.pbc)[0])]
    for candidate in (value, source.info.get(name), source.info.get(name.replace("_fraction", f"_{letter}frac"))):
        if candidate is not None:
            return float(candidate)
    raise ValueError(f"Set {name.upper()} or give {name} in the frame's info")


def prepare_frame(source):
    """The slab at Q = 0 with its dipole correction plane and counter charge sheet."""
    atoms = source.copy()
    atoms.calc = None
    atoms.info = {"total_charge": 0., "counter_charge": 0., "counter_charge_width": COUNTER_CHARGE_WIDTH,
                  "dipole_correction_fraction": plane(source, "dipole_correction_fraction", DIPOLE_CORRECTION_FRACTION),
                  "counter_charge_fraction": plane(source, "counter_charge_fraction", COUNTER_CHARGE_FRACTION)}
    return atoms


def main() -> None:
    from mace_scf.calculators.potential_response import MACEPotentialResponse
    from mace_scf.md import NVTPhiLangevin

    for path in (ATOMS_FILE, MODEL_FILE):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    rng = np.random.default_rng(SEED)
    atoms = prepare_frame(read(ATOMS_FILE, index=-1))
    atoms.calc = MACEPotentialResponse(model_path=MODEL_FILE, device=DEVICE, default_dtype="float64",
                                       constant_potential=True)
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
    axis = int(np.flatnonzero(~atoms.pbc)[0])
    cut, sheet = atoms.info["dipole_correction_fraction"], atoms.info["counter_charge_fraction"]
    z = (atoms.get_scaled_positions(wrap=True)[:, axis]-cut) % 1.
    print(f"Fractions above the dipole correction plane: atoms {z.min():.3f}..{z.max():.3f}, sheet "
          f"{(sheet-cut) % 1.:.3f}")

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
