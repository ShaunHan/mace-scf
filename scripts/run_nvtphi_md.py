#!/usr/bin/env python3

from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)

from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from mace_scf.md import NVTPhiLangevin


# ------------------------------ user settings ------------------------------ #
ATOMS_FILE = "IrO2_water_interface.traj"
MODEL_FILE = "fit_0.model"
TARGET_PAIR_VOLTAGE = None  # Physical electrode/counter transfer voltage; not V vs SHE.
VACUUM_POTENTIAL_KEY = "vacuum_potential"
FERMI_LEVEL_KEY = "fermi_level"

# Change both suffixes to xfrac or yfrac for a different open cell axis.
PLANE_FRACTIONS = {"vacuum_zfrac": 0.75, "dipole_correction_zfrac": 0.50}
COUNTER_SHEET_POINT = [0.0, 0.0, 10.0]  # Cartesian point on the sheet; converted to a cell fraction below.
INITIAL_COUNTER_CHARGE = 0.0
GAUSSIAN_WIDTH = 1.0

TEMPERATURE_K = 300.0
TIMESTEP_FS = 0.5
FRICTION_PER_FS = 0.01
TAUPHI_FS = 100.0
CAPACITANCE = 0.01
STEPS = 10000
SEED = 7777

SCF_STEPS = 50
SCF_TOLERANCE = 1.0e-5
STRICT_SCF = True

DEVICE = "cuda"
TRAJECTORY = "nvtphi.traj"
LOGFILE = "nvtphi.log"
LOGINTERVAL = 10
# -------------------------------------------------------------------------- #


def scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:
    if TARGET_PAIR_VOLTAGE is None or not np.isfinite(TARGET_PAIR_VOLTAGE):
        raise ValueError("Set the physical TARGET_PAIR_VOLTAGE; there is no arbitrary SHE offset")
    if not Path(ATOMS_FILE).is_file():
        raise FileNotFoundError(ATOMS_FILE)
    if not Path(MODEL_FILE).is_file():
        raise FileNotFoundError(MODEL_FILE)

    rng = np.random.default_rng(SEED)
    atoms = read(ATOMS_FILE, index=-1)
    axes = [i for i, letter in enumerate("xyz")
            if set(PLANE_FRACTIONS) == {f"{prefix}_{letter}frac" for prefix in
                                      ("vacuum", "dipole_correction")}]
    if len(axes) != 1:
        raise ValueError("Choose vacuum and dipole-correction planes on the same cell axis")
    axis = axes[0]
    atoms.pbc = True
    atoms.pbc[axis] = False
    for prefix in ("vacuum", "dipole_correction", "counter_charge"):
        for letter in "xyz":
            atoms.info.pop(f"{prefix}_{letter}frac", None)
    atoms.info.update(PLANE_FRACTIONS)
    atoms.info.pop("counter_charge_center", None)
    atoms.info[f"counter_charge_{'xyz'[axis]}frac"] = float(
        (np.asarray(COUNTER_SHEET_POINT) @ np.linalg.inv(atoms.cell.array))[axis] % 1.)
    atoms.info.update(
        counter_charge=float(INITIAL_COUNTER_CHARGE),
        total_charge=-float(INITIAL_COUNTER_CHARGE),
        counter_charge_width=GAUSSIAN_WIDTH,
        external_field=np.zeros(3, dtype=float),
    )

    atoms.calc = MACEFixedPointSCF(
        model_path=MODEL_FILE,
        device=DEVICE,
        external_field_key="external_field",
        total_charge_key="total_charge",
        pbc_handling="slab",
        counter_charge_width=GAUSSIAN_WIDTH,
        scf_restart=False,
        voltage_reference='pair',
        scf_options={
            "num_scf_steps": SCF_STEPS,
            "scf_tolerance": SCF_TOLERANCE,
            "constant_charge": True,
            "use_autograd_forces": True,
            "initial_density": "local_guess",
            "initial_fermi_level": "zero",
        },
        ignore_nonconverged=not STRICT_SCF,
    )

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    initial_energy = atoms.get_potential_energy()
    print(f"Initial DFT-convention energy: {initial_energy:.10f} eV")
    print(f"Initial pair voltage: {atoms.calc.results['md_charge_potential']:.10f} V")
    print(f"Target pair voltage: {TARGET_PAIR_VOLTAGE:.10f} V")

    dyn = NVTPhiLangevin(
        atoms=atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION_PER_FS / units.fs,
        target_potential=TARGET_PAIR_VOLTAGE,
        C0=CAPACITANCE,
        tauphi=TAUPHI_FS * units.fs,
        counter_charge_key="counter_charge",
        total_charge_key="total_charge",
        vacuum_potential_key=VACUUM_POTENTIAL_KEY,
        fermi_level_key=FERMI_LEVEL_KEY,
        reference_potential=None,
        trajectory=TRAJECTORY,
        logfile=LOGFILE,
        loginterval=LOGINTERVAL,
        rng=rng,
    )

    def print_status() -> None:
        results = atoms.calc.results
        print(
            f"step={dyn.nsteps:8d} "
            f"T={atoms.get_temperature():8.2f} K "
            f"U_pair={float(dyn.current_potential):10.5f} V "
            f"q={-scalar(atoms.info['counter_charge']):10.5f} e "
            f"SCF steps={results.get('num_scf_steps', 0)} "
            f"residual={results.get('scf_residual', np.nan):.3e}"
        )

    dyn.attach(print_status, interval=LOGINTERVAL)
    dyn.run(STEPS)


if __name__ == "__main__":
    main()
