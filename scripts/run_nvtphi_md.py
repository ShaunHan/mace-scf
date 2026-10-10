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
TARGET_POTENTIAL = 4.50
VOLTAGE_MODE = "relative"  # initial-structure reference for the derived potential
REFERENCE_POTENTIAL = 4.50  # physical potential assigned to the initial structure

# Change all three suffixes to xfrac or yfrac for a different open cell axis.
PLANE_FRACTIONS = {"vacuum_zfrac": 0.75, "dipole_correction_zfrac": 0.50,
                   "counter_charge_zfrac": 0.35}
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
    if not Path(ATOMS_FILE).is_file():
        raise FileNotFoundError(ATOMS_FILE)
    if not Path(MODEL_FILE).is_file():
        raise FileNotFoundError(MODEL_FILE)

    rng = np.random.default_rng(SEED)
    atoms = read(ATOMS_FILE, index=-1)
    axes = [i for i, letter in enumerate("xyz")
            if set(PLANE_FRACTIONS) == {f"{prefix}_{letter}frac" for prefix in
                                      ("vacuum", "dipole_correction", "counter_charge")}]
    if len(axes) != 1:
        raise ValueError("Choose vacuum, dipole-correction and counter-charge planes on the same cell axis")
    axis = axes[0]
    atoms.pbc = True
    atoms.pbc[axis] = False
    for prefix in ("vacuum", "dipole_correction", "counter_charge"):
        for letter in "xyz":
            atoms.info.pop(f"{prefix}_{letter}frac", None)
    atoms.info.update(PLANE_FRACTIONS)
    atoms.info.pop("counter_charge_center", None)
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
    if "workfunction" not in atoms.calc.results:
        raise RuntimeError(
            "The loaded model does not expose a workfunction observable. "
            "Use a model trained with potential and chemical-level observations."
        )
    if VOLTAGE_MODE not in {"relative", "absolute"}:
        raise ValueError("VOLTAGE_MODE must be 'relative' or 'absolute'")
    model_reference = scalar(atoms.calc.results["workfunction"])
    control_reference = (
        model_reference if REFERENCE_POTENTIAL is None else float(REFERENCE_POTENTIAL)
    )
    print(f"Initial energy: {initial_energy:.10f} eV")
    print(f"Initial model voltage: {model_reference:.10f} V")
    print(f"Initial control potential: {control_reference:.10f} V")
    print(f"Target potential: {TARGET_POTENTIAL:.10f} V")
    print(f"Voltage mode: {VOLTAGE_MODE}")

    dyn = NVTPhiLangevin(
        atoms=atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION_PER_FS / units.fs,
        target_potential=TARGET_POTENTIAL,
        C0=CAPACITANCE,
        tauphi=TAUPHI_FS * units.fs,
        counter_charge_key="counter_charge",
        total_charge_key="total_charge",
        workfunction_key="workfunction",
        use_relative_workfunction=VOLTAGE_MODE == "relative",
        reference_potential=control_reference,
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
            f"Phi={float(dyn.current_potential):10.5f} V "
            f"q={-scalar(atoms.info['counter_charge']):10.5f} e "
            f"SCF steps={results.get('num_scf_steps', 0)} "
            f"residual={results.get('scf_residual', np.nan):.3e}"
        )

    dyn.attach(print_status, interval=LOGINTERVAL)
    dyn.run(STEPS)


if __name__ == "__main__":
    main()
