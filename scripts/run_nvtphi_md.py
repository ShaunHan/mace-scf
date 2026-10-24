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

# ------------------------------ user settings ------------------------------ #
ATOMS_FILE = "IrO2_water_interface.traj"
MODEL_FILE = "models/fit_0_best_wf.model"  # Export from the new shared-energy training.
TARGET_U_SHE = 1.60  # Measured WF/e - conventional absolute SHE (4.44 V).
# Input-frame values take precedence; these defaults apply only if absent.
COUNTER_CHARGE_ZFRAC = 0.20   # Inside water for the supplied ~50 A cell.
VACUUM_ZFRAC = 0.40          # In the solution-side vacuum.
DIPOLE_CORRECTION_ZFRAC = 0.50
GAUSSIAN_WIDTH = 0.5         # Angstrom; use the width in the DFT data.

TEMPERATURE_K = 300.0
TIMESTEP_FS = 0.5
FRICTION_PER_FS = 0.01
TAUPHI_FS = 100.0
CAPACITANCE = 0.01
STEPS = 10000
SEED = 7777

SCF_STEPS = 50

DEVICE = "cuda"
TRAJECTORY = "nvtphi.traj"
LOGFILE = "nvtphi.log"
LOGINTERVAL = 10
# -------------------------------------------------------------------------- #


def scalar(value) -> float:
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f"Expected one finite scalar, got {value}")
    return float(a.item())


def prepare_frame(source):
    atoms, info = source.copy(), source.info
    if not np.allclose(atoms.cell.array[:2, 2], 0.):
        raise ValueError("This driver expects an xy-periodic slab, open along Cartesian z")
    atoms.pbc = [True, True, False]
    planes = dict(counter_charge_zfrac=COUNTER_CHARGE_ZFRAC,
                  vacuum_zfrac=VACUUM_ZFRAC, dipole_correction_zfrac=DIPOLE_CORRECTION_ZFRAC)
    planes = {key: scalar(info.get(key, value)) for key, value in planes.items()}
    if any(not 0. <= scalar(value) < 1. for value in planes.values()):
        raise ValueError("Plane zfrac values must be finite and in [0, 1)")
    charge = scalar(info.get("total_charge", 0.))
    width = scalar(info.get('counter_charge_width', GAUSSIAN_WIDTH))
    if width == 0. and charge == 0.:
        width = GAUSSIAN_WIDTH
    if width <= 0. or (charge != 0. and 'counter_charge_center' in info
                      and 'counter_charge_zfrac' not in info):
        raise ValueError('Specify a Gaussian sheet with positive width and counter_charge_zfrac')
    if charge != 0.:
        if not np.isclose(scalar(info.get("counter_charge", np.nan)), -charge):
            raise ValueError("A charged starting reference must already have counter_charge = -total_charge")
    for prefix in ("vacuum", "dipole_correction", "counter_charge"):
        for letter in "xyz":
            atoms.info.pop(f"{prefix}_{letter}frac", None)
    atoms.info.pop("counter_charge_center", None)
    atoms.info.pop("vacuum_reservoir", None)
    atoms.info.update(
        counter_charge=-charge, total_charge=charge,
        counter_charge_width=width,
        external_field=np.asarray(info.get("external_field", np.zeros(3)), dtype=float),
        **planes,
    )
    atoms.calc = None
    return atoms


def main() -> None:
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    from mace_scf.md import NVTPhiLangevin

    scalar(TARGET_U_SHE)
    if not Path(ATOMS_FILE).is_file():
        raise FileNotFoundError(ATOMS_FILE)
    if not Path(MODEL_FILE).is_file():
        raise FileNotFoundError(MODEL_FILE)
    rng = np.random.default_rng(SEED)
    atoms = prepare_frame(read(ATOMS_FILE, index=-1))

    atoms.calc = MACEFixedPointSCF(
        model_path=MODEL_FILE,
        device=DEVICE,
        pbc_handling="slab",
        counter_charge_width=GAUSSIAN_WIDTH,
        scf_restart=False,
        voltage_reference='SHE',
        scf_options={
            "num_scf_steps": SCF_STEPS,
            "constant_charge": True,
            "use_autograd_forces": True,
            "initial_density": "local_guess",
            "initial_fermi_level": "zero",
        },
        ignore_nonconverged=True,  # Finite unrolled model, as in the umbrella driver.
    )

    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    initial_energy = atoms.get_potential_energy()
    print(f"Initial potential energy: {initial_energy:.10f} eV")
    print(f"Initial voltage vs SHE: {atoms.calc.results['md_charge_potential']:.10f} V")
    print(f"Target voltage vs SHE: {TARGET_U_SHE:.10f} V")

    dyn = NVTPhiLangevin(
        atoms=atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURE_K,
        friction=FRICTION_PER_FS / units.fs,
        target_potential=TARGET_U_SHE,
        C0=CAPACITANCE,
        tauphi=TAUPHI_FS * units.fs,
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
            f"U_SHE={float(dyn.current_potential):10.5f} V "
            f"q={-scalar(atoms.info['counter_charge']):10.5f} e "
            f"SCF steps={results.get('num_scf_steps', 0)} "
            f"residual={results.get('scf_residual', np.nan):.3e}"
        )

    dyn.attach(print_status, interval=LOGINTERVAL)
    dyn.run(STEPS)


if __name__ == "__main__":
    main()
