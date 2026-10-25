#!/usr/bin/env python3
"""Constant-potential (NVT-Phi) MD with a Gaussian counter charge in the solution.

Hamiltonian Omega(R, Q) = E(R, Q, -Q) - e V* Q. Forces are the MACE-VOLT DFT
forces; the electrode charge follows the thermopotentiostat on the cell
voltage V = dE/dQ (one electronic evaluation per step). U_SHE = <WF>/e - 4.44 V
with WF on the solution side; V* = 4.44 V + U* - <Delta> is aligned during the
first ALIGNMENT_FS and then frozen (Delta = WF - eV contains no Fermi level).
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
# Counter charge (input-frame values take precedence): a sheet at a fractional
# coordinate along the slab normal, OR a 3D Gaussian centre (Cartesian, Angstrom).
COUNTER_CHARGE_ZFRAC = 0.20   # Inside the water; None when using COUNTER_CHARGE_CENTER.
COUNTER_CHARGE_CENTER = None  # e.g. (4.8, 3.2, 10.0) for a localized 3D Gaussian.
GAUSSIAN_WIDTH = 1.0          # Angstrom; the width used in the DFT plug-in data.
VACUUM_ZFRAC = 0.40           # Solution-side vacuum plane for WF (beyond the water).
DIPOLE_CORRECTION_ZFRAC = 0.50

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


def scalar(value) -> float:
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError(f"Expected one finite scalar, got {value}")
    return float(a.item())


def counter_charge_controls(info, normal_axis=2):
    """Sheet fraction or 3D centre, the Gaussian width, and the planes."""
    keys = [f"counter_charge_{a}frac" for a in "xyz"]
    sheet = {k: scalar(info[k]) for k in keys if info.get(k) is not None}
    center = info.get("counter_charge_center")
    if sheet and center is not None:
        raise ValueError("The frame specifies both a counter-charge sheet and a 3D centre")
    if not sheet and center is None:
        if (COUNTER_CHARGE_ZFRAC is None) == (COUNTER_CHARGE_CENTER is None):
            raise ValueError("Set exactly one of COUNTER_CHARGE_ZFRAC or COUNTER_CHARGE_CENTER")
        if COUNTER_CHARGE_ZFRAC is not None:
            sheet = {f"counter_charge_{'xyz'[normal_axis]}frac": float(COUNTER_CHARGE_ZFRAC)}
        else:
            center = COUNTER_CHARGE_CENTER
    controls = dict(sheet)
    if center is not None:
        center = np.asarray(center, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("counter_charge_center must be three finite Cartesian coordinates")
        controls["counter_charge_center"] = center
    if any(not 0. <= value < 1. for value in sheet.values()):
        raise ValueError("Counter-charge sheet fractions must be in [0, 1)")
    width = scalar(info.get("counter_charge_width") or GAUSSIAN_WIDTH)
    if width <= 0.:
        raise ValueError("The Gaussian width must be positive")
    controls["counter_charge_width"] = width
    for key, value in (("vacuum_zfrac", VACUUM_ZFRAC), ("dipole_correction_zfrac", DIPOLE_CORRECTION_ZFRAC)):
        controls[key] = scalar(info.get(key, value))
        if not 0. <= controls[key] < 1.:
            raise ValueError(f"{key} must be in [0, 1)")
    return controls


def prepare_frame(source):
    atoms, info = source.copy(), source.info
    if not np.allclose(atoms.cell.array[:2, 2], 0.):
        raise ValueError("This driver expects an xy-periodic slab, open along Cartesian z")
    atoms.pbc = [True, True, False]
    charge = scalar(info.get("total_charge", 0.))
    if charge != 0. and not np.isclose(scalar(info.get("counter_charge", np.nan)), -charge):
        raise ValueError("A charged starting frame must have counter_charge = -total_charge")
    controls = counter_charge_controls(info)
    atoms.info = dict(total_charge=charge, counter_charge=-charge,
                      external_field=np.asarray(info.get("external_field", np.zeros(3)), dtype=float),
                      **controls)
    if np.any(atoms.info["external_field"]):
        raise ValueError("The electrode potential is set by the potentiostat; use zero external field")
    report_geometry(atoms)
    atoms.calc = None
    return atoms


def report_geometry(atoms):
    """Print the planes along z on the dipole-cut branch, with warnings."""
    length = atoms.cell[2, 2]
    cut = atoms.info["dipole_correction_zfrac"]
    def branch(fraction):
        return ((fraction - cut) % 1.) * length
    z = branch(atoms.get_scaled_positions(wrap=False)[:, 2])
    vacuum = branch(atoms.info["vacuum_zfrac"])
    print(f"Along the normal (0 = dipole cut, L = {length:.3f} A): atoms {z.min():.2f}..{z.max():.2f} A, "
          f"vacuum observer {vacuum:.2f} A")
    if "counter_charge_zfrac" in atoms.info:
        print(f"  counter-charge sheet at {branch(atoms.info['counter_charge_zfrac']):.2f} A, "
              f"sigma {atoms.info['counter_charge_width']:.2f} A")
    else:
        center = atoms.info["counter_charge_center"]
        print(f"  3D counter-charge Gaussian at {np.round(center, 3)} A, sigma {atoms.info['counter_charge_width']:.2f} A")
    clearance = np.min(np.abs(z - vacuum))
    if clearance < 3.0:
        print(f"WARNING: the vacuum observer is only {clearance:.2f} A from an atom; WF needs a field-free plateau")


def main() -> None:
    import torch
    from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
    from mace_scf.md import NVTPhiLangevin

    for path in (ATOMS_FILE, MODEL_FILE):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    model_file, frame = MODEL_FILE, read(ATOMS_FILE, index=-1)
    model = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)
    if not bool(getattr(model.field_dependent_charges_map, "neutral_atom_fitted", False)):
        # A model trained before the form factors travelled with the data
        # takes them from the starting frame, if that frame carries them.
        from mace_scf.data.new_atomic_data import neutral_atom_reference
        from mace_scf.electrostatics.potential import install_neutral_atom_reference
        reference = neutral_atom_reference([frame.info])
        if not reference:
            raise ValueError(f"{MODEL_FILE} has no neutral-atom form factors and {ATOMS_FILE} carries none; "
                             "start from a frame collected by scripts/collect_vasp_data.py")
        install_neutral_atom_reference(model.field_dependent_charges_map, reference,
                                       [int(z) for z in model.atomic_numbers])
        model_file = str(Path(MODEL_FILE).with_suffix(".counter.model"))
        torch.save(model, model_file)
    del model
    rng = np.random.default_rng(SEED)
    atoms = prepare_frame(frame)
    atoms.calc = MACEFixedPointSCF(model_path=model_file, device=DEVICE, default_dtype="float64",
                                   constant_potential=True, scf_options={"num_scf_steps": SCF_STEPS},
                                   ignore_nonconverged=True)
    MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=rng)
    Stationary(atoms)
    ZeroRotation(atoms)

    dyn = NVTPhiLangevin(atoms=atoms, timestep=TIMESTEP_FS * units.fs, temperature_K=TEMPERATURE_K,
                         friction=FRICTION_PER_FS / units.fs, target_potential=TARGET_U_SHE,
                         she_offset=SHE_OFFSET, alignment_time=ALIGNMENT_FS * units.fs,
                         C0=CAPACITANCE, tauphi=TAUPHI_FS * units.fs, trajectory=TRAJECTORY,
                         logfile=LOGFILE, loginterval=LOGINTERVAL, rng=rng)
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
