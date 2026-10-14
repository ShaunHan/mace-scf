"""Batch single points: python dft_runs.py vasp_0_4.

Each output_i contains a neutral fixed-density reference (LOCPOT_proto,
AECCAR1_proto) and one converged SCF calculation. No manual interruption.
Charged slabs need a VASP build with Python plugins and LREMOVE_DRIFT support.
The generated plugin needs only NumPy; this driver needs ASE and pymatgen.
"""
from pathlib import Path
import json
import pickle
import re
import shutil
import sys
import time
import traceback

import numpy as np
from ase.calculators.vasp import Vasp, create_input
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Locpot

# Keep the compatibility fix from the original scripts, without adding duplicates.
if "pomass" in create_input.float_keys:
    create_input.float_keys.remove("pomass")
if "pomass" not in create_input.list_float_keys:
    create_input.list_float_keys.append("pomass")


BASE_PATH = Path("/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft/")
SYSTEM = "IrO2OH_110"
INPUT = None  # Optional XYZ/trajectory path; otherwise use the original batch file.
INDEXES = None  # Optional frame indexes in INPUT; None selects all frames.
COMMAND = "mpirun vasp_std"
VASP_VERSION = (6, 6, 1)  # Load your plugin-enabled VASP 6.6.1 executable.

# Written automatically into each charged calculation directory.
# Same Gaussian-sheet construction as eisenforschung/VASP-Python; the analytic
# ionic forces, countercharge self-energy and variational dipole energy are
# included here. Fixed cell only: no plugin stress is supplied.
PLUGIN = r'''
import json
from pathlib import Path
import numpy as np

C = json.loads(Path("dft_metadata.json").read_text())
K = 1.0 / 0.005526349406  # eV Angstrom / e^2, matching MACE-VOLT.
_geometry = None
_last = None


def geometry(c):
    global _geometry
    if _geometry is not None:
        return _geometry
    axis = C["slab_axis"]
    cell = np.asarray(c.lattice_vectors)
    normal = np.linalg.inv(cell)[:, axis]
    normal /= np.linalg.norm(normal)
    length = cell[axis] @ normal
    volume = abs(np.linalg.det(cell))
    shape = tuple(np.asarray(c.charge_density).shape)
    if len(shape) != 3:
        raise ValueError("This script uses ISPIN=1 and a scalar charge grid")
    size = shape[axis]
    z = np.arange(size) / size
    # Branch centered at zero, discontinuity halfway along the open cell axis.
    ramp = ((z + 0.5) % 1.0 - 0.5) * length
    ramp[np.isclose(z, 0.5, rtol=0, atol=1e-12)] = 0.0
    k = 2 * np.pi * np.fft.fftfreq(size, d=length / size)
    rho = (C["counter_charge"] / volume *
           np.exp(-0.5 * (C["counter_charge_width"] * k)**2) *
           np.exp(-1j * k * (np.asarray(C["counter_charge_center"]) @ normal)))
    if size % 2 == 0:
        rho[size // 2] = 0.0  # No unpaired complex Nyquist mode.
    phi = np.zeros_like(rho)
    np.divide(-K * rho, k*k, out=phi, where=k != 0)
    rho_grid = np.fft.ifft(size * rho).real
    phi_grid = np.fft.ifft(size * phi).real
    pcc = volume * np.mean(rho_grid * ramp)
    self_energy = -0.5 * volume * np.vdot(rho, phi).real
    zion = ((np.asarray(c.positions)[:, axis] + 0.5) % 1 - 0.5) * length
    phase = np.exp(1j * zion[:, None] * k)
    phi_ion = (phase @ phi).real
    grad_ion = (phase @ (1j * k * phi)).real
    valence = np.asarray(c.ZVAL)[np.asarray(c.ion_types, dtype=int)]
    if not np.isclose(valence.sum(), C["neutral_nelect"], atol=1e-6, rtol=0):
        raise ValueError("Plugin ZVAL/POTCAR electron count mismatch")
    broadcast = [1, 1, 1]
    broadcast[axis] = size
    _geometry = (axis, normal, volume, shape, ramp, phi_grid, pcc,
                 self_energy, zion, phi_ion, grad_ion, valence, broadcast)
    return _geometry


def local_potential(constants, additions):
    global _last
    (axis, normal, volume, shape, ramp, phi_cc, pcc, self_energy,
     zion, phi_ion, grad_ion, valence, broadcast) = geometry(constants)
    density = np.asarray(constants.charge_density)
    nelect = float(density.mean())  # VASP stores number density times volume.
    planar = density.mean(axis=tuple(j for j in range(3) if j != axis))
    p = float(valence @ zion - np.mean(planar * ramp))
    total_dipole = p + pcc
    field = float(np.asarray(C["external_field"]) @ normal)
    slope = field - K * total_dipole / volume
    delta = phi_cc + slope * ramp
    additions.total_potential += delta.reshape(broadcast)
    # VASP already adds integral(n * delta). Complete the physical energy:
    # ion interaction + cc self-energy - E*pcc + dipole double-counting term.
    energy = (-valence @ (phi_ion + slope * zion) + self_energy - field * pcc
              - 0.5 * K / volume * total_dipole**2
              + K / volume * total_dipole * pcc)
    additions.total_energy += float(energy)
    force = (valence * (grad_ion + slope))[:, None] * normal
    total = (np.asarray(constants.hartree_potential) +
             np.asarray(constants.ion_potential) + delta.reshape(broadcast))
    _last = (total, force, p, nelect, float(energy))


def force_and_stress(constants, additions):
    if _last is None:
        raise RuntimeError("local_potential was not called")
    total, force, p, nelect, energy = _last
    if not np.isclose(nelect, C["nelect"], atol=1e-4, rtol=0):
        raise ValueError("Final plugin density has the wrong electron count")
    additions.forces += force
    # Save once, after SCF, instead of writing a large grid every SCF iteration.
    np.save("plugin_total_potential.npy", total)
    np.savez("plugin_observables.npz", electrode_dipole=p,
             nelect=nelect, added_energy=energy,
             expected_forces=np.asarray(constants.forces) + force)
'''


def controls(atoms):
    """Read physical charge/field controls, never sampled ML energies or biases."""
    info = atoms.info
    specified_axes = [j for j, a in enumerate("xyz")
                      if f"dipole_correction_{a}frac" in info]
    open_axes = np.flatnonzero(~atoms.pbc)
    slab = info.get("config_type", "slab" if len(open_axes) == 1 else "bulk") == "slab"
    if len(specified_axes) > 1 or (slab and len(open_axes) not in (0, 1)):
        raise ValueError("Specify one slab normal and one dipole-correction plane")
    axis = (specified_axes[0] if specified_axes else
            int(open_axes[0]) if len(open_axes) == 1 else 2) if slab else None
    if slab and len(open_axes) == 1 and axis != open_axes[0]:
        raise ValueError("Dipole-correction axis conflicts with the open cell direction")
    if np.linalg.det(atoms.cell.array) <= 0:
        raise ValueError("Use a nondegenerate, right-handed cell")
    qcc = float(info.get("counter_charge", 0.0))
    charge = float(info.get("total_charge", -qcc))
    if not np.isfinite([qcc, charge]).all() or not np.isclose(charge + qcc, 0, atol=1e-8):
        raise ValueError("Neutral composite required: total_charge = -counter_charge")
    if info.get("excess_electrons") is not None and not np.isclose(
            float(info["excess_electrons"]), -charge, atol=1e-8):
        raise ValueError("Conflicting excess_electrons and physical charge metadata")
    center = info.get("counter_charge_center")
    width = float(info.get("counter_charge_width", 1.0))
    if qcc and (center is None or not slab):
        raise ValueError("A Gaussian countercharge requires a slab and Cartesian center")
    center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(width) or width <= 0:
        raise ValueError("Invalid countercharge center or Gaussian sigma (Angstrom)")
    field = np.asarray(info.get("external_field", [0., 0., 0.]), dtype=float)
    if field.shape != (3,) or not np.isfinite(field).all():
        raise ValueError("external_field must be three finite components in V/Angstrom")
    meta = dict(schema="mace_volt_cdce_v1", slab_axis=axis, total_charge=charge,
                counter_charge=qcc, counter_charge_center=center.tolist(),
                counter_charge_width=width, external_field=field.tolist(),
                plugin=bool(qcc), config_type="slab" if slab else "bulk")
    if slab:
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        if np.linalg.norm(np.cross(atoms.cell[axis], normal)) > 1e-8:
            raise ValueError("The slab cell vector must be perpendicular to its periodic plane")
        if "external_field" not in info:
            field = -float(info.get("efield", 0.0)) * normal  # VASP sign convention.
        elif "efield" in info and not np.isclose(float(info["efield"]), -field @ normal):
            raise ValueError("Conflicting efield/external_field metadata")
        if np.linalg.norm(field - (field @ normal) * normal) > 1e-9:
            raise ValueError("Only a slab-normal homogeneous field is supported")
        cut = float(info.get(f"dipole_correction_{'xyz'[axis]}frac", 0.5))
        if not np.isfinite(cut) or not 0 <= cut < 1:
            raise ValueError("Invalid dipole-correction fraction")
        # Keep the same physical boundary while giving charged MACE moments a
        # common origin: the slab is centered at zero, its cut is at 0.5.
        origin = (cut + 0.5) % 1.0
        shift = origin * atoms.cell[axis]
        atoms.positions -= shift
        center = center - shift
        fraction = atoms.get_scaled_positions(wrap=False)[:, axis]
        atoms.positions -= np.floor(fraction + 0.5)[:, None] * atoms.cell[axis]
        length = atoms.cell[axis] @ normal
        center -= np.floor((center @ normal) / length + 0.5) * atoms.cell[axis]
        vacuum_key = f"vacuum_{'xyz'[axis]}frac"
        if vacuum_key in info:
            vacuum = float(info[vacuum_key])
            if not np.isfinite(vacuum) or not 0 <= vacuum < 1:
                raise ValueError("Invalid vacuum-plane fraction")
            meta[vacuum_key] = (vacuum - origin) % 1.0
        elif qcc:
            raise ValueError("Keep the vacuum_*frac used by NVTPhi for charged snapshots")
        meta.update(external_field=field.tolist(), counter_charge_center=center.tolist(),
                    coordinate_shift=shift.tolist())
        meta[f"dipole_correction_{'xyz'[axis]}frac"] = 0.5
        if qcc and length / 2 - abs(center @ normal) < 6 * width:
            raise ValueError("Countercharge overlaps the dipole cut; move the cut into empty vacuum")
    elif np.any(field) or float(info.get("efield", 0)):
        raise ValueError("A homogeneous field requires an explicit slab normal")
    return meta


def make_vasp_calculator(atoms, directory, meta, proto=False):
    reciprocal = np.linalg.inv(atoms.cell.array).T
    kpts = np.maximum(1, np.rint(np.linalg.norm(reciprocal, axis=1) / 0.025)).astype(int)
    settings = dict(xc="rpbe", encut=550, algo="all", time=0.05, kpts=kpts,
                    ibrion=-1, nsw=0, isif=2, ncore=32, ediff=1e-6, isym=0,
                    amix=0.02, bmix=0.0001, lmaxmix=4, lreal="auto", ismear=-1,
                    sigma=0.1, ivdw=11, prec="normal", istart=0, icharg=2,
                    ispin=1, nelm=300, lwave=False, lasph=True, lcharg=True,
                    laechg=True, lvtot=False, lvhar=True, lvacpotav=True,
                    directory=str(directory), command=COMMAND)
    axis = meta["slab_axis"]
    if axis is not None:
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        settings.update(ldipol=True, idipol=axis + 1, dipol=[0., 0., 0.],
                        efield=-float(np.asarray(meta["external_field"]) @ normal))
    if proto:
        settings.update(algo="none", icharg=12, nelm=1, ldiag=False, efield=0.)
    elif meta["plugin"]:
        # Native LDIPOL cannot see the external charge. Do not apply it twice.
        settings.update(ldipol=False, idipol=0, efield=0., lvacpotav=False,
                        custom={"plugins/local_potential": ".TRUE.",
                                "plugins/force_and_stress": ".TRUE.",
                                "lremove_drift": ".FALSE."})
        # VASP 6.6 fixes plugin energy accounting for ALGO=All. On older
        # plugin-enabled builds use Normal; all physical settings are retained.
        if tuple(meta["vasp_version"]) < (6, 6, 0):
            settings["algo"] = "normal"
    # LVACPOTAV is newer than 6.4.3; the collector samples LOCPOT directly.
    if tuple(meta.get("vasp_version", VASP_VERSION)) <= (6, 4, 3):
        settings.pop("lvacpotav")
    elif "lvacpotav" not in create_input.bool_keys:
        settings.setdefault("custom", {})["lvacpotav"] = (
            ".TRUE." if settings.pop("lvacpotav") else ".FALSE.")
    calc = Vasp(**settings)
    calc.initialize(atoms)
    neutral = float(calc.default_nelect_from_ppp())
    calc.set(nelect=neutral if proto else neutral - meta["total_charge"])
    return calc, neutral


def run_vasp(atoms, outdir, source_index):
    atoms = atoms.copy()
    atoms.calc = None
    atoms.set_constraint()
    meta = controls(atoms)
    if meta["plugin"] and VASP_VERSION < (6, 5, 0):
        raise RuntimeError("VASP 6.4.3 has no Python local-potential plugin. "
                           "Use a newer plugin-enabled build for Gaussian countercharges.")
    outdir.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite completed calculations or reuse stale grids.
    if (outdir / "OUTCAR").exists() or (outdir / "proto" / "OUTCAR").exists():
        raise FileExistsError(f"Use a fresh output directory: {outdir}")
    meta["source_index"] = int(source_index)
    atoms.pbc = True
    atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))

    proto_dir = outdir / "proto"
    proto_dir.mkdir(exist_ok=True)
    proto, neutral = make_vasp_calculator(atoms, proto_dir, meta, proto=True)
    proto.write_input(atoms)
    # ALGO=None can lack energy/force records expected by ASE.calculate().
    # Run to normal completion and read its grids, without requesting energies.
    with (proto_dir / "vasp.out").open("w") as log:
        code, error = proto._run(command=COMMAND, out=log, directory=str(proto_dir))
    if code:
        raise RuntimeError(f"Protoatomic VASP failed ({code}): {error}")
    for src, dst, reader in (("LOCPOT", "LOCPOT_proto", Locpot),
                             ("AECCAR1", "AECCAR1_proto", Chgcar)):
        grid = reader.from_file(str(proto_dir / src))
        if not np.isfinite(grid.data["total"]).all():
            raise ValueError(f"Invalid protoatomic grid: {src}")
        if src == "AECCAR1" and abs(np.mean(grid.data["total"]) - neutral) > 1e-3:
            raise ValueError("Protoatomic density is not neutral")
        shutil.copy2(proto_dir / src, outdir / dst)
    with (proto_dir / "OUTCAR").open(errors="replace") as stream:
        version = re.search(r"vasp\.(\d+)\.(\d+)\.(\d+)", stream.read(8192), re.I)
    if not version:
        raise ValueError("Cannot identify VASP version from proto/OUTCAR")
    meta.update(vasp_version=list(map(int, version.groups())), neutral_nelect=neutral,
                nelect=neutral - meta["total_charge"])
    if meta["plugin"] and tuple(meta["vasp_version"]) < (6, 5, 0):
        raise ValueError("Charged calculations require VASP's Python plugin interface")
    (outdir / "dft_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    if meta["plugin"]:
        (outdir / "vasp_plugin.py").write_text(PLUGIN, encoding="utf-8")

    calc, checked_neutral = make_vasp_calculator(atoms, outdir, meta)
    if not np.isclose(neutral, checked_neutral):
        raise ValueError("SCF/proto POTCAR mismatch")
    atoms.calc = calc
    atoms.get_potential_energy()
    if not calc.converged:
        raise RuntimeError(f"SCF did not converge in {outdir}; do not train on it")
    if meta["plugin"]:
        if calc._xml_calc.parameters.get("lremove_drift") is not False:
            raise ValueError("VASP did not report LREMOVE_DRIFT=False; a build supporting "
                             "this flag is required for physical forces in a fixed external field")
        with np.load(outdir / "plugin_observables.npz") as data:
            if not np.allclose(atoms.get_forces(), data["expected_forces"][calc.resort],
                               rtol=0, atol=2e-5):
                raise ValueError("VASP altered plugin forces; check LREMOVE_DRIFT support")
        raw = Locpot.from_file(str(outdir / "LOCPOT"))
        total = np.load(outdir / "plugin_total_potential.npy")
        if total.shape != raw.data["total"].shape:
            raise ValueError("Plugin/LOCPOT grid mismatch")
        # Explicit H + ionic + plugin potential: never guess if native LOCPOT
        # already contains the plugin. Keep native LOCPOT untouched for audit.
        Locpot(raw.poscar, {"total": total}).write_file(str(outdir / "LOCPOT_total"))
    else:
        shutil.copy2(outdir / "LOCPOT", outdir / "LOCPOT_total")
    atoms.info = {key: value for key, value in meta.items() if value is not None}
    atoms.info["fermi_level"] = float(calc.get_fermi_level())
    write(outdir / f"{SYSTEM}_vasp_sp.traj", atoms)
    calc.write_json(str(outdir / "calc_state.json"))
    for filename, reader, target in (("LOCPOT_total", Locpot, "esp.pkl"),
                                     ("CHGCAR", Chgcar, "aed.pkl")):
        grid = reader.from_file(str(outdir / filename))
        axis = meta["slab_axis"] if meta["slab_axis"] is not None else 2
        with (outdir / target).open("wb") as stream:
            pickle.dump((grid.get_axis_grid(axis), grid.get_average_along_axis(axis)), stream)
    meta["complete"] = True
    (outdir / "dft_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    job = sys.argv[1]
    batch = BASE_PATH / job
    images = read(str(INPUT or batch / f"{SYSTEM}_{job[5:]}_vasp_init.traj"), index=":")
    selected = list(range(len(images))) if INDEXES is None else list(INDEXES)
    if len(set(selected)) != len(selected) or any(i < 0 or i >= len(images) for i in selected):
        raise ValueError("INDEXES must contain unique, valid frame indexes")
    if VASP_VERSION < (6, 5, 0) and any(float(images[i].info.get("counter_charge", 0)) for i in selected):
        raise SystemExit("No jobs launched: standard VASP 6.4.3 cannot apply Gaussian "
                         "countercharges through PLUGINS/LOCAL_POTENTIAL. Upgrade the "
                         "VASP executable and update VASP_VERSION first.")
    started, failed = time.time(), []
    for output_index, source_index in enumerate(selected):
        try:
            run_vasp(images[source_index], batch / f"output_{output_index}", source_index)
        except Exception:
            traceback.print_exc()
            failed.append(source_index)
    print(f"Calculation takes {(time.time() - started) / 3600:.3f} h", flush=True)
    if failed:
        raise SystemExit(f"Failed configuration indexes: {failed}")
