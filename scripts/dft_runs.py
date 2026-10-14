"""One SCF run per configuration: python dft_runs.py vasp_0_4.

The Python local-potential plugin captures LOCPOT_proto from the initial
atomic-density Hamiltonian, before adding the prescribed external potential.
Requires a plugin-enabled VASP executable. No separate proto calculation.
"""
from pathlib import Path
import json
import pickle
import shutil
import sys
import time
import traceback

import numpy as np
from ase.calculators.vasp import Vasp
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Locpot


BASE_PATH = Path("/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft/")
SYSTEM = "IrO2OH_110"
INPUT = None  # Optional XYZ/trajectory; otherwise use the original batch file.
INDEXES = None  # Optional frame indexes; None selects all frames.
COMMAND = "mpirun vasp_std"

# Installed automatically in each output directory; requires only NumPy.
PLUGIN = r'''
import json
from pathlib import Path
import numpy as np

C = json.loads(Path("dft_metadata.json").read_text())
K = 1.0 / 0.005526349406  # eV Angstrom / e^2.
_geometry = None
_last = None
_first = True


def write_grid(filename, constants, values):
    # ASE writes a standard POSCAR without selective dynamics for these runs.
    header = Path("POSCAR").read_text().splitlines()[:8 + constants.number_ions]
    flat = values.ravel(order="F")
    end = flat.size // 5 * 5
    with open(filename, "w") as stream:
        stream.write("\n".join(header) + "\n\n" + " ".join(map(str, values.shape)) + "\n")
        np.savetxt(stream, flat[:end].reshape(-1, 5), fmt="%.11E")
        if end < flat.size:
            np.savetxt(stream, flat[end:][None, :], fmt="%.11E")


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
    size = np.asarray(c.charge_density).shape[axis]
    z = np.arange(size) / size
    ramp = ((z + 0.5) % 1.0 - 0.5) * length
    ramp[np.isclose(z, 0.5, rtol=0, atol=1e-12)] = 0.0
    k = 2 * np.pi * np.fft.fftfreq(size, d=length / size)
    rho = (C["counter_charge"] / volume *
           np.exp(-0.5 * (C["counter_charge_width"] * k)**2) *
           np.exp(-1j * k * (np.asarray(C["counter_charge_center"]) @ normal)))
    if size % 2 == 0:
        rho[size // 2] = 0.0
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
    broadcast = [1, 1, 1]
    broadcast[axis] = size
    _geometry = (axis, normal, volume, ramp, phi_grid, pcc, self_energy,
                 zion, phi_ion, grad_ion, valence, broadcast)
    return _geometry


def local_potential(constants, additions):
    global _first, _last
    bare = (np.asarray(constants.hartree_potential) +
            np.asarray(constants.ion_potential))
    if _first:
        # ICHARG=2 and NELMDL=-5 retain the atomic starting Hamiltonian.
        # Capture its bare electrostatic potential before any plugin addition.
        write_grid("LOCPOT_proto", constants, bare)
        _first = False
    if not C["plugin"]:
        return
    (axis, normal, volume, ramp, phi_cc, pcc, self_energy,
     zion, phi_ion, grad_ion, valence, broadcast) = geometry(constants)
    density = np.asarray(constants.charge_density)
    planar = density.mean(axis=tuple(j for j in range(3) if j != axis))
    p = float(valence @ zion - np.mean(planar * ramp))
    total_dipole = p + pcc
    field = float(np.asarray(C["external_field"]) @ normal)
    slope = field - K * total_dipole / volume
    delta = phi_cc + slope * ramp
    additions.total_potential += delta.reshape(broadcast)
    # Complete VASP's electron contribution with the ionic, countercharge
    # self-energy and density-dependent dipole double-counting terms.
    energy = (-valence @ (phi_ion + slope * zion) + self_energy - field * pcc
              - 0.5 * K / volume * total_dipole**2
              + K / volume * total_dipole * pcc)
    additions.total_energy += float(energy)
    force = (valence * (grad_ion + slope))[:, None] * normal
    _last = (bare + delta.reshape(broadcast), force, p,
             float(density.mean()), float(energy))


def force_and_stress(constants, additions):
    total, force, p, nelect, energy = _last
    additions.forces += force
    write_grid("LOCPOT_total", constants, total)
    np.savez("plugin_observables.npz", electrode_dipole=p,
             nelect=nelect, added_energy=energy,
             expected_forces=np.asarray(constants.forces) + force)
'''


def configuration(atoms):
    """Preserve the snapshot's charge, field and physical reference planes."""
    info = atoms.info
    open_axes = np.flatnonzero(~atoms.pbc)
    slab = info.get("config_type", "slab" if len(open_axes) == 1 else "bulk") == "slab"
    axis = next((j for j, a in enumerate("xyz")
                 if f"dipole_correction_{a}frac" in info),
                int(open_axes[0]) if len(open_axes) == 1 else 2) if slab else None
    qcc = float(info.get("counter_charge", 0.0))
    charge = float(info.get("total_charge", -qcc))
    if not np.isclose(charge + qcc, 0, atol=1e-8):
        raise ValueError("total_charge must equal -counter_charge")
    center = np.asarray(info["counter_charge_center"] if qcc else
                        info.get("counter_charge_center", [0., 0., 0.]), dtype=float)
    field = np.asarray(info.get("external_field", [0., 0., 0.]), dtype=float)
    meta = dict(schema="mace_volt_cdce_v1", slab_axis=axis, total_charge=charge,
                counter_charge=qcc, counter_charge_center=center.tolist(),
                counter_charge_width=float(info.get("counter_charge_width", 1.0)),
                external_field=field.tolist(), plugin=bool(qcc),
                config_type="slab" if slab else "bulk")
    if slab:
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        field = np.asarray(info.get("external_field", -float(info.get("efield", 0.)) * normal))
        cut = float(info.get(f"dipole_correction_{'xyz'[axis]}frac", 0.5))
        origin = (cut + 0.5) % 1.0
        shift = origin * atoms.cell[axis]
        atoms.positions -= shift
        fraction = atoms.get_scaled_positions(wrap=False)[:, axis]
        atoms.positions -= np.floor(fraction + 0.5)[:, None] * atoms.cell[axis]
        center = center - shift
        length = atoms.cell[axis] @ normal
        center -= np.floor((center @ normal) / length + 0.5) * atoms.cell[axis]
        meta.update(external_field=field.tolist(), counter_charge_center=center.tolist())
        meta[f"dipole_correction_{'xyz'[axis]}frac"] = 0.5
        key = f"vacuum_{'xyz'[axis]}frac"
        if key in info:
            meta[key] = (float(info[key]) - origin) % 1.0
    return meta


def make_vasp_calculator(atoms, directory, meta):
    reciprocal = np.linalg.inv(atoms.cell.array).T
    kpts = np.maximum(1, np.rint(np.linalg.norm(reciprocal, axis=1) / 0.025)).astype(int)
    settings = dict(xc="rpbe", encut=550, algo="all", time=0.05, kpts=kpts,
                    ibrion=-1, nsw=0, isif=2, ncore=32, ediff=1e-6, isym=0,
                    amix=0.02, bmix=0.0001, lmaxmix=4, lreal="auto", ismear=-1,
                    sigma=0.1, ivdw=11, prec="normal", istart=0, icharg=2,
                    nelmdl=-5, ispin=1, nelm=300, lwave=False, lasph=True,
                    lcharg=True, laechg=True, lvtot=False, lvhar=True,
                    directory=str(directory), command=COMMAND,
                    custom={"plugins/local_potential": ".TRUE."})
    axis = meta["slab_axis"]
    if axis is not None:
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        settings.update(ldipol=True, idipol=axis + 1, dipol=[0., 0., 0.],
                        efield=-float(np.asarray(meta["external_field"]) @ normal))
        settings["custom"]["lvacpotav"] = ".TRUE."
    if meta["plugin"]:
        settings.update(ldipol=False, idipol=0, efield=0.)
        settings["custom"].update({"plugins/force_and_stress": ".TRUE.",
                                   "lremove_drift": ".FALSE.", "lvacpotav": ".FALSE."})
    calc = Vasp(**settings)
    calc.initialize(atoms)
    neutral = float(calc.default_nelect_from_ppp())
    calc.set(nelect=neutral - meta["total_charge"])
    meta.update(neutral_nelect=neutral, nelect=neutral - meta["total_charge"])
    return calc


def run_vasp(atoms, outdir, source_index):
    atoms = atoms.copy()
    atoms.calc = None
    atoms.set_constraint()
    meta = configuration(atoms)
    meta.update(source_index=int(source_index), complete=False)
    atoms.pbc = True
    atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
    outdir.mkdir(parents=True, exist_ok=True)
    atoms.calc = make_vasp_calculator(atoms, outdir, meta)
    (outdir / "dft_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (outdir / "vasp_plugin.py").write_text(PLUGIN, encoding="utf-8")
    atoms.get_potential_energy()
    if not atoms.calc.converged:
        raise RuntimeError(f"SCF did not converge in {outdir}")

    # AECCAR1 is already produced during this SCF run; retain the collector's name.
    shutil.copyfile(outdir / "AECCAR1", outdir / "AECCAR1_proto")
    if not meta["plugin"]:
        shutil.copyfile(outdir / "LOCPOT", outdir / "LOCPOT_total")
    meta["complete"] = True
    atoms.info = {key: value for key, value in meta.items() if value is not None}
    atoms.info["fermi_level"] = float(atoms.calc.get_fermi_level())
    write(outdir / f"{SYSTEM}_vasp_sp.traj", atoms)
    atoms.calc.write_json(str(outdir / "calc_state.json"))
    axis = meta["slab_axis"] if meta["slab_axis"] is not None else 2
    for grid, filename in ((Locpot.from_file(str(outdir / "LOCPOT_total")), "esp.pkl"),
                            (Chgcar.from_file(str(outdir / "CHGCAR")), "aed.pkl")):
        with (outdir / filename).open("wb") as stream:
            pickle.dump((grid.get_axis_grid(axis), grid.get_average_along_axis(axis)), stream)
    (outdir / "dft_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    job = sys.argv[1]
    batch = BASE_PATH / job
    images = read(str(INPUT or batch / f"{SYSTEM}_{job[5:]}_vasp_init.traj"), index=":")
    selected = range(len(images)) if INDEXES is None else INDEXES
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
