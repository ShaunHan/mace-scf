"""One SCF run per configuration: python dft_runs.py vasp_0_4.

The Python local-potential plugin captures LOCPOT_proto from the initial
atomic-density Hamiltonian, before adding the prescribed external potential.
Requires a plugin-enabled VASP executable. No separate proto calculation.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import pickle
import re
import sys
import time
import traceback

import numpy as np
import h5py
from ase.calculators.vasp import Vasp
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Locpot, Outcar


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

C = globals().get("C") or json.loads(Path("dft_metadata.json").read_text())
K = 1.0 / 0.005526349406  # eV Angstrom / e^2.
_geometry = None
_last = None
_first = True
_calls = 0


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
           np.exp(-1j * k * length * C[f"counter_charge_{'xyz'[axis]}frac"]))
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
    global _first, _last, _calls
    if _first:
        # ICHARG=2 starts from the superposition of atomic densities.
        # Capture its bare electrostatic potential before any plugin addition.
        write_grid("LOCPOT_proto", constants, np.asarray(constants.hartree_potential)
                   + np.asarray(constants.ion_potential))
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
    _last = (p, delta.copy())
    mode = 'w' if _calls == 0 else 'a'
    with open('plugin_scf.csv', mode) as stream:
        if _calls == 0:
            stream.write('callback,electrode_dipole_eA,electron_count,correction_energy_eV\n')
        stream.write(f'{_calls},{p:.12g},{density.mean():.12g},{energy:.12g}\n')
    _calls += 1


def force_and_stress(constants, additions):
    # The final force callback supplies the final density. The last potential
    # callback can be a trial state under a direct electronic optimizer.
    previous_p, previous_delta = _last
    (axis, normal, volume, ramp, phi_cc, pcc, self_energy,
     zion, phi_ion, grad_ion, valence, broadcast) = geometry(constants)
    density = np.asarray(constants.charge_density)
    planar = density.mean(axis=tuple(j for j in range(3) if j != axis))
    p = float(valence @ zion - np.mean(planar * ramp))
    slope = float(np.asarray(C['external_field']) @ normal)-K*(p+pcc)/volume
    delta = phi_cc+slope*ramp
    force = (valence*(grad_ion+slope))[:, None]*normal
    additions.forces += force
    np.savez("plugin_observables.npz", electrode_dipole=p,
             callback_dipole=previous_p, nelect=float(density.mean()),
             plugin_profile=delta, callback_plugin_profile=previous_delta,
             expected_forces=np.asarray(constants.forces) + force)
'''


def configuration(atoms):
    """Preserve the snapshot's charge, field and physical reference planes."""
    info = atoms.info
    if info.get("dipole_correction_center") is not None:
        raise NotImplementedError("This DFT runner is for slabs/bulk. Molecular reference data require native IDIPOL=4, LDIPOL=True and DIPOL in fractional coordinates.")
    open_axes = np.flatnonzero(~atoms.pbc)
    slab = info.get("config_type", "slab" if len(open_axes) == 1 else "bulk") == "slab"
    axis = next((j for j, a in enumerate("xyz")
                 if f"dipole_correction_{a}frac" in info),
                int(open_axes[0]) if len(open_axes) == 1 else 2) if slab else None
    qcc = float(info.get("counter_charge", 0.0))
    planes = [(j, f"counter_charge_{a}frac") for j, a in enumerate("xyz")
              if info.get(f"counter_charge_{a}frac") is not None]
    if len(planes) > 1 or (planes and info.get("counter_charge_center") is not None):
        raise ValueError("Choose one sheet fraction or a 3D counter_charge_center")
    if qcc and not planes:
        raise NotImplementedError("This VASP plugin implements Gaussian sheets. Use counter_charge_xfrac/yfrac/zfrac; a 3D counter_charge_center needs a 3D plugin.")
    if planes and (not slab or planes[0][0] != axis):
        raise ValueError("The counter-charge sheet must be parallel to the slab")
    charge = float(info.get("total_charge", -qcc))
    if not np.isclose(charge + qcc, 0, atol=1e-6):  # NELECT is commonly written to six decimals.
        raise ValueError("total_charge must equal -counter_charge")
    center = np.asarray(info.get("counter_charge_center", [0., 0., 0.]), dtype=float)
    field = np.asarray(info.get("external_field", [0., 0., 0.]), dtype=float)
    if not slab and np.linalg.norm(field) > 1e-10:
        raise ValueError('A homogeneous external field needs an open slab direction')
    meta = dict(schema="mace_volt_cdce_v2", slab_axis=axis, total_charge=charge,
                counter_charge=qcc, counter_charge_center=center.tolist(),
                counter_charge_width=float(info.get("counter_charge_width", 1.0)),
                external_field=field.tolist(), plugin=bool(qcc),
                config_type="slab" if slab else "bulk")
    if planes:
        key = planes[0][1]
        sheet_fraction = float(info[key])
        if not np.isfinite(sheet_fraction) or not 0. <= sheet_fraction < 1.:
            raise ValueError("Counter-charge sheet fractions must be in [0, 1)")
        meta.pop("counter_charge_center")
        meta[key] = sheet_fraction
    elif info.get("counter_charge_center") is None:
        meta.pop("counter_charge_center")
    if slab:
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        field = np.asarray(info.get("external_field", -float(info.get("efield", 0.)) * normal))
        if np.linalg.norm(field-(field @ normal)*normal) > 1e-10:
            raise ValueError('The homogeneous field must be normal to the slab')
        cut = float(info.get(f"dipole_correction_{'xyz'[axis]}frac", 0.5))
        origin = (cut + 0.5) % 1.0
        shift = origin * atoms.cell[axis]
        atoms.positions -= shift
        fraction = atoms.get_scaled_positions(wrap=False)[:, axis]
        atoms.positions -= np.floor(fraction + 0.5)[:, None] * atoms.cell[axis]
        center = center - shift
        length = atoms.cell[axis] @ normal
        center -= np.floor((center @ normal) / length + 0.5) * atoms.cell[axis]
        meta.update(external_field=field.tolist())
        if planes:
            meta[planes[0][1]] = (sheet_fraction-origin) % 1.
        elif "counter_charge_center" in meta:
            meta["counter_charge_center"] = center.tolist()
        meta[f"dipole_correction_{'xyz'[axis]}frac"] = 0.5
        key = f"vacuum_{'xyz'[axis]}frac"
        if key in info:
            meta[key] = (float(info[key]) - origin) % 1.0
    return meta


def make_vasp_calculator(atoms, directory, meta):
    reciprocal = np.linalg.inv(atoms.cell.array).T
    kpts = np.maximum(1, np.rint(np.linalg.norm(reciprocal, axis=1) / 0.025)).astype(int)
    # LORBIT adds projections of the converged orbitals. Set LWAVE=True only
    # when later band-resolved PARCHG analysis is needed (larger disk output).
    settings = dict(xc="rpbe", encut=550, algo="normal", kpts=kpts,
                    ibrion=-1, nsw=0, isif=2, ncore=32, ediff=1e-6, isym=0,
                    amix=0.02, bmix=0.0001, lmaxmix=4, lreal="auto", ismear=-1,
                    sigma=0.1, ivdw=11, prec="normal", istart=0, icharg=2,
                    nelmdl=-5, ispin=1, nelm=300, lwave=False, lasph=True, lorbit=11,
                    lcharg=True, laechg=True, lvtot=False, lvhar=True,
                    directory=str(directory), command=COMMAND,
                    custom={"plugins/local_potential": ".TRUE.",
                            "wrt_potential": "hartree ionic"})
    axis = meta["slab_axis"]
    if axis is not None:
        normal = np.linalg.inv(atoms.cell.array)[:, axis]
        normal /= np.linalg.norm(normal)
        settings.update(ldipol=True, idipol=axis + 1, dipol=[0., 0., 0.],
                        efield=-float(np.asarray(meta["external_field"]) @ normal))
        settings["custom"]["lvacpotav"] = ".TRUE."
    if meta["plugin"]:
        # Kerker damping suppresses long-wavelength charge sloshing without
        # making every local density update as small as straight AMIX=0.02.
        # AMIN must be explicit; its default would undo this damping.
        settings.update(ldipol=False, idipol=0, efield=0.,
                        amix=0.1, bmix=1.0, amin=0.001, nelmdl=-12, lwave=True)
        settings["custom"].update({"plugins/force_and_stress": ".TRUE.",
                                   "lvacpotav": ".FALSE."})
    calc = Vasp(**settings)
    calc.initialize(atoms)
    neutral = float(calc.default_nelect_from_ppp())
    calc.set(nelect=neutral - meta["total_charge"])
    meta.update(neutral_nelect=neutral, nelect=neutral - meta["total_charge"])
    return calc


def diagnose(atoms, outdir, meta):
    """Reuse the saved grids; no extra electronic calculation or wavefunctions."""
    axis = meta['slab_axis'] if meta['slab_axis'] is not None else 2
    native = Locpot.from_file(str(outdir/'LOCPOT'))
    density = Chgcar.from_file(str(outdir/'CHGCAR'))
    phi = np.asarray(native.get_average_along_axis(axis))
    n = np.asarray(density.get_average_along_axis(axis))/atoms.get_volume()
    outcar_text = (outdir/'OUTCAR').read_text(errors='replace')
    values = re.findall(r'Fermi energy:\s*([-+\d.eE]+)', outcar_text)
    if not values:
        values = re.findall(r'E-fermi\s*:\s*([-+\d.eE]+)', outcar_text)
    fermi = float(values[-1])
    converged = 'aborting loop because EDIFF is reached' in outcar_text
    report = dict(vasp_converged=converged,
                  fermi_level_eV=fermi,
                  dos_fermi_level_eV=float(atoms.calc.get_fermi_level()))
    meta['fermi_level'] = fermi
    meta['potential_file'] = 'LOCPOT'
    meta['electronic_converged'] = converged
    if meta['plugin']:
        with np.load(outdir/'plugin_observables.npz') as obs:
            # The saved native LOCPOT includes the final plugin contribution.
            # Verify this against independently stored Hartree + ionic grids.
            with h5py.File(outdir/'vaspout.h5') as handle:
                pot = handle['results/potential']
                bare_grid = (pot['hartree'][0]+pot['ionic'][0]).transpose(2,1,0)
                bare = bare_grid.mean(axis=tuple(j for j in range(3) if j != axis))
            # Reconstruct from final CHGCAR as an independent check; this also
            # works for older outputs whose callback saved a trial density.
            outcar = Outcar(str(outdir/'OUTCAR'))
            outcar.read_pseudo_zval()
            plugin_meta = dict(meta)
            if meta['schema'] == 'mace_volt_cdce_v1':
                plugin_meta[f"counter_charge_{'xyz'[axis]}frac"] = float(
                    (np.asarray(meta['counter_charge_center']) @ np.linalg.inv(atoms.cell.array))[axis] % 1.)
            namespace = {'C': plugin_meta}
            exec(PLUGIN, namespace)
            ions = read(outdir/'POSCAR')  # Full precision; CHGCAR headers can round coordinates.
            c = SimpleNamespace(lattice_vectors=ions.cell.array, positions=ions.get_scaled_positions(),
                charge_density=density.data['total'], ion_types=np.arange(len(atoms)),
                ZVAL=np.array([outcar.zval_dict[s] for s in ions.symbols]))
            _, normal, volume, ramp, cc, pcc, _, zion, _, _, valence, _ = namespace['geometry'](c)
            final_p = float(valence @ zion-np.mean(n*volume*ramp))
            delta = cc+(np.asarray(meta['external_field']) @ normal-namespace['K']*(final_p+pcc)/volume)*ramp
            def rms(x): return float(np.sqrt(np.mean((x-x.mean())**2)))
            total_error = rms(phi-bare-delta)
            report.update(native_vs_final_total_RMS_V=total_error,
                          callback_to_final_dipole_eA=final_p-float(obs.get('callback_dipole', obs['electrode_dipole'])))
            # A fixed external sheet can exert a net force on the atoms.
            # Record the native drift subtraction. ASE has already unsorted atoms.
            expected = obs['expected_forces'][getattr(atoms.calc, 'resort', np.arange(len(atoms)))]
            difference = atoms.get_forces(apply_constraint=False)-expected
            report['removed_force_drift_eV_A'] = difference.mean(0).tolist()
            report['nonuniform_force_difference_max_eV_A'] = float(np.max(abs(difference-difference.mean(0))))
            meta['forces_file'] = 'plugin_observables.npz'
    report['max_force_eV_A'] = float(np.linalg.norm(atoms.get_forces(apply_constraint=False),axis=1).max())
    # Keep OSZICAR residuals alongside the density diagnostics. A tiny dE alone
    # does not establish a stationary electronic state (especially ALGO=All).
    steps = []
    for line in (outdir/'OSZICAR').read_text().splitlines():
        match = re.match(r'\s*(DAV:|RMM:|CGA:|SDA:|CG\s*:|trl)\s*(\d+)\s+(.*)',line)
        if match:
            numbers = match[3].split()
            if len(numbers) >= 5:
                steps.append([int(match[2]), *[float(numbers[i]) for i in (0,1,2,4)],
                              float(numbers[5]) if len(numbers) > 5 else np.nan])
    if steps:
        np.savetxt(outdir/'scf_residuals.csv',steps,delimiter=',',
                   header='step,energy_eV,dE_eV,d_eps_eV,rms,rms_c',comments='')
        report['last_electronic_step'] = dict(zip(('step','energy_eV','dE_eV','d_eps_eV','rms'),steps[-1][:5]))
        charge_residuals = [row[-1] for row in steps if np.isfinite(row[-1])]
        report['last_density_residual_rms_c'] = charge_residuals[-1] if charge_residuals else None
    gamma = re.findall(r'average eigenvalue GAMMA=\s*([-+\d.eE]+)', outcar_text)
    report['last_mixing_mean_eigenvalue'] = float(gamma[-1]) if gamma else None
    length = 1/np.linalg.norm(np.linalg.inv(atoms.cell.array)[:,axis])
    z = np.arange(len(phi))*length/len(phi)
    az = atoms.get_scaled_positions()[:,axis]*length
    distance = abs((z[:,None]-az[None,:]+length/2)%length-length/2).min(1)
    vacuum = distance > 3.
    report.update(vacuum_definition='planes more than 3 Å from every atomic plane',
                  vacuum_electrons=float(n[vacuum].sum()*atoms.get_volume()/len(n)),
                  vacuum_density_min_e_A3=float(n[vacuum].min()) if vacuum.any() else None,
                  vacuum_density_max_e_A3=float(n[vacuum].max()) if vacuum.any() else None,
                  vacuum_min_phi_minus_fermi_V=float((phi-fermi)[vacuum].min()) if vacuum.any() else None)
    np.savetxt(outdir/'electrostatics_profile.csv',np.column_stack((z,phi,phi-fermi,n,distance)),
               delimiter=',',header='z_A,phi_V,phi_minus_fermi_V,electron_density_e_A3,distance_to_atomic_plane_A',comments='')
    (outdir/'electrostatics_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    for profile, filename in ((phi,'esp.pkl'),(n*atoms.get_volume(),'aed.pkl')):
        with (outdir/filename).open('wb') as stream: pickle.dump((z,profile),stream)


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
    diagnose(atoms, outdir, meta)
    print(f"{outdir.name}: electronic convergence = {meta['electronic_converged']}", flush=True)
    meta['complete'] = True
    atoms.info = {key: value for key, value in meta.items() if value is not None}
    write(outdir / f"{SYSTEM}_vasp_sp.traj", atoms)
    atoms.calc.write_json(str(outdir / "calc_state.json"))
    (outdir / "dft_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    # Existing output directory: diagnostics only, with no VASP launch or
    # changes to INCAR, metadata, grids or electronic results.
    existing = Path(sys.argv[1])
    if (existing/'OUTCAR').is_file():
        atoms = read(existing/'vasprun.xml', index=-1)
        diagnose(atoms, existing, json.loads((existing/'dft_metadata.json').read_text()))
        print('Diagnostics written to', existing)
        raise SystemExit(0)
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
