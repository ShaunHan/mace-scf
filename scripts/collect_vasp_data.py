#!/usr/bin/env python3
"""Collect matched VASP SCF/proto data for MACE-VOLT, without project helpers.

Dependencies: NumPy, SciPy, h5py, ASE and pymatgen. This script never runs VASP.
It retains the current field, gauge, Fourier, label and grouped-split conventions.
Use --preflight for read-only validation; add --poisson-check for a source audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re

import h5py
import numpy as np
from ase.geometry import Cell
from ase.io import read, write
from pymatgen.io.vasp import Chgcar, Incar, Locpot, Outcar
from scipy.optimize import linear_sum_assignment

BASE_PATH = Path("/gpfs/projects/qm_inorganics/electrocatalysis/IrO2_110/dft/")
PROTO_BASE_PATH = BASE_PATH.with_name("dft_proto")
SYSTEM_NAME = "IrO2OH_110"
CHUNK_SIZE, N_PROCS, REUSE_HDF = 5, 64, False
K_CUTOFF = 18.0
DENSITY_SIGMAS, POTENTIAL_SIGMAS = [1.5], [1.5]
FOURIER_CUTOFF_SAFETY_FACTOR = 1.25
FMAX_TOL = 6.0
VALID_FRACTION, SPLIT_SEED = 0.2, 7777
UNCONVERGED_IDS = []
POTENTIAL_REFERENCE = "applied_zero_mean"
PAIR_SCHEMA = "macevolt_vasp_pair"
RHO_HDF_NAME = "deformation_charge_density.hdf"
PHI_HDF_NAME = "deformation_hartree_potential.hdf"
PROTO_PHI_HDF_NAME = "proto_hartree_potential.hdf"
_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


# Essential raw-pair checks live here rather than in a sibling module.
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_incar(path):
    """Use the same installed INCAR parser as the volumetric collection code."""
    return dict(Incar.from_file(str(path)))


def _normal(value):
    if isinstance(value, str): return value.strip().lower()
    if isinstance(value, list): return [_normal(v) for v in value]
    return value


def validate_pair_settings(scf: dict, proto: dict, config_type: str, open_axis: int = 2):
    """Validate shared physical/numerical definitions, not identical SCF algorithms."""
    s = {key.upper(): value for key, value in scf.items()}
    p = {key.upper(): value for key, value in proto.items()}
    if s.get('LVHAR') is not True or p.get('LVHAR') is not True:
        raise ValueError('Both runs must explicitly request LVHAR=True (ionic plus Hartree, without XC)')
    if int(s.get('NSW',0)) > 1 or int(p.get('NSW',0)) > 1:
        raise ValueError('This paired collector expects a single-point run, not a multi-ionic-step trajectory')
    if int(s.get('ICHARG', 2)) >= 10 or str(s.get('ALGO','normal')).lower() in ('none','nothing','eigenval'):
        raise ValueError('The SCF member is frozen-density/postprocessing, not a ground-state SCF')
    if int(p.get('ICHARG', 2)) != 12 or int(p.get('NELM',60)) != 1:
        raise ValueError('This proto reference requires ICHARG=12 and NELM=1')
    if abs(float(p.get('EFIELD',0.))) > 1.e-12:
        raise ValueError('The proto reference must be field-free')
    # Some intentionally differ: ICHARG/ALGO/NELM/LDIAG, applied field and the
    # output-only LVACPOTAV. LVHAR=True takes precedence over old LVTOT=True.
    defaults = {'ISPIN':1, 'ISMEAR':1, 'SIGMA':.2, 'PREC':'normal', 'LREAL':False,
                'LASPH':False, 'LDAU':False, 'LHFCALC':False, 'LSORBIT':False,
                'LNONCOLLINEAR':False, 'IVDW':0, 'LMAXMIX':2, 'ADDGRID':False}
    names = (*defaults, 'ENCUT','GGA','METAGGA','AEXX','HFSCREEN','LEXCH',
             'LDAUTYPE','LDAUL','LDAUU','LDAUJ','NGX','NGY','NGZ','NGXF','NGYF','NGZF','KSPACING','KGAMMA','ISYM')
    for key in names:
        a=s.get(key,defaults.get(key)); b=p.get(key,defaults.get(key))
        if _normal(a) != _normal(b):
            raise ValueError(f'Paired reference settings differ for {key}: SCF={a!r}, proto={b!r}')
    if 'ENCUT' not in s or 'ENCUT' not in p:
        raise ValueError('Both runs must record explicit matching ENCUT')
    if config_type not in ('slab','bulk') or open_axis not in (0,1,2):
        raise ValueError('Only documented bulk/slab geometry and open_axis 0/1/2 are supported')
    if config_type == 'slab':
        for tag, settings in (('SCF',s),('proto',p)):
            if settings.get('LDIPOL') is not True or int(settings.get('IDIPOL',0)) != open_axis+1:
                raise ValueError(f'{tag} must use LDIPOL=True, IDIPOL={open_axis+1}')
            dip=settings.get('DIPOL')
            if not isinstance(dip,list) or len(dip)!=3 or not all(math.isfinite(float(x)) for x in dip):
                raise ValueError(f'{tag} must specify a finite 3-component DIPOL')
        if any(abs(float(a)-float(b))>1.e-12 for a,b in zip(s['DIPOL'],p['DIPOL'])):
            raise ValueError('SCF/proto DIPOL positions differ')
    elif bool(s.get('LDIPOL',False)) or bool(p.get('LDIPOL',False)):
        raise ValueError('Bulk reference pair must not apply a slab dipole correction')
    if config_type=='bulk' and abs(float(s.get('EFIELD',0.)))>1.e-12:
        raise ValueError('Bulk EFIELD is not supported by this collection convention')


def input_valence(directory: Path) -> float:
    """Sum POTCAR ZVAL times POSCAR counts, with no potential-file alteration."""
    directory=Path(directory)
    text=(directory/'POTCAR').read_text(errors='replace')
    vals=[float(v.replace('D','E').replace('d','e')) for v in re.findall(r'\bZVAL\s*=\s*('+_NUM+r')',text)]
    lines=(directory/'POSCAR').read_text().splitlines()
    if len(lines)<7: raise ValueError('Truncated POSCAR')
    index=5 if all(re.fullmatch(r'\d+',x) for x in lines[5].split()) else 6
    counts=[int(x) for x in lines[index].split()]
    if len(vals)!=len(counts) or not counts or any(n<=0 for n in counts):
        raise ValueError('Cannot uniquely match POTCAR ZVAL entries to POSCAR species counts')
    return sum(v*n for v,n in zip(vals,counts))


def inspect_run(directory: Path, role: str, *, allow_running=False) -> dict:
    """Check positive evidence of run completion and electron count."""
    directory=Path(directory)
    for name in ('INCAR','POSCAR','POTCAR','OUTCAR','LOCPOT'):
        p=directory/name
        if not p.is_file() or p.stat().st_size==0:
            raise ValueError(f'{role}: missing/empty {p}')
    status_path=directory/'macevolt_run_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else None
    if status is not None and status.get('status')!='complete':
        if not (allow_running and status.get('status')=='running'):
            raise ValueError(f'{role}: latest recorded run status is {status.get("status")!r}, not complete')
    inputs={name:sha256_file(directory/name) for name in ('INCAR','POSCAR','POTCAR','KPOINTS') if (directory/name).exists()}
    if status is not None and status.get('status')=='complete':
        recorded=status.get('raw',{}).get('input_sha256')
        if recorded is not None and recorded!=inputs:
            raise ValueError(f'{role}: inputs changed after the recorded completed run')
    settings=read_incar(directory/'INCAR')
    text=(directory/'OUTCAR').read_text(errors='replace')
    # Require VASP's normal end-of-run timing section; a partial output must not
    # pass merely because an earlier electronic cycle converged.
    completed = 'General timing and accounting informations for this job' in text
    if not completed: raise ValueError(f'{role}: OUTCAR has no normal completion footer in {directory}')
    nelects=re.findall(r'\bNELECT\s*=\s*('+_NUM+r')',text)
    if not nelects: raise ValueError(f'{role}: actual NELECT was not found in OUTCAR')
    nelect = float(nelects[-1].replace('D', 'E').replace('d', 'e'))
    neutral = input_valence(directory)
    if not math.isfinite(nelect): raise ValueError('Nonfinite actual NELECT')
    convergence='not_applicable_frozen_atomic_reference'
    if role=='scf':
        if int(settings.get('ICHARG',2))>=10:
            raise ValueError('SCF role uses frozen density')
        iterations = list(re.finditer(r"Iteration\s+\d+\s*\(\s*\d+\s*\)", text))
        final_cycle = text[iterations[-1].start():] if iterations else text
        if 'aborting loop because EDIFF is reached' in final_cycle:
            convergence='OUTCAR_EDIFF_reached'
        else:
            xml=directory/'vasprun.xml'
            try:
                from pymatgen.io.vasp.outputs import Vasprun
                vr=Vasprun(str(xml),parse_dos=False,parse_eigen=False,
                           parse_projected_eigen=False,parse_potcar_file=False)
                if not vr.converged_electronic:
                    raise ValueError('vasprun.xml says electronic SCF did not converge')
                convergence='vasprun_converged_electronic'
            except Exception as error:
                raise ValueError(f'SCF convergence unverified for {directory}: {error}; '
                                 'inspect the run, do not reuse a fresh LOCPOT alone') from error
    elif role!='proto': raise ValueError('role must be scf or proto')
    dipoles=re.findall(r'dipolmoment\s+('+_NUM+r')\s+('+_NUM+r')\s+('+_NUM+r')',text)
    return dict(role=role,completed=completed,convergence=convergence,nelect=nelect,
        neutral_valence=neutral,total_charge=neutral-nelect,
        outcar_dipole=[float(x.replace('D','E').replace('d','e')) for x in dipoles[-1]] if dipoles else None,
        potcar_sha256=inputs['POTCAR'],incar=settings,input_sha256=inputs)


def raw_signature(scf_dir: Path, proto_dir: Path) -> str:
    """Cache invalidation from full input hashes and raw-output size/mtime."""
    records=[]
    for role,d in (('scf',Path(scf_dir)),('proto',Path(proto_dir))):
        for name in ('INCAR','POSCAR','POTCAR','KPOINTS','OUTCAR','LOCPOT','AECCAR1','AECCAR2','vasprun.xml','macevolt_run_status.json'):
            p=d/name
            if not p.exists(): records.append([role,name,None]);continue
            st=p.stat();record=[role,name,st.st_size,st.st_mtime_ns]
            if name in ('INCAR','POSCAR','POTCAR','KPOINTS'): record.append(sha256_file(p))
            records.append(record)
    return hashlib.sha256(json.dumps(records,separators=(',',':')).encode()).hexdigest()


def validate_raw_pair(scf_dir, proto_dir, config_type='slab', open_axis=2, expected_charge=None):
    scf=inspect_run(Path(scf_dir),'scf');proto=inspect_run(Path(proto_dir),'proto')
    validate_pair_settings(scf['incar'],proto['incar'],config_type,open_axis)
    if scf['potcar_sha256']!=proto['potcar_sha256']:
        raise ValueError('SCF/proto POTCAR bytes differ: ionic cancellation cannot be assumed')
    if abs(proto['total_charge'])>1.e-5:
        raise ValueError(f'Proto run is not neutral according to actual NELECT: Q={proto["total_charge"]}')
    if expected_charge is not None and abs(float(expected_charge)-scf['total_charge'])>1.e-5:
        raise ValueError(f'Trajectory charge Q={expected_charge} disagrees with actual valence-NELECT={scf["total_charge"]}')
    # Equal explicit KPOINTS keeps reference discretization reproducible. Compare
    # normalized non-comment content rather than the arbitrary title line.
    def kpoints(d):
        p=Path(d)/'KPOINTS'
        return ' '.join(p.read_text().splitlines()[1:]).lower().split() if p.exists() else None
    if kpoints(scf_dir)!=kpoints(proto_dir): raise ValueError('SCF/proto KPOINTS differ')
    return dict(schema=PAIR_SCHEMA,potential_kind='ionic_plus_hartree',scf=scf,proto=proto,
                raw_signature=raw_signature(scf_dir,proto_dir),
                caveat='Grid/coordinate identity must also be checked by the volumetric collector. '
                       'Reconstructed AE density need not be the local ionic/Hartree potential source.')


def final_force_block(outcar_text: str, atom_count: int):
    """Final OUTCAR positions/forces, in VASP order; caller aligns atom ordering."""
    lines=outcar_text.splitlines()
    starts=[i for i,s in enumerate(lines) if 'POSITION' in s and 'TOTAL-FORCE' in s]
    if not starts:raise ValueError('No POSITION/TOTAL-FORCE block found in OUTCAR')
    values=[]
    for line in lines[starts[-1]+1:]:
        fields=line.split()
        if len(fields)!=6:
            if values:break
            continue
        try:row=[float(x.replace('D','E').replace('d','e')) for x in fields]
        except ValueError:
            if values:break
            continue
        values.append(row)
        if len(values)==atom_count:break
    if len(values)!=atom_count or not all(math.isfinite(x) for row in values for x in row):
        raise ValueError('Final OUTCAR force block is incomplete/nonfinite')
    return [r[:3] for r in values], [r[3:] for r in values]


# Fourier conventions and target construction (unchanged).
def _component_weight(source_info, key, default, require_nonzero=False):
    value = np.asarray(source_info.get(key, default), dtype=float).reshape(3)
    if (not np.all(np.isfinite(value))) or np.any(value < 0.0):
        raise ValueError(f"Invalid {key}: {value.tolist()}")
    if require_nonzero and not np.any(value > 0.0):
        raise ValueError(f"{key} must contain at least one positive component")
    return value


def max_n_cut(atoms, sigma=1.5, max_l=1):
    cutoff = (
        FOURIER_CUTOFF_SAFETY_FACTOR
        * 0.75
        * (1.0 / sigma)
        * (max_l + 1) ** 0.3
        * 3.0
    )

    cell_lengths = np.asarray(Cell(atoms.cell).lengths(), dtype=float)
    max_ns = cutoff * cell_lengths / (2.0 * np.pi)
    return (np.ceil(max_ns).astype(int) * 2).tolist()


def gaussian_kernel(shape, sigma, cell):
    integer_axes = [np.fft.fftfreq(n) * n for n in shape]
    mesh = np.meshgrid(*integer_axes, indexing="ij")
    integer_k = np.stack(mesh, axis=-1)
    rcell = 2.0 * np.pi * np.linalg.inv(np.asarray(cell, dtype=float).T)
    k_vectors = np.einsum("...i,ij->...j", integer_k, rcell)
    return np.exp(-0.5 * float(sigma) ** 2 * np.sum(k_vectors**2, axis=-1))


def project_hermitian_fft(fft_grid):
    fft_grid = np.asarray(fft_grid, dtype=np.complex128)
    partner = np.conj(
        np.roll(
            np.flip(fft_grid, axis=tuple(range(fft_grid.ndim))),
            shift=tuple(1 for _ in range(fft_grid.ndim)),
            axis=tuple(range(fft_grid.ndim)),
        )
    )
    return 0.5 * (fft_grid + partner)


def truncate_fft3(fft_grid, new_shape, apply_scale=True):
    mx, my, mz = [int(v) for v in new_shape]
    nx, ny, nz = fft_grid.shape

    if mx > nx or my > ny or mz > nz:
        raise ValueError(
            f"Requested truncated FFT shape {new_shape} is larger than full shape {fft_grid.shape}. "
            "Increase the VASP FFT grid density or reduce K_CUTOFF."
        )

    shifted = np.fft.fftshift(fft_grid)

    sx = nx // 2 - mx // 2
    sy = ny // 2 - my // 2
    sz = nz // 2 - mz // 2

    truncated = shifted[sx:sx + mx, sy:sy + my, sz:sz + mz]
    scale = (mx * my * mz) / (nx * ny * nz) if apply_scale else 1.0

    return project_hermitian_fft(np.fft.ifftshift(truncated) * scale)


def split_complex_fft(fft_grid):
    fft_grid = np.asarray(fft_grid)

    if fft_grid.ndim == 3:
        return np.stack([fft_grid.real, fft_grid.imag], axis=0)

    if fft_grid.ndim == 4 and fft_grid.shape[0] == 3:
        return np.stack([fft_grid.real, fft_grid.imag], axis=1)

    raise ValueError(f"Unsupported FFT shape: {fft_grid.shape}")


def fft_attrs(atoms, grid_shape):
    cell_lengths = np.asarray(Cell(atoms.cell).lengths(), dtype=float)
    grid_shape = np.asarray(grid_shape, dtype=int)
    i_cutoff = np.ceil(K_CUTOFF * cell_lengths / (2.0 * np.pi)).astype(int)

    return {
        "kcutoff": float(K_CUTOFF),
        "Ns": grid_shape,
        "i_cutoff": i_cutoff,
    }


def truncate_signal_fft(signal, atoms):
    attrs = fft_attrs(atoms, signal.shape)
    new_shape = tuple((2 * attrs["i_cutoff"]).astype(int))
    truncated = truncate_fft3(np.fft.fftn(signal), new_shape, apply_scale=True)
    return truncated, attrs


def align_electrostatic_gauge(scf_grid, proto_grid):
    """One explicit, recorded gauge for a periodic scalar field."""
    scf_grid = np.asarray(scf_grid, dtype=float)
    proto_grid = np.asarray(proto_grid, dtype=float)
    if scf_grid.shape != proto_grid.shape or scf_grid.ndim != 3 or not scf_grid.size:
        raise ValueError("SCF and proto potential grids must have the same nonempty 3D shape")
    if not np.isfinite(scf_grid).all() or not np.isfinite(proto_grid).all():
        raise ValueError("Nonfinite SCF/proto potential grid; gauge alignment cannot repair it")
    scf_mean, proto_mean = float(scf_grid.mean()), float(proto_grid.mean())
    proto = proto_grid - proto_mean
    deformation = scf_grid - scf_mean - proto
    return deformation, proto, scf_mean, proto_mean


def align_electronic_level(fermi_level, scf_mean):
    """Shift EF by exactly the scalar offset removed from the total potential."""
    if not np.isfinite(fermi_level) or not np.isfinite(scf_mean):
        raise ValueError("Nonfinite Fermi level or potential gauge")
    return float(fermi_level) - float(scf_mean)


def add_density_targets(atoms, shifted_rho_fft, sigmas, total_charge):
    volume = float(atoms.get_volume())

    for sigma in sigmas:
        new_shape = tuple(max_n_cut(atoms, sigma=sigma, max_l=1))
        smoothed_fft = truncate_fft3(shifted_rho_fft, new_shape, apply_scale=True)
        smoothed_fft *= gaussian_kernel(new_shape, sigma, atoms.cell.array)
        smoothed_fft[(0, 0, 0)] = float(total_charge) * np.prod(new_shape) / volume
        smoothed_fft = project_hermitian_fft(smoothed_fft)

        atoms.info[f"vasp_rho_fftn_shifted_{sigma}"] = split_complex_fft(smoothed_fft).astype(np.float32)


def open_axis_from_info(info):
    """One selected fractional plane fixes the slab cell axis."""
    selected = [(prefix, axis, info[f"{prefix}_{letter}frac"])
                for prefix in ("vacuum", "dipole_correction") for axis, letter in enumerate("xyz")
                if f"{prefix}_{letter}frac" in info]
    for prefix in ("vacuum", "dipole_correction"):
        if sum(item[0] == prefix for item in selected) > 1:
            raise ValueError(f"Specify one {prefix} fractional plane")
    axes = {axis for _, axis, _ in selected}
    if "open_axis" in info:
        axes.add(int(info["open_axis"]))
    if len(axes) > 1 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("Plane coordinates and open_axis must refer to the same cell axis")
    if any(not np.isfinite(float(v)) or not 0. <= float(v) < 1. for _, _, v in selected):
        raise ValueError("Fractional plane coordinates must be finite and in [0, 1)")
    return next(iter(axes), 2)


def add_potential_targets(atoms, shifted_phi_fft, sigmas, key_prefix="vasp_phi"):
    """Store one Gaussian-resolved target per requested width."""
    for sigma in sigmas:
        sigma = float(sigma)
        shape = tuple(max_n_cut(atoms, sigma=sigma, max_l=1))
        coefficients = truncate_fft3(shifted_phi_fft, shape, apply_scale=True)
        coefficients = project_hermitian_fft(coefficients*gaussian_kernel(shape, sigma, atoms.cell.array))
        coefficients[0, 0, 0] = 0.
        atoms.info[f"{key_prefix}_fftn_shifted_{sigma}"] = split_complex_fft(coefficients).astype(np.float32)


def compute_vacuum_potential(atoms, outdir, potential_grid=None):
    if potential_grid is None:
        potential_grid = np.asarray(Locpot.from_file(str(outdir / "LOCPOT")).data["total"], dtype=float)

    open_axes = np.flatnonzero(~np.asarray(atoms.pbc, dtype=bool))
    if open_axes.size != 1:
        raise ValueError(
            f"Vacuum-potential collection requires one open axis, got pbc={atoms.pbc}"
        )
    open_axis = int(open_axes[0])
    incar = Incar.from_file(str(outdir / "INCAR"))
    dipole_center = np.asarray(
        incar.get("DIPOL", [0.0, 0.0, 0.0]), dtype=float
    )
    correction_fraction = float((dipole_center[open_axis] + 0.5) % 1.0)
    supplied_branch = atoms.info.get(f"dipole_correction_{'xyz'[open_axis]}frac")
    if supplied_branch is not None and not np.isclose(float(supplied_branch), correction_fraction, rtol=0., atol=1.e-8):
        raise ValueError("Declared dipole-correction plane disagrees with VASP DIPOL")
    requested_fraction = atoms.info.get(f"vacuum_{'xyz'[open_axis]}frac")
    if requested_fraction is not None and np.isfinite(float(requested_fraction)):
        vacuum_fraction = float(requested_fraction) % 1.0
    else:
        scaled = atoms.get_scaled_positions(wrap=False)
        z_wrapped = np.mod(scaled[:, open_axis], 1.0)
        distance_from_atoms = np.mod(correction_fraction - z_wrapped, 1.0)
        positive = distance_from_atoms[distance_from_atoms > 1.0e-8]
        if positive.size == 0:
            raise ValueError("Cannot locate a vacuum interval before the DIPOL plane.")
        vacuum_fraction = float(
            (correction_fraction - 0.5 * float(np.min(positive))) % 1.0
        )

    planar_potential = potential_grid.mean(
        axis=tuple(axis for axis in range(3) if axis != open_axis)
    )
    nz = planar_potential.shape[0]
    z_grid = (float(vacuum_fraction) % 1.0) * nz
    iz0 = int(np.floor(z_grid)) % nz
    iz1 = (iz0 + 1) % nz
    weight = z_grid - np.floor(z_grid)
    vacuum_potential = (
        (1.0 - weight) * planar_potential[iz0]
        + weight * planar_potential[iz1]
    )

    cell = np.asarray(atoms.cell, dtype=float)
    periodic = [axis for axis in range(3) if axis != open_axis]
    normal = np.cross(cell[periodic[0]], cell[periodic[1]])
    normal /= max(float(np.linalg.norm(normal)), 1.0e-30)
    open_length = abs(float(np.dot(cell[open_axis], normal)))
    dz = open_length / nz
    slope = (planar_potential[(iz0 + 1) % nz] - planar_potential[(iz0 - 1) % nz]) / (
        2.0 * dz
    )
    return (
        float(vacuum_potential - np.mean(potential_grid)),
        float(vacuum_fraction),
        float(slope),
    )


def get_output_dirs(base_path):
    folders = sorted((p for p in Path(base_path).iterdir()
                      if p.is_dir() and p.name.startswith("vasp_")),
                     key=lambda p: int(p.name.split("_")[1]))
    return [folder / f"output_{i}" for folder in folders for i in range(CHUNK_SIZE)
            if int(folder.name.split("_")[1]) + i not in UNCONVERGED_IDS
            and (folder / f"output_{i}").is_dir()]


def proto_locpot_path_for_outdir(outdir):
    return PROTO_BASE_PATH / Path(outdir).relative_to(BASE_PATH) / "LOCPOT"


def get_external_field(outdir, atoms):
    incar = read_incar(Path(outdir) / "INCAR")
    value = float(incar.get("EFIELD", 0.0) or 0.0)
    if not np.isfinite(value):
        raise ValueError(f"Nonfinite EFIELD in {outdir}")
    if atoms.info.get("config_type") != "slab":
        if abs(value) > 1.e-12:
            raise ValueError("Periodic bulk EFIELD is not supported")
        return value, np.zeros(3)
    axis = open_axis_from_info(atoms.info)
    if axis not in (0, 1, 2) or int(incar.get("IDIPOL", 0)) != axis + 1:
        raise ValueError("IDIPOL must select the slab open axis")
    cell = np.asarray(atoms.cell)
    normal = np.cross(cell[(axis + 1) % 3], cell[(axis + 2) % 3])
    normal /= np.linalg.norm(normal)
    if cell[axis] @ normal < 0:
        normal = -normal
    return value, -value * normal


def read_chg_total(path):
    data = np.asarray(Chgcar.from_file(str(path)).data["total"], dtype=float)
    if data.ndim != 3 or not data.size or not np.isfinite(data).all():
        raise ValueError(f"Invalid charge grid: {path}")
    return data


def read_hdf(path, dataset_name="truncated_fftn"):
    with h5py.File(path, "r") as handle:
        ds = handle[dataset_name]
        return ds[...], dict(ds.attrs)


def write_hdf(path, info, dataset_name="truncated_fftn"):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with h5py.File(temporary, "w") as handle:
            ds = handle.create_dataset(dataset_name, data=info[dataset_name])
            for key, value in info.items():
                if key != dataset_name:
                    ds.attrs[key] = value
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_potential_runs(scf_path, proto_path, config_type, open_axis=2):
    # A direct call to the grid routine still validates the paired settings.
    settings = [read_incar(Path(path).parent / "INCAR") for path in (scf_path, proto_path)]
    validate_pair_settings(*settings, config_type, open_axis)
    hashes = [sha256_file(Path(path).parent / "POTCAR") for path in (scf_path, proto_path)]
    if hashes[0] != hashes[1]:
        raise ValueError("SCF and protoatomic POTCAR files do not match")
    return tuple(dict(incar=incar, potcar_sha256=digest)
                 for incar, digest in zip(settings, hashes))


def read_potential_pair(outdir, atoms, pair=None):
    paths = (Path(outdir) / "LOCPOT", proto_locpot_path_for_outdir(outdir))
    grids = [Locpot.from_file(str(path)) for path in paths]
    a, b = (grid.structure for grid in grids)
    if a.composition != b.composition or [str(x.specie) for x in a] != [str(x.specie) for x in b]:
        raise ValueError("SCF/proto LOCPOT species differ")
    if not np.allclose(a.lattice.matrix, b.lattice.matrix, rtol=0., atol=1.e-3):
        raise ValueError("SCF/proto LOCPOT cells differ")
    if not np.allclose(a.frac_coords, b.frac_coords, rtol=0., atol=1.e-3):
        raise ValueError("SCF/proto LOCPOT coordinates differ")
    cell = np.asarray(atoms.cell)
    symbols = np.asarray(atoms.get_chemical_symbols())
    grid_symbols = np.asarray([str(site.specie) for site in a])
    if sorted(symbols) != sorted(grid_symbols) or not np.allclose(cell, a.lattice.matrix, rtol=0., atol=1.e-3):
        print(sorted(symbols))
        print(sorted(grid_symbols))
        print(cell)
        print(a.lattice.matrix)
        raise ValueError("Trajectory and SCF LOCPOT composition/cell differ")
    # ASE may reorder species for VASP. Match positions within each species.
    positions = atoms.get_scaled_positions(wrap=False)
    for symbol in set(grid_symbols):
        delta = np.asarray(a.frac_coords)[grid_symbols == symbol, None, :] - positions[symbols == symbol][None, :, :]
        delta -= np.rint(delta)
        distance = np.linalg.norm(delta @ cell, axis=-1)
        rows, cols = linear_sum_assignment(distance)
        if np.max(distance[rows, cols], initial=0.) > 1.e-3:
            raise ValueError("Trajectory positions differ from the paired potential grids")
    if pair is None:
        validate_potential_runs(*paths, atoms.info["config_type"], open_axis_from_info(atoms.info))
    raw = [np.asarray(grid.data["total"], dtype=float) for grid in grids]
    deformation, proto, mean, proto_mean = align_electrostatic_gauge(*raw)
    metadata = dict(potential_reference=POTENTIAL_REFERENCE, scf_mean=mean,
                    proto_mean=proto_mean, gauge_version="joint_scf_zero_mean_v1",
                    potential_kind="ionic_plus_hartree", potcar_match=1)
    return deformation, proto, metadata, raw


def deformation_hartree_potential_grids(outdir, atoms):
    return read_potential_pair(outdir, atoms)[:3]


def scalar_labels(outdir, atoms):
    """Check the trajectory against final OUTCAR values before making labels."""
    forces = np.asarray(atoms.get_forces(apply_constraint=False), dtype=float)
    energy = float(atoms.get_potential_energy(force_consistent=True))
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all() or not np.isfinite(energy):
        raise ValueError(f"Nonfinite/incomplete energy or forces in {outdir}")
    text = (Path(outdir) / "OUTCAR").read_text(errors="replace")
    values = re.findall(r"free\s+energy\s+TOTEN\s*=\s*(" + _NUM + ")", text)
    if not values:
        raise ValueError(f"No final TOTEN in {outdir}")
    final_energy = float(values[-1].replace("D", "E").replace("d", "e"))
    positions, reference_forces = map(np.asarray, final_force_block(text, len(atoms)))
    delta = (positions[:, None] - atoms.positions[None, :]) @ np.linalg.inv(np.asarray(atoms.cell))
    delta -= np.rint(delta)
    distance = np.linalg.norm(delta @ np.asarray(atoms.cell), axis=-1)
    rows, cols = linear_sum_assignment(distance)
    if np.max(distance[rows, cols], initial=0.) > 2.e-5:
        raise ValueError(f"Trajectory/OUTCAR final positions differ in {outdir}")
    if np.max(np.abs(reference_forces[rows] - forces[cols]), initial=0.) > 2.e-5:
        raise ValueError(f"Trajectory/OUTCAR forces differ in {outdir}")
    if not np.isclose(energy, final_energy, rtol=0., atol=1.e-5):
        raise ValueError(f"Trajectory/OUTCAR free energies differ in {outdir}")
    try:
        dipole = np.asarray(atoms.get_dipole_moment(), dtype=float).reshape(3)
    except Exception:
        values = re.findall(r"dipolmoment\s+(" + _NUM + r")\s+(" + _NUM + r")\s+(" + _NUM + ")", text)
        if not values and atoms.info["config_type"] == "slab":
            raise ValueError(f"No slab dipole in trajectory or OUTCAR: {outdir}")
        dipole = np.array([float(x.replace("D", "E").replace("d", "e")) for x in values[-1]]) if values else np.zeros(3)
    oc = Outcar(str(Path(outdir) / "OUTCAR"))
    if not np.isfinite(dipole).all() or not np.isfinite(oc.efermi):
        raise ValueError(f"Nonfinite dipole or Fermi level in {outdir}")
    return energy, forces, dipole, oc


def check_pair(outdir, atoms):
    charge = atoms.info.get("total_charge", -float(atoms.info.get("excess_electrons", 0.)))
    return validate_raw_pair(outdir, proto_locpot_path_for_outdir(outdir).parent,
                             atoms.info["config_type"], open_axis_from_info(atoms.info), charge)


def current_cache(path, signature):
    if not Path(path).is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            attrs = handle["truncated_fftn"].attrs
            return (attrs.get("raw_signature") == signature
                    and attrs.get("pair_schema") == PAIR_SCHEMA
                    and attrs.get("kcutoff") == K_CUTOFF
                    and (Path(path).name == RHO_HDF_NAME
                         or attrs.get("potential_reference") == POTENTIAL_REFERENCE))
    except (OSError, KeyError):
        return False


def extract_hdf_job(job):
    outdir, reuse = job
    trajectory = outdir / f"{SYSTEM_NAME}_vasp_sp.traj"
    if not trajectory.is_file():
        return None, f"missing trajectory, skipping {outdir}"
    if not all((outdir / name).is_file() for name in ("AECCAR1", "AECCAR2")):
        return None, f"missing AECCAR1/2, skipping {outdir}"
    atoms = read(str(trajectory))
    _, forces, _, _ = scalar_labels(outdir, atoms)
    fmax = float(np.sqrt((forces**2).sum(axis=1).max()))
    if fmax > FMAX_TOL:
        return None, f"fmax {fmax:.3f} > {FMAX_TOL:.3f}, skipping {outdir}"
    pair = check_pair(outdir, atoms)
    signature = pair["raw_signature"]
    get_external_field(outdir, atoms)
    # Always check the raw pair's geometry, even when reusing Fourier caches.
    deformation, proto, metadata, _ = read_potential_pair(outdir, atoms, pair)
    paths = [outdir / name for name in (RHO_HDF_NAME, PHI_HDF_NAME, PROTO_PHI_HDF_NAME)]
    saved = 0
    for kind, path in enumerate(paths):
        if reuse and current_cache(path, signature):
            continue
        if kind == 0:
            first, second = (read_chg_total(outdir / name) for name in ("AECCAR1", "AECCAR2"))
            if first.shape != second.shape:
                raise ValueError(f"AECCAR grid mismatch in {outdir}")
            signal = (second - first) / atoms.get_volume()
            extra = {}
        else:
            signal = deformation if kind == 1 else proto
            extra = dict(metadata, actual_total_charge=pair["scf"]["total_charge"])
        coefficients, attrs = truncate_signal_fft(signal, atoms)
        write_hdf(path, dict(truncated_fftn=coefficients[None], **attrs, **extra,
                            raw_signature=signature, pair_schema=PAIR_SCHEMA))
        saved += 1
    return outdir, f"collected {outdir}; wrote {saved} caches; applied field retained"


def combine_job(outdir):
    atoms = read(str(outdir / f"{SYSTEM_NAME}_vasp_sp.traj"))
    source_info = dict(atoms.info)
    energy, forces, dipole, oc = scalar_labels(outdir, atoms)
    config_type = source_info["config_type"]
    open_axis = open_axis_from_info(source_info)
    atoms.pbc = True
    if config_type == "slab":
        atoms.pbc[open_axis] = False
    elif config_type != "bulk":
        raise ValueError(f"Unknown config_type: {config_type}")
    _, ext_field = get_external_field(outdir, atoms)
    paths = [outdir / name for name in (RHO_HDF_NAME, PHI_HDF_NAME, PROTO_PHI_HDF_NAME)]
    signature = raw_signature(outdir, proto_locpot_path_for_outdir(outdir).parent)
    if not all(current_cache(path, signature) for path in paths):
        raise ValueError(f"Stale/incompatible HDF cache in {outdir}; rebuild caches, not DFT")
    (rho_fftn, _), (phi_fftn, phi_attrs), (proto_phi_fftn, proto_attrs) = map(read_hdf, paths)
    rho_fft = -rho_fftn.sum(axis=0)  # Delta n in HDF; signed charge Delta rho in XYZ.
    phi_fft, proto_phi_fft = phi_fftn[0], proto_phi_fftn[0]
    density_sigmas, potential_sigmas = DENSITY_SIGMAS, POTENTIAL_SIGMAS
    total_charge = source_info.get("total_charge")
    if total_charge is None:
        total_charge = -float(source_info.get("excess_electrons", 0.0))

    atoms.arrays["vasp_forces"] = forces
    atoms.info = {
        "config_type": config_type,
        "open_axis": open_axis_from_info(source_info),
        "vasp_free_energy": energy,
        "vasp_efermi": align_electronic_level(oc.efermi, phi_attrs["scf_mean"]),
        "external_field": np.asarray(ext_field, dtype=float),
        "total_charge": float(total_charge),
        "dft_group_id": outdir.parent.name,
        "potential_reference": POTENTIAL_REFERENCE,
        "density_convention": "signed_charge_deformation",
        "potcar_match": int(phi_attrs.get("potcar_match", -1)),
        "potential_kind": "ionic_plus_hartree",
    }
    for prefix in ("vacuum", "dipole_correction"):
        key = f"{prefix}_{'xyz'[open_axis]}frac"
        if key in source_info:
            atoms.info[key] = float(source_info[key])

    add_density_targets(atoms, rho_fft, density_sigmas, float(total_charge))
    atoms.info["density_smearing_width"] = float(density_sigmas[0])
    atoms.info["potential_smearing_width"] = float(potential_sigmas[0])
    slab = config_type == "slab"
    dipole_weight = (_component_weight(source_info, "config_dipole_weight", np.eye(3)[open_axis])
                     if slab else np.zeros(3))
    atoms.info.update(
        vasp_dipole=(-np.asarray(dipole) if slab else np.asarray(dipole)).flatten(),
        config_dipole_weight=dipole_weight,
        config_vacuum_potential_weight=float(slab), config_fermi_level_weight=float(slab),
    )
    if slab:
        vacuum, fraction, _ = compute_vacuum_potential(atoms, outdir)
        center = read_incar(outdir / "INCAR")["DIPOL"]
        atoms.info.update(vacuum_potential=vacuum)
        atoms.info[f"vacuum_{'xyz'[open_axis]}frac"] = fraction
        atoms.info[f"dipole_correction_{'xyz'[open_axis]}frac"] = float((center[open_axis] + .5) % 1.)
    for field, prefix in ((phi_fft, "vasp_phi"), (proto_phi_fft, "vasp_proto_phi")):
        add_potential_targets(atoms, field, potential_sigmas, prefix)

    return atoms, f"processed {outdir}"


def split_configs(configs):
    groups = {}
    for index, atoms in enumerate(configs):
        group_id = str(atoms.info.get("dft_group_id", f"ungrouped_{index}"))
        groups.setdefault(group_id, []).append(atoms)
    strata = {}
    for group_id, group_configs in groups.items():
        config_types = tuple(sorted({item.info["config_type"] for item in group_configs}))
        strata.setdefault(config_types, []).append(group_id)
    rng = np.random.default_rng(SPLIT_SEED)
    valid_groups = set()
    for group_ids in strata.values():
        group_ids = np.asarray(sorted(group_ids), dtype=object)
        rng.shuffle(group_ids)
        if group_ids.size <= 1:
            continue
        count = int(round(VALID_FRACTION * group_ids.size))
        count = min(max(count, 1), group_ids.size - 1)
        valid_groups.update(str(value) for value in group_ids[:count])
    train_configs = [
        atoms for atoms in configs if str(atoms.info["dft_group_id"]) not in valid_groups
    ]
    valid_configs = [
        atoms for atoms in configs if str(atoms.info["dft_group_id"]) in valid_groups
    ]
    if not train_configs or not valid_configs:
        raise ValueError("A nonempty group-disjoint split needs at least two groups in a stratum")
    return train_configs, valid_configs


def _initialize_worker(settings):
    globals().update(settings)


def run_jobs(function, jobs):
    # Pass CLI paths explicitly to workers, also under spawn/forkserver.
    names = ("BASE_PATH", "PROTO_BASE_PATH", "SYSTEM_NAME", "K_CUTOFF", "DENSITY_SIGMAS",
             "POTENTIAL_SIGMAS", "POTENTIAL_REFERENCE", "FMAX_TOL", "PAIR_SCHEMA",
             "RHO_HDF_NAME", "PHI_HDF_NAME", "PROTO_PHI_HDF_NAME", "FOURIER_CUTOFF_SAFETY_FACTOR")
    settings = {name: globals()[name] for name in names}
    if N_PROCS is None or N_PROCS <= 1:
        return [function(job) for job in jobs]
    with mp.Pool(N_PROCS, initializer=_initialize_worker, initargs=(settings,)) as pool:
        return list(pool.imap(function, jobs))


def collect():
    results = run_jobs(extract_hdf_job, [(p, REUSE_HDF) for p in get_output_dirs(BASE_PATH)])
    for _, message in results:
        print(message)
    # Only accepted, currently discovered jobs are combined, never orphaned caches.
    outdirs = sorted(p for p, _ in results if p is not None)
    records = run_jobs(combine_job, outdirs)
    configs = []
    for atoms, message in records:
        print(message)
        configs.append(atoms)
    if not configs:
        raise SystemExit("No configurations collected")
    train, valid = split_configs(configs)
    for suffix, subset in (("", configs), ("_train", train), ("_valid", valid)):
        path = BASE_PATH / f"combined_VASP_shifted{suffix}.xyz"
        write(str(path), subset)
        print(f"Wrote {len(subset)} configurations to {path}")


# Optional raw-source diagnostic; never used to modify any training label.
def nonplanar_poisson_check(scf_potential, proto_potential, delta_electron_density,
                            cell, *, sigma=1.5, open_axis=2):
    """Diagnostic on full raw arrays; not a label conversion or pass/fail bound."""
    a,b,n=(np.asarray(x,dtype=float) for x in (scf_potential,proto_potential,delta_electron_density))
    if a.shape!=b.shape or a.shape!=n.shape or a.ndim!=3 or not all(np.isfinite(x).all() for x in (a,b,n)):
        raise ValueError('Poisson diagnostic requires finite, matching full 3D arrays')
    if sigma<=0 or open_axis not in (0,1,2):raise ValueError('Invalid diagnostic width or open axis')
    reciprocal=2*np.pi*np.linalg.inv(np.asarray(cell,dtype=float)).T
    axes=[np.fft.fftfreq(k)*k for k in a.shape]
    integer=np.stack(np.meshgrid(*axes,indexing='ij'),-1)
    wave=integer@reciprocal;k2=np.sum(wave*wave,axis=-1)
    periodic=[i for i in range(3) if i!=open_axis]
    selected=(k2>0)&np.any(integer[...,periodic]!=0,axis=-1)
    kernel=np.exp(-.5*sigma*sigma*k2)
    potential=np.fft.fftn(a-b)/a.size*kernel
    density=np.fft.fftn(n)/a.size*kernel
    # n is positive electron number; delta electron potential is +K*delta n/k^2.
    K=1/(5.526349406e-3)
    predicted=K*density/np.where(k2>0,k2,1.)
    error=(predicted-potential)[selected]
    return dict(modes=int(selected.sum()),sigma_A=float(sigma),
        nonplanar_potential_RMS_eV=float(np.linalg.norm(potential[selected])),
        nonplanar_Poisson_error_RMS_eV=float(np.linalg.norm(error)),
        electron_number_change=float(n.mean()*abs(np.linalg.det(cell))),
        scope='Smoothed nonplanar modes only; not a vacuum error floor or proof of exact PAW source identity')


def preflight(limit=None, poisson_check=False):
    directories = get_output_dirs(BASE_PATH)
    selected = directories if limit is None else directories[:limit]
    records = []
    for outdir in selected:
        try:
            atoms = read(str(outdir / f"{SYSTEM_NAME}_vasp_sp.traj"))
            report = check_pair(outdir, atoms)
            _, forces, _, _ = scalar_labels(outdir, atoms)
            _, _, metadata, (scf, proto) = read_potential_pair(outdir, atoms, report)
            get_external_field(outdir, atoms)
            axis = open_axis_from_info(atoms.info)
            first, second = (read_chg_total(outdir / name) for name in ("AECCAR1", "AECCAR2"))
            if first.shape != second.shape:
                raise ValueError("AECCAR grids differ")
            # Ensure the existing grids can support the configured Fourier cutoff.
            for shape in (first.shape, scf.shape):
                target_shape = 2 * fft_attrs(atoms, shape)["i_cutoff"]
                if np.any(target_shape > np.asarray(shape)):
                    raise ValueError(f"K_CUTOFF requires {target_shape.tolist()}, larger than raw grid {shape}")
            if atoms.info["config_type"] == "slab" and report["proto"]["outcar_dipole"] is not None:
                periodic = [i for i in range(3) if i != axis]
                area_vector = np.cross(atoms.cell[periodic[0]], atoms.cell[periodic[1]])
                area = float(np.linalg.norm(area_vector))
                step = (1 / 5.526349406e-3) * abs(np.asarray(report["proto"]["outcar_dipole"]) @ (area_vector / area)) / area
                report["proto_boundary_diagnostic"] = dict(
                    dipole_step_magnitude_eV=float(step),
                    note="Diagnostic only; a nonzero proto boundary is not silently subtracted.")
            if poisson_check:
                sources = {"AE_reconstructed_valence_difference": (second - first) / atoms.get_volume()}
                proto_dir = proto_locpot_path_for_outdir(outdir).parent
                if all((path / "CHGCAR").is_file() for path in (outdir, proto_dir)):
                    sources["CHGCAR_smooth_grid_difference"] = (read_chg_total(outdir / "CHGCAR") - read_chg_total(proto_dir / "CHGCAR")) / atoms.get_volume()
                report["source_comparators"] = {
                    key: nonplanar_poisson_check(scf, proto, value, atoms.cell,
                         sigma=float(POTENTIAL_SIGMAS[0]), open_axis=axis)
                    for key, value in sources.items()}
            fmax = float(np.linalg.norm(forces, axis=1).max())
            records.append(dict(path=str(outdir), ok=True, report=report, gauge=metadata,
                                excluded_by_fmax=fmax > FMAX_TOL))
        except Exception as error:
            records.append(dict(path=str(outdir), ok=False, error=f"{type(error).__name__}: {error}"))
    print(json.dumps(dict(discovered=len(directories), checked=len(records),
                         partial=len(selected) < len(directories), records=records), indent=2))
    if not records or any(not record["ok"] for record in records):
        raise SystemExit(2)


def main():
    global BASE_PATH, PROTO_BASE_PATH, N_PROCS, REUSE_HDF
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=BASE_PATH)
    parser.add_argument("--proto-base-path", type=Path, default=PROTO_BASE_PATH)
    parser.add_argument("--processes", type=int, default=N_PROCS)
    parser.add_argument("--reuse-hdf", action="store_true", default=REUSE_HDF)
    parser.add_argument("--preflight", action="store_true", help="Read-only validation; write no HDF/XYZ files")
    parser.add_argument("--poisson-check", action="store_true", help="With --preflight, add a nonplanar raw-source diagnostic")
    parser.add_argument("--limit", type=int, help="Limit the read-only preflight; default checks every discovered pair")
    args = parser.parse_args()
    if (args.poisson_check or args.limit is not None) and not args.preflight:
        parser.error("--poisson-check and --limit require --preflight")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.processes < 0:
        parser.error("--processes must be nonnegative; 0 or 1 runs serially")
    BASE_PATH, PROTO_BASE_PATH = args.base_path, args.proto_base_path
    N_PROCS, REUSE_HDF = args.processes, args.reuse_hdf
    if args.preflight:
        preflight(args.limit, args.poisson_check)
    else:
        collect()


if __name__ == "__main__":
    main()
