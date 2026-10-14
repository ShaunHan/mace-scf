#!/usr/bin/env python3
"""Plot independent MACE-VOLT predictions against starting-frame DFT profiles.

DFT data are used only for the comparison curve. A full ESP requires a fitted
protoatomic baseline in the model; otherwise auto mode plots deformation ESP.
Edit the settings below. One PNG/CSV pair is saved per selected XYZ frame.
"""
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from mace_scf.utils.foundation import accelerate_backbone

ATOMS_FILE = 'IrO2OH_110_mdinit_0_50_vasp_sp.xyz'
MODEL_FILE = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/ml_training/wf_training_186/fit_0_stage4_finetune_v383.model'
FRAME_INDEXES = list(range(10, 20))
DEVICE, ENABLE_CUEQ, SCF_STEPS = 'cuda:0', True, 50
PROFILE = 'auto'  # 'auto', 'total' (requires a fitted baseline), or 'deformation'.
DFT_PHI_KEY = 'vasp_phi_fftn_shifted_1.5'
DFT_PROTO_KEY = 'vasp_proto_phi_fftn_shifted_1.5'  # Used only for the DFT total curve.
OUTPUT_DIR, DPI = Path('esp_profiles'), 300


class ProfileCalculator(MACEFixedPointSCF):
    # Expose existing spectral outputs without changing the model/library.
    def _build_results(self, output, diagnostics):
        results = super()._build_results(output, diagnostics)
        for key in ('fourier_potential', 'fourier_total_potential', 'k_vectors_grid_shape', 'k_vectors_mask'):
            results[key] = output[key].detach().cpu().numpy()
        results['proto_fitted'] = bool(self.model.field_dependent_charges_map.proto_fitted)
        return results


def scalar(value):
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError('Expected one finite scalar')
    return float(a.item())


def stored_fft(atoms, key):
    a = np.asarray(atoms.info[key], dtype=float)
    if a.ndim != 4 or a.shape[0] != 2 or not np.isfinite(a).all():
        raise ValueError(f'{key} must have shape (2, nx, ny, nz)')
    # Historical "shifted" keys contain coefficients in native FFT order.
    return a[0]+1j*a[1]


def resize_fft(values, shape):
    import torch
    from mace_scf.electrostatics.potential import resize_fft as remap
    return remap(torch.as_tensor(values), shape).numpy()


def planar_profile(coefficients):
    return np.fft.ifftn(coefficients).real.mean(axis=(0, 1))


def periodic_coordinate(size, length):
    coordinate = np.arange(size, dtype=float)*length/size
    coordinate = np.where(coordinate > .5*length, coordinate-length, coordinate)
    return coordinate, np.argsort(coordinate)


def open_length(atoms):
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    normal /= np.linalg.norm(normal)
    return abs(float(np.dot(atoms.cell[2], normal)))


def profile_kind(results):
    if PROFILE not in ('auto', 'total', 'deformation'):
        raise ValueError("PROFILE must be 'auto', 'total', or 'deformation'")
    kind = ('total' if results['proto_fitted'] else 'deformation') if PROFILE == 'auto' else PROFILE
    if kind == 'total' and not results['proto_fitted']:
        raise ValueError('This checkpoint has no fitted protoatomic baseline: independent full ESP is unavailable. '
                         "Use PROFILE='deformation', or a model containing a fitted baseline.")
    return kind


def profiles(atoms, results):
    kind = profile_kind(results)
    dft_fft = stored_fft(atoms, DFT_PHI_KEY)
    if kind == 'total':
        proto = stored_fft(atoms, DFT_PROTO_KEY)
        if dft_fft.shape != proto.shape:
            raise ValueError('DFT deformation and protoatomic grids differ')
        dft_fft = dft_fft+proto  # Reconstruct only the DFT comparison curve.
    observed_shape = np.array(dft_fft.shape)
    shape = tuple(np.asarray(results['k_vectors_grid_shape'], dtype=int).ravel())
    key = 'fourier_total_potential' if kind == 'total' else 'fourier_potential'
    model = np.asarray(results[key]).reshape(*shape, 2)
    model_fft = model[..., 0]+1j*model[..., 1]
    modes = np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(n)*n).astype(int) for n in shape], indexing='ij'), axis=-1)
    # Match observed/model modes, excluding ambiguous even-grid Nyquist modes.
    common = (np.abs(modes) <= (observed_shape-1)//2).all(axis=-1)
    common &= np.asarray(results['k_vectors_mask'], dtype=bool).reshape(shape)
    dft = planar_profile(resize_fft(dft_fft, shape)*common)
    prediction = planar_profile(model_fft*common)
    if kind == 'total':
        fermi_key = 'fermi_level' if 'fermi_level' in atoms.info else 'vasp_efermi'
        dft -= scalar(atoms.info[fermi_key])
        prediction += scalar(results['potential_reference'])-scalar(results['fermi_level'])
    else:
        # The deformation fit excludes the uniform Fourier mode. Compare shapes
        # in the same zero-mean gauge, without claiming absolute total ESP.
        dft -= dft.mean()
        prediction -= prediction.mean()
    rmse = float(np.sqrt(np.mean((prediction-dft)**2)))
    return dft, prediction, rmse


def plot_frame(atoms, calculator, index):
    if not np.array_equal(atoms.pbc, [True, True, False]):
        raise ValueError('ESP z profiles require an xy-periodic slab with open z')
    if not np.allclose(atoms.info['external_field'], 0.) or scalar(atoms.info['total_charge']) != 0.:
        raise ValueError('This starting-frame comparison expects neutral, zero-field DFT data')
    model_atoms = atoms.copy()
    model_atoms.info = {key: atoms.info[key] for key in (
        'external_field', 'total_charge', 'vacuum_zfrac', 'dipole_correction_zfrac')}
    model_atoms.set_constraint([])
    model_atoms.calc = calculator
    model_atoms.get_potential_energy()
    results = calculator.results
    kind = profile_kind(results)
    dft, prediction, rmse = profiles(atoms, results)
    length = open_length(atoms)
    coordinate, order = periodic_coordinate(len(dft), length)
    plane = (scalar(atoms.info['vacuum_zfrac']) % 1.)*length
    if plane > .5*length:
        plane -= length
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.plot(coordinate[order], dft[order], linewidth=2., label=f'DFT {kind} potential')
    axis.plot(coordinate[order], prediction[order], linewidth=2., label=f'MACE-VOLT {kind} potential')
    axis.axvline(plane, color='0.25', linestyle=':', linewidth=1.)
    axis.set_xlabel(r'surface-normal coordinate ($\AA$)')
    axis.set_ylabel('Fermi-referenced electrostatic potential (V)' if kind == 'total'
                    else r'Zero-mean deformation potential $\Delta\phi$ (V)')
    axis.set_title(f'Frame {index}: {kind} profile, RMSE = {1000*rmse:.1f} mV', fontsize=11)
    axis.legend()
    figure.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output = OUTPUT_DIR/f'esp_profile_{index}'
    figure.savefig(output.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close(figure)
    suffix = 'total_fermi_referenced_V' if kind == 'total' else 'deformation_zero_mean_V'
    np.savetxt(output.with_suffix('.csv'), np.column_stack((coordinate[order], dft[order], prediction[order])),
               delimiter=',', header=f'z_A,dft_{suffix},model_{suffix}', comments='')
    print(f'Frame {index}: {kind} profile RMSE={1000*rmse:.2f} mV; saved {output}.png', flush=True)
    return dict(frame=index, profile=kind, rmse_V=rmse)


def main():
    calc = ProfileCalculator(model_path=MODEL_FILE, device=DEVICE, pbc_handling='slab', scf_restart=False,
        external_field_key='external_field', total_charge_key='total_charge', ignore_nonconverged=True,
        scf_options=dict(num_scf_steps=SCF_STEPS, constant_charge=True, use_autograd_forces=True,
                         initial_density='local_guess', initial_fermi_level='zero'))
    fitted = bool(calc.model.field_dependent_charges_map.proto_fitted)
    profile_kind(dict(proto_fitted=fitted))
    if not fitted:
        print('Checkpoint has no fitted protoatomic baseline; plotting deformation potential only.', flush=True)
    if ENABLE_CUEQ:
        accelerate_backbone(calc.model, DEVICE)
    calc.model.field_dependent_charges_map.deployment_mode = 'unroll_scf'
    calc.model.eval().requires_grad_(False)
    frames = read(ATOMS_FILE, index=':')
    for index in FRAME_INDEXES:
        plot_frame(frames[index], calc, index)


if __name__ == '__main__':
    main()
