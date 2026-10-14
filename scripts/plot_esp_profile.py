#!/usr/bin/env python3
"""Compare starting-frame ESP z profiles with the Fourier labels in the XYZ.

The stored potential is deformation + protoatomic potential on its retained
Fourier grid. Models without a fitted protoatomic profile use the DFT reference
explicitly, retaining the model's predicted vacuum level and Fermi level.
Edit the settings below; the model is loaded once for all selected frames.
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
DFT_PHI_KEY = 'vasp_phi_fftn_shifted_1.5'
DFT_PROTO_KEY = 'vasp_proto_phi_fftn_shifted_1.5'
OUTPUT_DIR, DPI = Path('esp_profiles'), 300


class ProfileCalculator(MACEFixedPointSCF):
    # Expose existing spectral outputs for plotting; the library remains unchanged.
    def _build_results(self, output, diagnostics):
        results = super()._build_results(output, diagnostics)
        for key in ('fourier_total_potential', 'k_vectors_grid_shape', 'k_vectors_mask'):
            results[key] = output[key].detach().cpu().numpy()
        response = self.model.field_dependent_charges_map
        results['uses_dft_proto_reference'] = not bool(response.proto_fitted)
        if results['uses_dft_proto_reference'] and response.response_reference != 'neutral':
            raise ValueError('A deformation-only profile requires the neutral-reference convention')
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
    # Despite the historical key name, these coefficients use native FFT order.
    return a[0]+1j*a[1]


def resize_fft(values, shape):
    import torch
    from mace_scf.electrostatics.potential import resize_fft as remap
    return remap(torch.as_tensor(values), shape).numpy()


def planar_profile(coefficients, open_axis=2):
    return np.fft.ifftn(coefficients).real.mean(axis=tuple(i for i in range(3) if i != open_axis))


def vacuum_plane(coefficients, zfrac):
    modes = np.fft.fftfreq(coefficients.shape[2])*coefficients.shape[2]
    return float(np.real(np.sum(coefficients[0, 0]*np.exp(2j*np.pi*modes*zfrac)))/coefficients.size)


def periodic_coordinate(size, length):
    coordinate = np.arange(size, dtype=float)*length/size
    coordinate = np.where(coordinate > .5*length, coordinate-length, coordinate)
    return coordinate, np.argsort(coordinate)


def open_length(atoms, open_axis=2):
    periodic = [i for i in range(3) if i != open_axis]
    normal = np.cross(atoms.cell[periodic[0]], atoms.cell[periodic[1]])
    normal /= np.linalg.norm(normal)
    return abs(float(np.dot(atoms.cell[open_axis], normal)))


def profiles(atoms, results):
    deformation, proto = stored_fft(atoms, DFT_PHI_KEY), stored_fft(atoms, DFT_PROTO_KEY)
    if deformation.shape != proto.shape:
        raise ValueError('DFT deformation and protoatomic grids differ')
    shape = tuple(np.asarray(results['k_vectors_grid_shape'], dtype=int).ravel())
    dft_fft = resize_fft(deformation+proto, shape)
    model = np.asarray(results['fourier_total_potential']).reshape(*shape, 2)
    model_fft = model[..., 0]+1j*model[..., 1]
    modes = np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(n)*n).astype(int) for n in shape], indexing='ij'), axis=-1)
    # Compare only modes observed by DFT and retained by the model. Even-grid
    # Nyquist modes are excluded, matching the training observation mask.
    common = (np.abs(modes) <= (np.array(deformation.shape)-1)//2).all(axis=-1)
    common &= np.asarray(results['k_vectors_mask'], dtype=bool).reshape(shape)
    dft_phi = planar_profile(dft_fft*common)
    model_phi = planar_profile(model_fft*common)+scalar(results['potential_reference'])
    if results.get('uses_dft_proto_reference', False):
        reference = resize_fft(proto, shape)*common
        # The model already includes a scalar vacuum reference. Add only the
        # spatial variation of the known protoatomic field about that plane.
        model_phi += planar_profile(reference)-vacuum_plane(reference, scalar(atoms.info['vacuum_zfrac']))
    fermi_key = 'fermi_level' if 'fermi_level' in atoms.info else 'vasp_efermi'
    dft_fermi, model_fermi = scalar(atoms.info[fermi_key]), scalar(results['fermi_level'])
    dft_profile, model_profile = dft_phi-dft_fermi, model_phi-model_fermi
    shape_rmse = np.sqrt(np.mean(((model_phi-model_phi.mean())-(dft_phi-dft_phi.mean()))**2))
    voltage_rmse = np.sqrt(np.mean((model_profile-dft_profile)**2))
    return dft_profile, model_profile, float(shape_rmse), float(voltage_rmse)


def plot_frame(atoms, calculator, index):
    if not np.array_equal(atoms.pbc, [True, True, False]):
        raise ValueError('ESP z profiles require an xy-periodic slab with open z')
    if not np.allclose(atoms.info['external_field'], 0.) or scalar(atoms.info['total_charge']) != 0.:
        raise ValueError('This starting-frame comparison expects neutral, zero-field DFT data')
    atoms = atoms.copy()
    atoms.set_constraint([])
    atoms.calc = calculator
    atoms.get_potential_energy()
    results = calculator.results
    dft, model, shape_rmse, voltage_rmse = profiles(atoms, results)
    length = open_length(atoms)
    coordinate, order = periodic_coordinate(len(dft), length)
    plane = (scalar(atoms.info['vacuum_zfrac']) % 1.)*length
    if plane > .5*length:
        plane -= length
    figure, axis = plt.subplots(figsize=(7.6, 4.6))
    axis.plot(coordinate[order], dft[order], linewidth=2., label='DFT (stored Fourier resolution)')
    label = 'MACE-VOLT + DFT protoatomic reference' if results['uses_dft_proto_reference'] else 'MACE-VOLT'
    axis.plot(coordinate[order], model[order], linewidth=2., label=label)
    axis.axvline(plane, color='0.25', linestyle=':', linewidth=1.)
    axis.set_xlabel(r'surface-normal coordinate ($\AA$)')
    axis.set_ylabel('Fermi-referenced electrostatic potential (V)')
    axis.set_title(f'Frame {index}: profile RMSE = {1000*voltage_rmse:.1f} mV', fontsize=11)
    axis.legend()
    figure.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output = OUTPUT_DIR/f'esp_profile_{index}'
    figure.savefig(output.with_suffix('.png'), dpi=DPI, bbox_inches='tight')
    plt.close(figure)
    np.savetxt(output.with_suffix('.csv'), np.column_stack((coordinate[order],dft[order],model[order])),
               delimiter=',', header='z_A,dft_fermi_referenced_V,model_fermi_referenced_V', comments='')
    dft_mu = scalar(atoms.info.get('fermi_level', atoms.info.get('vasp_efermi')))
    metrics = dict(frame=index, uses_dft_proto_reference=results['uses_dft_proto_reference'],
                   shape_rmse_V=shape_rmse, profile_rmse_V=voltage_rmse,
                   dft_vacuum_potential_V=scalar(atoms.info['vacuum_potential']),
                   dft_fermi_level_eV=dft_mu, model_vacuum_potential_V=scalar(results['vacuum_potential']),
                   model_fermi_level_eV=scalar(results['fermi_level']),
                   scf_steps=results['num_scf_steps'], scf_residual=scalar(results['scf_residual']))
    print(f'Frame {index}: shape RMSE={1000*shape_rmse:.2f} mV; profile RMSE={1000*voltage_rmse:.2f} mV; '
          f'DFT vacuum-minus-Fermi={metrics["dft_vacuum_potential_V"]-dft_mu:.4f} V; '
          f'model={metrics["model_vacuum_potential_V"]-metrics["model_fermi_level_eV"]:.4f} V; '
          f'{label}; saved {output}.png', flush=True)
    return metrics


def main():
    calc = ProfileCalculator(model_path=MODEL_FILE, device=DEVICE, pbc_handling='slab', scf_restart=False,
        external_field_key='external_field', total_charge_key='total_charge', ignore_nonconverged=True,
        scf_options=dict(num_scf_steps=SCF_STEPS, constant_charge=True, use_autograd_forces=True,
                         initial_density='local_guess', initial_fermi_level='zero'))
    if ENABLE_CUEQ:
        accelerate_backbone(calc.model, DEVICE)
    calc.model.field_dependent_charges_map.deployment_mode = 'unroll_scf'
    calc.model.eval().requires_grad_(False)
    frames = read(ATOMS_FILE, index=':')
    for index in FRAME_INDEXES:
        plot_frame(frames[index], calc, index)


if __name__ == '__main__':
    main()
