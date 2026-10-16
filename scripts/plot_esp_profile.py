#!/usr/bin/env python3
"""Plot an aligned structure strip, total ESP and deformation ESP per XYZ index.

Raw total DFT: locpots/LOCPOT_<index> (LOCPOT<index> also accepted), from LVHAR.
Raw protoatomic reference: locpots/LOCPOT_proto_<index> (required).
Raw deformation is the direct grid difference, never an inverse-smoothed FFT.
DFT is used for comparison only; the model must contain a fitted baseline.
Edit the settings below; one PNG/PDF/SVG/CSV set is saved per selected index.
"""
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0].strip()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from ase.geometry import find_mic
from ase.io import read
from ase.io.vasp import read_vasp_configuration
from ase.visualize.plot import plot_atoms
from scipy.signal import resample
from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF
from mace_scf.utils.foundation import accelerate_backbone

ATOMS_FILE = 'IrO2OH_110_mdinit_0_50_vasp_sp.xyz'
MODEL_FILE = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/ml_training/wf_training_186/fit_0_stage4_finetune_v387.model'
FRAME_INDEXES = list(range(10, 20))
DEVICE, ENABLE_CUEQ, SCF_STEPS = 'cuda:0', True, 50
SIGMA_PHI = 1.5  # Å; display smoothing applied to both DFT profiles.
MODEL_TARGET_SIGMA = 1.5  # Legacy checkpoint fallback; decoded fields and new proto fits are detected automatically.
LOCPOT_DIR, OUTPUT_DIR, DPI = Path('locpots'), Path('esp_profiles'), 300


class ProfileCalculator(MACEFixedPointSCF):
    def _build_results(self, output, diagnostics):
        results = super()._build_results(output, diagnostics)
        for key in ('fourier_potential', 'fourier_total_potential', 'k_vectors_grid_shape', 'k_vectors_mask'):
            results[key] = output[key].detach().cpu().numpy()
        results['proto_fitted'] = bool(self.model.field_dependent_charges_map.proto_fitted)
        response = self.model.field_dependent_charges_map
        decoder = getattr(response, 'raw_decoder', None)
        results['model_target_sigma'] = 0. if decoder is not None and bool(decoder.enabled[1]) else MODEL_TARGET_SIGMA
        width = getattr(response, 'proto_smearing_width', None)
        results['proto_target_sigma'] = MODEL_TARGET_SIGMA if width is None else width
        return results


def scalar(value):
    a = np.asarray(value, dtype=float)
    if a.size != 1 or not np.isfinite(a).all():
        raise ValueError('Expected one finite scalar')
    return float(a.item())


def locpot_path(index, proto=False):
    prefix = 'LOCPOT_proto' if proto else 'LOCPOT'
    paths = [LOCPOT_DIR/f'{prefix}{separator}{index}' for separator in ('_', '')]
    found = [p for p in paths if p.is_file()]
    if len(found) > 1:
        raise ValueError(f'Ambiguous files for index {index}: {found}')
    if found:
        return found[0]
    raise FileNotFoundError(f'Expected {paths[0]} or {paths[1]}')


def read_locpot(path, atoms):
    """Read the scalar LVHAR block directly, without CHGCAR volume scaling."""
    with Path(path).open() as stream:
        structure = read_vasp_configuration(stream)
        if not np.allclose(structure.cell, atoms.cell, atol=1.e-5, rtol=0.) or not np.array_equal(
                np.sort(structure.numbers), np.sort(atoms.numbers)):
            raise ValueError(f'{path}: cell/species do not match the XYZ frame')
        # VASP may reorder species. Compare periodic coordinates independently of order.
        for number in np.unique(atoms.numbers):
            first, second = structure.positions[structure.numbers == number], atoms.positions[atoms.numbers == number]
            distances = find_mic((first[:, None]-second).reshape(-1, 3), atoms.cell, pbc=True)[1]
            distances = distances.reshape(len(first), len(second))
            if max(distances.min(0).max(), distances.min(1).max()) > 1.e-3:
                raise ValueError(f'{path}: atomic positions do not match the XYZ frame')
        line = stream.readline()
        while line and not line.strip():
            line = stream.readline()
        shape = tuple(map(int, line.split()))
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError(f'{path}: invalid LOCPOT grid')
        profile = np.empty(shape[2])
        for iz in range(shape[2]):
            plane = np.fromfile(stream, count=shape[0]*shape[1], sep=' ')
            if plane.size != shape[0]*shape[1] or not np.isfinite(plane).all():
                raise ValueError(f'{path}: incomplete/nonfinite potential grid')
            profile[iz] = plane.mean()
    # VASP electron potential energy in eV/e: numerical values in V, same sign as training.
    return profile


def profiles(atoms, results, raw_total, raw_proto):
    """Compare actual grid profiles with the independently predicted spectrum."""
    if not results['proto_fitted']:
        raise ValueError('Total ESP requires a model with a fitted protoatomic baseline')
    potential_sigma = results.get('model_target_sigma', MODEL_TARGET_SIGMA)
    proto_sigma = results.get('proto_target_sigma', MODEL_TARGET_SIGMA)
    if SIGMA_PHI < max(potential_sigma, proto_sigma):
        raise ValueError('A plot cannot undo checkpoint target smoothing; lower MODEL_TARGET_SIGMA only for a raw-trained model')
    length = float(atoms.cell[2, 2])
    shape = tuple(np.asarray(results['k_vectors_grid_shape'], dtype=int).ravel())
    modes = np.rint(np.fft.fftfreq(shape[2])*shape[2]).astype(int)
    mask = np.asarray(results['k_vectors_mask'], dtype=bool).reshape(shape)[0, 0]
    wave = 2*np.pi*modes[mask]/length
    model_filter = np.exp(-.5*(SIGMA_PHI**2-potential_sigma**2)*wave**2)
    proto_filter = np.exp(-.5*(SIGMA_PHI**2-proto_sigma**2)*wave**2)
    predicted = []
    for key in ('fourier_potential', 'fourier_total_potential'):
        a = np.asarray(results[key]).reshape(*shape, 2)
        predicted.append((a[0, 0, mask, 0]+1j*a[0, 0, mask, 1])/np.prod(shape))
    model_delta = predicted[0]*model_filter
    model_total = model_delta+(predicted[1]-predicted[0])*proto_filter
    model_delta = model_delta*(modes[mask] != 0)
    fermi_key = 'fermi_level' if 'fermi_level' in atoms.info else 'vasp_efermi'
    fermi = scalar(atoms.info[fermi_key])
    reference = scalar(results['potential_reference'])-scalar(results['fermi_level'])
    z = np.linspace(-length/2, length/2, max(1201, len(raw_total)+1))
    phase = np.exp(1j*np.outer(wave, z))
    def interpolate(values):
        return np.interp(z % length, np.arange(len(values))*length/len(values), values, period=length)
    deformation = raw_total-resample(raw_proto, len(raw_total))
    deformation -= deformation.mean()
    grid_wave = 2*np.pi*np.fft.fftfreq(len(raw_total), d=length/len(raw_total))
    def smooth(values):
        return np.fft.ifft(np.fft.fft(values)*np.exp(-.5*SIGMA_PHI**2*grid_wave**2)).real
    curves = dict(z_A=z, model_total_V=(model_total@phase).real+reference,
                  model_deformation_V=(model_delta@phase).real,
                  dft_total_raw_V=interpolate(raw_total)-fermi,
                  dft_total_smoothed_V=interpolate(smooth(raw_total))-fermi,
                  dft_deformation_raw_V=interpolate(deformation),
                  dft_deformation_smoothed_V=interpolate(smooth(deformation)))
    # Report the errors of the curves actually plotted, without double-counting
    # the periodic endpoint. The raw DFT curves retain their full grid resolution.
    errors = {f'{kind}_rmse_V': float(np.sqrt(np.mean((curves[f'model_{kind}_V'][:-1]
               -curves[f'dft_{kind}_smoothed_V'][:-1])**2))) for kind in ('total','deformation')}
    return curves, 'Raw DFT', errors


def draw_structure(atoms, axis):
    length = float(atoms.cell[2, 2])
    display = atoms.copy()
    display.positions[:, 2] += length/2
    display.wrap(pbc=True)
    display.set_celldisp([0., 0., 0.])
    # Project z horizontally and y vertically; retain physical Angstrom scale.
    rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    cell_y = np.array([0., display.cell[0, 1], display.cell[1, 1], sum(display.cell[:2, 1])])
    ymin, ymax = min(cell_y.min(), display.positions[:, 1].min())-1., max(cell_y.max(), display.positions[:, 1].max())+1.
    plot_atoms(display, ax=axis, rotation=rotation, radii=.65, scale=1., show_unit_cell=2,
               bbox=(0., ymin, length, ymax))
    for patch in axis.patches:
        patch.set_transform(Affine2D().translate(-length/2, 0.)+axis.transData)
    axis.set_xlim(-length/2, length/2)
    axis.set_ylim(0., ymax-ymin)
    axis.set_aspect('equal', adjustable='datalim')


def save_plot(atoms, curves, raw_label, errors, index):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 8.1), sharex=True,
        gridspec_kw=dict(height_ratios=(1.15, 2.4, 2.4), hspace=.16))
    draw_structure(atoms, axes[0])
    axes[0].set_title(f'Configuration {index}', fontsize=11)
    z = curves['z_A']
    sigma = rf' ($\sigma_\phi={SIGMA_PHI:g}$ $\AA$)'
    for axis, kind, letter in zip(axes[1:], ('total', 'deformation'), ('a', 'b')):
        key = f'dft_{kind}_raw_V'
        axis.plot(z, curves[key], '--', color='#2166ac', lw=1.3, label=raw_label)
        axis.plot(z, curves[f'dft_{kind}_smoothed_V'], '-', color='#2166ac', lw=1.8, label='Smoothed DFT'+sigma)
        axis.plot(z, curves[f'model_{kind}_V'], '-', color='#e68626', lw=1.8, label='Smoothed MACE-VOLT'+sigma)
        axis.text(.015, .94, letter, transform=axis.transAxes, fontweight='bold', va='top', fontsize=13)
        axis.set_ylabel(r'Total ESP $\phi$ (V)'+'\nFermi-referenced' if kind == 'total'
                        else r'Deformation ESP $\Delta\phi$ (V)'+'\nzero mean')
        axis.legend(fontsize=8, loc='best', frameon=False)
        axis.set_xlim(z[0], z[-1])
        axis.grid(False)
    axes[-1].set_xlabel(r'$z$ ($\AA$)')
    figure.subplots_adjust(left=.13, right=.98, bottom=.08, top=.94)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output = OUTPUT_DIR/f'esp_profile_{index}'
    for extension in ('.png', '.pdf', '.svg'):
        figure.savefig(output.with_suffix(extension), dpi=DPI, bbox_inches='tight')
    np.savetxt(output.with_suffix('.csv'), np.column_stack(list(curves.values())), delimiter=',',
               header=','.join(curves), comments='')
    plt.close(figure)
    print(f'Frame {index}: plotted-profile RMSE total={1000*errors["total_rmse_V"]:.2f} mV, '
          f'deformation={1000*errors["deformation_rmse_V"]:.2f} mV; saved {output}', flush=True)


def plot_frame(atoms, calculator, index):
    if not np.array_equal(atoms.pbc, [True, True, False]) or atoms.cell[2, 2] <= 0 or not np.allclose(
            [*atoms.cell[:2, 2], *atoms.cell[2, :2]], 0.):
        raise ValueError('ESP z profiles require an xy-periodic slab with a perpendicular open z axis')
    raw = read_locpot(locpot_path(index), atoms)
    proto_path = locpot_path(index, proto=True)
    proto = read_locpot(proto_path, atoms)
    model_atoms = atoms.copy()
    model_atoms.info = {key: atoms.info[key] for key in (
        'external_field', 'total_charge', 'vacuum_zfrac', 'dipole_correction_zfrac',
        'counter_charge', 'counter_charge_width', 'counter_charge_center',
        'counter_charge_xfrac', 'counter_charge_yfrac', 'counter_charge_zfrac') if key in atoms.info}
    model_atoms.set_constraint([])
    model_atoms.calc = calculator
    model_atoms.get_potential_energy()
    curves, raw_label, errors = profiles(atoms, calculator.results, raw, proto)
    save_plot(atoms, curves, raw_label, errors, index)
    return errors


def main():
    calc = ProfileCalculator(model_path=MODEL_FILE, device=DEVICE, pbc_handling='slab', scf_restart=False,
        external_field_key='external_field', total_charge_key='total_charge', ignore_nonconverged=True,
        scf_options=dict(num_scf_steps=SCF_STEPS, constant_charge=True, use_autograd_forces=True,
                         initial_density='local_guess', initial_fermi_level='zero'))
    if not bool(calc.model.field_dependent_charges_map.proto_fitted):
        raise ValueError('Use the v387 export containing the fitted protoatomic baseline')
    if ENABLE_CUEQ:
        accelerate_backbone(calc.model, DEVICE)
    calc.model.field_dependent_charges_map.deployment_mode = 'unroll_scf'
    calc.model.eval().requires_grad_(False)
    frames = read(ATOMS_FILE, index=':')
    for index in FRAME_INDEXES:
        plot_frame(frames[index], calc, index)


if __name__ == '__main__':
    main()
