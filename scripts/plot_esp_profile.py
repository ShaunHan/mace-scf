#!/usr/bin/env python3
"""Plot the plane-averaged ESP of MACE-VOLT against DFT for XYZ frames with stored Fourier labels.

DFT: the deformation potential LOCPOT - LOCPOT_proto (info key vasp_phi_fftn_shifted) and
the neutral-atom potential LOCPOT_proto (vasp_proto_phi_fftn_shifted), unsmoothed Fourier
modes written by scripts/collect_vasp_data.py. Model: its deformation potential. Both are
averaged over the planes of the slab on the Fourier modes they share, raw and with the
model's Gaussian smearing, as in the potential loss. The total ESP adds the same neutral
atoms to both and is referenced to each Fermi level, so its vacuum plateau is the work
function. Edit the settings below; one PNG/PDF/SVG/CSV set is saved per selected index.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
from ase.io import read
from ase.visualize.plot import plot_atoms
from mace_scf.calculators.fixedpoint_scf import MACEFixedPointSCF

ATOMS_FILE = 'IrO2OH_110_mdinit_0_50_vasp_sp.xyz'
MODEL_FILE = '/fs/cephfs/data/qm_inorganics/shuang/electrocatalysis/IrO2_110/ml_training/wf_training_186/fit_0_stage4_finetune_v387.model'
FRAME_INDEXES = list(range(10, 20))
DEVICE = 'cuda:0'
POTENTIAL_KEY, PROTO_KEY, FERMI_KEY = 'vasp_phi_fftn_shifted', 'vasp_proto_phi_fftn_shifted', 'vasp_efermi'
OUTPUT_DIR, DPI = Path('esp_profiles'), 300


class ProfileCalculator(MACEFixedPointSCF):
    def _build_results(self, output, diagnostics):
        results = super()._build_results(output, diagnostics)
        for key in ('fourier_potential', 'k_vectors_grid_shape', 'k_vectors_mask'):
            results[key] = output[key].detach().cpu().numpy()
        return results


def planar_modes(coefficients, keep):
    """Modes m of the plane average sum_m c_m exp(2 pi i m s), s the fractional height.

    coefficients is an unnormalized FFT grid in native order, c along its third axis.
    keep(m) selects modes; an even grid's Nyquist mode is ambiguous and never kept.
    """
    n = coefficients.shape[2]
    modes = np.rint(np.fft.fftfreq(n)*n).astype(int)
    column = coefficients[0, 0]/coefficients.size
    return {int(m): c for m, c in zip(modes, column) if abs(m) <= (n-1)//2 and keep(m)}


def profile(modes, s):
    return sum((c*np.exp(2j*np.pi*m*s)).real for m, c in modes.items())


def profiles(atoms, results, sigma):
    """Curves on z in [-L/2, L/2] about the middle of the continuous branch, and RMSEs."""
    length = float(atoms.cell[2, 2])
    shape = tuple(int(n) for n in np.ravel(results['k_vectors_grid_shape']))
    grid = np.asarray(results['fourier_potential']).reshape(*shape, 2)
    inside = np.asarray(results['k_vectors_mask']).reshape(shape)[0, 0]
    model = planar_modes(grid[..., 0]+1j*grid[..., 1], lambda m: inside[m % shape[2]])

    def stored(key):
        return planar_modes(np.asarray(atoms.info[key][0])+1j*np.asarray(atoms.info[key][1]), lambda m: m in model)
    dft, proto = stored(POTENTIAL_KEY), stored(PROTO_KEY)
    model = {m: model[m] for m in dft}
    smear = {m: np.exp(-.5*(sigma*2*np.pi*m/length)**2) for m in dft}
    center = (float(atoms.info.get('dipole_correction_zfrac', .5))+.5) % 1.
    z = np.linspace(-length/2, length/2, 1201)
    s = center+z/length
    fermi = {'dft': float(atoms.info[FERMI_KEY]), 'model': float(results['fermi_level'])}
    curves = {'z_A': z}
    for name, deformation in (('dft', dft), ('model', model)):
        for kind, weight in (('raw', lambda m: 1.), ('smeared', smear.get)):
            curves[f'{name}_deformation_{kind}_V'] = profile({m: weight(m)*c for m, c in deformation.items()}, s)
            curves[f'{name}_total_{kind}_V'] = (profile({m: weight(m)*(c+proto[m]) for m, c in deformation.items()}, s)
                                               - fermi[name])
    # Parseval: the RMSE over the cell of a plane average from its Fourier modes.
    errors = {kind: float(np.sqrt(sum(abs(weight(m)*(model[m]-dft[m]))**2 for m in dft)))
              for kind, weight in (('raw', lambda m: 1.), ('smeared', smear.get))}
    errors['fermi'] = fermi['model']-fermi['dft']
    work = {'model': float(results['work_function'])}
    if 'vacuum_potential' in atoms.info:
        work['dft'] = float(atoms.info['vacuum_potential'])-fermi['dft']
    return curves, dict(center=center, errors=errors, work=work, kmax=2*np.pi*max(dft)/length)


def draw_structure(atoms, axis, center):
    length = float(atoms.cell[2, 2])
    display = atoms.copy()
    display.positions[:, 2] += (.5-center)*length
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


def save_plot(atoms, curves, summary, sigma, index):
    center, errors, work = summary['center'], summary['errors'], summary['work']
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'], 'svg.fonttype': 'none'})
    figure, axes = plt.subplots(3, 1, figsize=(8.2, 8.1), sharex=True,
                                gridspec_kw=dict(height_ratios=(1.15, 2.4, 2.4), hspace=.16))
    draw_structure(atoms, axes[0], center)
    axes[0].set_title(f'Configuration {index}', fontsize=11)
    z = curves['z_A']
    smeared = rf'smeared ($\sigma={sigma:g}$ $\AA$)'
    for axis, kind, letter in zip(axes[1:], ('total', 'deformation'), ('a', 'b')):
        for name, color, label in (('dft', '#2166ac', 'DFT'), ('model', '#e68626', 'MACE-VOLT')):
            axis.plot(z, curves[f'{name}_{kind}_raw_V'], '--', color=color, lw=1., label=f'{label} raw')
            axis.plot(z, curves[f'{name}_{kind}_smeared_V'], '-', color=color, lw=1.8, label=f'{label} {smeared}')
        axis.text(.015, .94, letter, transform=axis.transAxes, fontweight='bold', va='top', fontsize=13)
        axis.legend(fontsize=8, loc='best', frameon=False)
        axis.set_xlim(z[0], z[-1])
    if 'vacuum_zfrac' in atoms.info:
        axes[1].axvline(((float(atoms.info['vacuum_zfrac'])-center+.5) % 1.-.5)*float(atoms.cell[2, 2]),
                        color='0.6', lw=.8, ls=':')
    # The vacuum levels themselves: the model's exact level (no Fourier series) and the raw DFT label, both flat.
    for name, color in (('dft', 'C0'), ('model', 'C1')):
        if name in work:
            axes[1].axhline(work[name], color=color, lw=.8, ls=':')
    axes[1].set_title('Work function: '+', '.join(f'{label} {work[name]:.3f} eV' for name, label in
                                                  (('dft', 'DFT'), ('model', 'MACE-VOLT')) if name in work)
                      + rf'; Fourier modes $|k|\leq{summary["kmax"]:.2f}$ $\AA^{{-1}}$', fontsize=9)
    axes[1].set_ylabel(r'Total ESP $\phi-E_\mathrm{F}$ (V)')
    axes[2].set_ylabel(r'Deformation ESP $\Delta\phi$ (V)')
    axes[-1].set_xlabel(r'$z$ ($\AA$)')
    figure.subplots_adjust(left=.13, right=.98, bottom=.08, top=.94)
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    output = OUTPUT_DIR/f'esp_profile_{index}'
    for extension in ('.png', '.pdf', '.svg'):
        figure.savefig(output.with_suffix(extension), dpi=DPI, bbox_inches='tight')
    np.savetxt(output.with_suffix('.csv'), np.column_stack(list(curves.values())), delimiter=',',
               header=','.join(curves), comments='')
    plt.close(figure)
    print(f'Frame {index}: deformation ESP RMSE raw {1000*errors["raw"]:.1f} mV, smeared {1000*errors["smeared"]:.1f} '
          f'mV; Fermi level error {1000*errors["fermi"]:+.1f} meV; saved {output}', flush=True)


def plot_frame(atoms, calculator, index):
    if not np.array_equal(atoms.pbc, [True, True, False]) or atoms.cell[2, 2] <= 0 or not np.allclose(
            [*atoms.cell[:2, 2], *atoms.cell[2, :2]], 0.):
        raise ValueError('ESP z profiles require an xy-periodic slab with a perpendicular open z axis')
    model_atoms = atoms.copy()
    model_atoms.set_constraint([])
    model_atoms.calc = calculator
    model_atoms.get_potential_energy()
    sigma = float(calculator.model.field_dependent_charges_map.potential_amplitudes_smearing_width)
    curves, summary = profiles(atoms, calculator.results, sigma)
    save_plot(atoms, curves, summary, sigma, index)
    return summary['errors']


def main():
    calculator = ProfileCalculator(model_path=MODEL_FILE, device=DEVICE, pbc_handling='slab', scf_restart=False,
                                   external_field_key='external_field', total_charge_key='total_charge')
    calculator.model.eval().requires_grad_(False)
    frames = read(ATOMS_FILE, index=':')
    for index in FRAME_INDEXES:
        plot_frame(frames[index], calculator, index)


if __name__ == '__main__':
    main()
