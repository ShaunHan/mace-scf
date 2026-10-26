"""Constant-potential observables for a cell with an explicit counter charge.

The electrode carries charge Q (in e) and the prescribed Gaussian source -Q,
so the cell stays neutral. One energy E(R, Q, Q_cc) gives

    F_i = -dE/dR_i,      mu = -dE/dQ|_Qcc,      e V = dE/dQ on Q_cc = -Q,

and V = (<U>_cc - mu)/e is the electrode Fermi level measured from the
potential at the counter charge: the voltage across the two electrodes in the
thermopotentiostat Hamiltonian H = K + E - e V* Q (Deissenbeck et al.,
PRL 126, 136803). It is exactly the charge derivative of the VASP-Python
plug-in energy. The absolute (SHE) electrode potential is the solution-side
work function, U_SHE = WF/e - 4.44 V, and

    WF - e V = <U>_vac - <U>_cc = Delta,

the potential step from the counter charge to the vacuum beyond the solution.
Delta contains no Fermi level. In the fixed-V* ensemble <WF> = e V* + <Delta>,
so V* = 4.44 V + U* - <Delta>/e imposes <U_SHE> = U*.
"""
import numpy as np
import torch

SHE_ABSOLUTE_V = 4.44  # Trasatti's absolute SHE potential at 298.15 K, IUPAC.
# Electrode elements that place the counter-charge sheet: the d block.
D_BLOCK = frozenset([*range(21, 31), *range(39, 49), *range(57, 81), *range(89, 113)])


def slab_controls(atoms, sheet_height=6., sheet_width=1., cut=None):
    """Dipole cut, vacuum plane and counter-charge sheet of a slab open along z, from its geometry.

    Unless given, the cut lies in the middle of the largest empty gap along z.
    The electrode surface is the outermost d-block atom or O bonded to one
    within 2.3 Angstrom, on the side that holds more of the other atoms, the
    solvent. The sheet sits sheet_height beyond that surface, and the vacuum
    plane halfway between the cut and the solvent surface, or the sheet if that
    lies further out. Compute once per trajectory.
    """
    z = atoms.get_scaled_positions(wrap=True)[:, 2]
    if cut is None:
        order = np.sort(z)
        gaps = np.diff(np.append(order, order[0]+1.))
        i = int(np.argmax(gaps))
        cut = (order[i]+.5*gaps[i]) % 1.
    length = atoms.cell[2, 2]
    height = ((z-cut) % 1.)*length   # from the cut upwards, through the system
    numbers = atoms.get_atomic_numbers()
    electrode = np.isin(numbers, list(D_BLOCK))
    if not electrode.any():
        raise ValueError('No d-block electrode atoms to place the counter-charge sheet')
    surface = list(np.flatnonzero(electrode))
    for o in np.flatnonzero(numbers == 8):
        if (atoms.get_distances(int(o), np.flatnonzero(electrode), mic=True) < 2.3).any():
            surface.append(int(o))
    top, bottom = height[surface].max(), height[surface].min()
    if (height > top).sum() >= (height < bottom).sum():
        sheet = top+sheet_height
        vacuum = .5*(max(height.max(), sheet+4*sheet_width)+length)
    else:
        sheet = bottom-sheet_height
        vacuum = .5*min(height.min(), sheet-4*sheet_width)
    return {'dipole_correction_zfrac': float(cut), 'vacuum_zfrac': float((cut+vacuum/length) % 1.),
            'counter_charge_zfrac': float((cut+sheet/length) % 1.), 'counter_charge_width': float(sheet_width)}


def evaluate_cell(model, data, *, steps, mixing, tolerance):
    """E, F, cell voltage, WF and the SHE offset from ONE electronic trajectory."""
    from .coupled import evaluate_coupled
    response = model.field_dependent_charges_map
    if (not getattr(response, 'energy_derivatives', False)
            or getattr(response, 'energy_reference', 'dft') != 'dft'):
        raise ValueError('Constant-potential MD requires a model trained with the DFT energy '
                         'and derivative-based Fermi levels (energy_derivatives=True)')
    charge = data['total_charge'].reshape(-1)
    counter = data.get('counter_charge', torch.zeros_like(charge)).reshape(-1)
    if not torch.allclose(counter, -charge, atol=1.e-10, rtol=0):
        raise ValueError('Constant-potential evaluation requires counter_charge = -total_charge')
    width = data.get('counter_charge_width', torch.zeros_like(charge)).reshape(-1)
    if bool((width <= 0).any()):
        raise ValueError('Constant-potential evaluation needs an explicit Gaussian counter-charge '
                         'profile (width and sheet fraction or centre), also at Q = 0')
    output = evaluate_coupled(model, data, steps=steps, mixing=mixing,
                              tolerance=tolerance, compute_force=True)
    output['electrode_potential_she'] = output['work_function']-SHE_ABSOLUTE_V
    return output
