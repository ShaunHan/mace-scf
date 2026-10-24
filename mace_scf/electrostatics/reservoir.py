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
import torch

SHE_ABSOLUTE_V = 4.44  # Trasatti's absolute SHE potential at 298.15 K, IUPAC.


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
