"""Conservative electrode/counter-charge transfer using the DFT energy."""
import torch

SHE_ABSOLUTE_V = 4.44  # Conventional room-temperature absolute SHE reference.


def evaluate_pair(model, data, *, steps, mixing, tolerance):
    """One trajectory; E and -dE/dR stay in the supplied DFT convention.

    The controller's conjugate voltage is dE(R,Q,-Q)/dQ. The independently
    reported WF-4.44 is an electrode observation, not that transfer voltage.
    """
    from .coupled import evaluate_coupled
    response = model.field_dependent_charges_map
    if (not getattr(response, 'energy_derivatives', False)
            or getattr(response, 'energy_reference', 'dft') != 'dft'):
        raise ValueError('Charge-transfer MD requires a model TRAINED with DFT energy and derivative-based EF')
    if getattr(response, 'deployment_mode', 'unroll_scf') != 'unroll_scf':
        raise ValueError('Charge-transfer MD requires the finite unrolled energy model')
    charge = data['total_charge']
    counter = data.get('counter_charge', torch.zeros_like(charge))
    if not torch.allclose(counter.reshape(-1), -charge.reshape(-1), atol=1.e-8, rtol=0):
        raise ValueError('Charge-transfer MD requires counter_charge=-total_charge')
    output = evaluate_coupled(model, data, steps=steps, mixing=mixing,
                              tolerance=tolerance, compute_force=True)
    output['md_charge_potential'] = output['charging_potential']
    output['electrode_potential'] = output['work_function']-SHE_ABSOLUTE_V
    return output


def evaluate_she(*args, **kwargs):
    raise ValueError('Raw DFT energy with a changing counter-charge does not generally have WF as its '
                     'charge-transfer derivative. Use the physical pair voltage with evaluate_pair; '
                     'SHE control needs a specified reference-electrode/reservoir energy, not a fitted offset.')
