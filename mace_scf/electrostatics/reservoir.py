"""SHE control of a learned vacuum-referenced energy; one charge state per call."""
import torch

SHE_ABSOLUTE_V = 4.44  # Conventional room-temperature absolute SHE reference.


def evaluate_she(model, data, *, steps, mixing, tolerance):
    """A single ordinary evaluation supplies A, -dA/dR and dA/dQ along C=-Q.

    New models train EF=ESP_vac-dA/dQ. No quadrature, refit or neutral-state
    evaluation is performed. A physical solution-side vacuum reference remains
    necessary; this construction does not establish charged prediction accuracy.
    """
    from .coupled import evaluate_coupled
    response = model.field_dependent_charges_map
    if (not getattr(response, 'energy_derivatives', False)
            or getattr(response, 'energy_reference', 'dft') != 'vacuum'):
        raise ValueError('SHE dynamics requires a model TRAINED with the vacuum-energy definition; '
                         'legacy DFT-energy exports cannot be converted by toggling a flag')
    if getattr(response, 'deployment_mode', 'unroll_scf') != 'unroll_scf':
        raise ValueError('SHE dynamics currently requires the finite unrolled energy model')
    charge = data['total_charge']
    counter = data.get('counter_charge', torch.zeros_like(charge))
    if not torch.allclose(counter.reshape(-1), -charge.reshape(-1), atol=1.e-8, rtol=0):
        raise ValueError('SHE dynamics requires counter_charge=-total_charge')
    vacuum = data.get('vacuum_fraction')
    if vacuum is None or not bool((torch.isfinite(vacuum) & (vacuum >= 0.) & (vacuum < 1.)).all()):
        raise ValueError('Supply the physical solution-side vacuum observation plane')
    output = evaluate_coupled(model, data, steps=steps, mixing=mixing,
                              tolerance=tolerance, compute_force=True)
    output['electrode_potential'] = output['charging_potential']-SHE_ABSOLUTE_V
    return output
