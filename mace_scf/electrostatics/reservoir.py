"""Vacuum-referenced charging work for fixed-cell SHE-controlled dynamics.

The fitted energy describes the specified electrode/counter-charge calculation.
Its neutral-path derivative includes work on the artificial counter source.
For a vacuum-referenced electron reservoir use

    A(R,Q) = E(R,0,0) + integral_0^Q WF(R,q,-q) dq.

Thus dA/dQ=WF, and the grand potential is A-(U_SHE+4.44)*Q, in eV/e
units. Forces MUST differentiate A, including the charging-work correction.
This is an inference energy definition, not a refit or a change of DFT labels.
A genuine solution-side vacuum reference and accurate charged predictions are
still required. No temperature/pH correction or electrolyte model is inferred.
"""
import torch

SHE_ABSOLUTE_V = 4.44  # Conventional room-temperature absolute SHE reference.


def charging_work(evaluate, charge, positions):
    """Nested Simpson integration with energy, force and endpoint-voltage checks.

    evaluate(q) must retain derivatives in both q and positions. Refinement
    reuses all electronic evaluations; the caller can share one MACE backbone.
    The reported voltage differentiates the SAME numerical energy as the forces.
    """
    values = {}

    def at(fraction):
        if fraction not in values:
            values[fraction] = evaluate(charge*fraction)
        return values[fraction]

    neutral, endpoint = at(0.), at(1.)
    wf = endpoint['vacuum_potential']-endpoint['fermi_level']
    previous = None
    for intervals in (2, 4, 8, 16):
        integral = torch.zeros_like(wf)
        for j in range(intervals+1):
            point = at(j/intervals)
            weight = 1 if j in (0, intervals) else (4 if j % 2 else 2)
            integral = integral+weight*(point['vacuum_potential']-point['fermi_level'])
        energy = neutral['energy']+charge.reshape(-1)*integral/(3*intervals)
        voltage, gradient = torch.autograd.grad(energy.sum(), (charge, positions), retain_graph=True)
        voltage, forces = voltage.reshape(-1), -gradient
        voltage_error = (voltage-wf).detach().abs().max()
        if previous is not None:
            energy_error = (energy.detach()-previous[0]).abs().max()
            force_error = (forces.detach()-previous[1]).abs().max()
            errors = torch.stack((voltage_error, energy_error, force_error))
            limits = errors.new_tensor([5.e-4, 1.e-4, 1.e-3])
            if bool(torch.isfinite(errors).all() & (errors <= limits).all()):
                output = dict(endpoint)
                for name in ('energy', 'charge_potential', 'counter_charge_potential', 'charging_potential'):
                    if name in output:
                        output['model_'+name] = output.pop(name)
                output.update(energy=energy, forces=forces, charging_potential=voltage,
                              electrode_potential=voltage-SHE_ABSOLUTE_V,
                              reservoir_energy_correction=energy-endpoint['energy'],
                              charge_integration_error=voltage_error.reshape(1),
                              charge_integration_force_error=force_error.reshape(1),
                              charge_evaluations=charge.new_tensor([len(values)]))
                if 'scf_residual' in endpoint:
                    output['scf_residual'] = torch.stack([v['scf_residual'] for v in values.values()]).amax(0)
                return output
        previous = energy.detach(), forces.detach()
    raise RuntimeError('Vacuum-referenced charging integral did not converge: '
                       f'voltage={float(voltage_error):.3g} V, energy={float(energy_error):.3g} eV, '
                       f'force={float(force_error):.3g} eV/A. No unconverged MD force was returned.')


@torch.enable_grad()
def evaluate_she(model, data, *, steps, mixing, tolerance):
    """Preserve the trained model; construct its reservoir energy only for MD."""
    from .coupled import evaluate_coupled
    response = model.field_dependent_charges_map
    if not getattr(response, 'energy_derivatives', False):
        raise ValueError('SHE dynamics requires an energy-derived MACE-VOLT export')
    if getattr(response, 'deployment_mode', 'unroll_scf') == 'implicit':
        raise ValueError('SHE charging integration currently requires a finite unrolled model')
    data = dict(data)
    data['positions'] = data['positions'].detach().clone().requires_grad_(True)
    charge = data['total_charge'].detach().clone().requires_grad_(True)
    counter = data.get('counter_charge', torch.zeros_like(charge))
    if not torch.allclose(counter.reshape(-1), -charge.reshape(-1), atol=1.e-8, rtol=0):
        raise ValueError('SHE dynamics requires counter_charge=-total_charge')
    vacuum = data.get('vacuum_fraction')
    if vacuum is None or not bool((torch.isfinite(vacuum) & (vacuum >= 0.) & (vacuum < 1.)).all()):
        raise ValueError('Supply the physical solution-side vacuum observation plane')
    local = model.local_part(data, compute_force=True)

    def evaluate(q):
        return evaluate_coupled(model, dict(data, total_charge=q, counter_charge=-q),
            steps=steps, mixing=mixing, tolerance=tolerance, mode='unroll_scf',
            training=True, compute_force=False, local_state=local)

    return charging_work(evaluate, charge, data['positions'])
