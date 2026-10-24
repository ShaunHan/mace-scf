"""Conservative charge-transfer ensembles, with one electronic trajectory.

NonlinearEnergyResponse obtains vacuum-reference work algebraically from its
shared energy/ESP coefficients and supports a sheet inside the solvent. The
separated-vacuum functions below retain compatibility with earlier exports.

The prescribed Gaussian sheet lies beyond all atoms; an outer vacuum plateau
lies beyond that sheet. A sharp, uniform reference plane has zero spread self
energy. For a separated Gaussian of width sigma, restoring that plane adds
K*sigma*Q**2/(2*area*sqrt(pi)), K=1/epsilon_0 in eV Angstrom/e**2.
This is an electrostatic boundary term, not a fitted voltage offset. It assumes
negligible density overlap: check DFT density and ESP plateaus, not just atoms.
This uniform-plane boundary is not a full image-charge metallic ESM solver.
"""
import math

import torch

SHE_ABSOLUTE_V = 4.44  # Conventional room-temperature absolute SHE reference.


def vacuum_boundary(data):
    """Validate explicitly marked frames; return their mask and self coefficient.

    Unmarked neutral/charged DFT configurations keep their original convention.
    Positions and all three planes must use the same dipole-cut branch. Only
    the positive open direction is supported, towards the solution-side vacuum.
    """
    from .potential import FIELD_CONSTANT
    q = data['total_charge'].reshape(-1)
    marked = data.get('vacuum_reservoir', torch.zeros_like(q, dtype=torch.bool)).reshape(-1).bool()
    if not bool(marked.any()):
        return marked, torch.zeros_like(q)
    pbc = data['pbc'].reshape(-1, 3).bool()
    axes = (~pbc).long().argmax(-1)
    counter = data.get('counter_charge', torch.zeros_like(q)).reshape(-1)
    sheet_axis = data.get('counter_charge_axis', torch.full_like(axes, -1)).reshape(-1)
    sigma = data.get('counter_charge_width', torch.zeros_like(q)).reshape(-1)
    required = ('counter_charge_fraction', 'vacuum_fraction', 'dipole_correction_fraction')
    if any(name not in data for name in required):
        raise ValueError('Vacuum reservoir requires explicit counter, vacuum and dipole-correction planes')
    valid = ((pbc.sum(-1)==2) & (sheet_axis==axes) & (sigma>0)
             & ((counter+q).abs()<1.e-8) & torch.isfinite(sigma))
    field = data.get('external_field', q.new_zeros((len(q), 3))).reshape(-1, 3)
    valid &= torch.isfinite(field).all(-1) & (field.abs().amax(-1)<1.e-12)
    if 'counter_slab_bounds' in data or not bool(valid[marked].all()):
        raise ValueError('Vacuum reservoir requires a Gaussian slab-parallel sheet, C=-Q and zero external field')
    cell = data['cell'].reshape(-1, 3, 3)
    rows = torch.arange(len(q), device=q.device)
    normal = torch.linalg.cross(cell[rows, (axes+1)%3], cell[rows, (axes+2)%3])
    area = torch.linalg.vector_norm(normal, dim=-1)
    length = (cell[rows, axes]*normal/area[:, None]).sum(-1)
    if not bool((torch.isfinite(length[marked]) & (length[marked]>0)).all()):
        raise ValueError('Vacuum reservoir requires a finite, right-handed slab cell')
    branch = data['dipole_correction_fraction'].reshape(-1)
    if bool((marked & ((branch.remainder(1.)-.5).abs()>1.e-12)).any()):
        raise ValueError('Vacuum reservoir currently requires dipole_correction_fraction=0.5, '
                         'with the explicit slab rejoined about the cell origin, as in the VASP template')
    sheet = (data['counter_charge_fraction'].reshape(-1)-branch).remainder(1.)*length
    vacuum = (data['vacuum_fraction'].reshape(-1)-branch).remainder(1.)*length
    batch = data['batch']
    fractional = torch.einsum('ni,nij->nj', data['positions'], torch.linalg.inv(cell)[batch])
    z = ((fractional.gather(-1, axes[batch, None]).squeeze(-1)-branch[batch]).remainder(1.)
         *length[batch])
    for graph in torch.nonzero(marked).flatten().tolist():
        top = z[batch==graph].max()
        # Six sheet sigmas control Gaussian overlap; two Angstrom additionally
        # exclude an atomically empty but electron-rich surface tail.
        ok = ((sheet[graph]-top >= 6*sigma[graph]+2.)
              & (vacuum[graph]-sheet[graph] >= 6*sigma[graph])
              & (length[graph]-vacuum[graph] >= 2*sigma[graph]))
        if not bool(ok):
            raise ValueError('Vacuum reservoir geometry: require atoms < sheet < outer vacuum < dipole cut; '
                             'allow at least 6*sigma+2 Angstrom from atoms to sheet, 6*sigma to vacuum, '
                             'and 2*sigma from vacuum to cut. Add vacuum in DFT if needed.')
    alpha = FIELD_CONSTANT*sigma/(area*math.sqrt(math.pi))
    return marked, torch.where(marked, alpha, torch.zeros_like(alpha))


def _evaluate(model, data, *, steps, mixing, tolerance):
    from .potential import evaluate_electronic
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
    return evaluate_electronic(model, data, steps=steps, mixing=mixing,
                            tolerance=tolerance, compute_force=True)


def evaluate_pair(model, data, *, steps, mixing, tolerance):
    """Original DFT energy and its electrode/counter transfer derivative."""
    output = _evaluate(model, data, steps=steps, mixing=mixing, tolerance=tolerance)
    output['md_charge_potential'] = output['charging_potential']
    output['electrode_potential'] = output['work_function']-SHE_ABSOLUTE_V
    return output


def evaluate_she(model, data, *, steps, mixing, tolerance):
    """Return the vacuum-referenced energy and ALL its nuclear forces.

    For the shared nonlinear model, dE_MD/dQ=WF and the controlled potential
    is WF-4.44. Omega=E_MD-(4.44+U)*Q. Its reference work generally changes
    forces; no force term is detached. Training retains one DFT energy label.
    """
    response = model.field_dependent_charges_map
    if getattr(response, 'shared_nonlinear_energy', False):
        from .nonlinear_energy import evaluate_nonlinear
        return evaluate_nonlinear(model, data, steps=steps, mixing=mixing,
                                  tolerance=tolerance, compute_force=True, reservoir=True)
    if not getattr(response, 'vacuum_reservoir_trained', False):
        raise ValueError('SHE control needs training/finetuning with charged vacuum_reservoir=True DFT '
                         'frames and Fermi labels. An old model cannot be converted by changing its metadata.')
    q = data['total_charge'].reshape(-1)
    data = dict(data, vacuum_reservoir=torch.ones_like(q, dtype=torch.bool))
    output = _evaluate(model, data, steps=steps, mixing=mixing, tolerance=tolerance)
    output['dft_energy'] = output['energy']
    output['energy'] = output['energy']+output['reservoir_energy_correction']
    output['md_charge_potential'] = output['work_function']-SHE_ABSOLUTE_V
    output['electrode_potential'] = output['md_charge_potential']
    return output
