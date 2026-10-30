"""Validation diagnostics of MACE-VOLT slabs, logged at the DEBUG level.

Where the error sits by class of slab, where the error of a slab with a counter charge sits along
its normal, how the model charges at fixed geometry, what the electrolyte molecules carry, and
what law the network learned. Nothing here feeds back into training.
"""
import logging
import math

import numpy as np
import torch
from graph_longrange.utils import FIELD_CONSTANT

from mace_scf.electrostatics.reservoir import D_BLOCK

# Heights above the counter charge at which the planar potential error is read, relative to its
# value REFERENCE_DEPTH below the counter charge, inside the electrode.
PROFILE_HEIGHTS = (-8., -6., -4., -2., 0., 2., 4., 8., 15.)
REFERENCE_DEPTH = 10.
# The fixed-geometry charging response is measured on this many slabs with a counter charge per
# validation, by central differences of CHARGE_STEP along Q_cc = -Q.
CHARGING_FRAMES, CHARGE_STEP = 4, .05
# Regions of the induced charge, in A from the counter charge: the electrode side, the double layer
# around the counter charge, and the outer electrolyte.
REGION_EDGES = (-5., 3.)
# The net charges of the electrolyte molecules are read on this many slabs with a counter charge.
MOLECULE_FRAMES = 8
# The polarizability of a water molecule, A^3, against which the learned water law is read.
WATER_ALPHA = 1.45
# Coulomb's constant in eV A / e^2: a response in e A^2 / V times it is a polarizability in A^3.
K_COULOMB = FIELD_CONSTANT/(4*math.pi)
CLASSES = ('electrolyte', 'neutral slab', 'slab in a field', 'counter charge')


def slab_geometry(batch):
    """Per graph: slab flag, axis, unit normal, area and the length along the normal."""
    cell = batch.cell.reshape(-1, 3, 3)
    pbc = batch.pbc.reshape(-1, 3).bool()
    axis = (~pbc).long().argmax(-1)
    rows = torch.arange(batch.num_graphs, device=cell.device)
    normal = torch.linalg.cross(cell[rows, (axis+1) % 3], cell[rows, (axis+2) % 3])
    area = torch.linalg.vector_norm(normal, dim=-1)
    normal = normal/area[:, None]
    return pbc.sum(-1) == 2, axis, normal, area, (cell[rows, axis]*normal).sum(-1)


def _graph_scalar(batch, key, default=0.):
    value = getattr(batch, key, None)
    if value is None:
        return torch.full((batch.num_graphs,), default, dtype=batch.positions.dtype, device=batch.positions.device)
    return value.reshape(-1).to(batch.positions)


def slab_classes(model, batch):
    """Per graph: 'counter charge', 'slab in a field', 'neutral slab' (with d-block atoms) or 'electrolyte' (without)."""
    numbers = model.atomic_numbers[batch.node_attrs.argmax(-1)]
    metal = torch.tensor([int(z) in D_BLOCK for z in numbers.tolist()], device=numbers.device).long()
    metal = torch.zeros(batch.num_graphs, dtype=torch.long, device=numbers.device).index_add_(0, batch.batch, metal) > 0
    counter = _graph_scalar(batch, 'counter_charge')
    field = getattr(batch, 'external_field', None)
    field = (torch.zeros_like(counter) if field is None
             else torch.linalg.vector_norm(field.reshape(-1, 3).to(counter), dim=-1))
    return ['counter charge' if float(counter[i]) != 0. else 'electrolyte' if not bool(metal[i])
            else 'slab in a field' if float(field[i]) > 1e-8 else 'neutral slab' for i in range(batch.num_graphs)]


class SlabDiagnostics:
    """Collects per-slab rows during one evaluation pass and logs their summary."""

    def __init__(self, split):
        self.split = split
        self.records = []
        self.charging = []
        self.molecules = []
        self.law = None
        self.batch_classes = []

    def on_device(self, model, wrapper, batch, output):
        """With the batch on the device: its classes, the law summary once, the charging response and the
        molecular charges of a few slabs."""
        try:
            self.batch_classes = slab_classes(model, batch)
        except Exception as error:  # the diagnostics never stop a run
            logging.debug('%s slab classes failed: %r', self.split, error)
            self.batch_classes = ['slab']*batch.num_graphs
        if self.law is None:
            try:
                from mace_scf.electrostatics.coupled import describe_law
                self.law = describe_law(model, batch.to_dict())
            except Exception as error:
                logging.debug('%s law summary failed: %r', self.split, error)
                self.law = {}
        if len(self.charging) < CHARGING_FRAMES and output.get('density_coefficients') is not None:
            try:
                self.charging += charging_response(model, wrapper, batch, output, CHARGING_FRAMES-len(self.charging))
            except Exception as error:
                logging.debug('%s charging response failed: %r', self.split, error)
                self.charging += [None]*CHARGING_FRAMES
        if len(self.molecules) < MOLECULE_FRAMES and output.get('density_coefficients') is not None:
            try:
                self.molecules += molecule_charges(model, batch, output, MOLECULE_FRAMES-len(self.molecules))
            except Exception as error:
                logging.debug('%s molecular charges failed: %r', self.split, error)
                self.molecules += [None]*MOLECULE_FRAMES

    def rows(self, batch, output, error, weight):
        """Per slab with a work function: its class, errors, charge and planar potential error profile."""
        slab, axis, normal, area, length = slab_geometry(batch)
        charge = _graph_scalar(batch, 'total_charge')
        counter = _graph_scalar(batch, 'counter_charge')
        step = torch.full_like(error, float('nan'))
        if output.get('dipole') is not None and getattr(batch, 'dipole', None) is not None:
            delta = output['dipole'].reshape(-1, 3)-batch.dipole.reshape(-1, 3)
            step = FIELD_CONSTANT*(delta*normal).sum(-1)/area
        ef = (output['fermi_level'].reshape(-1)-batch.fermi_level.reshape(-1)
              if getattr(batch, 'fermi_level', None) is not None else torch.full_like(error, float('nan')))
        vac = (output['vacuum_potential'].reshape(-1)-batch.vacuum_potential.reshape(-1)
               if getattr(batch, 'vacuum_potential', None) is not None else torch.full_like(error, float('nan')))
        profile = planar_potential_errors(batch, output, slab, normal, length)
        classes = self.batch_classes if len(self.batch_classes) == batch.num_graphs else ['slab']*batch.num_graphs
        for i in torch.nonzero(weight > 0).flatten().tolist():
            self.records.append((float(weight[i]), float(error[i]), float(step[i]), float(charge[i]), float(counter[i]),
                                 float(ef[i]), float(vac[i]), profile[i], classes[i]))

    def log(self, objective=None, loss_weights=None):
        if objective:
            terms = ', '.join(f'{name}={float(loss_weights[name])*pair[0]/pair[1]:.4g}'
                              for name, pair in objective.items() if pair[1] > 0)
            logging.debug('%s loss terms, weight x mean: %s', self.split, terms)
        if self.records:
            self._log_records()
        done = [row for row in self.charging if row is not None]
        if done:
            self._log_charging(done)
        done = [row for row in self.molecules if row is not None]
        if done:
            self._log_molecules(done)
        if self.law:
            self._log_law()

    def _log_records(self):
        weight, wf, step, charge, counter, ef, vac = (np.array(column) for column in zip(*[r[:7] for r in self.records]))
        profile = np.array([r[7] for r in self.records])
        kind = np.array([r[8] for r in self.records])
        cc = counter != 0

        def rms(x, w):
            return 1000*math.sqrt(np.average(x*x, weights=w)) if len(x) else float('nan')
        for name in ('all',)+CLASSES:
            sel = np.ones_like(cc) if name == 'all' else kind == name
            if sel.sum() < 2:
                continue
            w, x, y = weight[sel], step[sel], wf[sel]
            use = np.isfinite(x)
            xc, yc = x[use]-np.average(x[use], weights=w[use]), y[use]-np.average(y[use], weights=w[use])
            xx, yy, xy = (np.average(v, weights=w[use]) for v in (xc*xc, yc*yc, xc*yc))
            logging.debug('%s %s slabs (n=%d): RMSE WF %.1f, EF %.1f, vacuum %.1f meV, mean WF error %+.1f meV; '
                          'dipole step %.1f mV, correlation with the WF error %.3f', self.split, name, int(sel.sum()),
                          rms(y, w), rms(ef[sel][np.isfinite(ef[sel])], w[np.isfinite(ef[sel])]),
                          rms(vac[sel][np.isfinite(vac[sel])], w[np.isfinite(vac[sel])]), 1000*np.average(y, weights=w),
                          1000*math.sqrt(xx), xy/max(math.sqrt(xx*yy), 1e-30))
        if cc.sum() > 2:
            w = weight[cc]
            a, b = np.polyfit(charge[cc], wf[cc], 1, w=np.sqrt(w))
            residual = wf[cc]-(a*charge[cc]+b)
            logging.debug('%s counter charge slabs: WF error = %.3f V/e x Q %+.3f V, residual RMSE %.1f meV '
                          '(a slope is a charging response the model lacks)', self.split, a, b, rms(residual, w))
            rows = profile[cc]
            use = np.isfinite(rows).all(-1)
            if use.sum() > 1:
                values = ' '.join(f'{h:+.0f}:{1000*math.sqrt(np.average(rows[use, j]**2, weights=w[use])):.0f}'
                                  for j, h in enumerate(PROFILE_HEIGHTS))
                logging.debug('%s planar potential error (model - DFT) relative to %.0f A below the counter charge, '
                              'RMSE in mV at heights above it: %s, vacuum:%.0f', self.split, REFERENCE_DEPTH, values,
                              1000*math.sqrt(np.average(rows[use, -1]**2, weights=w[use])))

    def _log_charging(self, rows):
        rows = np.array(rows)
        mean = rows.mean(0)
        centroid = rows[:, 5][np.isfinite(rows[:, 5])]
        logging.debug('%s charging at fixed geometry on %d counter charge slabs, dQ = %.2f e along Q_cc = -Q: '
                      'dV/dQ %.2f V/e, dmu/dQ %.2f V/e; induced charge per e in the electrode (below %.0f A under '
                      'the counter charge) %.2f, double layer %.2f, outer electrolyte %.2f; its centroid in the '
                      'electrode %s A below the counter charge', self.split, len(rows), CHARGE_STEP, mean[0], mean[1],
                      -REGION_EDGES[0], mean[2], mean[3], mean[4],
                      f'{-centroid.mean():.2f}' if len(centroid) else 'undefined')
        logging.debug('%s per slab dV/dQ: %s', self.split, ' '.join(f'{v:.2f}' for v in rows[:, 0]))

    def _log_molecules(self, rows):
        rows = np.array(rows)
        net, base = rows[:, 0], rows[:, 1]
        logging.debug('%s electrolyte molecules of the counter charge slabs (%d molecules): net charge RMS %.4f e, '
                      'mean %+.4f e, largest |net| %.3f e; from the MACE charges RMS %.4f e, from the placed charge '
                      'and the flows RMS %.4f e', self.split, len(rows), math.sqrt((net*net).mean()), net.mean(),
                      np.abs(net).max(), math.sqrt((base*base).mean()), math.sqrt(((net-base)**2).mean()))

    def _log_law(self):
        law = self.law
        names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 77: 'Ir', 78: 'Pt', 79: 'Au', 29: 'Cu', 28: 'Ni', 45: 'Rh', 25: 'Mn'}
        at = law.get('electrode', np.ones(len(law['numbers']), bool))
        parts = []
        for z in sorted(set(law['numbers'].tolist())):
            groups = [(' at the electrode', (law['numbers'] == z) & at), (' in the electrolyte', (law['numbers'] == z) & ~at)]
            if not all(m.any() for _, m in groups):
                groups = [('', law['numbers'] == z)]
            for label, m in groups:
                parts.append(f"{names.get(int(z), int(z))}{label}: fermi_dos {np.median(law['fermi_dos'][m]):.4f} "
                             f"({law['fermi_dos'][m].min():.4f}..{law['fermi_dos'][m].max():.4f}) e/V, alpha "
                             f"{np.median(law['alpha'][m]):.2f} A^3, bias {law['bias'][m].mean():+.2f}+-{law['bias'][m].std():.2f} V")
        logging.debug('%s law by element (median, min..max): %s', self.split, '; '.join(parts))
        if 'bonds' not in law:
            return
        za, zb, length, c = law['bonds'][:4]
        pair = np.array([f'{names.get(int(min(x, y)), int(min(x, y)))}-{names.get(int(max(x, y)), int(max(x, y)))}'
                         for x, y in zip(za, zb)])
        parts = []
        for name in sorted(set(pair.tolist())):
            m = pair == name
            bins = []
            for lo in np.arange(.8, length.max()+.4, .4):
                sel = m & (length >= lo) & (length < lo+.4)
                if sel.sum():
                    bins.append(f'{lo:.1f}A:{np.median(c[sel]):.3f}')
            parts.append(f'{name} '+' '.join(bins))
        logging.debug('%s bond conductances, median e/V by pair and 0.4 A bin: %s', self.split, '; '.join(parts))
        if len(law['bonds']) < 6:
            return
        ia, ib = law['bonds'][4:6]
        water = ((length < 1.3) & (np.minimum(za, zb) == 1) & (np.maximum(za, zb) == 8) & ~at[ia] & ~at[ib])
        o, h = (law['numbers'] == 8) & ~at, (law['numbers'] == 1) & ~at
        if water.any() and o.any() and h.any():
            bond = K_COULOMB*c[water]*length[water]**2
            alpha_o, alpha_h = np.median(law['alpha'][o]), np.median(law['alpha'][h])
            logging.debug('%s water molecule of the law: alpha O %.2f + 2 x H %.2f + 2 x O-H bond %.2f A^3 (conductance '
                          '%.4f e/V) = %.2f A^3, against %.2f A^3 of a DFT water molecule', self.split, alpha_o, alpha_h,
                          np.median(bond), np.median(c[water]), alpha_o+2*alpha_h+2*np.median(bond), WATER_ALPHA)


def planar_potential_errors(batch, output, slab, normal, length):
    """Model minus DFT planar smooth potential of every slab with a counter charge, at PROFILE_HEIGHTS above the
    counter charge and at the vacuum plane, relative to REFERENCE_DEPTH below it; NaN elsewhere."""
    result = np.full((batch.num_graphs, len(PROFILE_HEIGHTS)+1), np.nan)
    key = 'smooth_fourier_potential'
    if output.get(key) is None or output.get(key+'_dft') is None or output.get('k_vectors') is None:
        return result
    counter = _graph_scalar(batch, 'counter_charge')
    fraction, vacuum = _graph_scalar(batch, 'counter_charge_fraction'), _graph_scalar(batch, 'vacuum_fraction', .5)
    ngrid = float(output['k_vectors_grid_shape'].prod())
    model = torch.view_as_complex(output[key].contiguous())/ngrid
    label = torch.view_as_complex(output[key+'_dft'].contiguous())/ngrid
    wave, mask = output['k_vectors'], output['k_vectors_mask']
    label_mask = output.get('fourier_potential_dft_mask')
    for i in range(batch.num_graphs):
        if not bool(slab[i]) or float(counter[i]) == 0.:
            continue
        along = (wave[i]*normal[i]).sum(-1)
        planar = mask[i] & (torch.linalg.vector_norm(wave[i]-along[:, None]*normal[i], dim=-1) < 1e-8)
        if label_mask is not None:
            planar = planar & label_mask[i]
        kz, c = along[planar], model[i, planar]-label[i, planar]

        def value(z):
            return float((c*torch.exp(1j*kz*z)).real.sum())
        z_cc, z_vac = float(fraction[i]*length[i]), float(vacuum[i]*length[i])
        reference = value(z_cc-REFERENCE_DEPTH)
        result[i, :-1] = [value(z_cc+h)-reference for h in PROFILE_HEIGHTS]
        result[i, -1] = value(z_vac)-reference
    return result


def charging_response(model, wrapper, batch, output, count):
    """dV/dQ, dmu/dQ and the induced charge of up to count slabs with a counter charge, by central differences.

    Returns per slab [dV/dQ, dmu/dQ, electrode fraction, double layer fraction, outer fraction,
    centroid height of the induced electrode charge above the counter charge (NaN when that charge is small)].
    """
    counter = _graph_scalar(batch, 'counter_charge')
    width = _graph_scalar(batch, 'counter_charge_width')
    picks = [i for i in range(batch.num_graphs) if float(counter[i]) != 0. and float(width[i]) > 0][:count]
    if not picks:
        return []
    slab, axis, normal, area, length = slab_geometry(batch)
    fraction = _graph_scalar(batch, 'counter_charge_fraction')
    selected = torch.zeros(batch.num_graphs, dtype=batch.positions.dtype, device=batch.positions.device)
    selected[picks] = 1.
    sides = []
    for sign in (1., -1.):
        data = batch.to_dict()
        data['total_charge'] = data['total_charge']+sign*CHARGE_STEP*selected.reshape(data['total_charge'].shape)
        data['counter_charge'] = data['counter_charge']-sign*CHARGE_STEP*selected.reshape(data['counter_charge'].shape)
        out = wrapper(model, data, training=False, ema=None)
        sides.append((out['cell_voltage'].detach().reshape(-1), out['fermi_level'].detach().reshape(-1),
                      out['density_coefficients'][:, 0].detach()))
        if hasattr(model, 'batch_positions'):
            del model.batch_positions
    rows = []
    for i in picks:
        atoms = batch.batch == i
        induced = (sides[0][2][atoms]-sides[1][2][atoms])/(2*CHARGE_STEP)
        z = (batch.positions[atoms]*normal[i]).sum(-1)
        h = (z-fraction[i]*length[i]+.5*length[i]).remainder(length[i])-.5*length[i]
        electrode = h < REGION_EDGES[0]
        on_electrode = float(induced[electrode].sum())
        rows.append([float((sides[0][0][i]-sides[1][0][i])/(2*CHARGE_STEP)), float((sides[0][1][i]-sides[1][1][i])/(2*CHARGE_STEP)),
                     on_electrode, float(induced[(h >= REGION_EDGES[0]) & (h < REGION_EDGES[1])].sum()),
                     float(induced[h >= REGION_EDGES[1]].sum()),
                     float((induced[electrode]*h[electrode]).sum())/on_electrode if abs(on_electrode) > .05 else float('nan')])
    return rows


def molecule_charges(model, batch, output, count):
    """The net charge of every electrolyte molecule of up to count slabs with a counter charge.

    The molecules are the finite bonded fragments of electrolyte_molecules, a partition the model does
    not use. Returns rows [net charge of the solved state, sum of the MACE charges of the molecule].
    """
    from mace_scf.electrostatics.coupled import electrolyte_molecules
    counter = _graph_scalar(batch, 'counter_charge')
    picks = [i for i in range(batch.num_graphs) if float(counter[i]) != 0.][:count]
    if not picks:
        return []
    data = batch.to_dict()
    with torch.no_grad():
        local = model.local_part(dict(data), compute_force=False)
        members = electrolyte_molecules(model, data, local.positions)
        q0 = local.field_independent_charge_density[:, 0]
    q = output['density_coefficients'][:, 0].detach().to(q0)
    ptr = data['ptr']
    rows = []
    for i in picks:
        first, last = int(ptr[i]), int(ptr[i+1])
        m = members[i, :last-first].to(q0)
        present = m.sum(0) > 0
        if not bool(present.any()):
            continue
        net = (m[:, present]*q[first:last, None]).sum(0)
        base = (m[:, present]*q0[first:last, None]).sum(0)
        rows += [(float(a), float(b)) for a, b in zip(net.tolist(), base.tolist())]
    return rows
