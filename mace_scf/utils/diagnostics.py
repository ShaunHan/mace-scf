"""Validation diagnostics of MACE-VOLT slabs, logged at the DEBUG level.

Where the error of a slab with a counter charge sits, how the model charges at fixed geometry,
and what law the network learned. Nothing here feeds back into training.
"""
import logging
import math

import numpy as np
import torch
from graph_longrange.utils import FIELD_CONSTANT

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


class SlabDiagnostics:
    """Collects per-slab rows during one evaluation pass and logs their summary."""

    def __init__(self, split):
        self.split = split
        self.records = []
        self.charging = []
        self.law = None

    def on_device(self, model, wrapper, batch, output):
        """With the batch on the device: the law summary once, and the charging response of a few slabs."""
        if self.law is None:
            try:
                from mace_scf.electrostatics.coupled import describe_law
                self.law = describe_law(model, batch.to_dict())
            except Exception as error:  # the diagnostics never stop a run
                logging.debug('%s law summary failed: %r', self.split, error)
                self.law = {}
        if len(self.charging) < CHARGING_FRAMES and output.get('density_coefficients') is not None:
            try:
                self.charging += charging_response(model, wrapper, batch, output, CHARGING_FRAMES-len(self.charging))
            except Exception as error:
                logging.debug('%s charging response failed: %r', self.split, error)
                self.charging += [None]*CHARGING_FRAMES

    def rows(self, batch, output, error, weight):
        """Per slab with a work function: its errors, charge and planar potential error profile."""
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
        for i in torch.nonzero(weight > 0).flatten().tolist():
            self.records.append((float(weight[i]), float(error[i]), float(step[i]), float(charge[i]), float(counter[i]),
                                 float(ef[i]), float(vac[i]), profile[i]))

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
        if self.law:
            self._log_law()

    def _log_records(self):
        weight, wf, step, charge, counter, ef, vac = (np.array(column) for column in zip(*[r[:7] for r in self.records]))
        profile = np.array([r[7] for r in self.records])
        cc = counter != 0

        def rms(x, w):
            return 1000*math.sqrt(np.average(x*x, weights=w)) if len(x) else float('nan')
        for name, sel in (('all', np.ones_like(cc)), ('counter charge', cc), ('neutral', ~cc)):
            if sel.sum() < 2:
                continue
            w, x, y = weight[sel], step[sel], wf[sel]
            use = np.isfinite(x)
            xc, yc = x[use]-np.average(x[use], weights=w[use]), y[use]-np.average(y[use], weights=w[use])
            xx, yy, xy = (np.average(v, weights=w[use]) for v in (xc*xc, yc*yc, xc*yc))
            logging.debug('%s %s slabs (n=%d): RMSE WF %.1f, EF %.1f, vacuum %.1f meV; dipole step %.1f mV, '
                          'correlation with the WF error %.3f', self.split, name, int(sel.sum()), rms(y, w),
                          rms(ef[sel][np.isfinite(ef[sel])], w[np.isfinite(ef[sel])]),
                          rms(vac[sel][np.isfinite(vac[sel])], w[np.isfinite(vac[sel])]),
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
        logging.debug('%s charging at fixed geometry on %d counter charge slabs, dQ = %.2f e along Q_cc = -Q: '
                      'dV/dQ %.2f V/e, dmu/dQ %.2f V/e; induced charge per e in the electrode (below %.0f A under '
                      'the counter charge) %.2f, double layer %.2f, outer electrolyte %.2f; its centroid in the '
                      'electrode %.2f A below the counter charge', self.split, len(rows), CHARGE_STEP, mean[0], mean[1],
                      -REGION_EDGES[0], mean[2], mean[3], mean[4], -mean[5])
        logging.debug('%s per slab dV/dQ: %s', self.split, ' '.join(f'{v:.2f}' for v in rows[:, 0]))

    def _log_law(self):
        law = self.law
        names = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 77: 'Ir', 78: 'Pt', 79: 'Au', 29: 'Cu', 28: 'Ni', 45: 'Rh', 25: 'Mn'}
        parts = []
        for z in sorted(set(law['numbers'].tolist())):
            m = law['numbers'] == z
            parts.append(f"{names.get(int(z), int(z))}: fermi_dos {np.median(law['fermi_dos'][m]):.4f} "
                         f"({law['fermi_dos'][m].min():.4f}..{law['fermi_dos'][m].max():.4f}) e/V, alpha "
                         f"{np.median(law['alpha'][m]):.2f} A^3, bias {law['bias'][m].mean():+.2f}+-{law['bias'][m].std():.2f} V")
        logging.debug('%s law by element (median, min..max): %s', self.split, '; '.join(parts))
        if 'bonds' not in law:
            return
        za, zb, length, c = law['bonds']
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
    centroid height of the induced electrode charge above the counter charge].
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
        rows.append([float((sides[0][0][i]-sides[1][0][i])/(2*CHARGE_STEP)), float((sides[0][1][i]-sides[1][1][i])/(2*CHARGE_STEP)),
                     float(induced[electrode].sum()), float(induced[(h >= REGION_EDGES[0]) & (h < REGION_EDGES[1])].sum()),
                     float(induced[h >= REGION_EDGES[1]].sum()),
                     float((induced[electrode]*h[electrode]).sum()/induced[electrode].sum().clamp_min(1e-12))])
    return rows
