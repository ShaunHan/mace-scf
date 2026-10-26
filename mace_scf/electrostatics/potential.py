"""Variational multipoles and a moment-free local electrostatic completion.

The optional response uses the released MACE backbone.  Its electronic energy
is a positive local quadratic form plus the Coulomb energy of a completed
source.  The charge multiplier is the electron chemical potential.  Coarse
density and complete potential are distinct observations of the same state;
DFT coarse density is not assumed to be the Poisson source of a PAW potential.

This is a linear electronic-response approximation at fixed geometry, not a
claim of an exact density functional or a general nonlinear SCF response.

A prescribed Gaussian counter charge (a slab-parallel sheet selected by a
fractional coordinate, or a 3D Gaussian selected by a Cartesian centre) is the
classical source of the VASP-Python CDCE plug-in. Its own energy terms are
exact and independent of the learned Fourier cutoff (CounterCharge).
"""
from __future__ import annotations

import logging
import math
from math import pi
from functools import lru_cache
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F
from graph_longrange.utils import FIELD_CONSTANT
from mace.tools.scatter import scatter_sum
from .molecular import MolecularCorrection, enabled as molecular_enabled, moment_sum


class VariationalResponse(nn.Module):
    """Environment-dependent quadratic electronic functional.

    ``potential_widths`` are the Gaussian widths of the local potential basis.
    An empty list gives the ordinary monopole/dipole electronic functional.
    Completing the potential adds zero-charge, zero-dipole source functions,
    without adding another scalar prediction head.
    """

    variational = True
    spectral = True

    def __init__(self, node_feats_irreps, charges_irreps, num_elements,
                 potential_widths=(), **kwargs):
        super().__init__()
        self.charges_irreps = o3.Irreps(charges_irreps)
        self.coarse_dim = self.charges_irreps.dim
        if self.coarse_dim not in (1, 4):
            raise NotImplementedError("The released analytic GTO basis supports l <= 1")
        self.potential_widths = tuple(float(w) for w in potential_widths)
        if any(not math.isfinite(w) or w <= 0 for w in self.potential_widths):
            raise ValueError("potential_widths must contain positive finite widths")
        self.state_irreps = self.charges_irreps + o3.Irreps("0e + 1o") * len(self.potential_widths)
        # Linear readouts of nonlinear MACE features limit the unconstrained
        # common chemical-level capacity. There is one local drive, not a
        # separately centred charge head and a freely cancelling EF head.
        self.drive = o3.Linear(node_feats_irreps, self.state_irreps, biases=False)
        self.hardness = o3.Linear(node_feats_irreps, f"{len(self.state_irreps)}x0e", biases=True)
        self.scalar_indices = tuple(j for j, (_, ir) in enumerate(self.state_irreps) if ir.l == 0)
        self.species_drive = nn.Parameter(torch.zeros(num_elements, len(self.scalar_indices)))
        self.register_buffer("hardness_scale", torch.full((len(self.state_irreps),), 10.))
        self.register_buffer("feature_norms", torch.ones(o3.Irreps(node_feats_irreps).dim))
        self.register_buffer("proto_coefficients", torch.zeros(num_elements, 64))
        self.register_buffer("proto_fitted", torch.tensor(False))
        self.register_buffer("vacuum_reference_integrals", torch.zeros(num_elements))
        self.register_buffer("deployment_steps", torch.tensor(50))
        self.deployment_mode = "implicit"
        self.register_buffer("spectral_cutoff", torch.tensor(1.))
        with torch.no_grad():
            for p in self.drive.parameters():
                p.zero_()
            for p in self.hardness.parameters():
                p.zero_()

    def coefficients(self, features, node_attrs):
        """Return a drive and a strictly positive, equivariant local hardness."""
        features = features / self.feature_norms
        drive = self.drive(features)
        h = self.hardness_scale * (F.softplus(self.hardness(features)) / math.log(2.) + 1.e-6)
        species = node_attrs @ self.species_drive
        pieces, diagonals = [], []
        for j, ((mul, ir), sl) in enumerate(zip(self.state_irreps, self.state_irreps.slices())):
            value = drive[:, sl]
            if ir.l == 0:
                value = value + species[:, self.scalar_indices.index(j):self.scalar_indices.index(j)+1]
            pieces.append(value)
            diagonals.append(h[:, j:j+1].expand(-1, mul * ir.dim))
        return torch.cat(pieces, -1), torch.cat(diagonals, -1)

    def get_extra_state(self):
        return {"deployment_mode": getattr(self, "deployment_mode", "implicit")}

    def set_extra_state(self, state):
        mode = state.get("deployment_mode", "implicit")
        if mode not in ("implicit", "unroll_scf"):
            raise ValueError(f"Unsupported saved deployment mode: {mode}")
        self.deployment_mode = mode

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Original v367 checkpoints predate solver metadata and used implicit.
        key = prefix+"_extra_state"
        if key not in state_dict:
            state_dict = dict(state_dict)
            state_dict[key] = {"deployment_mode": "implicit"}
        reference = prefix+'vacuum_reference_integrals'
        if reference not in state_dict:
            state_dict = dict(state_dict)
            state_dict[reference] = torch.zeros_like(self.vacuum_reference_integrals)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def reciprocal_grid(cell, cutoff):
    """Full, odd FFT grid covering a physical sphere, independent of labels."""
    rcell = 2 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
    limits = torch.ceil(torch.linalg.vector_norm(cell, dim=-1) * float(cutoff) / (2*math.pi)).long()
    shape = tuple(2 * int(x) + 1 for x in limits.max(0).values.detach().cpu())
    axes = [torch.fft.fftfreq(n, d=1./n, device=cell.device).to(cell) for n in shape]
    modes = torch.cartesian_prod(*axes)
    wave = torch.einsum("ki,gij->gkj", modes, rcell)
    k2 = wave.square().sum(-1)
    mask = k2 <= float(cutoff)**2
    return wave, k2, mask, shape, modes


def radial_basis(k2, cutoff, knots=64):
    coordinate = (k2 / float(cutoff)**2).clamp(0., 1.) * (knots-1)
    left = coordinate.floor().long()
    fraction = coordinate - left
    basis = k2.new_zeros((*k2.shape, knots))
    basis.scatter_add_(-1, left[..., None], (1-fraction)[..., None])
    basis.scatter_add_(-1, (left+1).clamp_max(knots-1)[..., None], fraction[..., None])
    return basis


# Neutral-atom form factors on one uniform grid (inverse Angstrom); Gaussian
# source tails below exp(-TAIL) are dropped from every exact source sum.
FORM_FACTOR_SPACING, FORM_FACTOR_MAXIMUM, TAIL = .01, 40., 36.
# The neutral atoms a DFT counter charge overlaps, from DFT outputs alone. The
# plug-in's proto potential LOCPOT_proto = V_H[atomic densities] + V_ion is a sum
# of rigid spherical atoms P_Z(k). The plug-in couples its sheet to point valence
# ions, not to V_ion, which differs inside each core by VASP's alpha_Z, printed as
# PSCENC = sum_i alpha_i * sum_i Z_i / V. For sheets of width >= 0.7 A this core
# term is a contact, so the sheet sees A_Z(k) = P_Z(k) - alpha_Z. Rows hold
# f_Z = A_Z/K = [n_Z(k) - Z]/k^2 on the uniform grid (electron potential energy).
NEUTRAL_ATOM_KEYS = ('proto_fourier_potential', 'vasp_pscenc', 'vasp_nelect', 'neutral_atom_numbers',
                     'neutral_atom_alpha')
FORM_FACTOR_GRID = np.arange(0., FORM_FACTOR_MAXIMUM+.5*FORM_FACTOR_SPACING, FORM_FACTOR_SPACING)


def neutral_atom_alpha(frames):
    """{Z: alpha_Z} in eV A^3 from PSCENC across compositions, else from the frames' neutral_atom_alpha."""
    use = [(cell, numbers, info) for cell, _, numbers, info in frames
           if info.get('vasp_pscenc') is not None and info.get('vasp_nelect') is not None]
    elements = sorted({int(z) for _, numbers, _ in use for z in numbers})
    rows = np.asarray([[np.count_nonzero(np.asarray(numbers) == z) for z in elements] for _, numbers, _ in use], float)
    if elements and np.linalg.matrix_rank(rows) == len(elements):
        # VASP divides by the valence electrons sum_i Z_i = NELECT + total charge.
        values = np.asarray([float(info['vasp_pscenc'])*abs(np.linalg.det(np.asarray(cell, float)))
                             /(float(info['vasp_nelect'])+float(info.get('total_charge', 0.) or 0.))
                             for cell, _, info in use])
        return dict(zip(elements, np.linalg.lstsq(rows, values, rcond=None)[0].tolist()))
    alpha = {}
    for *_, info in frames:
        if info.get('neutral_atom_alpha') is not None:
            alpha.update(zip(np.asarray(info['neutral_atom_numbers'], int).reshape(-1).tolist(),
                             np.asarray(info['neutral_atom_alpha'], float).reshape(-1).tolist()))
    return alpha


def fit_neutral_atoms(frames, knots=64, max_frames=32):
    """{Z: row} of the neutral atoms the DFT counter charge sees, and a report.

    frames are (cell, positions, numbers, info). info holds proto_fourier_potential,
    the unsmoothed LOCPOT_proto modes of scripts/collect_vasp_data.py on their
    truncated grid, and vasp_pscenc with vasp_nelect, or neutral_atom_alpha. One
    linear fit gives P_Z(k) on hats in k^2 up to the common bandwidth of the
    spectra; beyond it A_Z falls as 1/k^2, where sheets of width >= 3.5/bandwidth
    no longer reach.
    """
    alpha, chosen, seen = neutral_atom_alpha(frames), [], {}
    for cell, positions, numbers, info in frames:
        numbers = np.asarray(numbers, int)
        if info.get('proto_fourier_potential') is None:
            continue
        if len(chosen) < max_frames or any(seen.get(int(z), 0) < 4 for z in numbers):
            proto = np.asarray(info['proto_fourier_potential'], float)
            if proto.ndim != 4 or proto.shape[0] != 2:
                raise ValueError(f'proto_fourier_potential must have shape [2, nx, ny, nz], got {proto.shape}')
            chosen.append((np.asarray(cell, float), np.asarray(positions, float), numbers, proto))
            for z in set(numbers.tolist()):
                seen[z] = seen.get(z, 0)+1
    if not chosen:
        return {}, 'No frame carries proto_fourier_potential: no neutral atoms fitted'
    elements = sorted(seen)
    missing = sorted(set(elements)-set(alpha))
    if missing:
        raise ValueError(f'alpha_Z of elements {missing} is unknown: give frames whose PSCENC spans enough '
                         'compositions, or their neutral_atom_alpha from scripts/collect_vasp_data.py')
    # The largest sphere inside every stored grid: |m_i| <= (n_i-1)/2 holds for |G| <= pi (n_i-1)/|a_i|.
    bandwidth = min(np.pi*(n-1)/np.linalg.norm(a) for cell, *_, proto in chosen for n, a in zip(proto.shape[1:], cell))
    size = len(elements)*knots
    gram, rhs, square = np.zeros((size, size)), np.zeros(size), 0.
    for cell, positions, numbers, proto in chosen:
        shape = proto.shape[1:]
        modes = np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(n)*n) for n in shape], indexing='ij'), -1).reshape(-1, 3)
        waves = 2*np.pi*modes@np.linalg.inv(cell).T
        k2 = np.square(waves).sum(-1)
        use = (k2 > 0) & (k2 <= bandwidth**2)
        # truncate_fft keeps the truncated grid's normalization: c(G) = coefficient/size.
        values = (proto[0]+1j*proto[1]).reshape(-1)[use]/np.prod(shape)
        structure = np.stack([np.exp(-1j*waves[use]@positions[numbers == z].T).sum(-1) for z in elements], -1)
        radial = radial_basis(torch.as_tensor(k2[use]), bandwidth, knots).numpy()
        design = (structure[:, :, None]*radial[:, None, :]).reshape(len(values), -1)/abs(np.linalg.det(cell))
        a, b = np.concatenate([design.real, design.imag]), np.concatenate([values.real, values.imag])
        gram += a.T@a
        rhs += a.T@b
        square += b@b
    curvature = np.diff(np.eye(knots), 2, axis=0)
    weight = 1.e-3*np.sqrt(np.diag(gram)).mean()
    solution = np.linalg.solve(gram+weight**2*np.kron(np.eye(len(elements)), curvature.T@curvature), rhs)
    residual = float(np.sqrt(max(square-2*solution@rhs+solution@gram@solution, 0.)/square))
    if residual > 1.e-2:
        raise ValueError(f'proto_fourier_potential does not match the atoms of its frames (relative residual '
                         f'{residual:.2g}); it must come with the geometry VASP computed it for')
    inside = FORM_FACTOR_GRID <= bandwidth
    radial = radial_basis(torch.as_tensor(FORM_FACTOR_GRID[inside]**2), bandwidth, knots).numpy()
    edge, rows = FORM_FACTOR_GRID[inside][-1], {}
    for row, z in enumerate(elements):
        atom = radial@solution[row*knots:(row+1)*knots]-alpha[z]
        tail = atom[-1]*(edge/FORM_FACTOR_GRID[~inside])**2
        rows[z] = np.concatenate([atom, tail])/FIELD_CONSTANT
    return rows, (f'Neutral atoms of {elements} from {len(chosen)} proto spectra to {bandwidth:.2f}/A '
                  f'(relative residual {residual:.1e}); alpha_Z {[round(alpha[z], 3) for z in elements]} eV A^3')


def install_neutral_atoms(response, atomic_numbers, rows):
    """Store {Z: row} of fit_neutral_atoms for a model's elements; the others keep zero rows.

    They are fixed physics without parameters and enter only counter-charge
    energies, so neutral predictions are unchanged. A placed counter charge
    refuses frames with an element that has no row.
    """
    table = np.zeros((len(atomic_numbers), len(FORM_FACTOR_GRID)))
    for i, z in enumerate(atomic_numbers):
        if int(z) in rows:
            table[i] = rows[int(z)]
    table = torch.as_tensor(table).to(response.source_widths)
    installed = torch.tensor(bool((table[:, 0] != 0).any()), device=table.device)
    for name, value in (('neutral_atom_form_factors', table), ('neutral_atom_fitted', installed)):
        response._buffers[name] = value
    return response


def source_waves(g, i, sigma, sheet):
    """Nonzero wavevectors of a Gaussian source, one of each +/-k pair."""
    kmax = math.sqrt(2*TAIL)/sigma
    if sheet:
        m = torch.arange(1, int(kmax*float(g.length[i])/(2*math.pi))+2, device=g.cell.device).to(g.cell)
        return (2*math.pi*m/g.length[i])[:, None]*g.normal[i]
    reciprocal = 2*math.pi*torch.linalg.inv(g.cell[i]).T
    limits = (kmax/torch.linalg.vector_norm(reciprocal, dim=-1)).ceil().long().tolist()
    modes = torch.cartesian_prod(*[torch.arange(-n, n+1, device=g.cell.device) for n in limits]).to(g.cell)
    x, y, z = modes.unbind(-1)
    modes = modes[(x > 0) | ((x == 0) & ((y > 0) | ((y == 0) & (z > 0))))]
    waves = modes@reciprocal
    return waves[waves.square().sum(-1) <= kmax**2]


class CounterCharge:
    """Prescribed source rho_cc = Q_cc g(r), a normalized Gaussian of width sigma.

    energy holds every term that does not involve the learned density,

        -Q_cc <U_atoms>_g + E_self + (K/2V) p^2 - (E_app.n) p,

    with p = Q_cc m the source moment on the zero-mean slab sawtooth and
    U_atoms the zero-mean periodic potential of the neutral atoms as the DFT
    counter charge sees them, from fit_neutral_atoms (point valence nuclei
    plus the atomic densities of LOCPOT_proto for the VASP-Python plug-in). These
    terms are summed to full bandwidth. density and potential are the model-grid
    source and its field (periodic part plus slab boundary term).
    """

    def __init__(self, g, data, positions):
        n, zero = len(g.volume), g.volume.new_zeros(len(g.volume))
        self.charge = data.get('counter_charge', zero).reshape(-1).to(zero)
        width = data.get('counter_charge_width', zero).reshape(-1).to(zero)
        axis = data.get('counter_charge_axis', torch.full_like(zero, -1).long()).reshape(-1)
        sheet, present = axis >= 0, width > 0
        if not bool(torch.isfinite(self.charge).all()) or bool(((self.charge != 0) & ~present).any()):
            raise ValueError('A nonzero, finite counter charge requires a Gaussian position and width')
        if bool((sheet & (~g.slab | (axis != g.axes))).any()):
            raise ValueError('A counter-charge sheet must be parallel to the slab')
        width = torch.where(present, width, 1.)
        rows = torch.arange(n, device=zero.device)
        fraction = data.get('counter_charge_fraction', zero).reshape(-1).to(zero)
        center = data.get('counter_charge_center', zero[:, None].expand(-1, 3)).reshape(-1, 3).to(zero)
        self.center = torch.where(sheet[:, None], fraction[:, None]*g.cell[rows, g.axes], center)
        phase = -torch.einsum('gkc,gc->gk', g.wave, self.center)
        profile = torch.exp(-.5*g.k2*width[:, None].square())*torch.where(sheet[:, None], g.planar, True)
        self.density = ((self.charge*present/g.volume)[:, None]*profile
                        *torch.complex(phase.cos(), phase.sin())*g.mask)
        # First moment of the periodic Gaussian on the sawtooth about the
        # branch centre, including any part beyond the correction plane.
        c = ((self.center*g.normal).sum(-1)/g.length-g.branch_center+.5).remainder(1.)-.5
        shift = (c[:, None]+torch.arange(-2., 3., device=zero.device))*g.length[:, None]
        a = (-.5*g.length[:, None]-shift)/width[:, None]
        b = (.5*g.length[:, None]-shift)/width[:, None]
        unit = (.5*shift*(torch.erf(b/math.sqrt(2))-torch.erf(a/math.sqrt(2)))
                +width[:, None]/math.sqrt(2*math.pi)*(torch.exp(-.5*a.square())-torch.exp(-.5*b.square()))).sum(-1)
        self.moment = self.charge*unit*(g.slab & present)
        species, energy = data['node_attrs'].argmax(-1), zero.clone()
        table = getattr(g.response, 'neutral_atom_form_factors', None)
        # Every placed source, also at Q_cc = 0: the linear term gives dE/dQ_cc.
        for i in torch.nonzero(present).flatten().tolist():
            if table is None or not bool(getattr(g.response, 'neutral_atom_fitted', False)):
                raise ValueError('A placed counter charge needs the neutral atoms of the DFT data: train on frames '
                                 'with proto_fourier_potential, run scripts/install_neutral_atoms.py, or give '
                                 'MACEFixedPointSCF such frames (neutral_atoms or atoms.info)')
            sigma = float(width[i])
            atoms = data['batch'] == i
            rows = table[species[atoms].unique()]
            covered = FORM_FACTOR_SPACING*(rows.shape[-1]-1-(rows != 0).flip(-1).int().argmax(-1))
            if bool((rows[:, 0] == 0).any()) or bool((covered < math.sqrt(2*TAIL)/sigma).any()):
                raise ValueError('The neutral atoms miss an element of this frame (fit them from frames that '
                                 'contain it) or end below '
                                 f'the {math.sqrt(2*TAIL)/sigma:.3g} 1/Angstrom that a {sigma:.3g} Angstrom '
                                 'counter charge needs')
            waves = source_waves(g, i, sigma, bool(sheet[i]))
            k2 = waves.square().sum(-1)
            # Self energy (K Q^2/2V) sum_k exp(-s^2 k^2)/k^2, summed over +/-k pairs.
            energy[i] = FIELD_CONSTANT*self.charge[i].square()/g.volume[i]*(torch.exp(-sigma**2*k2)/k2).sum()
            # -Q <U_atoms>_g = -(2KQ/V) sum_k exp(-s^2 k^2/2) sum_j f_j(k) cos k.(R_j-c)
            coordinate = (k2.sqrt()/FORM_FACTOR_SPACING).clamp_max(table.shape[-1]-1.001)
            left = coordinate.long()
            form = (table[:, left]*(left+1-coordinate)+table[:, left+1]*(coordinate-left))[species[atoms]]
            cosine = torch.cos((positions[atoms]-self.center[i])@waves.T)
            energy[i] = energy[i]-2*FIELD_CONSTANT*self.charge[i]/g.volume[i]*(
                torch.exp(-.5*sigma**2*k2)*(form*cosine).sum(0)).sum()
        field = (g.external*g.normal).sum(-1)
        self.energy = energy+.5*g.slab_factor*self.moment.square()-field*self.moment
        self.potential = -g.coulomb*self.density+g.ramp(-g.slab_factor*self.moment)


class SpectralGeometry:
    """Per-forward Fourier operators, with no persistent differentiable cache."""

    def __init__(self, model, data, positions):
        self.response = model.field_dependent_charges_map
        self.cell = data["cell"].reshape(-1, 3, 3)
        self.volume = torch.linalg.det(self.cell).abs()
        if bool((self.volume <= 0).any()):
            raise ValueError("Fourier observations require a nondegenerate cell")
        self.pbc = data["pbc"].reshape(-1, 3).bool()
        periodic = self.pbc.sum(-1)
        if bool((periodic < 2).any()):
            raise NotImplementedError("The spectral functional supports periodic bulk and two-periodic slabs; use the native update for isolated structures")
        self.slab = periodic == 2
        self.axes = (~self.pbc).long().argmax(-1)
        rows = torch.arange(len(self.cell), device=positions.device)
        normal = torch.linalg.cross(self.cell[rows, (self.axes+1)%3], self.cell[rows, (self.axes+2)%3])
        self.normal = normal / torch.linalg.vector_norm(normal, dim=-1, keepdim=True)
        self.length = (self.cell[rows, self.axes] * self.normal).sum(-1)
        self.wave, self.k2, self.mask, self.shape, self.modes = reciprocal_grid(self.cell, model.kspace_cutoff)
        self.ngrid = math.prod(self.shape)
        self.batch = data["batch"]
        self.ptr = data["ptr"]
        self.counts = (self.ptr[1:] - self.ptr[:-1]).detach().cpu().tolist()
        self.size = max(self.counts)
        self.slot = torch.arange(len(positions), device=positions.device)-self.ptr[:-1][self.batch]
        self.present = positions.new_zeros((len(self.counts), self.size))
        self.present[self.batch, self.slot] = 1.
        phase = torch.einsum("gkc,gnc->gkn", self.wave, self.pack(positions))
        self.phase = torch.complex(phase.cos(), -phase.sin()) * self.present[:, None, :]
        self.width = float(model.coulomb_energy.density_smearing_width)
        # Fourier transform of normalized monopoles/dipoles. The native GTO
        # convention is q, dy, dz, dx, with actual e and e*Angstrom moments.
        self.basis = self._basis(self.width, self.response.coarse_dim)
        for width in self.response.potential_widths:
            # -Laplacian(G) has zero integral and first moment. Its Coulomb
            # potential is a regular Gaussian even as k tends to zero.
            extra = self._basis(width, 4) * (self.width**2*self.k2)[..., None]
            self.basis = torch.cat((self.basis, extra), -1)
        self.basis = self.basis * self.mask[..., None]
        active = self.mask & (self.k2 > 0)
        self.coulomb = torch.where(active, FIELD_CONSTANT / self.k2.clamp_min(1.e-30), 0.)
        self.constraint = positions.new_zeros((len(self.counts), self.size, self.basis.shape[-1]))
        self.constraint[..., 0] = self.present
        self.dipole_map = positions.new_zeros((*self.constraint.shape, 3))
        self.dipole_map[..., 0, :] = self.pack(positions) * self.present[..., None]
        if self.response.coarse_dim == 4:
            self.dipole_map[..., 1:4, :] = torch.eye(3, device=positions.device, dtype=positions.dtype)[[1, 2, 0]] * self.present[..., None, None]
        self.normal_moment = (self.dipole_map*self.normal[:, None, None, :]).sum(-1)
        self.slab_factor = FIELD_CONSTANT / self.volume * self.slab
        self.external = data.get("external_field", positions.new_zeros((len(self.counts), 3))).reshape(-1, 3)
        if bool(((~self.slab) & (self.external.abs().sum(-1)>0)).any()):
            raise ValueError("A homogeneous field needs an explicit open direction")
        branch = data.get("dipole_correction_fraction", positions.new_full((len(self.counts),), .5)).reshape(-1)
        self.branch = torch.where(torch.isfinite(branch), branch, .5).remainder(1.)
        # Centre of the continuous slab branch, as a fraction of the normal
        # cell vector. Normal moments use the zero-mean sawtooth coordinate
        # about it, the same origin as the plug-in and the correction ramp.
        self.branch_center = (self.branch+.5).remainder(1.)*self.slab
        self.origin = self.branch_center[:, None]*self.cell[rows, self.axes]*self.slab[:, None]
        zfrac = data.get("vacuum_fraction", positions.new_full((len(self.counts),), .5)).reshape(-1)
        self.zfrac = torch.where(torch.isfinite(zfrac), zfrac, .5).remainder(1.)
        self.planar = (self.modes[None]*self.pbc[:, None]).abs().sum(-1) == 0
        self.axial = self.planar & active & self.slab[:, None]
        self.order = self.modes[None].expand(len(self.counts), -1, -1).gather(-1, self.axes[:, None, None].expand(-1, len(self.modes), 1)).squeeze(-1)
        self.attrs = self.pack(data["node_attrs"])
        self.cutoff = model.kspace_cutoff
        self.proto = torch.zeros_like(self.phase[..., 0])
        if bool(self.response.proto_fitted):
            structure = torch.einsum("gkn,gnz->gkz", self.phase, self.attrs.to(self.phase.dtype))
            radial = radial_basis(self.k2, self.cutoff)
            form_factors = torch.einsum("gkr,zr->gkz", radial, self.response.proto_coefficients)
            self.proto = (structure*form_factors).sum(-1)/self.volume[:, None]*active
        # Without an observed proto spectrum, only its far-vacuum reference
        # is identifiable from vacuum-minus-deformation labels. For localized
        # neutral atomic potentials this is -sum(integral(phi_atom))/volume.
        # It is a shared scalar gauge for EF/vacuum, never a substitute field.
        integrals = getattr(self.response, 'vacuum_reference_integrals', None)
        self.vacuum_reference = (self.attrs.sum(1)@integrals/self.volume*self.slab
            if integrals is not None and not bool(self.response.proto_fitted) else self.volume*0.)
        # Use the adjoint of the ACTUAL retained boundary field. Replacing
        # this with unwrapped z would change its uniform gauge and finite-grid
        # response, breaking the energy/EF/potential conjugacy for charged cells.
        if not getattr(self.response, 'coupled', False):
            self.normal_moment = self.adjoint(self.ramp(torch.ones_like(self.volume)))
        self.applied = self.ramp((self.external*self.normal).sum(-1))
        parallel = self.external-(self.external*self.normal).sum(-1)[:, None]*self.normal
        if bool((parallel.abs().max(-1).values > 1.e-10).any()):
            raise ValueError("A homogeneous field parallel to a periodic slab direction needs a separate boundary convention")
        if "counter_slab_bounds" in data:
            raise NotImplementedError("Uniform jellium counter slabs are not a spectral MACE-VOLT source; "
                                      "use a Gaussian sheet (counter_charge_zfrac) or counter_charge_center")
        # Prescribed Gaussian source (sheet or 3D); all its own terms are exact.
        self.counter_source = CounterCharge(self, data, positions)
        self.counter_density = self.counter_source.density
        self.counter = self.counter_source.potential
        self.counter_energy = self.counter_source.energy
        self.counter_moment = self.counter_source.moment

        self.molecular = None
        if bool(molecular_enabled(data, self.volume).any()):
            self.molecular = MolecularCorrection(self, data, positions)
            coupled = getattr(self.response, 'coupled', False)
            self.molecular_source = self.molecular.source_map(
                4 if coupled else self.basis.shape[-1], coarse_dim=4 if coupled else self.response.coarse_dim,
                scale=self.width if coupled else 1.,
                order=(0, 1, 2) if coupled else (1, 2, 0), completion=not coupled)

    def boundary_matrix(self):
        """Low-rank correction used by the implicit linear electronic solve."""
        if self.molecular is None:
            return self.normal_moment, self.slab_factor
        maps = torch.cat((self.normal_moment[..., None], self.molecular_source), -1)
        metric = self.volume.new_zeros((len(self.volume), 6, 6))
        metric[:, 0, 0] = self.slab_factor
        metric[:, 1:, 1:] = self.molecular.metric
        return maps, metric

    @property
    def proto_design(self):
        """Large linear design used only during the training-set reference fit."""
        structure = torch.einsum("gkn,gnz->gkz", self.phase, self.attrs.to(self.phase.dtype))
        radial = radial_basis(self.k2, self.cutoff)
        return structure[..., :, None]*radial[..., None, :]/self.volume[:, None, None, None]

    def _basis(self, width, dim):
        g = torch.exp(-.5 * width**2 * self.k2)
        if dim == 1:
            return torch.complex(g[..., None], torch.zeros_like(g[..., None]))
        imag = -g[..., None] * self.wave[..., [1, 2, 0]]
        real = torch.cat((g[..., None], torch.zeros_like(imag)), -1)
        imag = torch.cat((torch.zeros_like(g[..., None]), imag), -1)
        return torch.complex(real, imag)

    def pack(self, values):
        result = values.new_zeros((len(self.counts), self.size, *values.shape[1:]))
        result[self.batch, self.slot] = values
        return result

    def unpack(self, values):
        return values[self.batch, self.slot]

    def source(self, state, coarse=False):
        dim = self.response.coarse_dim if coarse else state.shape[-1]
        moments = torch.einsum("gkn,gnd->gkd", self.phase, state[..., :dim].to(self.phase.dtype))
        return (moments * self.basis[..., :dim]).sum(-1) / self.volume[:, None]

    def adjoint(self, potential):
        # Adjoint under integral rho*phi dV, not the Euclidean Fourier norm.
        projected = potential[..., None] * self.basis.conj()
        return torch.einsum("gkn,gkd->gnd", self.phase.conj(), projected).real

    def hartree(self, state):
        source = self.source(state)
        value = self.adjoint(self.coulomb * source)
        moment = (state*self.normal_moment).sum((1, 2))
        value = value + self.slab_factor[:, None, None]*moment[:, None, None]*self.normal_moment
        if self.molecular is not None:
            value = value + torch.einsum('gndm,gmk,gk->gnd', self.molecular_source,
                self.molecular.metric, moment_sum(state, self.molecular_source))
        return value

    def diagonal(self):
        diag = (self.basis.abs().square()*self.coulomb[..., None]).sum(1)/self.volume[:, None]
        diag = diag[:, None, :]*self.present[..., None] + self.slab_factor[:, None, None]*self.normal_moment.square()
        if self.molecular is not None:
            # Positive diagonal bound also bounds the projected Chebyshev
            # spectrum. Fixed-Q tangents remove the indefinite Q*M terms.
            maps = self.molecular_source.abs()
            diag = diag + torch.einsum('gndm,gmk,gndk->gnd', maps, self.molecular.metric.abs(), maps)
        return diag

    def ramp(self, field, width=None):
        safe = torch.where(self.axial, self.order, torch.ones_like(self.order))
        phase = -2*math.pi*safe*self.branch[:, None]
        # Fourier coefficients of a centred sawtooth in electron-energy units.
        amplitude = field[:, None]*self.length[:, None]/(2*math.pi*safe)
        width = self.width if width is None else width
        amplitude = amplitude * torch.exp(-.5*width**2*self.k2)*self.axial
        return torch.complex(-amplitude*phase.sin(), amplitude*phase.cos())

    def potentials(self, state):
        moment = (state*self.normal_moment).sum((1, 2))
        density = self.source(state, coarse=True)
        periodic = -self.coulomb*self.source(state)
        boundary = self.ramp(-self.slab_factor*moment)
        deformation = periodic + boundary + self.applied+self.counter
        if self.molecular is not None:
            deformation = deformation+self.molecular.spectrum(
                moment_sum(state, self.molecular_source)+self.molecular.counter)
        return density, deformation, deformation + self.proto

    def plane(self, spectrum):
        phase = 2*math.pi*self.order*self.zfrac[:, None]
        return (spectrum*torch.complex(phase.cos(), phase.sin())*self.axial).real.sum(-1)

    def ramp_plane(self, field):
        """Exact zero-mean sawtooth at the vacuum plane; midpoint at its jump.

        The measured vacuum scalar is not restricted to the retained FFT grid.
        Evaluating this known term directly avoids Fourier ringing at the
        dipole-correction discontinuity without changing any SCF field sample.
        """
        distance = (self.zfrac-self.branch).remainder(1.)
        return field*self.length*torch.where(distance == 0, 0., distance-.5)*self.slab


def conjugate_gradient(operator, rhs, diagonal, steps, constraint, charge):
    """Jacobi-preconditioned CG in the exact fixed-charge tangent space.

    The positive functional makes the projected Hessian positive definite.
    No state clipping, Anderson extrapolation or dense inverse is used.
    All iterations remain differentiable, including force-loss derivatives.
    """
    root = diagonal.rsqrt()
    normal = root*constraint
    norm2 = normal.square().sum((1, 2), keepdim=True)
    safe_norm2 = norm2.clamp_min(torch.finfo(rhs.dtype).tiny)

    def project(value):
        return value-normal*(value*normal).sum((1, 2), keepdim=True)/safe_norm2

    seed = normal * charge[:, None, None] / safe_norm2
    # A zero constraint selects constant chemical potential (grand canonical).
    seed = torch.where(norm2>0, seed, torch.zeros_like(seed))
    right = project(root*(rhs-operator(root*seed)))
    x = torch.zeros_like(rhs)
    residual = right
    direction = residual
    rr = residual.square().sum((1, 2), keepdim=True)
    initial = rr.detach()
    floor = (64*torch.finfo(rhs.dtype).eps)**2 * initial.clamp_min(1.)
    for _ in range(int(steps)):
        if not bool((rr.detach() > floor).any()):
            break
        product = project(root*operator(root*project(direction)))
        denominator = (direction*product).sum((1, 2), keepdim=True)
        active = rr.detach() > floor
        if bool((active & (denominator.detach() <= 0)).any()):
            raise RuntimeError("Nonpositive electronic curvature: check the functional and floating-point precision")
        alpha = torch.where(active, rr/torch.where(active, denominator, torch.ones_like(denominator)), 0.)
        x = x + alpha*direction
        residual = residual-alpha*product
        new_rr = residual.square().sum((1, 2), keepdim=True)
        beta = torch.where(active, new_rr/torch.where(active, rr, torch.ones_like(rr)), 0.)
        direction = residual+beta*direction
        rr = new_rr
    state = root*(seed+project(x))
    gradient = operator(state)-rhs
    multiplier = -(gradient*constraint).sum((1, 2))/constraint.square().sum((1, 2)).clamp_min(1.)
    projected = gradient+multiplier[:, None, None]*constraint
    return state, multiplier, projected.square().mean((1, 2)).sqrt()


def _project_charge(value, constraint):
    norm = constraint.square().sum((1, 2), keepdim=True).clamp_min(1.)
    return value-constraint*(value*constraint).sum((1, 2), keepdim=True)/norm


def _electronic_operator(value, hardness, phase, basis, coulomb, moment, factor, constraint):
    """Positive projected Hessian, extended by identity in the charge direction."""
    tangent = _project_charge(value, constraint)
    source = (torch.einsum('gkn,gnd->gkd', phase, tangent.to(phase.dtype))*basis).sum(-1)
    potential = (coulomb*source)[..., None]*basis.conj()
    hartree = torch.einsum('gkn,gkd->gnd', phase.conj(), potential).real
    if moment.ndim == 4:
        boundary = torch.einsum('gndm,gmk,gk->gnd', moment, factor, moment_sum(tangent, moment))
    else:
        boundary = factor[:, None, None]*moment*(moment*tangent).sum((1, 2), keepdim=True)
    return _project_charge(hardness*tangent+hartree+boundary, constraint)+value-tangent


class _ElectronicSolve(torch.autograd.Function):
    """Matrix-free linear solve with its analytic implicit derivative.

    Differentiating CG iterations loses the susceptibility at zero RHS and
    needlessly retains the iteration trajectory. The adjoint instead solves
    the same positive Hessian. Its backward is itself differentiable, so force
    training includes the complete second derivative. No dense inverse is built.
    """

    @staticmethod
    def forward(ctx, rhs, hardness, phase, basis, coulomb, moment, factor, constraint, diagonal, steps):
        matrix = (hardness, phase, basis, coulomb, moment, factor, constraint)
        zero = torch.zeros_like(constraint)
        unprojected = (*matrix[:-1], zero)
        operator = lambda value: _electronic_operator(value, *unprojected)
        solution, _, residual = conjugate_gradient(operator, rhs, diagonal, steps, constraint, rhs.new_zeros(len(rhs)))
        # Solve the identity charge direction analytically; do not let the
        # preconditioner mix it into the slower physical tangent solve.
        solution = solution+(rhs-_project_charge(rhs, constraint))
        tolerance = max(1.e-10, 256*torch.finfo(rhs.dtype).eps)*(1+rhs.square().mean((1, 2)).sqrt())
        if not bool(torch.isfinite(residual).all()) or bool((residual>tolerance).any()):
            raise RuntimeError(f'Electronic linear solve did not converge in {steps} steps: '
                               f'max residual={float(residual.max()):.5g}. Increase num_scf_steps or inspect conditioning.')
        ctx.save_for_backward(solution, *matrix, diagonal)
        ctx.steps = steps
        return solution

    @staticmethod
    def backward(ctx, gradient):
        solution, *saved = ctx.saved_tensors
        matrix, diagonal = saved[:-1], saved[-1]
        # Adjoint accuracy is independent of the deployment iteration budget.
        # CG needs at most the matrix dimension in exact arithmetic and stops
        # at precision sooner; storing its trajectory is unnecessary.
        adjoint_steps = max(ctx.steps, gradient[0].numel())
        adjoint = _ElectronicSolve.apply(gradient, *matrix, diagonal, adjoint_steps)
        # torch.func differentiates the explicit matrix inputs while retaining
        # the outer dependence on solution/adjoint for the second backward.
        _, pullback = torch.func.vjp(lambda *parameters: _electronic_operator(solution, *parameters), *matrix)
        derivatives = pullback(-adjoint)
        return (adjoint, *derivatives, None, None)


def unroll_electronic(operator, rhs, hardness, coulomb_diagonal, constraint, steps,
                      checkpoint_steps=False):
    """Finite Chebyshev iteration of the positive electronic functional.

    Local-hardness preconditioning preserves rotations within each irrep.
    The transformed Hessian is I + C with C positive semidefinite, so
    [1, 1 + trace(C)] bounds its spectrum, also after charge projection.
    This gives a stable polynomial iteration without a convergence requirement
    or a density-dependent line search. Its zero-RHS susceptibility is nonzero.
    All configured iterations are differentiated; checkpointing only recomputes
    intermediates and does not replace force derivatives with a terminal map.
    """
    from torch.utils.checkpoint import checkpoint

    root = hardness.rsqrt()
    normal = root*constraint
    norm = normal.square().sum((1, 2), keepdim=True).clamp_min(1.e-30)

    def project(value):
        return value-normal*(value*normal).sum((1, 2), keepdim=True)/norm

    upper = 1.+(coulomb_diagonal/hardness).sum((1, 2), keepdim=True)
    center = (upper+1.)/2.
    radius = (upper-1.)/2.
    right = project(root*rhs)
    solution = torch.zeros_like(right)
    direction = torch.zeros_like(right)
    alpha = 1./center
    for iteration in range(int(steps)):
        if iteration == 0:
            beta = torch.zeros_like(alpha)
        else:
            beta = (radius*alpha).square()*(.5 if iteration == 1 else .25)
            alpha = 1./(center-beta/alpha)

        def update(value, previous, step_size, momentum):
            residual = right-project(root*operator(root*project(value)))
            next_direction = residual+momentum*previous
            return value+step_size*next_direction, next_direction

        if checkpoint_steps and torch.is_grad_enabled():
            solution, direction = checkpoint(update, solution, direction, alpha, beta,
                                             use_reentrant=False, preserve_rng_state=False)
        else:
            solution, direction = update(solution, direction, alpha, beta)
    return root*project(solution)


@torch.enable_grad()
def evaluate_variational(model, data, steps=50, training=False, compute_force=True,
                         constant_charge=True, compute_stress=False, mode=None):
    """Evaluate a finite trajectory or a converged implicit electronic state."""
    response = model.field_dependent_charges_map
    mode = getattr(response, "deployment_mode", "implicit") if mode is None else mode
    if mode not in ("unroll_scf", "shortcut_scf", "implicit"):
        raise ValueError(f"Unknown variational electronic mode: {mode}")
    finite_steps = mode != "implicit"
    if int(steps) < 1:
        raise ValueError("num_scf_steps must be positive")
    # Strain all coordinates, cells and periodic image shifts together.
    data = dict(data)
    positions = data["positions"]
    if compute_force:
        positions.requires_grad_(True)
    strain = None
    if compute_stress:
        if bool(molecular_enabled(data, data['total_charge'].reshape(-1)).any()):
            raise NotImplementedError('The cubic molecular correction supports fixed-cell energies and forces; general strain/stress is not implemented')
        strain = positions.new_zeros((len(data["ptr"])-1, 3, 3), requires_grad=True)
        strain_symmetric = .5*(strain+strain.transpose(-1, -2))
        data["positions"] = positions + torch.einsum("ni,nij->nj", positions, strain_symmetric[data["batch"]])
        cell = data["cell"].reshape(-1, 3, 3)
        data["cell"] = cell + cell@strain_symmetric
        edges = data["batch"][data["edge_index"][0]]
        data["shifts"] = data["shifts"]+torch.einsum("ni,nij->nj", data["shifts"], strain_symmetric[edges])
    local = model.local_part(data, compute_force=compute_force)
    geom = SpectralGeometry(model, data, local.positions)
    level_reference = model.fermi_level_offset+geom.vacuum_reference
    drive, hardness = response.coefficients(local.all_layer_feats, data["node_attrs"])
    if hasattr(model, "foundation_element_map"):
        # MACE-POLAR uses Cartesian SH input; native graph_longrange uses y,z,x.
        pieces = []
        for (_, ir), sl in zip(response.state_irreps, response.state_irreps.slices()):
            value = drive[:, sl]
            pieces.append(value[:, [1, 2, 0]] if ir.l == 1 else value)
        drive = torch.cat(pieces, -1)
    drive, hardness = geom.pack(drive), geom.pack(hardness)
    present = geom.present[..., None]
    hardness = hardness*present + (1-present)
    # The state is DEFORMATION density around the neutral proto reference.
    # Its local linear energy includes the exact reference counterterm
    # +<delta_rho,v_proto>. This cancels the fixed proto drive, not any learned
    # error, and gives zero deformation for zero readouts at zero applied field.
    # Otherwise a fresh model spuriously screens the entire neutral-atom core
    # potential. The complete evolving LR+SR deformation still enters H at
    # every iteration, and v_proto remains in the absolute potential observer.
    fixed_drive = drive+geom.adjoint(geom.applied+geom.counter)
    if geom.molecular is not None:
        fixed_drive = fixed_drive-torch.einsum('gndm,gmk,gk->gnd', geom.molecular_source,
                                             geom.molecular.metric, geom.molecular.counter)
    rhs = fixed_drive
    target = data["total_charge"].reshape(-1)
    if finite_steps and constant_charge:
        # Away from stationarity the KKT multiplier alone is not -dE/dQ.
        # Differentiate the actual finite-budget energy, just as for forces.
        target.requires_grad_(True)
    constraint = geom.constraint
    if not constant_charge:
        mu = data["fermi_level"].reshape(-1)-level_reference
        rhs = rhs-mu[:, None, None]*constraint
        constraint = torch.zeros_like(constraint)
        target = torch.zeros_like(target)

    def operator(value):
        return hardness*value+geom.hartree(value)

    diagonal = hardness+geom.diagonal()
    # Eliminate Q exactly. Adding identity in its orthogonal direction makes
    # the tangent-space Hessian positive definite on the full storage space.
    norm = constraint.square().sum((1, 2), keepdim=True).clamp_min(1.)
    seed = constraint*target[:, None, None]/norm
    tangent_rhs = _project_charge(rhs-operator(seed), constraint)
    matrix = (hardness, geom.phase, geom.basis, geom.coulomb/geom.volume[:, None],
              *geom.boundary_matrix(), constraint)
    if finite_steps:
        correction = unroll_electronic(operator, tangent_rhs, hardness,
            geom.diagonal(), constraint, int(steps), checkpoint_steps=mode == "shortcut_scf")
    else:
        correction = _ElectronicSolve.apply(tangent_rhs, *matrix, diagonal, int(steps))
    state = seed+_project_charge(correction, constraint)
    gradient = operator(state)-rhs
    multiplier = -(gradient*constraint).sum((1, 2))/norm.reshape(-1)
    residual = _project_charge(gradient, constraint).square().mean((1, 2)).sqrt()
    residual = residual*(geom.size/(geom.ptr[1:]-geom.ptr[:-1]).to(residual)).sqrt()
    state = state*present
    if not bool(torch.isfinite(state.detach()).all() & torch.isfinite(residual.detach()).all()):
        raise RuntimeError("Nonfinite electronic state; the optimizer was not updated")
    charge = (state*geom.constraint).sum((1, 2))
    mu = multiplier+level_reference if constant_charge else data["fermi_level"].reshape(-1)
    electronic = (.5*state*operator(state)-state*fixed_drive).sum((1, 2))
    electronic = electronic-level_reference*charge+geom.counter_energy
    if geom.molecular is not None:
        electronic = electronic+geom.molecular.energy(geom.molecular.counter)
    energy = local.energies.sum(-1)+electronic
    # For fixed mu the differentiated mechanical potential is E + mu*Q.
    mechanical = energy if constant_charge else energy+mu*charge
    forces, stress = None, None
    charge_derivative = finite_steps and constant_charge
    if compute_force or compute_stress or charge_derivative:
        inputs = ([target] if charge_derivative else [])+([positions] if compute_force else [])+([strain] if compute_stress else [])
        derivatives = torch.autograd.grad(mechanical.sum(), inputs, create_graph=training, retain_graph=training, allow_unused=True)
        if charge_derivative:
            mu = -derivatives[0]
        if compute_force:
            derivative = derivatives[int(charge_derivative)]
            forces = -derivative if derivative is not None else torch.zeros_like(positions)
        if compute_stress:
            derivative = derivatives[-1]
            stress = derivative/geom.volume[:, None, None] if derivative is not None else torch.zeros_like(strain)
    density, deformation, total = geom.potentials(state)
    dipole = (geom.dipole_map*state[..., None]).sum((1, 2))
    if geom.molecular is not None:
        dipole = torch.where(geom.molecular.active[:, None],
                             moment_sum(state, geom.molecular_source)[:, 1:4], dipole)
    vacuum = geom.plane(total)+geom.vacuum_reference
    output = {"energy": energy, "forces": forces, "stress": stress,
              "virials": -stress*geom.volume[:, None, None] if stress is not None else None,
              "density_coefficients": geom.unpack(state[..., :response.coarse_dim]),
              "dipole": dipole, "total_charge": charge, "fermi_level": mu,
              "vacuum_potential": vacuum,
              "electrostatic_energy": electronic, "electron_energy": electronic,
              "external_field": geom.external,
              "electrostatic_features": geom.unpack(-geom.hartree(state)),
              "fourier_density": torch.view_as_real(density)*geom.ngrid,
              "fourier_potential": torch.view_as_real(deformation)*geom.ngrid,
              "fourier_total_potential": torch.view_as_real(total)*geom.ngrid,
              "k_vectors_mask": geom.mask, "k_vectors": geom.wave,
              "k_vectors_grid_shape": torch.tensor(geom.shape, device=positions.device),
              "scf_residual": residual, "scf_steps": positions.new_full((len(mu),), int(steps)),
              "potential_coefficients": geom.unpack(state[..., response.coarse_dim:]),
              "esps": None, "esps_dft": None,
              "charges_history": geom.unpack(state[..., :response.coarse_dim])[..., None]}
    return attach_observations(data, geom, output)


def attach_observations(data, geom, output):
    """Attach measured labels after prediction, using the same retained grid."""
    output['density_smearing_width'] = geom.wave.new_tensor(geom.width)
    widths = getattr(geom.response, 'source_widths', geom.response.potential_widths)
    width = getattr(geom.response, 'potential_amplitudes_smearing_width',
                    float(widths[0]) if len(widths) else geom.width)
    output['potential_smearing_width'] = geom.wave.new_tensor(width)
    # Labels are attached only AFTER computing the physical state/observables.
    for key in ("fourier_density", "fourier_potential", "fourier_proto_potential"):
        target_fft = _target_fft(data, key, key+"_shape", geom.shape, geom.wave.dtype)
        if target_fft is not None:
            output[key+"_dft"] = target_fft
            output[key+"_dft_mask"] = target_mode_mask(data[key+"_shape"], geom.modes)
            if key != "fourier_density":
                # Raw labels retain their gauge for audits. Uniform offsets
                # never enter Poisson, the recurrent field, or spatial loss.
                output[key+"_dft_mean"] = (target_fft[...,0]*(geom.k2==0)).sum(-1)/geom.ngrid
    phi = output.get("fourier_potential_dft")
    proto = output.get("fourier_proto_potential_dft")
    if phi is not None and proto is not None:
        output["fourier_total_potential_dft"] = phi+proto
        output["fourier_total_potential_dft_mask"] = output["fourier_potential_dft_mask"] & output["fourier_proto_potential_dft_mask"]
        reference = torch.view_as_complex((phi+proto).contiguous())/geom.ngrid
        output["vacuum_potential_dft"] = (geom.plane(reference)
            +output["fourier_potential_dft_mean"]+output["fourier_proto_potential_dft_mean"])
    return output


@torch.no_grad()
def initialize_response(model, loader, device, apply_potential_smearing=False, fit_proto=True):
    """Fit only training-set feature scales and frozen proto form factors."""
    from mace.tools import torch_geometric
    loader = torch_geometric.dataloader.DataLoader(loader.dataset,
        batch_size=getattr(loader,'batch_size',1) or 1,shuffle=False,drop_last=False,
        generator=torch.Generator().manual_seed(0))
    response = model.field_dependent_charges_map
    response.spectral_cutoff.copy_(model.kspace_cutoff)
    norms = torch.zeros_like(response.feature_norms)
    count = 0
    size = response.proto_coefficients.numel()
    gram = torch.zeros(size, size, dtype=torch.float64, device=device)
    rhs = torch.zeros(size, dtype=torch.float64, device=device)
    square = torch.zeros((), dtype=torch.float64, device=device)
    observations = 0
    norm_graphs = 0
    for batch in loader:
        data = batch.to(device).to_dict()
        if not getattr(response, 'coupled', False) and norm_graphs < 64:
            local = model.local_part(data, compute_force=False)
            norms += local.all_layer_feats.square().sum(0)
            count += len(local.positions)
            norm_graphs += len(data["ptr"])-1
        if not fit_proto or not bool((data.get("fourier_proto_potential_weight", torch.zeros(1, device=device))>0).any()):
            continue
        geom = SpectralGeometry(model, data, data["positions"])
        target = _target_fft(data, "fourier_proto_potential", "fourier_proto_potential_shape", geom.shape, data["positions"].dtype)
        if apply_potential_smearing:
            width = getattr(response, 'potential_amplitudes_smearing_width', geom.width)
            target = target*torch.exp(-.5*width**2*geom.k2)[..., None]
        weights = data["weight"].reshape(-1)*data["fourier_proto_potential_weight"].reshape(-1)
        use = geom.mask & target_mode_mask(data["fourier_proto_potential_shape"], geom.modes) & (geom.k2>0) & (weights[:, None]>0)
        design = torch.view_as_real(geom.proto_design).reshape(len(weights), -1, size, 2)
        design = design.permute(0, 1, 3, 2)[use].reshape(-1, size).double()
        values = (target/geom.ngrid)[use].reshape(-1).double()
        w = weights[:, None].expand_as(use)[use].repeat_interleave(2).sqrt().double()
        design = design*w[:, None]
        values = values*w
        gram += design.T@design
        rhs += design.T@values
        square += values.square().sum()
        observations += values.numel()
    # One RMS per irrep channel, preserving rotations. No per-component scale.
    if not getattr(response, 'coupled', False):
        feature_irreps = model.products[0].linear.irreps_out
        for mul_ir, sl in zip(feature_irreps, feature_irreps.slices()):
            mul, ir = mul_ir
            values = norms[sl].reshape(mul, ir.dim).mean(-1).div(max(count, 1)).sqrt().clamp_min(1.e-3)
            response.feature_norms[sl] = values[:, None].expand(-1, ir.dim).reshape(-1)
    if observations:
        response.proto_smearing_width = (float(getattr(response, 'potential_amplitudes_smearing_width', geom.width))
                                         if apply_potential_smearing else 0.)
        solution = torch.linalg.lstsq(gram.cpu(), rhs.cpu()[:, None],
                                     rcond=1.e-10, driver="gelsd").solution[:, 0].to(gram)
        response.proto_coefficients.copy_(solution.reshape_as(response.proto_coefficients))
        response.proto_fitted.fill_(True)
        error = (square-2*solution@rhs+solution@(gram@solution)).clamp_min(0)
        logging.debug("Frozen training-only proto fit: spectral component RMS %.6g eV, observations %d", float((error/observations).sqrt()), observations)
    logging.info("Electronic response %s: %d coarse + %d regular coefficients per atom; deployment uses %d steps", type(response).__name__, response.coarse_dim, response.state_irreps.dim-response.coarse_dim, int(response.deployment_steps))


@torch.no_grad()
def initialize_vacuum_reference(model, loader, device, apply_potential_smearing=False):
    """Fit the missing atomic vacuum reference from training observations only.

    Observed vacuum minus the retained deformation plane determines a small
    atomic reference model. For a neutral-reference response, the same scalar
    observer is used with and without a proto spectrum: its finite-grid tail
    must not redefine a measured real-space vacuum value. Absolute training
    observations fix the reference without subtracting a fitted offset.
    The residual measures its far-vacuum/finite-spectrum approximation.
    """
    response = model.field_dependent_charges_map
    response.vacuum_reference_integrals.zero_()
    if bool(response.proto_fitted):
        if getattr(response, 'response_reference', 'total') != 'neutral':
            return
        # When no paired scalar observation exists, the fitted atomic radial
        # k=0 limit is a reference estimate, not an observed absolute zero.
        response.vacuum_reference_integrals.copy_(-response.proto_coefficients[:,0])
    from .loss import vacuum_reference_weights
    from mace_scf.utils.foundation import calibration_loader
    designs, targets, weights = [], [], []
    for batch in calibration_loader(loader, maximum=None):
        batch = batch.to(device)
        label, weight = vacuum_reference_weights(batch, batch.total_charge)
        weight = weight*batch.fourier_potential_weight
        if not bool((weight > 0).any()):
            continue
        data = batch.to_dict()
        geom = SpectralGeometry(model, data, data['positions'])
        phi = _target_fft(data, 'fourier_potential', 'fourier_potential_shape', geom.shape, geom.volume.dtype)
        if phi is None:
            continue
        if apply_potential_smearing:
            width = getattr(response, 'potential_amplitudes_smearing_width', geom.width)
            phi = phi*torch.exp(-.5*width**2*geom.k2)[..., None]
        complete = (~geom.axial | target_mode_mask(data['fourier_potential_shape'], geom.modes)).all(-1)
        use = (weight > 0) & complete
        if bool(use.any()):
            plane = geom.plane(torch.view_as_complex(phi.contiguous())/geom.ngrid)
            designs.append((geom.attrs.sum(1)/geom.volume[:,None])[use])
            targets.append((label-plane)[use])
            weights.append(weight[use])
    if not designs:
        logging.info('No paired deformation/vacuum reference: retaining %s scalar reference',
                     'proto radial-limit' if bool(response.proto_fitted) else 'zero')
        return
    x, y, w = torch.cat(designs), torch.cat(targets), torch.cat(weights)
    scale = x.square().mul(w[:,None]).sum(0).sqrt().clamp_min(1.e-12)
    solution = torch.linalg.lstsq((x/scale*w.sqrt()[:,None]).cpu(), (y*w.sqrt()).cpu(),
                                  rcond=1.e-10, driver='gelsd').solution.to(x)/scale
    response.vacuum_reference_integrals.copy_(solution)
    residual = ((x@solution-y).square()*w).sum()/w.sum()
    logging.debug('Training-only atomic vacuum reference: labels=%d, residual_RMSE=%.6g V; '
                 'shared EF/vacuum gauge, fitted independently of optional spatial proto data',len(y),float(residual.sqrt()))

def _fft_shapes(values: torch.Tensor, num_graphs: int) -> list[tuple[int, int, int]]:
    """Decode one stored three-dimensional FFT shape per graph."""
    values = values.reshape(-1)
    if values.numel() == 0:
        return []
    if values.numel() != 3 * num_graphs:
        raise ValueError(
            f"Expected three FFT dimensions per graph, got {values.numel()} values "
            f"for {num_graphs} graphs"
        )
    return [
        tuple(int(x) for x in values[3 * graph : 3 * graph + 3].tolist())
        for graph in range(num_graphs)
    ]


def target_mode_mask(shapes, modes):
    """Known reciprocal coefficients, excluding ambiguous even-grid Nyquist modes.

    Padding a stored spectrum supplies no observation at new higher frequencies.
    In particular, missing data must never become a target of zero potential.
    """
    shapes = shapes.reshape(-1, 3).to(device=modes.device)
    maximum = torch.div(shapes-1, 2, rounding_mode='floor')
    return (modes[None].abs() <= maximum[:, None]).all(-1)

def project_hermitian_fft(values: torch.Tensor) -> torch.Tensor:
    """Project a full FFT grid onto the Hermitian subspace of a real field."""
    partner = torch.conj(
        torch.roll(
            torch.flip(values, dims=tuple(range(values.ndim))),
            shifts=tuple(1 for _ in range(values.ndim)),
            dims=tuple(range(values.ndim)),
        )
    )
    return 0.5 * (values + partner)

def _fft_integer_modes(size: int) -> tuple[int, ...]:
    """Integer reciprocal modes in native FFT order."""
    size = int(size)
    positive_stop = (size - 1) // 2
    positive = list(range(0, positive_stop + 1))
    negative_start = -(size // 2)
    negative = list(range(negative_start, 0))
    return tuple(positive + negative)

@lru_cache(maxsize=256)
def _fft_resize_index_map(
    old_shape: tuple[int, int, int],
    new_shape: tuple[int, int, int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Map identical integer reciprocal modes between two FFT grids.

    Array-centred slicing is not mode exact across even/odd grid changes because
    Nyquist/negative-frequency indices move.  Match integer reciprocal modes
    explicitly so batching variable DFT FFT shapes does not shift coefficients.
    """
    old_modes = [_fft_integer_modes(n) for n in old_shape]
    new_lookup = [
        {mode: index for index, mode in enumerate(_fft_integer_modes(n))}
        for n in new_shape
    ]
    old_flat: list[int] = []
    new_flat: list[int] = []
    old_stride1 = old_shape[1] * old_shape[2]
    old_stride2 = old_shape[2]
    new_stride1 = new_shape[1] * new_shape[2]
    new_stride2 = new_shape[2]
    for i0, mode0 in enumerate(old_modes[0]):
        new0 = new_lookup[0].get(mode0)
        if new0 is None:
            continue
        for i1, mode1 in enumerate(old_modes[1]):
            new1 = new_lookup[1].get(mode1)
            if new1 is None:
                continue
            for i2, mode2 in enumerate(old_modes[2]):
                new2 = new_lookup[2].get(mode2)
                if new2 is None:
                    continue
                old_flat.append(i0 * old_stride1 + i1 * old_stride2 + i2)
                new_flat.append(new0 * new_stride1 + new1 * new_stride2 + new2)
    return tuple(old_flat), tuple(new_flat)

def resize_fft(values: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Losslessly remap common reciprocal modes to ``shape``.

    The scale factor preserves the unnormalised FFT convention used by the DFT
    collector.  Hermitian projection restores the +/- partner when parity
    changes make a former Nyquist coefficient a distinct reciprocal pair.
    """
    shape = tuple(int(x) for x in shape)
    old_shape = tuple(int(x) for x in values.shape)
    if old_shape == shape:
        return values
    old_flat, new_flat = _fft_resize_index_map(old_shape, shape)
    resized = values.new_zeros(shape).reshape(-1)
    if old_flat:
        src = torch.as_tensor(old_flat, dtype=torch.long, device=values.device)
        dst = torch.as_tensor(new_flat, dtype=torch.long, device=values.device)
        resized.index_copy_(0, dst, values.reshape(-1).index_select(0, src))
    resized = resized.reshape(shape)
    scale = math.prod(shape) / float(math.prod(old_shape))
    return project_hermitian_fft(resized * scale)

def _target_fft(
    data: Dict[str, torch.Tensor],
    key: str,
    shape_key: str,
    grid_shape: Sequence[int],
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Collate a variable-shape stored FFT target onto the current batch grid.

    DFT targets are stored as flattened per-configuration FFT arrays plus a
    three-integer shape.  Equal shapes are only reshaped; unequal shapes are
    remapped by exact integer reciprocal mode with ``resize_fft``.  The returned
    tensor uses the model convention ``[..., 2] = (real, imag)``.
    """
    values = data.get(key)
    shapes = data.get(shape_key)
    if values is None or shapes is None or values.numel() == 0:
        return None
    num_graphs = data["ptr"].numel() - 1
    values = values.to(device=data["positions"].device)
    if not torch.is_complex(values):
        if values.shape[-1] != 2:
            raise ValueError(f"{key} must be complex or end in a real/imaginary axis")
        values = torch.complex(values[..., 0], values[..., 1])
    values = values.reshape(-1)
    fft_shapes = _fft_shapes(shapes, num_graphs)
    target_shape = tuple(int(x) for x in grid_shape)
    target_size = math.prod(target_shape)
    if (
        len(fft_shapes) == num_graphs
        and all(shape == target_shape for shape in fft_shapes)
        and values.numel() == num_graphs * target_size
    ):
        output = values.reshape(num_graphs, target_size)
    else:
        if len(fft_shapes) != num_graphs:
            raise ValueError(
                f"{shape_key} contains {len(fft_shapes)} FFT shapes for "
                f"{num_graphs} graphs"
            )
        output_items = []
        offset = 0
        for shape in fft_shapes:
            size = math.prod(shape)
            if size == 0:
                output_items.append(values.new_zeros(target_shape).reshape(-1))
                continue
            item = values[offset : offset + size].reshape(shape)
            output_items.append(resize_fft(item, target_shape).reshape(-1))
            offset += size
        if offset != values.numel():
            raise ValueError(
                f"{key} contains {values.numel()} values but shapes consume {offset}"
            )
        output = torch.stack(output_items)
    return torch.stack([output.real, output.imag], dim=-1).to(dtype=dtype)


def evaluate_electronic(model, data, **kwargs):
    """Dispatch the saved response family without reinterpreting its weights."""
    if getattr(model.field_dependent_charges_map, 'coupled', False):
        from .coupled import evaluate_coupled
        return evaluate_coupled(model, data, **kwargs)
    kwargs.pop('mixing', None)
    kwargs.pop('tolerance', None)
    return evaluate_variational(model, data, **kwargs)
