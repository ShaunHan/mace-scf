"""Fourier grids, neutral atoms, the Gaussian counter charge and the DFT field labels of MACE-VOLT.

Potentials are electron referenced: a value in V equals the electron potential
energy in eV written by plane wave codes.
"""
import math
from functools import lru_cache
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from graph_longrange.utils import FIELD_CONSTANT


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
    """Piecewise linear hats in k^2 on [0, cutoff^2]."""
    coordinate = (k2 / float(cutoff)**2).clamp(0., 1.) * (knots-1)
    left = coordinate.floor().long()
    fraction = coordinate - left
    basis = k2.new_zeros((*k2.shape, knots))
    basis.scatter_add_(-1, left[..., None], (1-fraction)[..., None])
    basis.scatter_add_(-1, (left+1).clamp_max(knots-1)[..., None], fraction[..., None])
    return basis


# The neutral atoms of the reference calculations. Their potential, the proto
# potential (info key fourier_proto_potential), is a sum of rigid spherical atoms
# P_Z(k). P_Z(0) is the volume integral of an atom's potential: minus its sum over
# the atoms per volume is the vacuum level in the zero-mean origin of periodic DFT.
# A counter charge sees f_Z = P_Z(k)/K, tabulated on a uniform k grid.
# Gaussian source tails below exp(-TAIL) are dropped from every exact source sum.
FORM_FACTOR_SPACING, FORM_FACTOR_MAXIMUM, TAIL = .01, 40., 36.
FORM_FACTOR_GRID = np.arange(0., FORM_FACTOR_MAXIMUM+.5*FORM_FACTOR_SPACING, FORM_FACTOR_SPACING)


def fit_neutral_atoms(frames, knots=64, max_frames=32):
    """The neutral atoms of the reference calculations, from their proto potentials.

    frames are (cell, positions, numbers, info), where info holds the unsmoothed
    proto modes [2, nx, ny, nz] as fourier_proto_potential. One linear fit gives
    P_Z(k) on hats in k^2 up to the common bandwidth of the spectra. Returns
    ({Z: P_Z(0)}, {Z: form factor row}, report); beyond the bandwidth the form
    factors fall as 1/k^2, where sheets of width >= 3.5/bandwidth no longer reach.
    """
    chosen, seen = [], {}
    for cell, positions, numbers, info in frames:
        numbers = np.asarray(numbers, int)
        if info.get('fourier_proto_potential') is None:
            continue
        if len(chosen) < max_frames or any(seen.get(int(z), 0) < 4 for z in numbers):
            proto = np.asarray(info['fourier_proto_potential'], float)
            if proto.ndim != 4 or proto.shape[0] != 2:
                raise ValueError(f'fourier_proto_potential must have shape [2, nx, ny, nz], got {proto.shape}')
            chosen.append((np.asarray(cell, float), np.asarray(positions, float), numbers, proto))
            for z in set(numbers.tolist()):
                seen[z] = seen.get(z, 0)+1
    if not chosen:
        return {}, {}, 'No frame carries fourier_proto_potential: no neutral atoms fitted'
    elements = sorted(seen)
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
        # Stored modes keep the truncated grid's normalization: c(G) = coefficient/size.
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
        raise ValueError(f'fourier_proto_potential does not match the atoms of its frames (relative residual '
                         f'{residual:.2g}); it must come with the geometry it was computed for')
    inside = FORM_FACTOR_GRID <= bandwidth
    radial = radial_basis(torch.as_tensor(FORM_FACTOR_GRID[inside]**2), bandwidth, knots).numpy()
    edge, integrals, rows = FORM_FACTOR_GRID[inside][-1], {}, {}
    for row, z in enumerate(elements):
        atom = radial@solution[row*knots:(row+1)*knots]
        integrals[z] = float(atom[0])
        rows[z] = np.concatenate([atom, atom[-1]*(edge/FORM_FACTOR_GRID[~inside])**2])/FIELD_CONSTANT
    report = (f'Neutral atoms of {elements} from {len(chosen)} proto spectra to {bandwidth:.2f}/A (relative '
              f'residual {residual:.1e}); potential integrals {[round(integrals[z], 2) for z in elements]} eV A^3')
    return integrals, rows, report


def install_neutral_atoms(response, atomic_numbers, integrals, rows):
    """Store the fitted neutral atoms in a response: -P_Z(0) per element and the form factors."""
    table = np.zeros((len(atomic_numbers), len(FORM_FACTOR_GRID)))
    origin = np.zeros(len(atomic_numbers))
    for i, z in enumerate(atomic_numbers):
        if int(z) in rows:
            table[i] = rows[int(z)]
        origin[i] = -integrals.get(int(z), 0.)
    table = torch.as_tensor(table).to(response.source_widths)
    response.neutral_atom_form_factors = table
    response.vacuum_reference_integrals = torch.as_tensor(origin).to(table)
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

    energy holds the terms without the learned density: the self energy, the slab
    terms (K/2V) p^2 - (E_app.n) p of its moment p and, unless the response sets
    counter_charge_overlap False as constant-potential MD does, the overlap
    -Q_cc <phi_NA>_g with the neutral atoms. All are summed to full bandwidth.
    density and potential are the model-grid source and its field.
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
        # First moment of the periodic Gaussian on the sawtooth about the branch centre.
        c = ((self.center*g.normal).sum(-1)/g.length-g.branch_center+.5).remainder(1.)-.5
        shift = (c[:, None]+torch.arange(-2., 3., device=zero.device))*g.length[:, None]
        a = (-.5*g.length[:, None]-shift)/width[:, None]
        b = (.5*g.length[:, None]-shift)/width[:, None]
        unit = (.5*shift*(torch.erf(b/math.sqrt(2))-torch.erf(a/math.sqrt(2)))
                +width[:, None]/math.sqrt(2*math.pi)*(torch.exp(-.5*a.square())-torch.exp(-.5*b.square()))).sum(-1)
        self.moment = self.charge*unit*(g.slab & present)
        self.width, self.present = width, present
        species, energy = data['node_attrs'].argmax(-1), zero.clone()
        table = g.response.neutral_atom_form_factors
        overlap = getattr(g.response, 'counter_charge_overlap', True)
        # Every placed source, also at Q_cc = 0: the linear term gives dE/dQ_cc.
        for i in torch.nonzero(present).flatten().tolist():
            sigma = float(width[i])
            atoms = data['batch'] == i
            if overlap:
                rows = table[species[atoms].unique()]
                covered = FORM_FACTOR_SPACING*(rows.shape[-1]-1-(rows != 0).flip(-1).int().argmax(-1))
                if bool((rows[:, 0] == 0).any()) or bool((covered < math.sqrt(2*TAIL)/sigma).any()):
                    raise ValueError('A placed counter charge needs the neutral atoms of every element up to '
                                     '6/width: train on frames with fourier_proto_potential')
            waves = source_waves(g, i, sigma, bool(sheet[i]))
            k2 = waves.square().sum(-1)
            # Self energy (K Q^2/2V) sum_k exp(-s^2 k^2)/k^2, summed over +/-k pairs.
            energy[i] = FIELD_CONSTANT*self.charge[i].square()/g.volume[i]*(torch.exp(-sigma**2*k2)/k2).sum()
            if not overlap:
                continue
            # -Q <phi_NA>_g = -(2KQ/V) sum_k exp(-s^2 k^2/2) sum_j f_j(k) cos k.(R_j-c)
            coordinate = (k2.sqrt()/FORM_FACTOR_SPACING).clamp_max(table.shape[-1]-1.001)
            left = coordinate.long()
            form = (table[:, left]*(left+1-coordinate)+table[:, left+1]*(coordinate-left))[species[atoms]]
            cosine = torch.cos((positions[atoms]-self.center[i])@waves.T)
            energy[i] = energy[i]-2*FIELD_CONSTANT*self.charge[i]/g.volume[i]*(
                torch.exp(-.5*sigma**2*k2)*(form*cosine).sum(0)).sum()
        field = (g.external*g.normal).sum(-1)
        self.energy = energy+.5*g.slab_factor*self.moment.square()-field*self.moment
        self.potential = -g.coulomb*self.density+g.ramp(-g.slab_factor*self.moment)


def attach_labels(data, g, output):
    """Attach the DFT density and deformation potential on the model's Fourier grid."""
    for key in ("fourier_density", "fourier_potential"):
        target = _target_fft(data, key, key+"_shape", g.shape, g.wave.dtype)
        if target is not None:
            output[key+"_dft"] = target
            output[key+"_dft_mask"] = target_mode_mask(data[key+"_shape"], g.modes)
    return output


def _fft_shapes(values: torch.Tensor, num_graphs: int) -> list[tuple[int, int, int]]:
    """Decode one stored three-dimensional FFT shape per graph."""
    values = values.reshape(-1)
    if values.numel() != 3 * num_graphs:
        raise ValueError(f"Expected three FFT dimensions per graph, got {values.numel()} for {num_graphs} graphs")
    return [tuple(int(x) for x in values[3*graph:3*graph+3].tolist()) for graph in range(num_graphs)]


def target_mode_mask(shapes, modes):
    """Known reciprocal coefficients, excluding ambiguous even-grid Nyquist modes."""
    shapes = shapes.reshape(-1, 3).to(device=modes.device)
    maximum = torch.div(shapes-1, 2, rounding_mode='floor')
    return (modes[None].abs() <= maximum[:, None]).all(-1)


def project_hermitian_fft(values: torch.Tensor) -> torch.Tensor:
    """Project a full FFT grid onto the Hermitian subspace of a real field."""
    dims = tuple(range(values.ndim))
    partner = torch.conj(torch.roll(torch.flip(values, dims=dims), shifts=(1,)*values.ndim, dims=dims))
    return 0.5 * (values + partner)


def _fft_integer_modes(size: int) -> tuple[int, ...]:
    """Integer reciprocal modes in native FFT order."""
    return tuple(range(0, (size-1)//2+1)) + tuple(range(-(size//2), 0))


@lru_cache(maxsize=256)
def _fft_resize_index_map(old_shape, new_shape):
    """Flat indices of identical integer reciprocal modes in two FFT grids."""
    old_modes = [_fft_integer_modes(n) for n in old_shape]
    lookup = [{mode: i for i, mode in enumerate(_fft_integer_modes(n))} for n in new_shape]
    old_flat, new_flat = [], []
    for i0, m0 in enumerate(old_modes[0]):
        for i1, m1 in enumerate(old_modes[1]):
            for i2, m2 in enumerate(old_modes[2]):
                n0, n1, n2 = lookup[0].get(m0), lookup[1].get(m1), lookup[2].get(m2)
                if None not in (n0, n1, n2):
                    old_flat.append((i0*old_shape[1]+i1)*old_shape[2]+i2)
                    new_flat.append((n0*new_shape[1]+n1)*new_shape[2]+n2)
    return tuple(old_flat), tuple(new_flat)


def resize_fft(values: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Remap the common reciprocal modes to ``shape`` in the unnormalized FFT convention."""
    shape, old_shape = tuple(int(x) for x in shape), tuple(int(x) for x in values.shape)
    if old_shape == shape:
        return values
    old_flat, new_flat = _fft_resize_index_map(old_shape, shape)
    resized = values.new_zeros(shape).reshape(-1)
    if old_flat:
        src = torch.as_tensor(old_flat, dtype=torch.long, device=values.device)
        dst = torch.as_tensor(new_flat, dtype=torch.long, device=values.device)
        resized.index_copy_(0, dst, values.reshape(-1).index_select(0, src))
    return project_hermitian_fft(resized.reshape(shape) * (math.prod(shape)/float(math.prod(old_shape))))


def _target_fft(data: Dict[str, torch.Tensor], key: str, shape_key: str, grid_shape: Sequence[int],
                dtype: torch.dtype) -> Optional[torch.Tensor]:
    """Collate the flattened per-graph FFT targets onto the batch grid as [..., (real, imag)]."""
    values, shapes = data.get(key), data.get(shape_key)
    if values is None or shapes is None or values.numel() == 0:
        return None
    num_graphs = data["ptr"].numel() - 1
    values = values.to(device=data["positions"].device).reshape(-1)
    target_shape = tuple(int(x) for x in grid_shape)
    output, offset = [], 0
    for shape in _fft_shapes(shapes, num_graphs):
        size = math.prod(shape)
        if size == 0:
            output.append(values.new_zeros(target_shape).reshape(-1))
            continue
        output.append(resize_fft(values[offset:offset+size].reshape(shape), target_shape).reshape(-1))
        offset += size
    if offset != values.numel():
        raise ValueError(f"{key} contains {values.numel()} values but its shapes consume {offset}")
    output = torch.stack(output)
    return torch.stack([output.real, output.imag], dim=-1).to(dtype=dtype)
