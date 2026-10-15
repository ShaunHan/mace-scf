"""Cubic molecule-in-box correction for the represented Gaussian charge density.

Makov--Payne / Dabo correction through order L**-3 (vacuum dielectric):
E_corr = A Q**2/2 + B (|p|**2 - Q M)/2,
A = k_e alpha/L, B = 4 pi k_e/(3 V), M = integral rho |r-center|**2.
This is the IDIPOL=4 type multipole correction, not a VASP PAW implementation.
The local field is sampled analytically; Fourier outputs contain its periodic
polynomial continuation, truncated to the model's observation cutoff.
"""
import math

import torch
from graph_longrange.utils import FIELD_CONSTANT


def enabled(data, volume):
    flag = data.get("molecular_correction")
    if flag is not None:
        return flag.reshape(-1).bool()
    return torch.full_like(volume, "dipole_correction_center" in data, dtype=torch.bool)


class MolecularCorrection:
    def __init__(self, geometry, data, positions):
        self.active = enabled(data, geometry.volume)
        self.cell, self.volume = geometry.cell, geometry.volume
        gram = self.cell @ self.cell.transpose(-1, -2)
        self.length = self.volume.pow(1./3.)
        cubic = self.length[:, None, None].square()*torch.eye(3, device=gram.device, dtype=gram.dtype)
        relative_error = ((gram-cubic).abs()/self.length[:, None, None].square()).amax((1, 2))
        if bool((self.active & (relative_error > 1.e-7)).any()):
            raise ValueError("dipole_correction_center currently requires a cubic molecular supercell")
        if bool((self.active & ~geometry.pbc.all(-1)).any()):
            raise ValueError("dipole_correction_center requires pbc=True; a truly open system needs no image correction")
        self.center = data["dipole_correction_center"].reshape(-1, 3).to(positions)
        if not bool(torch.isfinite(self.center).all()):
            raise ValueError("dipole_correction_center must contain finite Cartesian coordinates")
        self.present = geometry.present
        self.relative = self.relative_positions(geometry.pack(positions))
        self.width = geometry.width
        self.a = FIELD_CONSTANT/(4*math.pi)*2.837297479480619/self.length*self.active
        self.b = FIELD_CONSTANT/(3*self.volume)*self.active
        # Moments are (Q, p_x, p_y, p_z, M); E = m.T metric m / 2.
        self.metric = positions.new_zeros((len(self.volume), 5, 5))
        self.metric[:, 0, 0] = self.a
        for j in range(1, 4):
            self.metric[:, j, j] = self.b
        self.metric[:, 0, 4] = self.metric[:, 4, 0] = -.5*self.b
        self.counter = positions.new_zeros((len(self.volume), 5))
        charge = data.get("counter_charge", torch.zeros_like(self.volume)).reshape(-1)
        axis = data.get("counter_charge_axis", torch.full_like(charge, -1, dtype=torch.long)).reshape(-1)
        if bool((self.active & (axis >= 0)).any()) or "counter_slab_bounds" in data:
            raise ValueError("A molecular correction needs a localized source, not a counter-charge sheet")
        if bool((charge != 0).any()):
            center = data["counter_charge_center"].reshape(-1, 1, 3).to(positions)
            relative = self.relative_positions(center)[:, 0]
            width = data["counter_charge_width"].reshape(-1).to(positions)
            self.counter = torch.cat((charge[:, None], charge[:, None]*relative,
                (charge*(relative.square().sum(-1)+3*width.square()))[:, None]), -1)
        self.modes, self.mask = geometry.modes, geometry.mask

    def relative_positions(self, positions):
        relative = positions-self.center[:, None]
        fraction = relative @ torch.linalg.inv(self.cell)
        return relative-torch.floor(fraction+.5) @ self.cell

    def source_map(self, dimension, *, coarse_dim=4, scale=1., order=(0, 1, 2), completion=False):
        """Map atomic Gaussian coefficients to exact moments, including widths."""
        r = self.relative
        result = r.new_zeros((*r.shape[:2], dimension, 5))
        result[..., 0, 0] = 1.
        result[..., 0, 1:4] = r
        result[..., 0, 4] = r.square().sum(-1)+3*self.width**2
        if coarse_dim == 4:
            for j, axis in enumerate(order):
                result[..., j+1, axis+1] = scale
                result[..., j+1, 4] = 2*scale*r[..., axis]
        if completion:
            # Variational completion sources are -sigma_rho**2 Laplacian(g).
            result[..., coarse_dim::4, 4] = -6*self.width**2
        return result*self.present[..., None, None]

    def sample_map(self, widths):
        """Map total moments to (phi, -grad phi) at each Gaussian receiver."""
        widths = torch.as_tensor(widths, device=self.relative.device, dtype=self.relative.dtype)
        r = self.relative
        output = r.new_zeros((*r.shape[:2], len(widths), 4, 5))
        a, b = self.a[:, None, None], self.b[:, None, None]
        output[..., 0, 0] = -a+.5*b*(r.square().sum(-1)[..., None]+3*widths.square())
        output[..., 0, 1:4] = -b[..., None]*r[:, :, None]
        output[..., 0, 4] = .5*b
        for j in range(3):
            output[..., j+1, 0] = -b*r[..., j, None]
            output[..., j+1, j+1] = b
        return output*self.present[..., None, None, None]

    def energy(self, moments):
        return .5*torch.einsum("gi,gij,gj->g", moments, self.metric, moments)

    def spectrum(self, moments):
        """Electron-referenced correction, including its determined constant."""
        modes = self.modes
        nonzero = (modes != 0).sum(-1)
        zero = nonzero == 0
        fraction = torch.einsum("gi,gij->gj", self.center, torch.linalg.inv(self.cell))
        linear = torch.zeros((*self.mask.shape, 3), device=moments.device,
                             dtype=torch.complex128 if moments.dtype == torch.float64 else torch.complex64)
        square = torch.zeros_like(linear[..., 0])
        for j in range(3):
            order = modes[:, j]
            safe = torch.where(order != 0, order, 1.)
            select = (nonzero == 1) & (order != 0)
            phase = -2*math.pi*safe[None]*(fraction[:, j, None]+.5)
            phase = torch.complex(phase.cos(), phase.sin())*select[None]
            ramp = 1j*self.length[:, None]/(2*math.pi*safe[None])*phase
            direction = self.cell[:, j]/self.length[:, None]
            linear = linear+ramp[..., None]*direction[:, None]
            square = square+self.length[:, None].square()/(2*math.pi**2*safe[None].square())*phase
        square = square+zero[None]*self.length[:, None].square()/4
        q, p, second = moments[:, 0], moments[:, 1:4], moments[:, 4]
        phi = ((-self.a*q+.5*self.b*second)[:, None]*zero[None]
               +.5*(self.b*q)[:, None]*square
               -self.b[:, None]*(linear*p[:, None]).sum(-1))
        return phi*self.mask


def moment_sum(state, source_map):
    return torch.einsum("gnd,gndm->gm", state[..., :source_map.shape[2]], source_map)
