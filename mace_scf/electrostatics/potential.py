"""Variational multipoles and a moment-free local electrostatic completion.

The optional response uses the released MACE backbone.  Its electronic energy
is a positive local quadratic form plus the Coulomb energy of a completed
source.  The charge multiplier is the electron chemical potential.  Coarse
density and complete potential are distinct observations of the same state;
DFT coarse density is not assumed to be the Poisson source of a PAW potential.

This is a linear electronic-response approximation at fixed geometry, not a
claim of an exact density functional or a general nonlinear SCF response.
"""
from __future__ import annotations

import logging
import math
from math import pi
from functools import lru_cache
from typing import Dict, Optional, Sequence

import torch
from e3nn import o3
from torch import nn
from torch.nn import functional as F
from graph_longrange.utils import FIELD_CONSTANT
from mace.tools.scatter import scatter_sum


class VariationalResponse(nn.Module):
    """Environment-dependent quadratic electronic functional.

    ``potential_widths`` are the Gaussian widths of the local potential basis.
    An empty list gives the ordinary monopole/dipole electronic functional.
    Completing the potential adds zero-charge, zero-dipole source functions,
    not another Fermi-level or work-function prediction head.
    """

    variational = True

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
        branch = data.get("dipole_correction_zfrac", positions.new_full((len(self.counts),), .5)).reshape(-1)
        self.branch = torch.where(torch.isfinite(branch), branch, .5).remainder(1.)
        zfrac = data.get("vacuum_zfrac", positions.new_full((len(self.counts),), .5)).reshape(-1)
        self.zfrac = torch.where(torch.isfinite(zfrac), zfrac, .5).remainder(1.)
        self.axial = (self.modes[None]*self.pbc[:, None]).abs().sum(-1) == 0
        self.axial = self.axial & active & self.slab[:, None]
        self.order = self.modes[None].expand(len(self.counts), -1, -1).gather(-1, self.axes[:, None, None].expand(-1, len(self.modes), 1)).squeeze(-1)
        self.attrs = self.pack(data["node_attrs"])
        self.cutoff = model.kspace_cutoff
        self.proto = torch.zeros_like(self.phase[..., 0])
        if bool(self.response.proto_fitted):
            structure = torch.einsum("gkn,gnz->gkz", self.phase, self.attrs.to(self.phase.dtype))
            radial = radial_basis(self.k2, self.cutoff)
            form_factors = torch.einsum("gkr,zr->gkz", radial, self.response.proto_coefficients)
            self.proto = (structure*form_factors).sum(-1)/self.volume[:, None]*active
        # Use the adjoint of the ACTUAL retained boundary field. Replacing
        # this with unwrapped z would change its uniform gauge and finite-grid
        # response, breaking the energy/EF/potential conjugacy for charged cells.
        self.normal_moment = self.adjoint(self.ramp(torch.ones_like(self.volume)))
        self.applied = self.ramp((self.external*self.normal).sum(-1))
        parallel = self.external-(self.external*self.normal).sum(-1)[:, None]*self.normal
        if bool((parallel.abs().max(-1).values > 1.e-10).any()):
            raise ValueError("A homogeneous field parallel to a periodic slab direction needs a separate boundary convention")
        self.counter_density = torch.zeros_like(self.proto)
        if "counter_charge" in data:
            counter = data["counter_charge"].reshape(-1)
            if "counter_slab_bounds" in data:
                bounds = data["counter_slab_bounds"].reshape(-1, 2)
                width = bounds[:, 1]-bounds[:, 0]
                if bool((width<=0).any()):
                    raise ValueError("Counter-charge slab bounds must have positive width")
                kz = (self.wave*self.normal[:, None]).sum(-1)
                midpoint = bounds.mean(-1)
                profile = torch.sinc(kz*width[:, None]/(2*math.pi))
                phase_counter = -kz*midpoint[:, None]
                planar = (self.modes[None]*self.pbc[:, None]).abs().sum(-1) == 0
                profile = profile*planar
            else:
                center = data["counter_charge_center"].reshape(-1, 3)
                width = data["counter_charge_width"].reshape(-1)
                if bool((width<=0).any()):
                    raise ValueError("Counter-charge Gaussian widths must be positive")
                phase_counter = -torch.einsum("gkc,gc->gk", self.wave, center)
                profile = torch.exp(-.5*self.k2*width[:, None].square())
            self.counter_density = (counter/self.volume)[:, None]*profile*torch.complex(phase_counter.cos(), phase_counter.sin())*self.mask
        counter_moment = self.volume*(self.counter_density.conj()*self.ramp(torch.ones_like(self.volume))).real.sum(-1)
        self.counter = -self.coulomb*self.counter_density+self.ramp(-self.slab_factor*counter_moment)
        self.counter_energy = (.5*self.volume*(self.coulomb*self.counter_density.abs().square()).sum(-1)
            +.5*self.slab_factor*counter_moment.square()
            -self.volume*(self.counter_density.conj()*(self.proto+self.applied)).real.sum(-1))

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
        return value + self.slab_factor[:, None, None]*moment[:, None, None]*self.normal_moment

    def diagonal(self):
        diag = (self.basis.abs().square()*self.coulomb[..., None]).sum(1)/self.volume[:, None]
        return diag[:, None, :]*self.present[..., None] + self.slab_factor[:, None, None]*self.normal_moment.square()

    def ramp(self, field):
        safe = torch.where(self.axial, self.order, torch.ones_like(self.order))
        phase = -2*math.pi*safe*self.branch[:, None]
        # Fourier coefficients of a centred sawtooth in electron-energy units.
        amplitude = field[:, None]*self.length[:, None]/(2*math.pi*safe)
        amplitude = amplitude * torch.exp(-.5*self.width**2*self.k2)*self.axial
        return torch.complex(-amplitude*phase.sin(), amplitude*phase.cos())

    def potentials(self, state):
        moment = (state*self.normal_moment).sum((1, 2))
        density = self.source(state, coarse=True)
        periodic = -self.coulomb*self.source(state)
        boundary = self.ramp(-self.slab_factor*moment)
        deformation = periodic + boundary + self.applied+self.counter
        return density, deformation, deformation + self.proto

    def plane(self, spectrum):
        phase = 2*math.pi*self.order*self.zfrac[:, None]
        return (spectrum*torch.complex(phase.cos(), phase.sin())*self.axial).real.sum(-1)


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
        strain = positions.new_zeros((len(data["ptr"])-1, 3, 3), requires_grad=True)
        strain_symmetric = .5*(strain+strain.transpose(-1, -2))
        data["positions"] = positions + torch.einsum("ni,nij->nj", positions, strain_symmetric[data["batch"]])
        cell = data["cell"].reshape(-1, 3, 3)
        data["cell"] = cell + cell@strain_symmetric
        edges = data["batch"][data["edge_index"][0]]
        data["shifts"] = data["shifts"]+torch.einsum("ni,nij->nj", data["shifts"], strain_symmetric[edges])
    local = model.local_part(data, compute_force=compute_force)
    geom = SpectralGeometry(model, data, local.positions)
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
    rhs = fixed_drive
    target = data["total_charge"].reshape(-1)
    if finite_steps and constant_charge:
        # Away from stationarity the KKT multiplier alone is not -dE/dQ.
        # Differentiate the actual finite-budget energy, just as for forces.
        target.requires_grad_(True)
    constraint = geom.constraint
    if not constant_charge:
        mu = data["fermi_level"].reshape(-1)-model.fermi_level_offset
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
              geom.normal_moment, geom.slab_factor, constraint)
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
    mu = multiplier+model.fermi_level_offset if constant_charge else data["fermi_level"].reshape(-1)
    electronic = (.5*state*operator(state)-state*fixed_drive).sum((1, 2))
    electronic = electronic-model.fermi_level_offset*charge+geom.counter_energy
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
    vacuum = geom.plane(total)
    output = {"energy": energy, "forces": forces, "stress": stress,
              "virials": -stress*geom.volume[:, None, None] if stress is not None else None,
              "density_coefficients": geom.unpack(state[..., :response.coarse_dim]),
              "dipole": dipole, "total_charge": charge, "fermi_level": mu,
              "workfunction": vacuum-mu, "vacuum_potential": vacuum,
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
    # Labels are attached only AFTER computing the physical state/observables.
    for key in ("fourier_density", "fourier_potential", "fourier_proto_potential"):
        target_fft = _target_fft(data, key, key+"_shape", geom.shape, positions.dtype)
        if target_fft is not None:
            output[key+"_dft"] = target_fft
            output[key+"_dft_mask"] = target_mode_mask(data[key+"_shape"], geom.modes)
    phi = output.get("fourier_potential_dft")
    proto = output.get("fourier_proto_potential_dft")
    if phi is not None:
        output["potential_mode_weight"] = potential_mode_weights(geom.wave, geom.cell, geom.pbc,
            data.get("potential_weight", positions.new_ones((len(mu), 3))).reshape(-1, 3), geom.mask)
    if phi is not None and proto is not None:
        output["fourier_total_potential_dft"] = phi+proto
        output["fourier_total_potential_dft_mask"] = output["fourier_potential_dft_mask"] & output["fourier_proto_potential_dft_mask"]
        reference = torch.view_as_complex((phi+proto).contiguous())/geom.ngrid
        output["vacuum_potential_dft"] = geom.plane(reference)
    return output


@torch.no_grad()
def initialize_response(model, loader, device):
    """Fit only training-set feature scales and frozen proto form factors."""
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
        if norm_graphs < 64:
            local = model.local_part(data, compute_force=False)
            norms += local.all_layer_feats.square().sum(0)
            count += len(local.positions)
            norm_graphs += len(data["ptr"])-1
        if not bool((data.get("fourier_proto_potential_weight", torch.zeros(1, device=device))>0).any()):
            continue
        geom = SpectralGeometry(model, data, data["positions"])
        target = _target_fft(data, "fourier_proto_potential", "fourier_proto_potential_shape", geom.shape, data["positions"].dtype)
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
    feature_irreps = model.products[0].linear.irreps_out
    for mul_ir, sl in zip(feature_irreps, feature_irreps.slices()):
        mul, ir = mul_ir
        values = norms[sl].reshape(mul, ir.dim).mean(-1).div(max(count, 1)).sqrt().clamp_min(1.e-3)
        response.feature_norms[sl] = values[:, None].expand(-1, ir.dim).reshape(-1)
    if observations:
        solution = torch.linalg.lstsq(gram.cpu(), rhs.cpu()[:, None],
                                     rcond=1.e-10, driver="gelsd").solution[:, 0].to(gram)
        response.proto_coefficients.copy_(solution.reshape_as(response.proto_coefficients))
        response.proto_fitted.fill_(True)
        error = (square-2*solution@rhs+solution@(gram@solution)).clamp_min(0)
        logging.info("Frozen training-only proto fit: spectral component RMS %.6g eV, observations %d", float((error/observations).sqrt()), observations)
    logging.info("Electronic functional: %d coarse + %d moment-free coefficients per atom; deployment uses %d steps", response.coarse_dim, response.state_irreps.dim-response.coarse_dim, int(response.deployment_steps))
def potential_mode_weights(
    k_vectors: torch.Tensor,
    cell: torch.Tensor,
    pbc: Optional[torch.Tensor],
    component_weights: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Lift ``config_potential_weight`` to reciprocal-space observables.

    The three components select potential observables, never boundary
    conditions.  One or two active components select the union of the
    corresponding one-dimensional reciprocal lines (planar-average profiles),
    while three active components select the complete retained 3-D spectrum.
    PBC enters the electrostatic solver elsewhere and does not override this
    user-supplied observation mask.
    """
    del pbc
    num_graphs = k_vectors.shape[0]
    weights = component_weights.reshape(num_graphs, 3).to(k_vectors)
    lattice_modes = torch.einsum(
        "bki,bji->bkj", k_vectors, cell.reshape(num_graphs, 3, 3)
    ) / (2.0 * pi)
    active_axis = lattice_modes.abs() > 1.0e-6
    active_count = active_axis.sum(dim=-1)
    selected = weights > 0.0
    selected_count = selected.sum(dim=-1)
    weight_sum = weights.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(weights.dtype).eps
    )

    # Axial-profile mode: exactly one reciprocal lattice component is nonzero.
    axial_lines = active_axis & (active_count == 1).unsqueeze(-1)
    axial_norm = weights * selected_count.clamp_min(1).to(weights)[:, None] / weight_sum
    axial_weight = torch.sum(
        axial_lines.to(weights) * axial_norm[:, None, :], dim=-1
    )

    # Full-3D mode: all three components active.  Directional non-unit weights
    # smoothly reweight mixed modes; [1,1,1] reduces exactly to one.
    mode_square = lattice_modes.square()
    full_norm = weights * 3.0 / weight_sum
    full_weight = torch.sum(mode_square * full_norm[:, None, :], dim=-1) / (
        mode_square.sum(dim=-1).clamp_min(torch.finfo(weights.dtype).eps)
    )
    output = torch.where(
        (selected_count == 3)[:, None], full_weight, axial_weight
    )
    output = torch.where((selected_count > 0)[:, None], output, torch.zeros_like(output))
    return output * mask.to(output)


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
