"""Checked electronic solves and exact derivatives from v366.

Linear moment screening is factored once per geometry. Finite trajectories
and converged-root implicit differentiation are distinct explicit policies.
"""
from dataclasses import dataclass
from typing import Callable, Tuple
import logging
import math
import torch
from torch.utils.checkpoint import checkpoint
from .coupled_response import sample_spectrum

def ensure_cuda_linalg(device):
    """Select cuSOLVER before any CUDA solve, including backward solves.

    PyTorch's default heuristics can choose MAGMA for large batched matrices.
    The v365 failure reports include a fatal MAGMA pointer-array error. Select
    the supported cuSOLVER/cuBLAS route explicitly and keep that process-wide
    preference for later autograd/audit calls. Never retry after a CUDA fault.
    CPU and other device backends are unchanged.
    """
    if torch.device(device).type != 'cuda':
        return
    preferred = torch.backends.cuda.preferred_linalg_library
    if preferred().name != 'Cusolver':
        preferred('cusolver')
        logging.info('SCF CUDA linear algebra: cuSOLVER/cuBLAS selected '
                     '(process-wide); exact cached matrix-solve derivatives; '
                     'torch=%s CUDA=%s', torch.__version__, torch.version.cuda)


def factor_linear_system(matrix):
    """Return checked, non-differentiated workspace for a live square matrix."""
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2] or matrix.shape[-1] == 0:
        raise ValueError('Linear operator must contain nonempty square matrices')
    ensure_cuda_linalg(matrix.device)
    with torch.no_grad():
        if not bool(torch.isfinite(matrix).all()):
            raise RuntimeError('Nonfinite constrained linear operator before LU factorization')
        lu, pivots, info = torch.linalg.lu_factor_ex(matrix)
        n = matrix.shape[-1]
        # Invalid pivots must never reach a native permutation/indexing kernel.
        valid = ((info == 0).all() & torch.isfinite(lu).all()
                 & (pivots >= 1).all() & (pivots <= n).all())
        if not bool(valid):
            raise RuntimeError('Constrained LU factorization failed its status/finite/pivot '
                               'checks; no shift or fallback applied')
    return lu, pivots


class _CachedLinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, matrix, lu, pivots, right, adjoint):
        ensure_cuda_linalg(matrix.device)
        solved = torch.linalg.lu_solve(lu, pivots, right, adjoint=adjoint)
        operator = matrix.mH if adjoint else matrix
        defect = operator @ solved - right
        scale = operator.abs() @ solved.abs() + right.abs()
        allowance = (64 * matrix.shape[-1] * torch.finfo(matrix.dtype).eps) * scale.clamp_min(1.)
        if not bool(torch.isfinite(solved).all() & (defect.abs() <= allowance).all()):
            raise RuntimeError('Constrained linear solve failed its finite/backward-error '
                               'check; no diagonal shift or solver substitution applied')
        ctx.adjoint = adjoint
        # Saving the output (not a detached internal copy) makes second and
        # higher derivatives include the solution's dependence on A and B.
        ctx.save_for_backward(matrix, lu, pivots, solved)
        return solved

    @staticmethod
    def backward(ctx, grad_solution):
        matrix, lu, pivots, solved = ctx.saved_tensors
        dual = _CachedLinearSolve.apply(matrix, lu, pivots, grad_solution, not ctx.adjoint)
        grad_matrix = None
        if ctx.needs_input_grad[0]:
            grad_matrix = -(solved @ dual.mH if ctx.adjoint else dual @ solved.mH)
        return grad_matrix, None, None, dual if ctx.needs_input_grad[3] else None, None


def solve_factored_system(matrix, lu, pivots, right):
    """Solve using factors from THIS matrix/evaluation; no batch broadcasting.

    Keep matrix attached. Factors are reusable workspace and carry no independent
    derivative; all operator dependence goes through the exact solve identity.
    """
    if (matrix.shape != lu.shape or pivots.shape != matrix.shape[:-1]
            or right.ndim != matrix.ndim or right.shape[:-1] != matrix.shape[:-1]
            or pivots.dtype != torch.int32):
        raise ValueError('Incompatible matrix, LU, pivot or right-hand-side shapes/dtype')
    if (matrix.device != lu.device or matrix.device != pivots.device or matrix.device != right.device
            or matrix.dtype != lu.dtype or matrix.dtype != right.dtype):
        raise ValueError('Linear solve inputs must share device and floating dtype')
    return _CachedLinearSolve.apply(matrix, lu.detach(), pivots, right, False)


def prepare_charge_factorization(softness, kernel, mask, constraint=None):
    """Factor the constrained geometry-only charge operator once per forward.

    The matrix and scaling retain parameter/coordinate derivatives. Numerical
    LU workspace is detached; solve_factored_system supplies the exact matrix
    derivative, including double backward. Nothing is cached across calls.
    LU handles the existing generally nonsymmetric Fourier slab operator;
    no symmetrization, diagonal shift, or alternate boundary convention is used.
    The optional constraint selects charge coordinates in a joint moment solve;
    dipoles then remain unconstrained. The default constrains every coordinate.
    """
    present = mask.bool()
    if kernel.shape != (*mask.shape, mask.shape[-1]) or softness.shape != mask.shape:
        raise ValueError('Charge kernel/softness dimensions disagree with node mask')
    if not bool(torch.isfinite(kernel).all() & torch.isfinite(softness).all()) or bool((softness[present] <= 0).any()):
        raise RuntimeError('Charge kernel/softness must be finite, with positive atomic softness')
    s = softness * mask
    total = s.sum(-1, keepdim=True)
    root = torch.sqrt(torch.where(present, s/total, torch.ones_like(s))) * mask
    n = mask.shape[-1]
    matrix = (kernel * root[..., None] * root[:, None, :]) * total[..., None]
    matrix = matrix + torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    constraint = mask if constraint is None else constraint
    if constraint.shape != mask.shape or not bool(torch.isfinite(constraint).all()) or bool((constraint.square().sum(-1)==0).any()):
        raise ValueError('Each graph needs a finite, nonzero constraint with the coordinate-mask shape')
    border = root * constraint
    upper = torch.cat((matrix, border[..., None]), -1)
    lower = torch.cat((border, torch.zeros_like(root[:, :1])), -1)[:, None, :]
    bordered = torch.cat((upper, lower), -2)
    lu, pivots = factor_linear_system(bordered)
    return bordered, root, lu, pivots


def screened_charge_step(current, raw_proposal, softness, kernel, mask, factorization=None, constraint=None):
    """Precondition the neutral charge residual without changing fixed points.

    (S^-1+C) delta + lambda*1 = S^-1 (q_raw-q), sum(delta)=sum(q_raw-q).
    q_screened=q+delta. S is frozen only with respect to the within-step charge
    solve, NOT detached from model/coordinate derivatives. Nonlinear S and all
    other response channels are reevaluated on every outer-SCF update.
    An optional factorization may be reused only for exactly the same S, C and
    mask; the geometry-hardness model supplies those live per-evaluation factors.

    Whiten with sqrt(s/sum(s)) and normalize the constraint to unit norm.
    This is algebraically equivalent to the existing sqrt(s) system.
    Solve the bordered charge-constrained system directly. Unlike inverting
    the unconstrained block and using its Schur complement, this also handles
    a singular/unphysical uniform-charge direction when the fixed-Q tangent
    problem is well defined. It is one direct (N_atom+1) solve, not a nonlinear
    root iteration. The multiplier returned here corrects the frozen-field
    multiplier; the reported Fermi level is still read from the final state.
    """
    present = mask.bool()
    constraint = mask if constraint is None else constraint
    if current.shape != mask.shape or softness.shape != mask.shape or raw_proposal.shape != mask.shape:
        raise ValueError('Charge-block inputs must have shape [graphs, max_atoms]')
    if kernel.shape != (*mask.shape, mask.shape[-1]):
        raise ValueError('Charge kernel must have shape [graphs, max_atoms, max_atoms]')
    if not bool(torch.isfinite(kernel).all() & torch.isfinite(softness).all()) or bool((softness[present] <= 0).any()):
        raise RuntimeError('Charge-block kernel/softness must be finite, with positive atomic softness')
    s = softness * mask
    total = s.sum(-1, keepdim=True)
    # Unit-norm bordered constraint. The old [sqrt(s),0] border acquired a
    # spurious 1/sum(s) condition number for a uniformly stiff material, even
    # when its physical charge-neutral system was well conditioned.
    if factorization is None:
        factorization = prepare_charge_factorization(softness, kernel, mask, constraint)
    bordered, root, lu, pivots = factorization
    residual = (raw_proposal-current)*mask
    rhs = residual / torch.where(present, root, torch.ones_like(root))
    right = torch.cat((rhs, (constraint*residual).sum(-1, keepdim=True)), -1)[..., None]
    if not bool(torch.isfinite(right).all()):
        raise RuntimeError('Charge-block residual is nonfinite before the constrained solve')
    solved = solve_factored_system(bordered, lu, pivots, right)
    delta = root*solved[:, :-1, 0]
    multiplier = solved[:, -1, 0] / total[:, 0]
    result = (current+delta)*mask
    # The raw proposal already has the prescribed Q. Only roundoff is removed.
    result = result + constraint*((constraint*(raw_proposal-result)).sum(-1)/constraint.square().sum(-1))[:, None]
    return result, multiplier


def moment_kernel(geometry, kernels, mask, sigma):
    co,si,wave=geometry[:3]
    kr,ki=(k[..., :4] for k in kernels)
    source_re=(co[...,None]*kr[:,:,None,:]+si[...,None]*ki[:,:,None,:]).flatten(-2)
    source_im=(co[...,None]*ki[:,:,None,:]-si[...,None]*kr[:,:,None,:]).flatten(-2)
    receiver=geometry[6][...,0,None,None]
    observer_re=torch.cat((co[...,None],-sigma*si[...,None]*wave[:,:,None,:]),-1)*receiver
    observer_im=torch.cat((-si[...,None],-sigma*co[...,None]*wave[:,:,None,:]),-1)*receiver
    matrix=-(observer_re.flatten(-2).transpose(1,2)@source_re
             +observer_im.flatten(-2).transpose(1,2)@source_im)
    unit_v,unit_e=sample_spectrum(geometry[8],geometry)
    observed=torch.cat((unit_v[...,:1],-sigma*unit_e[...,0,:]),-1).flatten(-2)
    dipole=torch.cat((geometry[10][...,None],sigma*geometry[11][:,None,:].expand(-1,mask.shape[1],-1)),-1).flatten(-2)
    matrix=matrix-observed[...,None]*dipole[:,None,:]
    present=mask[...,None].expand(-1,-1,4).flatten(-2)
    return matrix*present[...,None]*present[:,None,:]


def moment_coordinates(reference, mask):
    # Last entries remain [dipole softness, chi0, charge softness].
    softness=torch.cat((reference[...,-1:],reference[...,-3:-2].expand(-1,-1,3)),-1)
    present=mask[...,None].expand_as(softness)
    constraint=torch.cat((mask[...,None],torch.zeros_like(softness[...,1:])), -1)
    return softness.flatten(-2),present.flatten(-2),constraint.flatten(-2)


def factor_moments(reference, kernel, mask):
    softness,present,constraint=moment_coordinates(reference,mask)
    return prepare_charge_factorization(softness,kernel,present,constraint)


def screen_moments(current, proposal, reference, kernel, mask, factors):
    softness,present,constraint=moment_coordinates(reference,mask)
    value,_=screened_charge_step(current[...,:4].flatten(-2),proposal[...,:4].flatten(-2),
        softness,kernel,present,factors,constraint)
    return torch.cat((value.reshape(*mask.shape,4),proposal[...,4:]),-1)


@dataclass(frozen=True)
class RootOptions:
    max_steps: int = 100
    tolerance: float = 1.e-7
    mixing: float = .5
    history: int = 5
    linear_tolerance: float = 1.e-9
    linear_max_steps: int = 160
    linear_restart: int = 24
    fixed_steps: bool = False

    def __post_init__(self):
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps < 1:
            raise ValueError("Coupled SCF step count must be a positive integer")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0 or not 0 < self.mixing <= 1:
            raise ValueError("Invalid coupled fixed-point solver options")
        if not isinstance(self.fixed_steps, bool):
            raise TypeError("fixed_steps must be bool")
        if min(self.linear_max_steps, self.linear_restart) < 1 or self.linear_tolerance <= 0:
            raise ValueError("Invalid coupled adjoint solver options")


class SCFConvergenceError(RuntimeError):
    """A valid strict root iteration exhausted its convergence budget."""


class SCFNumericalError(RuntimeError):
    """The selected trajectory became nonfinite or catastrophically amplified.

    This is an error, not a clipped state, accepted earlier iterate, skipped batch,
    altered objective or solver-mode substitution.
    """


def _trajectory_scale(initial):
    # A precision-scaled runaway budget in the defined state coordinates, not
    # a physical bound or a theorem that smaller trajectories are stable. Large
    # state amplification precedes high-degree energy/force-loss contractions.
    # This fail-fast convention never changes an accepted finite trajectory.
    x=initial.detach()
    if not bool(torch.isfinite(x).all()):
        raise SCFNumericalError('SCF initial state is not finite')
    peak=x.abs().amax() if x.numel() else x.new_zeros(())
    return peak.clamp_min(1.) / math.sqrt(torch.finfo(x.dtype).eps)


def _check_trajectory(state,limit,step,kind):
    with torch.no_grad():
        peak=state.detach().abs().amax() if state.numel() else state.new_zeros(())
        # A single scalar decision covers NaN/Inf and amplitude. Avoid separate
        # device synchronizations for finiteness and magnitude on every step.
        if bool((~torch.isfinite(peak)) | (peak>limit)):
            flat=state.detach().abs().reshape(-1)
            index=int(torch.nan_to_num(flat,nan=float('inf'),posinf=float('inf')).argmax()) if flat.numel() else 0
            per_graph=state[0].numel() if state.ndim>=2 and len(state) else max(1,state.numel())
            graph=index//per_graph if state.ndim>=2 else 0
            detail = ""
            if state.ndim == 3 and state.shape[-1] >= 4 and (state.shape[-1]-4) % 4 == 0:
                c = (state.shape[-1]-4)//4
                sample = state[graph].detach()
                blocks = {"q": (0, 1), "d_scaled": (1, 4),
                          "scalar_potential": (4, 4+c), "vector_potential": (4+c, state.shape[-1])}
                peaks = {name: float(sample[:, a:b].abs().max()) for name, (a, b) in blocks.items() if b > a}
                detail = f" Failing-graph block maxima={peaks}."
            raise SCFNumericalError(f'{kind} SCF numerical runaway at step={step}, graph={graph}: '
                f'max|state|={float(peak):.6e}, fixed initial-scale precision budget={float(limit):.6e}. '
                'Finite-step unrolling is not a convergence solver. No state was clipped or substituted; '
                'check the requested update count and the nonlinear response. Do not relax spectrum parity.' + detail)


def _stopping_residual(function, state, args, proposal):
    physical = getattr(function, "convergence_residual", None)
    return proposal-state if physical is None else physical(state,*args)


def _norm(x):
    return torch.linalg.vector_norm(x.reshape(-1))


def solve_root(function: Callable, initial: torch.Tensor, args: Tuple[torch.Tensor, ...],
               options: RootOptions):
    """Safeguarded Anderson iteration.  The residual is F(x)-x, NOT a mixed step.

    Call in no_grad for implicit solving.  This routine does not detach inputs
    itself and is not used as a surrogate derivative for implicit force training.
    Nonconvergence is an error, never a silent rollback to another physical state.
    """
    x = initial.clone()
    limit = _trajectory_scale(initial)
    xs, fs = [], []
    previous_x = previous_f = None
    previous_error = float("inf")
    rejected = 0
    for step in range(options.max_steps + 1):
        _check_trajectory(x,limit,step,'strict-root iterate')
        f = function(x, *args)
        _check_trajectory(f,limit,step,'strict-root proposal')
        residual = f - x
        stopping = _stopping_residual(function, x, args, f)
        error = float(stopping.detach().abs().max()) if x.numel() else 0.
        if not torch.isfinite(residual).all() or not torch.isfinite(stopping).all():
            raise RuntimeError(f"Coupled SCF produced a nonfinite state at step {step}")
        if error <= options.tolerance:
            return x, {"iterations": step, "residual": error, "rejected": rejected}
        if step == options.max_steps:
            break
        # Reject an extrapolation that substantially worsens the fixed-point
        # residual.  A fresh damped step uses the *same* map and charge law.
        if previous_x is not None and error > 1.5 * previous_error and len(xs) > 1:
            x = previous_x + options.mixing * (previous_f - previous_x)
            xs, fs = [], []
            previous_x = previous_f = None
            rejected += 1
            continue
        previous_x, previous_f, previous_error = x, f, error
        xs.append(x); fs.append(f)
        if len(xs) > max(1, options.history):
            xs.pop(0); fs.pop(0)
        if len(xs) < 2 or options.history < 2:
            x = x + options.mixing * residual
            continue
        X = torch.stack([z.reshape(-1) for z in xs])
        F = torch.stack([z.reshape(-1) for z in fs])
        R = F - X
        gram = R @ R.T
        scale = gram.diagonal().mean().clamp_min(torch.finfo(x.dtype).tiny)
        gram = gram + (1.e-4 * scale) * torch.eye(len(xs), dtype=x.dtype, device=x.device)
        weights = torch.linalg.solve(gram, torch.ones(len(xs), dtype=x.dtype, device=x.device))
        denom = weights.sum()
        if not torch.isfinite(weights).all() or denom.abs() <= torch.finfo(x.dtype).eps:
            x = x + options.mixing * residual
            xs, fs = [], []
        else:
            weights = weights / denom
            x = ((1 - options.mixing) * (weights @ X) + options.mixing * (weights @ F)).reshape_as(x)
    raise SCFConvergenceError(
        f"Coupled SCF did not converge in {options.max_steps} steps: "
        f"max |F(x)-x|={error:.3e}, tolerance={options.tolerance:.3e}. "
        "Increase the SCF budget or reduce mixing; no state/backend was substituted.")


def gmres(operator: Callable, rhs: torch.Tensor, options: RootOptions):
    """Restarted matrix-free GMRES with a verified true residual.

    The Krylov vectors are constants during an implicit solve.  Derivatives of
    the solution are supplied by _LinearSolve, including its double backward.
    """
    shape = rhs.shape
    b = rhs.reshape(-1)
    x = torch.zeros_like(b)
    bnorm = _norm(b)
    if float(bnorm) == 0.:
        return x.reshape(shape)
    tol = max(options.linear_tolerance, 20. * torch.finfo(b.dtype).eps)
    target = tol * bnorm
    total = 0
    def matvec(v):
        return operator(v.reshape(shape)).reshape(-1)
    while total < options.linear_max_steps:
        r = b - matvec(x)
        beta = _norm(r)
        if beta <= target:
            return x.reshape(shape)
        length = min(options.linear_restart, options.linear_max_steps - total, b.numel())
        vectors = [r / beta]
        H = b.new_zeros((length + 1, length))
        # Small least-squares problems use normal QR/lstsq, never normal equations.
        for j in range(length):
            v = matvec(vectors[j])
            for _ in range(2):  # reorthogonalization matters near a solved root
                for k in range(j + 1):
                    coeff = torch.dot(vectors[k], v)
                    H[k, j] += coeff
                    v = v - coeff * vectors[k]
            hnext = _norm(v)
            H[j + 1, j] = hnext
            total += 1
            beta_e1 = b.new_zeros(j + 2); beta_e1[0] = beta
            small = H[:j + 2, :j + 1]
            # 'gels' is also supported on CUDA and Arnoldi columns have full rank
            # until happy breakdown.  QR handles that breakdown without a zero
            # trailing row being promoted to a column.
            coeffs = torch.linalg.lstsq(small, beta_e1, driver="gels").solution
            estimate = _norm(beta_e1 - small @ coeffs)
            candidate = x + torch.stack(vectors[:j + 1], dim=1) @ coeffs
            breakdown = float(hnext) <= 10. * torch.finfo(b.dtype).eps
            if estimate <= target or breakdown or j == length - 1:
                true_error = _norm(b - matvec(candidate))
                if true_error <= target:
                    return candidate.reshape(shape)
                if breakdown or j == length - 1:
                    x = candidate
                    break
            vectors.append(v / hnext)
    relative = float(_norm(b - matvec(x)) / bnorm)
    raise RuntimeError(f"Coupled SCF adjoint did not converge: relative residual={relative:.3e} "
                       f"after {total} Krylov steps (required {tol:.3e})")


def _linear_operator(function, transpose, state, args):
    """A = I-dF/dx or its transpose at frozen state/inputs."""
    with torch.enable_grad():
        z = state.detach().requires_grad_(True)
        f = function(z, *(a.detach() for a in args))
        if not f.requires_grad:
            return lambda v: v
        if transpose:
            def operator(v):
                with torch.enable_grad():
                    jt = torch.autograd.grad(f, z, v, retain_graph=True, allow_unused=True)[0]
                return v if jt is None else v - jt
        else:
            # Reverse-over-reverse JVP works for operations without forward-AD
            # kernels.  It does not materialize a Jacobian or a Hessian.
            seed = torch.zeros_like(f, requires_grad=True)
            jt_seed = torch.autograd.grad(f, z, seed, create_graph=True, retain_graph=True,
                                          allow_unused=True)[0]
            def operator(v):
                if jt_seed is None or not jt_seed.requires_grad:
                    return v
                with torch.enable_grad():
                    jv = torch.autograd.grad(jt_seed, seed, v, retain_graph=True,
                                             allow_unused=True)[0]
                return v if jv is None else v - jv
    return operator


def _proxies(values):
    # An intermediate proxy makes the following autograd.grad a PARTIAL
    # derivative, while preserving original-input dependence for double backward.
    # Leaf constants also need a proxy for the local Jacobian, but no derivative
    # is returned to a non-differentiable original input.
    return [x + 0. if x.requires_grad else x.detach() for x in values]


class _LinearSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, options, transpose, rhs, state, *args):
        solution = gmres(_linear_operator(function, transpose, state, args), rhs, options)
        ctx.function, ctx.options, ctx.transpose = function, options, transpose
        ctx.save_for_backward(solution, rhs, state, *args)
        return solution

    @staticmethod
    def backward(ctx, grad_solution):
        solution, rhs, state, *args = ctx.saved_tensors
        higher = torch.is_grad_enabled()
        with torch.enable_grad():
            adjoint = _LinearSolve.apply(ctx.function, ctx.options, not ctx.transpose,
                                        grad_solution, state, *args)
            z, *xs = _proxies([state, *args])
            f = ctx.function(z, *xs)
            left, right = (solution, adjoint) if ctx.transpose else (adjoint, solution)
            jt_left = (torch.autograd.grad(f, z, left, create_graph=True,
                                         retain_graph=True, allow_unused=True)[0]
                       if f.requires_grad and z.requires_grad else None)
            if jt_left is None or not jt_left.requires_grad:
                derivatives = [None] * (1 + len(args))
            else:
                variables = [z, *xs]
                chosen = [i for i, value in enumerate(variables) if value.requires_grad]
                calculated = torch.autograd.grad((jt_left * right).sum(),
                    [variables[i] for i in chosen], create_graph=higher, allow_unused=True)
                derivatives = [None] * len(variables)
                for i, derivative in zip(chosen, calculated): derivatives[i] = derivative
        derivatives = tuple(d if original.requires_grad else None
                            for d, original in zip(derivatives, [state, *args]))
        return (None, None, None, adjoint, *derivatives)


class _ImplicitRoot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, function, options, initial, *args):
        if options.fixed_steps:
            raise ValueError("implicit_root cannot differentiate an unconverged finite trajectory; "
                             "use unroll_fixed_steps explicitly")
        state, stats = solve_root(function, initial, args, options)
        # The root belongs to the physical constitutive equation, not the
        # nonlinear preconditioning path. Differentiate that equation directly.
        # The forward solver has already checked its raw residual.
        ctx.function, ctx.options = getattr(function, "raw", function), options
        ctx.save_for_backward(state, *args)
        # Statistics are not model inputs, never part of a training target.
        info = initial.new_tensor([stats["iterations"], stats["residual"], stats["rejected"]])
        ctx.mark_non_differentiable(info)
        return state, info

    @staticmethod
    def backward(ctx, grad_state, grad_info):
        state, *args = ctx.saved_tensors
        higher = torch.is_grad_enabled()
        with torch.enable_grad():
            z, *xs = _proxies([state, *args])
            f = ctx.function(z, *xs)
            adjoint = _LinearSolve.apply(ctx.function, ctx.options, True, grad_state,
                                        state, *args)
            chosen = [i for i, value in enumerate(xs) if value.requires_grad]
            calculated = (torch.autograd.grad(f, [xs[i] for i in chosen], adjoint,
                                            create_graph=higher, allow_unused=True)
                          if chosen and f.requires_grad else [None] * len(chosen))
            derivatives = [None] * len(xs)
            for i, derivative in zip(chosen, calculated): derivatives[i] = derivative
        derivatives = tuple(d if a.requires_grad else None for d, a in zip(derivatives, args))
        return (None, None, None, *derivatives)


def implicit_root(function, initial, *args, options=None):
    """Return a verified root and [iterations, residual, rejected extrapolations].

    Every differentiable dependency MUST be passed in args. The starting guess
    is not an implicit variable. A failed root or verified adjoint solve is an
    error, not an approximate gradient, restored state, or backend substitution.
    """
    options = options or RootOptions()
    if options.fixed_steps:
        raise ValueError('implicit_root requires a converged root, not fixed_steps=True')
    return _ImplicitRoot.apply(function, options, initial, *args)

