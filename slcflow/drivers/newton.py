"""Global Newton driver (Theory Manual section 6.3; ARCH-5.3).

Solves the section 6.1 residual ``R(x) = 0`` *simultaneously* (as opposed to
the classical driver's nested scalar-solve/reposition scheme), which is the
robust path near choke/stall where the classical fixed point stalls, and the
route to an exportable Jacobian. Newton with line-search globalization over
the pure ``ResidualAssembler.residual`` (AD-3 is what makes this well-posed —
no hidden state to desynchronize between the residual and its finite-
difference Jacobian).

Warm start is **mandatory** (ARCH-5.3): Newton is a local method; the seed is
a classical-driver iterate or a neighbouring converged operating point.
Closures stay lagged across an outer quasi-Newton loop (section 6.3); within
one inner Newton solve the ``FrozenInputs`` are fixed, so the inner system is
exactly the frozen-closure residual (and, at Tier 2, the exact full system —
the curvature/lean lag terms carry zero flags).

Jacobian: **dense forward-difference** here. Section 6.3 permits dense Newton
for the typical problem size (N_sl * N_qo <~ 10^3 unknowns); it is
the unconditionally-correct baseline against which the ARCH-5.3
colored-finite-difference Jacobian (exploiting the near-block-tridiagonal
station-index structure) must be validated column-for-column when it lands.
That optimization is the recorded next step, not a correctness prerequisite.

Globalization detail (the M2/M3 carryover): a Newton trial step can drive a
q-o's nodes non-monotone (a *crossing streamline*), which the assembler's
PCHIP construction cannot represent — the classical driver structurally never
produces one, but Newton can. The line search treats a non-monotone (or
non-finite-residual) trial as an infeasible point and backtracks, so such
steps are rejected rather than raising (AD-10 letter closed for the Newton
path).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # driver layer: orchestration, not residual path  # ad6: allow

from ..assembly.assembler import ResidualAssembler
from ..assembly.inputs import ClosureFields, FrozenInputs
from ..assembly.pack import unpack
from ..diagnostics.record import (ConvergenceRecord, IterationRecord,
                                  SolveStatus)
from ..errors import ConfigError
from ..grid.core import MetricsConfig
from ..transport.streamwise import TransportFields, TransportStep, row_steps, sweep
from ..types import FidelityConfig, MassFlowSpec
from .classical import (ClassicalResult, _closure_norm, _evaluate_rows,
                        _resolve_rows)

__all__ = ["NewtonConfig", "newton_solve", "solve_newton"]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class NewtonConfig:
    """Newton driver settings (section 6.3).

    Parameters
    ----------
    max_outer : quasi-Newton outer iterations (closure re-lag + transport
        sweep). One pass suffices for static-closure (duct) problems.
    max_iter : inner Newton iterations per outer pass.
    tol_res : target infinity norm of the SCALED residual (continuity rows by
        ``mdot``, position rows by ``mdot / 2*pi``), so the two row families
        are compared on one dimensionless footing.
    tol_closure : outer closure-update norm target (section 6.2.5).
    fd_rel, fd_abs : forward-difference step ``h = fd_rel*|x_k| + fd_abs``.
    max_backtrack : Armijo backtracks before declaring the step infeasible.
    armijo : sufficient-decrease constant on the merit ``0.5 ||r_scaled||^2``.
    closure_relax : under-relaxation of the lagged closure outputs between
        outer passes (section 6.2.4; same rationale as the classical driver).
    """

    max_outer: int = 40
    max_iter: int = 60
    tol_res: float = 1e-9
    tol_closure: float = 1e-9
    fd_rel: float = 1e-7
    fd_abs: float = 1e-9
    max_backtrack: int = 40
    armijo: float = 1e-4
    closure_relax: float = 0.25

    def __post_init__(self):
        if self.max_iter < 1 or self.max_outer < 1:
            raise ConfigError("max_iter and max_outer must be >= 1")
        if not (0.0 < self.closure_relax <= 1.0):
            raise ConfigError(
                f"closure_relax must be in (0, 1], got {self.closure_relax}")


# ---------------------------------------------------------------------------
# State geometry helpers (independent of the assembler's spline-raising path)
# ---------------------------------------------------------------------------
def _q_full_from_x(x, frozen: FrozenInputs):
    """Reconstruct the nodal q-positions from ``x`` WITHOUT building the
    section 5.3 interpolants — so the monotonicity guard can inspect a trial
    step that the assembler's PCHIP construction would reject by raising."""
    n_sl, n_qo = frozen.n_sl, frozen.n_qo
    _, q_int = unpack(x, n_sl, n_qo)
    if n_sl == 1:
        return frozen.q_fixed                    # meanline: single fixed node
    lengths = np.array([qo.length
                        for qo in frozen.topology.flowpath.qo_curves])
    return np.concatenate([np.zeros((1, n_qo)), q_int,
                           lengths.reshape(1, n_qo)], axis=0)


def _is_feasible_q(x, frozen: FrozenInputs) -> bool:
    """A trial ``x`` is feasible iff every q-o's nodes stay strictly
    increasing (no crossing streamline, section 6.1 / assembler contract)."""
    if frozen.n_sl <= 1:
        return True
    q_full = _q_full_from_x(x, frozen)
    return bool(np.all(np.diff(q_full, axis=0) > 0.0))


def _residual_scale(frozen: FrozenInputs):
    """Per-row scale making continuity (``~mdot``) and position (``~mdot/2pi``)
    residual rows dimensionless-comparable (section 6.2.5 in one norm)."""
    n_sl, n_qo = frozen.n_sl, frozen.n_qo
    mdot = frozen.spec.mdot
    n_pos = max(n_sl - 2, 0) * n_qo
    return np.concatenate([np.full(n_qo, mdot),
                           np.full(n_pos, mdot / _TWO_PI)])


def _safe_residual(asm: ResidualAssembler, x, scale):
    """Scaled residual at ``x``, or ``None`` if the point is infeasible
    (crossing streamline) or produces a non-finite residual (out-of-domain
    Vm along the ODE). No exception crosses this boundary (AD-10)."""
    if not _is_feasible_q(x, asm.frozen):
        return None
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        r = asm.residual(x)
    if not np.all(np.isfinite(r)):
        return None
    return r / scale


# ---------------------------------------------------------------------------
# Inner Newton core
# ---------------------------------------------------------------------------
def _fd_jacobian(asm, x, r0, scale, config: NewtonConfig):
    """Dense forward-difference Jacobian of the SCALED residual (section 6.3).

    Falls back to a backward step for any column whose forward perturbation is
    infeasible; returns ``None`` if a column cannot be formed at all (the
    caller then reports NUMERICAL_FAILURE rather than stepping on a corrupt
    Jacobian)."""
    n = x.size
    jac = np.empty((r0.size, n))
    for k in range(n):
        h = config.fd_rel * abs(x[k]) + config.fd_abs
        xf = x.copy()
        xf[k] += h
        rf = _safe_residual(asm, xf, scale)
        if rf is not None:
            jac[:, k] = (rf - r0) / h
            continue
        xb = x.copy()
        xb[k] -= h
        rb = _safe_residual(asm, xb, scale)
        if rb is None:
            return None
        jac[:, k] = (r0 - rb) / h
    return jac


def newton_solve(asm: ResidualAssembler, x0, config: NewtonConfig = NewtonConfig()):
    """Drive ``asm.residual(x) = 0`` from warm start ``x0`` (section 6.3).

    Returns ``(x, status, iteration_records)``. ``status`` is CONVERGED,
    MAX_ITER, or NUMERICAL_FAILURE (a singular/unobtainable Jacobian or a step
    the line search cannot make feasible-and-decreasing). Pure over the frozen
    assembler — closures/transport are the caller's outer concern.
    """
    scale = _residual_scale(asm.frozen)
    x = np.array(x0, dtype=float)
    records = []
    r = _safe_residual(asm, x, scale)
    if r is None:
        return x, SolveStatus.NUMERICAL_FAILURE, records
    for it in range(1, config.max_iter + 1):
        rnorm = float(np.max(np.abs(r)))
        if rnorm < config.tol_res:
            return x, SolveStatus.CONVERGED, records
        jac = _fd_jacobian(asm, x, r, scale, config)
        if jac is None:
            records.append(_rec(it, rnorm, 0.0))
            return x, SolveStatus.NUMERICAL_FAILURE, records
        try:
            dx = np.linalg.solve(jac, -r)
        except np.linalg.LinAlgError:
            dx, *_ = np.linalg.lstsq(jac, -r, rcond=None)  # singular fallback
        # Armijo backtracking line search on 0.5||r||^2, with the
        # feasibility (monotone-q) guard folded in: an infeasible trial is a
        # merit of +inf, so it backtracks (the crossing-streamline rejection).
        phi0 = 0.5 * float(r @ r)
        alpha, accepted = 1.0, None
        for _ in range(config.max_backtrack):
            r_try = _safe_residual(asm, x + alpha * dx, scale)
            if r_try is not None:
                phi = 0.5 * float(r_try @ r_try)
                if phi <= (1.0 - 2.0 * config.armijo * alpha) * phi0:
                    accepted = r_try
                    break
            alpha *= 0.5
        if accepted is None:
            records.append(_rec(it, rnorm, alpha))
            return x, SolveStatus.NUMERICAL_FAILURE, records
        x = x + alpha * dx
        r = accepted
        records.append(_rec(it, float(np.max(np.abs(r))), alpha))
    return x, SolveStatus.MAX_ITER, records


def _rec(it, rnorm, alpha) -> IterationRecord:
    # Newton reports its residual norm as cont_norm (already scaled by mdot);
    # pos_norm is folded into the same scaled infinity norm, and the accepted
    # step length rides in the omega_sl slot ("the relaxation actually used").
    return IterationRecord(iteration=it, cont_norm=rnorm, pos_norm=rnorm,
                           closure_norm=0.0, omega_sl=alpha)


# ---------------------------------------------------------------------------
# Outer driver (mandatory warm start + quasi-Newton closure lagging)
# ---------------------------------------------------------------------------
def solve_newton(topology, fluid, fidelity: FidelityConfig,
                 spec: MassFlowSpec, inlet: TransportFields, *,
                 warm_start: ClassicalResult,
                 rows=(), steps=None, blockage=None,
                 metrics_config: MetricsConfig = None,
                 config: NewtonConfig = NewtonConfig()) -> ClassicalResult:
    """Solve one operating point by global Newton (section 6.3, ARCH-5.3).

    ``warm_start`` is a converged-or-partial :class:`ClassicalResult` on the
    SAME topology (mandatory: Newton is local). Its packed state seeds the
    inner solve and its lagged transport/closures seed the outer loop. Returns
    a :class:`ClassicalResult` (same shape as the classical driver, so the
    facade/continuation layers consume both identically).
    """
    if warm_start is None or warm_start.frozen is None:
        raise ConfigError(
            "solve_newton requires a warm_start ClassicalResult with a frozen "
            "iterate (ARCH-5.3: Newton is local, the seed is mandatory)")
    n_sl, n_qo = topology.n_sl, topology.n_qo
    if rows and steps is not None:
        raise ConfigError("steps and rows are mutually exclusive")
    resolved_rows = _resolve_rows(topology, rows) if rows else []
    if steps is None:
        steps = [TransportStep()] * (n_qo - 1)
    if blockage is None:
        blockage = np.zeros((n_sl, n_qo))
    if metrics_config is None:
        metrics_config = MetricsConfig()

    def col0(a):
        a = np.asarray(a, dtype=float)
        return a[:, 0] if a.ndim == 2 else np.broadcast_to(a, (n_sl,))

    inlet_h0, inlet_s, inlet_rvt = (col0(inlet.h0), col0(inlet.s),
                                    col0(inlet.rvt))

    # Seed from the warm start (ARCH-5.3): its state and lagged fields.
    x = np.array(warm_start.x, dtype=float)
    transported = warm_start.frozen.transported
    closures = warm_start.frozen.closures
    vm_lagged = warm_start.frozen.vm_lagged
    q_fixed = warm_start.frozen.q_fixed
    kappa_relax = 0.3 if fidelity.curvature_term > 0.0 else 1.0
    kappa_prev = warm_start.frozen.kappa_lagged
    history = []
    status, reason = SolveStatus.MAX_ITER, ""
    frozen = asm = None

    for outer in range(1, config.max_outer + 1):
        if not all(np.all(np.isfinite(a)) for a in
                   (transported.h0, transported.s, transported.rvt,
                    vm_lagged, x)):
            status, reason = SolveStatus.NUMERICAL_FAILURE, (
                f"non-finite lagged inputs at outer {outer} (AD-10)")
            break
        frozen = FrozenInputs(topology=topology, fluid=fluid,
                              fidelity=fidelity, spec=spec,
                              transported=transported, closures=closures,
                              vm_lagged=vm_lagged, kappa_lagged=kappa_prev,
                              kappa_relax=kappa_relax, q_fixed=q_fixed,
                              metrics_config=metrics_config)
        asm = ResidualAssembler(frozen)
        x, in_status, in_recs = newton_solve(asm, x, config)
        history.extend(in_recs)
        if in_status is not SolveStatus.CONVERGED:
            status, reason = in_status, (
                f"inner Newton {in_status.value} at outer {outer}")
            break

        fields = asm.split(x)
        # Lag refresh (quasi-Newton outer, section 6.3): re-evaluate closures
        # from the converged inner field, under-relax, re-sweep transport.
        if resolved_rows:
            new_steps, exit_rvt, delta_s_row, validity = _evaluate_rows(
                resolved_rows, topology, fluid, transported, fields)
            if closures.row_exit_rvt:
                w = config.closure_relax
                exit_rvt = {k: closures.row_exit_rvt[k]
                            + w * (v - closures.row_exit_rvt[k])
                            for k, v in exit_rvt.items()}
                delta_s_row = {k: closures.row_delta_s[k]
                               + w * (v - closures.row_delta_s[k])
                               for k, v in delta_s_row.items()}
                for rspec, j_le, _j_te in resolved_rows:
                    new_steps[j_le] = row_steps(
                        omega=rspec.omega,
                        rvt_le=transported.rvt[:, j_le],
                        rvt_te=exit_rvt[rspec.row_id],
                        delta_s_row=delta_s_row[rspec.row_id])[0]
            cnorm = _closure_norm(exit_rvt, delta_s_row,
                                  closures.row_exit_rvt, closures.row_delta_s)
            closures = ClosureFields(blockage, row_exit_rvt=exit_rvt,
                                     row_delta_s=delta_s_row,
                                     validity=validity, iteration_tag=outer)
            transported = sweep(inlet_h0, inlet_s, inlet_rvt, new_steps)
        else:
            cnorm = 0.0
        # Recursive curvature lag (section 5.5), mirroring the classical EMA.
        if kappa_prev is None:
            kappa_prev = fields.metrics.kappa_m
        else:
            kappa_prev = (kappa_relax * fields.metrics.kappa_m
                          + (1.0 - kappa_relax) * kappa_prev)
        vm_lagged = fields.vm
        history.append(IterationRecord(iteration=len(history) + 1,
                                       cont_norm=0.0, pos_norm=0.0,
                                       closure_norm=cnorm, omega_sl=1.0))
        if cnorm < config.tol_closure:
            status = SolveStatus.CONVERGED
            break

    record = ConvergenceRecord(status=status, iterations=tuple(history),
                               reason=reason)
    if asm is None:
        return ClassicalResult(status=status, x=x, fields=None,
                               frozen=None, record=record)
    return ClassicalResult(status=status, x=x, fields=asm.split(x),
                           frozen=frozen, record=record)
