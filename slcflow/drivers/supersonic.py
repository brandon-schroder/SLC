"""Meridional-supersonic-branch continuation driver (Theory Manual sections
6.6, C.9; ARCH-5.4).

The mass-flow-specified continuity relation at a q-o station is FOLDED at the
sonic meridional condition (``M_m = 1``): below the station capacity there are
two ``Vm`` roots — a subsonic-meridional root and a supersonic-meridional one —
that coalesce at the capacity peak, where the continuity Jacobian is singular.
The classical driver takes the subsonic root by construction (``_solve_qo``),
and natural-parameter continuation in ``mdot`` (or exit pressure) cannot cross
the peak: it either chokes (``CHOKE_LIMITED``) or pins at ``M_m = 1`` (measured
2026-07, C.9). Reaching the supersonic-meridional branch therefore needs
**pseudo-arclength continuation** (Keller): parametrise the solution curve in
``(state, mdot)`` by arclength, so the augmented Jacobian stays non-singular AT
the fold and the traversal walks smoothly from the subsonic branch, through the
choke turning point, onto the supersonic branch.

This is the "V5 supersonic-branch traversal" milestone, built as a general,
reusable capability against a purpose-designed nozzle test (a meanline
converging–diverging duct, whose supersonic root the isentropic area–Mach
relation gives in closed form) — **not** a V5 blocker (the transonic V5 gate is
met on the ordinary branch; the in-window condition is loading, not the
meridional branch — see C.9 and ``V5TransonicRotor``).

Two paths, selected by whether ``rows`` (closure-fed blade rows) are supplied:

* **Prescribed transport** (duct / explicit :class:`TransportStep` steps, no
  rows). The frozen inputs — transported fields, closures, lag — are constant
  along the branch (a duct has none to re-lag), so the continuation is purely
  over the continuity/position state and ``mdot``: one arclength crossing plus
  the landing Newton.
* **Closure-lagged blade rows.** The row swirl/loss closures depend on the flow
  field and are lagged (AD-4); the supersonic-branch field differs from the
  subsonic seed's (a different ``mdot``, and — where the row's own flow goes
  supersonic — a different inflow), so the closures must be re-evaluated at the
  landed supersonic state. This path wraps the arclength inner in the SAME
  outer quasi-Newton closure-lag loop as :func:`solve_newton` (section 6.3):
  bootstrap onto the supersonic branch ONCE by arclength, then on each outer
  pass re-lag the closures from the supersonic field, re-sweep transport, and
  re-land at ``target_mdot`` by fixed-``mdot`` Newton (the branch is already
  selected, so the fold need not be re-crossed) until the closure-update norm
  converges (section 6.2.5). Robust for a single dominant fold (the binding
  station's ``M_m=1``); a fully supersonic ROW inflow that folds several
  stations at once (measured on a transonic rotor, C.9) is the harder
  multi-fold regime this simple-fold arclength does not claim.

Layering (ARCH-2): a ``drivers`` orchestration over the pure
:class:`ResidualAssembler` pieces (``split``, ``mass_cumulative``,
``continuity_position_rows``) and the shared classical row-evaluation helpers.
Data-dependent branching and iteration are the driver's job (AD-6 binds
assembly, not orchestration); failures return typed statuses (ARCH-6), never
exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np  # driver layer: orchestration, not residual path  # ad6: allow

from ..assembly.assembler import ResidualAssembler
from ..assembly.inputs import FrozenInputs
from ..diagnostics.record import ConvergenceRecord, SolveStatus
from ..errors import ConfigError
from ..grid.core import MetricsConfig
from ..transport.streamwise import TransportFields
from ..types import FidelityConfig, MassFlowSpec
from .classical import ClassicalResult
from .newton import NewtonConfig, newton_solve, solve_newton

__all__ = ["ArclengthConfig", "BranchPoint", "MeridionalBranchResult",
           "solve_supersonic_branch"]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ArclengthConfig:
    """Pseudo-arclength continuation settings (section 6.6 / C.9).

    Parameters
    ----------
    ds0 : initial arclength step (in the SCALED state, so O(0.1) is a fraction
        of the characteristic velocity/mass-flow, not an SI length).
    ds_max, ds_min : step bounds; the corrector shrinks ``ds`` on failure down
        to ``ds_min`` before declaring a numerical failure.
    ds_grow, ds_shrink : step adaptation factors on corrector success/failure.
    max_steps : hard cap on predictor steps (runaway guard).
    tol : corrector convergence tolerance on the scaled augmented residual.
    max_corrector : Newton iterations per corrector.
    fd_rel, fd_abs : forward-difference step for the state Jacobian columns
        (the ``mdot`` column is analytic — ``mdot`` enters continuity linearly).
    vm_scale, mdot_scale, q_scale : arclength variable scales; ``None`` picks
        physical defaults from the seed (sonic speed, seed/target mass flow,
        mean q-o length) so the arclength balances velocity and mass-flow
        components (measured necessary — an unscaled arclength creeps in
        ``mdot`` because ``Vm`` dominates the norm near the fold).
    newton : settings for BOTH the fixed-``mdot`` landing Newton (prescribed-
        transport path) and the outer closure-lag loop (blade-row path, which
        hands off to :func:`solve_newton`). The default raises ``max_outer`` to
        120: the closure lag is a Picard iteration converging at rate
        ``(1 - closure_relax)``, so the conservative ``closure_relax = 0.25``
        (kept — it must stay safe for a stiff swirl-continuity loop, M4-4) needs
        ~60 outer passes to reach ``tol_closure = 1e-9`` from a typical few-%
        seed mismatch. Weakly-coupled cases (e.g. a row upstream of the folding
        throat) tolerate a larger ``closure_relax`` and converge in far fewer.
    """

    ds0: float = 0.1
    ds_max: float = 0.3
    ds_min: float = 1e-4
    ds_grow: float = 1.15
    ds_shrink: float = 0.5
    max_steps: int = 400
    tol: float = 1e-8
    max_corrector: int = 30
    fd_rel: float = 1e-7
    fd_abs: float = 1e-9
    vm_scale: float = None
    mdot_scale: float = None
    q_scale: float = None
    newton: NewtonConfig = field(
        default_factory=lambda: NewtonConfig(max_outer=120))

    def __post_init__(self):
        if self.max_steps < 1:
            raise ConfigError("max_steps must be >= 1")
        if not (0.0 < self.ds_min <= self.ds0 <= self.ds_max):
            raise ConfigError("need 0 < ds_min <= ds0 <= ds_max")
        if self.ds_grow < 1.0 or not (0.0 < self.ds_shrink < 1.0):
            raise ConfigError("need ds_grow >= 1 and 0 < ds_shrink < 1")


@dataclass(frozen=True)
class BranchPoint:
    """One converged point on the traversed solution branch."""

    mdot: float                  # mass flow at this point (an OUTPUT here)
    mach_m_max: float            # max meridional Mach across stations
    dmdot_ds: float              # tangent's mdot component (sign flips at fold)
    ds: float                    # arclength step that reached it


@dataclass(frozen=True)
class MeridionalBranchResult:
    """A meridional-branch traversal (section 6.6 / C.9).

    ``result`` is the final on-``target_mdot`` supersonic solution as an
    ordinary :class:`ClassicalResult` (so the facade/tests consume it exactly
    like a mass-flow solve). ``fold_crossed`` records whether the traversal
    passed the sonic turning point; ``fold_mdot`` is the peak mass flow there
    (the station capacity, A.7). ``path`` is the branch for diagnostics."""

    status: SolveStatus
    result: ClassicalResult
    fold_crossed: bool
    fold_mdot: float
    path: tuple = ()
    reason: str = ""

    @property
    def converged(self) -> bool:
        return self.status is SolveStatus.CONVERGED


# ---------------------------------------------------------------------------
# Augmented residual G(state, mdot) and its Jacobian
# ---------------------------------------------------------------------------
def _safe_G(asm: ResidualAssembler, x_state, mdot):
    """Continuity + position rows at ``(x_state, mdot)``, or ``None`` if the
    point is infeasible — a crossing streamline, or a non-finite / non-positive
    ``Vm`` field (the master ODE's spurious sub-zero branch). Mirrors the
    Newton driver's positive-branch guard; the supersonic root has large but
    strictly positive ``Vm``, so it passes. No exception crosses (AD-10)."""
    fz = asm.frozen
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        fields = asm.split(x_state)
        if not bool(np.all(np.isfinite(fields.vm))
                    and np.all(fields.vm > 0.0)
                    and np.all(np.isfinite(fields.rho))
                    and np.all(np.isfinite(fields.mach_m))):
            return None, None
        if fz.n_sl > 1 and not bool(np.all(np.diff(fields.q, axis=0) > 0.0)):
            return None, None
        rows = asm.continuity_position_rows(fields, mdot)
    if not np.all(np.isfinite(rows)):
        return None, None
    return rows, fields


def _mdot_column(asm: ResidualAssembler) -> np.ndarray:
    """Analytic ``d(rows)/d(mdot)``: ``mdot`` enters continuity linearly, so
    the continuity rows contribute ``-1`` and the position rows ``-psi/2pi``
    (from ``c[1:-1] - psi_int * mdot/2pi``)."""
    fz = asm.frozen
    psi_int = fz.topology.psi[1:-1]
    cont = np.full(fz.n_qo, -1.0)
    pos = -np.repeat(psi_int, fz.n_qo) / _TWO_PI      # empty for n_sl <= 2
    return np.concatenate([cont, pos])


def _jacobian(asm, x_state, mdot, r0, config, mdot_col):
    """``d(rows)/d(state, mdot)`` (N x N_state+1): forward-difference state
    columns (backward fallback), analytic ``mdot`` column. ``None`` if a
    column cannot be formed."""
    n = x_state.size
    jac = np.empty((r0.size, n + 1))
    for k in range(n):
        h = config.fd_rel * abs(x_state[k]) + config.fd_abs
        xf = x_state.copy()
        xf[k] += h
        rf, _ = _safe_G(asm, xf, mdot)
        if rf is not None:
            jac[:, k] = (rf - r0) / h
            continue
        xb = x_state.copy()
        xb[k] -= h
        rb, _ = _safe_G(asm, xb, mdot)
        if rb is None:
            return None
        jac[:, k] = (r0 - rb) / h
    jac[:, n] = mdot_col
    return jac


# ---------------------------------------------------------------------------
# Arclength core (fixed frozen inputs) + fixed-mdot landing
# ---------------------------------------------------------------------------
def _arclength_scales(frozen: FrozenInputs, seed_mdot, target_mdot,
                      config: ArclengthConfig):
    """Per-variable arclength scales ``D`` (balance Vm and mdot; measured
    necessary) and the ``mdot`` normaliser conditioning the augmented solve."""
    g = frozen.fluid.gamma
    vm_scale = config.vm_scale or float(np.sqrt(
        2.0 * np.mean(frozen.transported.h0) * (g - 1.0) / (g + 1.0)))
    mdot_scale = config.mdot_scale or max(float(seed_mdot), float(target_mdot))
    q_scale = config.q_scale or float(np.max(
        [qo.length for qo in frozen.topology.flowpath.qo_curves]))
    n_int = max(frozen.n_sl - 2, 0)
    D = np.concatenate([np.full(frozen.n_qo, vm_scale),
                        np.full(n_int * frozen.n_qo, q_scale),
                        np.array([mdot_scale])])
    return D, mdot_scale


def _arclength_to_supersonic(asm: ResidualAssembler, x_subsonic, seed_mdot,
                             target_mdot, config: ArclengthConfig):
    """Pseudo-arclength from the subsonic seed across the ``M_m=1`` fold to a
    state on the SUPERSONIC branch at ``mdot <= target_mdot`` (fixed closures /
    frozen). Returns ``(x_arc, status, fold_crossed, fold_mdot, path)`` — the
    pre-landing arclength endpoint (``x_arc`` is ``None`` on failure)."""
    D, row_scale = _arclength_scales(asm.frozen, seed_mdot, target_mdot, config)
    mdot_col = _mdot_column(asm)

    def augmented(y, yk, t, ds):
        rows, fields = _safe_G(asm, y[:-1], y[-1])
        if rows is None:
            return None, None
        arc = float(t @ ((y - yk) / D)) - ds
        return np.append(rows / row_scale, arc), fields

    y = np.append(np.array(x_subsonic, dtype=float), float(seed_mdot))
    r0, _ = _safe_G(asm, y[:-1], y[-1])
    if r0 is None:
        return None, SolveStatus.NUMERICAL_FAILURE, False, float("nan"), ()
    jac0 = _jacobian(asm, y[:-1], y[-1], r0, config, mdot_col)
    if jac0 is None:
        return None, SolveStatus.NUMERICAL_FAILURE, False, float("nan"), ()
    _, _, vt = np.linalg.svd(jac0 * D[None, :] / row_scale)
    t = vt[-1]
    if t[-1] < 0.0:                                  # orient toward rising mdot
        t = -t
    t /= np.linalg.norm(t)

    ds = config.ds0
    path = []
    fold_crossed = False
    fold_mdot = float("nan")
    prev_mdot = y[-1]

    for _step in range(config.max_steps):
        yk = y.copy()
        y_try = y + ds * (D * t)
        accepted = None
        for _ in range(config.max_corrector):
            F, fields = augmented(y_try, yk, t, ds)
            if F is None:
                break
            if float(np.max(np.abs(F))) < config.tol:
                accepted = (y_try.copy(), fields)
                break
            rows, _ = _safe_G(asm, y_try[:-1], y_try[-1])
            jac = _jacobian(asm, y_try[:-1], y_try[-1], rows, config, mdot_col)
            if jac is None:
                break
            jf = np.vstack([jac * D[None, :] / row_scale, t[None, :]])
            try:
                dy = np.linalg.solve(jf, F)
            except np.linalg.LinAlgError:
                dy, *_ = np.linalg.lstsq(jf, F, rcond=None)
            y_try = y_try - D * dy
        if accepted is None:
            ds *= config.ds_shrink
            if ds < config.ds_min:
                return (None, SolveStatus.NUMERICAL_FAILURE, fold_crossed,
                        fold_mdot, tuple(path))
            continue

        y, fields = accepted
        rows, _ = _safe_G(asm, y[:-1], y[-1])
        jac = _jacobian(asm, y[:-1], y[-1], rows, config, mdot_col)
        jf = np.vstack([jac * D[None, :] / row_scale, t[None, :]])
        tn = np.linalg.solve(jf, np.append(np.zeros(rows.size), 1.0))
        tn /= np.linalg.norm(tn)
        if float(tn @ t) < 0.0:
            tn = -tn
        t = tn

        path.append(BranchPoint(mdot=float(y[-1]),
                                mach_m_max=float(np.max(fields.mach_m)),
                                dmdot_ds=float(t[-1]), ds=ds))
        if not fold_crossed and y[-1] < prev_mdot:
            fold_crossed = True
            fold_mdot = float(prev_mdot)
        prev_mdot = y[-1]
        ds = min(ds * config.ds_grow, config.ds_max)

        if fold_crossed and y[-1] <= target_mdot:
            return (y[:-1].copy(), SolveStatus.CONVERGED, fold_crossed,
                    fold_mdot, tuple(path))

    return (None, SolveStatus.MAX_ITER, fold_crossed, fold_mdot, tuple(path))


def _land(frozen: FrozenInputs, x_seed, target_mdot, newton_config):
    """Fixed-``mdot`` Newton at ``target_mdot`` from a supersonic seed: the
    branch is selected (the fold is behind), so this is a regular mass-flow
    root. Reuses the tested inner Newton core. Returns
    ``(asm_t, frozen_t, x, status, records)``."""
    frozen_t = replace(frozen, spec=MassFlowSpec(float(target_mdot)))
    asm_t = ResidualAssembler(frozen_t)
    # Branch-preserving trials from the LANDED seed (section 6.3): the
    # arclength already selected the supersonic branch; the guard keeps the
    # landing Newton from falling back across the fold (and, per the
    # 2026-07-16 root cause, from jumping any other station's root).
    from .newton import _branch_masks
    x, status, recs = newton_solve(asm_t, np.array(x_seed, dtype=float),
                                   newton_config,
                                   branch=_branch_masks(asm_t, x_seed))
    return asm_t, frozen_t, x, status, recs


def _result(status, x, asm_t, frozen_t, fold_crossed, fold_mdot, path, reason):
    record = ConvergenceRecord(status=status, iterations=(), reason=reason)
    fields = (asm_t.split(x) if status in
              (SolveStatus.CONVERGED, SolveStatus.MAX_ITER) else None)
    result = ClassicalResult(status=status, x=x, fields=fields,
                             frozen=frozen_t, record=record)
    return MeridionalBranchResult(status, result, fold_crossed, fold_mdot,
                                  tuple(path), reason=reason)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def solve_supersonic_branch(topology, fluid, fidelity: FidelityConfig,
                            inlet: TransportFields, *,
                            subsonic_seed: ClassicalResult,
                            target_mdot: float,
                            steps=None, rows=(), blockage=None,
                            metrics_config: MetricsConfig = None,
                            config: ArclengthConfig = ArclengthConfig()
                            ) -> MeridionalBranchResult:
    """Traverse from a subsonic-branch seed onto the meridional-supersonic
    branch and land at ``target_mdot`` (section 6.6 / C.9).

    ``subsonic_seed`` is a converged mass-flow :class:`ClassicalResult` on the
    SAME topology, at ``mdot`` below the binding station capacity. ``rows``
    (closure-fed :class:`RowSpec`) selects the **closure-lagged** path (the
    outer quasi-Newton loop re-lagging closures at the supersonic field); an
    empty ``rows`` (with prescribed ``steps`` / a duct) is the fixed-frozen
    path. ``target_mdot`` is the mass flow to land on, on the far (supersonic)
    side of the fold — below the capacity peak the traversal discovers.

    Pseudo-arclength climbs the subsonic branch (rising ``mdot``) to the sonic
    turning point, crosses it, then descends the supersonic branch; a
    fixed-``mdot`` Newton lands the exact on-target supersonic solution (the
    branch is selected, so that root is regular). Returns typed statuses
    (ARCH-6)."""
    if subsonic_seed is None or subsonic_seed.frozen is None:
        raise ConfigError(
            "solve_supersonic_branch requires a converged subsonic_seed "
            "ClassicalResult with frozen inputs")
    if not target_mdot > 0.0:
        raise ConfigError(f"target_mdot must be > 0, got {target_mdot}")
    frozen0 = subsonic_seed.frozen
    if frozen0.topology.n_sl != topology.n_sl \
            or frozen0.topology.n_qo != topology.n_qo:
        raise ConfigError("subsonic_seed must be on the same topology")
    seed_mdot = (frozen0.spec.mdot if isinstance(frozen0.spec, MassFlowSpec)
                 else float(target_mdot))

    if rows:
        return _solve_with_closure_lag(topology, fluid, fidelity, inlet,
                                       subsonic_seed, target_mdot, seed_mdot,
                                       rows, metrics_config, config)

    # Prescribed-transport (fixed-frozen) path: one crossing + landing.
    asm = ResidualAssembler(frozen0)
    x_arc, status, fold_crossed, fold_mdot, path = _arclength_to_supersonic(
        asm, subsonic_seed.x, seed_mdot, target_mdot, config)
    if x_arc is None:
        reason = ("corrector failed" if status is SolveStatus.NUMERICAL_FAILURE
                  else f"reached max_steps ({config.max_steps}) without "
                       f"landing on target_mdot={target_mdot}")
        return MeridionalBranchResult(status, subsonic_seed, fold_crossed,
                                      fold_mdot, path, reason=reason)
    asm_t, frozen_t, x, lstatus, _recs = _land(frozen0, x_arc, target_mdot,
                                               config.newton)
    reason = ("" if lstatus is SolveStatus.CONVERGED else
              "fixed-mdot landing Newton did not converge on the supersonic root")
    return _result(lstatus, x, asm_t, frozen_t, fold_crossed, fold_mdot, path,
                   reason)


def _solve_with_closure_lag(topology, fluid, fidelity, inlet, subsonic_seed,
                            target_mdot, seed_mdot, rows, metrics_config,
                            config: ArclengthConfig):
    """Closure-lagged blade-row path: bootstrap onto the supersonic branch by
    arclength ONCE (with the seed's frozen closures), then hand the supersonic
    seed to :func:`solve_newton` at ``target_mdot``, which runs the SAME outer
    quasi-Newton closure-lag loop it uses everywhere (section 6.3) — re-lagging
    the flow-dependent closures at the supersonic field and re-solving until the
    closure-update norm converges (section 6.2.5). The fold is behind the
    bootstrap seed, so the Newton inner stays on the supersonic branch (its
    positive-``Vm`` feasibility guard admits the supersonic root); the lag
    cadence is ``config.newton``. Reusing the tested/tuned Newton machinery is
    why this path adds almost no numerical surface of its own."""
    frozen0 = subsonic_seed.frozen
    asm = ResidualAssembler(frozen0)
    x_arc, status, fold_crossed, fold_mdot, path = _arclength_to_supersonic(
        asm, subsonic_seed.x, seed_mdot, target_mdot, config)
    if x_arc is None:
        return MeridionalBranchResult(
            status, subsonic_seed, fold_crossed, fold_mdot, path,
            reason=f"arclength bootstrap {status.value} before any landing")

    # Warm start for solve_newton: the supersonic STATE (x_arc) with the seed's
    # lagged fields/closures (subsonic-consistent) — the first outer pass
    # re-lags them at the supersonic field.
    warm = ClassicalResult(status=SolveStatus.CONVERGED, x=x_arc,
                           fields=asm.split(x_arc), frozen=frozen0,
                           record=ConvergenceRecord(SolveStatus.CONVERGED, ()))
    kw = {} if metrics_config is None else {"metrics_config": metrics_config}
    res = solve_newton(topology, fluid, fidelity, MassFlowSpec(target_mdot),
                       inlet, warm_start=warm, rows=rows, config=config.newton,
                       **kw)
    return MeridionalBranchResult(res.status, res, fold_crossed, fold_mdot,
                                  path, reason=res.record.reason)
