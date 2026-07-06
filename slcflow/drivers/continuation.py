"""Off-design map / continuation driver (Theory Manual sections 6.6, 6.7;
ARCH-5.4).

Traverses a speedline by natural-parameter continuation in mass flow: order
points from choke toward stall, initialise each from the previous converged
solution (the warm start the classical/Newton drivers now accept), adapt the
step with cut-back on failure, and escalate classical -> Newton before
rejecting a step. Numerical stall/surge is **reported, not solved through**
(section 6.7): the driver flags the operating point where a stall criterion
fires and records *which* criterion — surge-margin definitions must be
traceable.

Layering (ARCH-2): this sits in ``drivers`` with the two point-solvers it
orchestrates, so it cannot reach up to the ``machine`` facade's performance
reduction. It computes only what operability needs — the annulus choke margin
(section 6.6) and a mass-averaged total-to-total pressure ratio (the map's
ordinate and the turnover criterion's signal). Full performance reduction
(efficiency, spanwise profiles) stays the facade's job; ``MapPoint`` carries
the raw :class:`ClassicalResult` so the facade can reduce each point.

Scope (M5-2): the traversal runs in **mass-flow-specified (normal) mode**
throughout. The section 6.6 back-pressure BC-switch for the choke-proximal
branch is M5-3; here the choke margin is reported and choke-limiting ends the
traversal with a recorded flag rather than switching boundary conditions.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # driver layer: orchestration, not residual path  # ad6: allow

from ..assembly.assembler import ResidualAssembler
from ..diagnostics.record import SolveStatus
from ..errors import ConfigError
from ..grid.core import MetricsConfig
from ..transport.streamwise import TransportFields
from ..types import FidelityConfig, MassFlowSpec
from .classical import ClassicalConfig, ClassicalResult, solve_classical
from .newton import NewtonConfig, solve_newton

__all__ = ["SpeedlineConfig", "MapPoint", "StallFlag", "MapResult",
           "solve_speedline"]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class SpeedlineConfig:
    """Continuation settings (sections 6.6, 6.7).

    Parameters
    ----------
    step_min_frac : smallest step, as a fraction of ``mdot_start``, the
        cut-back will shrink to before declaring a solver-failure stall.
    cutback : step-shrink factor on a failed point (section 6.7).
    step_grow : step-regrow factor after a clean success (bounded by the
        caller's initial step), so the traversal re-accelerates on easy runs.
    validity_min : converged-point closure validity below which the point is
        flagged as a saturation stall (section 6.7 criterion (c)).
    pr_rise_min : the pressure ratio must first rise this much above the
        starting point to ARM the turnover check — so a flat (duct) or
        still-rising characteristic never false-flags (criterion (b)).
    escalate : try the Newton driver (warm-started) on a point the classical
        driver fails, before cutting the step (section 6.7 escalation).
    max_points : hard cap on traversed points (runaway guard).
    """

    step_min_frac: float = 0.02
    cutback: float = 0.5
    step_grow: float = 1.5
    validity_min: float = 0.1
    pr_rise_min: float = 1e-3
    escalate: bool = True
    max_points: int = 200
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    newton: NewtonConfig = field(default_factory=NewtonConfig)

    def __post_init__(self):
        if not (0.0 < self.cutback < 1.0):
            raise ConfigError(f"cutback must be in (0, 1), got {self.cutback}")
        if self.step_grow < 1.0:
            raise ConfigError("step_grow must be >= 1")


@dataclass(frozen=True)
class MapPoint:
    """One converged operating point on the speedline (ARCH-5.4)."""

    mdot: float
    pressure_ratio: float
    choke_margin: float          # c = min_j (1 - mdot/mdot_max_j), section 6.6
    validity: float
    driver: str                  # "classical" | "newton" (which converged it)
    result: ClassicalResult = field(repr=False)


@dataclass(frozen=True)
class StallFlag:
    """Recorded stall/surge onset (section 6.7: report, don't solve through)."""

    mdot: float
    criterion: str               # "solver_failure"|"pr_turnover"|"validity_saturated"
    detail: str = ""


@dataclass(frozen=True)
class MapResult:
    """A traversed speedline: converged points choke->stall, plus the stall
    flag that ended it (``None`` if the traversal reached ``mdot_min``)."""

    points: tuple                # tuple[MapPoint], choke -> stall order
    stall: StallFlag = None

    @property
    def converged_points(self) -> int:
        return len(self.points)

    @property
    def peak_pressure_ratio(self) -> float:
        return max((p.pressure_ratio for p in self.points), default=float("nan"))


# ---------------------------------------------------------------------------
# Performance/operability reductions the driver layer is allowed to do
# ---------------------------------------------------------------------------
def _mass_avg(fluid, result: ClassicalResult, phi, j) -> float:
    """Mass-flux-weighted (``rho Vm cos(eps) r``) span average at station j;
    a single meanline node returns itself (section 3.2)."""
    phi = np.asarray(phi, dtype=float)
    if phi.size == 1:
        return float(phi[0])
    f = result.fields
    w = (f.rho[:, j] * f.vm[:, j] * np.cos(f.metrics.eps[:, j])
         * f.metrics.r[:, j])
    q = f.q[:, j]
    return float(np.trapezoid(w * phi, q) / np.trapezoid(w, q))


def _pressure_ratio(fluid, result: ClassicalResult) -> float:
    """Mass-averaged total-to-total stagnation pressure ratio (exit/inlet).
    Stagnation pressure is ``p(h0, s)`` (isentropic-to-rest, section 3.7)."""
    tr = result.frozen.transported
    j_in, j_ex = 0, result.frozen.n_qo - 1
    p0_in = _mass_avg(fluid, result, fluid.p(tr.h0[:, j_in], tr.s[:, j_in]),
                      j_in)
    p0_ex = _mass_avg(fluid, result, fluid.p(tr.h0[:, j_ex], tr.s[:, j_ex]),
                      j_ex)
    return p0_ex / p0_in


def _choke_margin(result: ClassicalResult) -> float:
    """Annulus choke margin ``c = min_j (1 - mdot / mdot_max,j)`` (section
    6.6). Row-throat margins (section 4.5) need the INBLADE throat model
    (M7); until then this is annulus-only, and that is what it reports."""
    asm = ResidualAssembler(result.frozen)
    mdot = result.frozen.spec.mdot
    caps = [asm.qo_capacity(j, result.fields)
            for j in range(result.frozen.n_qo)]
    return float(min(1.0 - mdot / c for c in caps))


# ---------------------------------------------------------------------------
# Point solve with classical -> Newton escalation
# ---------------------------------------------------------------------------
def _solve_point(mdot, warm, *, topology, fluid, fidelity, inlet, rows, steps,
                 blockage, metrics_config, config: SpeedlineConfig):
    """Solve at ``mdot`` from warm start ``warm`` (a previous MapPoint's
    result, or ``None`` for the cold first point). Classical first; on failure
    escalate to Newton warm-started from the same seed (section 6.7)."""
    common = dict(rows=rows, steps=steps, blockage=blockage,
                  metrics_config=metrics_config)
    res = solve_classical(topology, fluid, fidelity, MassFlowSpec(mdot), inlet,
                          warm_start=warm, config=config.classical, **common)
    if res.converged:
        return res, "classical"
    if config.escalate and warm is not None:
        res_n = solve_newton(topology, fluid, fidelity, MassFlowSpec(mdot),
                             inlet, warm_start=warm, config=config.newton,
                             **common)
        if res_n.converged:
            return res_n, "newton"
        return res_n, "newton"
    return res, "classical"


# ---------------------------------------------------------------------------
# Speedline traversal
# ---------------------------------------------------------------------------
def solve_speedline(topology, fluid, fidelity: FidelityConfig,
                    inlet: TransportFields, *, mdot_start: float,
                    mdot_min: float, mdot_step: float, rows=(), steps=None,
                    blockage=None, metrics_config: MetricsConfig = None,
                    config: SpeedlineConfig = SpeedlineConfig()) -> MapResult:
    """Traverse a speedline choke -> stall (section 6.7).

    Points run from ``mdot_start`` (choke side, higher flow) down toward
    ``mdot_min`` (stall side) in steps of ``mdot_step``, each warm-started
    from its predecessor. A failed point cuts the step (``config.cutback``)
    and retries closer to the last success; when the step falls below
    ``step_min_frac * mdot_start`` the traversal ends with a ``solver_failure``
    stall flag. Converged points are also checked for the section 6.7 turnover
    and validity-saturation criteria.
    """
    if not (mdot_start > mdot_min > 0.0):
        raise ConfigError("need mdot_start > mdot_min > 0")
    if not (0.0 < mdot_step <= mdot_start - mdot_min):
        raise ConfigError("need 0 < mdot_step <= mdot_start - mdot_min")
    if metrics_config is None:
        metrics_config = MetricsConfig()

    step_min = config.step_min_frac * mdot_start
    solve_kw = dict(topology=topology, fluid=fluid, fidelity=fidelity,
                    inlet=inlet, rows=rows, steps=steps, blockage=blockage,
                    metrics_config=metrics_config, config=config)

    points = []
    stall = None
    mdot = mdot_start
    step = mdot_step
    last_ok = None                 # last converged result (warm-start seed)
    pr_start = None
    armed = False
    prev_pr = None

    while mdot >= mdot_min - 1e-12 and len(points) < config.max_points:
        result, driver = _solve_point(mdot, last_ok, **solve_kw)
        if not result.converged:
            step *= config.cutback
            if step < step_min or last_ok is None:
                stall = StallFlag(
                    mdot=mdot, criterion="solver_failure",
                    detail=f"{driver} {result.status.value}; step {step:.4g} "
                           f"< min {step_min:.4g}"
                    if last_ok is not None else
                    f"cold first point failed: {driver} "
                    f"{result.status.value}")
                break
            mdot = points[-1].mdot - step      # retry nearer the last success
            continue

        pr = _pressure_ratio(fluid, result)
        validity = float(result.frozen.closures.validity)
        if validity < config.validity_min:
            stall = StallFlag(mdot=mdot, criterion="validity_saturated",
                              detail=f"closure validity {validity:.3g} "
                                     f"< {config.validity_min:.3g}")
            break
        # Turnover: armed once PR has risen above the start; fires when PR
        # stops rising as mdot falls (the peak was the previous point).
        if armed and prev_pr is not None and pr < prev_pr:
            stall = StallFlag(
                mdot=mdot, criterion="pr_turnover",
                detail=f"PR {pr:.5g} < previous {prev_pr:.5g} at falling mdot "
                       f"(peak ~{points[-1].mdot:.4g})")
            break

        points.append(MapPoint(
            mdot=mdot, pressure_ratio=pr, choke_margin=_choke_margin(result),
            validity=validity, driver=driver, result=result))
        last_ok = result
        if pr_start is None:
            pr_start = pr
        if pr > pr_start + config.pr_rise_min:
            armed = True
        prev_pr = pr
        step = min(step * config.step_grow, mdot_step)
        mdot -= step

    return MapResult(points=tuple(points), stall=stall)
