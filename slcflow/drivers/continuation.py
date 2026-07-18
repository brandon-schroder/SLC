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
from ..assembly.pack import unpack
from ..diagnostics.record import SolveStatus
from ..errors import ConfigError
from ..grid.core import MetricsConfig
from ..transport.streamwise import TransportFields
from ..types import BackPressureSpec, FidelityConfig, MassFlowSpec
from .classical import (ClassicalConfig, ClassicalResult, _resolve_rows,
                        solve_classical)
from .newton import NewtonConfig, solve_newton

__all__ = ["SpeedlineConfig", "BCSwitchConfig", "MapPoint", "StallFlag",
           "SwitchEvent", "MapResult", "solve_speedline"]

_TWO_PI = 2.0 * np.pi

# Boundary-condition modes (section 6.6). The value is also the human label
# recorded on each MapPoint / SwitchEvent.
_NORMAL = "mdot"             # mass-flow specified
_BACKPRESSURE = "p_exit"     # choke-proximal: exit static pressure specified


@dataclass(frozen=True)
class BCSwitchConfig:
    """Hysteretic choke<->normal BC-switch policy (section 6.6).

    Switch to back-pressure mode when the choke margin falls below ``c_sw``
    (or the mdot-specified solve loses its root); switch back to normal mode
    only once the margin recovers past ``c_sw + delta_hys`` — the hysteresis
    band that prevents limit-cycling at the boundary. In back-pressure mode
    the traversal throttles by raising the exit pressure ``bp_step_frac`` per
    point (which lowers mdot, i.e. moves toward stall).
    """

    c_sw: float = 0.05
    delta_hys: float = 0.03
    bp_step_frac: float = 0.01

    def __post_init__(self):
        if not (0.0 < self.c_sw < 1.0):
            raise ConfigError(f"c_sw must be in (0, 1), got {self.c_sw}")
        if self.delta_hys < 0.0:
            raise ConfigError("delta_hys must be >= 0")
        if not (0.0 < self.bp_step_frac < 1.0):
            raise ConfigError("bp_step_frac must be in (0, 1)")


def _next_mode(mode, c, cfg: BCSwitchConfig):
    """Hysteretic mode decision (section 6.6). Returns
    ``(new_mode, switched, reason)`` — pure, so it is directly unit-testable
    for the automatic-and-hysteretic requirement."""
    if mode == _NORMAL and c < cfg.c_sw:
        return _BACKPRESSURE, True, (
            f"choke margin {c:.4g} < c_sw {cfg.c_sw:.4g}")
    if mode == _BACKPRESSURE and c > cfg.c_sw + cfg.delta_hys:
        return _NORMAL, True, (
            f"choke margin {c:.4g} recovered past "
            f"{cfg.c_sw + cfg.delta_hys:.4g}")
    return mode, False, ""


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
    d_factor_max : opt-in blade-loading stall criterion (default ``None`` =
        off). When set, a converged point whose maximum ROTOR-tip Lieblein
        diffusion factor (NACA RM E53D01) reaches this value flags a
        ``blade_loading`` stall — the grounded loading limit for the stall
        LINE (measured on Rotor 37/38 to predict stall within ~3% at 0.60;
        see docs/references/ROTOR37.md gate #5). Checked BEFORE the
        validity/turnover criteria: blade loading is the physical stall
        signal and takes precedence over the closure-window validity gate
        (which, for the transonic-rotor endwall family, saturates as a
        bookkeeping artifact well before real stall — pair ``d_factor_max``
        with a low ``validity_min`` there).
    bc_switch : hysteretic choke<->normal BC-switch policy (section 6.6). When
        ``None`` (default) the traversal stays in mass-flow mode and a lost
        mdot root ends it with a stall flag; when set, choke-proximal points
        switch to a back-pressure branch and switch back on recovery.
    max_points : hard cap on traversed points (runaway guard).
    """

    step_min_frac: float = 0.02
    cutback: float = 0.5
    step_grow: float = 1.5
    validity_min: float = 0.1
    pr_rise_min: float = 1e-3
    escalate: bool = True
    d_factor_max: float = None
    bc_switch: BCSwitchConfig = None
    max_points: int = 200
    classical: ClassicalConfig = field(default_factory=ClassicalConfig)
    newton: NewtonConfig = field(default_factory=NewtonConfig)

    def __post_init__(self):
        if not (0.0 < self.cutback < 1.0):
            raise ConfigError(f"cutback must be in (0, 1), got {self.cutback}")
        if self.step_grow < 1.0:
            raise ConfigError("step_grow must be >= 1")
        if self.d_factor_max is not None and self.d_factor_max <= 0.0:
            raise ConfigError("d_factor_max must be > 0 when set")


@dataclass(frozen=True)
class MapPoint:
    """One converged operating point on the speedline (ARCH-5.4). ``mdot`` is
    the achieved mass flow (the spec in normal mode, an output in
    back-pressure mode); ``mode`` records which boundary condition produced
    it (section 6.6)."""

    mdot: float
    pressure_ratio: float
    choke_margin: float          # c = min_j (1 - mdot/mdot_max_j), section 6.6
    validity: float
    driver: str                  # "classical" | "newton" (which converged it)
    mode: str                    # "mdot" | "p_exit" (section 6.6 BC in force)
    result: ClassicalResult = field(repr=False)


@dataclass(frozen=True)
class StallFlag:
    """Recorded stall/surge onset (section 6.7: report, don't solve through)."""

    mdot: float
    criterion: str               # "solver_failure"|"pr_turnover"|"validity_saturated"
    detail: str = ""


def _tip_diffusion_factor(result: ClassicalResult, resolved_rows):
    """Maximum ROTOR-tip Lieblein diffusion factor over a converged point's
    rotor rows (NACA RM E53D01; section 6.7 blade-loading diagnostic).

    ``D = 1 - W2/W1 + |Vtheta1 - Vtheta2| / (2 sigma W1)`` in the relative
    frame, at the outermost (tip) q-o streamline of each row's EDGE_LE..TE.
    Only rows with ``omega != 0`` (rotors) contribute; returns ``None`` when
    there are none. Driver-layer reduction (post-solve, off the residual
    path — plain arithmetic is fine, cf. ``row_throat_capacity``)."""
    f, fz = result.fields, result.frozen
    r = f.metrics.r
    tr = fz.transported
    n_sl = r.shape[0]
    best = None
    for spec, j_le, j_te, _t in resolved_rows:
        if spec.omega == 0.0:
            continue
        sl = int(np.argmax(r[:, j_le]))            # tip streamline
        r1, r2 = float(r[sl, j_le]), float(r[sl, j_te])
        vm1, vm2 = float(f.vm[sl, j_le]), float(f.vm[sl, j_te])
        wt1 = float(tr.rvt[sl, j_le]) / r1 - spec.omega * r1
        wt2 = float(tr.rvt[sl, j_te]) / r2 - spec.omega * r2
        w1 = float(np.hypot(vm1, wt1))
        w2 = float(np.hypot(vm2, wt2))
        col = r[:, j_le]
        span = ((r1 - col.min()) / (col.max() - col.min())
                if n_sl > 1 and col.max() > col.min() else 0.5)
        sigma = float(spec.geometry.solidity(span))
        d = 1.0 - w2 / w1 + abs(wt1 - wt2) / (2.0 * sigma * w1)
        best = d if best is None else max(best, d)
    return best


def _classify_stall(out_mdot, pr, prev_pr, validity, armed, peak_mdot, config,
                    d_factor=None):
    """Section 6.7 stall-onset classification for one converged point (pure).

    Order is normative: the opt-in **blade-loading** criterion
    (``d_factor_max``) is checked first — it is the physical stall signal and
    takes precedence over the closure-window validity gate (which saturates as
    a bookkeeping artifact for the transonic-rotor endwall family). Then
    **validity saturation before PR turnover**: for a compressor whose
    loss/deviation correlation loses validity as incidence climbs toward stall
    (e.g. Lieblein), validity collapses to 0 at the high-incidence end *before*
    the (correct) loss turns the PR over — so that family flags
    ``validity_saturated``. ``pr_turnover`` fires only when a point on an ARMED
    (already-risen) characteristic actually falls below its predecessor while
    validity is still admissible. Returns a :class:`StallFlag` or ``None``.
    """
    if (config.d_factor_max is not None and d_factor is not None
            and d_factor >= config.d_factor_max):
        return StallFlag(
            mdot=out_mdot, criterion="blade_loading",
            detail=f"rotor-tip diffusion factor {d_factor:.3g} "
                   f">= {config.d_factor_max:.3g} (NACA RM E53D01)")
    if validity < config.validity_min:
        return StallFlag(mdot=out_mdot, criterion="validity_saturated",
                         detail=f"closure validity {validity:.3g} "
                                f"< {config.validity_min:.3g}")
    if armed and prev_pr is not None and pr < prev_pr:
        return StallFlag(
            mdot=out_mdot, criterion="pr_turnover",
            detail=f"PR {pr:.5g} < previous {prev_pr:.5g} at falling mdot "
                   f"(peak ~{peak_mdot:.4g})")
    return None


@dataclass(frozen=True)
class SwitchEvent:
    """A logged boundary-condition switch (section 6.6: automatic + logged)."""

    mdot: float                  # achieved mass flow at the switch
    from_mode: str
    to_mode: str
    reason: str


@dataclass(frozen=True)
class MapResult:
    """A traversed speedline: converged points choke->stall, the BC-switch
    log, and the stall flag that ended it (``None`` if the traversal reached
    ``mdot_min``)."""

    points: tuple                # tuple[MapPoint], choke -> stall order
    stall: StallFlag = None
    switches: tuple = ()         # tuple[SwitchEvent], section 6.6

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


def _achieved_mdot(result: ClassicalResult) -> float:
    """Mass flow at a converged point: the spec value in normal mode, the
    solved trailing state component in back-pressure mode (section 6.6)."""
    fz = result.frozen
    if isinstance(fz.spec, BackPressureSpec):
        return float(unpack(result.x, fz.n_sl, fz.n_qo, backpressure=True)[2])
    return float(fz.spec.mdot)


def _choke_margin(result: ClassicalResult) -> float:
    """Annulus choke margin ``c = min_j (1 - mdot / mdot_max,j)`` (section
    6.6). Row-throat margins (section 4.5) need the INBLADE throat model
    (M7); until then this is annulus-only, and that is what it reports."""
    asm = ResidualAssembler(result.frozen)
    mdot = _achieved_mdot(result)
    caps = [asm.qo_capacity(j, result.fields)
            for j in range(result.frozen.n_qo)]
    return float(min(1.0 - mdot / c for c in caps))


def _exit_static_pressure(result: ClassicalResult, station: int) -> float:
    """Static pressure at the ``q = 0`` node of ``station`` — the
    back-pressure handle a normal-mode point hands to the switch (section
    6.6), so the back-pressure branch starts from a consistent state."""
    fz = result.frozen
    f, tr = result.fields, fz.transported
    r0 = f.metrics.r[0, station]
    vt0 = tr.rvt[0, station] / r0
    h = tr.h0[0, station] - 0.5 * (f.vm[0, station] ** 2 + vt0 ** 2)
    return float(fz.fluid.p(h, tr.s[0, station]))


# ---------------------------------------------------------------------------
# Point solve with classical -> Newton escalation (spec-generic)
# ---------------------------------------------------------------------------
def _solve_point(spec, warm, *, topology, fluid, fidelity, inlet, rows, steps,
                 blockage, metrics_config, config: SpeedlineConfig):
    """Solve for ``spec`` from warm start ``warm``. A MassFlowSpec goes to the
    classical driver first, escalating to Newton on failure (section 6.7). A
    BackPressureSpec is Newton-only (the classical scalar-solve path is
    normal-mode); its warm start is mandatory and always available because the
    driver only enters back-pressure mode after a converged point."""
    common = dict(rows=rows, steps=steps, blockage=blockage,
                  metrics_config=metrics_config)
    if isinstance(spec, BackPressureSpec):
        res = solve_newton(topology, fluid, fidelity, spec, inlet,
                           warm_start=warm, config=config.newton, **common)
        return res, "newton-bp"
    res = solve_classical(topology, fluid, fidelity, spec, inlet,
                          warm_start=warm, config=config.classical, **common)
    if res.converged:
        return res, "classical"
    if config.escalate and warm is not None:
        res_n = solve_newton(topology, fluid, fidelity, spec, inlet,
                             warm_start=warm, config=config.newton, **common)
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
    """Traverse a speedline choke -> stall (sections 6.6, 6.7).

    Progress is measured by the ACHIEVED mass flow, which falls from
    ``mdot_start`` (choke side) toward ``mdot_min`` (stall side). In normal
    mode the mdot spec is stepped down; a failed point cuts the step and
    retries nearer the last success. With ``config.bc_switch`` set, a
    choke-proximal point (margin below ``c_sw``, or a lost mdot root) switches
    to a back-pressure branch that throttles by raising exit pressure — and
    switches back once the margin recovers past the hysteresis band (section
    6.6). Converged points are checked for the section 6.7 turnover and
    validity-saturation stall criteria.
    """
    if not (mdot_start > mdot_min > 0.0):
        raise ConfigError("need mdot_start > mdot_min > 0")
    if not (0.0 < mdot_step <= mdot_start - mdot_min):
        raise ConfigError("need 0 < mdot_step <= mdot_start - mdot_min")
    if metrics_config is None:
        metrics_config = MetricsConfig()

    bc = config.bc_switch
    station = topology.n_qo - 1
    step_min = config.step_min_frac * mdot_start
    resolved_rows = (_resolve_rows(topology, rows)
                     if rows and config.d_factor_max is not None else [])
    solve_kw = dict(topology=topology, fluid=fluid, fidelity=fidelity,
                    inlet=inlet, rows=rows, steps=steps, blockage=blockage,
                    metrics_config=metrics_config, config=config)

    points, switches = [], []
    stall = None
    mode = _NORMAL
    target_mdot, target_p = mdot_start, None
    step = mdot_step
    last_ok = None
    pr_start = None
    armed = False
    prev_pr = None

    while len(points) < config.max_points:
        spec = (MassFlowSpec(target_mdot) if mode == _NORMAL
                else BackPressureSpec(p_exit=target_p, station=station))
        result, driver = _solve_point(spec, last_ok, **solve_kw)

        if not result.converged:
            # Section 6.6 automatic switch: a lost mdot root is the canonical
            # choke-proximal trigger -> jump to the back-pressure branch,
            # seeded from the last good point's exit pressure.
            if (mode == _NORMAL and bc is not None and last_ok is not None
                    and result.status is SolveStatus.CHOKE_LIMITED):
                target_p = _exit_static_pressure(last_ok, station)
                switches.append(SwitchEvent(
                    mdot=_achieved_mdot(last_ok), from_mode=_NORMAL,
                    to_mode=_BACKPRESSURE,
                    reason="mdot root lost (choke-limited, section 6.6)"))
                mode = _BACKPRESSURE
                continue
            if mode == _NORMAL and last_ok is not None:
                step *= config.cutback
                if step >= step_min:
                    target_mdot = _achieved_mdot(last_ok) - step
                    continue
            flag_mdot = (_achieved_mdot(last_ok) if last_ok is not None
                         else target_mdot)
            stall = StallFlag(
                mdot=flag_mdot, criterion="solver_failure",
                detail=(f"{driver} {result.status.value} in {mode} mode"
                        if last_ok is not None else
                        f"cold first point failed: {driver} "
                        f"{result.status.value}"))
            break

        out_mdot = _achieved_mdot(result)
        pr = _pressure_ratio(fluid, result)
        validity = float(result.frozen.closures.validity)
        c = _choke_margin(result)

        peak_mdot = points[-1].mdot if points else out_mdot
        d_factor = (_tip_diffusion_factor(result, resolved_rows)
                    if config.d_factor_max is not None else None)
        stall = _classify_stall(out_mdot, pr, prev_pr, validity, armed,
                                peak_mdot, config, d_factor)
        if stall is not None:
            break

        points.append(MapPoint(
            mdot=out_mdot, pressure_ratio=pr, choke_margin=c,
            validity=validity, driver=driver, mode=mode, result=result))
        last_ok = result
        if pr_start is None:
            pr_start = pr
        if pr > pr_start + config.pr_rise_min:
            armed = True
        prev_pr = pr
        if out_mdot <= mdot_min + 1e-12:
            break

        # Hysteretic mode update (section 6.6), then advance the active
        # parameter toward stall (lower achieved mdot).
        if bc is not None:
            new_mode, switched, reason = _next_mode(mode, c, bc)
            if switched:
                switches.append(SwitchEvent(mdot=out_mdot, from_mode=mode,
                                            to_mode=new_mode, reason=reason))
                if new_mode == _BACKPRESSURE:
                    target_p = _exit_static_pressure(result, station)
                else:
                    target_mdot, step = out_mdot, mdot_step
                mode = new_mode
        if mode == _NORMAL:
            step = min(step * config.step_grow, mdot_step)
            target_mdot = out_mdot - step
        else:
            target_p *= (1.0 + bc.bp_step_frac)   # throttle -> lower mdot

    return MapResult(points=tuple(points), stall=stall,
                     switches=tuple(switches))
