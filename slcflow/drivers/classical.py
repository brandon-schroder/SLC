"""Classical nested driver (Theory Manual section 6.2; ARCH-5.2).

Orchestration over ``ResidualAssembler`` pieces, exactly the section 6.2
scheme: initialize (area-rule streamlines, 1-D continuity Vm, one transport
sweep), then outer-iterate { geometry pass -> per-q-o safeguarded scalar
solves on the subsonic branch -> streamline repositioning by cumulative
mass-flow inversion with adaptive relaxation -> lagged field refresh } until
all three section 6.2.5 norms converge.

Driver-layer code is *outside* the pure residual path: data-dependent
branching, bracketing, and root-finding iteration are its job (AD-6 binds
assembly, not orchestration). Failures return typed statuses (ARCH-6),
never exceptions. This driver also owns the single NaN/Inf boundary check
of AD-10/ARCH-6 (the reproducer-bundle serialization itself is M5 scope;
the reason string records what failed).

Section 6.4 relaxation: omega_sl <= C (1 - Mm^2) (dm/dq)^2, evaluated from
the per-iteration worst-case grid aspect ratio and meridional Mach, capped
by the user maximum. The constant C is the [VERIFY] Wilkinson constant,
default 1.0 until the M3 free-vortex calibration (ARCH-8).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # driver layer: orchestration, not residual path  # ad6: allow
from scipy.optimize import brentq

from ..assembly.assembler import AssembledFields, ResidualAssembler
from ..assembly.inputs import ClosureFields, FrozenInputs
from ..assembly.pack import pack
from ..closures.interfaces import (LossModel, RowFlowView, RowView,
                                   SwirlClosure)
from ..diagnostics.record import (ConvergenceRecord, IterationRecord,
                                  SolveStatus)
from ..errors import ConfigError
from ..geometry.flowpath import StationType
from ..grid.core import GridTopology, MetricsConfig, initialize_positions
from ..grid.quadrature import invert_cumulative
from ..transport.streamwise import (TransportFields, TransportStep,
                                    row_steps, sweep)
from ..types import FidelityConfig, MassFlowSpec

__all__ = ["ClassicalConfig", "ClassicalResult", "RowSpec", "solve_classical"]

_TWO_PI = 2.0 * np.pi
_BRACKET_SCAN = 64      # coarse-scan points for the 6.5 subsonic bracket


@dataclass(frozen=True)
class ClassicalConfig:
    """Driver settings (section 6.2, 6.4). All tolerances are the relative
    norms of section 6.2.5."""

    max_outer: int = 200
    tol_pos: float = 1e-9       # max |dq| / q-o length
    tol_cont: float = 1e-9      # max_j |F_j| / mdot
    tol_closure: float = 1e-9   # closure-update norm (static closures: 0)
    omega_sl_max: float = 0.7   # user cap on the relaxation factor
    # Section 6.4 relaxation constant, CALIBRATED at M3-3 (Appendix C.3,
    # tools/calibrate_wilkinson.py): omega = wilkinson_c * (1 - Mm^2) *
    # (dm_min/L_qo)^1.5 with the kappa lag at 0.3. Measured threshold
    # constant ~7.3; 4.4 is the 0.6x-margin default. Recalibrate (rerun the
    # tool) after any change to repositioning or curvature-lag machinery.
    wilkinson_c: float = 4.4
    # Section 5.5 curvature under-relaxation. None resolves per tier: 0.3
    # when the curvature term is active ("on by default in Tier 3"), 1.0
    # (off) otherwise. Without it the curvature-repositioning feedback is
    # unstable at ANY omega_sl on station-dense curved paths (measured at
    # M3-1: the streamwise odd-even mode diverges even at omega = 0.05).
    kappa_relax: float = None
    # Section 6.2.4: closure outputs are updated UNDER-RELAXED. Applied to
    # the lagged per-row (exit rVt, delta_s) between outer iterates;
    # measured at M4-3: flow-coupled swirl closures oscillate and blow up
    # without it.
    closure_relax: float = 0.5
    brentq_rtol: float = 1e-12

    def __post_init__(self):
        if self.max_outer < 1:
            raise ConfigError(f"max_outer must be >= 1, got {self.max_outer}")
        if not (0.0 < self.omega_sl_max <= 1.0):
            raise ConfigError(
                f"omega_sl_max must be in (0, 1], got {self.omega_sl_max}")
        if self.kappa_relax is not None \
                and not (0.0 < self.kappa_relax <= 1.0):
            raise ConfigError(
                f"kappa_relax must be in (0, 1], got {self.kappa_relax}")


@dataclass(frozen=True)
class ClassicalResult:
    """Solve outcome: typed status, final packed state, assembled fields,
    the frozen inputs of the last iterate (deterministic replay, AD-3), and
    the full convergence record (ARCH-6).

    ``fields``/``frozen`` are ``None`` only when the input-side boundary
    check fired before the first assembly completed (the record's ``reason``
    says why)."""

    status: SolveStatus
    x: np.ndarray
    fields: AssembledFields | None
    frozen: FrozenInputs | None
    record: ConvergenceRecord

    @property
    def converged(self) -> bool:
        return self.status is SolveStatus.CONVERGED


@dataclass(frozen=True)
class RowSpec:
    """One blade row for the solve: its identity, rotation, and closures
    (sections 4.2, 7.1). The row's stations come from the topology
    (StationDef.row_id); edge-only rows until INBLADE lands (M7)."""

    row_id: str
    omega: float
    swirl: SwirlClosure
    loss: LossModel
    blade_count: int = 0
    geometry: object = None     # BladeRowGeometry (section 4.1), M4-3


def _resolve_rows(topology: GridTopology, rows):
    """Map RowSpecs to (spec, j_le, j_te) via the station list; config
    boundary (AD-10): raise on mismatches between rows and stations."""
    stations = topology.flowpath.stations
    by_id = {}
    for j, st in enumerate(stations):
        if st.row_id is not None:
            by_id.setdefault(st.row_id, {})[st.stype] = j
    declared = set(by_id)
    specified = {r.row_id for r in rows}
    if declared != specified:
        raise ConfigError(
            f"row/station mismatch: stations declare {sorted(declared)}, "
            f"RowSpecs provide {sorted(specified)}")
    resolved = []
    for r in rows:
        idx = by_id[r.row_id]
        if StationType.EDGE_LE not in idx or StationType.EDGE_TE not in idx:
            raise ConfigError(
                f"row {r.row_id!r} needs EDGE_LE and EDGE_TE stations")
        j_le, j_te = idx[StationType.EDGE_LE], idx[StationType.EDGE_TE]
        if j_te != j_le + 1:
            raise ConfigError(
                f"row {r.row_id!r}: EDGE_TE must directly follow EDGE_LE "
                "(INBLADE stations land at M7)")
        resolved.append((r, j_le, j_te))
    return resolved


def _row_flow_view(topology, fluid, transported, fields, j, j_te, omega):
    """Section 7.2 LE flow view at station j from the current iterate,
    with lagged TE quantities for iterative closures ("and TE where
    iterative"). Relative-frame quantities use the section 2.4 convention
    (``W_theta = V_theta - omega r``, angles from the meridional)."""
    r = fields.metrics.r[:, j]
    vm = fields.vm[:, j]
    vtheta = transported.rvt[:, j] / r
    w_theta = vtheta - omega * r
    h, s = fields.h[:, j], transported.s[:, j]
    return RowFlowView(psi=topology.psi, r=r, vm=vm, vtheta=vtheta,
                       w_theta=w_theta, alpha=np.arctan2(vtheta, vm),
                       beta=np.arctan2(w_theta, vm), h=h, s=s,
                       T=fields.T[:, j], rho=fields.rho[:, j],
                       a=fluid.a(h, s), fluid=fluid,
                       r_te=fields.metrics.r[:, j_te],
                       vm_te=fields.vm[:, j_te])


def _evaluate_rows(resolved, topology, fluid, transported, fields):
    """Lagged closure evaluation (section 6.2.2.4, AD-4): one swirl + loss
    call per row from the current iterate's LE flow; returns the transport
    steps and the per-row output dicts for ClosureFields/the closure norm."""
    n_qo = topology.n_qo
    steps = [TransportStep()] * (n_qo - 1)
    exit_rvt, delta_s, validity = {}, {}, 1.0
    for spec, j_le, j_te in resolved:
        view = _row_flow_view(topology, fluid, transported, fields, j_le,
                              j_te, spec.omega)
        row = RowView(row_id=spec.row_id, omega=spec.omega,
                      blade_count=spec.blade_count, geometry=spec.geometry)
        swirl = spec.swirl.exit_rvt(row, view)
        loss = spec.loss.evaluate(row, view)
        rvt_te = np.broadcast_to(np.asarray(swirl.rvt, dtype=float),
                                 view.vm.shape)
        ds = np.broadcast_to(np.asarray(loss.delta_s, dtype=float),
                             view.vm.shape)
        # rvt_le is the SWEPT field arriving at EDGE_LE — the section 3.4
        # consistency contract of transport.row_steps.
        steps[j_le] = row_steps(omega=spec.omega,
                                rvt_le=transported.rvt[:, j_le],
                                rvt_te=rvt_te, delta_s_row=ds)[0]
        exit_rvt[spec.row_id] = rvt_te
        delta_s[spec.row_id] = ds
        validity = min(validity, float(swirl.validity),
                       float(loss.validity))
    return steps, exit_rvt, delta_s, validity


def _closure_norm(new_rvt, new_ds, old_rvt, old_ds):
    """Closure-update norm (section 6.2.5, third convergence criterion):
    max relative change of the lagged row outputs between outer iterates."""
    worst = 0.0
    for key in new_rvt:
        if key not in old_rvt:
            return 1.0          # first evaluation: closures just switched on
        scale_rvt = float(np.max(np.abs(new_rvt[key]))) + 1e-30
        scale_ds = float(np.max(np.abs(new_ds[key]))) + 1e-12
        worst = max(
            worst,
            float(np.max(np.abs(new_rvt[key] - old_rvt[key]))) / scale_rvt,
            float(np.max(np.abs(new_ds[key] - old_ds[key]))) / scale_ds)
    return worst


def _vm_upper_bound(frozen: FrozenInputs, fields: AssembledFields, j):
    """Static-enthalpy-positivity bound on Vm at the q = 0 node (the same
    window the A.7 capacity search uses)."""
    r0 = fields.metrics.r[0, j]
    vt0 = frozen.transported.rvt[0, j] / r0
    return 0.999 * float(np.sqrt(np.maximum(
        2.0 * frozen.transported.h0[0, j] - vt0 * vt0, 1e-12)))


def _solve_qo(asm: ResidualAssembler, j, fields, v_hi, rtol, v_prev=None):
    """Solve F_j(Vm_q0) = 0 on the subsonic branch (sections 6.5, 5.4).

    F rises from -mdot at Vm_q0 -> 0 to the capacity peak (A.7), so the
    subsonic root is bracketed between the scan point below the peak where
    F < 0 and the first point at or below the peak where F >= 0. Returns
    ``None`` when the peak stays negative: the q-o cannot pass mdot
    (choke-limited, section 6.6).

    ``v_prev`` warm-starts the bracket from the previous outer iterate
    (streamlines move little per iteration under section 6.4 relaxation);
    the full scan is the cold-start / fallback path. Out-of-domain trial
    velocities produce non-finite F by design (AD-10 saturation happens at
    the fluid domain edge); they are mapped to -inf here, hence the local
    errstate."""
    def F_of(v):
        # NaN-safe for root-finding: a non-finite F means the trial velocity
        # left the fluid domain (h < 0 somewhere along the ODE) — map it to
        # a huge mass deficit so brentq bisects back toward the physical
        # branch instead of raising on an interior NaN.
        return float(np.nan_to_num(asm.continuity_F(j, v, fields),
                                   nan=-1e30, posinf=-1e30, neginf=-1e30))

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        if v_prev is not None:
            lo, hi = 0.7 * v_prev, min(1.3 * v_prev, v_hi)
            F_lo, F_hi = F_of(lo), F_of(hi)
            if np.isfinite(F_lo) and np.isfinite(F_hi) \
                    and F_lo < 0.0 < F_hi:
                return float(brentq(F_of, lo, hi, rtol=rtol))

        grid = np.linspace(v_hi / _BRACKET_SCAN, v_hi, _BRACKET_SCAN)
        F = np.array([F_of(v) for v in grid])
        F = np.where(np.isfinite(F), F, -np.inf)
        k_peak = int(np.argmax(F))
        if F[k_peak] < 0.0:
            return None
        k_up = int(np.argmax(F[:k_peak + 1] >= 0.0))  # first F >= 0 below peak
        if k_up == 0:
            lo = 1e-12 * v_hi  # root below the first scan point
        else:
            lo = grid[k_up - 1]
        return float(brentq(F_of, lo, grid[k_up], rtol=rtol))


def _omega_sl(config: ClassicalConfig, fields: AssembledFields,
              curvature_on: bool = True):
    """Adaptive streamline relaxation factor (section 6.4, as CALIBRATED at
    M3-3 -- Appendix C.3, ``tools/calibrate_wilkinson.py``):

        omega = wilkinson_c * (1 - Mm^2) * (dm_min / L_qo)^1.5

    The measured stability envelope of this implementation's odd-even
    curvature-repositioning mode tracks the *station density* alone --
    thresholds are identical for n_sl = 5, 9, 17 at fixed stations -- so
    Wilkinson's literature (dm/dq)^2 aspect form is deliberately NOT used
    (it both over-throttles coarse grids and, un-capped, licenses divergent
    factors on fine spanwise grids; both measured). Exponent 1.5 and the
    threshold constant ~7.3 are fitted to the C.3 envelope;
    ``wilkinson_c = 4.4`` is the 0.6x-margin default (the fit passes within
    2% of a measured-unstable point, so margin is not optional).

    The throttle exists for the curvature-repositioning feedback loop; with
    the curvature term inactive (Tier 1/2) repositioning is a plain fixed
    point and runs at the user cap (config/tier branching, AD-1)."""
    if not curvature_on:
        return config.omega_sl_max
    dm_min = float(np.min(np.diff(fields.metrics.m, axis=1)))
    span = float(np.max(fields.metrics.qo_length))
    mm2 = float(np.max(fields.mach_m ** 2))
    om = (config.wilkinson_c * max(1.0 - mm2, 0.0)
          * (dm_min / span) ** 1.5)
    return float(np.clip(om, 1e-3, config.omega_sl_max))


def solve_classical(topology: GridTopology, fluid, fidelity: FidelityConfig,
                    spec: MassFlowSpec, inlet: TransportFields,
                    steps=None, rows=(), blockage=None,
                    metrics_config: MetricsConfig = None,
                    config: ClassicalConfig = ClassicalConfig()
                    ) -> ClassicalResult:
    """Run the section 6.2 nested scheme to a converged operating point.

    Parameters
    ----------
    inlet : fields at station 0 as ``(n_sl,)`` columns of a
        :class:`TransportFields`-like bundle (only column 0 is read if 2-D);
        the station march re-sweeps them each outer iterate (section 6.2.2).
    steps : per-interval :class:`TransportStep` sequence for PRESCRIBED
        transport (default: all-duct, the V1/V2 configuration). Mutually
        exclusive with ``rows``.
    rows : :class:`RowSpec` sequence — blade rows whose swirl/loss come
        from closures, evaluated lagged per outer iterate (AD-4, section
        6.2.2.4). The first iterate runs on a duct-only sweep (closures
        need a flow field to evaluate against); rows join from the second.
    blockage : prescribed ``B(i, j)`` schedule (section 7.2), default zero.
    """
    n_sl, n_qo = topology.n_sl, topology.n_qo
    if rows and steps is not None:
        raise ConfigError("steps and rows are mutually exclusive: closure-fed"
                          " rows build their own transport steps")
    resolved_rows = _resolve_rows(topology, rows) if rows else []
    if not rows and steps is None and any(
            st.row_id is not None for st in topology.flowpath.stations):
        raise ConfigError("topology declares blade rows: provide RowSpecs "
                          "(closure-fed) or explicit transport steps")
    if steps is None:
        steps = [TransportStep()] * (n_qo - 1)
    if len(steps) != n_qo - 1:
        raise ConfigError(f"need {n_qo - 1} transport steps, got {len(steps)}")
    if blockage is None:
        blockage = np.zeros((n_sl, n_qo))
    if metrics_config is None:
        metrics_config = MetricsConfig()

    def col0(a):
        a = np.asarray(a, dtype=float)
        return a[:, 0] if a.ndim == 2 else np.broadcast_to(a, (n_sl,))

    inlet_h0, inlet_s, inlet_rvt = (col0(inlet.h0), col0(inlet.s),
                                    col0(inlet.rvt))

    # --- section 6.2.1 initialization -----------------------------------
    q_full = initialize_positions(topology)     # area rule (G-5)
    transported = sweep(inlet_h0, inlet_s, inlet_rvt, steps)
    # 1-D continuity Vm guess: mean-state density, annulus-integral area.
    rho0 = float(np.mean(fluid.rho(np.mean(inlet_h0), np.mean(inlet_s))))
    lengths = np.array([qo.length for qo in topology.flowpath.qo_curves])
    r_mid = np.array([np.mean(qo.point(np.linspace(0, qo.length, 32))[1])
                      for qo in topology.flowpath.qo_curves])
    vm_q0 = np.maximum(spec.mdot / (_TWO_PI * rho0 * r_mid * lengths), 1e-6)
    vm_lagged = np.tile(vm_q0[None, :], (n_sl, 1))

    closures = ClosureFields(blockage, iteration_tag=0)
    # Section 5.5 default resolution: curvature lag on whenever the
    # curvature term is active (config/tier branching, not flow branching).
    kappa_relax = config.kappa_relax if config.kappa_relax is not None \
        else (0.3 if fidelity.curvature_term > 0.0 else 1.0)
    kappa_prev = None      # section 5.5 lag; None on the first iterate
    history = []
    status, reason = SolveStatus.MAX_ITER, ""
    frozen = asm = fields = x = None

    for it in range(1, config.max_outer + 1):
        # AD-10/ARCH-6 boundary check, input side: non-finite lagged fields
        # must become a typed status BEFORE scipy's interpolant constructors
        # can raise on them inside assembly.
        if not all(np.all(np.isfinite(a)) for a in
                   (transported.h0, transported.s, transported.rvt,
                    vm_lagged, vm_q0)):
            status, reason = SolveStatus.NUMERICAL_FAILURE, (
                f"non-finite lagged inputs at outer iteration {it} "
                "(AD-10 boundary check)")
            break

        # (6.2.2.1) geometry pass + lagged-field freeze for this iterate.
        frozen = FrozenInputs(topology=topology, fluid=fluid,
                              fidelity=fidelity, spec=spec,
                              transported=transported, closures=closures,
                              vm_lagged=vm_lagged,
                              kappa_lagged=kappa_prev,
                              kappa_relax=kappa_relax,
                              metrics_config=metrics_config)
        asm = ResidualAssembler(frozen)
        x = pack(vm_q0, q_full[1:-1, :])
        fields = asm.split(x)
        # Section 5.5 lag is recursive (blend against the previously USED
        # field): replicate the assembler's blend to carry the EMA forward.
        if kappa_prev is None:
            kappa_prev = fields.metrics.kappa_m
        else:
            kappa_prev = (kappa_relax * fields.metrics.kappa_m
                          + (1.0 - kappa_relax) * kappa_prev)

        if not all(np.all(np.isfinite(a)) for a in
                   (fields.vm, fields.rho, fields.mach_m)):
            status, reason = SolveStatus.NUMERICAL_FAILURE, (
                f"non-finite assembled fields at outer iteration {it} "
                "(AD-10 boundary check)")
            break

        # (6.2.2.2) station-march scalar solves on the subsonic branch.
        vm_new = np.empty(n_qo)
        choked_j = None
        for j in range(n_qo):
            v = _solve_qo(asm, j, fields, _vm_upper_bound(frozen, fields, j),
                          config.brentq_rtol, v_prev=float(vm_q0[j]))
            if v is None:
                choked_j = j
                break
            vm_new[j] = v
        if choked_j is not None:
            status, reason = SolveStatus.CHOKE_LIMITED, (
                f"q-o {choked_j} cannot pass mdot = {spec.mdot} "
                f"(capacity below target, section 6.6)")
            break
        vm_q0 = vm_new
        fields = asm.split(pack(vm_q0, q_full[1:-1, :]))

        # (6.2.2.3) reposition streamlines: invert THE mass cumulative.
        omega = _omega_sl(config, fields,
                          curvature_on=fidelity.curvature_term > 0.0)
        q_target_cols = []
        for j in range(n_qo):
            cum = asm.mass_cumulative(j, fields.vm[:, j], fields)
            q_target_cols.append(invert_cumulative(
                fields.q[:, j], cum, topology.psi * cum[-1]))
        q_target = np.stack(q_target_cols, axis=1)
        q_next = q_full + omega * (q_target - q_full)   # walls map to selves

        # (6.2.2.4) lagged refreshes: closures re-evaluated from the current
        # iterate (AD-4), transported fields re-swept through the resulting
        # steps, Vm field lagged for the dVm/dm term.
        if resolved_rows:
            steps, exit_rvt, delta_s_row, validity = _evaluate_rows(
                resolved_rows, topology, fluid, transported, fields)
            # Section 6.2.4: closure updates are under-relaxed against the
            # previous lagged outputs (flow-coupled closures oscillate
            # otherwise; measured at M4-3).
            if closures.row_exit_rvt:
                w_cl = config.closure_relax
                exit_rvt = {k: closures.row_exit_rvt[k]
                            + w_cl * (v - closures.row_exit_rvt[k])
                            for k, v in exit_rvt.items()}
                delta_s_row = {k: closures.row_delta_s[k]
                               + w_cl * (v - closures.row_delta_s[k])
                               for k, v in delta_s_row.items()}
                # Rebuild the row steps from the RELAXED outputs so the
                # sweep and ClosureFields stay consistent.
                for rspec, j_le, _j_te in resolved_rows:
                    steps[j_le] = row_steps(
                        omega=rspec.omega,
                        rvt_le=transported.rvt[:, j_le],
                        rvt_te=exit_rvt[rspec.row_id],
                        delta_s_row=delta_s_row[rspec.row_id])[0]
            closure_norm = _closure_norm(exit_rvt, delta_s_row,
                                         closures.row_exit_rvt,
                                         closures.row_delta_s)
            closures = ClosureFields(blockage, row_exit_rvt=exit_rvt,
                                     row_delta_s=delta_s_row,
                                     validity=validity, iteration_tag=it)
        else:
            closure_norm = 0.0      # static closures: norm identically 0
            closures = ClosureFields(blockage, iteration_tag=it)
        transported = sweep(inlet_h0, inlet_s, inlet_rvt, steps)
        vm_lagged = fields.vm

        # (6.2.2.5) all three norms, reported every iteration.
        pos_norm = float(np.max(np.abs(q_next - q_full)) / np.max(lengths))
        cont_norm = float(np.max(np.abs(
            [asm.continuity_F(j, vm_q0[j], fields) for j in range(n_qo)]
        )) / spec.mdot)
        history.append(IterationRecord(iteration=it, cont_norm=cont_norm,
                                       pos_norm=pos_norm,
                                       closure_norm=closure_norm,
                                       omega_sl=omega))
        q_full = q_next
        if (pos_norm < config.tol_pos and cont_norm < config.tol_cont
                and closure_norm < config.tol_closure):
            status = SolveStatus.CONVERGED
            break

    record = ConvergenceRecord(status=status, iterations=tuple(history),
                               reason=reason)
    x_final = pack(vm_q0, q_full[1:-1, :])
    if asm is None:  # input-side boundary check fired before first assembly
        return ClassicalResult(status=status, x=x_final, fields=None,
                               frozen=None, record=record)
    return ClassicalResult(status=status, x=x_final,
                           fields=asm.split(x_final) if status in
                           (SolveStatus.CONVERGED, SolveStatus.MAX_ITER)
                           else fields,
                           frozen=frozen, record=record)
