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
from ..diagnostics.record import (ConvergenceRecord, IterationRecord,
                                  SolveStatus)
from ..errors import ConfigError
from ..grid.core import GridTopology, MetricsConfig, initialize_positions
from ..grid.quadrature import invert_cumulative
from ..transport.streamwise import TransportFields, TransportStep, sweep
from ..types import FidelityConfig, MassFlowSpec

__all__ = ["ClassicalConfig", "ClassicalResult", "solve_classical"]

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
    # Section 6.4 Wilkinson constant. Provisionally calibrated at M3-1 on
    # the V2 bend: the measured instability threshold (with the section 5.5
    # kappa lag at 0.3) is ~0.3 x the local aspect factor; 0.15 keeps ~50%
    # margin and was validated at two station densities. [VERIFY] the
    # formal stability-envelope sweep is M3-3.
    wilkinson_c: float = 0.15
    # Section 5.5 curvature under-relaxation. None resolves per tier: 0.3
    # when the curvature term is active ("on by default in Tier 3"), 1.0
    # (off) otherwise. Without it the curvature-repositioning feedback is
    # unstable at ANY omega_sl on station-dense curved paths (measured at
    # M3-1: the streamwise odd-even mode diverges even at omega = 0.05).
    kappa_relax: float = None
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
        return asm.continuity_F(j, v, fields)

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


def _omega_sl(config: ClassicalConfig, fields: AssembledFields):
    """Adaptive streamline relaxation factor (section 6.4, Wilkinson form):
    worst-case local (dm/dq)^2 aspect ratio times (1 - Mm^2), scaled by the
    [VERIFY] constant and capped by the user maximum."""
    dm = np.diff(fields.metrics.m, axis=1)          # (n_sl, n_qo-1)
    dq = np.diff(fields.q, axis=0)                  # (n_sl-1, n_qo)
    aspect2 = (dm[:-1, :] / dq[:, :-1]) ** 2        # per-cell worst pairing
    mm2 = float(np.max(fields.mach_m ** 2))
    om = config.wilkinson_c * max(1.0 - mm2, 0.0) * float(np.min(aspect2))
    return float(np.clip(om, 1e-3, config.omega_sl_max))


def solve_classical(topology: GridTopology, fluid, fidelity: FidelityConfig,
                    spec: MassFlowSpec, inlet: TransportFields,
                    steps=None, blockage=None,
                    metrics_config: MetricsConfig = None,
                    config: ClassicalConfig = ClassicalConfig()
                    ) -> ClassicalResult:
    """Run the section 6.2 nested scheme to a converged operating point.

    Parameters
    ----------
    inlet : fields at station 0 as ``(n_sl,)`` columns of a
        :class:`TransportFields`-like bundle (only column 0 is read if 2-D);
        the station march re-sweeps them each outer iterate (section 6.2.2).
    steps : per-interval :class:`TransportStep` sequence (default: all-duct,
        the M2/V1 configuration; rows arrive with their closures, M4+).
    blockage : prescribed ``B(i, j)`` schedule (section 7.2), default zero.
    """
    n_sl, n_qo = topology.n_sl, topology.n_qo
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
        omega = _omega_sl(config, fields)
        q_target_cols = []
        for j in range(n_qo):
            cum = asm.mass_cumulative(j, fields.vm[:, j], fields)
            q_target_cols.append(invert_cumulative(
                fields.q[:, j], cum, topology.psi * cum[-1]))
        q_target = np.stack(q_target_cols, axis=1)
        q_next = q_full + omega * (q_target - q_full)   # walls map to selves

        # (6.2.2.4) lagged refreshes: transported fields re-swept, Vm field
        # for the dVm/dm term, closures static in M2 (closure norm 0).
        transported = sweep(inlet_h0, inlet_s, inlet_rvt, steps)
        vm_lagged = fields.vm
        closures = ClosureFields(blockage, iteration_tag=it)

        # (6.2.2.5) all three norms, reported every iteration.
        pos_norm = float(np.max(np.abs(q_next - q_full)) / np.max(lengths))
        cont_norm = float(np.max(np.abs(
            [asm.continuity_F(j, vm_q0[j], fields) for j in range(n_qo)]
        )) / spec.mdot)
        closure_norm = 0.0
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
