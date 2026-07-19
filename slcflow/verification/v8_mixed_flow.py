"""V8 - Mixed-flow compressor (Theory Manual section 9.8; ARCH-7, ARCH-8 M8).

A mixed-flow impeller turns the meridional flow only *part* of the way from
axial to radial -- exit meridional angle ``phi_max`` in ``(0, 90) deg``,
between the axial V5 and the fully-radial V7. Same centrifugal set (Wiesner
slip + representative internal loss), same parametric-phi machinery (M1), a
milder bend. The partial turn still raises the radius from inducer to exit, so
the impeller does centrifugal-plus-axial work: exit ``rV_theta`` rises from ~0
to the slipped value and ``dh0 > 0`` (compression, PR > 1), exiting at ``phi
~ phi_max``.

**Status (M8 entry point; Tier 3 added 2026-07).** Structural gate at all
three tiers: converges on the partial-phi path, does real work with real
loss, exits at the intended intermediate angle with a radius rise,
PR/efficiency in sane bands. Point-by-point reproduction of a specific
mixed-flow rotor is **[VERIFY]** (reference-library calibration + deferred
loss); the geometry is representative, not a digitised design. Efficiency
reads a realistic ~0.86-0.87 now that the dominant blade-loading loss is
modelled (with the Coppage/Oh-1997 diffusion factor, ratio-corrected
2026-07-12; tip-clearance/disk-friction stay deferred).

**Tier-3 status (2026-07-12): converges at the re-centred mdot = 14 with the
realistic loss.** History: M8-4 originally recorded Tier-3 full-SLC
repositioning as failing across a wide ``(n_sl, n_inblade, omega)`` grid; the
2026-07 stabilization refuted that attribution (the failure was the driver's
stale-split boundary check + spurious negative-Vm continuity branches + the
unrelaxed closure switch-on -- fixed by the solved-state check, positive-branch
root validation with freeze-and-patience choke handling, and the section 6.2.4
closure ramp) and Tier 3 converged on the *pre*-blade-loading bend. Adding the
dominant blade-loading loss then pushed Tier 3 into a narrow choke/max-iter
pocket (choke_limited at the old mdot=12). The **Coppage/Oh-1997 D_f ratio fix**
(2026-07-12, ~2.3x less loss -> ~27-30% less spanwise stratification) LOWERED
that pocket into a genuine converging window ``mdot in {13, 14}`` (choke at 12,
slow-max-iter at 15/16). At the re-centred ``mdot = 14`` all three tiers
converge (validity 1, Tier 3 PR agrees Tier 2 to ~2.5%).

**Tier-3 acceleration + pocket widening (2026-07-19, ``wilkinson_c = 13``).**
The section 6.4 throttle was the recorded slowness follow-up (395 outer
iterations at the duct default 4.4). Appendix C.3 measured ``c* = 13.2``
SAFE for exactly this layout (``phi = 55 deg``, ``n_inblade = 6``) with
identical answers, so the case carries a per-case ``wilkinson_c = 13``: Tier 3
converges in ~153 iterations (2.6x faster, inside the stock ``max_outer =
200``) and the pocket WIDENS from ``{13, 14}`` to ``{13, 14, 15}`` (mdot 15
lifts from slow-max-iter into a 264-iteration convergence). The boundaries are
unmoved by relaxation speed: ``mdot = 12`` stays a capacity/stratification
CHOKE fold and ``mdot >= 16`` the upper feasibility edge (still MAX_ITER at 13).
``test_tier3_converges_at_recentred_mdot`` (speed) and
``test_tier3_pocket_widened_by_wilkinson_c`` (widen) pin it. (V7's tighter
90-deg bend is NOT liftable this way -- it stays an infeasible fold; C.7.)

Provenance: M8 sub-step 4, written with the mixed-flow case; Tier-3 status
revised at the 2026-07 stabilization and the 2026-07-12 blade-loading fix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # verification layer: case definitions  # ad6: allow

from ..closures.centrifugal import CENTRIFUGAL
from ..drivers.classical import ClassicalConfig
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..geometry.bladerow import ParamRowGeometry
from ..machine import (FidelityConfig, InletCondition, Machine, MassFlowSpec,
                      PerformanceResult, RowSpec)

__all__ = ["V8MixedFlow"]

_DEG = np.pi / 180.0


@dataclass(frozen=True)
class V8MixedFlow:
    """Representative mixed-flow compressor impeller (section 9.8).

    Concentric partial-bend walls: quarter-arcs swept only through
    ``phi_max`` (not the full 90 deg of V7), bend centre on the axis at
    machine radius ``rc``. Bend radii ``r_inner``/``r_outer`` give an axial
    inducer annulus turning to an exit inclined at ``phi_max`` with a radius
    rise. Metal angles mid-span, both negative (as V7). Angles in degrees at
    this I/O boundary.
    """

    rc: float = 0.25               # bend-centre machine radius
    r_inner: float = 0.08
    r_outer: float = 0.18
    phi_max_deg: float = 55.0      # exit meridional angle (mixed-flow)
    omega: float = 1450.0          # rad/s
    # Re-centred 12 -> 14 kg/s (2026-07-12) after the Coppage/Oh-1997 blade-
    # loading ratio fix (~2.3x less loss). The reduced loss/stratification
    # LOWERED the Tier-3 feasible pocket into a converging window mdot in
    # {13, 14} (was choke_limited at 12 and a knife-edge ~15 pre-fix): at 12 the
    # exit q-o still chokes, 15/16 slow-max-iter. mdot = 14 sits in the pocket
    # with all three tiers converging (validity 1, T3 agrees T2 to ~2.5%). Same
    # category of operating-point re-centre as V7 T2 (12 -> 17). See Appendix C.8.
    mdot: float = 14.0             # kg/s
    h0_in: float = 3.0e5           # J/kg
    s_in: float = 0.0
    beta1_blade_deg: float = -60.0  # inducer relative metal angle (sgn = -1)
    beta2_blade_deg: float = -30.0  # exit backsweep
    solidity: float = 2.0
    chord: float = 0.08
    blade_count: int = 18
    n_inblade: int = 6             # subdivide the bend (Tier-2 stability, M7-4)
    n_sl_rep: int = 7
    # Per-case section 6.4 relaxation override (Appendix C.3, 2026-07-19).
    # The duct-calibrated default (4.4) is 2-3x conservative on blade-row
    # bends; C.3 measured c* = 13.2 SAFE for exactly this layout (phi=55 deg,
    # n_inblade=6; 17.6 fails, 22 diverges) with IDENTICAL answers. Using 13.0
    # (a hair inside the measured-safe point) accelerates Tier 3 ~2.6x
    # (395 -> 153 outer iterations at mdot 14) AND widens the converging
    # pocket from {13, 14} to {13, 14, 15}. Only affects Tier 3 (curvature
    # on); Tier 1/2 run at omega_sl_max regardless (_omega_sl).
    wilkinson_c: float = 13.0
    gas: PerfectGas = field(default_factory=PerfectGas)

    pr_band: tuple = (1.2, 3.0)
    eta_band: tuple = (0.6, 0.999)

    def _flowpath(self) -> FlowPath:
        rc, phi = self.rc, self.phi_max_deg * _DEG

        def wall(R):
            return lambda u: (R * np.sin(phi * u), rc - R * np.cos(phi * u))

        w0 = WallCurve.from_callable(wall(self.r_inner), n=201)
        w1 = WallCurve.from_callable(wall(self.r_outer), n=201)
        a_le, a_te = 0.12, 0.90
        stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                    StationDef(StationType.EDGE_LE, a_le, a_le, row_id="imp")]
        for k in range(self.n_inblade):
            f = a_le + (k + 1) / (self.n_inblade + 1) * (a_te - a_le)
            stations.append(StationDef(StationType.INBLADE, f, f,
                                       row_id="imp"))
        stations += [StationDef(StationType.EDGE_TE, a_te, a_te, row_id="imp"),
                     StationDef(StationType.DUCT, 1.0, 1.0)]
        return FlowPath(w0, w1, stations)

    def machine(self) -> Machine:
        geom = ParamRowGeometry(
            blade_count=self.blade_count,
            beta1=self.beta1_blade_deg * _DEG,
            beta2=self.beta2_blade_deg * _DEG,
            chord_len=self.chord, solidity_val=self.solidity)
        row = RowSpec(row_id="imp", omega=self.omega,
                      swirl=CENTRIFUGAL.swirl, loss=CENTRIFUGAL.loss,
                      blade_count=self.blade_count, geometry=geom)
        return Machine(self._flowpath(), self.gas,
                       InletCondition(h0=self.h0_in, s=self.s_in, rvt=0.0),
                       rows=[row])

    def evaluate(self, n_sl: int = 1, fidelity: FidelityConfig = None,
                 config: ClassicalConfig = None) -> PerformanceResult:
        """Solve at the requested fidelity (default Tier-1 meanline). The
        default config applies the case's C.3-grounded ``wilkinson_c`` so a
        Tier-3 solve converges in ~153 outer iterations (inside the stock
        ``max_outer = 200``); Tier 1/2 are unaffected (curvature off)."""
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        if config is None:
            config = ClassicalConfig(wilkinson_c=self.wilkinson_c)
        return self.machine().evaluate(MassFlowSpec(self.mdot), fidelity,
                                       n_sl=n_sl, config=config)
