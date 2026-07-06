"""V8 - Mixed-flow compressor (Theory Manual section 9.8; ARCH-7, ARCH-8 M8).

A mixed-flow impeller turns the meridional flow only *part* of the way from
axial to radial -- exit meridional angle ``phi_max`` in ``(0, 90) deg``,
between the axial V5 and the fully-radial V7. Same centrifugal set (Wiesner
slip + representative internal loss), same parametric-phi machinery (M1), a
milder bend. The partial turn still raises the radius from inducer to exit, so
the impeller does centrifugal-plus-axial work: exit ``rV_theta`` rises from ~0
to the slipped value and ``dh0 > 0`` (compression, PR > 1), exiting at ``phi
~ phi_max``.

**Status (M8 entry point).** Structural gate at **Tier 1 (meanline) and Tier
2 (REE)**: converges on the partial-phi path, does real work with real loss,
exits at the intended intermediate angle with a radius rise, PR/efficiency in
sane bands. Point-by-point reproduction of a specific mixed-flow rotor is
**[VERIFY]** (reference-library calibration + deferred loss); the geometry is
representative, not a digitised design. Efficiency reads high for the same
reason as V7 (only incidence + skin friction modelled).

**Measured (M8-4): Tier-3 mixed-flow repositioning is beyond the current
stabilization.** V7 converges Tier 3 on the *full* 90-degree bend in a narrow
``(n_sl, n_inblade)`` pocket (M7-4); that pocket does **not** transfer to the
partial mixed-flow bend -- across a wide ``(n_sl, n_inblade, omega)`` grid at
``phi_max`` in 45-70 deg, Tier-3 full-SLC repositioning fails to converge
(the section 6.4 odd-even mode again, but the working pocket is not just
narrow, it is *angle-specific*). So the robust radial/mixed repositioning
stabilization carried past M7 (see the module-level stability notes) is the
V8 Tier-3 blocker; it is a driver-stabilization gap, not a closure or
geometry one. ``test_tier3_is_the_known_repositioning_carryover`` pins this as
a tripwire: it will flag the day a stabilization makes Tier 3 converge.

Provenance: M8 sub-step 4, written with the mixed-flow case.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # verification layer: case definitions  # ad6: allow

from ..closures.centrifugal import CENTRIFUGAL
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
    mdot: float = 12.0             # kg/s
    h0_in: float = 3.0e5           # J/kg
    s_in: float = 0.0
    beta1_blade_deg: float = -60.0  # inducer relative metal angle (sgn = -1)
    beta2_blade_deg: float = -30.0  # exit backsweep
    solidity: float = 2.0
    chord: float = 0.08
    blade_count: int = 18
    n_inblade: int = 6             # subdivide the bend (Tier-2 stability, M7-4)
    n_sl_rep: int = 7
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

    def evaluate(self, n_sl: int = 1,
                 fidelity: FidelityConfig = None) -> PerformanceResult:
        """Solve at the requested fidelity (default Tier-1 meanline)."""
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        return self.machine().evaluate(MassFlowSpec(self.mdot), fidelity,
                                       n_sl=n_sl)
