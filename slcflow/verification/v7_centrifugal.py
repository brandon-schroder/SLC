"""V7 — Centrifugal compressor impeller (Theory Manual section 9.7; ARCH-7,
ARCH-8 M7).

An Eckardt-style radial impeller composed through the :class:`Machine` facade
with the centrifugal correlation set (Wiesner slip + representative internal
loss). The meridional passage turns axial -> radial (phi: 0 -> 90 deg) with a
radius rise from the inducer to the exit, so the blade speed rises
(``U2 = Omega r2 >> U1``) and the impeller does large centrifugal work: the
exit swirl from the slip closure lifts ``rV_theta`` from ~0 (axial inflow) to
``~ sigma U2 r2``, and ``dh0 = Omega d(rV_theta) > 0`` (compression), with a
total-pressure *rise* (PR > 1). This is the first radial end-to-end -- it
exercises the parametric phi -> 90 geometry path (M1) with a blade row.

**Status (M7 entry point).** As with V5/V6, this binds the *structural* half
of V7: the case converges end-to-end at all three tiers, does real centrifugal
work with real loss, reaches phi ~ 90 deg at the exit, and lands PR and
efficiency in physically sane bands. The quantitative half -- reproducing a
specific Eckardt impeller (O/A/B) exit profile or stage map point-by-point --
is **[VERIFY]**, blocked on the reference-library correlation calibration
(every Wiesner/loss coefficient is [VERIFY]) and the deferred loss components.
The reference geometry is a representative backswept impeller, not a digitised
Eckardt rotor; efficiency reads high (~0.98) because only incidence + skin
friction are modelled (blade-loading/clearance/disk-friction deferred).

**Measured finding (M7-4).** Unlike the axial V5/V6 (where a straight annulus
makes Tier 3 == Tier 2 bit-for-bit, V3), this is the first curved-path case
carrying a blade row AND streamline repositioning. Tier-3 full SLC on the
90-degree bend **requires the in-blade subdivision** (``n_inblade = 6``): with
edge-only stations the section 6.4 odd-even streamwise mode diverges at any
relaxation, and the M5 Newton driver inherits the same stiff seed and cannot
recover it. Subdividing the passage keeps the per-step curvature inside the
envelope -- the concrete physical reason radial rows want in-blade stations
(M7-3). The stable pocket is narrow (odd ``n_inblade`` counts near the edge
still flake); a robust radial-repositioning stabilization is a carryover past
M7.

Provenance: M7 sub-step 4, written with the centrifugal set.
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

__all__ = ["V7Centrifugal"]

_DEG = np.pi / 180.0


@dataclass(frozen=True)
class V7Centrifugal:
    """Representative backswept centrifugal compressor impeller (section 9.7).

    The meridional walls are concentric quarter-arcs of the 90-degree
    axial->radial bend whose center sits on the machine axis at the exit
    radius ``r2``: bend radii ``r_inner``/``r_outer`` give an inducer annulus
    ``r in [r2 - r_outer, r2 - r_inner]`` turning to a radial exit at ``r2``.
    Metal angles are mid-span; both negative (inducer relative inflow and the
    backsweep, section 2.4). All angles in degrees at this I/O boundary.
    """

    r2: float = 0.25               # exit radius (bend-center machine radius)
    r_inner: float = 0.08          # inner-bend wall radius
    r_outer: float = 0.18          # outer-bend wall radius
    omega: float = 1450.0          # rad/s -> U2 = 362 m/s (Eckardt-class)
    mdot: float = 12.0             # kg/s
    h0_in: float = 3.0e5           # J/kg
    s_in: float = 0.0
    beta1_blade_deg: float = -60.0  # inducer relative metal angle (sgn = -1)
    beta2_blade_deg: float = -30.0  # exit backsweep (b2b = +30 deg via sgn)
    solidity: float = 2.0
    chord: float = 0.08
    blade_count: int = 18
    # Six INBLADE stations subdivide the passage: MEASURED-necessary for
    # Tier-3 full-SLC repositioning on this 90-degree bend (n_inblade = 0
    # diverges the section 6.4 odd-even streamwise mode; the subdivision keeps
    # the per-step curvature inside the envelope). This is the physical reason
    # radial rows want in-blade stations -- see the module docstring.
    n_inblade: int = 6
    n_sl_rep: int = 7              # spanwise nodes for Tier 2/3 (measured pocket)
    gas: PerfectGas = field(default_factory=PerfectGas)

    # Structural plausibility bands (NOT V7 validation tolerances; [VERIFY]).
    pr_band: tuple = (1.5, 4.0)    # total-to-total compression (PR > 1)
    eta_band: tuple = (0.6, 0.999)  # high end: deferred loss components

    def _flowpath(self) -> FlowPath:
        rc = self.r2

        def wall(R):
            return lambda u: (R * np.sin(0.5 * np.pi * u),
                              rc - R * np.cos(0.5 * np.pi * u))

        w0 = WallCurve.from_callable(wall(self.r_inner), n=201)
        w1 = WallCurve.from_callable(wall(self.r_outer), n=201)
        a_le, a_te = 0.12, 0.90
        stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                    StationDef(StationType.EDGE_LE, a_le, a_le, row_id="imp")]
        # INBLADE stations evenly subdivide the LE->TE passage: on a sharp
        # radial bend they keep the per-step curvature within the section 6.4
        # repositioning envelope (M7-3), the physical reason radial rows want
        # in-blade stations at all.
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
        """Solve the impeller at the requested fidelity (default: Tier-1
        meanline, ``n_sl = 1``)."""
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        return self.machine().evaluate(MassFlowSpec(self.mdot), fidelity,
                                       n_sl=n_sl)
