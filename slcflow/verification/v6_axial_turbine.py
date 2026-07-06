"""V6 — Axial turbine (Theory Manual section 9.6; ARCH-7, ARCH-8 M6).

A pre-swirled axial-turbine rotor composed through the :class:`Machine`
facade with the Kacker-Okapuu correlation set (throat exit angle + profile/
secondary/trailing-edge/shock loss), runnable at any fidelity — Tier 1
meanline (``n_sl = 1``) for meanline-level checks, or a spanwise grid for
Tier 2/3.

The rotor is fed high inlet swirl (as from an upstream nozzle) and turns it
back toward axial, so it *extracts* work: ``dh0 = omega d(rVtheta) < 0``, the
defining turbine behaviour, with a net total-pressure *drop* (expansion +
loss). The blade orientation is negative (turning against rotation), the
turbine sign convention (section 2.4).

**Status (M6 entry point).** As with V5, this binds the *structural* half of
V6: the case converges end-to-end, extracts real work with real loss, and
lands its expansion ratio and efficiency in physically sane bands. The
quantitative half — reproducing a *specific* Kacker-Okapuu validation case or
published stage map point-by-point — is **[VERIFY]** and blocked on the same
two deferrals as V5: the calibrated correlation coefficients (every K-O fit
coefficient is ``[VERIFY]`` pending the reference library — see the closure
docstrings), and speedline/choke traversal. So the bands here are generous
plausibility gates, not validation tolerances.

Provenance: M6 sub-step 5, written with the K-O turbine set.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # verification layer: case definitions  # ad6: allow

from ..closures.axial_turbine import KACKER_OKAPUU
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..geometry.bladerow import ParamRowGeometry
from ..machine import (FidelityConfig, InletCondition, Machine, MassFlowSpec,
                      PerformanceResult, RowSpec)

__all__ = ["V6AxialTurbine"]

_DEG = np.pi / 180.0


@dataclass(frozen=True)
class V6AxialTurbine:
    """Representative axial-turbine rotor with inlet pre-swirl (section 9.6).

    Cylindrical annulus, one rotor row fed by :data:`KACKER_OKAPUU`. Metal
    angles are mid-span values (negative: turbine orientation, section 2.4);
    a spanwise run may pass arrays via ``geometry``. All angles in degrees at
    this I/O boundary, converted to radians for the geometry object (AD-7).
    """

    r0: float = 0.35
    r1: float = 0.50
    length: float = 1.0
    omega: float = 250.0             # rad/s
    mdot: float = 40.0               # kg/s
    h0_in: float = 3.0e5             # J/kg (moderate inlet; s=0 keeps rho sane)
    s_in: float = 0.0
    rvt_in: float = 30.0             # inlet pre-swirl r*Vtheta [m^2/s]
    beta1_blade_deg: float = -25.0   # relative metal angle, LE (turbine sign)
    beta2_blade_deg: float = -58.0   # relative metal angle, TE
    solidity: float = 1.4
    chord: float = 0.03
    thickness: float = 0.10
    throat: float = 0.030            # throat opening o [m]
    blade_count: int = 60
    gas: PerfectGas = field(default_factory=PerfectGas)

    # Structural plausibility bands (NOT V6 validation tolerances; [VERIFY]).
    pr_band: tuple = (0.85, 0.99)    # total-to-total expansion (PR < 1)
    eta_band: tuple = (1.01, 1.15)   # inverted turbine efficiency (>1; facade)

    def _flowpath(self) -> FlowPath:
        z = np.linspace(0.0, self.length, 8)
        w0 = WallCurve.from_points(
            np.column_stack([z, np.full_like(z, self.r0)]))
        w1 = WallCurve.from_points(
            np.column_stack([z, np.full_like(z, self.r1)]))
        stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                    StationDef(StationType.EDGE_LE, 0.35, 0.35, row_id="r1"),
                    StationDef(StationType.EDGE_TE, 0.55, 0.55, row_id="r1"),
                    StationDef(StationType.DUCT, 1.0, 1.0)]
        return FlowPath(w0, w1, stations)

    def machine(self) -> Machine:
        geom = ParamRowGeometry(
            blade_count=self.blade_count,
            beta1=self.beta1_blade_deg * _DEG,
            beta2=self.beta2_blade_deg * _DEG,
            chord_len=self.chord, solidity_val=self.solidity,
            thickness=self.thickness, throat_val=self.throat)
        row = RowSpec(row_id="r1", omega=self.omega,
                      swirl=KACKER_OKAPUU.swirl, loss=KACKER_OKAPUU.loss,
                      blade_count=self.blade_count, geometry=geom)
        return Machine(self._flowpath(), self.gas,
                       InletCondition(h0=self.h0_in, s=self.s_in,
                                      rvt=self.rvt_in),
                       rows=[row])

    def evaluate(self, n_sl: int = 1,
                 fidelity: FidelityConfig = None) -> PerformanceResult:
        """Solve the rotor at the requested fidelity (default: Tier-1
        meanline, ``n_sl = 1``)."""
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        return self.machine().evaluate(MassFlowSpec(self.mdot), fidelity,
                                       n_sl=n_sl)
