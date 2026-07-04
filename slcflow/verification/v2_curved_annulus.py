"""V2 — Swirl-free flow through a curved annulus (Theory Manual section 9.2;
ARCH-7).

Full Tier-3 problem: curvature term AND streamline repositioning coupled
(the M1 gate froze the streamlines; this case moves them). Reference: the
**planar-limit concentric-bend solution**. For a 90-degree concentric-arc
annulus whose bend center sits at machine radius ``rc`` with bend radii
``R in [R_inner, R_outer]``, the exact swirl-free solution in the limit
``rc >> R`` is the meridional free vortex of A.5 case 2,

    Vm(R) = A / R,     streamlines concentric,

with positions from 1-D continuity (mass weight ``rho Vm``, the ``2 pi rc``
factor common) — computed here on a dense grid, independent of the kernel's
nodal quadrature. At finite ``rc`` the axisymmetric weight ``r(R, theta) =
rc - R cos(theta)`` perturbs the solution at O(R/rc); the default
``rc = 400`` puts that reference deviation (~1e-3) well below the target
grids' discretization error, and the planar-limit family test asserts the
solver converges toward the reference as ``rc`` grows.

This fulfils the analytic side of V2; a cross-check against an external
potential-flow/CFD reference at moderate ``rc`` remains **[VERIFY]** and is
recorded in Appendix C.2. The case is defined by the vortex constant ``A``;
``mdot`` is *derived* from the dense reference (no root-find needed).

The q-o rays of this geometry pass through the bend center, so the bend
radius at a node is exactly ``R = R_outer - q`` (q runs from the outer-bend
wall per A.1.1 — this case exercises the orientation flip for real).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # verification layer: reference implementations  # ad6: allow
from scipy.integrate import cumulative_trapezoid

from ..drivers.classical import ClassicalConfig, ClassicalResult, solve_classical
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..grid import GridTopology
from ..transport import TransportFields
from ..types import FidelityConfig, MassFlowSpec

__all__ = ["V2Exact", "V2CurvedAnnulus"]

_DENSE = 20001


@dataclass(frozen=True)
class V2Exact:
    """Dense planar-limit reference: vortex constant, derived mass flow,
    and the concentric-solution profiles on the q coordinate (from the
    outer-bend wall, matching the solver's A.1.1 orientation)."""

    A: float
    mdot: float
    q_dense: np.ndarray
    psi_dense: np.ndarray
    R_dense: np.ndarray      # bend radius R_outer - q
    vm_dense: np.ndarray     # A / R

    def q_of_psi(self, psi):
        return np.interp(psi, self.psi_dense, self.q_dense)

    def vm_of_q(self, q):
        return np.interp(q, self.q_dense, self.vm_dense)


@dataclass(frozen=True)
class V2CurvedAnnulus:
    """90-degree concentric bend, swirl-free, Tier 3 (section 9.2).

    ``rc`` is the machine radius of the bend center; ``rc >> R_outer`` is
    the planar limit the reference is exact in. ``A = Vm * R`` is the
    target meridional free-vortex constant (sets the speed level).
    """

    A: float = 20.0            # m^2/s -> Vm 100..40 m/s across the span
    rc: float = 400.0
    r_inner: float = 0.2       # bend radii of the two walls
    r_outer: float = 0.5
    h0: float = 3.0e5
    s: float = 0.0
    n_stations: int = 7
    gas: PerfectGas = PerfectGas()

    def topology(self, n_sl, n_stations=None) -> GridTopology:
        n_st = self.n_stations if n_stations is None else n_stations
        zc, rc = 0.0, self.rc

        def wall(R):
            return lambda u: (zc + R * np.sin(0.5 * np.pi * u),
                              rc - R * np.cos(0.5 * np.pi * u))

        w0 = WallCurve.from_callable(wall(self.r_inner), n=201)
        w1 = WallCurve.from_callable(wall(self.r_outer), n=201)
        fracs = np.linspace(0.0, 1.0, n_st)
        fp = FlowPath(w0, w1,
                      [StationDef(StationType.DUCT, f, f) for f in fracs])
        return GridTopology(fp, n_sl=n_sl)

    def exact(self) -> V2Exact:
        q = np.linspace(0.0, self.r_outer - self.r_inner, _DENSE)
        R = self.r_outer - q
        vm = self.A / R
        h = self.h0 - 0.5 * vm * vm
        rho = self.gas.rho(h, self.s)
        # Planar-limit mass weight: rho Vm (the 2 pi rc factor is common to
        # the cumulative and the total, but kept so mdot is dimensional).
        cum = 2.0 * np.pi * self.rc * cumulative_trapezoid(rho * vm, q,
                                                           initial=0.0)
        return V2Exact(A=self.A, mdot=float(cum[-1]), q_dense=q,
                       psi_dense=cum / cum[-1], R_dense=R, vm_dense=vm)

    def solve(self, n_sl, n_stations=None,
              config: ClassicalConfig = ClassicalConfig()
              ) -> ClassicalResult:
        exact = self.exact()
        topo = self.topology(n_sl, n_stations)
        inlet = TransportFields(h0=np.full(n_sl, self.h0),
                                s=np.full(n_sl, self.s),
                                rvt=np.zeros(n_sl))
        return solve_classical(topo, self.gas, FidelityConfig.tier3(),
                               MassFlowSpec(exact.mdot), inlet, config=config)
