"""V1 — Analytic radial equilibrium in a cylindrical annulus
(Theory Manual section 9.1; ARCH-7).

Importable problem definitions with *independent* exact solutions:
free-vortex (rVt = const, incompressible-limit and compressible) and
forced-vortex (Vtheta = Omega_f r) swirling duct flow, Tier 2. The exact
solutions are built from the closed-form REE Vm(r) families plus a dense
(20001-point) 1-D continuity inversion — deliberately NOT the kernel's nodal
quadrature, so agreement is evidence, not tautology. Verifies
master-equation integration, continuity, and repositioning together; the
grid-convergence order check (expect 2nd) binds in tests/.

Closed forms (section A.5 special case 1, uniform h0 and s):
  free vortex:   dVm/dr = 0                    -> Vm(r) = Vm0
  forced vortex: (1/2) dVm^2/dr = -2 Omega^2 r -> Vm^2 = Vm0^2
                                                  - 2 Omega^2 (r^2 - r0^2)
with Vm0 closed by 1-D continuity at the specified mdot.

Tolerances for the bound regression tests are recorded in Theory Manual
Appendix C.1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # verification layer: reference implementations  # ad6: allow
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

from ..drivers.classical import ClassicalConfig, ClassicalResult, solve_classical
from ..errors import ConfigError
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..grid import GridTopology
from ..transport import TransportFields
from ..types import FidelityConfig, MassFlowSpec

__all__ = ["annulus_topology", "V1Exact", "V1FreeVortex", "V1ForcedVortex"]

_DENSE = 20001   # dense-grid resolution of the reference solution


def annulus_topology(r0, r1, length, n_sl, n_stations=4) -> GridTopology:
    """Cylindrical annulus with radial q-o stations (the V1 geometry)."""
    z = np.linspace(0.0, length, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, r0)]))
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, r1)]))
    fracs = np.linspace(0.0, 1.0, n_stations)
    fp = FlowPath(w0, w1, [StationDef(StationType.DUCT, f, f) for f in fracs])
    return GridTopology(fp, n_sl=n_sl)


@dataclass(frozen=True)
class V1Exact:
    """Dense reference solution: hub meridional velocity, exact streamline
    radii at given mass fractions, and the exact Vm(r) profile."""

    vm0: float
    r_dense: np.ndarray
    psi_dense: np.ndarray
    vm_dense: np.ndarray

    def r_of_psi(self, psi):
        return np.interp(psi, self.psi_dense, self.r_dense)

    def vm_of_r(self, r):
        return np.interp(r, self.r_dense, self.vm_dense)


@dataclass(frozen=True)
class _V1Base:
    """Shared annulus/operating definition. Subclasses supply the swirl
    field and the closed-form Vm(r) family."""

    h0: float = 3.0e5
    s: float = 0.0
    mdot: float = 100.0
    r0: float = 0.3
    r1: float = 0.6
    length: float = 1.0
    n_stations: int = 4
    gas: PerfectGas = PerfectGas()

    # --- case-specific closed forms (A.5 case 1) -----------------------
    def vtheta(self, r):
        raise NotImplementedError

    def vm_family(self, r, vm0):
        raise NotImplementedError

    def vm0_floor(self) -> float:
        """Smallest hub velocity for which the Vm(r) family is real across
        the whole annulus (0 unless the family loses speed outward)."""
        return 0.0

    def inlet_rvt(self, psi, exact: V1Exact):
        raise NotImplementedError

    # --- dense reference solution (independent of kernel quadrature) ----
    def exact(self) -> V1Exact:
        r = np.linspace(self.r0, self.r1, _DENSE)
        vt = self.vtheta(r)

        def mass(vm0):
            vm = self.vm_family(r, vm0)
            h = self.h0 - 0.5 * (vm * vm + vt * vt)
            rho = self.gas.rho(h, self.s)
            return 2.0 * np.pi * float(np.trapezoid(rho * vm * r, r))

        # Subsonic branch: bracket Vm0 between the family's reality floor
        # and the 1-D mass-flux peak, located by a dense scan (mass rises
        # to the choke peak).
        vm_hi = 0.999 * np.sqrt(2.0 * self.h0 - float(np.max(vt * vt)))
        vm_lo = self.vm0_floor() + 1e-9 * vm_hi
        scan = np.linspace(vm_lo + (vm_hi - vm_lo) / 400, vm_hi, 400)
        with np.errstate(invalid="ignore"):
            m_scan = np.array([mass(v) for v in scan])
        m_scan = np.where(np.isfinite(m_scan), m_scan, -np.inf)
        k = int(np.argmax(m_scan))
        if m_scan[k] < self.mdot:
            raise ConfigError(
                f"V1 case is choked: capacity {m_scan[k]:.1f} < mdot "
                f"{self.mdot}")
        if mass(vm_lo) >= self.mdot:
            raise ConfigError(
                "V1 case ill-posed: the family's reality floor already "
                "passes mdot; reduce swirl or raise mdot")
        vm0 = float(brentq(lambda v: mass(v) - self.mdot,
                           vm_lo, scan[k], rtol=1e-14))

        vm = self.vm_family(r, vm0)
        h = self.h0 - 0.5 * (vm * vm + vt * vt)
        rho = self.gas.rho(h, self.s)
        cum = 2.0 * np.pi * cumulative_trapezoid(rho * vm * r, r, initial=0.0)
        return V1Exact(vm0=vm0, r_dense=r, psi_dense=cum / cum[-1],
                       vm_dense=vm)

    # --- solver invocation ----------------------------------------------
    def solve(self, n_sl, config: ClassicalConfig = ClassicalConfig()
              ) -> ClassicalResult:
        topo = annulus_topology(self.r0, self.r1, self.length, n_sl,
                                self.n_stations)
        exact = self.exact()
        inlet = TransportFields(
            h0=np.full(n_sl, self.h0), s=np.full(n_sl, self.s),
            rvt=self.inlet_rvt(topo.psi, exact))
        return solve_classical(topo, self.gas, FidelityConfig.tier2(),
                               MassFlowSpec(self.mdot), inlet, config=config)


@dataclass(frozen=True)
class V1FreeVortex(_V1Base):
    """Free vortex, rVt = const: uniform Vm(r) (A.5 case 1)."""

    rvt: float = 15.0

    def vtheta(self, r):
        return self.rvt / r

    def vm_family(self, r, vm0):
        return np.full_like(np.asarray(r, dtype=float), vm0)

    def inlet_rvt(self, psi, exact):
        # Streamtube-attached rVt is the same constant everywhere: the
        # free-vortex case has no prescription-sampling ambiguity.
        return np.full(np.shape(psi), self.rvt)

    @classmethod
    def incompressible(cls) -> "V1FreeVortex":
        """Low-Mach limit (Mm ~ 0.02): positions must match the closed-form
        area rule r_i = sqrt(psi (r1^2 - r0^2) + r0^2) to the compressibility
        residue."""
        return cls(rvt=1.2, mdot=7.0)

    @classmethod
    def compressible(cls) -> "V1FreeVortex":
        """Mm ~ 0.45 with swirl: exercises the density coupling."""
        return cls(rvt=15.0, mdot=150.0)


@dataclass(frozen=True)
class V1ForcedVortex(_V1Base):
    """Forced vortex, Vtheta = Omega_f r: Vm^2(r) = Vm0^2 - 2 Omega_f^2
    (r^2 - r0^2) (A.5 case 1)."""

    omega_f: float = 60.0

    def vtheta(self, r):
        return self.omega_f * np.asarray(r, dtype=float)

    def vm_family(self, r, vm0):
        r = np.asarray(r, dtype=float)
        return np.sqrt(vm0 * vm0
                       - 2.0 * self.omega_f**2 * (r * r - self.r0 * self.r0))

    def vm0_floor(self):
        # Vm(r1) reaches zero when vm0^2 = 2 Omega_f^2 (r1^2 - r0^2).
        return float(np.sqrt(2.0) * self.omega_f
                     * np.sqrt(self.r1**2 - self.r0**2))

    def inlet_rvt(self, psi, exact):
        # The forced-vortex field is prescribed per streamtube at the EXACT
        # streamline radii (part of the reference solution), so the solved
        # problem is the analytic one and not a sampling perturbation of it.
        r_star = exact.r_of_psi(psi)
        return self.omega_f * r_star**2
