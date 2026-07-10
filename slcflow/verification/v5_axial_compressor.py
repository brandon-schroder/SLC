"""V5 — Axial compressor (Theory Manual section 9.5; ARCH-7, ARCH-8 M4).

An axial-compressor rotor composed through the :class:`Machine` facade with
the Lieblein NACA-65 correlation set (incidence/deviation + equivalent-
diffusion profile loss), runnable at any fidelity — Tier 1 meanline
(``n_sl = 1``) for the rotor-67-style meanline-level checks the ladder calls
for, or a spanwise grid for Tier 2/3.

**Status (M4 entry point).** This binds the *structural* half of V5: the case
converges end-to-end, does real work with real loss, and lands its
total-to-total pressure ratio and efficiency in physically sane bands. As of
the 2026-07 retune it also runs **inside** the Lieblein correlation's validity
window (closure validity 1.0 at the design point and across most of its
operating line), so the reported loss is in-domain rather than saturated — see
the geometry comment below. The
quantitative half — reproducing a *specific* NASA case (e.g. the two-stage fan
or rotor 67) point-by-point against published throughflow/test data, and
speedline generation with choke-side behaviour — is **[VERIFY]** and blocked
on two things the project has deliberately deferred:

  * the calibrated correlation coefficients (every Lieblein fit coefficient is
    ``[VERIFY]`` pending the reference library — see the closure docstrings),
  * the continuation/BC-switching driver for speedlines and choke traversal
    (M5/V9, ARCH-8).

So the bands here are generous plausibility gates, not validation tolerances;
the reference geometry/operating point is a representative subsonic stage, not
a digitised NASA rotor. Tightening these to Appendix-C V5 tolerances is the
M5-and-beyond task, recorded so the gap is not mistaken for a pass.

Provenance: M4 sub-step 5, written with the Tier-1 machine facade.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # verification layer: case definitions  # ad6: allow

from ..closures.axial_compressor import LIEBLEIN_NACA65
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..geometry.bladerow import ParamRowGeometry
from ..machine import (FidelityConfig, InletCondition, Machine, MassFlowSpec,
                       PerformanceResult, RowSpec)

__all__ = ["V5AxialRotor", "V5MultistageCompressor"]

_DEG = np.pi / 180.0


@dataclass(frozen=True)
class V5AxialRotor:
    """Representative subsonic axial-compressor rotor (section 9.5).

    Cylindrical annulus, uniform axial inflow, one blade row fed by
    :data:`LIEBLEIN_NACA65`. Metal angles are mid-span values (the meanline
    sees only mid-span; a spanwise run may pass arrays via ``geometry``).
    All angles in degrees at this I/O boundary, converted to radians for the
    geometry object (AD-7).
    """

    # Geometry/loading tuned (2026-07) to run INSIDE the Lieblein SP-36
    # validity window at every tier. The original -63/-45 over a hub/tip 0.5
    # annulus was an over-loaded, untwisted blade: with zero inlet swirl the
    # relative inlet angle beta1_flow = atan(omega r / Vm) swings hard across a
    # 2:1 radius ratio, so a constant metal angle drove the equivalent
    # diffusion factor D_eq to ~3 (window [1,2]) and beta1 past 70 deg at the
    # spanwise tiers -> closure validity 0 (the loss became saturated garbage,
    # even though PR/eta stayed in-band). Narrowing to hub/tip 0.75 (a more
    # representative NACA-65 mid-stage annulus) and moderating the turning
    # keeps D_eq in [1.2, 1.6] and beta1 < 55 deg, validity 1.0.
    r0: float = 0.45
    r1: float = 0.6
    length: float = 1.0
    omega: float = 400.0             # rad/s
    mdot: float = 100.0              # kg/s (valid window ~[80, 115]; choke ~118)
    h0_in: float = 3.0e5             # J/kg
    s_in: float = 0.0
    beta1_blade_deg: float = -52.0   # relative metal angle, LE
    beta2_blade_deg: float = -40.0   # relative metal angle, TE
    solidity: float = 1.2
    chord: float = 0.06
    thickness: float = 0.08
    blade_count: int = 31
    gas: PerfectGas = field(default_factory=PerfectGas)

    # Structural plausibility bands (NOT V5 validation tolerances; [VERIFY]).
    pr_band: tuple = (1.02, 1.8)
    eta_band: tuple = (0.80, 0.995)

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
            thickness=self.thickness)
        row = RowSpec(row_id="r1", omega=self.omega,
                      swirl=LIEBLEIN_NACA65.swirl, loss=LIEBLEIN_NACA65.loss,
                      blade_count=self.blade_count, geometry=geom)
        return Machine(self._flowpath(), self.gas,
                       InletCondition(h0=self.h0_in, s=self.s_in, rvt=0.0),
                       rows=[row])

    def evaluate(self, n_sl: int = 1,
                 fidelity: FidelityConfig = None) -> PerformanceResult:
        """Solve the rotor at the requested fidelity (default: Tier-1
        meanline, ``n_sl = 1``)."""
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        return self.machine().evaluate(MassFlowSpec(self.mdot), fidelity,
                                       n_sl=n_sl)


@dataclass(frozen=True)
class V5MultistageCompressor:
    """Repeating-stage axial compressor for the M8 mixing revisit (section
    9.5 / 3.6). ``n_stages`` rotor+stator pairs on a cylindrical annulus, all
    fed by :data:`LIEBLEIN_NACA65`. Each rotor adds swirl and work; each
    stator de-swirls back toward axial, so the stage repeats. Across several
    rows the spanwise entropy stratification accumulates (loss varies with the
    local span flow) -- the configuration section 3.6 mixing exists for. The
    metal angles are a representative matched set, **not** a digitised design.
    """

    n_stages: int = 2
    # hub/tip 0.73 (was 0.64): keeps the untwisted matched-stage blades inside
    # the Lieblein SP-36 validity window (D_eq <= ~1.45, was 2.25 -> validity 0)
    # at every tier, so the loss the mixing comparison sees is in-domain. Same
    # root cause + fix as the single-rotor V5AxialRotor retune (2026-07).
    r0: float = 0.40
    r1: float = 0.55
    length: float = 1.0
    omega: float = 300.0
    mdot: float = 90.0
    h0_in: float = 3.0e5
    s_in: float = 0.0
    rotor_beta_deg: tuple = (-48.0, -30.0)   # relative LE, TE (matched stage)
    stator_beta_deg: tuple = (25.0, -5.0)    # absolute LE, TE (de-swirl to ~0)
    solidity: float = 1.3
    chord: float = 0.05
    thickness: float = 0.08
    blade_count: int = 35
    gas: PerfectGas = field(default_factory=PerfectGas)

    pr_band: tuple = (1.05, 3.0)
    # Endwall loss (Howell secondary + annulus, 2026-07) roughly halves this
    # lightly-loaded matched-stage testbed's net pressure rise (PR 1.18 -> 1.09
    # over 2 stages), so efficiency is loss-dominated and low (~0.64). That is
    # honest physics for a low-PR mixing demonstrator, not an efficiency
    # benchmark -- the band stays structural (widened lower bound for the newly
    # modelled endwall loss), the mixing measurement is what this case exists for.
    eta_band: tuple = (0.55, 0.999)

    def _rows_and_stations(self):
        n_rows = 2 * self.n_stages
        # Blade stations occupy (0.05, 0.95); each row is an LE/TE pair.
        edges = np.linspace(0.06, 0.94, 2 * n_rows)
        stations = [StationDef(StationType.DUCT, 0.0, 0.0)]
        specs = []
        for k in range(n_rows):
            is_rotor = (k % 2 == 0)
            rid = f"{'r' if is_rotor else 's'}{k // 2 + 1}"
            le, te = edges[2 * k], edges[2 * k + 1]
            stations.append(StationDef(StationType.EDGE_LE, le, le, row_id=rid))
            stations.append(StationDef(StationType.EDGE_TE, te, te, row_id=rid))
            b1, b2 = (self.rotor_beta_deg if is_rotor
                      else self.stator_beta_deg)
            geom = ParamRowGeometry(
                blade_count=self.blade_count, beta1=b1 * _DEG, beta2=b2 * _DEG,
                chord_len=self.chord, solidity_val=self.solidity,
                thickness=self.thickness)
            specs.append(RowSpec(
                row_id=rid, omega=(self.omega if is_rotor else 0.0),
                swirl=LIEBLEIN_NACA65.swirl, loss=LIEBLEIN_NACA65.loss,
                blade_count=self.blade_count, geometry=geom))
        stations.append(StationDef(StationType.DUCT, 1.0, 1.0))
        return specs, stations

    def machine(self) -> Machine:
        specs, stations = self._rows_and_stations()
        z = np.linspace(0.0, self.length, 8)
        w0 = WallCurve.from_points(
            np.column_stack([z, np.full_like(z, self.r0)]))
        w1 = WallCurve.from_points(
            np.column_stack([z, np.full_like(z, self.r1)]))
        return Machine(FlowPath(w0, w1, stations), self.gas,
                       InletCondition(h0=self.h0_in, s=self.s_in, rvt=0.0),
                       rows=specs)

    def evaluate(self, n_sl: int = 9, fidelity: FidelityConfig = None,
                 *, mixing=None) -> PerformanceResult:
        """Solve the stack. Default Tier 3 with spanwise mixing on; pass
        ``fidelity=FidelityConfig.tier3()`` (mixing off) for the comparison."""
        if fidelity is None:
            fidelity = FidelityConfig.tier3(mixing_term=1.0)
        return self.machine().evaluate(MassFlowSpec(self.mdot), fidelity,
                                       n_sl=n_sl, mixing=mixing)
