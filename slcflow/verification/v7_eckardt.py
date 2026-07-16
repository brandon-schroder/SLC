"""V7 — Eckardt rotor O, geometry-faithful endpoints (Theory Manual section
9.7; the point-by-point V7 validation case, ``model-readiness`` gate #1).

The DFVLR radial-discharge laser-anemometry impeller (Eckardt 1976), the
canonical centrifugal validation article. Upgrades the first-order
``tools/eckardt_anchor.py`` (concentric-bend approximation, assumed hub/width)
to the PRIMARY-PAPER geometry.

Provenance — Eckardt, "Detailed Flow Investigations Within a High-Speed
Centrifugal Compressor Impeller", ASME J. Fluids Eng. 98 (1976); extracted
verbatim from the paper via the "Turbomachinery: Test Cases" NotebookLM
notebook (2026-07-15; see docs/references/ECKARDT.md):

  * D2 = 400 mm (r2 = 0.200 m), **20 radially-ending blades** (Z = 20,
    beta2b = 0) — both previously "widely cited, unconfirmed".
  * Inducer LE: tip diameter 280 mm (r1t = 0.140 m), **hub diameter 90 mm
    (r1h = 0.045 m)** — the previously-missing hub radius.
  * Exit width **b2 = 26 mm**; axial length of impeller **130 mm**;
    relative tip clearance z_s/b2 = 0.027 (0.70 mm).
  * Laser operating point: **14 000 rpm, 5.31 kg/s — measured stage
    PR 2.1, stage eta_is 0.88** (near stage optimum). Design speed
    18 000 rpm at 7.16 kg/s (paper Fig. 16; Cumpsty quotes design PR 3.0).

Modelling choices (recorded):

  * Wall contours: quarter-ELLIPSES through the grounded endpoints — hub
    (z 0 -> 0.130 m, r 0.045 -> 0.200 m), shroud (z 0 -> 0.104 m,
    r 0.140 -> 0.200 m; 0.104 = 0.130 - b2) — axial-tangent at inlet,
    radial-tangent at exit, so the constant-radius inlet duct and the
    radial vaneless-diffuser stub join C1. The paper's Fig. 1 contours are
    a raster figure; the ellipse assumption is the recorded residual
    geometry infidelity (the blade camber lines are themselves elliptic).
  * Inducer metal angle: not stated numerically in the paper ("elliptic
    camber"); set for ZERO INCIDENCE at the 18 000-rpm design triangle,
    spanwise: beta1(y) = -atan(omega_design r1(y) / Vm1_design) with
    Vm1_design = 112 m/s (compressible one-D at 7.16 kg/s through the
    grounded inducer annulus). Recorded assumption, hub -37 deg / tip -67.
  * chord/solidity are friction-length representatives (meridional blade
    length ~ 0.18 m), not transcription — the skin-friction loss is the
    only consumer.
  * slcflow models the IMPELLER only: measured PR/eta are STAGE values
    (impeller + vaneless diffuser). PR comparison is nearly direct (the
    diffuser only loses a few % p0), so slcflow impeller-exit PR should sit
    AT-OR-SLIGHTLY-ABOVE the measured stage PR; stage efficiency is NOT
    directly comparable (deferred parasitic disk/recirculation/leakage +
    diffuser losses) — see docs/references/ECKARDT.md.

Status: measured-agreement record, pinned in ``tests/test_v7_eckardt.py``.
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

__all__ = ["EckardtO", "LASER_POINT", "DESIGN_POINT"]

_DEG = np.pi / 180.0

# Measured anchors (stage values; impeller+vaneless diffuser — see docstring).
LASER_POINT = {"rpm": 14000.0, "mdot": 5.31, "stage_pr": 2.1,
               "stage_eta": 0.88}
DESIGN_POINT = {"rpm": 18000.0, "mdot": 7.16, "stage_pr": 3.0}


@dataclass(frozen=True)
class EckardtO:
    """Geometry-faithful-endpoints Eckardt rotor O (section 9.7)."""

    rpm: float = 14000.0          # laser-measurement speed
    mdot: float = 5.31            # kg/s
    T0_in: float = 288.15         # K (ambient rig inlet)
    p0_in: float = 101325.0       # Pa
    r1h: float = 0.045
    r1t: float = 0.140
    r2: float = 0.200
    b2: float = 0.026
    z_len: float = 0.130          # impeller axial length
    blade_count: int = 20
    beta2_blade_deg: float = 0.0  # radially-ending blades
    vm1_design: float = 112.0     # m/s, 18 000-rpm design (docstring)
    rpm_design: float = 18000.0
    chord: float = 0.18           # friction-length representative
    solidity: float = 2.5
    clearance: float = 7.0e-4     # z_s = 0.027 b2
    n_inblade: int = 6
    n_sl_rep: int = 7
    n_span_nodes: int = 21
    gas: PerfectGas = field(default_factory=PerfectGas)

    @property
    def omega(self) -> float:
        return self.rpm * 2.0 * np.pi / 60.0

    # ------------------------------------------------------------------
    def _walls(self):
        """Hub/shroud parametric callables: inlet straight + quarter-ellipse
        + radial diffuser stub, C1 at the joins (axial/radial tangents)."""
        z_in = -0.05
        r_ext = 0.230              # radial stub end (vaneless entry)

        def make(r_le, z_te):
            a_z, a_r = z_te, self.r2 - r_le
            # segment arc lengths (straight, ellipse ~ Ramanujan, straight)
            l1 = -z_in
            h = ((a_z - a_r) / (a_z + a_r)) ** 2
            l2 = 0.25 * np.pi * (a_z + a_r) * (
                1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))
            l3 = r_ext - self.r2
            L = l1 + l2 + l3
            u1, u2 = l1 / L, (l1 + l2) / L

            def wall(u):
                u = np.asarray(u, dtype=float)
                th = (u - u1) / (u2 - u1) * 0.5 * np.pi
                z = np.where(u < u1, z_in + u * L,
                             np.where(u > u2, z_te,
                                      a_z * np.sin(th)))
                r = np.where(u < u1, r_le,
                             np.where(u > u2,
                                      self.r2 + (u - u2) * L,
                                      self.r2 - a_r * np.cos(th)))
                return z, r

            return wall, u1, u2

        hub = make(self.r1h, self.z_len)
        shd = make(self.r1t, self.z_len - self.b2)
        return hub, shd

    def _flowpath(self) -> FlowPath:
        (hub, h1, h2), (shd, s1, s2) = self._walls()
        w0 = WallCurve.from_callable(hub, n=301)
        w1 = WallCurve.from_callable(shd, n=301)
        stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                    StationDef(StationType.EDGE_LE, h1, s1, row_id="imp")]
        for k in range(self.n_inblade):
            t = (k + 1) / (self.n_inblade + 1)
            stations.append(StationDef(
                StationType.INBLADE, h1 + t * (h2 - h1), s1 + t * (s2 - s1),
                row_id="imp"))
        stations += [StationDef(StationType.EDGE_TE, h2, s2, row_id="imp"),
                     StationDef(StationType.DUCT, 1.0, 1.0)]
        return FlowPath(w0, w1, stations)

    def _geometry(self) -> ParamRowGeometry:
        y = np.linspace(0.0, 1.0, self.n_span_nodes)
        r1 = self.r1h + y * (self.r1t - self.r1h)
        om_d = self.rpm_design * 2.0 * np.pi / 60.0
        beta1 = -np.arctan2(om_d * r1, self.vm1_design)
        return ParamRowGeometry(
            blade_count=self.blade_count,
            beta1=beta1,
            beta2=self.beta2_blade_deg * _DEG if self.beta2_blade_deg
            else 0.0,
            chord_len=self.chord, solidity_val=self.solidity,
            clearance=self.clearance)

    def machine(self) -> Machine:
        cp = self.gas.gamma * self.gas.R / (self.gas.gamma - 1.0)
        row = RowSpec(row_id="imp", omega=self.omega,
                      swirl=CENTRIFUGAL.swirl, loss=CENTRIFUGAL.loss,
                      blade_count=self.blade_count, geometry=self._geometry())
        return Machine(self._flowpath(), self.gas,
                       InletCondition(h0=cp * self.T0_in, s=0.0, rvt=0.0),
                       rows=[row])

    def evaluate(self, n_sl: int = 1, fidelity: FidelityConfig = None,
                 mdot: float = None) -> PerformanceResult:
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        target = self.mdot if mdot is None else mdot
        return self.machine().evaluate(MassFlowSpec(target), fidelity,
                                       n_sl=n_sl)

    # ------------------------------------------------------------------
    def parasitic_breakdown(self, result: PerformanceResult,
                            mdot: float = None) -> dict:
        """Aungier ch.-4 parasitic (shaft-side) works [J/kg] evaluated on
        the converged state (post-solve scalars; gate-#3 components —
        see closures/centrifugal/parasitic.py for provenance and why
        these are machine-level, not per-streamtube). Disk backface gap
        ratio 0.02 is a recorded assumption (not published for the rig).
        """
        from ..closures.centrifugal.parasitic import (
            disk_friction_work, leakage_work, recirculation_work)
        md = self.mdot if mdot is None else mdot
        res = result.result
        f, tr = res.fields, res.frozen.transported
        j_le, j_te = 1, 2 + self.n_inblade
        r1 = float(np.mean(f.metrics.r[:, j_le]))
        r2 = float(np.mean(f.metrics.r[:, j_te]))
        cu1 = float(np.mean(tr.rvt[:, j_le])) / r1
        cu2 = float(np.mean(tr.rvt[:, j_te])) / r2
        vm1 = float(np.mean(f.vm[:, j_le]))
        cm2 = float(np.mean(f.vm[:, j_te]))
        rho2 = float(np.mean(f.rho[:, j_te]))
        u1, u2 = self.omega * r1, self.omega * r2
        w1 = float(np.hypot(vm1, u1 - cu1))
        wu2 = u2 - cu2
        w2 = float(np.hypot(cm2, wu2))
        return {
            "disk_friction": disk_friction_work(md, rho2, u2, self.r2),
            "leakage": leakage_work(
                md, rho2, u2, r1, self.r2, self.r1t - self.r1h, self.b2,
                cu1, cu2, self.blade_count, self.clearance, self.chord),
            "recirculation": recirculation_work(
                u2, w1, w2, cm2, wu2, 0.0,   # radial blades: cot(90) = 0
                r1 * cu1, self.r2 * cu2, self.blade_count, self.chord),
        }

    def stage_efficiency(self, result: PerformanceResult,
                         mdot: float = None) -> float:
        """Total-to-total efficiency with the parasitic works debited from
        the shaft side: ``eta = dh_ideal(PR) / (dh0_flow + sum dh_par)``.
        Still excludes the vaneless-diffuser p0 loss (the measured 'stage'
        plane sits at R/R2 = 2) — the recorded remaining gap."""
        par = sum(self.parasitic_breakdown(result, mdot).values())
        cp = self.gas.gamma * self.gas.R / (self.gas.gamma - 1.0)
        kappa = (self.gas.gamma - 1.0) / self.gas.gamma
        dh_id = cp * self.T0_in * (result.pressure_ratio ** kappa - 1.0)
        dh0 = dh_id / result.efficiency
        return float(dh_id / (dh0 + par))
