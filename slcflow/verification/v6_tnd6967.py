"""V6 — NASA TN D-6967 two-stage turbine, geometry-faithful (Theory Manual
section 9.6; the stage-level point-by-point V6 case, ``model-readiness``
gate #1).

The Kofskey & Nusbaum cold-air two-stage turbine for a small low-cost
turbofan — the first machine-level measured turbine in the verification set
(LS-89 covers the cascade level; see ``tests/test_v6_ls89.py``).

Provenance (transcribed 2026-07-16 from the primary report):
    Kofskey & Nusbaum, "Design and Cold-Air Investigation of a Turbine for
    a Small Low-Cost Turbofan Engine", NASA TN D-6967 (1972). NTRS
    19720024422 (public domain); also in the "Turbomachinery: Test Cases"
    NotebookLM notebook.

  * Table I  — design operating values (two-stage: 977.78 K, 28.544 N/cm²,
    2.994 kg/s, 28 000 rpm; AIR-EQUIVALENT: 1.989 kg/s, 15 336 rpm,
    PR_tt,eq 3.765, design eta_tt 0.880).
  * Table II — per-row aerodynamic parameters at hub/mean/tip (turning,
    chord, solidity, reaction, aspect ratio, blade counts 35/42/43/44,
    rotor tip clearances 0.030/0.038 cm).
  * Figure 1 — complete design free-stream velocity diagrams (hub/mean/
    tip, stations 1-5; angles from AXIAL): the flow-angle set below.
  * Text     — constant mean diameter 20.32 cm; stator-1 height constant
    3.363 cm; stator-2 height 3.945 -> 4.483 cm; stator p0 losses 4%/5%
    design intent; free-vortex design.
  * Table IV — MEASURED at design-equivalent speed and pressure ratio:
    two-stage eta_tt 0.93 (design 0.88!), equivalent specific work
    84.90 J/g (design 80.41), equivalent mass flow 2.004 kg/s (1.989).

Modelling choices (recorded):

  * Metal angles = the Figure 1 design flow angles (zero-incidence design;
    exit deviation is carried by the throat rule, not the TE metal angle).
    Stator-1's true inlet is exactly axial, which the section 4.1 contract
    excludes (orientation needs a sign) — its LE metal is set to +3 deg as
    a sign carrier (the K-O loss uses the FLOW inlet angle, so this only
    fixes the frame sign).
  * Throats from the design exit flow angles: o = cos(alpha_exit) * pitch
    per span (the design-intent gauging; TN D-6967 gives no throat table).
  * Rotor-1/rotor-2 exit heights derived from Table II aspect ratios
    (height_avg = AR * chord_mean): 3.945 cm (= stator-2 inlet, quoted)
    and 5.06 cm.
  * Blade max t/c 0.15 and TE t_te/o 0.05 are representative assumptions
    (Table III coordinates untranscribed); Reynolds ~ 4e5 (equivalent
    cold-air chord Re, inside the K-O flat band).
  * The K-O set carries NO tip-clearance loss; the rotors ran 0.9%/0.8%
    clearance/height — efficiency reads accordingly optimistic.

Status: measured-agreement record, pinned in ``tests/test_v6_tnd6967.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # verification layer: case definitions  # ad6: allow

from ..closures.axial_turbine import KACKER_OKAPUU
from ..closures.axial_turbine.loss import KackerOkapuuLoss
from ..drivers.classical import ClassicalConfig
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..geometry.bladerow import ParamRowGeometry
from ..machine import (FidelityConfig, InletCondition, Machine, MassFlowSpec,
                       PerformanceResult, RowSpec)

__all__ = ["TND6967Turbine", "TND6967FirstStage", "MEASURED_EQ",
           "DESIGN_EQ", "MEASURED_MAP", "MEASURED_EQ_S1", "DESIGN_EQ_S1"]

_DEG = np.pi / 180.0

# Air-equivalent design / measured anchors (Tables I and IV).
DESIGN_EQ = {"mdot": 1.989, "rpm": 15336.0, "pr_tt": 3.765,
             "eta_tt": 0.88, "work_J_per_g": 80.41}
MEASURED_EQ = {"mdot": 2.004, "eta_tt": 0.93, "work_J_per_g": 84.90}

# Figure 17(a) multi-speed map points (digitized 2026-07-17 from a 300-dpi
# page render, calibrated on the axis ticks; the plotted equivalent-design
# dot reproduces its published coordinates (3.22e3, 84.9) exactly — the
# calibration control). Each entry: percent equivalent design speed, the
# PR_tt contour value, equivalent specific work [J/g] (read +-1.5), and the
# equivalent mass flow recovered from the map abscissa (x / omega_eq; the
# near-vertical speed lines make this the rig's CHOKED flow, ~2.0 kg/s at
# every speed — rising ~1.5 percent from 100 to 50 percent speed).
MEASURED_MAP = {
    "speed_pct": np.array([90.0, 90.0, 70.0, 50.0]),
    "pr_tt": np.array([3.4, 3.0, 2.6, 2.2]),
    "work_J_per_g": np.array([79.0, 71.0, 62.5, 46.9]),
    "mdot": np.array([2.00, 1.99, 2.01, 2.03]),
}

# FIRST-STAGE-ONLY operation (Table IV first-stage columns + report text
# p. 16/18: equivalent design-inlet to rotor-exit PR_tt 2.018, PR_ts
# 2.298; the rig removed stage 2 and installed fairing pieces for a
# smooth first-stage exit). The isentropic back-check from the design
# column (work 45.83 / eta 0.870) reproduces PR_tt 2.020 — transcription
# control.
DESIGN_EQ_S1 = {"mdot": 1.989, "pr_tt": 2.018, "eta_tt": 0.870,
                "work_J_per_g": 45.83}
MEASURED_EQ_S1 = {"mdot": 2.005, "eta_tt": 0.93, "work_J_per_g": 49.28}

_R_MEAN = 0.1016                    # constant mean radius [m]

# Per-row transcription: (blade_count, chord_mean [m], solidity h/m/t,
# LE metal deg h/m/t, TE metal deg h/m/t, aspect ratio, exit height [m]).
# Angles: Figure 1 design flow angles from axial; stator exit +, rotor
# exit - (positive toward rotor rotation, section 2.4).
_ROWS = {
    "s1": dict(z=35, chord=0.02616, sol=(1.35, 1.43, 1.39),
               le=(3.0, 3.0, 3.0), te=(68.7, 65.0, 61.5),
               ar=1.29, h_exit=0.03363, omega=False),
    "r1": dict(z=42, chord=0.02606, sol=(2.18, 1.71, 1.50),
               le=(51.4, 29.6, 0.1), te=(-59.6, -61.6, -63.8),
               ar=1.39, h_exit=0.03945, omega=True),
    "s2": dict(z=43, chord=0.02182, sol=(1.56, 1.47, 1.42),
               le=(-31.3, -26.1, -22.3), te=(61.6, 55.2, 49.7),
               ar=1.95, h_exit=0.04483, omega=False),
    "r2": dict(z=44, chord=0.02408, sol=(2.16, 1.66, 1.34),
               le=(42.7, 13.9, 15.5), te=(-42.4, -48.9, -54.3),
               ar=1.98, h_exit=0.0506, omega=True),
}
_H_INLET = 0.03363                  # stator-1 inlet height (constant row)


@dataclass(frozen=True)
class TND6967Turbine:
    """Geometry-faithful two-stage TN D-6967 turbine (section 9.6),
    at the air-equivalent operating point."""

    # Rows included in this configuration (class hook, cf. Rotor37.TABLES;
    # the first-stage-only rig build subclasses with ("s1", "r1")).
    ROW_IDS = ("s1", "r1", "s2", "r2")

    mdot: float = MEASURED_EQ["mdot"]
    rpm: float = DESIGN_EQ["rpm"]
    T0_in: float = 288.15
    p0_in: float = 101325.0
    reynolds: float = 4.0e5
    tc: float = 0.15                # representative max t/c (recorded)
    te_o_ratio: float = 0.05        # representative TE t/o (recorded)
    gas: PerfectGas = field(default_factory=PerfectGas)

    @property
    def omega(self) -> float:
        return self.rpm * 2.0 * np.pi / 60.0

    # ------------------------------------------------------------------
    def _axial_layout(self):
        """Axial extents: rows at fixed z with small gaps; heights linear
        between the transcribed station heights (constant mean radius).
        Only ``ROW_IDS`` rows are laid out; the exit duct sits one
        inter-row-gap + margin past the last TE at its exit height (the
        first-stage rig's fairing pieces = a smooth constant exit)."""
        zrow = {"s1": (0.000, 0.020), "r1": (0.028, 0.048),
                "s2": (0.056, 0.076), "r2": (0.084, 0.104)}
        last = self.ROW_IDS[-1]
        z = {"in": -0.03, "out": zrow[last][1] + 0.036}
        z.update({rid: zrow[rid] for rid in self.ROW_IDS})
        heights = [(z["in"], _H_INLET), (z["s1"][0], _H_INLET)]
        heights += [(z[rid][1], _ROWS[rid]["h_exit"]) for rid in self.ROW_IDS]
        heights.append((z["out"], _ROWS[last]["h_exit"]))
        return z, heights

    def _flowpath(self) -> FlowPath:
        zpos, heights = self._axial_layout()
        zs = np.array([p[0] for p in heights])
        hs = np.array([p[1] for p in heights])
        hub = np.column_stack([zs, _R_MEAN - hs / 2.0])
        tip = np.column_stack([zs, _R_MEAN + hs / 2.0])
        w0 = WallCurve.from_points(hub)
        w1 = WallCurve.from_points(tip)

        def frac(poly, zq):
            seg = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            k = int(np.searchsorted(poly[:, 0], zq) - 1)
            k = min(max(k, 0), len(seg) - 1)
            t = (zq - poly[k, 0]) / (poly[k + 1, 0] - poly[k, 0])
            return float((cum[k] + t * seg[k]) / cum[-1])

        stations = [StationDef(StationType.DUCT, 0.0, 0.0)]
        for rid in self.ROW_IDS:
            for stype, zq in zip((StationType.EDGE_LE, StationType.EDGE_TE),
                                 zpos[rid]):
                stations.append(StationDef(
                    stype, frac(hub, zq), frac(tip, zq), row_id=rid))
        stations.append(StationDef(StationType.DUCT, 1.0, 1.0))
        return FlowPath(w0, w1, stations)

    def _row_specs(self):
        specs = []
        for rid in self.ROW_IDS:
            d = _ROWS[rid]
            r_ex = _R_MEAN + np.array([-0.5, 0.0, 0.5]) * d["h_exit"]
            pitch = 2.0 * np.pi * r_ex / d["z"]
            throat = np.cos(np.abs(np.array(d["te"])) * _DEG) * pitch
            geom = ParamRowGeometry(
                blade_count=d["z"],
                beta1=np.array(d["le"]) * _DEG,
                beta2=np.array(d["te"]) * _DEG,
                chord_len=d["chord"], solidity_val=np.array(d["sol"]),
                thickness=self.tc, throat_val=throat)
            loss = KackerOkapuuLoss(reynolds=self.reynolds,
                                    aspect_ratio=d["ar"],
                                    te_o_ratio=self.te_o_ratio)
            specs.append(RowSpec(
                row_id=rid, omega=self.omega if d["omega"] else 0.0,
                swirl=KACKER_OKAPUU.swirl, loss=loss,
                blade_count=d["z"], geometry=geom))
        return specs

    def machine(self) -> Machine:
        cp = self.gas.gamma * self.gas.R / (self.gas.gamma - 1.0)
        return Machine(self._flowpath(), self.gas,
                       InletCondition(h0=cp * self.T0_in, s=0.0, rvt=0.0),
                       rows=self._row_specs())

    def evaluate(self, n_sl: int = 1, fidelity: FidelityConfig = None,
                 mdot: float = None,
                 config: ClassicalConfig = None) -> PerformanceResult:
        """Solve one operating point. Default config raises ``max_outer``
        to 800: the four-row swirl-continuity Picard chain at the safe
        ``closure_relax = 0.25`` converges cleanly but needs ~500 outer
        iterations (measured 2026-07-16; MAX_ITER at the stock 200)."""
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        if config is None:
            config = ClassicalConfig(max_outer=800)
        target = self.mdot if mdot is None else mdot
        return self.machine().evaluate(MassFlowSpec(target), fidelity,
                                       n_sl=n_sl, config=config)


@dataclass(frozen=True)
class TND6967FirstStage(TND6967Turbine):
    """First-stage-only rig build (report Procedure section: stage 2
    removed, fairing pieces installed for a smooth first-stage exit).
    Measured anchors in :data:`MEASURED_EQ_S1`; the equivalent design
    point is PR_tt 2.018 (PR_ts 2.298) at 15 336 rpm."""

    ROW_IDS = ("s1", "r1")

    mdot: float = MEASURED_EQ_S1["mdot"]
