"""V5 — NASA Rotor 37, geometry-faithful (Theory Manual section 9.5; the
point-by-point V5 validation case, ``model-readiness`` gate #1).

The first *digitised NASA rotor* in the verification set: NASA Rotor 37
(Reid & Moore transonic core-compressor inlet rotor; design PR 2.106 at
20.188 kg/s, 17188.7 rpm, tip speed 454.14 m/s, M1_rel ≈ 1.5 tip / 1.13 hub).

Provenance (all values transcribed 2026-07-15 from the primary report):
    Moore & Reid, "Performance of Single-Stage Axial-Flow Transonic
    Compressor With Rotor and Stator Aspect Ratios of 1.19 and 1.26,
    Respectively, and With Design Pressure Ratio of 2.05", NASA TP-1659
    (1980). NTRS 19800012840 (public domain). Also a source in the
    "Turbomachinery: Test Cases" NotebookLM notebook.

  * Blade geometry: Table III(a) — radii RI/RO, cone-plane metal angles
    KIC/KOC, max thickness TM, aero chord, solidity, at 11 spanwise blade
    elements (0..100% span from the TIP); 36 blades.
  * Design intent: Table I (rotor PR 2.106, eta_ad 0.889, mdot 20.188).
  * Measured 100%-speed rotor line: Table IV(a), five readings choke→stall
    (:data:`MEASURED_100`).

Modelling choices (recorded; each is a fidelity bound, not a transcription
gap):

  * Metal angles are the report's CONE-PLANE angles applied in the section
    2.4 cascade frame along the meridional — the standard blade-element
    reduction; signed negative (rotor turns W toward less-negative angles,
    the V5 convention).
  * The annulus walls are piecewise-linear hub/casing lines through the
    Table III(a) LE/TE endpoint radii with constant-radius duct extensions
    (the true flowpath has gentle curvature between stations; TP-1659
    Fig. 1 shows the wall contraction is close to linear over the rotor).
  * Tip clearance 0.04 cm (report's nominal running clearance order;
    ``[VERIFY]`` against AGARD-AR-355 for the blind-test value).
  * The stator (Table III(b)) is NOT modelled — rotor-only validation
    against the rotor rows of Table IV(a), which the report tabulates
    separately from the stage values.

Status: this case exists to MEASURE agreement, not to assert it. The
Lieblein NACA-65 correlation set is out of pedigree on MCA transonic
sections — measured 2026-07-15 as a −3.6 deg mean deviation gap driving
PR +12-16%. Since 2026-07-16 the case defaults the **Cetin AGARD-R-745
Eq 3.5 transonic deviation correction** ON (the library-grounded
correction for exactly this blade family, applied as published): per-span
deviation RMS 3.8 -> 1.2 deg, Tier-2 PR lands on the measured value
(2.051 vs 2.056), Tier-1 +3.8%. The choke-side speedline collapse remains
un-modelled (Swan's M1-dependent off-design rule is the recorded next
lever), and the blockage schedule stays open. The pinned tests encode the
MEASURED agreement bands — see ``tests/test_v5_rotor37.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # verification layer: case definitions  # ad6: allow
from scipy.interpolate import PchipInterpolator

from ..closures.axial_compressor import LIEBLEIN_NACA65
from ..closures.axial_compressor.lieblein import LieblienSwirl
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..geometry.bladerow import ParamRowGeometry
from ..machine import (FidelityConfig, InletCondition, Machine, MassFlowSpec,
                       PerformanceResult, RowSpec)

__all__ = ["Rotor37", "MEASURED_100", "MEASURED_BE_4182", "DESIGN"]

_DEG = np.pi / 180.0

# --- NASA TP-1659 Table III(a), rotor 37, tip -> hub (percent span from tip)
_PCT_SPAN = np.array([0., 5., 10., 15., 30., 50., 70., 85., 90., 95., 100.])
_RI_CM = np.array([25.230, 24.935, 24.597, 24.254, 23.211, 21.761,
                   20.246, 19.030, 18.603, 18.161, 17.780])
_RO_CM = np.array([24.506, 24.218, 23.929, 23.641, 22.775, 21.622,
                   20.468, 19.603, 19.314, 19.026, 18.738])
_KIC_DEG = np.array([62.53, 61.66, 60.76, 60.07, 58.48, 56.53,
                     54.24, 52.67, 52.37, 52.18, 52.04])
_KOC_DEG = np.array([49.98, 49.07, 48.18, 47.34, 44.22, 38.87,
                     32.37, 25.28, 22.68, 19.75, 16.75])
_TM_CM = np.array([0.175, 0.186, 0.199, 0.211, 0.250, 0.303,
                   0.360, 0.407, 0.425, 0.443, 0.458])
_CHORD_CM = np.array([5.592, 5.609, 5.603, 5.598, 5.583, 5.570,
                      5.571, 5.591, 5.604, 5.622, 5.627])
_SOLIDITY = np.array([1.288, 1.308, 1.323, 1.339, 1.391, 1.471,
                      1.568, 1.658, 1.693, 1.732, 1.766])
_SETTING_DEG = np.array([60.63, 59.61, 58.54, 57.65, 55.11, 51.16,
                         46.54, 42.82, 41.48, 40.17, 38.92])
# LE/TE axial positions [cm] (ZI / ZO): the swept edge lines.
_ZI_CM = np.array([0.713, 0.665, 0.615, 0.574, 0.466, 0.317,
                   0.176, 0.079, 0.048, 0.021, 0.000])
_ZO_CM = np.array([3.372, 3.424, 3.475, 3.520, 3.644, 3.822,
                   4.015, 4.153, 4.198, 4.241, 4.283])

# --- Table I design intent / Table IV(a) measured 100%-speed rotor line ----
DESIGN = {"mdot": 20.188, "rotor_pr": 2.106, "rotor_eta": 0.889,
          "rpm": 17188.7, "tip_speed": 454.136, "hub_tip": 0.70}
# Five readings, choke -> near-stall (orifice airflow [kg/s], rotor PR,
# rotor adiabatic efficiency).
MEASURED_100 = {
    "mdot": np.array([20.93, 20.83, 20.74, 20.43, 19.60]),
    "rotor_pr": np.array([1.785, 1.917, 2.056, 2.157, 2.196]),
    "rotor_eta": np.array([0.842, 0.862, 0.876, 0.867, 0.852]),
}

# --- Table V(c): measured rotor blade-element data, reading 4182 (100%
# speed, the peak-eta point; transcribed 2026-07-16 from a page render).
# RP 1..9 = 5..95 percent span FROM TIP (no 0/100 rows in Table V). The
# measured DEV column is the direct calibration target for the Lieblein
# deviation on MCA transonic sections (docs/references/ROTOR37.md).
MEASURED_BE_4182 = {
    "pct_span_from_tip": np.array([5., 10., 15., 30., 50., 70., 85., 90., 95.]),
    "beta1_rel": np.array([69.1, 65.4, 64.5, 62.2, 59.8, 58.1, 57.2, 57.0,
                           57.3]),
    "beta2_rel": np.array([59.4, 56.9, 55.5, 51.3, 46.3, 39.5, 36.4, 34.0,
                           30.7]),
    "incidence_mean": np.array([7.3, 4.5, 4.4, 3.7, 3.2, 3.8, 4.5, 4.5, 5.0]),
    "deviation": np.array([10.1, 8.5, 7.9, 7.0, 7.4, 7.2, 11.0, 11.2, 10.7]),
    "rel_mach_in": np.array([1.448, 1.487, 1.477, 1.454, 1.403, 1.331,
                             1.260, 1.234, 1.197]),
    "eff": np.array([0.793, 0.773, 0.831, 0.856, 0.881, 0.902, 0.938,
                     0.931, 0.936]),
    "loss_tot": np.array([0.199, 0.213, 0.156, 0.137, 0.121, 0.108, 0.071,
                          0.082, 0.082]),
    "loss_prof": np.array([0.067, 0.087, 0.035, 0.032, 0.030, 0.028, 0.001,
                           0.019, 0.026]),
}


def _resample_hub_to_tip(values, n=21):
    """Table order is tip->hub at non-uniform percent span; the section 4.1
    contract wants uniform span nodes from wall_0 (hub) to wall_1 (tip).
    PCHIP through the report nodes, sampled uniformly (C1 preserved)."""
    y_nodes = (1.0 - _PCT_SPAN / 100.0)[::-1]   # hub-first span fractions
    p = PchipInterpolator(y_nodes, np.asarray(values)[::-1])
    return p(np.linspace(0.0, 1.0, n))


@dataclass(frozen=True)
class Rotor37:
    """Geometry-faithful NASA Rotor 37 (section 9.5, point-by-point V5)."""

    mdot: float = 20.74          # kg/s — the measured peak-eta reading
    rpm: float = 17188.7
    # Rotor 37's MCA transonic sections are exactly the AGARD-R-745 blade
    # family; the Cetin Eq 3.5 design-deviation correction is ON by default
    # (validated per-span vs MEASURED_BE_4182: RMS 3.8 -> 1.2 deg; see
    # cetin_deviation_correction and docs/references/AGARD745.md). Set
    # "none" to reproduce the uncorrected 2026-07-15 record.
    transonic_correction: str = "cetin_agard745"
    T0_in: float = 288.15        # K   (report standard-day: 288.2)
    p0_in: float = 101325.0      # Pa  (report: 10.13 N/cm^2)
    tip_clearance_m: float = 4.0e-4   # [VERIFY] AGARD-AR-355 blind-test value
    n_span_nodes: int = 21
    gas: PerfectGas = field(default_factory=PerfectGas)

    @property
    def omega(self) -> float:
        return self.rpm * 2.0 * np.pi / 60.0

    # ------------------------------------------------------------------
    def _walls(self):
        """Hub/casing polylines [m]: constant-radius duct extensions +
        linear taper across the swept rotor (see module docstring)."""
        zi_hub, zo_hub = _ZI_CM[-1] / 100.0, _ZO_CM[-1] / 100.0
        zi_tip, zo_tip = _ZI_CM[0] / 100.0, _ZO_CM[0] / 100.0
        r_hub_le, r_hub_te = _RI_CM[-1] / 100.0, _RO_CM[-1] / 100.0
        r_tip_le, r_tip_te = _RI_CM[0] / 100.0, _RO_CM[0] / 100.0
        z_in, z_out = -0.06, 0.10
        hub = [(z_in, r_hub_le), (zi_hub, r_hub_le),
               (zo_hub, r_hub_te), (z_out, r_hub_te)]
        tip = [(z_in, r_tip_le), (zi_tip, r_tip_le),
               (zo_tip, r_tip_te), (z_out, r_tip_te)]
        return hub, tip

    @staticmethod
    def _frac(poly, z):
        """Arc-length fraction of axial position ``z`` along a wall
        polyline (dense resampling of the same points WallCurve gets)."""
        pts = np.asarray(poly)
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        # locate z on the polyline (z is monotone here)
        zs = pts[:, 0]
        k = int(np.searchsorted(zs, z) - 1)
        k = min(max(k, 0), len(seg) - 1)
        t = (z - zs[k]) / (zs[k + 1] - zs[k])
        return float((cum[k] + t * seg[k]) / cum[-1])

    def _flowpath(self) -> FlowPath:
        hub, tip = self._walls()
        w0 = WallCurve.from_points(np.asarray(hub))
        w1 = WallCurve.from_points(np.asarray(tip))
        f_le0 = self._frac(hub, _ZI_CM[-1] / 100.0)
        f_le1 = self._frac(tip, _ZI_CM[0] / 100.0)
        f_te0 = self._frac(hub, _ZO_CM[-1] / 100.0)
        f_te1 = self._frac(tip, _ZO_CM[0] / 100.0)
        stations = [
            StationDef(StationType.DUCT, 0.0, 0.0),
            StationDef(StationType.EDGE_LE, f_le0, f_le1, row_id="r37"),
            StationDef(StationType.EDGE_TE, f_te0, f_te1, row_id="r37"),
            StationDef(StationType.DUCT, 1.0, 1.0),
        ]
        return FlowPath(w0, w1, stations)

    def _geometry(self) -> ParamRowGeometry:
        n = self.n_span_nodes
        return ParamRowGeometry(
            blade_count=36,
            beta1=-_resample_hub_to_tip(_KIC_DEG, n) * _DEG,
            beta2=-_resample_hub_to_tip(_KOC_DEG, n) * _DEG,
            chord_len=_resample_hub_to_tip(_CHORD_CM, n) / 100.0,
            solidity_val=_resample_hub_to_tip(_SOLIDITY, n),
            thickness=_resample_hub_to_tip(_TM_CM / _CHORD_CM, n),
            stagger_val=-_resample_hub_to_tip(_SETTING_DEG, n) * _DEG,
            clearance=self.tip_clearance_m)

    def machine(self) -> Machine:
        cp = self.gas.gamma * self.gas.R / (self.gas.gamma - 1.0)
        # PerfectGas reference state is (288.15 K, 101325 Pa) with s=0 —
        # the report's standard-day inlet to within 0.02%.
        h0 = cp * self.T0_in
        swirl = LieblienSwirl(transonic_correction=self.transonic_correction)
        row = RowSpec(row_id="r37", omega=self.omega,
                      swirl=swirl, loss=LIEBLEIN_NACA65.loss,
                      blade_count=36, geometry=self._geometry())
        return Machine(self._flowpath(), self.gas,
                       InletCondition(h0=h0, s=0.0, rvt=0.0), rows=[row])

    def evaluate(self, n_sl: int = 1, fidelity: FidelityConfig = None,
                 mdot: float = None) -> PerformanceResult:
        """Solve one operating point (default: Tier-1 meanline at the
        measured peak-efficiency flow)."""
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        target = self.mdot if mdot is None else mdot
        return self.machine().evaluate(MassFlowSpec(target), fidelity,
                                       n_sl=n_sl)
