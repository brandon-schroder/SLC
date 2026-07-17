"""V5 — NASA Rotor 38, geometry-faithful (Theory Manual section 9.5): the
SECOND digitised transonic axial rotor — the high-aspect-ratio sibling of
Rotor 37 from the same four-stage family, testing whether the
Cetin-corrected axial validation GENERALIZES (the axial analogue of the
Krain second-impeller check).

Provenance (transcribed 2026-07-17 from page renders of the primary
report): Moore & Reid, "Performance of Single-Stage Axial-Flow Transonic
Compressor With Rotor and Stator Aspect Ratios of 1.63 and 1.77,
Respectively, and With Design Pressure Ratio of 2.05", NASA TP-2001
(1982). NTRS 19820014395 (public domain); in the Test Cases notebook.

  * Table I: design rotor PR 2.105 / eta_ad 0.878 at 20.188 kg/s,
    17 188.7 rpm (same speed/flow/annulus family as Stage 37), tip speed
    455.1 m/s, hub/tip 0.70, **48 rotor blades** (vs 36 — the high-AR
    short-chord design), 62 stator blades.
  * Table III(a): rotor geometry at 11 span elements (below).
  * Table IV(a): measured 100%-speed rotor line, SIX readings
    choke -> stall (:data:`MEASURED_100_R38`). Note the rotor never
    reaches its design PR at 100% speed (max 2.004 near stall, eta
    ~0.848) — the historically documented high-aspect-ratio shortfall
    relative to Stage 37 (peak eta 0.876, PR 2.196 near stall).

Same modelling frame as Rotor 37 (the ``TABLES`` hook); the
generalization question this case answers is whether the model TRACKS the
measured Stage 37 -> 38 differences (blade count/chord/solidity are the
dominant geometry changes on a near-identical annulus).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # verification layer: case definitions  # ad6: allow

from .v5_rotor37 import Rotor37

__all__ = ["Rotor38", "MEASURED_100_R38", "DESIGN_R38"]

# --- NASA TP-2001 Table III(a), rotor 38, tip -> hub ----------------------
_T = {
    "pct": np.array([0., 5., 10., 15., 30., 50., 70., 85., 90., 95., 100.]),
    "ri": np.array([25.283, 24.979, 24.641, 24.297, 23.231, 21.762,
                    20.236, 19.020, 18.593, 18.151, 17.780]),
    "ro": np.array([24.770, 24.459, 24.148, 23.837, 22.904, 21.660,
                    20.416, 19.483, 19.172, 18.861, 18.550]),
    "kic": np.array([62.69, 62.05, 61.34, 60.59, 58.53, 56.51,
                     54.16, 52.74, 52.55, 52.51, 52.52]),
    "koc": np.array([55.39, 53.70, 52.12, 50.83, 47.21, 41.52,
                     34.46, 26.64, 23.76, 20.74, 17.69]),
    "tm": np.array([0.149, 0.157, 0.166, 0.174, 0.202, 0.239,
                    0.278, 0.311, 0.323, 0.336, 0.346]),
    "chord": np.array([4.215, 4.235, 4.232, 4.228, 4.218, 4.208,
                       4.210, 4.228, 4.239, 4.255, 4.253]),
    "sol": np.array([1.287, 1.309, 1.325, 1.342, 1.397, 1.481,
                     1.582, 1.678, 1.715, 1.756, 1.789]),
    "set": np.array([62.82, 61.77, 60.64, 59.51, 56.18, 52.04,
                     47.50, 43.45, 42.09, 40.83, 39.62]),
    "zi": np.array([0.504, 0.479, 0.450, 0.418, 0.311, 0.200,
                    0.109, 0.039, 0.021, 0.009, 0.000]),
    "zo": np.array([2.371, 2.425, 2.477, 2.525, 2.648, 2.797,
                    2.958, 3.083, 3.122, 3.159, 3.194]),
}

DESIGN_R38 = {"mdot": 20.188, "rotor_pr": 2.105, "rotor_eta": 0.878,
              "rpm": 17188.7, "blades": 48}
# Table IV(a): six readings, choke -> near-stall.
MEASURED_100_R38 = {
    "mdot": np.array([20.97, 20.91, 20.91, 20.83, 20.67, 20.44]),
    "rotor_pr": np.array([1.799, 1.846, 1.858, 1.912, 1.969, 2.004]),
    "rotor_eta": np.array([0.842, 0.847, 0.847, 0.848, 0.849, 0.848]),
}


@dataclass(frozen=True)
class Rotor38(Rotor37):
    """Geometry-faithful NASA Rotor 38 (TP-2001; high-AR sibling)."""

    TABLES = _T
    BLADES = 48

    mdot: float = 20.67          # the measured near-peak-eta reading (4120)
