#!/usr/bin/env python3
"""Digitized Kacker-Okapuu (1982) Fig. 14 -> trailing_edge_zeta base curves.

Provenance
----------
``kacker_okapuu.trailing_edge_zeta`` interpolates the TE kinetic-energy loss
coefficient ``dphi^2_TET`` between two reference curves vs the trailing-edge-
thickness/throat-opening ratio ``t/o`` (KO82 Fig. 14):

  * AXIAL-ENTRY NOZZLE (beta1 = 0)  -- the UPPER curve (higher TE loss)
  * IMPULSE BLADING     (beta1 = alpha2) -- the LOWER curve

and interpolates on KO82's signed angle-ratio weight ``|b1/a2|(b1/a2)`` (Eq. 17).

Digitized 2026-07-10 from the K-O 1982 paper (S.C. Kacker & U. Okapuu, "A Mean
Line Prediction Method for Axial Flow Turbine Efficiency", ASME J. Eng. Power
104, 1982) -- the source PDF from the user's library (kacker_mean_1980.pdf).
Fig. 14 was rasterized, the axes calibrated off the tick marks, and the two
curves extracted by column scan. Reading precision ~ +/-0.005.

This pass FIXED a real bug: the old base curves ``phi2_ax = 0.4 t + 2.0 t^2`` /
``phi2_imp = 0.7 t + 4.0 t^2`` had the nozzle/impulse ordering SWAPPED
(impulse > nozzle) AND were ~3x too high (0.48/0.92 at t/o = 0.4 vs the chart's
0.14/0.074). Refit to ``phi2_ax = 0.148 t + 0.530 t^2`` (nozzle) and
``phi2_imp = 0.078 t + 0.277 t^2`` (impulse), reproducing the chart to <0.008.
The interpolation weight was also symmetric ``r^2``; corrected to the signed
Eq. 17 form (matching profile_loss_am).

Separately CONFIRMED (no change): the K_p Mach factor K1 is printed as an
equation ON Fig. 8 -- ``K1 = 1 - 1.25(M2 - 0.2) for M2 > 0.2`` -- exactly the
coded ramp; it was never a surrogate.

Run:  python tools/digitize_ko82_fig14.py
"""
from __future__ import annotations

import numpy as np

from slcflow.closures.axial_turbine.kacker_okapuu import trailing_edge_zeta

# Digitized Fig. 14: (t/o, nozzle dphi^2, impulse dphi^2).
FIG14 = [(0.10, 0.0143, 0.0073), (0.15, 0.029, 0.015),
         (0.20, 0.0506, 0.0283), (0.25, 0.0741, 0.0388),
         (0.30, 0.100, 0.051), (0.35, 0.115, 0.0606),
         (0.40, 0.140, 0.0739)]


def main():
    t = np.array([p[0] for p in FIG14])
    for j, name in ((1, "NOZZLE (b1=0) "), (2, "IMPULSE(b1=a2)")):
        y = np.array([p[j] for p in FIG14])
        (a, b), *_ = np.linalg.lstsq(np.vstack([t, t * t]).T, y, rcond=None)
        # code output: nozzle via alpha1=0, impulse via alpha1=alpha2
        a1 = 0.0 if j == 1 else 60.0
        code = np.array([float(trailing_edge_zeta(a1, 60.0, float(tt))[0])
                         for tt in t])
        print(f"{name}: fit dphi^2 = {a:.3f} t + {b:.3f} t^2   "
              f"max|code-chart| = {np.abs(code - y).max():.4f}")


if __name__ == "__main__":
    main()
