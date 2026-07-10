#!/usr/bin/env python3
"""Digitized Ainley-Mathieson R&M 2974 Fig. 4 -> Kacker-Okapuu yp1/yp2 fit.

Provenance
----------
The two reference profile-loss curves in
``slcflow/closures/axial_turbine/kacker_okapuu.py`` (``profile_loss_am``)
are smooth surrogate fits to Ainley & Mathieson's classic profile-loss chart
(ARC R&M 2974, 1951, Fig. 4; ``t/c = 20%``, ``Re = 2e5``, ``M < 0.6``):

  * Fig. 4a -- NOZZLE blades (beta1 = 0):   Y_p vs pitch/chord, per exit angle
  * Fig. 4b -- IMPULSE blades (beta1 = -alpha2): same

Kacker-Okapuu (1982) use these AM curves unchanged for the AMDC profile-loss
term (their recalibration is the 2/3 bracket, K_p, f_Re, etc. -- all applied
*outside* ``profile_loss_am``), so R&M 2974 Fig. 4 is the correct source.

The chart was digitized (2026-07-10) by rasterizing the figure page and
reading each labelled curve's minimum against the printed grid, cross-checked
against an automated dark-pixel column scan (gridline detection + bottom/top
envelope trace) and the canonical values reproduced in Dixon and
Cohen-Rogers-Saravanamuttoo. The AM51 report is a public-domain ARC document
but is not vendored here (it was deliberately removed, commit 2916c57); the
human-verified anchor points are embedded below so the calibration is
reproducible without redistributing the scan. Reading precision is about
+/- 0.005 in Y_p (a 1951 raster chart) -- honest tolerances follow.

What is calibrated (and what is not)
------------------------------------
* minimum-loss level vs exit angle -> ``A + B*u**4`` (u = alpha2/70);
* loading-optimum pitch/chord vs exit angle -> ``C + D*u`` (linear);
* off-optimum curvature -> a single symmetric-parabola coefficient.

The chart curves are *asymmetric* (steep for s/c < s_opt, flat for s/c >
s_opt) and their curvature grows with exit angle; the symmetric,
angle-independent parabola is a deliberate compromise, accurate near the
loading optimum where rows operate. That residual is documented, not hidden.

Run:  python tools/digitize_am_fig4.py
"""
from __future__ import annotations

import numpy as np

# Digitized minima off R&M 2974 Fig. 4: (alpha2_deg, s/c at min, Y_p,min).
NOZZLE = [(40, 0.86, 0.021), (50, 0.83, 0.023), (60, 0.79, 0.026),
          (65, 0.76, 0.030), (70, 0.72, 0.035), (75, 0.68, 0.042),
          (80, 0.63, 0.049)]
IMPULSE = [(40, 0.75, 0.067), (50, 0.72, 0.075), (55, 0.70, 0.086),
           (60, 0.65, 0.102), (65, 0.62, 0.115), (70, 0.58, 0.135)]


def fit(points):
    """Least-squares fit of the coded surrogate form to the digitized minima.

    Returns ``(A, B, C, D, max_min_err, max_sopt_err)`` for
    ``Y_p,min = A + B*u**4`` and ``s_opt = C + D*u`` (``u = alpha2/70``).
    """
    a2 = np.array([p[0] for p in points], float)
    sopt = np.array([p[1] for p in points], float)
    ymin = np.array([p[2] for p in points], float)
    u = a2 / 70.0
    (A, B), *_ = np.linalg.lstsq(np.vstack([np.ones_like(u), u ** 4]).T,
                                 ymin, rcond=None)
    (C, D), *_ = np.linalg.lstsq(np.vstack([np.ones_like(u), u]).T,
                                 sopt, rcond=None)
    e_min = np.abs(A + B * u ** 4 - ymin).max()
    e_sopt = np.abs(C + D * u - sopt).max()
    return A, B, C, D, e_min, e_sopt


def main():
    for label, pts in [("NOZZLE (yp1)", NOZZLE), ("IMPULSE (yp2)", IMPULSE)]:
        A, B, C, D, em, es = fit(pts)
        print(f"{label}: Y_min = {A:.4f} + {B:.4f}*u**4   "
              f"s_opt = {C:.3f} + {D:.3f}*u")
        print(f"    max|Y_min err| = {em:.4f}  max|s_opt err| = {es:.3f}  "
              f"(chart read ~+/-0.005)")
    print("Coded (kacker_okapuu.profile_loss_am): "
          "yp1 = 0.0178 + 0.0179 u^4 + 0.020((s/c - (1.109-0.397u))/0.35)^2; "
          "yp2 = 0.0572 + 0.0782 u^4 + 0.045((s/c - (1.000-0.408u))/0.35)^2")


if __name__ == "__main__":
    main()
