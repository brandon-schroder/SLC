#!/usr/bin/env python3
"""Digitized Lieblein (1959) Fig. 6 -> wake_momentum_thickness validation.

Provenance
----------
``axial_compressor/loss.wake_momentum_thickness`` implements Lieblein's
wake-momentum-thickness correlation

    (theta/c)_2 = e / (1 - k_s * ln(V_max,s / V2))     (Lieblein 1959, Eq. 8)

with ``e = 0.004`` and ``k_s = 1.17``, using the equivalent diffusion ratio
``D_eq`` as the computable estimate of the suction-surface diffusion ratio
``V_max,s / V2``. The two constants were confirmed verbatim at the
coefficient level (docs/references/LIEB59.md, vs Aungier/Cumpsty/Dixon); this
pass validates the fit OUTPUT against Lieblein's own published chart.

Fig. 6 of S. Lieblein, "Loss and Stall Analysis of Compressor Cascades,"
ASME J. Basic Eng. 81 (1959) 387-400 -- "Experimental variation of wake
momentum thickness with suction-surface diffusion ratio at minimum loss" --
plots (theta/c)_2 vs V_max,s/V2 for NACA 65-(A10) and C.4 circular-arc cascade
data, with the dashed curve labelled "EQUATION [8] WITH k_s = 1.17 AND
e = 0.004". Digitized 2026-07-10 from the paper (user's Google Drive,
lieblein_loss_1959.pdf): rendered at 600 dpi, axes calibrated off the tick
labels (DR 1.0->2.4, theta/c 0->.05), the dashed curve read by column scan in
the regions where it is separable from the data markers. Reading precision
~ +/-0.0006 in (theta/c)_2.

Result: CLEAN VALIDATION, no bug (like the SP-36 incidence/deviation pass).
The coded curve lands on Lieblein's dashed EQUATION-[8] line and through the
centre of his data cloud across the whole range. The data span DR ~ 1.15 to
~ 2.25; the code's compact-support calibration window (1.0, 2.0) is sound and
slightly conservative on the upper end. Lieblein states the k_s = 1.17 fit
diverges at the "limit V_max,s/V2 = 2.35" -- exactly the code's denominator
zero e^(1/1.17) = 2.35 (ceiling at 2.2 sits safely at the data edge).

Run:  python tools/digitize_lieblein_loss.py
"""
from __future__ import annotations

import numpy as np

from slcflow.closures.axial_compressor.loss import wake_momentum_thickness

# Digitized Fig. 6 dashed-curve readings: (diffusion ratio, (theta/c)_2).
# Curve-isolated columns only (floor + steep flank); the near-vertical tail
# past DR ~ 2.1 is dropped -- 1 px in DR maps to a large d(theta/c) there, so
# the reading precision collapses (the overlay confirms the tail visually).
FIG6 = [(1.10, 0.0046), (1.15, 0.0049), (1.20, 0.0049), (1.40, 0.0063),
        (1.60, 0.0089), (1.80, 0.0128), (2.05, 0.0247), (2.10, 0.0298)]

# Data-cloud extent read off the chart (for the validity-window check).
DATA_DR_RANGE = (1.15, 2.25)
# Lieblein's stated divergence limit (= e^(1/1.17)); the code ceils below it.
DIVERGENCE_LIMIT = 2.35


def main():
    dr = np.array([p[0] for p in FIG6])
    chart = np.array([p[1] for p in FIG6])
    coded = np.array([float(wake_momentum_thickness(float(x))[0]) for x in dr])
    resid = np.abs(coded - chart)
    print("  DR    chart    coded    |diff|")
    for x, c, k in zip(dr, chart, coded):
        print(f"  {x:.2f}  {c:.4f}  {k:.4f}  {abs(k-c):.4f}")
    print(f"max |coded - chart| = {resid.max():.4f}  (reading precision ~0.0006)")
    print(f"denominator zero at D_eq = e^(1/1.17) = "
          f"{np.exp(1/1.17):.3f}  (Lieblein limit 2.35)")


if __name__ == "__main__":
    main()
