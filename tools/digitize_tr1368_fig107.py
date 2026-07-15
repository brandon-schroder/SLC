#!/usr/bin/env python3
"""Digitized NACA TR-1368 (RM L51G31) Fig. 107 -> off-design deviation-slope
validation for lieblein.py.

Provenance
----------
NACA RM L51G31 (Herrig, Emery & Erwin, 1951; superseded by TR-1368) is the
PRIMARY 65-series cascade dataset the SP-36/Aungier correlations in
``slcflow/closures/axial_compressor/lieblein.py`` were ultimately fit to.
``tools/digitize_sp36.py`` validated the reference-point fits against the
derived SP-36 design charts; this tool goes one level deeper and checks the
**off-design slope** against the measured raw-data summary:

    Fig. 107 (p. 214) -- "Variation of turning-angle, angle-of-attack slope
    at design with solidity and inlet angle. The slopes are averages for the
    moderate camber range."  Three curves: beta1 = 30/45 (one faired curve),
    60, 70 deg.

With fixed blade geometry the predicted design-point turning slope is
``(d theta / d alpha)_d = 1 - (d delta / d i)`` and ``d delta / d i`` is
exactly ``lieblein.deviation_slope`` (section 4.3) -- so the chart validates
that fit directly, at the raw-data level, with no camber-equivalence step.

Digitized 2026-07-15 from the NTRS scan (19930092353, public domain, not
vendored -- 116 MB), page 216 of the PDF rendered at 300 dpi. Calibration was
anchored to the printed uniform grid (x: 148 px per 0.2 solidity from the
sigma=0 axis at px 671; y: 147 px per 0.1 slope from the 1.2 line at px 576 --
the digitize_sp36 lesson: anchor to the tick grid, never ink-weight frame
detection). Curves were traced by seeded bidirectional nearest-continuation
tracking of small ink runs (gridline hits and label glyphs rejected), and the
DECISIVE check was overlaying the traced points on the chart image: all three
traces ride the printed curves over their full span. Reading precision
~ +/-0.01 in slope (2-3 px line width on a 14.7 px per 0.01 grid).

Findings (2026-07-15, code at the reference-calibration-complete state):
  * Overall: n = 39 (beta1, sigma) points, RMS error 0.030, mean +0.002 --
    the Aungier fit reproduces the measured slope structure (rising with
    sigma, decreasing with beta1 at low sigma).
  * beta1 = 30: |err| <= 0.013 everywhere. The chart fairs 30 and 45 into a
    single curve; the code's 30-vs-45 spread (~0.03) is the same order as
    that fairing width, with the 45-deg branch reading ~0.02-0.03 low.
  * beta1 = 60: +0.04..+0.06 at sigma <= 0.8, tapering through zero to
    -0.025 at sigma = 1.5.
  * beta1 = 70 (documented deviation region): the fit's sigma-dependence is
    shallower than measured -- +0.097 at sigma = 1.0 tapering to -0.039 at
    sigma = 1.5. That corner is the edge of the calibrated domain
    (CALIBRATED beta1 hi = 70) and the most extreme cascade condition in the
    dataset; the chart is itself "averages for the moderate camber range",
    and the NACA design point (smooth pressure distribution) is not
    identically SP-36's minimum-loss reference that Aungier fit. Recorded
    honestly rather than tuned away; see docs/references/TR1368.md.

Run:  python tools/digitize_tr1368_fig107.py
"""
from __future__ import annotations

import numpy as np

from slcflow.closures.axial_compressor.lieblein import deviation_slope

# Fig. 107 traced samples: {beta1 (deg) or (30, 45): {sigma: (dtheta/dalpha)_d}}
FIG107 = {
    (30, 45): {0.5: 0.748, 0.6: 0.812, 0.7: 0.855, 0.8: 0.890, 0.9: 0.923,
               1.0: 0.945, 1.1: 0.959, 1.2: 0.972, 1.3: 0.980, 1.4: 0.983,
               1.5: 0.979},
    60:       {0.5: 0.594, 0.6: 0.651, 0.7: 0.726, 0.8: 0.773, 0.9: 0.830,
               1.0: 0.871, 1.1: 0.896, 1.2: 0.924, 1.3: 0.947, 1.4: 0.965,
               1.5: 0.979},
    70:       {1.0: 0.751, 1.1: 0.807, 1.2: 0.852, 1.3: 0.907, 1.4: 0.943,
               1.5: 0.979},
}


def predicted(beta1_deg: float, sigma: float) -> float:
    """Code-side design-point turning slope (section 4.3)."""
    return 1.0 - float(deviation_slope(beta1_deg, sigma))


def main() -> None:
    errs = []
    for key, pts in FIG107.items():
        betas = key if isinstance(key, tuple) else (key,)
        for s, meas in sorted(pts.items()):
            for b in betas:
                e = predicted(float(b), float(s)) - meas
                errs.append(e)
                print(f"beta1={b:2d} sigma={s:.1f}  chart={meas:.3f}  "
                      f"code={predicted(float(b), float(s)):.3f}  err={e:+.3f}")
    e = np.asarray(errs)
    print(f"\nn={len(e)}  mean={e.mean():+.4f}  "
          f"rms={np.sqrt((e * e).mean()):.4f}  max|e|={np.abs(e).max():.4f}")


if __name__ == "__main__":
    main()
