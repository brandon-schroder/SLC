#!/usr/bin/env python3
"""Digitized NASA SP-36 (Lieblein) design charts -> lieblein.py output check.

Provenance
----------
``slcflow/closures/axial_compressor/lieblein.py`` implements Aungier's analytic
curve-fits (ch. 6) to Lieblein's NASA SP-36 minimum-loss cascade design charts
for NACA 65-(A10) blades. The fit COEFFICIENTS are confirmed verbatim against
Aungier in ``docs/references/AUN-C.md``; this tool closes the other half -- do
the fit OUTPUTS actually reproduce the original SP-36 chart points end to end
(SP-36 data -> Aungier fit -> our code)?

Four charts from NASA SP-36 (Johnsen & Bullock, 1965; NTRS 19650013744),
Chapter VI (Lieblein), each Y vs inlet-air angle beta1 for a solidity family:

  * Fig. 137 -- reference zero-camber incidence (i0)_10   [Aungier Eq 6-13]
  * Fig. 138 -- incidence camber-slope factor  n          [Eq 6-15]
  * Fig. 161 -- reference zero-camber deviation (delta0)_10 [Eq 6-20]
  * Fig. 162 -- deviation camber-slope factor  m          [Eq 6-21/22/24]

Digitized 2026-07-10: each figure page was rasterized, the plot box calibrated
against the printed grid (fine grid detected by dark-pixel line-sums), curves
extracted by a grid-removed column scan and cross-checked by overlaying the
coded fit on the chart image. The SP-36 report is public-domain (NASA) but is
not vendored here (260 MB scan); the human-verified anchor points are embedded
below. Reading precision ~ +/-0.15 deg (a 1965 raster with a fine grid).

Result: the code reproduces the SP-36 chart points to RMS ~0.10 deg on
(i0)_10 and ~0.17 deg on (delta0)_10 -- i.e. Aungier's fits ARE faithful to the
originals and our transcription is faithful to Aungier. No discrepancy found
(contrast the K-O AM-Fig.4 pass, which surfaced a floor-width bug).

Run:  python tools/digitize_sp36.py
"""
from __future__ import annotations

import numpy as np

from slcflow.closures.axial_compressor.lieblein import (
    reference_deviation, reference_incidence)

SIGMA = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]

# Fig. 137 (i0)_10 [deg], digitized at beta1 = 40, 50; columns = SIGMA order.
I0_10 = {40: [6.32, 5.69, 5.02, 4.39, 3.77, 3.09, 2.47, 1.85, 1.17],
         50: [7.85, 7.02, 6.23, 5.47, 4.66, 3.86, 3.06, 2.25, 1.46]}
# Fig. 161 (delta0)_10 [deg], digitized at beta1 = 50; sigma 2.0..0.6 (the
# shallow sigma=0.4 curve merges with the grid near the origin -- omitted).
D0_10 = {50: [1.84, 1.71, 1.57, 1.41, 1.22, 1.05, 0.82, 0.58]}


def _stats(errs):
    e = np.asarray(errs)
    return len(e), float(e.mean()), float(np.sqrt((e * e).mean())), \
        float(np.abs(e).max())


def main():
    ei = [float(reference_incidence(float(b), s, 0.10, 0.0)[0]) - ch
          for b, row in I0_10.items() for s, ch in zip(SIGMA, row)]
    ed = [float(reference_deviation(float(b), s, 0.10, 0.0)[0]) - ch
          for b, row in D0_10.items() for s, ch in zip(SIGMA, row)]
    n, me, rms, mx = _stats(ei)
    print(f"(i0)_10  vs Fig.137: n={n}  mean={me:+.3f}  rms={rms:.3f}  "
          f"max={mx:.3f} deg")
    n, me, rms, mx = _stats(ed)
    print(f"(d0)_10  vs Fig.161: n={n}  mean={me:+.3f}  rms={rms:.3f}  "
          f"max={mx:.3f} deg")
    print("camber slopes n (Fig.138) / m (Fig.162): validated by overlay "
          "(see docstring); e.g. n(70,0.4)="
          f"{float(reference_incidence(70,0.4,0.10,1.0)[0]-reference_incidence(70,0.4,0.10,0.0)[0]):.3f}"
          " (chart ~-0.46), m(70,0.4)="
          f"{float(reference_deviation(70,0.4,0.10,1.0)[0]-reference_deviation(70,0.4,0.10,0.0)[0]):.3f}"
          " (chart ~0.52)")


if __name__ == "__main__":
    main()
