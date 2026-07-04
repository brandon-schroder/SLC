#!/usr/bin/env python3
"""Wilkinson stability-envelope calibration study (Theory Manual section 6.4
[VERIFY]; section 9.2 "calibrate the section 6.4 constant here"; M3-3).

Sweeps the V2 curved-annulus case (Tier 3, kappa_relax = 0.3) over station
density x spanwise density x fixed relaxation factor, and reports the
divergence threshold omega* per grid. The fixed omega is forced by setting
``wilkinson_c`` large so the adaptive formula always hits the
``omega_sl_max`` cap.

Findings feed the ``_omega_sl`` model in ``drivers/classical.py`` and
Appendix C.3 of the theory manual. Rerun after any change to the
repositioning or curvature-lag machinery:

    python tools/calibrate_wilkinson.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slcflow.drivers import ClassicalConfig  # noqa: E402
from slcflow.verification import V2CurvedAnnulus  # noqa: E402

R_INNER = 0.2
ARC = 0.5 * np.pi * R_INNER          # inner-bend arc length
SPAN = 0.3

OMEGAS = (0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40, 0.56, 0.70)
GRIDS = [(9, 5), (9, 7), (9, 10), (9, 13), (9, 19), (17, 7), (17, 13),
         (5, 13)]


def solve_at(n_sl, n_st, omega):
    # wilkinson_c huge so the adaptive formula always saturates at the
    # omega_sl_max cap: the sweep tests FIXED relaxation factors.
    case = V2CurvedAnnulus()
    cfg = ClassicalConfig(omega_sl_max=omega, wilkinson_c=1000.0,
                          kappa_relax=0.3, max_outer=150, tol_pos=1e-8)
    t0 = time.time()
    res = case.solve(n_sl=n_sl, n_stations=n_st, config=cfg)
    return res, time.time() - t0


def main():
    np.seterr(all="ignore")
    print(f"{'grid':>9} {'dm_min':>7} {'dm/span':>8} | threshold scan "
          "(o = converged, x = failed, m = max_iter)")
    rows = []
    for n_sl, n_st in GRIDS:
        dm = ARC / (n_st - 1)
        marks, omega_star = [], None
        for om in OMEGAS:
            res, dt = solve_at(n_sl, n_st, om)
            tag = {"converged": "o", "numerical_failure": "x",
                   "max_iter": "m"}.get(res.status.value, "?")
            marks.append(f"{om:.2f}{tag}")
            if tag == "o":
                omega_star = om          # highest converged so far
        rows.append((n_sl, n_st, dm, omega_star))
        print(f"({n_sl:2d},{n_st:2d}) {dm:7.4f} {dm / SPAN:8.4f} | "
              + " ".join(marks) + f"  -> omega* ~ {omega_star}")

    print("\nmodel fit omega* = K (dm/span)^p:")
    pts = [(dm / SPAN, om) for _, _, dm, om in rows if om is not None]
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    p, logk = np.polyfit(x, y, 1)
    print(f"  p = {p:.2f}, K = {np.exp(logk):.1f}")
    for (n_sl, n_st, dm, om) in rows:
        pred = np.exp(logk) * (dm / SPAN) ** p
        print(f"  ({n_sl:2d},{n_st:2d}): measured {om}  predicted {pred:.3f}")


if __name__ == "__main__":
    main()
