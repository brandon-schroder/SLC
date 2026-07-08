#!/usr/bin/env python3
"""Wilkinson stability-envelope calibration study (Theory Manual section 6.4;
section 9.2 "calibrate the section 6.4 constant here"; M3-3, extended at the
2026-07 recalibration).

Two case FAMILIES, each swept over station density x fixed relaxation
factor; the divergence threshold omega* per grid feeds the ``_omega_sl``
model in ``drivers/classical.py`` and Appendix C.3:

* ``duct`` — the original M3-3 study: V2 curved annulus (Tier 3, duct-only,
  kappa_relax = 0.3), spanwise x station density grids.
* ``bladerow`` — the 2026-07 extension: the V8 parametric blade-row bend
  (centrifugal set, Tier 3) at exit angles 55 deg (mixed-flow) and 90 deg
  (the V7 radial geometry — same walls, same bend radii), in-blade
  resolution varied through ``n_inblade``. Run AFTER the Tier-3
  stabilization: before it, blade-row divergences were dominated by the
  driver artifact family (stale-split deaths, spurious branches), not the
  genuine section 6.4 mode, so pre-fix blade-row anecdotes are not
  calibration data.

Protocols differ deliberately. The duct family pins FIXED omega
(``wilkinson_c`` huge, so the adaptive formula saturates at the
``omega_sl_max`` cap) — clean, because ducts have no closure switch-on
transient. Blade-row cases DO: a fixed omega dies in the ramp-in transient
at values the asymptotic mode tolerates fine (measured: fixed 0.20 fails
where the adaptive formula runs 0.18 steady and converges), because the
adaptive ``(1 - Mm^2)`` factor backs off exactly when the transient spikes
the Mach number. So the blade-row family sweeps ``wilkinson_c`` itself —
the shipped object, transient handling included — and reports c* = the
largest converged constant, directly comparable to the duct threshold
(~7.3) and the shipped default (4.4). Note ``n_inblade`` barely moves the
formula's density ``x = dm_min/L_qo`` here (the duct-adjacent station gaps
pin dm_min), so blade-row points probe the threshold CONSTANT, not the
exponent; the exponent remains duct-calibrated.

Rerun after any change to the repositioning or curvature-lag machinery:

    python tools/calibrate_wilkinson.py [duct|bladerow|all]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slcflow.drivers import ClassicalConfig  # noqa: E402
from slcflow.grid import (GridTopology, evaluate_metrics,  # noqa: E402
                          initialize_positions)
from slcflow.types import FidelityConfig, MassFlowSpec  # noqa: E402
from slcflow.verification import V2CurvedAnnulus  # noqa: E402
from slcflow.verification.v8_mixed_flow import V8MixedFlow  # noqa: E402

R_INNER = 0.2
ARC = 0.5 * np.pi * R_INNER          # inner-bend arc length
SPAN = 0.3

OMEGAS = (0.05, 0.07, 0.10, 0.14, 0.20, 0.28, 0.40, 0.56, 0.70)
GRIDS = [(9, 5), (9, 7), (9, 10), (9, 13), (9, 19), (17, 7), (17, 13),
         (5, 13)]

# Blade-row family: (phi_max_deg, n_inblade); n_sl fixed at the V8/V7
# representative 7. The reference density x = dm_min / L_qo is computed
# from the initialized grid metrics EXACTLY as drivers._omega_sl does.
BLADEROW_POINTS = [(55.0, 2), (55.0, 6), (55.0, 12),
                   (90.0, 2), (90.0, 6), (90.0, 12)]
BLADEROW_CS = (4.4, 6.6, 8.8, 13.2, 17.6, 22.0, 30.0)
_N_SL_BR = 7
_TAG = {"converged": "o", "numerical_failure": "x", "choke_limited": "x",
        "max_iter": "m"}


def solve_at(n_sl, n_st, omega):
    # wilkinson_c huge so the adaptive formula always saturates at the
    # omega_sl_max cap: the sweep tests FIXED relaxation factors.
    case = V2CurvedAnnulus()
    cfg = ClassicalConfig(omega_sl_max=omega, wilkinson_c=1000.0,
                          kappa_relax=0.3, max_outer=150, tol_pos=1e-8)
    t0 = time.time()
    res = case.solve(n_sl=n_sl, n_stations=n_st, config=cfg)
    return res, time.time() - t0


def bladerow_solve_at(phi_deg, n_inblade, c):
    case = V8MixedFlow(phi_max_deg=phi_deg, n_inblade=n_inblade)
    cfg = ClassicalConfig(wilkinson_c=c, kappa_relax=0.3, max_outer=500,
                          tol_pos=1e-8)
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier3(), n_sl=_N_SL_BR,
                                config=cfg)
    return r.result


def bladerow_density_x(phi_deg, n_inblade):
    """dm_min / L_qo on the initialized grid, matching drivers._omega_sl."""
    case = V8MixedFlow(phi_max_deg=phi_deg, n_inblade=n_inblade)
    topo = GridTopology(case._flowpath(), n_sl=_N_SL_BR)
    mtr = evaluate_metrics(topo, initialize_positions(topo))
    return float(np.min(np.diff(mtr.m, axis=1)) / np.max(mtr.qo_length))


def _fit(tag, pts):
    """Log-log fit omega* = K x^p over (x, omega*) points."""
    if len(pts) < 2:
        print(f"  [{tag}] too few converged points to fit")
        return
    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    p, logk = np.polyfit(lx, ly, 1)
    print(f"  [{tag}] p = {p:.2f}, K = {np.exp(logk):.1f}   "
          f"(and K at fixed p = 1.5: "
          f"{np.exp(np.mean(ly - 1.5 * lx)):.1f})")


def run_duct():
    print(f"{'grid':>9} {'dm_min':>7} {'dm/span':>8} | threshold scan "
          "(o = converged, x = failed, m = max_iter)")
    rows = []
    for n_sl, n_st in GRIDS:
        dm = ARC / (n_st - 1)
        marks, omega_star = [], None
        for om in OMEGAS:
            res, dt = solve_at(n_sl, n_st, om)
            tag = _TAG.get(res.status.value, "?")
            marks.append(f"{om:.2f}{tag}")
            if tag == "o":
                omega_star = om          # highest converged so far
        rows.append((n_sl, n_st, dm, omega_star))
        print(f"({n_sl:2d},{n_st:2d}) {dm:7.4f} {dm / SPAN:8.4f} | "
              + " ".join(marks) + f"  -> omega* ~ {omega_star}")

    print("\nduct model fit omega* = K (dm/span)^p:")
    _fit("duct", [(dm / SPAN, om) for _, _, dm, om in rows if om is not None])


def run_bladerow():
    print(f"{'family point':>14} {'x=dm/L':>8} | wilkinson_c scan "
          "(o = converged, x = failed/choked, m = max_iter)")
    c_stars = []
    for phi, nib in BLADEROW_POINTS:
        x = bladerow_density_x(phi, nib)
        marks, c_star, n_iters = [], None, None
        for c in BLADEROW_CS:
            res = bladerow_solve_at(phi, nib, c)
            tag = _TAG.get(res.status.value, "?")
            marks.append(f"{c:.1f}{tag}")
            if tag == "o":
                c_star = c
                n_iters = res.record.n_iterations
        if c_star is not None:
            c_stars.append(c_star)
        print(f"(phi={phi:3.0f},ib={nib:2d}) {x:8.4f} | " + " ".join(marks)
              + f"  -> c* ~ {c_star} ({n_iters} iters at c*)")

    if c_stars:
        print(f"\nbladerow family threshold constant: c* in "
              f"[{min(c_stars):.1f}, {max(c_stars):.1f}] across points; "
              "duct family threshold ~7.3, shipped default 4.4 (0.6x of the "
              "BINDING family's threshold)")


def main():
    np.seterr(all="ignore")
    which = sys.argv[1] if len(sys.argv) > 1 else "duct"
    if which in ("duct", "all"):
        run_duct()
    if which in ("bladerow", "all"):
        run_bladerow()


if __name__ == "__main__":
    main()
