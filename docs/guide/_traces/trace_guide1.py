"""Regenerate the numbers cited in Guide 1 (Life of a Solve).

Source of truth for Guide 1's snapshot values. Run it, and paste the labeled
block into Guide 1's snapshot table / §6 / §7 / §8, then bump the commit
stamp. See docs/guide/_traces/README.md for the convention.

    <project-venv>/python docs/guide/_traces/trace_guide1.py

Feeds: §1 run table, §D.4 Vm(q0), §6 convergence tables + Euler check,
       §7 reduction, §8 validity-saturation (run C) + CHOKE_LIMITED.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")  # perfect-gas domain edge warnings are expected

from slcflow.machine import FidelityConfig, MassFlowSpec
from slcflow.verification.v5_axial_compressor import (V5AxialRotor,
                                                      V5MultistageCompressor)


def _iters(perf):
    return perf.result.record.iterations


def _row(perf, keep):
    recs = _iters(perf)
    n = len(recs)
    for k in sorted({0, 1, n // 2, n - 1} & set(range(n))) if keep is None else keep:
        if 0 <= k < n:
            it = recs[k]
            print(f"    it{it.iteration:<3d}: cont={it.cont_norm:.1e} "
                  f"pos={it.pos_norm:.1e} clo={it.closure_norm:.1e} "
                  f"omega_sl={it.omega_sl:.4f}")


def main():
    print("# Guide 1 snapshot — run at the current checkout\n")

    # --- Run A: meanline rotor -------------------------------------------
    a = V5AxialRotor().evaluate(n_sl=1)
    fz = a.result.frozen
    print(f"RUN A (n_sl=1, tier1): {a.status.name}  iters={len(_iters(a))}  "
          f"PR={a.pressure_ratio:.4f}  eta={a.efficiency:.4f}  "
          f"validity={a.validity:.4f}")
    _row(a, None)
    print(f"    Vm(q0)={np.round(a.result.x[:fz.n_qo], 2)}  "
          f"rvt_exit={fz.transported.rvt[0, -1]:.2f}  "
          f"h0_exit={fz.transported.h0[0, -1]:.1f}  "
          f"s_exit={fz.transported.s[0, -1]:.4f}")
    r, vm = float(a.r[0]), float(a.vm[0])
    vt = fz.transported.rvt[0, -1] / r
    U = 400.0 * r
    print(f"    Euler: 400*{fz.transported.rvt[0, -1]:.2f}="
          f"{400 * fz.transported.rvt[0, -1]:.0f} J/kg vs dh0="
          f"{fz.transported.h0[0, -1] - 3.0e5:.0f}")
    print(f"    exit angle: vt={vt:.1f} vm={vm:.1f} alpha={np.degrees(np.arctan2(vt, vm)):.1f} "
          f"beta2={np.degrees(np.arctan2(vt - U, vm)):.1f} (metal -45)")

    # --- Run B / B': multistage, mixing on vs off ------------------------
    b = V5MultistageCompressor().evaluate()                       # tier3+mixing
    b0 = V5MultistageCompressor().evaluate(fidelity=FidelityConfig.tier3())
    for lbl, p in (("RUN B  mixing ON ", b), ("RUN B' mixing OFF", b0)):
        s = p.result.frozen.transported.s[:, -1]
        print(f"\n{lbl} (n_sl=9, tier3): {p.status.name}  iters={len(_iters(p))}"
              f"  PR={p.pressure_ratio:.4f}  eta={p.efficiency:.4f}  "
              f"s_spread={float(s.max() - s.min()):.3f}")
        _row(p, None)

    # --- Run C: same rotor spanwise (mid-span-only angles) ---------------
    c = V5AxialRotor().evaluate(n_sl=9, fidelity=FidelityConfig.tier2())
    print(f"\nRUN C (n_sl=9, tier2): {c.status.name}  iters={len(_iters(c))}  "
          f"PR={c.pressure_ratio:.4f}  validity={c.validity:.4f}"
          f"{'  reason=' + c.result.record.reason if c.result.record.reason else ''}")

    # --- A genuine CHOKE_LIMITED via over-throttle -----------------------
    print("\nCHOKE sweep (meanline rotor, rising mdot):")
    for md in (150.0, 180.0):
        p = V5AxialRotor(mdot=md).evaluate(n_sl=1)
        print(f"    mdot={md:.0f} -> {p.status.name}"
              f"{'  ' + p.result.record.reason if p.result.record.reason else ''}")


if __name__ == "__main__":
    main()
