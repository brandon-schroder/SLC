"""Regenerate the numbers cited in Guide 3 (Numerics & Stability).

Feeds: §4.4 refutation table (V7 edge-only/subdivided, V8 Tier 3, multistage
mixing off), §4.5 Newton quadratic record, §5 mixing snapshot. The Wilkinson
C.3 sweep is NOT here — it is `tools/calibrate_wilkinson.py` (the canonical
generator). See docs/guide/_traces/README.md.

    <project-venv>/python docs/guide/_traces/trace_guide3.py

Note: the V7/V8 Tier-3 runs are slow (V8 ~1-2 min at omega_sl ~ 0.06).
"""
from __future__ import annotations

import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from slcflow.drivers.classical import ClassicalConfig, solve_classical
from slcflow.machine import FidelityConfig, MassFlowSpec
from slcflow.drivers.newton import solve_newton
from slcflow.transport.streamwise import TransportFields
from slcflow.verification.v1_analytic_ree import (V1ForcedVortex,
                                                  annulus_topology)
from slcflow.verification.v5_axial_compressor import V5MultistageCompressor
from slcflow.verification.v7_centrifugal import V7Centrifugal
from slcflow.verification.v8_mixed_flow import V8MixedFlow


def _conv(label, perf_or_res):
    r = getattr(perf_or_res, "result", perf_or_res)
    n = len(r.record.iterations)
    pr = getattr(perf_or_res, "pressure_ratio", float("nan"))
    val = getattr(perf_or_res, "validity", float("nan"))
    print(f"  {label}: {r.status.name} iters={n} PR={pr:.4f} validity={val:.4f}")


def main():
    print("# Guide 3 snapshot\n")

    # --- §4.5 Newton quadratic from a partial warm start (V1c) -----------
    case = V1ForcedVortex()
    topo = annulus_topology(case.r0, case.r1, case.length, 17, case.n_stations)
    inlet = TransportFields(h0=np.full(17, case.h0), s=np.full(17, case.s),
                            rvt=case.inlet_rvt(topo.psi, case.exact()))
    cl = case.solve(17)
    warm = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(case.mdot), inlet,
                           config=ClassicalConfig(max_outer=3))
    nt = solve_newton(topo, case.gas, FidelityConfig.tier2(),
                      MassFlowSpec(case.mdot), inlet, warm_start=warm)
    print(f"§4.5 Newton (V1c, 3-iter warm start): classical={len(cl.record.iterations)} it")
    for it in nt.record.iterations:
        print(f"    rec{it.iteration}: scaled|r|={it.cont_norm:.2e} alpha={it.omega_sl:.2f}")

    # --- §4.4 refutation table -------------------------------------------
    print("\n§4.4 stabilized Tier-3 (slow — please wait):")
    _conv("V7 edge-only T3 (n_inblade=0, n_sl=7)",
          V7Centrifugal(n_inblade=0).evaluate(n_sl=7, fidelity=FidelityConfig.tier3()))
    _conv("V7 subdivided T3 (n_inblade=6, n_sl=7)",
          V7Centrifugal(n_inblade=6).evaluate(n_sl=7, fidelity=FidelityConfig.tier3()))
    v8 = V8MixedFlow()
    t0 = time.perf_counter()
    p8 = v8.machine().evaluate(MassFlowSpec(v8.mdot), FidelityConfig.tier3(),
                               n_sl=v8.n_sl_rep, config=ClassicalConfig(max_outer=600))
    print(f"  V8 mixed-flow T3 (n_sl={v8.n_sl_rep}): {p8.status.name} "
          f"iters={len(p8.result.record.iterations)} PR={p8.pressure_ratio:.4f} "
          f"omega_sl={p8.result.record.iterations[-1].omega_sl:.4f} "
          f"wall={time.perf_counter() - t0:.0f}s")

    # --- §5 multistage mixing on vs off (WATCH validity) -----------------
    print("\n§5 multistage mixing (n_sl=9, tier3):")
    for lbl, fid in (("mixing ON ", None),
                     ("mixing OFF", FidelityConfig.tier3())):
        p = V5MultistageCompressor().evaluate(fidelity=fid)
        s = p.result.frozen.transported.s[:, -1]
        print(f"  {lbl}: {p.result.status.name} iters={len(p.result.record.iterations)}"
              f" PR={p.pressure_ratio:.4f} validity={p.validity:.4f}"
              f" s_spread={float(s.max() - s.min()):.3f}")


if __name__ == "__main__":
    main()
