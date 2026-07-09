"""Regenerate the numbers cited in Guide 2 (Theory Companion), §8.

The B.2/B.3/B.4 worked conversions use FIXED illustrative coefficients on a
fixed LE state, so they do not move with calibration. Only the "closing the
loop to Guide 1" check reads run A's actual converged entropy, which does
move. See docs/guide/_traces/README.md.

    <project-venv>/python docs/guide/_traces/trace_guide2.py
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

from slcflow.closures import conversions as cv
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.verification.v5_axial_compressor import V5AxialRotor


def main():
    gas = PerfectGas()
    print("# Guide 2 §8 snapshot\n")
    print(f"gas: cp={gas.cp:.1f} gamma={gas.gamma:.3f} R={gas.R:.3f}")

    # --- Fixed illustrative conversions (do NOT move with calibration) ---
    vm, r = 91.16, 0.4743                 # run A LE state (inlet, stable)
    U1 = 400.0 * r
    w1 = np.hypot(vm, U1)
    h1 = 3.0e5 - 0.5 * vm * vm
    T1, p1 = gas.T(h1, 0.0), gas.p(h1, 0.0)
    T0r1, p0r1 = cv.relative_stagnation(gas, T1, p1, w1)
    print(f"\nLE state: T1={T1:.2f} p1={p1:.0f}  W1={w1:.2f}  "
          f"T0r1={T0r1:.2f} p0r1={p0r1:.0f}  head={p0r1 - p1:.0f}")
    ds2, _ = cv.delta_s_compressor_omega_bar(gas, 0.06, T0r1, p0r1, p1, U1, U1)
    print(f"B.2 omega_bar=0.06 (axial): ds={ds2:.4f} J/kgK  (fixed illustration)")
    T2 = 1200.0 - 550.0 ** 2 / (2 * gas.cp)
    p2 = 8.0e5 * (T2 / 1200.0) ** (gas.gamma / (gas.gamma - 1.0))
    dsY, _ = cv.delta_s_turbine_Y(gas, 0.10, 1200.0, 8.0e5, p2, 0.0, 0.0)
    dsZ = cv.delta_s_kinetic_energy_zeta(gas, 0.05, T2, 550.0)
    print(f"B.3 Y=0.10: ds={dsY:.4f}   B.4 zeta=0.05: ds={dsZ:.4f}  (fixed)")

    # --- Closing check: reads run A's actual entropy (MOVES) -------------
    a = V5AxialRotor().evaluate(n_sl=1)
    tr = a.result.frozen.transported
    ds, h0ex = float(tr.s[0, -1]), float(tr.h0[0, -1])
    ideal = (h0ex / 3.0e5) ** (gas.gamma / (gas.gamma - 1.0))
    disc = np.exp(-ds / gas.R)
    print(f"\nCLOSING CHECK (moves with loss calibration):")
    print(f"  run A: ds={ds:.4f}  h0_exit={h0ex:.0f}  PR_reported={a.pressure_ratio:.4f}")
    print(f"  ideal={ideal:.4f}  exp(-ds/R)={disc:.4f}  product={ideal * disc:.4f}")


if __name__ == "__main__":
    main()
