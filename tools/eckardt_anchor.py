"""Eckardt rotor O first-order design-point ANCHOR (docs/references/ECKARDT.md).

A rerunnable study, NOT a rigorous validation and NOT a CI test. It checks that
slcflow's Euler-work + Wiesner-slip + (corrected 2026-07-12) blade-loading loss
reproduce the *grounded* Eckardt O design-point pressure ratios to first order,
using a concentric-bend meridional APPROXIMATION of the real impeller.

Grounded from the library (Cumpsty 1989; see docs/references/ECKARDT.md):
  * D2 = 400 mm (r2 = 0.2 m), RADIAL exit (beta2b = 0), inducer tip ~ 0.7 D2,
    vaneless diffuser, ambient inlet (~288 K).
  * design:      18000 rpm, mdot 7.2 kg/s, measured stage PR 3.0.
  * laser point: 14000 rpm (78%), mdot 5.31 kg/s, measured PR 2.1.
ASSUMED / not confirmed from the library: Z = 20, r1_hub, b2, blade angles, and
the meridional profile (approximated by a concentric quarter-arc bend). The
~5-7% PR under-prediction below is the size of that approximation error.

Run:  .venv/Scripts/python.exe tools/eckardt_anchor.py
"""
import numpy as np

from slcflow.fluid.perfectgas import PerfectGas
from slcflow.verification.v7_centrifugal import V7Centrifugal

_PI = np.pi
_GAS = PerfectGas()
_H0_IN = _GAS.cp * 288.0                      # ambient inlet ~ 288 K


def eckardt_O(rpm, mdot, Z=20):
    """A V7-class radial impeller sized to Eckardt O's grounded numbers."""
    omega = 2 * _PI * rpm / 60.0
    # r2 = 0.2; inducer tip 0.14 (= 0.7 r2), hub 0.06 -> concentric-bend radii.
    return V7Centrifugal(
        r2=0.20, r_inner=0.06, r_outer=0.14, omega=omega, mdot=mdot,
        h0_in=_H0_IN, s_in=0.0,
        beta1_blade_deg=-60.0, beta2_blade_deg=0.0,   # radial exit (O)
        blade_count=Z, solidity=2.0, chord=0.06,
        n_inblade=6, n_sl_rep=7,
        pr_band=(1.0, 5.0), eta_band=(0.0, 1.0))


def main():
    print("Eckardt O anchor (grounded targets in brackets; Tier-1 meanline):")
    for rpm, mdot, pr_meas in [(14000, 5.31, 2.1), (18000, 7.2, 3.0)]:
        r = eckardt_O(rpm, mdot).evaluate(n_sl=1)
        u2 = (2 * _PI * rpm / 60.0) * 0.20
        print(f"  {rpm} rpm, mdot {mdot:>4}: U2={u2:.0f} m/s -> "
              f"slcflow PR={r.pressure_ratio:.3f} eta={r.efficiency:.3f} "
              f"(converged={r.converged})   [measured PR {pr_meas}]")
    print("\nZ sensitivity of PR at 18000 rpm (mdot 7.2):")
    for Z in (16, 18, 20, 24, 30):
        r = eckardt_O(18000, 7.2, Z=Z).evaluate(n_sl=1)
        print(f"  Z={Z:>2}: sigma={1 - 1 / Z ** 0.7:.3f} "
              f"PR={r.pressure_ratio:.3f} eta={r.efficiency:.3f}")


if __name__ == "__main__":
    main()
