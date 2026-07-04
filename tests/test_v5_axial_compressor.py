"""V5 — Axial-compressor entry point (Theory Manual section 9.5; ARCH-5.5,
ARCH-8 M4). Structural half only.

These bind the *structural* V5 gate: an axial-compressor rotor composed
through the Machine facade and run at Tier-1 meanline (section 9.5's
"rotor-67 meanline-level checks") converges, does real work with real loss,
and lands total-to-total PR and efficiency in physically sane bands. The
quantitative half — matching a specific NASA case point-by-point and
generating speedlines through choke — is **[VERIFY]**, blocked on the
reference-library correlation calibration and the M5 continuation driver
(documented in ``verification/v5_axial_compressor.py``). The bands here are
plausibility gates, not Appendix-C tolerances.

Provenance: M4 sub-step 5, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.machine import FidelityConfig, PerformanceResult
from slcflow.verification.v5_axial_compressor import V5AxialRotor

DEG = 180.0 / np.pi


# --------------------------------------------------------------------------
# Section 9.5: Tier-1 meanline rotor through the facade
# --------------------------------------------------------------------------
def test_meanline_converges_and_is_physical():
    case = V5AxialRotor()
    pr = case.evaluate(n_sl=1)          # Tier-1 meanline (section 8)
    assert isinstance(pr, PerformanceResult)
    assert pr.converged
    # Single mid-psi streamline: length-1 exit profiles (the meanline).
    assert pr.vm.shape == (1,)

    # Real work and real loss across the rotor (sections 3.3, 3.5).
    tr = pr.result.frozen.transported
    assert tr.h0[0, 2] - tr.h0[0, 1] > 0.0      # compressor raises h0
    assert tr.s[0, 2] - tr.s[0, 1] > 0.0        # entropy rises (loss)


def test_meanline_pressure_ratio_and_efficiency_bands():
    # Structural plausibility gate for a subsonic stage; NOT a V5 tolerance
    # (point-by-point NASA reproduction is [VERIFY]).
    case = V5AxialRotor()
    pr = case.evaluate(n_sl=1)
    lo, hi = case.pr_band
    assert lo < pr.pressure_ratio < hi
    lo, hi = case.eta_band
    assert lo < pr.efficiency < hi
    assert pr.validity > 0.5


def test_meanline_adds_swirl_and_stays_subsonic():
    # Axial inflow (alpha_in = 0) -> rotor adds absolute swirl, so the exit
    # absolute flow angle is positively deflected (section 2.4). Meridional
    # Mach stays subsonic on the converged field (section 6.5 branch).
    case = V5AxialRotor()
    pr = case.evaluate(n_sl=1)
    assert pr.alpha[0] > 5.0 / DEG               # meaningful exit swirl
    mm = pr.result.fields.mach_m
    assert np.all(mm < 1.0)


def test_facade_is_deterministic_replay():
    # AD-3: identical inputs -> identical outputs (the facade adds no hidden
    # state over the pure residual path).
    case = V5AxialRotor()
    a = case.evaluate(n_sl=1)
    b = case.evaluate(n_sl=1)
    assert a.pressure_ratio == b.pressure_ratio
    assert a.efficiency == b.efficiency
    np.testing.assert_array_equal(a.result.x, b.result.x)


def test_tier1_is_pure_data_switch_not_a_code_path():
    # AD-1: the meanline is n_sl = 1 through the SAME facade/kernel; the
    # FidelityConfig.tier1 flag set is the Tier-2 set (types.py), and the
    # degeneration is purely the single mid-psi streamline (section 8).
    assert FidelityConfig.tier1() == FidelityConfig.tier2()
    case = V5AxialRotor()
    pr = case.evaluate(n_sl=1, fidelity=FidelityConfig.tier1())
    assert pr.converged
