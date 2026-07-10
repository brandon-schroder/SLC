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
from slcflow.verification.v5_axial_compressor import (V5AxialRotor,
                                                      V5TransonicRotor)

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


# --------------------------------------------------------------------------
# Transonic rotor: an in-window transonic meanline point with the Aungier 6.7
# shock loss active (theory manual C.9). The earlier two-branch "tripwire" was
# a misdiagnosis -- the in-window condition is set by loading (D_eq), not the
# meridional-continuity branch; the ordinary mass-flow driver reaches it once
# the loading is in-window (V5TransonicRotor docstring; theory manual C.9).
# --------------------------------------------------------------------------
def test_transonic_rotor_converges_and_is_supersonic_relative():
    # The shock stack runs end-to-end: a high-omega rotor converges (T1 and
    # T2), the relative inlet Mach is supersonic (the shock regime -- the shock
    # component's engagement at M1 > 1 is pinned in test_lieblein_loss.py), and
    # it compresses.
    case = V5TransonicRotor()
    for n_sl in (1, 9):
        pr = case.evaluate(n_sl=n_sl)
        assert pr.converged
        lo, hi = case.pr_band
        assert lo < pr.pressure_ratio < hi          # real transonic compression
    pr1 = case.evaluate(n_sl=1)
    assert case.meanline_inlet_rel_mach(pr1) > 1.0  # supersonic relative inlet


def test_transonic_meanline_is_in_window_and_shock_active():
    # MEASURED 2026-07 (theory manual C.9, replacing the earlier out-of-window
    # "tripwire"): with the loading in-window (beta2 retuned so D_eq < 2.0), the
    # ordinary mass-flow meanline converges a genuine in-window transonic point
    # -- no supersonic-branch driver needed. The relative inlet is supersonic
    # (so the Aungier 6.7 shock loss is active; the shock component's engagement
    # at M1_rel > 1 is unit-pinned in test_lieblein_loss.py) and the closure
    # runs INSIDE the Lieblein validity window.
    case = V5TransonicRotor()
    pr = case.evaluate(n_sl=1)
    assert pr.converged
    assert case.meanline_inlet_rel_mach(pr) > 1.0    # supersonic relative -> shock
    assert pr.validity > 0.5                          # in the Lieblein window
    lo, hi = case.pr_band
    assert lo < pr.pressure_ratio < hi
    lo, hi = case.eta_band
    assert lo < pr.efficiency < hi
