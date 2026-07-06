"""V6 — Axial-turbine entry point (Theory Manual section 9.6; ARCH-5.5,
ARCH-8 M6). Structural half only.

These bind the *structural* V6 gate: a pre-swirled axial-turbine rotor
composed through the Machine facade with the Kacker-Okapuu set, run at
Tier-1 meanline and a spanwise grid, converges, *extracts* real work with
real loss, and lands total-to-total expansion ratio and efficiency in
physically sane bands. The quantitative half — matching a specific
Kacker-Okapuu validation case / published stage map point-by-point, and
speedline/choke traversal — is **[VERIFY]**, blocked on the reference-library
correlation calibration (every K-O fit coefficient is [VERIFY]) as for V5.
The bands here are plausibility gates, not Appendix-C tolerances.

Provenance: M6 sub-step 5, written with the K-O turbine set.
"""
import numpy as np
import pytest

from slcflow.machine import FidelityConfig, PerformanceResult
from slcflow.verification.v6_axial_turbine import V6AxialTurbine

DEG = 180.0 / np.pi


# --------------------------------------------------------------------------
# Section 9.6: Tier-1 meanline rotor through the facade
# --------------------------------------------------------------------------
def test_meanline_converges_and_extracts_work():
    case = V6AxialTurbine()
    pr = case.evaluate(n_sl=1)          # Tier-1 meanline (section 8)
    assert isinstance(pr, PerformanceResult)
    assert pr.converged
    assert pr.vm.shape == (1,)

    # Turbine EXTRACTS work (h0 drops) with real loss (entropy rises) across
    # the rotor (sections 3.3, 3.5). This is the defining V6 behaviour, the
    # sign opposite of V5.
    tr = pr.result.frozen.transported
    assert tr.h0[0, 2] - tr.h0[0, 1] < 0.0      # turbine lowers h0 (work out)
    assert tr.s[0, 2] - tr.s[0, 1] > 0.0        # entropy rises (loss)


def test_meanline_expansion_and_efficiency_bands():
    # Structural plausibility gate: total-to-total PR < 1 (expansion) and the
    # facade's inverted turbine efficiency (> 1) in a sane band. NOT a V6
    # tolerance (point-by-point reproduction is [VERIFY]).
    case = V6AxialTurbine()
    pr = case.evaluate(n_sl=1)
    lo, hi = case.pr_band
    assert lo < pr.pressure_ratio < hi
    assert pr.pressure_ratio < 1.0              # genuine expansion
    lo, hi = case.eta_band
    assert lo < pr.efficiency < hi              # turbine eta ~ 1/1.05 ~ 0.95
    assert pr.validity > 0.5


def test_meanline_deswirls_toward_axial_exit():
    # The rotor removes the inlet pre-swirl (that is where the work comes
    # from): exit rVtheta well below inlet, exit absolute flow near axial.
    case = V6AxialTurbine()
    pr = case.evaluate(n_sl=1)
    tr = pr.result.frozen.transported
    assert tr.rvt[0, 2] < 0.5 * tr.rvt[0, 1]    # strong de-swirl
    assert abs(pr.alpha[0]) < 30.0 / DEG        # exit close-ish to axial
    assert np.all(pr.result.fields.mach_m < 1.0)


# --------------------------------------------------------------------------
# Section 8: tiers are data (spanwise run + Tier 2/3 consistency)
# --------------------------------------------------------------------------
def test_spanwise_run_agrees_with_meanline():
    # A spanwise Tier-2 grid converges and its mass-averaged work/expansion
    # track the meanline (the section-8 consistency requirement, structural
    # form for the turbine set).
    case = V6AxialTurbine()
    m1 = case.evaluate(n_sl=1, fidelity=FidelityConfig.tier1())
    m2 = case.evaluate(n_sl=9, fidelity=FidelityConfig.tier2())
    assert m2.converged and m2.vm.shape == (9,)
    assert m2.pressure_ratio == pytest.approx(m1.pressure_ratio, rel=5e-3)
    assert m2.efficiency == pytest.approx(m1.efficiency, rel=5e-3)


def test_tier2_tier3_consistency_straight_annulus():
    # Straight annulus: Tier 2 (REE only) and Tier 3 (curvature on) agree
    # closely. NOT the bit-identical V3 gate -- that holds only for clean
    # free/forced-vortex cases (test_v3_tier_consistency.py). Here the
    # throat-based exit swirl is spanwise-varying, so repositioning gives the
    # streamlines slight meridional curvature and Tier 3's curvature term is
    # small-but-nonzero; the tiers track to well under a percent.
    case = V6AxialTurbine()
    t2 = case.evaluate(n_sl=9, fidelity=FidelityConfig.tier2())
    t3 = case.evaluate(n_sl=9, fidelity=FidelityConfig.tier3())
    assert t2.converged and t3.converged
    np.testing.assert_allclose(t3.pressure_ratio, t2.pressure_ratio,
                               rtol=2e-3)
    np.testing.assert_allclose(t3.vm, t2.vm, rtol=1.5e-2)


def test_facade_is_deterministic_replay():
    # AD-3: identical inputs -> identical outputs.
    case = V6AxialTurbine()
    a = case.evaluate(n_sl=1)
    b = case.evaluate(n_sl=1)
    assert a.pressure_ratio == b.pressure_ratio
    np.testing.assert_array_equal(a.result.x, b.result.x)


def test_tier1_is_pure_data_switch():
    # AD-1: the meanline is n_sl = 1 through the same facade/kernel.
    assert FidelityConfig.tier1() == FidelityConfig.tier2()
    pr = V6AxialTurbine().evaluate(n_sl=1, fidelity=FidelityConfig.tier1())
    assert pr.converged
