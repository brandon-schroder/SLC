"""Cetin/Swan AGARD-R-745 transonic deviation corrections (section 4.3, 7.3).

Coefficient pins + smoothness + config-boundary behavior for
``cetin_deviation_correction``, ``swan_offdesign_deviation``, and the
``LieblienSwirl`` option seams. Source note: docs/references/AGARD745.md
(Eq. 3.5 / Eq. 70 verbatim via the loss-models notebook, 2026-07-16);
end-to-end validation vs the measured Rotor 37 blade elements lives in
``test_v5_rotor37.py`` (where the Swan rule's measured NON-adoption on the
Rotor 37 line is also recorded).
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.lieblein import (
    LieblienSwirl, cetin_deviation_correction, swan_offdesign_deviation)
from slcflow.errors import ConfigError


def _assert_c1_continuous(f, x_lo, x_hi):
    """Refinement-scaling C1 check (the test_smoothmath pattern: the
    max-second-difference indicator halves under dx-halving iff the first
    derivative is continuous)."""
    def indicator(n):
        x = np.linspace(x_lo, x_hi, n)
        dx = x[1] - x[0]
        y = f(x)
        d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
        assert np.all(np.isfinite(y))
        return np.max(np.abs(d2)) / dx

    ratio = indicator(4001) / (indicator(2001) + 1e-300)
    assert ratio < 0.75, f"kink suspected (refinement ratio {ratio:.3f})"


def test_eq35_coefficients_verbatim():
    # AGARD-R-745 Eq. 3.5: delta*cor = -1.099379 + 3.0186 d - 0.1988 d^2.
    # Exact (to fp) DEEP inside the saturation window; within the C1
    # saturation skin (~1e-2) toward the window edges (the documented
    # _saturate behavior, section 7.3.2).
    got, v = cetin_deviation_correction(4.0)
    assert float(got) == pytest.approx(7.794221, abs=1e-6)
    assert float(v) == pytest.approx(1.0, abs=1e-9)
    for d in (2.0, 6.0):
        expect = -1.099379 + 3.0186 * d - 0.1988 * d * d
        got, v = cetin_deviation_correction(d)
        assert float(got) == pytest.approx(expect, abs=1e-2)
        assert float(v) == pytest.approx(1.0, abs=1e-6)


def test_correction_increases_deviation_in_fitted_range():
    # The whole point (and AGARD's finding): classical rules UNDERESTIMATE
    # transonic deviation, so the correction must raise it over the fitted
    # branch (section 4.3).
    for d in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        got, _ = cetin_deviation_correction(d)
        assert float(got) > d


def test_c1_across_saturation_knees():
    # Section 7.3: the saturated polynomial is C1 across both window knees
    # (refinement-scaling check, the smoothmath pattern).
    _assert_c1_continuous(
        lambda x: np.asarray(cetin_deviation_correction(x)[0]), -2.0, 12.0)


def test_validity_falls_outside_fitted_branch():
    # Compact-support validity: 1 inside (0.5, 7.5), toward 0 outside
    # (section 7.3.3) -- out-of-range inputs saturate, never extrapolate
    # the non-monotone branch of the parabola.
    _, v_in = cetin_deviation_correction(4.0)
    _, v_hi = cetin_deviation_correction(10.0)
    _, v_lo = cetin_deviation_correction(0.0)
    assert float(v_in) == pytest.approx(1.0, abs=1e-9)
    assert float(v_hi) < 0.1 and float(v_lo) < 0.1
    # And the saturated output can never exceed the vertex value:
    hi, _ = cetin_deviation_correction(50.0)
    assert float(hi) <= -1.099379 + 3.0186 * 7.59 - 0.1988 * 7.59 ** 2 + 0.1


def test_swan_eq70_coefficients_verbatim():
    # AGARD-R-745 App. II Eq. 70: delta - delta* =
    # [6.40 - 9.45 (M1 - 0.60)] (D_eq - D_eq*), inside the ceiling.
    for m1, ddeq in ((0.8, 0.10), (1.0, -0.15), (1.2, 0.20)):
        expect = (6.40 - 9.45 * (m1 - 0.60)) * ddeq
        got, v = swan_offdesign_deviation(m1, 1.5 + ddeq, 1.5)
        assert float(got) == pytest.approx(expect, abs=1e-6)
        assert float(v) == pytest.approx(1.0, abs=1e-6)


def test_swan_bracket_sign_change_at_transonic_mach():
    # The bracket crosses zero at M1 = 1.277: above it, LOWER loading
    # (D_eq < D_eq*, the choke side) RAISES deviation — the transonic
    # reversal the rule exists for (section 4.3 / AGARD745.md).
    lo, _ = swan_offdesign_deviation(1.0, 1.3, 1.5)    # subcritical bracket
    hi, _ = swan_offdesign_deviation(1.45, 1.3, 1.5)   # supercritical
    assert float(lo) < 0.0 < float(hi)


def test_swan_increment_ceiling_and_validity_window():
    # Section 7.3.2 guards: the increment is smoothly ceilinged at +-8 deg
    # (lagged-state transients can produce wild D_eq excursions) and
    # validity ends at the AGARD data range (M1 ~ 1.5).
    big, _ = swan_offdesign_deviation(0.7, 6.0, 1.4)
    assert float(big) <= 8.0 + 1e-6
    small, _ = swan_offdesign_deviation(1.45, 6.0, 1.4)
    assert float(small) >= -8.0 - 1e-6
    _, v_in = swan_offdesign_deviation(1.0, 1.5, 1.5)
    _, v_out = swan_offdesign_deviation(1.8, 1.5, 1.5)
    assert float(v_in) > 0.95 and float(v_out) < 0.1


def test_swan_c1_in_mach():
    # C1 in the flow Mach across the ceiling knees (section 7.3).
    _assert_c1_continuous(
        lambda m: np.asarray(
            swan_offdesign_deviation(m, 2.6, 1.4)[0]), 0.3, 1.6)


def test_unknown_correction_option_raises_at_construction():
    # Config boundary (AD-10): a typo'd option fails loudly at build time,
    # never on the closure-evaluation path.
    with pytest.raises(ConfigError):
        LieblienSwirl(transonic_correction="cetin")


def test_unknown_offdesign_rule_raises_at_construction():
    with pytest.raises(ConfigError):
        LieblienSwirl(offdesign_rule="swan")


def test_default_swirl_is_uncorrected():
    # Behavior preservation: the shipped LIEBLEIN_NACA65 set stays the
    # SP-36 NACA-65 pedigree (correction off, Aungier slope, by default).
    from slcflow.closures.axial_compressor import LIEBLEIN_NACA65
    assert LieblienSwirl().transonic_correction == "none"
    assert LieblienSwirl().offdesign_rule == "aungier"
    assert LIEBLEIN_NACA65.swirl.transonic_correction == "none"
    assert LIEBLEIN_NACA65.swirl.offdesign_rule == "aungier"
