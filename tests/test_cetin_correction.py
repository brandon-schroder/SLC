"""Cetin AGARD-R-745 transonic deviation correction (section 4.3, 7.3).

Coefficient pins + smoothness + config-boundary behavior for
``cetin_deviation_correction`` and the ``LieblienSwirl.transonic_correction``
seam. Source note: docs/references/AGARD745.md (Eq. 3.5 verbatim via the
loss-models notebook, 2026-07-16); end-to-end validation vs the measured
Rotor 37 blade elements lives in ``test_v5_rotor37.py``.
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.lieblein import (
    LieblienSwirl, cetin_deviation_correction)
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


def test_unknown_correction_option_raises_at_construction():
    # Config boundary (AD-10): a typo'd option fails loudly at build time,
    # never on the closure-evaluation path.
    with pytest.raises(ConfigError):
        LieblienSwirl(transonic_correction="cetin")


def test_default_swirl_is_uncorrected():
    # Behavior preservation: the shipped LIEBLEIN_NACA65 set stays the
    # SP-36 NACA-65 pedigree (correction off by default).
    from slcflow.closures.axial_compressor import LIEBLEIN_NACA65
    assert LieblienSwirl().transonic_correction == "none"
    assert LIEBLEIN_NACA65.swirl.transonic_correction == "none"
