"""Contract tests for slcflow.closures.smoothmath.

These verify the properties the correlation layer depends on: C1 continuity
(no derivative jumps, especially at regime boundaries), correct limiting
behavior as width -> 0, vectorization/purity, and monotonicity.
"""
import numpy as np
import pytest

from slcflow.closures import smoothmath as sm


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def numerical_derivative(f, x, dx=1e-6):
    return (f(x + dx) - f(x - dx)) / (2 * dx)


def assert_c1_continuous(f, x_lo, x_hi):
    """Assert f has a continuous first derivative via refinement scaling.

    Indicator = max|second difference| / dx  ~=  f''_max * dx for a C1 (indeed
    C2) function, so halving dx halves it (ratio -> 0.5). A genuine derivative
    discontinuity of jump J gives a second difference ~ J*dx at the kink, so the
    indicator ~ J stays constant under refinement (ratio -> 1). We require the
    indicator to at least halve, which cleanly separates smooth curvature
    (however concentrated) from a kink.
    """

    def indicator(n):
        x = np.linspace(x_lo, x_hi, n)
        dx = x[1] - x[0]
        y = f(x)
        d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
        assert np.all(np.isfinite(y))
        return np.max(np.abs(d2)) / dx

    coarse = indicator(2001)
    fine = indicator(4001)  # exactly half the dx
    ratio = fine / (coarse + 1e-300)
    assert ratio < 0.75, (
        f"first derivative appears discontinuous (refinement ratio {ratio:.3f}; "
        "expected ~0.5 for a smooth function, ~1.0 for a kink)"
    )


# --------------------------------------------------------------------------
# smoothstep
# --------------------------------------------------------------------------
def test_smoothstep_endpoints_and_midpoint():
    assert sm.smoothstep(-1.0, 0.0, 1.0) == 0.0
    assert sm.smoothstep(2.0, 0.0, 1.0) == 1.0
    assert sm.smoothstep(0.5, 0.0, 1.0) == pytest.approx(0.5)


def test_smoothstep_zero_derivative_at_edges():
    # Quintic smoothstep has zero 1st derivative at both edges.
    d_lo = numerical_derivative(lambda x: sm.smoothstep(x, 0.0, 1.0), 0.0)
    d_hi = numerical_derivative(lambda x: sm.smoothstep(x, 0.0, 1.0), 1.0)
    assert d_lo == pytest.approx(0.0, abs=1e-6)
    assert d_hi == pytest.approx(0.0, abs=1e-6)


def test_smoothstep_c1_across_edges():
    assert_c1_continuous(lambda x: sm.smoothstep(x, 0.0, 1.0), -0.5, 1.5)


def test_smoothstep_monotone():
    x = np.linspace(-0.5, 1.5, 500)
    y = sm.smoothstep(x, 0.0, 1.0)
    assert np.all(np.diff(y) >= -1e-15)


# --------------------------------------------------------------------------
# smooth_max / smooth_min
# --------------------------------------------------------------------------
@pytest.mark.parametrize("width", [1e-3, 1e-2, 1e-1])
def test_smooth_max_converges_to_max(width):
    a = np.array([1.0, -2.0, 3.5, 0.0])
    b = np.array([2.0, -1.0, 3.0, 0.0])
    approx = sm.smooth_max(a, b, width)
    true = np.maximum(a, b)
    # overestimate bounded by width*ln2
    assert np.all(approx >= true - 1e-12)
    assert np.all(approx <= true + width * np.log(2) + 1e-12)


def test_smooth_max_bound_tight_at_equality():
    approx = sm.smooth_max(3.0, 3.0, 0.1)
    assert approx == pytest.approx(3.0 + 0.1 * np.log(2))


def test_smooth_min_is_dual():
    a, b, w = 2.0, 5.0, 0.3
    assert sm.smooth_min(a, b, w) == pytest.approx(-sm.smooth_max(-a, -b, w))


def test_smooth_max_c1():
    assert_c1_continuous(lambda x: sm.smooth_max(x, 0.0, 0.2), -2.0, 2.0)


def test_smooth_max_stable_large_arguments():
    # logaddexp guards against overflow that naive exp() would hit.
    val = sm.smooth_max(700.0, 705.0, 1.0)
    assert np.isfinite(val)
    assert val == pytest.approx(705.0, abs=0.05)


# --------------------------------------------------------------------------
# soft_clip
# --------------------------------------------------------------------------
def test_soft_clip_interior_identity():
    x = np.array([0.0, 0.5, -0.5])
    y = sm.soft_clip(x, -5.0, 5.0, 0.05)
    assert np.allclose(y, x, atol=1e-3)


def test_soft_clip_saturates():
    assert sm.soft_clip(100.0, -1.0, 1.0, 0.05) == pytest.approx(1.0, abs=1e-2)
    assert sm.soft_clip(-100.0, -1.0, 1.0, 0.05) == pytest.approx(-1.0, abs=1e-2)


def test_soft_clip_c1_across_both_corners():
    assert_c1_continuous(lambda x: sm.soft_clip(x, -1.0, 1.0, 0.1), -2.0, 2.0)


# --------------------------------------------------------------------------
# logistic / blend / blend_between
# --------------------------------------------------------------------------
def test_logistic_midpoint_and_range():
    assert sm.logistic(0.0, 0.0, 0.1) == pytest.approx(0.5)
    x = np.linspace(-5, 5, 100)
    y = sm.logistic(x, 0.0, 0.3)
    assert np.all((y > 0) & (y < 1))
    assert np.all(np.diff(y) > 0)  # strictly monotone


def test_blend_compact_support():
    assert sm.blend(-1.0, 0.0, 0.5) == 0.0  # below band -> exactly 0
    assert sm.blend(1.0, 0.0, 0.5) == 1.0   # above band -> exactly 1
    assert sm.blend(0.0, 0.0, 0.5) == pytest.approx(0.5)


def test_blend_between_regime_values():
    # Far into each regime returns the exact regime value.
    lo = sm.blend_between(-10.0, 2.0, 9.0, 0.0, 0.5)
    hi = sm.blend_between(10.0, 2.0, 9.0, 0.0, 0.5)
    assert lo == pytest.approx(2.0)
    assert hi == pytest.approx(9.0)


def test_blend_between_c1():
    assert_c1_continuous(
        lambda x: sm.blend_between(x, 2.0, 9.0, 0.0, 0.5), -1.5, 1.5
    )


# --------------------------------------------------------------------------
# abs_smooth
# --------------------------------------------------------------------------
def test_abs_smooth_bound_and_limit():
    eps = 1e-3
    x = np.array([-2.0, -0.1, 0.0, 0.1, 2.0])
    y = sm.abs_smooth(x, eps)
    assert np.all(y >= np.abs(x))
    assert np.all(y <= np.abs(x) + eps)


def test_abs_smooth_c1_through_zero():
    assert_c1_continuous(lambda x: sm.abs_smooth(x, 1e-2), -1.0, 1.0)


# --------------------------------------------------------------------------
# vectorization / purity
# --------------------------------------------------------------------------
def test_vectorization_shape_preserved():
    x = np.linspace(-1, 1, 37).reshape(37, 1)
    for f in [
        lambda x: sm.smoothstep(x, -0.5, 0.5),
        lambda x: sm.soft_clip(x, -0.5, 0.5, 0.1),
        lambda x: sm.logistic(x, 0.0, 0.2),
        lambda x: sm.blend(x, 0.0, 0.3),
        lambda x: sm.abs_smooth(x, 1e-2),
    ]:
        y = f(x)
        assert y.shape == x.shape


def test_purity_no_mutation():
    x = np.linspace(-1, 1, 10)
    x_copy = x.copy()
    sm.soft_clip(x, -0.5, 0.5, 0.1)
    sm.smooth_max(x, 0.0, 0.2)
    assert np.array_equal(x, x_copy)


# --------------------------------------------------------------------------
# softplus
# --------------------------------------------------------------------------
def test_softplus_asymptotes_and_knee():
    w = 0.1
    assert sm.softplus(-5.0, w) == pytest.approx(0.0, abs=1e-12)
    assert sm.softplus(5.0, w) == pytest.approx(5.0, rel=1e-12)
    assert sm.softplus(0.0, w) == pytest.approx(w * np.log(2))


def test_softplus_c1_and_monotone():
    assert_c1_continuous(lambda x: sm.softplus(x, 0.05), -1.0, 1.0)
    x = np.linspace(-1, 1, 400)
    assert np.all(np.diff(sm.softplus(x, 0.05)) >= 0)


def test_softplus_converges_to_relu():
    x = np.array([-2.0, -0.01, 0.01, 2.0])
    for w in [1e-2, 1e-3]:
        assert np.allclose(sm.softplus(x, w), np.maximum(x, 0.0), atol=w)


# --------------------------------------------------------------------------
# numerical stability & warnings hygiene
# --------------------------------------------------------------------------
def test_logistic_no_overflow_in_either_tail():
    with np.errstate(over="raise", invalid="raise"):
        lo = sm.logistic(np.array([-1e6, -1e3]), 0.0, 1.0)
        hi = sm.logistic(np.array([1e3, 1e6]), 0.0, 1.0)
    assert np.allclose(lo, 0.0)
    assert np.allclose(hi, 1.0)


def test_softplus_smoothmax_stable_extremes():
    with np.errstate(over="raise", invalid="raise"):
        assert np.isfinite(sm.softplus(1e6, 1.0))
        assert np.isfinite(sm.smooth_max(-1e6, 1e6, 1.0))


# --------------------------------------------------------------------------
# parameter validation (config boundary, AD-10)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    lambda: sm.smooth_max(1.0, 2.0, 0.0),
    lambda: sm.smooth_min(1.0, 2.0, -0.1),
    lambda: sm.softplus(1.0, 0.0),
    lambda: sm.logistic(1.0, 0.0, 0.0),
    lambda: sm.abs_smooth(1.0, 0.0),
    lambda: sm.smoothstep(0.5, 1.0, 1.0),   # edge1 must exceed edge0
    lambda: sm.soft_clip(0.0, 1.0, -1.0, 0.1),  # hi must exceed lo
])
def test_invalid_parameters_raise(call):
    with pytest.raises(ValueError):
        call()


# --------------------------------------------------------------------------
# explicit xp injection path (AD-6)
# --------------------------------------------------------------------------
def test_explicit_xp_injection_matches_default():
    x = np.linspace(-1, 1, 11)
    assert np.array_equal(sm.softplus(x, 0.1, xp=np), sm.softplus(x, 0.1))
    assert np.array_equal(
        sm.logistic(x, 0.0, 0.2, xp=np), sm.logistic(x, 0.0, 0.2)
    )