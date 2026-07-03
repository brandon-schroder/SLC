"""Independent adjudication suite for slcflow.fluid and slcflow.closures.smoothmath
(ARCH-4.1, ARCH-4.2; Theory Manual sections 3.7, 7.3).

Provenance note: unlike the grid/geometry adjudication precedent
(test_grid_adjudication.py), fluid/perfectgas.py and closures/smoothmath.py
are not of uncertain provenance -- they were written deliberately against
these contracts. This suite exists for a different reason: a coverage-gap
check, not a trust exercise. The checklist below was derived from ARCH-4.1 /
ARCH-4.2 and Theory Manual sections 3.7 / 7.3 *before* reading
test_fluid_conformance.py, test_perfectgas.py, or test_smoothmath.py, then
diffed against their actual assertions (see the audit session notes for the
full diff). Six coverage gaps were found; none were contract violations --
each was empirically verified to already hold before a test was written for
it, so nothing here should ever fail against the current implementation.

1. dp/dh|_s = rho -- the Gibbs-relation slice complementary to the existing
   dh/ds|_p = T check (test_fluid_conformance.py #2); an independent
   projection of the same relation, not redundant with it.
2. Direct degenerate broadcast of T(h,s)/p(h,s)/a(h,s)/rho(h,s) against
   mismatched h/s shapes -- the existing broadcast test only exercises
   h_from_Tp/s_from_Tp's degenerate args, never calls e.g. T(h, s) with h
   and s of different shapes directly (the exact case perfectgas.py's own
   "conformance test 5" comment calls out).
3. logistic() has no C1-via-refinement-scaling proof (every other primitive
   in __all__ does; smooth_min/blend are structurally redundant with
   smooth_max/smoothstep -- smooth_min is -smooth_max(-a,-b,w) and blend is
   smoothstep with shifted edges -- and are correctly not re-proved).
4. No signature-level test of the "mandatory explicit width, no hidden
   default" contract (ARCH-4.2).
5. The existing AD-6 xp-injection test only passes xp=np, which *is* the
   default -- it cannot distinguish "routes through the injected xp" from
   "silently calls numpy regardless of xp". This suite injects a recording
   proxy namespace instead.
6. No negative control proving a C1 refinement-scaling checker actually
   rejects a known-discontinuous function (required by CLAUDE.md for any
   new smoothness checker; never added for the existing one). Verified by
   hand first that the checker rejects abs(x)/relu(x) (ratio -> 1.0) and
   passes sin(x) (ratio -> 0.5) -- this was a coverage gap, not a bug.
"""
import inspect

import numpy as np
import pytest

from slcflow.closures import smoothmath as sm
from slcflow.fluid import PerfectGas

# --- registered backends, mirroring test_fluid_conformance.py's pattern ----
BACKENDS = [
    pytest.param(PerfectGas(), {"T": (250.0, 1500.0), "p": (3e4, 3e6)}, id="PerfectGas-air"),
]
RNG_SEED = 20260703
N_SAMPLES = 20


def _sample_states(box, n=N_SAMPLES):
    rng = np.random.default_rng(RNG_SEED)
    T = rng.uniform(*box["T"], size=n)
    p = np.exp(rng.uniform(np.log(box["p"][0]), np.log(box["p"][1]), size=n))
    return T, p


# ---------------------------------------------------------------------------
# Gap 1 -- dp/dh|_s = rho (Gibbs relation, fixed-s slice)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fluid,box", BACKENDS)
def test_gibbs_relation_dp_dh_at_const_s(fluid, box):
    """dp/dh|_s = rho, by central difference at fixed s.

    From dh = T ds + dp/rho: holding s fixed (ds=0) gives dp = rho*dh, i.e.
    dp/dh|_s = rho. This is an independent slice of the same Gibbs relation
    from test_fluid_conformance.py's dh/ds|_p = T check (that one holds p
    fixed and probes T; this one holds s fixed and probes rho) -- neither
    implies the other.
    """
    T, p = _sample_states(box)
    h = fluid.h_from_Tp(T, p)
    s = fluid.s_from_Tp(T, p)
    dh = np.maximum(np.abs(h), 1.0) * 1e-6
    dp = fluid.p(h + dh, s) - fluid.p(h - dh, s)
    dp_dh = dp / (2 * dh)
    assert np.allclose(dp_dh, fluid.rho(h, s), rtol=1e-5)


# ---------------------------------------------------------------------------
# Gap 2 -- degenerate broadcast of T/p/a/rho against mismatched h,s shapes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fluid,box", BACKENDS)
def test_degenerate_broadcast_against_mismatched_hs_shapes(fluid, box):
    """T(h,s), p(h,s), a(h,s), rho(h,s) called with h and s of *different*
    shapes must broadcast, not just work when h and s already share a shape.

    T(h,s) is mathematically independent of s (T = h/cp); the existing
    broadcast test (test_fluid_conformance.py #5) only ever calls rho(h,s)
    with an already-matched-shape (h,s) pair built from h_from_Tp/s_from_Tp.
    This directly exercises the case perfectgas.py's own comment calls out.
    """
    T, p = _sample_states(box, n=4)
    h = fluid.h_from_Tp(T, p)          # shape (4,)
    s_col = fluid.s_from_Tp(T, p)[:, None]  # reshape to (4, 1): mismatched vs h

    T_out = fluid.T(h, s_col)
    p_out = fluid.p(h, s_col)
    a_out = fluid.a(h, s_col)
    rho_out = fluid.rho(h, s_col)
    for out in (T_out, p_out, a_out, rho_out):
        assert out.shape == (4, 4)

    # T depends only on h -- every row (fixed s) must reproduce the same
    # per-column values regardless of which s row produced it.
    assert np.allclose(T_out, T_out[0:1, :])


# ---------------------------------------------------------------------------
# Gap 3 -- logistic() C1 continuity via refinement scaling
# ---------------------------------------------------------------------------
def _indicator(f, x_lo, x_hi, n):
    x = np.linspace(x_lo, x_hi, n)
    dx = x[1] - x[0]
    y = np.asarray(f(x), dtype=float)
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    assert np.all(np.isfinite(y))
    return np.max(np.abs(d2)) / dx


def _assert_c1_continuous(f, x_lo, x_hi):
    """Independent re-implementation of the refinement-scaling C1 check
    (same algorithm as test_smoothmath.py::assert_c1_continuous): a genuine
    derivative discontinuity gives an indicator that stays ~constant under
    grid refinement (ratio -> 1); a C1 function's indicator roughly halves
    (ratio -> 0.5) since it scales with the (shrinking) step size.
    """
    coarse = _indicator(f, x_lo, x_hi, 2001)
    fine = _indicator(f, x_lo, x_hi, 4001)
    ratio = fine / (coarse + 1e-300)
    assert ratio < 0.75, f"derivative discontinuity suspected (ratio {ratio:.3f})"


def test_logistic_c1_continuous():
    _assert_c1_continuous(lambda x: sm.logistic(x, 0.0, 0.2), -3.0, 3.0)


# ---------------------------------------------------------------------------
# Gap 6 -- negative control: the refinement-scaling check must reject a kink
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kinked_fn", [
    lambda x: np.abs(x),
    lambda x: np.maximum(x, 0.0),
])
def test_c1_checker_rejects_known_kink(kinked_fn):
    with pytest.raises(AssertionError):
        _assert_c1_continuous(kinked_fn, -1.0, 1.0)


def test_c1_checker_passes_known_smooth_function():
    _assert_c1_continuous(lambda x: np.sin(x), -2.0, 2.0)


# ---------------------------------------------------------------------------
# Gap 4 -- no hidden default on any smoothing-scale (or other) parameter
# ---------------------------------------------------------------------------
def test_no_hidden_defaults_except_xp():
    """ARCH-4.2: 'mandatory, explicit width -- there is no hidden default
    smoothing length'. Generalized: every parameter except xp must be
    required, on every primitive in the public API."""
    for name in sm.__all__:
        fn = getattr(sm, name)
        for pname, param in inspect.signature(fn).parameters.items():
            if pname == "xp":
                assert param.default is None, f"{name}.xp should default to None"
            else:
                assert param.default is inspect.Parameter.empty, (
                    f"{name}.{pname} has a hidden default ({param.default!r}); "
                    "ARCH-4.2 requires explicit, mandatory smoothing-scale params"
                )


# ---------------------------------------------------------------------------
# Gap 5 -- xp is genuinely routed through, not just accepted and ignored
# ---------------------------------------------------------------------------
class _RecordingXp:
    """Numpy-backed proxy that records which attributes were accessed, so a
    test can prove a primitive actually dispatches through the *injected*
    namespace object rather than a module-level numpy import (AD-6).
    Passing xp=np (as the existing test_explicit_xp_injection_matches_default
    does) cannot distinguish this, since np is already the default.
    """

    def __init__(self):
        self.accessed = set()

    def __getattr__(self, item):
        self.accessed.add(item)
        return getattr(np, item)


@pytest.mark.parametrize("call,expected_ufunc", [
    (lambda xp: sm.smooth_max(1.0, 2.0, 0.5, xp=xp), "logaddexp"),
    (lambda xp: sm.smooth_min(1.0, 2.0, 0.5, xp=xp), "logaddexp"),
    (lambda xp: sm.softplus(1.0, 0.5, xp=xp), "logaddexp"),
    (lambda xp: sm.logistic(1.0, 0.0, 0.5, xp=xp), "tanh"),
    (lambda xp: sm.abs_smooth(1.0, 0.1, xp=xp), "sqrt"),
    (lambda xp: sm.smoothstep(0.5, 0.0, 1.0, xp=xp), "clip"),
])
def test_xp_injection_actually_dispatches_through_injected_namespace(call, expected_ufunc):
    recorder = _RecordingXp()
    result = call(recorder)
    default_result = call(np)
    assert result == pytest.approx(default_result)
    assert expected_ufunc in recorder.accessed, (
        f"expected {expected_ufunc!r} to be called on the injected xp; "
        f"got {recorder.accessed} -- suggests numpy is hardcoded internally"
    )
