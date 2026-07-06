"""V8 mixed-flow compressor structural tests (Theory Manual section 9.8;
Appendix C.8; M8 sub-step 4).

Structural verification at Tier 1 (meanline) and Tier 2 (REE): a mixed-flow
impeller on a partial axial->radial bend converges, compresses (PR > 1) with
real loss, and exits at an intermediate meridional angle with a radius rise
(r_LE < r_exit < rc, the signature of 0 < phi < 90). Point-by-point
reproduction is [VERIFY].

Tier 3 (full-SLC repositioning) does NOT converge on the mixed-flow bend --
the M7-4 radial-repositioning pocket is angle-specific and does not transfer.
That is pinned here as an explicit tripwire, not hidden.

Provenance: M8 sub-step 4, written with the V8 case.
"""
import numpy as np
import pytest

from slcflow.types import FidelityConfig, MassFlowSpec
from slcflow.verification.v8_mixed_flow import V8MixedFlow


@pytest.fixture(scope="module")
def meanline():
    return V8MixedFlow().evaluate(n_sl=1)


# --------------------------------------------------------------------------
# Section 9.8: mixed-flow compression at the meanline
# --------------------------------------------------------------------------
def test_meanline_converges_and_compresses(meanline):
    r = meanline
    assert r.converged
    lo, hi = V8MixedFlow().pr_band
    assert lo < r.pressure_ratio < hi
    assert r.pressure_ratio > 1.0
    elo, ehi = V8MixedFlow().eta_band
    assert elo < r.efficiency < ehi
    tr = r.result.frozen.transported
    assert float(tr.h0[0, -1]) > float(tr.h0[0, 0])     # work in (dh0 > 0)
    assert float(tr.s[0, -1]) > float(tr.s[0, 0])       # real loss
    assert float(tr.rvt[0, -1]) > 0.0                   # spun up from axial


def test_exit_is_mixed_flow(meanline):
    # r_LE < r_exit < rc: the flow turned partway (0 < phi < 90) with a
    # radius rise -- neither axial (V5) nor fully radial (V7).
    case = V8MixedFlow()
    m = meanline.result.fields.metrics
    r_le = float(np.mean(m.r[:, 1]))
    r_exit = float(np.mean(m.r[:, -1]))
    assert r_le < r_exit < case.rc


def test_exit_swirl_is_slipped(meanline):
    case = V8MixedFlow()
    tr = meanline.result.frozen.transported
    u2 = case.omega * case.rc
    vtheta_exit = float(tr.rvt[0, -1]) / float(
        meanline.result.fields.metrics.r[0, -1])
    assert 0.0 < vtheta_exit < u2                       # Wiesner sigma < 1


# --------------------------------------------------------------------------
# One kernel: Tier 2 converges; Tier 3 is the documented carryover
# --------------------------------------------------------------------------
def test_tier2_ree_converges():
    case = V8MixedFlow()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier2(), n_sl=case.n_sl_rep)
    assert r.converged
    assert r.pressure_ratio > 1.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_tier3_is_the_known_repositioning_carryover():
    # TRIPWIRE (M8-4): Tier-3 full-SLC repositioning on the mixed-flow bend is
    # beyond the current stabilization -- the V7 90-degree pocket does not
    # transfer to intermediate angles (module docstring / Appendix C.8). When
    # a robust radial/mixed repositioning stabilization lands, this assertion
    # flips and the test fails LOUDLY -- flip it to `assert r.converged` then.
    case = V8MixedFlow()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier3(), n_sl=case.n_sl_rep)
    assert not r.converged
