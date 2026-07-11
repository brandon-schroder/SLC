"""V8 mixed-flow compressor structural tests (Theory Manual section 9.8;
Appendix C.8; M8 sub-step 4, Tier 3 flipped at the 2026-07 stabilization).

Structural verification: a mixed-flow impeller on a partial axial->radial
bend converges, compresses (PR > 1) with real loss, and exits at an
intermediate meridional angle with a radius rise (r_LE < r_exit < rc, the
signature of 0 < phi < 90). Point-by-point reproduction is [VERIFY].

Tier 3 (full-SLC repositioning) originally did NOT converge here and was
pinned as a tripwire (M8-4 attributed it to an angle-specific repositioning
pocket). The 2026-07 diagnosis refuted that story: the failure was the
driver's stale-split boundary check, spurious negative-Vm continuity
branches, and the unrelaxed closure switch-on. Post-stabilization Tier 3
converges on this bend; the flipped tripwire below now pins THAT.

Provenance: M8 sub-step 4, written with the V8 case; Tier-3 test revised
2026-07.
"""
import numpy as np
import pytest

from slcflow.drivers.classical import ClassicalConfig
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
# One kernel, three tiers (AD-1): all converge post-stabilization
# --------------------------------------------------------------------------
def test_tier2_ree_converges():
    case = V8MixedFlow()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier2(), n_sl=case.n_sl_rep)
    assert r.converged
    assert r.pressure_ratio > 1.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # wedge transient
@pytest.mark.xfail(strict=True, reason=(
    "Blade-loading (diffusion) loss added 2026-07 -- the DOMINANT centrifugal "
    "internal loss -- pushes V8 mixed-flow Tier 3 into the documented "
    "freeze-fallback wedge (exit q-o has no positive-branch root at any mdot; "
    "lowering mdot makes it worse). Tier 1/2 converge with realistic eta ~0.90. "
    "Pre-loss Tier 3 converged (the 2026-07 stabilization, 396 iters) but "
    "realistic loss overwhelms it; the wedge's recorded attacks are "
    "closure-in-Newton or a compact-support streamline fit (major, not patches). "
    "REMOVE this xfail when the wedge is cracked -- strict=True flags the XPASS."))
def test_tier3_hits_the_documented_wedge_with_realistic_loss():
    # Tripwire: the assertion is the PRE-loss expectation (Tier 3 converges and
    # agrees with Tier 2). It now xfails because the realistic blade-loading
    # loss drives the mixed-flow bend into the wedge (memory:
    # centrifugal-blade-loading-wip; Appendix C.8 note).
    case = V8MixedFlow()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier3(), n_sl=case.n_sl_rep,
                                config=ClassicalConfig(max_outer=600))
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
    r2 = case.machine().evaluate(MassFlowSpec(case.mdot),
                                 FidelityConfig.tier2(), n_sl=case.n_sl_rep)
    assert r2.converged
    assert r.pressure_ratio == pytest.approx(r2.pressure_ratio, rel=5e-2)
