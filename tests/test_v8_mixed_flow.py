"""V8 mixed-flow compressor structural tests (Theory Manual section 9.8;
Appendix C.8; M8 sub-step 4, Tier 3 flipped at the 2026-07 stabilization).

Structural verification: a mixed-flow impeller on a partial axial->radial
bend converges, compresses (PR > 1) with real loss, and exits at an
intermediate meridional angle with a radius rise (r_LE < r_exit < rc, the
signature of 0 < phi < 90). Point-by-point reproduction is [VERIFY].

Tier 3 (full-SLC repositioning) history: originally a tripwire (M8-4), then
converged post-stabilization on the pre-blade-loading bend, then pushed into a
choke/max-iter pocket by the dominant blade-loading loss (xfail at the old
mdot=12). The 2026-07-12 Coppage/Oh-1997 D_f ratio fix (~2.3x less loss)
lowered the pocket into a converging window mdot in {13, 14}; the case is
re-centred to mdot=14 where all three tiers converge, pinned by
``test_tier3_converges_at_recentred_mdot``.

Provenance: M8 sub-step 4, written with the V8 case; Tier-3 test revised
2026-07 and 2026-07-12 (blade-loading fix + mdot re-centre).
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


# V8 Tier 3 with the dominant blade-loading loss: a PASSING test at the
# re-centred mdot=14. The Coppage/Oh-1997 D_f ratio fix (2026-07-12) lowered
# the pocket into a converging window (choke at 12; at the duct-default
# wilkinson_c=4.4, mdot 14 needed 395 slow outer iterations and 15/16 were
# slow-max-iter). The C.3-grounded per-case wilkinson_c=13 (2026-07-19; safe
# for phi=55/n_inblade=6 by Appendix C.3, c*=13.2, IDENTICAL answers)
# accelerates it ~2.6x AND widens the pocket to {13, 14, 15}.
def test_tier3_converges_at_recentred_mdot():
    # Tier 3 at mdot=14 converges with the realistic blade-loading loss and
    # agrees with Tier 2 (validity 1). With the case wilkinson_c=13 it fits
    # inside the stock max_outer=200 (~153 iters) - a 2.6x speedup over the
    # 395 at wilkinson_c=4.4, guarded here by max_outer. See Appendix C.8/C.3.
    case = V8MixedFlow()
    r = case.evaluate(n_sl=case.n_sl_rep, fidelity=FidelityConfig.tier3())
    assert r.converged                               # inside default max_outer
    assert r.result.record.n_iterations < 250        # ~153; guards the speedup
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
    assert r.validity > 0.0
    r2 = case.evaluate(n_sl=case.n_sl_rep, fidelity=FidelityConfig.tier2())
    assert r2.converged
    assert r.pressure_ratio == pytest.approx(r2.pressure_ratio, rel=5e-2)


def test_tier3_pocket_widened_by_wilkinson_c():
    # POCKET WIDENING (2026-07-19): the C.3-grounded wilkinson_c=13 lifts
    # mdot=15 from slow-max-iter (at the duct default 4.4) into the converging
    # window (measured 264 iters), so the pocket is {13, 14, 15} not {13, 14}.
    # Control: at the duct default 4.4, mdot=15 does NOT converge within the
    # same budget - the override is load-bearing. (mdot 12 stays a capacity
    # CHOKE fold and 16+ the upper feasibility edge; only 13-15 lift.)
    case = V8MixedFlow(mdot=15.0)
    cfg = ClassicalConfig(max_outer=400, wilkinson_c=case.wilkinson_c)
    r = case.machine().evaluate(MassFlowSpec(15.0), FidelityConfig.tier3(),
                                n_sl=case.n_sl_rep, config=cfg)
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi

    ctrl = case.machine().evaluate(
        MassFlowSpec(15.0), FidelityConfig.tier3(), n_sl=case.n_sl_rep,
        config=ClassicalConfig(max_outer=400, wilkinson_c=4.4))
    assert not ctrl.converged                        # duct default can't
