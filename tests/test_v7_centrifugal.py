"""V7 centrifugal-impeller structural tests (Theory Manual section 9.7;
Appendix C.7; M7 sub-step 4).

Structural (not quantitative) verification, mirroring V5/V6: the impeller
converges end-to-end at all three tiers on the phi -> 90 deg radial path, does
real centrifugal work with real loss, exits radially with sub-blade-speed
(slipped) swirl, and lands PR/efficiency in sane bands. Point-by-point Eckardt
reproduction is [VERIFY] (reference-library calibration + deferred loss).

Provenance: M7 sub-step 4, written with the V7 case.
"""
import numpy as np
import pytest

from slcflow.drivers.classical import ClassicalConfig
from slcflow.types import FidelityConfig, MassFlowSpec
from slcflow.verification.v7_centrifugal import V7Centrifugal


@pytest.fixture(scope="module")
def meanline():
    return V7Centrifugal().evaluate(n_sl=1)


# --------------------------------------------------------------------------
# Section 9.7: the impeller compresses (work in, PR > 1)
# --------------------------------------------------------------------------
def test_meanline_converges_and_compresses(meanline):
    r = meanline
    assert r.converged
    lo, hi = V7Centrifugal().pr_band
    assert lo < r.pressure_ratio < hi          # total-to-total PR > 1
    assert r.pressure_ratio > 1.0
    elo, ehi = V7Centrifugal().eta_band
    assert elo < r.efficiency < ehi
    assert r.validity > 0.0


def test_does_centrifugal_work_and_loss(meanline):
    tr = meanline.result.frozen.transported
    # Euler work raises h0 (dh0 > 0, a compressor), entropy rises (real loss).
    assert float(tr.h0[0, -1]) > float(tr.h0[0, 0])
    assert float(tr.s[0, -1]) > float(tr.s[0, 0])
    # Axial inflow (rVt = 0) is spun up to positive exit swirl (compression).
    assert float(tr.rvt[0, 0]) == pytest.approx(0.0, abs=1e-9)
    assert float(tr.rvt[0, -1]) > 0.0


def test_exit_is_radial_phi_90(meanline):
    # The exit (last) station is the radial line at r = r2: the parametric
    # phi -> 90 deg path (M1) carried end-to-end with a blade row.
    case = V7Centrifugal()
    r_exit = meanline.result.fields.metrics.r[:, -1]
    np.testing.assert_allclose(r_exit, case.r2, rtol=1e-6)


def test_exit_swirl_is_slipped(meanline):
    # Wiesner slip: exit tangential velocity is below blade speed U2 (sigma<1).
    case = V7Centrifugal()
    tr = meanline.result.frozen.transported
    u2 = case.omega * case.r2
    vtheta_exit = float(tr.rvt[0, -1]) / case.r2
    assert 0.0 < vtheta_exit < u2


def test_inblade_stations_ramp_the_work(meanline):
    # The 6 INBLADE stations distribute the rVt rise monotonically LE -> TE
    # (section 3.4 schedule); no station overshoots the TE value.
    tr = meanline.result.frozen.transported
    rvt = tr.rvt[0]
    # Columns: DUCT, LE, INBLADE x6, TE, DUCT (n_qo = 10).
    seg = rvt[1:9]                              # LE .. TE inclusive
    assert np.all(np.diff(seg) >= -1e-9)        # monotone non-decreasing
    assert float(seg[-1]) == pytest.approx(float(rvt[-1]), rel=1e-9)


# --------------------------------------------------------------------------
# One kernel, three tiers (AD-1): all converge and agree structurally
# --------------------------------------------------------------------------
def test_all_three_tiers_converge_and_agree():
    case = V7Centrifugal()
    m = case.machine()
    n = case.n_sl_rep
    specs = [("t1", FidelityConfig.tier1(), 1),
             ("t2", FidelityConfig.tier2(), n),
             ("t3", FidelityConfig.tier3(), n)]
    prs = []
    for _name, fid, nsl in specs:
        # Tier 3 measured at 197 iterations post-stabilization (the closure
        # switch-on ramp adds a few): give headroom over the default 200.
        res = m.evaluate(MassFlowSpec(case.mdot), fid, n_sl=nsl,
                         config=ClassicalConfig(max_outer=400))
        assert res.converged, f"{_name} did not converge"
        assert res.pressure_ratio > 1.0
        prs.append(res.pressure_ratio)
    # Tier consistency: the three PRs agree to a few percent (meanline vs.
    # spanwise-resolved + repositioning differ only at second order here).
    assert max(prs) - min(prs) < 0.05 * np.mean(prs)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # switch-on transient
def test_tier3_edge_only_converges_after_stabilization():
    # TRIPWIRE FLIPPED (2026-07): C.7's M7-4 finding -- that Tier-3
    # repositioning on the 90-degree bend REQUIRES the INBLADE subdivision
    # (edge-only "diverges the section 6.4 odd-even mode at any
    # relaxation") -- was refuted by the diagnosis: the edge-only failure
    # was the driver accepting a spurious negative-Vm continuity branch
    # (decreasing mass cumulative), the same artifact family as V8 Tier 3,
    # not a repositioning envelope. Post-stabilization the edge-only row
    # converges (measured 173 iterations) to the same PR as the subdivided
    # case to <1%. INBLADE stations remain the RESOLUTION choice for
    # in-blade quantities (sections 2.5, 4.5) -- they are just not a
    # convergence crutch anymore (Appendix C.7, revised).
    case = V7Centrifugal(n_inblade=0)
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier3(), n_sl=case.n_sl_rep,
                                config=ClassicalConfig(max_outer=400))
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
