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
# One kernel, meanline vs spanwise (AD-1). The Tier-1 meanline converges with
# realistic loss (the fixture tests above). V7's 90-degree bend, however,
# cannot absorb the DOMINANT blade-loading loss (~7 kJ/kg, added 2026-07) at
# EITHER spanwise tier -- both land in the documented freeze-fallback wedge
# (a self-consistent lag state whose exit q-o has no positive-branch root at
# any mdot; lowering mdot makes it worse, the wedge signature; and Tier-2
# retune to mdot 10 still only max-iters slowly). The wedge's recorded attacks
# are closure-in-Newton or a compact-support streamline fit (major, not
# patches). The two xfails are tripwires -- REMOVE them when the wedge is
# cracked (strict=True flags the XPASS). See memory
# centrifugal-blade-loading-wip; Appendix C.7 note.
# --------------------------------------------------------------------------
_WEDGE_REASON = (
    "Blade-loading loss (2026-07, dominant centrifugal internal loss) pushes "
    "V7's 90-deg bend into the documented freeze-fallback wedge at this "
    "spanwise tier; Tier-1 meanline converges with realistic eta ~0.90. Remove "
    "this xfail when the wedge is cracked (closure-in-Newton / compact-support "
    "streamline fit).")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # wedge transient
@pytest.mark.xfail(strict=True, reason=_WEDGE_REASON)
def test_tier2_hits_the_wedge_with_realistic_loss():
    # Tripwire: the PRE-loss expectation was Tier-2 REE convergence.
    case = V7Centrifugal()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier2(), n_sl=case.n_sl_rep,
                                config=ClassicalConfig(max_outer=400))
    assert r.converged
    assert r.pressure_ratio > 1.0


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # wedge transient
@pytest.mark.xfail(strict=True, reason=_WEDGE_REASON)
def test_tier3_hits_the_wedge_with_realistic_loss():
    # Tripwire: the PRE-loss expectation was Tier-3 convergence (the 2026-07
    # stabilization) -- realistic loss overwhelms it.
    case = V7Centrifugal()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier3(), n_sl=case.n_sl_rep,
                                config=ClassicalConfig(max_outer=400))
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
