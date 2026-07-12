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
# One kernel, meanline vs spanwise (AD-1). The 2026-07 diagnosis (probe_cin_*,
# memory wedge-closure-in-newton) SEPARATED what had been called one "wedge"
# into two distinct diseases:
#
#   Tier 2 = an OPERATING-POINT stratification-capacity fold. The dominant
#   blade-loading loss (~7 kJ/kg) stratifies the exit profile and drives an
#   interior streamtube's Vm toward the master-ODE Vm=0 singularity, so the
#   COUPLED flow folds BELOW a mass-flow floor (~15 kg/s). It is NOT closure
#   coupling (fixed prescribed stratified transport folds identically) and NOT
#   the classical repositioning algorithm (global Newton folds too). Raising
#   mdot lifts every Vm off the singularity: at the re-centred mdot = 17 (the
#   case default) Tier 2 converges with realistic loss (below) -- so this is a
#   PASSING test now, the operating-point crack landed.
#
#   Tier 3 = a SEPARATE curvature-repositioning collapse on the 90-deg bend:
#   it fails at EVERY mdot (including the Tier-2-feasible window), dies early
#   (outer it 3-5), and the section 6.4 wilkinson_c throttle is inert -- the
#   standing "robust radial/mixed repositioning" open item, now isolated from
#   the operating-point confound. It stays an xfail tripwire; REMOVE it when a
#   robust Tier-3 repositioning lands (strict=True flags the XPASS). See memory
#   wedge-closure-in-newton; Appendix C.7.
# --------------------------------------------------------------------------
def test_tier2_converges_with_realistic_loss():
    # Operating-point crack (2026-07): at the re-centred mdot the Tier-2 REE
    # solve converges with the full realistic loss, validity 1, sane PR/eta.
    case = V7Centrifugal()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier2(), n_sl=case.n_sl_rep)
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
    elo, ehi = case.eta_band
    assert elo < r.efficiency < ehi
    assert r.validity > 0.0


_TIER3_REASON = (
    "Tier-3 curvature+lean streamline repositioning on the 90-deg bend "
    "collapses early (outer it 3-5) at EVERY mdot with the dominant "
    "blade-loading loss -- a SEPARATE failure from the Tier-2 operating-point "
    "wedge (which the mdot re-centre cracks; see test_tier2_converges...). This "
    "is the standing robust-radial/mixed-repositioning open item. Remove this "
    "xfail when a compact-support streamline fit / closure-in-repositioning "
    "lands (memory wedge-closure-in-newton, Appendix C.7).")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # repositioning transient
@pytest.mark.xfail(strict=True, reason=_TIER3_REASON)
def test_tier3_hits_the_repositioning_collapse():
    # Tripwire: Tier 3 fails at the re-centred (Tier-2-feasible) mdot too, so
    # this is the repositioning mode, NOT the operating-point fold.
    case = V7Centrifugal()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier3(), n_sl=case.n_sl_rep,
                                config=ClassicalConfig(max_outer=400))
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
