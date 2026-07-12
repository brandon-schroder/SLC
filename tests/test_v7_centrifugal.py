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
#   Tier 3 = a physical FEASIBILITY FOLD at realistic loss (diagnosed 2026-07,
#   probe_v7t3_*): NOT a repositioning-mechanism gap. A damped-Newton +
#   curvature-strength continuation showed the flow branch folds (interior
#   Vm -> 0, the master-ODE singularity) at only ~9% of the full Tier-3
#   curvature at mdot=17 (~26% at mdot=20). The tight 0.08 m bend (kappa~20)
#   plus the realistic-loss stratification drive an interior streamtube to
#   Vm -> 0 (incipient meridional reversal the inviscid model refuses). The
#   fold is mdot-liftable (like the Tier-2 wedge) but reaching full curvature
#   needs mdot ~ 32 >> the Tier-2 choke ceiling ~22 -- so NO non-choked mass
#   flow admits full Tier-3 radial equilibrium. There is no positive-Vm root,
#   so a stiff integrator / compact-support fit / damped Newton cannot help;
#   the "repositioning failed" symptom is downstream of the fold. The levers
#   are case-side: a calibrated/lower blade-loading loss ([VERIFY], likely
#   high), a gentler bend, or accepting beyond-model-validity at this loading.
#   Stays an xfail tripwire; REMOVE if a case-side change makes it feasible
#   (strict=True flags the XPASS). See memory v7-tier3-root-cause; Appendix C.7.
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
    "V7 Tier 3 at realistic loss is a physical FEASIBILITY FOLD, not a solver "
    "gap (diagnosed 2026-07, probe_v7t3_*): a damped-Newton + curvature-strength "
    "continuation showed the flow branch folds (interior Vm -> 0) at ~9% of full "
    "Tier-3 curvature at mdot=17 / ~26% at mdot=20; reaching full curvature needs "
    "mdot ~32 >> the Tier-2 choke ceiling ~22, so no non-choked mass flow admits "
    "full radial equilibrium on the tight 0.08 m bend (kappa~20) with this loss. "
    "No positive-Vm root exists -> stiff integrator / compact-support fit / "
    "damped Newton cannot help. The 'calibrated/lower loss' case-side lever was "
    "TRIED (2026-07-12: the Coppage/Oh-1997 D_f ratio fix cut the blade-loading "
    "loss ~2.3x): it EASED the fold (Tier 3 now fails at sane PR/eta ~2.3/0.9 "
    "rather than garbage) but did NOT crack it -- Tier 3 still fails at every "
    "mdot in 13..32. Remaining levers: a further-calibrated/lower loss, a gentler "
    "bend, or beyond-model validity. Remove this xfail if a case-side change makes "
    "it feasible (memory v7-tier3-root-cause, Appendix C.7).")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # fold transient
@pytest.mark.xfail(strict=True, reason=_TIER3_REASON)
def test_tier3_infeasible_fold_at_realistic_loss():
    # Tripwire: Tier 3 fails at the re-centred (Tier-2-feasible) mdot -- the
    # curvature x loss fold into Vm -> 0, not the operating-point wedge and not
    # a repositioning-mechanism gap (there is no positive-Vm root to reach).
    case = V7Centrifugal()
    r = case.machine().evaluate(MassFlowSpec(case.mdot),
                                FidelityConfig.tier3(), n_sl=case.n_sl_rep,
                                config=ClassicalConfig(max_outer=400))
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
