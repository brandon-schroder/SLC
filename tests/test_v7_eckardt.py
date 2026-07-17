"""Eckardt rotor O point-by-point validation gate (section 9.7, V7).

Geometry-faithful-endpoints case from the PRIMARY paper (Eckardt 1976 via
the Test Cases notebook; slcflow/verification/v7_eckardt.py and
docs/references/ECKARDT.md for provenance and the modelling choices).

MEASURED AGREEMENT (2026-07-15, corrected blade-loading closures), pinned
here as recorded findings:

  * ALL THREE TIERS converge on the real geometry at the laser point with
    validity 1.0, agreeing with each other to ~0.1% — including Tier 3,
    which is INFEASIBLE (physical fold) on the synthetic V7 testbed. The
    real Eckardt bend (hub quarter-ellipse 0.130/0.155 m) is ~2.5x gentler
    than the testbed's 0.08 m arc: the fold was a property of that
    aggressive synthetic bend, not of radial machines.
  * Laser point (14 000 rpm, 5.31 kg/s; measured STAGE PR 2.1): impeller-
    exit PR 2.20 (+4.7%) — the right side and roughly the right size, since
    slcflow stops at the impeller exit and the vaneless diffuser loses a
    few % p0 before the stage measurement plane.
  * Design point (18 000 rpm, 7.16 kg/s; measured stage PR 3.0): PR 3.38
    (+12.6%) — the gap grows with speed, consistent with the deferred
    parasitic (disk/recirculation/leakage) + clearance + diffuser losses
    (measured stage eta ~0.856 here vs code impeller-internal 0.966).
  * Efficiency is NOT stage-comparable (documented): code ~0.97 is
    impeller-internal-only vs measured stage 0.88.
  * Slip data point: the measured (PR, eta) pair implies a work input ~3%
    ABOVE Wiesner's sigma = 1 - 1/20^0.7 = 0.877 (implied sigma ~0.90) —
    a calibration observation for the slip closure, recorded not tuned.
"""
import pytest

from slcflow.machine import FidelityConfig
from slcflow.verification.v7_eckardt import (DESIGN_POINT, LASER_POINT,
                                             EckardtO)


@pytest.fixture(scope="module")
def case():
    return EckardtO()


@pytest.fixture(scope="module")
def tier1_laser(case):
    return case.evaluate(n_sl=1)


def test_transcribed_geometry_anchors(case):
    # Primary-paper transcription guards (section 4.1 / ECKARDT.md):
    # r2 = 0.200 m, r1h/r1t = 0.045/0.140 m, b2 = 26 mm, Z = 20 radial.
    assert case.r2 == 0.200 and case.b2 == 0.026
    assert case.r1h == 0.045 and case.r1t == 0.140
    assert case.blade_count == 20
    g = case._geometry()
    assert float(g.beta2_blade(0.5)) == 0.0            # radially ending
    assert case.omega == pytest.approx(1466.1, abs=0.1)


def test_laser_point_meanline_pr_vs_measured_stage(tier1_laser):
    # Impeller-exit PR must sit AT-OR-ABOVE the measured stage PR (the
    # diffuser can only lose p0) and within the measured-agreement band
    # (+4.7% observed; bounded at +10%).
    r = tier1_laser
    assert r.converged and r.validity == pytest.approx(1.0, abs=1e-6)
    assert r.pressure_ratio >= LASER_POINT["stage_pr"] * 0.995
    assert r.pressure_ratio <= LASER_POINT["stage_pr"] * 1.10
    # Impeller-internal-only efficiency exceeds the stage measurement.
    assert LASER_POINT["stage_eta"] < r.efficiency < 0.995


def test_design_point_meanline_pr_vs_measured_stage(case):
    # 18 000 rpm design: +12.6% observed (unmodelled parasitic/clearance/
    # diffuser grow with speed) — bounded, documented, not tuned away.
    r = EckardtO(rpm=DESIGN_POINT["rpm"],
                 mdot=DESIGN_POINT["mdot"]).evaluate(n_sl=1)
    assert r.converged
    assert DESIGN_POINT["stage_pr"] <= r.pressure_ratio \
        <= DESIGN_POINT["stage_pr"] * 1.15


def test_krain_second_impeller_measured_agreement():
    # The SECOND centrifugal point (Krain 1988 / Krain-Hoffmann 1989 via
    # the Test Cases notebook): 30-deg backswept, 24 blades, PR-4.7-class
    # — twice Eckardt's loading, cross-checking Wiesner slip + the loss
    # set. MEASURED (2026-07-17, docs/references/ECKARDT.md "Krain"):
    #   * Tier-1 AND Tier-2 converge with validity 1.0, agreeing to 0.2%
    #     (impeller-exit PR ~5.00, internal eta 0.972).
    #   * Stage chain: PR_stage 4.714 vs the measured stage max ~4.5
    #     (+4.8%; design rotor PR_tt 4.7) — the PR side holds up.
    #   * eta_stage 0.905 vs measured stage 0.84: +6.5 pt — the loss set
    #     that CLOSES at Eckardt's PR 2.1 reads LIGHT at PR 4.7 (measured
    #     impeller eta_poly 0.95 ~ eta_is 0.938 vs internal 0.972, so
    #     ~3.4 pt of it is internal-loss level at high loading; the
    #     recirculation term floors to exactly 0 at design backsweep;
    #     clearance is an assumption). RECORDED trend finding, not tuned.
    from slcflow.machine import FidelityConfig
    from slcflow.verification.v7_eckardt import KRAIN_DESIGN, KrainImpeller
    case = KrainImpeller()
    r1 = case.evaluate(n_sl=1)
    assert r1.converged and r1.validity == pytest.approx(1.0, abs=1e-6)
    assert r1.pressure_ratio == pytest.approx(5.00, abs=0.15)
    r2 = case.evaluate(n_sl=7, fidelity=FidelityConfig.tier2())
    assert r2.converged
    assert r2.pressure_ratio == pytest.approx(r1.pressure_ratio, rel=0.02)
    sp = case.stage_performance(r1)
    assert sp["pr_stage"] == pytest.approx(4.714, abs=0.1)
    assert sp["pr_stage"] >= KRAIN_DESIGN["stage_pr_max"]  # +4.8% recorded
    assert sp["eta_stage"] == pytest.approx(0.905, abs=0.015)
    assert sp["eta_stage"] > KRAIN_DESIGN["stage_eta_max"]  # +6.5 pt gap
    # Design-backsweep recirculation floors to zero (the -2 cot term):
    assert case.parasitic_breakdown(r1)["recirculation"] == 0.0


def test_all_tiers_converge_and_agree_on_real_geometry(case, tier1_laser):
    # The headline structural finding: Tier 2 AND Tier 3 (which folds on
    # the synthetic V7 testbed bend) converge on the real Eckardt geometry
    # with validity 1, agreeing with the meanline to ~0.1% (pinned at 2%).
    for fid in (FidelityConfig.tier2(), FidelityConfig.tier3()):
        r = case.evaluate(n_sl=case.n_sl_rep, fidelity=fid)
        assert r.converged, fid
        assert r.validity == pytest.approx(1.0, abs=1e-6)
        assert r.pressure_ratio == pytest.approx(
            tier1_laser.pressure_ratio, rel=0.02)
