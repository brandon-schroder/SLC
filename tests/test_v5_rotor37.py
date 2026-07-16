"""NASA Rotor 37 point-by-point validation gate (section 9.5, V5).

The first digitised-NASA-rotor validation in the suite (model-readiness
gate #1): geometry transcribed from NASA TP-1659 Table III(a), measured
100%-speed rotor line from Table IV(a) (see slcflow/verification/v5_rotor37.py
and docs/references/ROTOR37.md for provenance).

MEASURED AGREEMENT, pinned here so drift is visible — recorded findings,
not success claims. Two epochs:

2026-07-15 (uncorrected NACA-65 deviation): Tier-1 PR +16% / Tier-2 +12%
vs the measured 2.056, decomposed by probe into ~5 pts zero-blockage
modelling + ~7 pts Lieblein deviation under-prediction on the MCA
transonic sections (raw per-span gap: mean -3.6 deg, RMS 3.8 —
test_measured_deviation_gap_on_mca_sections keeps that record); closure
validity 0 (D_eq at/above the SP-36 ceiling).

2026-07-16 (Cetin AGARD-R-745 Eq 3.5 transonic deviation correction ON —
the library-grounded correction for exactly this blade family, applied
as published with no locally fitted constant):

  * Per-span deviation error: RMS 3.8 -> 1.2 deg, mean +-0
    (test_cetin_corrected_deviation_matches_measured).
  * Peak-eta point (20.74 kg/s, measured PR 2.056 / eta 0.876):
    Tier-1 PR 2.135 (+3.8%), eta 0.864; Tier-2 (n_sl=5) PR 2.051
    (+0.2% — on the measured value), eta 0.854.
  * Closure validity rose 0 -> ~0.8 at Tier 1 (the corrected work state
    pulls D_eq toward the window); Tier-2 spanwise ends still read 0.
  * The speedline remains shallower than measured on the choke side
    (2.11 vs 1.785 at 20.93 kg/s): the choke-side collapse is shock/
    choking physics outside the subsonic off-design bucket (Swan's
    M1-dependent off-design deviation rule from AGARD-R-745 is the
    recorded next lever). Slope sign pinned only.

Remaining known modelling gaps: blockage schedule, choke-side physics.
"""
import numpy as np
import pytest

from slcflow.machine import FidelityConfig
from slcflow.verification.v5_rotor37 import DESIGN, MEASURED_100, Rotor37


@pytest.fixture(scope="module")
def case():
    return Rotor37()


def test_transcribed_geometry_anchors(case):
    # Transcription guards (section 4.1 contract vs TP-1659 Table III(a)):
    # tip LE radius 25.230 cm, hub/tip 17.780/25.230 = 0.7048 (Table I: 0.70),
    # mid-span solidity 1.471, tip metal angles KIC 62.53 / KOC 49.98 deg.
    g = case._geometry()
    assert case.omega == pytest.approx(1800.0, abs=0.1)
    assert float(g.solidity(0.5)) == pytest.approx(1.471, abs=0.01)
    assert np.degrees(float(g.beta1_blade(1.0))) == pytest.approx(-62.53,
                                                                  abs=0.05)
    assert np.degrees(float(g.beta2_blade(1.0))) == pytest.approx(-49.98,
                                                                  abs=0.05)
    assert np.degrees(float(g.beta1_blade(0.0))) == pytest.approx(-52.04,
                                                                  abs=0.05)
    hub, tip = case._walls()
    assert tip[1][1] == pytest.approx(0.25230, abs=1e-5)
    assert hub[1][1] / tip[1][1] == pytest.approx(0.7048, abs=0.001)


def test_meanline_converges_at_measured_peak_eta_point(case):
    # Structural gate + measured-agreement pin at 20.74 kg/s (reading 4182:
    # PR 2.056, eta 0.876). PR reads +16% (recorded finding, see module
    # docstring); eta within 3 points. Cetin correction ON (case default):
    # PR 2.135 vs measured 2.056 (+3.8%; was +16% uncorrected).
    r = case.evaluate(n_sl=1)
    assert r.converged
    assert r.pressure_ratio == pytest.approx(2.135, abs=0.09)
    assert abs(r.efficiency - 0.876) < 0.03
    # The corrected work state pulls D_eq toward the SP-36 window
    # (validity ~0.8; it read 0 uncorrected):
    assert r.validity > 0.5


def test_tier2_spanwise_converges_on_faithful_geometry(case):
    # The spanwise-REE tier on the digitised twisted geometry: with the
    # Cetin correction the mass-averaged PR lands ON the measured value
    # (2.051 vs 2.056; was +12% uncorrected). Eta reads -2.2 pt.
    r = case.evaluate(n_sl=5, fidelity=FidelityConfig.tier2())
    assert r.converged
    assert r.pressure_ratio == pytest.approx(MEASURED_100["rotor_pr"][2],
                                             abs=0.09)
    assert abs(r.efficiency - MEASURED_100["rotor_eta"][2]) < 0.04


def test_speedline_slope_sign_matches_measured(case):
    # Across the measured 100%-speed flow range the code PR must FALL with
    # rising mdot (the measured line does, steeply). Level is not pinned —
    # the shallow-slope gap is the recorded finding.
    lo = case.evaluate(n_sl=1, mdot=float(MEASURED_100["mdot"][-1]))
    hi = case.evaluate(n_sl=1, mdot=float(MEASURED_100["mdot"][0]))
    assert lo.converged and hi.converged
    assert lo.pressure_ratio > hi.pressure_ratio


def test_measured_deviation_gap_on_mca_sections(case):
    # Table V(c) reading 4182: the MEASURED per-span deviation vs the
    # Lieblein chain evaluated at the measured incidence and the
    # transcribed blade-element geometry (section 4.3). MEASURED GAP
    # (2026-07-16): the NACA-65 correlation UNDER-predicts deviation on
    # Rotor 37's MCA transonic sections by mean -3.6 deg / RMS 3.8 deg
    # (-1.6 at 70% span, worst -5.5 near the tip; endwall stations carry
    # secondary/leakage contamination in the measured value). This is the
    # quantified calibration target behind the +7-point PR excess. This
    # test KEEPS the uncorrected record (the raw chain, correction off) --
    # the corrected chain is pinned by
    # test_cetin_corrected_deviation_matches_measured below.
    import numpy as np

    from slcflow.closures.axial_compressor.lieblein import (
        deviation_slope, reference_deviation, reference_incidence)
    from slcflow.verification.v5_rotor37 import (_CHORD_CM, _KIC_DEG,
                                                 _KOC_DEG, _PCT_SPAN,
                                                 _SOLIDITY, _TM_CM,
                                                 MEASURED_BE_4182 as BE)

    geo_idx = [int(np.where(_PCT_SPAN == p)[0][0])
               for p in BE["pct_span_from_tip"]]
    errs = []
    for k, gi in enumerate(geo_idx):
        camber = _KIC_DEG[gi] - _KOC_DEG[gi]
        b1f = float(BE["beta1_rel"][k])
        i = b1f - _KIC_DEG[gi]
        sol = _SOLIDITY[gi]
        tc = _TM_CM[gi] / _CHORD_CM[gi]
        i_ref, _ = reference_incidence(b1f, sol, tc, camber)
        d_ref, _ = reference_deviation(b1f, sol, tc, camber)
        dev = float(d_ref) + float(deviation_slope(b1f, sol)) \
            * (i - float(i_ref))
        errs.append(dev - float(BE["deviation"][k]))
    e = np.asarray(errs)
    assert -5.0 < float(e.mean()) < -1.5
    assert float(np.sqrt(np.mean(e * e))) < 5.0


def test_cetin_corrected_deviation_matches_measured(case):
    # The same per-span comparison with the Cetin AGARD-R-745 Eq 3.5
    # correction applied to the reference deviation (section 4.3, as in
    # LieblienSwirl with transonic_correction="cetin_agard745"): the
    # AS-PUBLISHED polynomial (no locally fitted constant) takes the error
    # from RMS 3.8 deg to 1.2, mean ~0 (measured 2026-07-16). Pinned:
    # RMS < 1.8 deg, |mean| < 0.6.
    import numpy as np

    from slcflow.closures.axial_compressor.lieblein import (
        cetin_deviation_correction, deviation_slope, reference_deviation,
        reference_incidence)
    from slcflow.verification.v5_rotor37 import (_CHORD_CM, _KIC_DEG,
                                                 _KOC_DEG, _PCT_SPAN,
                                                 _SOLIDITY, _TM_CM,
                                                 MEASURED_BE_4182 as BE)

    geo_idx = [int(np.where(_PCT_SPAN == p)[0][0])
               for p in BE["pct_span_from_tip"]]
    errs = []
    for k, gi in enumerate(geo_idx):
        camber = _KIC_DEG[gi] - _KOC_DEG[gi]
        b1f = float(BE["beta1_rel"][k])
        i = b1f - _KIC_DEG[gi]
        sol = _SOLIDITY[gi]
        tc = _TM_CM[gi] / _CHORD_CM[gi]
        i_ref, _ = reference_incidence(b1f, sol, tc, camber)
        d_ref, _ = reference_deviation(b1f, sol, tc, camber)
        d_cor, v = cetin_deviation_correction(float(d_ref))
        # Inside (or, at the 95%-span station, exactly ON) the fitted
        # branch: d_ref there is 7.51 deg, the window knee (v ~ 0.5).
        assert float(v) > 0.25
        dev = float(d_cor) + float(deviation_slope(b1f, sol)) \
            * (i - float(i_ref))
        errs.append(dev - float(BE["deviation"][k]))
    e = np.asarray(errs)
    assert abs(float(e.mean())) < 0.6
    assert float(np.sqrt(np.mean(e * e))) < 1.8


def test_swan_offdesign_rule_runs_but_is_not_adopted(case):
    # Swan Eq. 70 (AGARD-R-745) IMPLEMENTED + MEASURED on this case,
    # NOT adopted as the default (2026-07-16): across the measured line
    # the incidence stays within ~3 deg of reference, so off-design
    # deviation is small under either rule — Swan shifts PR a uniform
    # ~+0.03 (its negative Mach bracket at M1~1.4 cuts deviation) and
    # does NOT steepen the choke side; that collapse is loss/choking
    # physics. This pins (a) the opt-in path solves end-to-end, (b) the
    # measured shift stays small — if either changes, re-measure adoption.
    from slcflow.verification.v5_rotor37 import Rotor37
    swan = Rotor37(offdesign_rule="swan_agard745")
    r = swan.evaluate(n_sl=1)
    base = case.evaluate(n_sl=1)
    assert r.converged
    assert abs(r.pressure_ratio - base.pressure_ratio) < 0.08
    assert r.pressure_ratio == pytest.approx(2.165, abs=0.09)


def test_design_intent_record_matches_report(case):
    # Table I anchors used by the docs (guards the transcription record).
    assert DESIGN["rotor_pr"] == 2.106
    assert DESIGN["mdot"] == 20.188
    assert MEASURED_100["rotor_pr"][2] == 2.056
    assert MEASURED_100["rotor_eta"][2] == 0.876
