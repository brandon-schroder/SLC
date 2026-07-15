"""NASA Rotor 37 point-by-point validation gate (section 9.5, V5).

The first digitised-NASA-rotor validation in the suite (model-readiness
gate #1): geometry transcribed from NASA TP-1659 Table III(a), measured
100%-speed rotor line from Table IV(a) (see slcflow/verification/v5_rotor37.py
and docs/references/ROTOR37.md for provenance).

MEASURED AGREEMENT (2026-07-15, post-reference-calibration closures), pinned
here so drift is visible — these are recorded findings, not success claims:

  * Both tiers converge on the faithful geometry (meanline and spanwise REE)
    across the measured flow range — the structural result.
  * At the measured peak-efficiency point (20.74 kg/s): Tier-1 PR 2.38 vs
    measured 2.056 (+16%), eta 0.872 vs 0.876; Tier-2 (n_sl=5) PR 2.31
    (+12%), eta 0.864. Efficiency is close; PR is systematically HIGH.
  * Decomposition (probed): ~5 points of the PR excess is zero-blockage
    modelling (a uniform 4% blockage drops Tier-2 PR 2.31 -> 2.20); the
    remaining ~7 points is Euler over-work from the Lieblein NACA-65
    deviation under-predicting (~3.5 deg at mid-span) on Rotor 37's MCA
    transonic sections — beta2_flow 44.2 deg vs the report's design-intent
    47.7 deg at mid-span. Applying a 65-series subsonic deviation
    correlation to an MCA transonic rotor is out-of-pedigree by design;
    this measures the gap.
  * Closure validity reads 0 at all measured points: the equivalent-
    diffusion factor sits at/above the SP-36 window ceiling (D_eq ~ 2.0 at
    the meanline) — Rotor 37's loading is outside the Lieblein calibration
    window, so the loss is ceiling-saturated (the efficiency agreement is
    partly that ceiling, not a validated loss level).
  * The code speedline is much shallower than measured (PR 2.36..2.49 over
    the measured 1.785..2.196): the measured choke-side PR collapse (shock/
    choking-dominated) is not captured by the subsonic off-design bucket.
    Only the SIGN of the slope is pinned.

Calibrating the deviation/loss set against this dataset is model-readiness
gate #2 work — the point of landing the case is that the gap is now a
measured number instead of a [VERIFY].
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
    # docstring); eta within 3 points.
    r = case.evaluate(n_sl=1)
    assert r.converged
    assert r.pressure_ratio == pytest.approx(2.38, abs=0.12)
    assert abs(r.efficiency - 0.876) < 0.03
    # Loading sits outside the SP-36 window on this rotor (documented):
    assert r.validity < 0.5


def test_tier2_spanwise_converges_on_faithful_geometry(case):
    # The spanwise-REE tier on the digitised twisted geometry (the first
    # real-rotor spanwise run): converges, mass-averaged PR/eta at the
    # measured-agreement level (PR +12%, eta -1.2 pt).
    r = case.evaluate(n_sl=5, fidelity=FidelityConfig.tier2())
    assert r.converged
    assert r.pressure_ratio == pytest.approx(2.31, abs=0.12)
    assert abs(r.efficiency - MEASURED_100["rotor_eta"][2]) < 0.04


def test_speedline_slope_sign_matches_measured(case):
    # Across the measured 100%-speed flow range the code PR must FALL with
    # rising mdot (the measured line does, steeply). Level is not pinned —
    # the shallow-slope gap is the recorded finding.
    lo = case.evaluate(n_sl=1, mdot=float(MEASURED_100["mdot"][-1]))
    hi = case.evaluate(n_sl=1, mdot=float(MEASURED_100["mdot"][0]))
    assert lo.converged and hi.converged
    assert lo.pressure_ratio > hi.pressure_ratio


def test_design_intent_record_matches_report(case):
    # Table I anchors used by the docs (guards the transcription record).
    assert DESIGN["rotor_pr"] == 2.106
    assert DESIGN["mdot"] == 20.188
    assert MEASURED_100["rotor_pr"][2] == 2.056
    assert MEASURED_100["rotor_eta"][2] == 0.876
