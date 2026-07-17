"""NASA Rotor 38 second-point validation gate (section 9.5, V5).

The high-aspect-ratio sibling of Rotor 37 (TP-2001; same annulus/speed/
flow family, 48 short-chord blades vs 36) — the axial generalization test
of the Cetin-corrected set, the counterpart of the Krain check on the
centrifugal side (slcflow/verification/v5_rotor38.py for provenance).

MEASURED AGREEMENT (2026-07-17), pinned as recorded findings:

  * Both tiers converge across the measured line; validity 0 (same
    above-SP-36-window loading class as Rotor 37).
  * Efficiency LEVEL generalizes: near-peak (20.67 kg/s, measured
    0.849): Tier-1 0.860 (+1.1 pt), Tier-2 0.854 (+0.5 pt).
  * PR does NOT track the measured high-AR shortfall: at matched flow
    Tier-2 reads 2.098 vs measured 1.969 (+6.6%) where Rotor 37 read
    +0.2% — the rig's own summary shows the high-AR stage STALLED before
    design flow (peak eta at the minimum flow; never reached design PR
    2.105), an early-stall/endwall-sensitivity mechanism the correlation
    set does not carry (Howell's s/h scaling even moves slightly the
    wrong way with the 48-blade pitch; no part-span damper is mentioned
    in TP-2001). THE AXIAL TWO-POINT TREND FINDING, recorded not tuned.
    (The matched-mdot frame also sits near the vertical characteristic
    here, as for Rotor 37 — the differential +0.2% vs +6.6% between
    siblings is the frame-robust statement.)
"""
import numpy as np
import pytest

from slcflow.machine import FidelityConfig
from slcflow.verification.v5_rotor38 import (DESIGN_R38, MEASURED_100_R38,
                                             Rotor38)


@pytest.fixture(scope="module")
def case():
    return Rotor38()


def test_transcribed_geometry_anchors(case):
    g = case._geometry()
    assert case.BLADES == 48
    assert float(g.solidity(0.5)) == pytest.approx(1.481, abs=0.01)
    assert np.degrees(float(g.beta1_blade(1.0))) == pytest.approx(-62.69,
                                                                  abs=0.05)
    assert np.degrees(float(g.beta2_blade(0.0))) == pytest.approx(-17.69,
                                                                  abs=0.05)
    hub, tip = case._walls()
    assert tip[1][1] == pytest.approx(0.25283, abs=1e-5)


def test_meanline_measured_agreement(case):
    # Near-peak reading 4120 (20.67 kg/s; measured PR 1.969, eta 0.849):
    # eta +1.1 pt, PR +10.9% (matched-mdot near the vertical
    # characteristic; the DIFFERENTIAL vs Rotor 37 is the finding).
    r = case.evaluate(n_sl=1)
    assert r.converged
    assert r.pressure_ratio == pytest.approx(2.183, abs=0.1)
    assert abs(r.efficiency - MEASURED_100_R38["rotor_eta"][4]) < 0.02
    assert r.validity < 0.5      # above-window loading, as Rotor 37


def test_tier2_and_the_high_ar_differential(case):
    # Tier-2 PR 2.098 vs measured 1.969 (+6.6%) where Rotor 37's Tier-2
    # read +0.2% at its peak-eta point: the model does NOT carry the
    # measured high-AR degradation (early stall / endwall sensitivity).
    r = case.evaluate(n_sl=5, fidelity=FidelityConfig.tier2())
    assert r.converged
    assert r.pressure_ratio == pytest.approx(2.098, abs=0.1)
    assert abs(r.efficiency - MEASURED_100_R38["rotor_eta"][4]) < 0.015


def test_speedline_slope_sign(case):
    lo = case.evaluate(n_sl=1, mdot=float(MEASURED_100_R38["mdot"][-1]))
    hi = case.evaluate(n_sl=1, mdot=float(MEASURED_100_R38["mdot"][0]))
    assert lo.converged and hi.converged
    assert lo.pressure_ratio > hi.pressure_ratio


def test_design_record(case):
    assert DESIGN_R38["rotor_pr"] == 2.105
    assert DESIGN_R38["blades"] == 48
    # The rig never reached design PR at 100% (stall-truncated line):
    assert float(np.max(MEASURED_100_R38["rotor_pr"])) < DESIGN_R38[
        "rotor_pr"]
