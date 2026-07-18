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


def test_tip_diffusion_factor_predicts_the_sibling_stall_differential():
    # GATE #5 FOLLOW-ON (2026-07-18): the Lieblein tip diffusion factor
    # (NACA RM E53D01) is the grounded loading criterion the operability
    # disposition called for. Characterized at Tier 2 (n_sl=5) across the
    # measured lines of BOTH transonic siblings.
    #
    # MEASURED: the tip D-factor at each rotor's measured stall flow is
    # ~0.6 (R37 0.63 at 19.60; R38 0.595 at 20.44) - ABOVE the report's
    # tip DESIGN limit 0.45 (eta=0.90) but at the 2-D-cascade
    # sharp-loss-rise value, i.e. the rigs run past design loading to
    # stall. A D_tip = 0.60 stall threshold crosses within ~3% of each
    # measured stall (R37 ~20.2, +3.0%; R38 ~20.3, -0.6%) AND ORDERS THE
    # SIBLINGS CORRECTLY: the high-AR R38 reaches the loading limit at
    # HIGHER flow -> stalls earlier, reproducing the measured differential
    # (20.44 > 19.60) that the LOSS set does NOT carry (test_tier2_and_the
    # _high_ar_differential: model PR +6.6% vs +0.2%). Loading, not loss,
    # is the right variable for the stall LINE. Zero tuning.
    import numpy as np

    from slcflow.drivers.classical import ClassicalConfig, solve_classical
    from slcflow.grid.core import GridTopology
    from slcflow.machine import FidelityConfig, MassFlowSpec
    from slcflow.verification.v5_rotor37 import (Rotor37, MEASURED_100,
                                                 tip_diffusion_factor)
    from slcflow.verification.v5_rotor38 import Rotor38, MEASURED_100_R38

    def d_tip_curve(case_cls):
        case = case_cls()
        m = case.machine()
        topo = GridTopology(m.flowpath, n_sl=5)
        inlet = m.inlet.fields(topo.psi)
        warm, rows = None, []
        for md in np.arange(21.0, 19.39, -0.2):
            r = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                                MassFlowSpec(md), inlet, rows=m.rows,
                                warm_start=warm,
                                config=ClassicalConfig(max_outer=800))
            if not r.converged:
                continue
            warm = r
            rows.append((float(md), tip_diffusion_factor(case, r)))
        a = np.array(rows)
        return a[np.argsort(a[:, 1])]      # ascending in D for np.interp

    def cross06(a):
        return float(np.interp(0.60, a[:, 1], a[:, 0]))

    def d_at(a, md):
        b = a[np.argsort(a[:, 0])]
        return float(np.interp(md, b[:, 0], b[:, 1]))

    c37 = d_tip_curve(Rotor37)
    c38 = d_tip_curve(Rotor38)
    stall37 = float(np.min(MEASURED_100["mdot"]))          # 19.60
    stall38 = float(np.min(MEASURED_100_R38["mdot"]))      # 20.44

    # D_tip at each measured stall sits at the ~0.6 sharp-loss value:
    assert 0.60 <= d_at(c37, stall37) <= 0.68
    assert 0.56 <= d_at(c38, stall38) <= 0.62

    # a D_tip=0.60 threshold predicts each measured stall within ~4%:
    x37, x38 = cross06(c37), cross06(c38)
    assert abs(x37 - stall37) / stall37 < 0.04
    assert abs(x38 - stall38) / stall38 < 0.04

    # and ORDERS the siblings the way the loss set could not: R38 (early
    # stall) reaches the loading limit at higher flow than R37:
    assert x38 > x37
    assert stall38 > stall37       # the measured differential it tracks
