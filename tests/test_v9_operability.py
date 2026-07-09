"""V9 operability regression (Theory Manual sections 6.6, 6.7, 9 item 9;
ARCH-5.4; M5-4).

Two behaviours, on the two cases each is well-posed for (see the module
docstring for why the split is honest rather than a shortcut):

  * surge-flag behaviour on the V5 rotor operating line (structural; the
    reported-surge-line match is [VERIFY]);
  * stable BC-switching across the choke-proximal region on the swirling-duct
    testbed (the switch machinery is case-independent; the V5 choke-knee
    traversal itself is [VERIFY], blocked on M6 shock-loss closures).

Provenance: M5 sub-step 4, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.drivers import BCSwitchConfig, SpeedlineConfig
from slcflow.drivers.continuation import _BACKPRESSURE, _NORMAL
from slcflow.verification.v9_operability import V9Operability


# --------------------------------------------------------------------------
# Surge-flag behaviour (V5 rotor operating line)
# --------------------------------------------------------------------------
def test_v9_v5_operating_line_reports_margin_and_flags_surge():
    line = V9Operability.v5_rotor().operating_line(
        mdot_start=110.0, mdot_min=40.0, mdot_step=10.0)
    # A rising characteristic that ends by REPORTING stall onset, with the
    # firing criterion recorded (section 6.7). For the Lieblein rotor the
    # correlation validity collapses to 0 as incidence climbs toward stall
    # before the (correctly lower, post-omega_bar-fix) loss turns PR over, so
    # the recorded criterion is ``validity_saturated``. [VERIFY vs a reported
    # surge line — blocked on the reference data, as for V5.]
    prs = [p.pressure_ratio for p in line.points]
    assert len(prs) >= 3 and all(b > a for a, b in zip(prs, prs[1:]))
    assert line.stall is not None
    assert line.stall.criterion == "validity_saturated"
    assert "validity" in line.stall.detail
    # Choke margin is reported per point and grows away from choke (falling
    # mdot => larger margin), section 6.6.
    margins = [p.choke_margin for p in line.points]
    assert margins == sorted(margins)
    assert all(0.0 < c < 1.0 for c in margins)


# --------------------------------------------------------------------------
# Stable BC-switching across choke (well-posed testbed)
# --------------------------------------------------------------------------
def test_v9_bc_switch_is_stable_across_choke_proximal():
    tb = V9Operability.bc_switch_testbed()
    cfg = SpeedlineConfig(bc_switch=BCSwitchConfig(
        c_sw=0.10, delta_hys=0.05, bp_step_frac=0.02))
    line = tb.operating_line(mdot_start=210.0, mdot_min=150.0,
                             mdot_step=10.0, config=cfg)
    # Enters back-pressure mode near choke and returns to normal mode: exactly
    # one out-and-back, i.e. NO limit-cycling (section 6.6 hysteresis works).
    assert [(s.from_mode, s.to_mode) for s in line.switches] == [
        (_NORMAL, _BACKPRESSURE), (_BACKPRESSURE, _NORMAL)]
    assert {p.mode for p in line.points} == {_NORMAL, _BACKPRESSURE}
    # The excursion is stable and productive: achieved mdot falls monotonically
    # throughout (progress unbroken by the mode change) and it does not
    # dead-end at the choke boundary.
    mdots = [p.mdot for p in line.points]
    assert all(b < a for a, b in zip(mdots, mdots[1:]))
    assert line.stall is None
    # The switch back happened only after the margin cleared the hysteresis
    # band (section 6.6): the last back-pressure point's margin exceeds c_sw.
    bp_margins = [p.choke_margin for p in line.points if p.mode == _BACKPRESSURE]
    assert max(bp_margins) > 0.10


def test_v9_no_switch_without_a_policy():
    # Default (bc_switch=None): the testbed traverses in mass-flow mode only,
    # never switching — the switch is opt-in operability, not always-on.
    tb = V9Operability.bc_switch_testbed()
    line = tb.operating_line(mdot_start=200.0, mdot_min=150.0, mdot_step=10.0)
    assert line.switches == ()
    assert all(p.mode == _NORMAL for p in line.points)
