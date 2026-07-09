"""Continuation / map driver tests (Theory Manual sections 6.6, 6.7;
ARCH-5.4; M5-2).

Binds the speedline traversal mechanics on a duct (deterministic: point
ordering, choke-margin trend, warm-start answer-invariance, config guards),
the real rising-then-stalling characteristic on the V5 meanline rotor (the
section 6.7 stall flag with its recorded criterion -- validity_saturated for
the Lieblein family, with pr_turnover unit-covered on _classify_stall), and
the classical->Newton escalation at the point-solver level.

Provenance: M5 sub-step 2, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.drivers import (BCSwitchConfig, ClassicalConfig, MapResult,
                             SpeedlineConfig, solve_classical, solve_speedline)
from slcflow.drivers.classical import RowSpec
from slcflow.drivers.continuation import (_BACKPRESSURE, _NORMAL, _classify_stall,
                                          _next_mode, _solve_point)
from slcflow.errors import ConfigError
from slcflow.closures.axial_compressor import LIEBLEIN_NACA65
from slcflow.geometry.bladerow import ParamRowGeometry
from slcflow.grid import GridTopology
from slcflow.transport import TransportFields
from slcflow.types import FidelityConfig, MassFlowSpec
from slcflow.verification.v1_analytic_ree import V1FreeVortex, annulus_topology
from slcflow.verification.v5_axial_compressor import V5AxialRotor

_DEG = np.pi / 180.0


def _duct():
    case = V1FreeVortex.compressible()
    topo = annulus_topology(case.r0, case.r1, case.length, 9, case.n_stations)
    inlet = TransportFields(h0=np.full(9, case.h0), s=np.full(9, case.s),
                            rvt=np.full(9, case.rvt))
    return case, topo, inlet


def _v5_meanline():
    case = V5AxialRotor()
    topo = GridTopology(case._flowpath(), n_sl=1)
    inlet = TransportFields(h0=np.full(1, case.h0_in), s=np.full(1, case.s_in),
                            rvt=np.zeros(1))
    geom = ParamRowGeometry(blade_count=31, beta1=case.beta1_blade_deg * _DEG,
                            beta2=case.beta2_blade_deg * _DEG, chord_len=0.06,
                            solidity_val=1.2, thickness=0.08)
    row = RowSpec(row_id="r1", omega=case.omega, swirl=LIEBLEIN_NACA65.swirl,
                  loss=LIEBLEIN_NACA65.loss, blade_count=31, geometry=geom)
    return case, topo, inlet, row


# --------------------------------------------------------------------------
# Section 6.7 traversal mechanics (duct: deterministic, no work)
# --------------------------------------------------------------------------
def test_duct_speedline_orders_choke_to_stall_with_margin_trend():
    case, topo, inlet = _duct()
    m = solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet,
                        mdot_start=150.0, mdot_min=110.0, mdot_step=10.0)
    assert isinstance(m, MapResult)
    assert m.stall is None                       # duct never stalls
    mdots = [p.mdot for p in m.points]
    assert mdots == sorted(mdots, reverse=True)  # choke -> stall (descending)
    assert all(p.result.converged for p in m.points)
    # Choke margin grows as mdot falls away from choke (section 6.6).
    margins = [p.choke_margin for p in m.points]
    assert margins == sorted(margins)
    assert all(0.0 < c < 1.0 for c in margins)


def test_warm_start_does_not_change_the_answer():
    # A speedline point (warm-started from its neighbour) must match an
    # independent cold classical solve at the same mdot: warm start is a
    # convergence accelerant, never a different fixed point.
    case, topo, inlet = _duct()
    m = solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet,
                        mdot_start=150.0, mdot_min=120.0, mdot_step=10.0)
    p = m.points[-1]                              # a warm-started point
    cold = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(p.mdot), inlet)
    np.testing.assert_allclose(p.result.x, cold.x, atol=1e-6)


def test_speedline_is_deterministic():
    case, topo, inlet = _duct()
    kw = dict(mdot_start=150.0, mdot_min=120.0, mdot_step=10.0)
    a = solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet, **kw)
    b = solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet, **kw)
    assert [p.mdot for p in a.points] == [p.mdot for p in b.points]
    np.testing.assert_array_equal(a.points[-1].result.x,
                                  b.points[-1].result.x)


def test_speedline_config_guards():
    case, topo, inlet = _duct()
    with pytest.raises(ConfigError, match="mdot_start > mdot_min"):
        solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet,
                        mdot_start=100.0, mdot_min=120.0, mdot_step=5.0)
    with pytest.raises(ConfigError, match="mdot_step"):
        solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet,
                        mdot_start=150.0, mdot_min=110.0, mdot_step=100.0)


# --------------------------------------------------------------------------
# Section 6.7 characteristic + stall flag (V5 meanline rotor)
# --------------------------------------------------------------------------
def test_v5_meanline_characteristic_rises_then_flags_stall():
    case, topo, inlet, row = _v5_meanline()
    # mdot_start below the retuned annulus's choke (~118); the traversal runs
    # in-window down to ~50 kg/s (the Aungier off-design model keeps the loss
    # in-domain far further than the old pessimistic bucket), then saturates
    # near stall (~45 kg/s).
    m = solve_speedline(topo, case.gas, FidelityConfig.tier1(), inlet,
                        rows=[row], mdot_start=110.0, mdot_min=40.0,
                        mdot_step=10.0)
    # Real compressor characteristic: PR rises monotonically toward stall...
    prs = [p.pressure_ratio for p in m.points]
    assert len(prs) >= 3
    assert all(b > a for a, b in zip(prs, prs[1:]))   # strictly rising
    assert all(p.pressure_ratio > 1.0 for p in m.points)
    # ...and the traversal ends by REPORTING stall onset, not solving through.
    # For the Lieblein rotor the correlation's validity collapses to 0 as
    # incidence climbs toward stall BEFORE the (now correctly lower, post-
    # omega_bar fix) loss can turn the PR over -- so this family flags
    # ``validity_saturated``. (Before the omega_bar inversion was fixed the
    # inflated loss produced a spurious pr_turnover; pr_turnover coverage is
    # preserved as a unit test on ``_classify_stall`` below.)
    assert m.stall is not None
    assert m.stall.criterion == "validity_saturated"
    assert "validity" in m.stall.detail
    # The last accepted point is the peak (highest PR on the line).
    assert m.points[-1].pressure_ratio == pytest.approx(m.peak_pressure_ratio)


def test_classify_stall_pr_turnover_and_priority():
    # Unit coverage of the section 6.7 classifier's pr_turnover branch, which
    # is not reachable end-to-end for the Lieblein family (validity saturates
    # first) but IS the correct flag for a correlation that stays valid across
    # the peak. Priority is normative: validity is checked before turnover.
    cfg = SpeedlineConfig()          # validity_min=0.1
    # Armed, valid, PR fell below its predecessor -> pr_turnover.
    f = _classify_stall(out_mdot=90.0, pr=1.40, prev_pr=1.42, validity=0.9,
                        armed=True, peak_mdot=100.0, config=cfg)
    assert f is not None and f.criterion == "pr_turnover"
    assert "peak" in f.detail
    # Not armed yet (still on the rising branch): no flag despite pr<prev.
    assert _classify_stall(90.0, 1.40, 1.42, 0.9, False, 100.0, cfg) is None
    # Rising and valid: no flag.
    assert _classify_stall(90.0, 1.45, 1.42, 0.9, True, 100.0, cfg) is None
    # Validity wins over a simultaneous turnover (order is normative).
    f = _classify_stall(90.0, 1.40, 1.42, 0.0, True, 100.0, cfg)
    assert f is not None and f.criterion == "validity_saturated"


# --------------------------------------------------------------------------
# Section 6.7 escalation + failure flagging
# --------------------------------------------------------------------------
def test_point_solver_escalates_classical_to_newton():
    # A classical budget too small to converge forces escalation: Newton,
    # warm-started from the previous converged point, recovers the point.
    case, topo, inlet = _duct()
    warm = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(150.0), inlet)
    assert warm.converged
    cfg = SpeedlineConfig(classical=ClassicalConfig(max_outer=1))  # cripple
    res, driver = _solve_point(
        MassFlowSpec(145.0), warm, topology=topo, fluid=case.gas,
        fidelity=FidelityConfig.tier2(), inlet=inlet, rows=(), steps=None,
        blockage=None, metrics_config=None, config=cfg)
    assert driver == "newton" and res.converged


# --------------------------------------------------------------------------
# Section 6.6 hysteretic BC-switch
# --------------------------------------------------------------------------
def test_next_mode_hysteresis_is_automatic_and_banded():
    # The core section 6.6 requirement, as a pure decision: switch to
    # back-pressure below c_sw, and back to normal only past c_sw + delta_hys
    # (no switch-back inside the band -> no limit cycling).
    cfg = BCSwitchConfig(c_sw=0.05, delta_hys=0.03)
    assert _next_mode(_NORMAL, 0.10, cfg)[:2] == (_NORMAL, False)
    assert _next_mode(_NORMAL, 0.04, cfg)[:2] == (_BACKPRESSURE, True)
    # In back-pressure mode: stay until margin clears the hysteresis band.
    assert _next_mode(_BACKPRESSURE, 0.06, cfg)[:2] == (_BACKPRESSURE, False)
    assert _next_mode(_BACKPRESSURE, 0.09, cfg)[:2] == (_NORMAL, True)


def test_bc_switch_traverses_choke_proximal_region_and_recovers():
    # Starting near choke (small margin), the driver switches to the
    # back-pressure branch, throttles until the margin clears the hysteresis
    # band, switches back, and continues -- instead of dead-ending. The whole
    # excursion is logged (section 6.6: automatic + logged).
    case, topo, inlet = _duct()
    cfg = SpeedlineConfig(bc_switch=BCSwitchConfig(
        c_sw=0.10, delta_hys=0.05, bp_step_frac=0.02))
    m = solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet,
                        mdot_start=210.0, mdot_min=150.0, mdot_step=10.0,
                        config=cfg)
    modes = [p.mode for p in m.points]
    assert _BACKPRESSURE in modes and _NORMAL in modes    # both branches used
    # A switch out to back-pressure and a switch back to normal, both logged.
    assert [(s.from_mode, s.to_mode) for s in m.switches] == [
        (_NORMAL, _BACKPRESSURE), (_BACKPRESSURE, _NORMAL)]
    # Back-pressure points were solved for their (output) mdot by Newton.
    assert all(p.driver == "newton-bp"
               for p in m.points if p.mode == _BACKPRESSURE)
    # Achieved mdot falls monotonically across the whole excursion (progress
    # toward stall is unbroken by the mode change).
    mdots = [p.mdot for p in m.points]
    assert all(b < a for a, b in zip(mdots, mdots[1:]))
    # The margin recovered above the hysteresis ceiling before switching back.
    assert m.switches[1].mdot > 0.0
    assert m.stall is None


def test_cold_first_point_failure_flags_solver_failure():
    # mdot_start above the annulus capacity: the cold first point cannot
    # converge and there is no seed to escalate from -> solver_failure flag.
    case, topo, inlet = _duct()
    m = solve_speedline(topo, case.gas, FidelityConfig.tier2(), inlet,
                        mdot_start=380.0, mdot_min=120.0, mdot_step=20.0)
    assert m.converged_points == 0
    assert m.stall is not None
    assert m.stall.criterion == "solver_failure"
    assert "cold first point" in m.stall.detail
