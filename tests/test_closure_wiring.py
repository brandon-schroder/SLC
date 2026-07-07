"""Closure-fed blade-row wiring tests (M4 sub-step 1): Theory Manual
sections 3.3-3.5 through the closure path, 6.2.2.4 lagged evaluation (AD-4),
7.1-7.2 interfaces, and the section 6.2.5 closure-update norm.

Includes the rvt_le consistency test carried from the M2 transport review:
the driver must hand ``row_steps`` the SWEPT field arriving at EDGE_LE, so
the Euler-work h0 rise across the row is exactly omega * (rVt_TE - rVt_LE)
of the swept transported arrays.

Provenance: M4 sub-step 1, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures.interfaces import (LossBreakdown, LossModel,
                                         RowFlowView, RowView, SwirlClosure,
                                         SwirlResult)
from slcflow.closures.simple import PrescribedLoss, PrescribedSwirl
from slcflow.drivers import ClassicalConfig, RowSpec, solve_classical
from slcflow.errors import ConfigError
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import FlowPath, StationDef, StationType, WallCurve
from slcflow.grid import GridTopology
from slcflow.transport import TransportFields, TransportStep
from slcflow.types import FidelityConfig, MassFlowSpec

GAS = PerfectGas()
H0, S0 = 3.0e5, 0.0
R0, R1 = 0.3, 0.6


def rotor_topology(n_sl=9):
    """Cylinder annulus with one blade row: DUCT, LE, TE, DUCT stations."""
    z = np.linspace(0.0, 1.0, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R0)]))
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R1)]))
    stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                StationDef(StationType.EDGE_LE, 0.35, 0.35, row_id="r1"),
                StationDef(StationType.EDGE_TE, 0.55, 0.55, row_id="r1"),
                StationDef(StationType.DUCT, 1.0, 1.0)]
    return GridTopology(FlowPath(w0, w1, stations), n_sl=n_sl)


def solve_with_row(swirl, loss, omega=800.0, rvt_inlet=6.0, n_sl=9,
                   mdot=100.0, **kw):
    topo = rotor_topology(n_sl)
    inlet = TransportFields(h0=np.full(n_sl, H0), s=np.full(n_sl, S0),
                            rvt=np.full(n_sl, float(rvt_inlet)))
    row = RowSpec(row_id="r1", omega=omega, swirl=swirl, loss=loss)
    return topo, solve_classical(topo, GAS, FidelityConfig.tier2(),
                                 MassFlowSpec(mdot), inlet, rows=[row], **kw)


# --------------------------------------------------------------------------
# Section 7.1 interfaces: conformance (ARCH-7 protocol tests)
# --------------------------------------------------------------------------
def test_prescribed_closures_satisfy_protocols():
    assert isinstance(PrescribedSwirl(rvt=10.0), SwirlClosure)
    assert isinstance(PrescribedLoss(delta_s=1.0), LossModel)
    with pytest.raises(ConfigError):
        SwirlResult(rvt=1.0, validity=1.5)
    with pytest.raises(ConfigError):
        LossBreakdown(components={}, delta_s=0.0, validity=-0.1)


# --------------------------------------------------------------------------
# Sections 3.3-3.5 through the closure path + rvt_le consistency (carryover)
# --------------------------------------------------------------------------
def test_rotor_row_euler_work_and_entropy_via_closures():
    omega, rvt_in, rvt_out, ds = 800.0, 6.0, 20.0, 2.5
    topo, res = solve_with_row(PrescribedSwirl(rvt=rvt_out),
                               PrescribedLoss(delta_s=ds), omega=omega,
                               rvt_inlet=rvt_in)
    assert res.converged
    tr = res.frozen.transported
    # Section 3.4: TE rVt converges to the closure value to the closure-lag
    # tolerance (the section 6.2.4 under-relaxation now also ramps the FIRST
    # application from the duct baseline -- 2026-07 Tier-3 stabilization --
    # so a prescribed-constant closure is approached geometrically, residual
    # ~ tol_closure/closure_relax, rather than hit exactly on switch-on).
    # Ducts still conserve it exactly.
    np.testing.assert_allclose(tr.rvt[:, 2], np.full(topo.n_sl, rvt_out),
                               rtol=1e-7)
    np.testing.assert_array_equal(tr.rvt[:, 3], tr.rvt[:, 2])
    np.testing.assert_array_equal(tr.rvt[:, 1], np.full(topo.n_sl, rvt_in))
    # rvt_le CONSISTENCY (section 3.4 carryover): the h0 rise equals the
    # Euler work computed from the SWEPT LE and TE columns -- if the driver
    # had passed a stale rvt_le to row_steps, this identity breaks.
    np.testing.assert_allclose(
        tr.h0[:, 2] - tr.h0[:, 1],
        omega * (tr.rvt[:, 2] - tr.rvt[:, 1]), rtol=1e-14)
    # Section 3.5: entropy rises by the loss model's delta_s, only in-row
    # (closure-lag tolerance, as for rVt above).
    np.testing.assert_allclose(tr.s[:, 2] - tr.s[:, 1], ds, rtol=1e-7)
    np.testing.assert_array_equal(tr.s[:, 1], tr.s[:, 0])
    np.testing.assert_array_equal(tr.s[:, 3], tr.s[:, 2])
    # Rothalpy conserved through the rotor (section 3.3).
    I_le = tr.h0[:, 1] - omega * tr.rvt[:, 1]
    I_te = tr.h0[:, 2] - omega * tr.rvt[:, 2]
    np.testing.assert_allclose(I_te, I_le, rtol=1e-12)


def test_stator_row_conserves_h0():
    topo, res = solve_with_row(PrescribedSwirl(rvt=2.0),
                               PrescribedLoss(delta_s=1.0), omega=0.0,
                               rvt_inlet=6.0)
    assert res.converged
    tr = res.frozen.transported
    np.testing.assert_allclose(tr.h0, H0, rtol=1e-14)  # stator: h0 const
    np.testing.assert_allclose(tr.rvt[:, 2], np.full(topo.n_sl, 2.0),
                               rtol=1e-7)   # closure-lag tolerance


def test_closure_outputs_recorded_in_frozen_inputs():
    # AD-4/ARCH-3.3: the lagged closure outputs live in ClosureFields with
    # the producing iteration's tag, for replay/attribution.
    _, res = solve_with_row(PrescribedSwirl(rvt=20.0),
                            PrescribedLoss(delta_s=0.5))
    cf = res.frozen.closures
    assert set(cf.row_exit_rvt) == {"r1"} and set(cf.row_delta_s) == {"r1"}
    np.testing.assert_allclose(cf.row_exit_rvt["r1"],
                               np.full(res.frozen.n_sl, 20.0),
                               rtol=1e-7)   # closure-lag tolerance
    # result.frozen is the bundle that PRODUCED the final iterate, so its
    # closures carry the previous iterate's tag (lagged, AD-4).
    assert cf.iteration_tag == res.record.n_iterations - 1
    assert cf.validity == 1.0


# --------------------------------------------------------------------------
# Section 6.2.5: the closure-update norm with a flow-dependent closure
# --------------------------------------------------------------------------
class _VmProportionalSwirl:
    """Test-only flow-dependent closure: exit rVt = k * mean(Vm_LE) * r.
    Converges as the flow field settles, exercising the third norm."""

    def __init__(self, k):
        self.k = k

    def exit_rvt(self, row: RowView, flow: RowFlowView) -> SwirlResult:
        return SwirlResult(rvt=self.k * float(np.mean(flow.vm)) * flow.r,
                           validity=1.0)


def test_flow_dependent_closure_converges_with_closure_norm():
    topo, res = solve_with_row(_VmProportionalSwirl(k=0.05),
                               PrescribedLoss(delta_s=0.5), omega=600.0)
    assert res.converged
    norms = [rec.closure_norm for rec in res.record.iterations]
    assert norms[0] == 1.0                    # switch-on iterate
    assert any(n > 0.0 for n in norms[1:])    # genuinely flow-dependent
    assert norms[-1] < 1e-9                   # converged (tol_closure)
    # And the recorded outputs match a fresh evaluation at the answer.
    rvt_te = res.frozen.closures.row_exit_rvt["r1"]
    r_le = res.fields.metrics.r[:, 1]
    vm_le = res.fields.vm[:, 1]
    np.testing.assert_allclose(rvt_te, 0.05 * np.mean(vm_le) * r_le,
                               rtol=1e-6)


# --------------------------------------------------------------------------
# Config boundary (AD-10)
# --------------------------------------------------------------------------
def test_row_station_mismatches_raise():
    topo = rotor_topology()
    inlet = TransportFields(h0=np.full(9, H0), s=np.full(9, S0),
                            rvt=np.full(9, 6.0))
    row = RowSpec(row_id="r1", omega=0.0, swirl=PrescribedSwirl(rvt=1.0),
                  loss=PrescribedLoss())
    wrong = RowSpec(row_id="nope", omega=0.0, swirl=PrescribedSwirl(rvt=1.0),
                    loss=PrescribedLoss())
    with pytest.raises(ConfigError, match="mismatch"):
        solve_classical(topo, GAS, FidelityConfig.tier2(), MassFlowSpec(50.0),
                        inlet, rows=[wrong])
    with pytest.raises(ConfigError, match="mutually exclusive"):
        solve_classical(topo, GAS, FidelityConfig.tier2(), MassFlowSpec(50.0),
                        inlet, rows=[row],
                        steps=[TransportStep()] * (topo.n_qo - 1))
    with pytest.raises(ConfigError, match="declares blade rows"):
        solve_classical(topo, GAS, FidelityConfig.tier2(), MassFlowSpec(50.0),
                        inlet)
