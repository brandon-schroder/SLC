"""INBLADE-station tests (Theory Manual sections 2.5, 3.4, 3.5, B.5.1;
M7 sub-step 3).

An INBLADE station sits between a row's EDGE_LE and EDGE_TE; the row's work
(rVt change) and loss (delta_s) are distributed across the sub-intervals by
the section 3.4/3.5 DistributionSchedule. The TE values are exact regardless
of subdivision (the schedule reaches 1 there), so an edge-only row and a
subdivided row must agree at the TE -- the consistency gate here.

Provenance: M7 sub-step 3, written with the constraint-lift.
"""
import numpy as np
import pytest

from slcflow.drivers import RowSpec, solve_classical
from slcflow.drivers.classical import _resolve_rows
from slcflow.closures.simple import PrescribedLoss, PrescribedSwirl
from slcflow.errors import ConfigError
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import FlowPath, StationDef, StationType, WallCurve
from slcflow.grid import GridTopology
from slcflow.transport import (SmoothRampSchedule, TransportFields,
                               assert_valid_schedule)
from slcflow.types import FidelityConfig, MassFlowSpec

GAS = PerfectGas()
H0, S0 = 3.0e5, 0.0
R0, R1 = 0.3, 0.6
_A_LE, _A_TE = 0.35, 0.55


def _walls():
    z = np.linspace(0.0, 1.0, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R0)]))
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R1)]))
    return w0, w1


def inblade_topology(n_sl=9, t_inblade=0.5):
    """DUCT, EDGE_LE, INBLADE, EDGE_TE, DUCT -- one row with one in-blade
    station at meridional fraction ``t_inblade`` of the LE->TE chord."""
    w0, w1 = _walls()
    a_ib = _A_LE + t_inblade * (_A_TE - _A_LE)
    stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                StationDef(StationType.EDGE_LE, _A_LE, _A_LE, row_id="r1"),
                StationDef(StationType.INBLADE, a_ib, a_ib, row_id="r1"),
                StationDef(StationType.EDGE_TE, _A_TE, _A_TE, row_id="r1"),
                StationDef(StationType.DUCT, 1.0, 1.0)]
    return GridTopology(FlowPath(w0, w1, stations), n_sl=n_sl)


def edge_only_topology(n_sl=9):
    w0, w1 = _walls()
    stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                StationDef(StationType.EDGE_LE, _A_LE, _A_LE, row_id="r1"),
                StationDef(StationType.EDGE_TE, _A_TE, _A_TE, row_id="r1"),
                StationDef(StationType.DUCT, 1.0, 1.0)]
    return GridTopology(FlowPath(w0, w1, stations), n_sl=n_sl)


def _solve(topo, omega=500.0, rvt_le=6.0, rvt_te=20.0, ds=2.0, mdot=100.0):
    n_sl = topo.n_sl
    inlet = TransportFields(h0=np.full(n_sl, H0), s=np.full(n_sl, S0),
                            rvt=np.full(n_sl, rvt_le))
    row = RowSpec(row_id="r1", omega=omega, swirl=PrescribedSwirl(rvt=rvt_te),
                  loss=PrescribedLoss(delta_s=ds))
    return solve_classical(topo, GAS, FidelityConfig.tier2(),
                           MassFlowSpec(mdot), inlet, rows=[row])


# --------------------------------------------------------------------------
# Section 3.4/3.5: the in-blade station carries the scheduled distribution
# --------------------------------------------------------------------------
def test_inblade_station_holds_scheduled_values():
    res = _solve(inblade_topology(t_inblade=0.5))
    assert res.converged
    tr = res.frozen.transported
    # Columns: 0 DUCT, 1 LE, 2 INBLADE, 3 TE, 4 DUCT.
    f = float(SmoothRampSchedule()(0.5))             # = 0.5 (quintic)
    rvt_ib_expected = (1.0 - f) * 6.0 + f * 20.0     # = 13.0
    # Closure-lag tolerance: the section 6.2.4 under-relaxation now also
    # ramps the FIRST closure application from the duct baseline (2026-07
    # Tier-3 stabilization), so prescribed-constant closure values are
    # approached to ~tol_closure/closure_relax, not hit exactly.
    np.testing.assert_allclose(tr.rvt[:, 2], rvt_ib_expected, rtol=1e-7)
    np.testing.assert_allclose(tr.rvt[:, 3], 20.0, rtol=1e-7)   # TE value
    # Euler work follows the swept rVt (section 3.3); entropy follows the
    # same schedule (B.5.1): half the row's delta_s by the INBLADE station.
    np.testing.assert_allclose(tr.h0[:, 2] - H0,
                               500.0 * (rvt_ib_expected - 6.0), rtol=1e-7)
    np.testing.assert_allclose(tr.s[:, 2] - S0, f * 2.0, rtol=1e-7)
    np.testing.assert_allclose(tr.s[:, 3] - S0, 2.0, rtol=1e-7)


def test_inblade_rvt_monotone_through_row():
    res = _solve(inblade_topology(t_inblade=0.3))
    tr = res.frozen.transported
    # rVt ramps monotonically LE -> INBLADE -> TE (6 -> . -> 20).
    seq = [float(tr.rvt[0, j]) for j in (1, 2, 3)]
    assert seq[0] < seq[1] < seq[2]
    assert seq[0] == pytest.approx(6.0) and seq[2] == pytest.approx(20.0)


def test_edge_only_and_subdivided_agree_at_te():
    # The TE state is exact regardless of INBLADE subdivision (the schedule
    # reaches 1 there): edge-only and subdivided rows must match at the TE.
    edge = _solve(edge_only_topology())
    sub = _solve(inblade_topology(t_inblade=0.4))
    te_edge, te_sub = edge.frozen.transported, sub.frozen.transported
    np.testing.assert_allclose(te_sub.rvt[:, -2], te_edge.rvt[:, -2],
                               rtol=1e-10)
    np.testing.assert_allclose(te_sub.h0[:, -2], te_edge.h0[:, -2], rtol=1e-10)
    np.testing.assert_allclose(te_sub.s[:, -2], te_edge.s[:, -2], rtol=1e-10)


# --------------------------------------------------------------------------
# Config boundary (AD-10): row-station resolution
# --------------------------------------------------------------------------
def test_resolve_rows_derives_t_stations():
    topo = inblade_topology(t_inblade=0.5)
    row = RowSpec(row_id="r1", omega=0.0, swirl=PrescribedSwirl(rvt=1.0),
                  loss=PrescribedLoss())
    (spec, j_le, j_te, ts), = _resolve_rows(topo, [row])
    assert (j_le, j_te) == (1, 3)
    np.testing.assert_allclose(ts, (0.5, 1.0))       # INBLADE at mid, TE at 1


def test_resolve_rows_rejects_noncontiguous_row():
    # A DUCT station inside the row (between LE and TE) is rejected.
    w0, w1 = _walls()
    stations = [StationDef(StationType.EDGE_LE, 0.2, 0.2, row_id="r1"),
                StationDef(StationType.DUCT, 0.4, 0.4),
                StationDef(StationType.EDGE_TE, 0.6, 0.6, row_id="r1")]
    topo = GridTopology(FlowPath(w0, w1, stations), n_sl=5)
    row = RowSpec(row_id="r1", omega=0.0, swirl=PrescribedSwirl(rvt=1.0),
                  loss=PrescribedLoss())
    with pytest.raises(ConfigError, match="contiguous"):
        _resolve_rows(topo, [row])


# --------------------------------------------------------------------------
# Section 7.3.4: schedule contract helper (M4 carryover)
# --------------------------------------------------------------------------
def test_assert_valid_schedule_accepts_default():
    assert_valid_schedule(SmoothRampSchedule())      # must not raise


def test_assert_valid_schedule_rejects_bad_schedule():
    # A linear ramp satisfies f(0)=0, f(1)=1 and monotonicity but has nonzero
    # end slopes -> the composite in-blade field would kink at the LE/TE
    # joins. The helper must reject it (section 3.4 C1 requirement).
    class LinearSchedule:
        def __call__(self, t, *, xp=None):
            return np.clip(np.asarray(t, dtype=float), 0.0, 1.0)

    with pytest.raises(AssertionError):
        assert_valid_schedule(LinearSchedule())
