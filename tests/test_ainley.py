"""Tests for the axial-turbine throat-based exit-angle closure and the
``throat`` geometry-contract slot (Theory Manual sections 4.1, 4.5, 3.4,
7.3; V6-class structural checks per section 9.6).

V6 status: STRUCTURAL anchors, trends, and the section 7.3.4 smoothness
sweeps are bound here; point-by-point reproduction of published turbine
exit-angle / stage data is **[VERIFY]** pending the reference library and
the M6-4 transonic deviation correction (this step ships the throat cosine
rule alone — the correct sonic asymptote).

Provenance: M6 sub-step 1, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures.axial_turbine import AinleyTurbineSwirl, throat_exit_angle
from slcflow.closures.interfaces import RowFlowView, RowView, SwirlClosure
from slcflow.closures.simple import PrescribedLoss
from slcflow.drivers import RowSpec, solve_classical
from slcflow.errors import ConfigError
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import BladeRowGeometry, ParamRowGeometry
from slcflow.transport import TransportFields
from slcflow.types import FidelityConfig, MassFlowSpec
from tests.test_closure_wiring import H0, S0, rotor_topology

GAS = PerfectGas()
DEG = np.pi / 180.0


def _assert_c1_continuous(f, x_lo, x_hi):
    """Refinement-scaling C1 check (see test_smoothmath.py docstring)."""
    def indicator(n):
        x = np.linspace(x_lo, x_hi, n)
        dx = x[1] - x[0]
        y = f(x)
        d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
        assert np.all(np.isfinite(y))
        return np.max(np.abs(d2)) / dx

    ratio = indicator(4001) / (indicator(2001) + 1e-300)
    assert ratio < 0.75, f"derivative discontinuity (ratio {ratio:.3f})"


# --------------------------------------------------------------------------
# Section 4.1 / 4.5: the throat geometry-contract slot
# --------------------------------------------------------------------------
def test_geometry_throat_slot_scalar_and_array():
    g = ParamRowGeometry(blade_count=40, beta1=5 * DEG, beta2=65 * DEG,
                         chord_len=0.04, solidity_val=1.5, throat_val=0.03)
    assert isinstance(g, BladeRowGeometry)
    assert g.throat(0.5) == pytest.approx(0.03)
    y = np.linspace(0, 1, 7)
    assert np.shape(g.throat(y)) == (7,)
    # Array throat -> PCHIP over span, C1, honoring the endpoints.
    ga = ParamRowGeometry(blade_count=40, beta1=5 * DEG, beta2=65 * DEG,
                          chord_len=0.04, solidity_val=1.5,
                          throat_val=np.array([0.028, 0.030, 0.032, 0.034]))
    assert ga.throat(0.0) == pytest.approx(0.028)
    assert ga.throat(1.0) == pytest.approx(0.034)


def test_geometry_throat_absent_raises_only_when_asked():
    # A compressor-style row that never sets a throat is valid to construct
    # (throat is optional) and only complains if a turbine closure asks.
    g = ParamRowGeometry(blade_count=30, beta1=-60 * DEG, beta2=-42 * DEG,
                         chord_len=0.05, solidity_val=1.2)
    assert isinstance(g, BladeRowGeometry)
    with pytest.raises(ConfigError, match="throat"):
        g.throat(0.5)


def test_geometry_throat_nonpositive_rejected():
    with pytest.raises(ConfigError, match="throat must be > 0"):
        ParamRowGeometry(blade_count=40, beta1=5 * DEG, beta2=65 * DEG,
                         chord_len=0.04, solidity_val=1.5, throat_val=-0.01)


# --------------------------------------------------------------------------
# Section 4.5: throat_exit_angle cosine rule (structural anchors) [VERIFY]
# --------------------------------------------------------------------------
def test_throat_exit_angle_cosine_anchors():
    # arccos(o/s): o/s = cos(60 deg) = 0.5 -> alpha2 = 60 deg (to within the
    # smooth-cap residue ~ width*exp(-(85-60)/width), a few 1e-6 deg).
    a60, v = throat_exit_angle(0.5)
    assert float(a60) == pytest.approx(60.0, abs=1e-4)
    assert 0.999 < float(v) <= 1.0
    # o/s = cos(45 deg) -> 45 deg.
    a45, _ = throat_exit_angle(np.cos(45 * DEG))
    assert float(a45) == pytest.approx(45.0, abs=1e-4)
    # Monotone: a wider throat (o/s -> 1) gives less turning.
    xs = np.array([0.3, 0.45, 0.6, 0.75, 0.9])
    a = throat_exit_angle(xs)[0]
    assert np.all(np.diff(a) < 0.0)


def test_throat_exit_angle_saturation_and_validity():
    # Degenerate throats are flagged (validity -> 0) but never NaN, and the
    # magnitude is smoothly capped below 90 deg (no runaway tangent).
    a_tight, v_tight = throat_exit_angle(0.01)   # near-closed throat
    assert np.isfinite(a_tight) and float(a_tight) < 90.0
    assert float(v_tight) < 0.05
    a_open, v_open = throat_exit_angle(0.995)    # near-axial exit
    assert np.isfinite(a_open) and float(a_open) >= 0.0
    assert float(v_open) < 0.5
    # Validity is a genuine measure in [0, 1] across the whole range.
    v = throat_exit_angle(np.linspace(-0.2, 1.3, 400))[1]
    assert np.all((v >= 0.0) & (v <= 1.0))


def test_throat_exit_angle_c1_and_finite():
    # Section 7.3: C1 across the arccos-domain soft-clip and the magnitude
    # cap, finite even outside the physical (0, 1) throat/pitch range.
    _assert_c1_continuous(lambda x: throat_exit_angle(x)[0], -0.2, 1.3)
    a, v = throat_exit_angle(np.linspace(-0.2, 1.3, 500))
    assert np.all(np.isfinite(a))


# --------------------------------------------------------------------------
# Section 3.4 / 7.1: AinleyTurbineSwirl on a synthetic view
# --------------------------------------------------------------------------
def _view_and_row(vm=120.0, omega=0.0, r=0.45, rvt=0.0, throat=0.03,
                  blade_count=40, beta2_deg=65.0):
    """Single-streamtube view + turbine row (nozzle by default: omega=0)."""
    geom = ParamRowGeometry(blade_count=blade_count, beta1=5 * DEG,
                            beta2=beta2_deg * DEG, chord_len=0.04,
                            solidity_val=1.5, throat_val=throat)
    vtheta = rvt / r
    w_theta = vtheta - omega * r
    h = H0 - 0.5 * (vm**2 + vtheta**2)
    T = GAS.T(h, S0)
    view = RowFlowView(psi=np.array([0.5]), r=np.array([r]),
                       vm=np.array([vm]), vtheta=np.array([vtheta]),
                       w_theta=np.array([w_theta]),
                       alpha=np.arctan2([vtheta], [vm]),
                       beta=np.arctan2([w_theta], [vm]),
                       h=np.array([h]), s=np.array([S0]),
                       T=np.array([T]), rho=GAS.rho(np.array([h]), S0),
                       a=GAS.a(np.array([h]), S0), fluid=GAS,
                       r_te=np.array([r]), vm_te=np.array([vm]))
    row = RowView(row_id="n1", omega=omega, blade_count=blade_count,
                  geometry=geom)
    return row, view


def test_swirl_is_protocol_and_nozzle_turns_flow():
    assert isinstance(AinleyTurbineSwirl(), SwirlClosure)
    row, view = _view_and_row(rvt=0.0)          # axial inflow nozzle
    out = AinleyTurbineSwirl().exit_rvt(row, view)
    # Nozzle imparts positive swirl; magnitude matches the throat angle:
    # rVt = r * vm * tan(alpha2), alpha2 the closure's own exit angle.
    r = float(view.r_te[0])
    pitch = 2.0 * np.pi * r / row.geometry.blade_count
    alpha2 = np.deg2rad(float(throat_exit_angle(0.03 / pitch)[0]))
    assert float(out.rvt[0]) == pytest.approx(
        r * float(view.vm_te[0]) * np.tan(alpha2), rel=1e-9)
    assert float(out.rvt[0]) > 0.0
    assert 0.0 < out.validity <= 1.0


def test_tighter_throat_turns_more():
    # Smaller o/s => larger exit angle => more exit swirl (section 4.5).
    lo = AinleyTurbineSwirl().exit_rvt(*_view_and_row(throat=0.024)).rvt[0]
    hi = AinleyTurbineSwirl().exit_rvt(*_view_and_row(throat=0.036)).rvt[0]
    assert float(lo) > float(hi) > 0.0


def test_rotor_relative_to_absolute_mapping():
    # For a rotor the throat sets the RELATIVE exit angle; the absolute
    # swirl adds the blade speed (section 2.4 V_theta = W_theta + omega r).
    row, view = _view_and_row(omega=500.0, rvt=8.0)
    out = AinleyTurbineSwirl().exit_rvt(row, view)
    r = float(view.r_te[0])
    pitch = 2.0 * np.pi * r / row.geometry.blade_count
    alpha2 = np.deg2rad(float(throat_exit_angle(0.03 / pitch)[0]))
    w_theta_2 = float(view.vm_te[0]) * np.tan(alpha2)
    vtheta_2 = w_theta_2 + row.omega * r
    assert float(out.rvt[0]) == pytest.approx(r * vtheta_2, rel=1e-9)


def test_swirl_c1_in_flow_input():
    # Section 7.3: C1 in the flow input (a vm_te sweep is linear here, but a
    # throat sweep crosses the arccos soft-clip smoothly).
    def rvt_of_throat(throats):
        return np.array([float(AinleyTurbineSwirl().exit_rvt(
            *_view_and_row(throat=float(o))).rvt[0]) for o in throats])

    _assert_c1_continuous(rvt_of_throat, 0.018, 0.060)


# --------------------------------------------------------------------------
# End-to-end: the swirl closure through the classical driver (nozzle stator)
# --------------------------------------------------------------------------
def test_nozzle_stator_end_to_end():
    # A turbine nozzle (omega=0) with axial inflow: the throat closure must
    # produce a converged solve that adds real swirl at the TE with no work
    # and no prescribed loss.
    n_sl = 9
    topo = rotor_topology(n_sl)
    inlet = TransportFields(h0=np.full(n_sl, H0), s=np.full(n_sl, S0),
                            rvt=np.zeros(n_sl))
    # A near-free-vortex throat schedule (throat scales with radius so o/s
    # stays ~0.65-0.75, exit angle ~40-50 deg, rVt roughly uniform across
    # span): the healthy-regime wiring case. A constant throat over a 2:1
    # radius range is geometrically inconsistent (o/s > 1 at the hub) and a
    # constant metal angle over-turns the tip -- a physics statement, not a
    # wiring one, so the wiring test stays in the healthy regime.
    geom = ParamRowGeometry(
        blade_count=40, beta1=5 * DEG, beta2=65 * DEG, chord_len=0.04,
        solidity_val=1.5, throat_val=np.array([0.028, 0.044, 0.061, 0.078]))
    row = RowSpec(row_id="r1", omega=0.0, swirl=AinleyTurbineSwirl(),
                  loss=PrescribedLoss(delta_s=0.0), blade_count=40,
                  geometry=geom)
    res = solve_classical(topo, GAS, FidelityConfig.tier2(),
                          MassFlowSpec(100.0), inlet, rows=[row])
    assert res.converged
    tr = res.frozen.transported
    # Swirl created across the row; no Euler work (stator) and no loss.
    assert np.all(tr.rvt[:, 2] > tr.rvt[:, 1])
    np.testing.assert_allclose(tr.h0[:, 2], H0, rtol=1e-10)
    np.testing.assert_allclose(tr.s[:, 2], S0, atol=1e-12)
    assert res.frozen.closures.validity > 0.5
