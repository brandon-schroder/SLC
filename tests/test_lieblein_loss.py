"""Tests for the Lieblein diffusion-factor profile loss (Theory Manual
sections 4.3, 4.4, 7.3; Appendix B.2; V4-class structural checks).

Same V4 status as the deviation set: structural anchors, trends, bands,
and smoothness sweeps bound here; point-by-point published-figure
reproduction is **[VERIFY]** pending the reference library.

Provenance: M4 sub-step 4, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures import conversions as cv
from slcflow.closures.axial_compressor import (LIEBLEIN_NACA65, LieblienLoss,
                                               LieblienSwirl,
                                               equivalent_diffusion,
                                               wake_momentum_thickness)
from slcflow.closures.interfaces import CorrelationSet, LossModel
from slcflow.drivers import RowSpec, solve_classical
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import ParamRowGeometry
from slcflow.transport import TransportFields
from slcflow.types import FidelityConfig, MassFlowSpec
from tests.test_closure_wiring import H0, S0, rotor_topology

GAS = PerfectGas()
DEG = np.pi / 180.0


def _assert_c1_continuous(f, x_lo, x_hi):
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
# Structural anchors and trends (section 9.4 V4, structural part) [VERIFY]
# --------------------------------------------------------------------------
def test_equivalent_diffusion_anchors():
    # No turning, equal speeds: D_eq = 1.12 exactly (the fit's floor term).
    b = 40 * DEG
    assert equivalent_diffusion(150.0, 150.0, b, b, 1.2) \
        == pytest.approx(1.12)
    # More turning -> more diffusion; more solidity -> less.
    d_lo = equivalent_diffusion(150.0, 120.0, 55 * DEG, 40 * DEG, 1.2)
    d_hi = equivalent_diffusion(150.0, 120.0, 55 * DEG, 30 * DEG, 1.2)
    assert d_hi > d_lo > 1.12
    d_dense = equivalent_diffusion(150.0, 120.0, 55 * DEG, 40 * DEG, 1.8)
    assert d_dense < d_lo


def test_wake_momentum_thickness_anchor_and_growth():
    # theta/c = 0.004 at D_eq = 1 (ln 1 = 0) by construction of the fit;
    # steep growth toward the D_eq ~ 2 stall end. [VERIFY figure]
    t1, v1 = wake_momentum_thickness(1.0)
    assert t1 == pytest.approx(0.004, rel=1e-6)
    t2, _ = wake_momentum_thickness(1.9)
    assert t2 > 3.0 * t1
    # D_eq = 1.0 is exactly the calibration edge (v = 0.5 by construction
    # of the compact blend window); strictly inside gives 1.
    assert v1 == pytest.approx(0.5)
    assert wake_momentum_thickness(1.3)[1] == 1.0
    # Saturation: finite and bounded even past the fit's denominator zero.
    t3, v3 = wake_momentum_thickness(5.0)
    assert np.isfinite(t3) and t3 > 0.0
    assert v3 == 0.0


def test_wake_momentum_thickness_c1_and_finite():
    _assert_c1_continuous(lambda d: wake_momentum_thickness(d)[0], 0.0, 6.0)
    d = np.linspace(-1.0, 10.0, 500)
    t, v = wake_momentum_thickness(d)
    assert np.all(np.isfinite(t)) and np.all(t > 0.0)
    assert np.all((v >= 0.0) & (v <= 1.0))


# --------------------------------------------------------------------------
# Full loss model on a synthetic view (bucket, bands, B.2 consistency)
# --------------------------------------------------------------------------
def make_view_and_row(vm=110.0, omega=400.0, r=0.45, rvt=0.0,
                      beta1_blade_deg=-63.0):
    """Hand-built single-streamtube view + row at a controllable incidence."""
    from slcflow.closures.interfaces import RowFlowView, RowView

    geom = ParamRowGeometry(blade_count=31, beta1=beta1_blade_deg * DEG,
                            beta2=-45.0 * DEG, chord_len=0.06,
                            solidity_val=1.2, thickness=0.08)
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
    row = RowView(row_id="r1", omega=omega, blade_count=31, geometry=geom)
    return row, view


def test_loss_band_and_positivity_typical_section():
    # Typical loaded section: profile omega_bar in the classic 0.005-0.08
    # band, delta_s > 0. [VERIFY point values]
    row, view = make_view_and_row()
    out = LieblienLoss().evaluate(row, view)
    wbar = float(out.components["profile_omega_bar"][0])
    assert 0.005 < wbar < 0.08
    assert float(out.delta_s[0]) > 0.0
    assert out.validity > 0.0


def test_off_design_bucket_minimum_near_reference():
    # Section 4.3: the loss bucket's minimum sits near the reference
    # incidence; both deep-positive and deep-negative incidence cost more.
    def wbar_at(vm):
        row, view = make_view_and_row(vm=vm)
        return float(LieblienLoss().evaluate(
            row, view).components["profile_omega_bar"][0])

    # vm sweep moves beta1_flow, hence incidence.
    w_design = wbar_at(110.0)
    assert wbar_at(80.0) > w_design      # higher |beta1| -> +incidence
    assert wbar_at(160.0) > w_design     # lower |beta1| -> -incidence


def test_delta_s_matches_b2_conversion_of_native_coefficient():
    # Section 4.4 bookkeeping: the recorded native omega_bar and delta_s
    # must be linked by exactly the B.2 conversion at the view's state.
    row, view = make_view_and_row()
    out = LieblienLoss().evaluate(row, view)
    wbar = out.components["profile_omega_bar"]
    w1 = float(view.vm[0]) / np.cos(np.arctan2(-view.w_theta[0],
                                               view.vm[0]))
    p1 = GAS.p(view.h, view.s)
    T0r1, p0r1 = cv.relative_stagnation(GAS, view.T, p1, abs(w1))
    ds, _ = cv.delta_s_compressor_omega_bar(
        GAS, wbar, T0r1, p0r1, p1, row.omega * view.r,
        row.omega * view.r_te)
    # rtol reflects the closure's interior soft-clip residue on beta1
    # (~4e-4 deg at 8.5 transition widths inside the limit), not slack in
    # the B.2 conversion itself.
    np.testing.assert_allclose(out.delta_s, ds, rtol=1e-4)


def test_loss_c1_in_flow_input():
    # Section 7.3: C1 in the flow input (vm sweep crosses the bucket and
    # the D_eq saturation smoothly).
    def wbar_of_vm(vm_arr):
        out = []
        for v in vm_arr:
            row, view = make_view_and_row(vm=v)
            out.append(float(LieblienLoss().evaluate(
                row, view).components["profile_omega_bar"][0]))
        return np.array(out)

    _assert_c1_continuous(wbar_of_vm, 70.0, 180.0)


# --------------------------------------------------------------------------
# End-to-end: the full CorrelationSet through the classical driver
# --------------------------------------------------------------------------
def test_full_correlation_set_rotor_end_to_end():
    assert isinstance(LIEBLEIN_NACA65, CorrelationSet)
    assert isinstance(LIEBLEIN_NACA65.loss, LossModel)
    # A physically sensible twisted rotor (near-free-vortex design for
    # rVt_te ~ 12 at Omega = 400, vm ~ 91): metal angles matched to the
    # spanwise inlet triangles with small incidence and light loading. A
    # constant-metal-angle blade at this Omega over-turns the hub (negative
    # absolute swirl, ferocious spanwise gradients) -- a physics statement,
    # not a wiring one; the wiring test stays in the healthy regime.
    geom = ParamRowGeometry(
        blade_count=31,
        beta1=np.array([-53.0, -62.5, -66.5, -69.0]) * DEG,
        beta2=np.array([-36.0, -53.0, -59.0, -62.5]) * DEG,
        chord_len=0.06, solidity_val=1.2, thickness=0.08)
    topo = rotor_topology(n_sl=9)
    inlet = TransportFields(h0=np.full(9, H0), s=np.full(9, S0),
                            rvt=np.zeros(9))
    row = RowSpec(row_id="r1", omega=400.0, swirl=LIEBLEIN_NACA65.swirl,
                  loss=LIEBLEIN_NACA65.loss, blade_count=31, geometry=geom)
    res = solve_classical(topo, GAS, FidelityConfig.tier2(),
                          MassFlowSpec(100.0), inlet, rows=[row])
    assert res.converged
    tr = res.frozen.transported
    ds_row = tr.s[:, 2] - tr.s[:, 1]
    assert np.all(ds_row > 0.0)          # real loss, every streamtube
    assert np.all(tr.h0[:, 2] > H0)      # real work
    # Rotor efficiency in a sane band: eta ~ dh0_ideal/dh0 with
    # dh0_ideal = dh0 - T ds (small-loss approx). [sanity, not V5]
    T_mean = float(np.mean(res.fields.T[:, 2]))
    eta = 1.0 - T_mean * np.mean(ds_row) / np.mean(tr.h0[:, 2] - H0)
    assert 0.80 < eta < 0.99
    assert res.frozen.closures.validity > 0.3