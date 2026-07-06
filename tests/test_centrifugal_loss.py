"""Tests for the centrifugal internal-loss model (Theory Manual sections
4.3, 4.4, 7.3; Appendix B enthalpy-loss conversion; V7-class structural
checks).

V7 status (same boundary as the axial sets): structural anchors, trends,
bands, and smoothness sweeps are bound here; point-by-point reproduction of
published impeller (Eckardt) data is **[VERIFY]** pending the reference
library and the deferred loss components.

Provenance: M7 sub-step 2, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures import conversions as cv
from slcflow.closures.centrifugal import (CENTRIFUGAL, CentrifugalLoss,
                                          WiesnerSlip, incidence_loss,
                                          skin_friction_loss, wiesner_slip)
from slcflow.closures.interfaces import (CorrelationSet, LossModel,
                                         RowFlowView, RowView)
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import ParamRowGeometry

GAS = PerfectGas()
DEG = np.pi / 180.0
H0, S0 = 4.0e5, 0.0


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
# Section 4.3: loss-component functions
# --------------------------------------------------------------------------
def test_incidence_and_friction_functions():
    # Incidence: half the squared tangential-velocity mismatch, zero at match.
    assert float(incidence_loss(50.0, 50.0)) == 0.0
    assert float(incidence_loss(80.0, 50.0)) == pytest.approx(0.5 * 30.0**2)
    # Friction: 2 Cf (L/D) W^2, grows with each factor.
    base = float(skin_friction_loss(200.0, 0.005, 4.0))
    assert base == pytest.approx(2.0 * 0.005 * 4.0 * 200.0**2)
    assert float(skin_friction_loss(200.0, 0.010, 4.0)) > base
    assert float(skin_friction_loss(260.0, 0.005, 4.0)) > base


# --------------------------------------------------------------------------
# Section 4.4 / Appendix B: full loss model on a synthetic impeller view
# --------------------------------------------------------------------------
def _view_and_row(vm1=120.0, vm2=80.0, omega=3000.0, r1=0.06, r2=0.15,
                  vtheta1=0.0, beta1_blade_deg=-56.0, backsweep_deg=30.0,
                  blade_count=20):
    geom = ParamRowGeometry(blade_count=blade_count,
                            beta1=beta1_blade_deg * DEG,
                            beta2=backsweep_deg * DEG, chord_len=0.05,
                            solidity_val=2.0)
    w_theta1 = vtheta1 - omega * r1
    h = H0 - 0.5 * (vm1**2 + vtheta1**2)
    view = RowFlowView(psi=np.array([0.5]), r=np.array([r1]),
                       vm=np.array([vm1]), vtheta=np.array([vtheta1]),
                       w_theta=np.array([w_theta1]),
                       alpha=np.arctan2([vtheta1], [vm1]),
                       beta=np.arctan2([w_theta1], [vm1]),
                       h=np.array([h]), s=np.array([S0]),
                       T=GAS.T(np.array([h]), S0),
                       rho=GAS.rho(np.array([h]), S0),
                       a=GAS.a(np.array([h]), S0), fluid=GAS,
                       r_te=np.array([r2]), vm_te=np.array([vm2]))
    row = RowView(row_id="imp", omega=omega, blade_count=blade_count,
                  geometry=geom)
    return row, view


def test_loss_components_present_and_positive():
    row, view = _view_and_row()
    out = CentrifugalLoss().evaluate(row, view)
    assert float(out.components["incidence_dh"][0]) >= 0.0
    assert float(out.components["friction_dh"][0]) > 0.0
    assert float(out.delta_s[0]) > 0.0
    assert out.validity > 0.0


def test_incidence_loss_grows_with_mismatch():
    # Blade matched to the axial-inlet inducer flow -> minimal incidence loss;
    # a mismatched metal angle costs more.
    matched = CentrifugalLoss().evaluate(
        *_view_and_row(beta1_blade_deg=-56.0))     # ~ arctan(-180/120)
    mismatched = CentrifugalLoss().evaluate(
        *_view_and_row(beta1_blade_deg=-30.0))
    assert float(mismatched.components["incidence_dh"][0]) \
        > float(matched.components["incidence_dh"][0])


def test_friction_loss_grows_with_cf():
    lo = CentrifugalLoss(cf=0.004).evaluate(*_view_and_row())
    hi = CentrifugalLoss(cf=0.008).evaluate(*_view_and_row())
    assert float(hi.components["friction_dh"][0]) \
        > float(lo.components["friction_dh"][0])


def test_delta_s_matches_enthalpy_loss_conversion():
    # Section 4.4 / B.5: recorded component dh's convert to delta_s
    # individually at the exit static T, then sum.
    row, view = _view_and_row()
    out = CentrifugalLoss().evaluate(row, view)
    dh_inc = out.components["incidence_dh"]
    dh_sf = out.components["friction_dh"]

    sgn = row.geometry.orientation
    b2b = sgn * float(row.geometry.beta2_blade(0.5))
    sigma = float(wiesner_slip(b2b, row.geometry.blade_count)[0])
    u1, u2 = row.omega * view.r, row.omega * view.r_te
    w_theta_2 = (sigma - 1.0) * u2 - view.vm_te * np.tan(b2b)
    w1 = np.sqrt(view.vm**2 + view.w_theta**2)
    w2 = np.sqrt(view.vm_te**2 + w_theta_2**2)
    T0r1 = view.T + 0.5 * w1 * w1 / GAS.cp
    T0r2 = T0r1 + 0.5 * (u2 * u2 - u1 * u1) / GAS.cp
    T2 = T0r2 - 0.5 * w2 * w2 / GAS.cp
    ds = (cv.delta_s_enthalpy_loss(GAS, dh_inc, T2)
          + cv.delta_s_enthalpy_loss(GAS, dh_sf, T2))
    np.testing.assert_allclose(out.delta_s, ds, rtol=1e-6)


def test_loss_c1_in_flow_input():
    # Section 7.3: C1 in the inducer meridional velocity (both components are
    # smooth and genuinely nonlinear in vm).
    def ds_of_vm(vms):
        return np.array([float(CentrifugalLoss().evaluate(
            *_view_and_row(vm1=float(v))).delta_s[0]) for v in vms])

    _assert_c1_continuous(ds_of_vm, 70.0, 180.0)


def test_centrifugal_set_is_correlation_set():
    assert isinstance(CENTRIFUGAL, CorrelationSet)
    assert isinstance(CENTRIFUGAL.loss, LossModel)
    assert isinstance(CENTRIFUGAL.swirl, WiesnerSlip)
