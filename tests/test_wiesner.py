"""Tests for the Wiesner centrifugal slip closure (Theory Manual sections
3.4, 7.1, 7.3; V7-class structural checks per section 9.7).

V7 status: STRUCTURAL anchors, trends, and the section 7.3.4 smoothness
sweeps are bound here; point-by-point reproduction of published impeller
(Eckardt) exit data is **[VERIFY]** pending the reference library and the
M7-2..M7-4 loss/INBLADE/verification work.

Provenance: M7 sub-step 1, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures.centrifugal import WiesnerSlip, wiesner_slip
from slcflow.closures.interfaces import RowFlowView, RowView, SwirlClosure
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import ParamRowGeometry

GAS = PerfectGas()
DEG = np.pi / 180.0
H0, S0 = 3.0e5, 0.0


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
# Section 3.4: Wiesner slip factor sigma = 1 - sqrt(cos b2b) / Z^0.7 [VERIFY]
# --------------------------------------------------------------------------
def test_wiesner_slip_anchor_and_trends():
    # Radial blades (beta2b = 0), Z = 20: sigma = 1 - 1/20^0.7 = 0.877.
    s0, v = wiesner_slip(0.0, 20)
    assert float(s0) == pytest.approx(1.0 - 20 ** -0.7, rel=1e-9)
    assert 0.999 < float(v) <= 1.0
    # More backsweep -> higher sigma (less slip, cos falls).
    assert float(wiesner_slip(45 * DEG, 20)[0]) > float(s0)
    # More blades -> higher sigma (less slip). Fewer -> lower.
    assert float(wiesner_slip(0.0, 30)[0]) > s0 > float(wiesner_slip(0.0, 8)[0])
    # sigma stays a physical fraction in (0, 1) across the range.
    for z in (5, 12, 25, 40):
        for b in (0.0, 30 * DEG, 60 * DEG):
            assert 0.0 < float(wiesner_slip(b, z)[0]) < 1.0


def test_wiesner_slip_validity_and_c1():
    # Validity fades for extreme backsweep and out-of-band blade counts.
    assert float(wiesner_slip(20 * DEG, 20)[1]) > 0.9
    assert float(wiesner_slip(85 * DEG, 20)[1]) < 0.2
    assert float(wiesner_slip(0.0, 2)[1]) < 0.5        # too few blades
    v = wiesner_slip(np.linspace(-10 * DEG, 100 * DEG, 300), 20)[1]
    assert np.all((v >= 0.0) & (v <= 1.0))
    # Section 7.3: C1 in the backsweep (magnitude soft-saturation).
    _assert_c1_continuous(lambda b: wiesner_slip(b, 20)[0], -0.2, 1.7)


# --------------------------------------------------------------------------
# Section 7.1: WiesnerSlip closure on a synthetic impeller-exit view
# --------------------------------------------------------------------------
def _view_and_row(vm2=80.0, omega=3000.0, r2=0.15, backsweep_deg=0.0,
                  blade_count=20, r1=0.06):
    """Single-streamtube impeller-exit view (radial exit) + row."""
    geom = ParamRowGeometry(blade_count=blade_count, beta1=30 * DEG,
                            beta2=backsweep_deg * DEG, chord_len=0.05,
                            solidity_val=2.0)
    # LE (inducer) fields are unused by the slip closure; set plausible values.
    view = RowFlowView(psi=np.array([0.5]), r=np.array([r1]),
                       vm=np.array([120.0]), vtheta=np.array([0.0]),
                       w_theta=np.array([-omega * r1]),
                       alpha=np.array([0.0]),
                       beta=np.arctan2([-omega * r1], [120.0]),
                       h=np.array([H0]), s=np.array([S0]),
                       T=GAS.T(np.array([H0]), S0),
                       rho=GAS.rho(np.array([H0]), S0),
                       a=GAS.a(np.array([H0]), S0), fluid=GAS,
                       r_te=np.array([r2]), vm_te=np.array([vm2]))
    row = RowView(row_id="imp", omega=omega, blade_count=blade_count,
                  geometry=geom)
    return row, view


def test_slip_is_protocol_and_radial_blade_anchor():
    assert isinstance(WiesnerSlip(), SwirlClosure)
    row, view = _view_and_row(backsweep_deg=0.0)
    out = WiesnerSlip().exit_rvt(row, view)
    # Radial blades: V_theta2 = sigma U2 (no backsweep term), rVt2 = r2 sigma U2.
    r2 = float(view.r_te[0])
    sigma = float(wiesner_slip(0.0, row.geometry.blade_count)[0])
    assert float(out.rvt[0]) == pytest.approx(
        r2 * sigma * row.omega * r2, rel=1e-9)
    assert float(out.rvt[0]) > 0.0
    assert 0.0 < out.validity <= 1.0


def test_backsweep_reduces_exit_swirl_and_work():
    # More backsweep -> less exit rVtheta (less Euler work), the defining
    # centrifugal design lever.
    radial = WiesnerSlip().exit_rvt(*_view_and_row(backsweep_deg=0.0)).rvt[0]
    swept = WiesnerSlip().exit_rvt(*_view_and_row(backsweep_deg=45.0)).rvt[0]
    assert float(radial) > float(swept) > 0.0


def test_more_blades_raise_exit_swirl():
    # Higher blade count -> higher slip factor -> more exit swirl.
    few = WiesnerSlip().exit_rvt(*_view_and_row(blade_count=10)).rvt[0]
    many = WiesnerSlip().exit_rvt(*_view_and_row(blade_count=30)).rvt[0]
    assert float(many) > float(few) > 0.0


def test_slip_c1_across_backsweep():
    # Section 7.3: C1 across the backsweep, the closure's genuinely nonlinear
    # path (sigma via sqrt(cos), plus the tan(beta2b) term and the magnitude
    # soft-saturation). The vm dependence is exactly linear, so it is
    # trivially C1 but not a useful refinement-scaling target (see
    # test_smoothmath.py docstring on why linear functions defeat the check).
    def rvt_of_backsweep(bs_deg):
        return np.array([float(WiesnerSlip().exit_rvt(
            *_view_and_row(backsweep_deg=float(b))).rvt[0]) for b in bs_deg])

    _assert_c1_continuous(rvt_of_backsweep, -10.0, 88.0)
