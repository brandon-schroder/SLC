"""Tests for the Kacker-Okapuu axial-turbine profile loss (Theory Manual
sections 4.3, 4.4, 7.3; Appendix B.3; V6-class structural checks).

V6 status (same boundary as the compressor V4/V5 set): structural anchors,
trends, magnitude bands, and the section 7.3.4 smoothness sweeps are bound
here; point-by-point reproduction of the published AM/K-O charts and stage
maps is **[VERIFY]** pending the reference library — the fit coefficients
were encoded from general knowledge of the published forms.

Provenance: M6 sub-step 2, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures import conversions as cv
from slcflow.closures.axial_turbine import (KACKER_OKAPUU, AinleyTurbineSwirl,
                                            KackerOkapuuLoss,
                                            mach_profile_correction,
                                            profile_loss_am,
                                            reynolds_correction,
                                            throat_exit_angle)
from slcflow.closures.interfaces import (CorrelationSet, LossModel,
                                         RowFlowView, RowView)
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
# Section 4.3: profile_loss_am (AM nozzle/impulse interpolation) [VERIFY]
# --------------------------------------------------------------------------
def test_profile_loss_am_bands_and_impulse_over_nozzle():
    # Classic turbine profile-loss band, and impulse (b1=b2) loses more than
    # a nozzle (b1=0) at the same pitch/chord and exit angle.
    yp_noz = float(profile_loss_am(0.8, 0.0, 65.0, 0.15)[0])
    yp_imp = float(profile_loss_am(0.8, 65.0, 65.0, 0.15)[0])
    assert 0.02 < yp_noz < 0.15
    assert yp_imp > yp_noz
    # More turning (higher exit angle) costs more.
    assert float(profile_loss_am(0.8, 0.0, 72.0, 0.15)[0]) > yp_noz


def test_profile_loss_am_nozzle_independent_of_thickness():
    # Section 4.3 / K-O: the thickness correction is (t/c / 0.2)^(b1/b2), so
    # at b1 = 0 (nozzle) it is exactly 1 -- nozzle loss is thickness-free.
    thin = float(profile_loss_am(0.8, 0.0, 60.0, 0.08)[0])
    thick = float(profile_loss_am(0.8, 0.0, 60.0, 0.28)[0])
    # Equal to within the b1/b2 soft-clip residue at r = 0 (a few 1e-6 rel),
    # not an exact algebraic identity.
    assert thin == pytest.approx(thick, rel=1e-4)
    # For an impulse blade the thickness DOES bite (exponent 1).
    assert float(profile_loss_am(0.8, 60.0, 60.0, 0.28)[0]) \
        != pytest.approx(float(profile_loss_am(0.8, 60.0, 60.0, 0.08)[0]))


def test_profile_loss_am_optimum_pitch_and_validity():
    # A shallow minimum in pitch/chord: mid-range beats both tight and wide.
    yy = np.array([float(profile_loss_am(s, 30.0, 60.0, 0.15)[0])
                   for s in (0.45, 0.75, 1.05)])
    assert yy[1] < yy[0] and yy[1] < yy[2]
    # Validity is a measure in [0, 1], -> 0 outside the calibrated band.
    v = profile_loss_am(np.linspace(0.1, 1.5, 300), 30.0, 60.0, 0.15)[1]
    assert np.all((v >= 0.0) & (v <= 1.0))
    assert float(profile_loss_am(0.75, 30.0, 60.0, 0.15)[1]) > 0.9
    assert float(profile_loss_am(0.15, 30.0, 60.0, 0.15)[1]) < 0.1


def test_profile_loss_am_c1_and_finite():
    _assert_c1_continuous(lambda s: profile_loss_am(s, 30.0, 60.0, 0.15)[0],
                          0.1, 1.5)
    _assert_c1_continuous(lambda a: profile_loss_am(0.8, 30.0, a, 0.15)[0],
                          10.0, 85.0)


# --------------------------------------------------------------------------
# Section 4.3: Mach (K_p) and Reynolds (f_Re) corrections
# --------------------------------------------------------------------------
def test_mach_correction_unity_low_speed_and_drop_high_subsonic():
    # K_p ~ 1 well below M2 = 0.2; drops below 1 at high subsonic (K-O found
    # the incompressible AM charts over-predict there).
    assert float(mach_profile_correction(0.05, 0.10)) == pytest.approx(
        1.0, abs=0.02)
    assert float(mach_profile_correction(0.4, 0.8)) < 0.95
    # Bounded below by the floor, never negative, finite across the range.
    m2 = np.linspace(0.01, 1.4, 400)
    kp = mach_profile_correction(0.3 * m2, m2)
    assert np.all(np.isfinite(kp)) and np.all(kp >= 0.0)
    _assert_c1_continuous(lambda m: mach_profile_correction(0.4 * m, m),
                          0.05, 1.3)


def test_reynolds_correction_flat_band_and_tails():
    # Unity in the flat band [2e5, 1e6]; low-Re penalty above 1; mild
    # high-Re rise below 1.
    assert float(reynolds_correction(5.0e5)) == pytest.approx(1.0, abs=1e-6)
    assert float(reynolds_correction(5.0e4)) > 1.0
    assert float(reynolds_correction(5.0e6)) < 1.0
    _assert_c1_continuous(lambda lre: reynolds_correction(10.0 ** lre),
                          4.0, 7.5)


# --------------------------------------------------------------------------
# Section 4.4 / B.3: the full loss model on a synthetic rotor view
# --------------------------------------------------------------------------
def _rotor_view_and_row(vm=140.0, omega=500.0, r=0.45, rvt=30.0, vm_te=160.0,
                        throat=0.030, blade_count=50, beta1_metal_deg=35.0):
    geom = ParamRowGeometry(blade_count=blade_count,
                            beta1=beta1_metal_deg * DEG, beta2=60 * DEG,
                            chord_len=0.03, solidity_val=1.2, thickness=0.12,
                            throat_val=throat)
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
                       r_te=np.array([r]), vm_te=np.array([vm_te]))
    row = RowView(row_id="r1", omega=omega, blade_count=blade_count,
                  geometry=geom)
    return row, view


def test_loss_band_positivity_and_validity():
    row, view = _rotor_view_and_row()
    out = KackerOkapuuLoss().evaluate(row, view)
    Y = float(out.components["profile_Y"][0])
    assert 0.02 < Y < 0.2
    assert float(out.delta_s[0]) > 0.0        # real loss (entropy rise)
    assert out.validity > 0.0


def test_delta_s_matches_b3_conversion_of_native_coefficient():
    # Section 4.4 bookkeeping: the recorded native Y and delta_s must be
    # linked by exactly the B.3 conversion at the loss model's exit state.
    row, view = _rotor_view_and_row()
    out = KackerOkapuuLoss().evaluate(row, view)
    Y = out.components["profile_Y"]

    # Reconstruct the ideal exit state the loss model evaluated at.
    sgn = row.geometry.orientation
    pitch = 2.0 * np.pi * float(view.r_te[0]) / row.geometry.blade_count
    a2_deg = float(throat_exit_angle(float(row.geometry.throat(0.5))
                                     / pitch)[0])
    b1r = sgn * float(view.beta[0])
    w1 = float(view.vm[0]) / np.cos(b1r)
    w2 = float(view.vm_te[0]) / np.cos(np.deg2rad(a2_deg))
    p1 = GAS.p(view.h, view.s)
    T0r1, p0r1 = cv.relative_stagnation(GAS, view.T, p1, abs(w1))
    u1, u2 = row.omega * view.r, row.omega * view.r_te
    T0r2, p0r2_id = cv.ideal_exit_relative_stagnation(GAS, T0r1, p0r1, u1, u2)
    T2 = T0r2 - 0.5 * w2 * w2 / GAS.cp
    p2 = p0r2_id * (T2 / T0r2) ** (GAS.gamma / (GAS.gamma - 1.0))
    ds, _ = cv.delta_s_turbine_Y(GAS, Y, T0r1, p0r1, p2, u1, u2)
    # rtol reflects the interior soft-saturation residue on the T2 floor and
    # the b1 soft-clip, not slack in the B.3 conversion itself.
    np.testing.assert_allclose(out.delta_s, ds, rtol=1e-4)


def test_loss_c1_in_flow_input():
    def y_of_vm(vm_arr):
        return np.array([float(KackerOkapuuLoss().evaluate(
            *_rotor_view_and_row(vm=float(v))).components["profile_Y"][0])
            for v in vm_arr])

    _assert_c1_continuous(y_of_vm, 90.0, 200.0)


# --------------------------------------------------------------------------
# End-to-end: the full KACKER_OKAPUU CorrelationSet through the driver
# --------------------------------------------------------------------------
def test_correlation_set_nozzle_end_to_end():
    assert isinstance(KACKER_OKAPUU, CorrelationSet)
    assert isinstance(KACKER_OKAPUU.loss, LossModel)
    n_sl = 9
    topo = rotor_topology(n_sl)
    inlet = TransportFields(h0=np.full(n_sl, H0), s=np.full(n_sl, S0),
                            rvt=np.zeros(n_sl))
    # Near-free-vortex nozzle (healthy regime; see test_ainley).
    geom = ParamRowGeometry(
        blade_count=40, beta1=5 * DEG, beta2=65 * DEG, chord_len=0.04,
        solidity_val=1.5, thickness=0.12,
        throat_val=np.array([0.028, 0.044, 0.061, 0.078]))
    row = RowSpec(row_id="r1", omega=0.0, swirl=KACKER_OKAPUU.swirl,
                  loss=KACKER_OKAPUU.loss, blade_count=40, geometry=geom)
    res = solve_classical(topo, GAS, FidelityConfig.tier2(),
                          MassFlowSpec(100.0), inlet, rows=[row])
    assert res.converged
    tr = res.frozen.transported
    # Nozzle: real loss (entropy rises across the row), no Euler work.
    assert np.all(tr.s[:, 2] - tr.s[:, 1] > 0.0)
    np.testing.assert_allclose(tr.h0[:, 2], H0, rtol=1e-10)
    # Nozzle total-pressure loss coefficient in a sane band [VERIFY].
    T_mean = float(np.mean(res.fields.T[:, 2]))
    ds_mean = float(np.mean(tr.s[:, 2] - tr.s[:, 1]))
    assert 0.0 < ds_mean < 30.0
    assert res.frozen.closures.validity > 0.3
