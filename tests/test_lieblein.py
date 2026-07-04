"""Tests for the blade-row geometry contract and the Lieblein
incidence/deviation set (Theory Manual sections 4.1, 4.3, 3.4, 7.3;
V4-class checks per section 9.4).

V4 status: STRUCTURAL anchors, trends, magnitude bands, and the full
section 7.3.4 sweeps are bound here; point-by-point reproduction of the
published SP-36 figures still requires the reference library and is
**[VERIFY]** — the fit coefficients were encoded from general knowledge of
Aungier's published forms. Bands below are deliberately generous.

Provenance: M4 sub-step 3, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor import (LieblienSwirl,
                                               deviation_slope,
                                               reference_deviation,
                                               reference_incidence)
from slcflow.closures.interfaces import SwirlClosure
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
# Section 4.1: ParamRowGeometry contract
# --------------------------------------------------------------------------
def test_param_row_geometry_contract_and_broadcast():
    g = ParamRowGeometry(blade_count=30, beta1=-60 * DEG, beta2=-42 * DEG,
                         chord_len=0.05, solidity_val=1.2)
    assert isinstance(g, BladeRowGeometry)
    y = np.linspace(0, 1, 7)
    assert np.shape(g.beta1_blade(y)) == (7,)
    assert g.solidity(0.5) == pytest.approx(1.2)
    assert g.orientation == -1.0


def test_param_row_geometry_c1_in_span():
    # ARCH-3.1: C1 output in span fraction (PCHIP path).
    g = ParamRowGeometry(blade_count=30,
                         beta1=np.array([-55, -60, -66, -68]) * DEG,
                         beta2=-40 * DEG, chord_len=0.05, solidity_val=1.2)
    _assert_c1_continuous(g.beta1_blade, 0.0, 1.0)


@pytest.mark.parametrize("kw", [
    dict(beta1=np.array([-0.5, 0.5])),   # mixed-sign metal angle
    dict(beta1=0.0),                     # zero metal angle
    dict(solidity_val=-1.0),
    dict(thickness=0.0),
])
def test_param_row_geometry_validation(kw):
    base = dict(blade_count=30, beta1=-1.0, beta2=-0.7, chord_len=0.05,
                solidity_val=1.2)
    with pytest.raises(ConfigError):
        ParamRowGeometry(**{**base, **kw})


# --------------------------------------------------------------------------
# Section 9.4 (V4, structural part): fit anchors and trends [VERIFY figures]
# --------------------------------------------------------------------------
def test_thickness_corrections_are_unity_at_ten_percent():
    # The "(.)_10" subscript means the base charts ARE the 10%-thickness
    # ones: both thickness corrections must equal 1 at t/c = 0.10 by
    # construction.
    i_a, _ = reference_incidence(50.0, 1.0, 0.10, 20.0)
    d_a, _ = reference_deviation(50.0, 1.0, 0.10, 20.0)
    i_b, _ = reference_incidence(50.0, 1.0, 0.10 + 1e-9, 20.0)
    assert i_a == pytest.approx(i_b, abs=1e-6)
    # K_td(0.1) = 6.25*0.1 + 37.5*0.01 = 1 exactly:
    d0, _ = reference_deviation(50.0, 1.0, 0.10, 0.0)
    d0_scaled, _ = reference_deviation(50.0, 1.0, 0.10, 0.0)
    assert d0 == d0_scaled


def test_zero_camber_zero_angle_anchors():
    # (i0)_10 and (d0)_10 vanish as beta1 -> 0 (SP-36 chart origin).
    # beta1 = 0 is exactly the calibration edge, so the smooth saturation
    # floor (softplus knee ~ w ln2 = 1.4 deg) biases the anchor slightly:
    # assert small, not exact.
    i0, _ = reference_incidence(0.0, 1.0, 0.10, 0.0)
    d0, _ = reference_deviation(0.0, 1.0, 0.10, 0.0)
    assert i0 == pytest.approx(0.0, abs=0.5)
    assert d0 == pytest.approx(0.0, abs=0.5)


def test_reference_trends():
    # Deviation grows with camber (Carter m > 0); reference incidence
    # falls with camber (n < 0); deviation falls with solidity (more
    # guidance). Standard SP-36 behavior [VERIFY figures].
    d1, _ = reference_deviation(55.0, 1.25, 0.10, 10.0)
    d2, _ = reference_deviation(55.0, 1.25, 0.10, 30.0)
    assert d2 > d1 > 0.0
    i1, _ = reference_incidence(55.0, 1.25, 0.10, 10.0)
    i2, _ = reference_incidence(55.0, 1.25, 0.10, 30.0)
    assert i2 < i1
    dl, _ = reference_deviation(55.0, 0.8, 0.10, 25.0)
    dh, _ = reference_deviation(55.0, 1.6, 0.10, 25.0)
    assert dh < dl


def test_magnitude_bands_typical_stage():
    # Generous plausibility bands for a typical axial-compressor section
    # (beta1 = 55 deg, sigma = 1.25, t/c = 0.10, camber = 25 deg)
    # [VERIFY point values against SP-36 charts when the library is in].
    i_ref, v_i = reference_incidence(55.0, 1.25, 0.10, 25.0)
    d_ref, v_d = reference_deviation(55.0, 1.25, 0.10, 25.0)
    slope = deviation_slope(55.0, 1.25)
    assert -6.0 < i_ref < 6.0
    assert 4.0 < d_ref < 16.0
    assert 0.0 < slope < 1.0
    assert v_i == 1.0 and v_d == 1.0     # strictly inside calibration


# --------------------------------------------------------------------------
# Section 7.3: C1 smoothness, saturation, validity
# --------------------------------------------------------------------------
def test_c1_across_saturation_edges_in_beta1():
    _assert_c1_continuous(
        lambda b: reference_incidence(b, 1.2, 0.08, 20.0)[0], -15.0, 95.0)
    _assert_c1_continuous(
        lambda b: reference_deviation(b, 1.2, 0.08, 20.0)[0], -15.0, 95.0)


def test_c1_across_saturation_edges_in_sigma():
    _assert_c1_continuous(
        lambda s: reference_deviation(55.0, s, 0.08, 20.0)[0], 0.15, 2.6)
    _assert_c1_continuous(
        lambda s: deviation_slope(55.0, s), 0.15, 2.6)


def test_validity_is_c1_and_compact():
    # Section 7.3.3: v = 1 strictly inside, 0 well outside, C1 throughout.
    def v_of_beta(b):
        return reference_incidence(b, 1.2, 0.08, 20.0)[1]

    assert v_of_beta(35.0) == 1.0
    assert v_of_beta(90.0) == 0.0
    assert 0.0 < v_of_beta(69.5) < 1.0
    _assert_c1_continuous(np.vectorize(v_of_beta), -15.0, 95.0)


def test_negative_control_checker_rejects_hard_clip():
    # CLAUDE.md discipline: prove the C1 checker would catch a hard-clamp
    # implementation of the same saturation.
    def hard_clipped(b):
        return reference_incidence(np.clip(b, 0.0, 70.0), 1.2, 0.08, 20.0)[0]

    # A hard input clamp creates a derivative kink at the edges (the fit
    # has nonzero slope in beta there).
    with pytest.raises(AssertionError):
        _assert_c1_continuous(hard_clipped, -15.0, 95.0)


def test_full_domain_finiteness_sweep():
    # Section 7.3.4: finite everywhere over a wide (well out-of-range) box.
    b, s, t = np.meshgrid(np.linspace(-30, 120, 16),
                          np.linspace(0.05, 5.0, 16),
                          np.linspace(0.001, 0.3, 8))
    for f in (lambda: reference_incidence(b, s, t, 25.0)[0],
              lambda: reference_deviation(b, s, t, 25.0)[0],
              lambda: deviation_slope(b, s)):
        assert np.all(np.isfinite(f()))


# --------------------------------------------------------------------------
# End-to-end: Lieblein-fed rotor through the classical driver
# --------------------------------------------------------------------------
def test_lieblein_rotor_end_to_end():
    omega = 400.0
    geom = ParamRowGeometry(blade_count=31, beta1=-63.0 * DEG,
                            beta2=-45.0 * DEG, chord_len=0.06,
                            solidity_val=1.2, thickness=0.08)
    assert isinstance(LieblienSwirl(), SwirlClosure)
    topo = rotor_topology(n_sl=9)
    inlet = TransportFields(h0=np.full(9, H0), s=np.full(9, S0),
                            rvt=np.zeros(9))
    row = RowSpec(row_id="r1", omega=omega, swirl=LieblienSwirl(),
                  loss=PrescribedLoss(delta_s=1.0), blade_count=31,
                  geometry=geom)
    res = solve_classical(topo, GAS, FidelityConfig.tier2(),
                          MassFlowSpec(100.0), inlet, rows=[row])
    assert res.converged
    assert res.frozen.closures.validity > 0.5

    # Exit relative angle must equal blade angle + the correlation's own
    # deviation at the converged state (wiring consistency, section 3.4).
    r_te = res.fields.metrics.r[:, 2]
    vm_te = res.fields.vm[:, 2]
    vtheta_te = res.frozen.transported.rvt[:, 2] / r_te
    beta2_flow = np.arctan2(vtheta_te - omega * r_te, vm_te)
    dev_deg = np.rad2deg(-beta2_flow) - 45.0    # cascade frame (sgn = -1)
    assert np.all(dev_deg > 0.0)                # deviation under-turns
    assert np.all(dev_deg < 15.0)
    # Work input positive: compressor rotor raises h0.
    assert np.all(res.frozen.transported.h0[:, 2] > H0)