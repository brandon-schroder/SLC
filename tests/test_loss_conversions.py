"""Tests for slcflow.closures.conversions (Theory Manual section 4.4,
Appendix B — each assertion cites its clause).

The independent oracle throughout is the PerfectGas entropy function:
``s_from_Tp(T0, p0_actual) - s_from_Tp(T0, p0_ideal)`` must equal the
conversion's delta_s, since ``-R ln(p_a/p_i)`` IS the perfect-gas entropy
difference at common temperature.

Provenance: M4 sub-step 2, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures import conversions as cv
from slcflow.fluid.perfectgas import PerfectGas

GAS = PerfectGas()

# A representative rotor-inlet relative state (air, SI).
T1, P1 = 288.0, 9.0e4
W1 = 180.0
U1, U2 = 250.0, 320.0     # radius change: U2 > U1 (centrifugal-ish)


def rel_stag_inlet():
    return cv.relative_stagnation(GAS, T1, P1, W1)


# --------------------------------------------------------------------------
# Master relation + relative stagnation
# --------------------------------------------------------------------------
def test_master_relation_matches_entropy_oracle():
    # Appendix B master relation == perfect-gas entropy difference at
    # common T0 (independent oracle: fluid.s_from_Tp).
    T0 = 400.0
    p0_id, p0_act = 2.0e5, 1.8e5
    ds = cv.delta_s_from_p0_deficit(GAS, p0_act, p0_id)
    oracle = GAS.s_from_Tp(T0, p0_act) - GAS.s_from_Tp(T0, p0_id)
    assert ds == pytest.approx(oracle, rel=1e-12)
    assert ds > 0.0
    # Lossless: exactly zero.
    assert cv.delta_s_from_p0_deficit(GAS, p0_id, p0_id) == 0.0


def test_relative_stagnation_is_isentropic_recovery():
    T0r, p0r = rel_stag_inlet()
    # Same entropy as the static state (isentropic, section 3.7 oracle).
    assert GAS.s_from_Tp(T0r, p0r) == pytest.approx(GAS.s_from_Tp(T1, P1),
                                                    abs=1e-10)
    assert T0r == pytest.approx(T1 + W1**2 / (2 * GAS.cp))


# --------------------------------------------------------------------------
# B.1 ideal exit state
# --------------------------------------------------------------------------
def test_b1_stator_degeneracy():
    # B.1: stator/duct — ideal exit state is the inlet stagnation state.
    T0r, p0r = rel_stag_inlet()
    T0_2, p0_2 = cv.ideal_exit_relative_stagnation(GAS, T0r, p0r, 0.0, 0.0)
    assert T0_2 == T0r and p0_2 == p0r


def test_b1_rotor_rereferencing_is_isentropic():
    # B.1: the loss-free re-referencing across the radius change is an
    # isentropic process at constant rothalpy — entropy oracle must agree,
    # and T0r must rise by (U2^2 - U1^2)/(2 cp). This is the radial-machine
    # correction the manual bans omitting.
    T0r_1, p0r_1 = rel_stag_inlet()
    T0r_2, p0r_2id = cv.ideal_exit_relative_stagnation(GAS, T0r_1, p0r_1,
                                                       U1, U2)
    assert T0r_2 == pytest.approx(T0r_1 + (U2**2 - U1**2) / (2 * GAS.cp))
    assert GAS.s_from_Tp(T0r_2, p0r_2id) == pytest.approx(
        GAS.s_from_Tp(T0r_1, p0r_1), abs=1e-10)
    assert p0r_2id > p0r_1    # compression along the radius increase


# --------------------------------------------------------------------------
# B.2 compressor omega_bar
# --------------------------------------------------------------------------
def test_b2_omega_bar_definition_roundtrip():
    # B.2: recovered p0r_2 must satisfy the coefficient definition, and
    # delta_s must match the entropy oracle at the exit state.
    T0r_1, p0r_1 = rel_stag_inlet()
    omega_bar = 0.08
    ds, p0r_2 = cv.delta_s_compressor_omega_bar(GAS, omega_bar, T0r_1,
                                                p0r_1, P1, U1, U2)
    _, p0r_2id = cv.ideal_exit_relative_stagnation(GAS, T0r_1, p0r_1, U1, U2)
    assert (p0r_2id - p0r_2) / (p0r_1 - P1) == pytest.approx(omega_bar,
                                                             rel=1e-12)
    T0r_2 = T0r_1 + (U2**2 - U1**2) / (2 * GAS.cp)
    oracle = GAS.s_from_Tp(T0r_2, p0r_2) - GAS.s_from_Tp(T0r_2, p0r_2id)
    assert ds == pytest.approx(oracle, rel=1e-12)
    assert ds > 0.0


def test_b2_zero_loss_zero_entropy():
    T0r_1, p0r_1 = rel_stag_inlet()
    ds, _ = cv.delta_s_compressor_omega_bar(GAS, 0.0, T0r_1, p0r_1, P1,
                                            U1, U2)
    assert ds == 0.0


def test_b2_vectorized_and_monotone():
    # Section 7.3.4 sweep: finite and strictly increasing in omega_bar
    # over the admissible domain.
    T0r_1, p0r_1 = rel_stag_inlet()
    w = np.linspace(0.0, 0.5, 200)
    ds, _ = cv.delta_s_compressor_omega_bar(GAS, w, T0r_1, p0r_1, P1, U1, U2)
    assert ds.shape == w.shape and np.all(np.isfinite(ds))
    assert np.all(np.diff(ds) > 0.0)


# --------------------------------------------------------------------------
# B.3 turbine Y
# --------------------------------------------------------------------------
def test_b3_Y_definition_roundtrip():
    # B.3: the manual's rearrangement must reproduce the exit-dynamic-head
    # definition Y = (p0r_2,id - p0r_2)/(p0r_2 - p2), and the entropy must
    # match the oracle.
    T0r_1, p0r_1 = rel_stag_inlet()
    p2, Y = 7.0e4, 0.15
    ds, p0r_2 = cv.delta_s_turbine_Y(GAS, Y, T0r_1, p0r_1, p2, U1, U2)
    _, p0r_2id = cv.ideal_exit_relative_stagnation(GAS, T0r_1, p0r_1, U1, U2)
    assert (p0r_2id - p0r_2) / (p0r_2 - p2) == pytest.approx(Y, rel=1e-12)
    T0r_2 = T0r_1 + (U2**2 - U1**2) / (2 * GAS.cp)
    oracle = GAS.s_from_Tp(T0r_2, p0r_2) - GAS.s_from_Tp(T0r_2, p0r_2id)
    assert ds == pytest.approx(oracle, rel=1e-12)
    assert ds > 0.0
    # Y = 0: lossless.
    ds0, _ = cv.delta_s_turbine_Y(GAS, 0.0, T0r_1, p0r_1, p2, U1, U2)
    assert ds0 == 0.0


# --------------------------------------------------------------------------
# B.4 kinetic-energy zeta
# --------------------------------------------------------------------------
def test_b4_zeta_matches_entropy_oracle():
    # B.4: delta_s = cp ln(T2/T2s) == s(T2, p2) - s(T2s, p2) at common p2.
    T2, V2, zeta, p2 = 300.0, 200.0, 0.06, 8.0e4
    ds = cv.delta_s_kinetic_energy_zeta(GAS, zeta, T2, V2)
    T2s = T2 - zeta * V2**2 / (2 * GAS.cp)
    oracle = GAS.s_from_Tp(T2, p2) - GAS.s_from_Tp(T2s, p2)
    assert ds == pytest.approx(oracle, rel=1e-12)
    assert ds > 0.0
    assert cv.delta_s_kinetic_energy_zeta(GAS, 0.0, T2, V2) == 0.0


def test_b4_guard_asserts_not_clamps():
    # B.4: "assert, don't clamp" — an unsaturated coefficient reaching the
    # conversion is an upstream correlation bug and must fail loudly.
    with pytest.raises(AssertionError, match="B.4 guard"):
        cv.delta_s_kinetic_energy_zeta(GAS, 20.0, 250.0, 600.0)


# --------------------------------------------------------------------------
# B.5.2: individual conversion then summation
# --------------------------------------------------------------------------
def test_b52_components_convert_individually():
    # B.5.2: delta_s of components converted individually and summed is NOT
    # the delta_s of summed coefficients (log is nonlinear) — the rule
    # exists because the difference is real. Both computed here to pin it.
    T0r_1, p0r_1 = rel_stag_inlet()
    w_profile, w_tip = 0.05, 0.03
    ds_p, _ = cv.delta_s_compressor_omega_bar(GAS, w_profile, T0r_1, p0r_1,
                                              P1, U1, U2)
    ds_t, _ = cv.delta_s_compressor_omega_bar(GAS, w_tip, T0r_1, p0r_1,
                                              P1, U1, U2)
    ds_sum, _ = cv.delta_s_compressor_omega_bar(GAS, w_profile + w_tip,
                                                T0r_1, p0r_1, P1, U1, U2)
    assert ds_p + ds_t == pytest.approx(ds_sum, rel=1e-2)  # close...
    assert ds_p + ds_t != ds_sum                           # ...but not equal
