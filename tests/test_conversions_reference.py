"""Reference-verified Appendix-B loss->entropy definitions.

Pins the coefficient definitions + reference dynamic heads confirmed against
Denton/Cumpsty/Aungier/Dixon/Lakshminarayana in ``docs/references/CONV-B.md``
(2026-07-09). These are the foundational conversions every loss set routes
through, so the reference-head (inlet vs exit) is what matters. Stator inputs
(u1=u2=0) isolate the definition from the rothalpy re-referencing.
"""
import numpy as np
import pytest

from slcflow.closures import conversions as cv
from slcflow.fluid.perfectgas import PerfectGas

GAS = PerfectGas()
R = GAS.R


def test_master_relation_is_minus_R_ln_p0_ratio():
    # Denton 4a: delta_s = -R ln(p0_actual / p0_ideal).
    ds = float(cv.delta_s_from_p0_deficit(GAS, 0.95e5, 1.0e5))
    assert ds == pytest.approx(-R * np.log(0.95e5 / 1.0e5), rel=1e-12)


def test_compressor_omega_bar_uses_inlet_dynamic_head():
    # B.2 / Cumpsty: omega_bar = (p0_id - p0_2)/(p01 - p1), INLET dyn head.
    # Stator (u1=u2=0) -> p0_id = p0r1, so p0_2 = p0r1 - omega*(p0r1 - p1).
    p0r1, p1, omega = 1.20e5, 1.00e5, 0.06
    ds, p0r2 = cv.delta_s_compressor_omega_bar(
        GAS, omega, T0r_1=350.0, p0r_1=p0r1, p1=p1, u1=0.0, u2=0.0)
    assert float(p0r2) == pytest.approx(p0r1 - omega * (p0r1 - p1), rel=1e-9)
    # Recover omega from the inlet-referenced definition.
    omega_back = (p0r1 - float(p0r2)) / (p0r1 - p1)
    assert omega_back == pytest.approx(omega, rel=1e-9)
    assert float(ds) == pytest.approx(-R * np.log(float(p0r2) / p0r1),
                                      rel=1e-9)


def test_turbine_Y_uses_exit_dynamic_head():
    # B.3 / Aungier: Y = (p0_id - p0_2)/(p0_2 - p2), EXIT ("discharge") dyn
    # head. Stator -> p0_id = p0r1. Check the exit-referenced identity holds.
    p0r1, p2, Y = 1.50e5, 1.00e5, 0.08
    ds, p0r2 = cv.delta_s_turbine_Y(
        GAS, Y, T0r_1=500.0, p0r_1=p0r1, p2=p2, u1=0.0, u2=0.0)
    Y_back = (p0r1 - float(p0r2)) / (float(p0r2) - p2)   # EXIT reference
    assert Y_back == pytest.approx(Y, rel=1e-9)
    assert float(ds) == pytest.approx(-R * np.log(float(p0r2) / p0r1),
                                      rel=1e-9)


def test_kinetic_energy_zeta_definition():
    # B.4 / Denton: T2s = T2 - zeta V2^2/(2 cp); delta_s = cp ln(T2/T2s).
    zeta, T2, V2 = 0.05, 300.0, 200.0
    ds = float(cv.delta_s_kinetic_energy_zeta(GAS, zeta, T2, V2))
    T2s = T2 - zeta * V2 * V2 / (2.0 * GAS.cp)
    assert ds == pytest.approx(GAS.cp * np.log(T2 / T2s), rel=1e-12)


def test_rothalpy_rereference_stator_is_identity():
    # B.1: u1=u2=0 -> ideal exit stagnation equals inlet exactly.
    T0r2, p0r2id = cv.ideal_exit_relative_stagnation(
        GAS, T0r_1=400.0, p0r_1=1.3e5, u1=0.0, u2=0.0)
    assert float(T0r2) == pytest.approx(400.0, rel=1e-12)
    assert float(p0r2id) == pytest.approx(1.3e5, rel=1e-12)
