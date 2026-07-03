"""Contract tests for slcflow.fluid.PerfectGas.

Verifies thermodynamic self-consistency (round-trips), known reference values
for air, the isentropic stagnation identity, speed-of-sound consistency, and
vectorization. This backend is the oracle for future real-gas backends, so its
correctness is load-bearing.
"""
import numpy as np
import pytest

from slcflow.fluid import PerfectGas, StagState, WorkingFluid


@pytest.fixture
def air():
    return PerfectGas()  # gamma=1.4, R=287.05


# --------------------------------------------------------------------------
# Protocol conformance
# --------------------------------------------------------------------------
def test_satisfies_protocol(air):
    assert isinstance(air, WorkingFluid)


def test_derived_constants(air):
    assert air.cp == pytest.approx(1004.675, rel=1e-4)
    assert air.cv == pytest.approx(717.625, rel=1e-4)
    assert air.cp - air.cv == pytest.approx(air.R)


# --------------------------------------------------------------------------
# Known reference values (sea-level standard air)
# --------------------------------------------------------------------------
def test_reference_density_and_sound_speed(air):
    T, p = 288.15, 101325.0
    h = air.h_from_Tp(T, p)
    s = air.s_from_Tp(T, p)
    assert air.rho(h, s) == pytest.approx(1.2250, rel=1e-3)
    assert air.a(h, s) == pytest.approx(340.3, rel=1e-3)
    assert air.T(h, s) == pytest.approx(T)
    assert air.p(h, s) == pytest.approx(p, rel=1e-9)


def test_sound_speed_formula(air):
    T = np.array([250.0, 300.0, 400.0])
    p = np.full_like(T, 90000.0)
    h = air.h_from_Tp(T, p)
    s = air.s_from_Tp(T, p)
    assert np.allclose(air.a(h, s), np.sqrt(air.gamma * air.R * T))


# --------------------------------------------------------------------------
# Round-trip consistency
# --------------------------------------------------------------------------
def test_Tp_to_hs_to_Tp_roundtrip(air):
    T = np.array([220.0, 288.15, 350.0, 500.0])
    p = np.array([50000.0, 101325.0, 200000.0, 1.5e6])
    h = air.h_from_Tp(T, p)
    s = air.s_from_Tp(T, p)
    assert np.allclose(air.T(h, s), T, rtol=1e-12)
    assert np.allclose(air.p(h, s), p, rtol=1e-12)


def test_rho_matches_ideal_gas_law(air):
    T = np.array([300.0, 450.0])
    p = np.array([120000.0, 80000.0])
    h = air.h_from_Tp(T, p)
    s = air.s_from_Tp(T, p)
    assert np.allclose(air.rho(h, s), p / (air.R * T))


# --------------------------------------------------------------------------
# Stagnation / isentropic identities
# --------------------------------------------------------------------------
def test_stagnation_is_isentropic(air):
    T, p, V = 300.0, 100000.0, 200.0
    h = air.h_from_Tp(T, p)
    s = air.s_from_Tp(T, p)
    stag = air.stag_from_static(h, s, V)
    assert isinstance(stag, StagState)
    # entropy preserved
    assert stag.s == pytest.approx(s)
    # T0 = T + V^2/(2 cp)
    assert stag.T0 == pytest.approx(T + V * V / (2 * air.cp))
    # isentropic p0/p = (T0/T)^(gamma/(gamma-1))
    ratio = (stag.T0 / T) ** (air.gamma / (air.gamma - 1.0))
    assert stag.p0 / p == pytest.approx(ratio, rel=1e-10)


def test_stagnation_matches_mach_relations(air):
    # Cross-check against classic 1-D isentropic Mach relations.
    T, p = 288.15, 101325.0
    h = air.h_from_Tp(T, p)
    s = air.s_from_Tp(T, p)
    a = air.a(h, s)
    M = 0.8
    V = M * a
    stag = air.stag_from_static(h, s, V)
    g = air.gamma
    T0_over_T = 1 + 0.5 * (g - 1) * M * M
    p0_over_p = T0_over_T ** (g / (g - 1))
    assert stag.T0 / T == pytest.approx(T0_over_T, rel=1e-10)
    assert stag.p0 / p == pytest.approx(p0_over_p, rel=1e-10)


def test_static_stag_inverse(air):
    h0, V = 3.2e5, 180.0
    h = air.static_h_from_stag(h0, V)
    assert h == pytest.approx(h0 - 0.5 * V * V)
    # and stag_from_static recovers h0
    s = 100.0
    stag = air.stag_from_static(h, s, V)
    assert stag.h0 == pytest.approx(h0)


# --------------------------------------------------------------------------
# Entropy behavior
# --------------------------------------------------------------------------
def test_entropy_reference_zero(air):
    assert air.s_from_Tp(air.T_ref, air.p_ref) == pytest.approx(0.0)


def test_entropy_increases_with_T_decreases_with_p(air):
    s1 = air.s_from_Tp(300.0, 100000.0)
    s_hotter = air.s_from_Tp(350.0, 100000.0)
    s_higher_p = air.s_from_Tp(300.0, 200000.0)
    assert s_hotter > s1
    assert s_higher_p < s1


# --------------------------------------------------------------------------
# Non-air gas & vectorization
# --------------------------------------------------------------------------
def test_combustion_gas_constants():
    gas = PerfectGas(gamma=1.33, R=287.0)
    assert gas.cp == pytest.approx(1.33 * 287.0 / 0.33, rel=1e-6)


def test_broadcasting(air):
    T = np.array([[250.0], [300.0], [350.0]])   # (3,1)
    p = np.array([80000.0, 100000.0, 120000.0])  # (3,)
    h = air.h_from_Tp(T, p)
    s = air.s_from_Tp(T, p)
    rho = air.rho(h, s)
    assert rho.shape == (3, 3)