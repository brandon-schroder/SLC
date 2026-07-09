"""Reference-verified Aungier ch.6 fit constants for the Lieblein set.

Pins the incidence/deviation/off-design-slope fit COEFFICIENTS confirmed
term-by-term against Aungier *Axial-Flow Compressors* (2003) ch. 6 in
``docs/references/AUN-C.md`` (extracted 2026-07-09 from the NotebookLM theory
library, citation-backed, equation numbers Aungier's own). A drift in any
constant turns this red -- including a regression of the K_ti fix (Eq 6-11
uses (t/c)^0.3, not the old (10 t/c)^0.3).

Tolerance note: the closure soft-saturates its inputs (C1 domain safety), so
even deep in-domain the code differs from the raw fit by <0.1%. The rtol here
absorbs that but is far tighter than any single-constant error (>=~1%). Points
are chosen central in the calibrated domain to keep saturation negligible.
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.lieblein import (
    deviation_slope, reference_deviation, reference_incidence)

RTOL = 3e-3


def _i0_10(b, s):                       # Aungier 6-13/6-14
    p = 0.914 + s ** 3 / 160.0
    return b ** p / (5.0 + 46.0 * np.exp(-2.3 * s)) \
        - 0.1 * s ** 3 * np.exp((b - 70.0) / 4.0)


def _k_ti(t):                           # Aungier 6-10/6-11 ((t/c)^0.3, no x10)
    return (10.0 * t) ** (0.28 / (0.1 + t ** 0.3))


def _n(b, s):                           # Aungier 6-15
    return 0.025 * s - 0.06 - (b / 90.0) ** (1.0 + 1.2 * s) / (1.5 + 0.43 * s)


def _d0_10(b, s):                       # Aungier 6-20
    return 0.01 * s * b + (0.74 * s ** 1.9 + 3.0 * s) * (b / 90.0) ** (1.67 + 1.09 * s)


def _k_td(t):                           # Aungier 6-25
    return 6.25 * t + 37.5 * t * t


def _m(b, s):                           # Aungier 6-21/6-22/6-24
    m10 = 0.17 - 0.0333 * (b / 100.0) + 0.333 * (b / 100.0) ** 2
    bexp = 0.9625 - 0.17 * (b / 100.0) - 0.85 * (b / 100.0) ** 3
    return m10 / s ** bexp


def _slope(b, s):                       # Aungier 6-76
    return (1.0 + (s + 0.25 * s ** 4) * (b / 53.0) ** 2.5) / np.exp(3.1 * s)


@pytest.mark.parametrize("b,s,t", [(45.0, 1.0, 0.08), (40.0, 1.2, 0.10),
                                   (50.0, 0.9, 0.06)])
def test_reference_incidence_i0_and_kti(b, s, t):
    # camber=0 isolates K_ti*(i0)_10 (Eqs 6-13/6-14 and the 6-10/6-11 fix).
    got = float(reference_incidence(b, s, t, 0.0)[0])
    assert got == pytest.approx(_k_ti(t) * _i0_10(b, s), rel=RTOL)


def test_kti_fix_bites_off_reference_thickness():
    # The K_ti fix only shows away from t/c=0.10. Pin that the coded value
    # tracks Aungier (t/c)^0.3 and NOT the old (10 t/c)^0.3 at t/c=0.08.
    b, s, t = 45.0, 1.0, 0.08
    aungier = _k_ti(t) * _i0_10(b, s)
    old_bug = (10.0 * t) ** (0.28 / (0.1 + (10.0 * t) ** 0.3)) * _i0_10(b, s)
    got = float(reference_incidence(b, s, t, 0.0)[0])
    assert got == pytest.approx(aungier, rel=RTOL)
    assert abs(got - old_bug) > 0.01 * abs(aungier)   # distinct from the bug


@pytest.mark.parametrize("b,s", [(45.0, 1.0), (35.0, 1.4)])
def test_camber_slope_n_matches_6_15(b, s):
    # inc(camber=C) - inc(camber=0) = n*C isolates the slope factor n.
    c = 20.0
    n_code = (float(reference_incidence(b, s, 0.08, c)[0])
              - float(reference_incidence(b, s, 0.08, 0.0)[0])) / c
    assert n_code == pytest.approx(_n(b, s), abs=2e-3)


@pytest.mark.parametrize("b,s,t", [(45.0, 1.0, 0.08), (40.0, 1.2, 0.10),
                                   (55.0, 0.8, 0.06)])
def test_reference_deviation_d0_and_ktd(b, s, t):
    got = float(reference_deviation(b, s, t, 0.0)[0])
    assert got == pytest.approx(_k_td(t) * _d0_10(b, s), rel=RTOL)


@pytest.mark.parametrize("b,s", [(45.0, 1.0), (35.0, 1.4)])
def test_camber_slope_m_matches_6_21(b, s):
    c = 20.0
    m_code = (float(reference_deviation(b, s, 0.08, c)[0])
              - float(reference_deviation(b, s, 0.08, 0.0)[0])) / c
    assert m_code == pytest.approx(_m(b, s), abs=2e-3)


@pytest.mark.parametrize("b,s", [(45.0, 1.0), (30.0, 1.5), (55.0, 0.8)])
def test_deviation_slope_matches_6_76(b, s):
    assert float(deviation_slope(b, s)) == pytest.approx(_slope(b, s), rel=RTOL)
