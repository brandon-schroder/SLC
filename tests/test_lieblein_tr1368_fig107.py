"""TR-1368 raw-data validation of the off-design deviation slope (section 4.3).

Distinct from ``test_lieblein_reference.py`` (Aungier fit COEFFICIENTS) and
``test_lieblein_sp36_charts.py`` (derived SP-36 chart outputs): this pins the
code against the PRIMARY measured cascade data -- NACA TR-1368 / RM L51G31
Fig. 107, the measured design-point turning-angle-vs-angle-of-attack slope
``(d theta / d alpha)_d`` per (beta1, sigma). With fixed blade geometry,

    (d theta / d alpha)_d = 1 - (d delta / d i)  =  1 - deviation_slope(...)

so the chart validates ``lieblein.deviation_slope`` directly at the raw-data
level with no camber-equivalence step (section 4.3).

Digitization: tools/digitize_tr1368_fig107.py (grid-anchored calibration +
seeded curve tracing + overlay verification; reading precision ~ +/-0.01).
Provenance + findings: docs/references/TR1368.md.

Measured agreement (2026-07-15): overall RMS 0.030 / mean +0.002 over 39
points; beta1=30 within 0.013; the chart fairs beta1=30 and 45 into one curve
and the code's 30-vs-45 spread (~0.03) is the same order as that fairing;
beta1=70 low-sigma is a DOCUMENTED deviation region (fit sigma-dependence
shallower than measured, worst +0.097 at sigma=1.0 -- the calibrated-domain
edge). Tolerances below pin those measured levels: an unnoticed drift of the
fit form turns this red, while the known deviation region is bounded, not
hidden.
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.lieblein import deviation_slope

# Traced Fig. 107 samples (see tools/digitize_tr1368_fig107.py).
FAIRED_30_45 = {0.5: 0.748, 0.6: 0.812, 0.7: 0.855, 0.8: 0.890, 0.9: 0.923,
                1.0: 0.945, 1.1: 0.959, 1.2: 0.972, 1.3: 0.980, 1.4: 0.983,
                1.5: 0.979}
B60 = {0.5: 0.594, 0.6: 0.651, 0.7: 0.726, 0.8: 0.773, 0.9: 0.830,
       1.0: 0.871, 1.1: 0.896, 1.2: 0.924, 1.3: 0.947, 1.4: 0.965,
       1.5: 0.979}
B70 = {1.0: 0.751, 1.1: 0.807, 1.2: 0.852, 1.3: 0.907, 1.4: 0.943,
       1.5: 0.979}


def slope_d(beta1: float, sigma: float) -> float:
    """(d theta / d alpha)_d predicted by the code (section 4.3)."""
    return 1.0 - float(deviation_slope(beta1, sigma))


@pytest.mark.parametrize("sigma,chart", sorted(FAIRED_30_45.items()))
def test_beta30_matches_fig107(sigma, chart):
    # beta1=30 branch of the faired 30/45 curve: the tight family (section
    # 4.3). Measured max |err| 0.013 -> tolerance 0.025.
    assert slope_d(30.0, sigma) == pytest.approx(chart, abs=0.025)


@pytest.mark.parametrize("sigma,chart", sorted(FAIRED_30_45.items()))
def test_beta45_within_fairing_band_of_fig107(sigma, chart):
    # beta1=45 reads ~0.02-0.03 below the faired curve -- the same order as
    # the chart's own 30/45 fairing width. Bounded at 0.05.
    assert slope_d(45.0, sigma) == pytest.approx(chart, abs=0.05)


@pytest.mark.parametrize("sigma,chart", sorted(B60.items()))
def test_beta60_matches_fig107(sigma, chart):
    # Measured max |err| 0.059 (low sigma) -> tolerance 0.075.
    assert slope_d(60.0, sigma) == pytest.approx(chart, abs=0.075)


@pytest.mark.parametrize("sigma,chart", sorted(B70.items()))
def test_beta70_bounded_against_fig107(sigma, chart):
    # DOCUMENTED deviation region (calibrated-domain edge): fit shallower in
    # sigma than the data, measured worst +0.097 at sigma=1.0. Bound 0.12 --
    # this is a tripwire against getting WORSE, not a claim of agreement.
    assert slope_d(70.0, sigma) == pytest.approx(chart, abs=0.12)


def test_fig107_aggregate_rms_and_bias():
    # Whole-chart aggregate (section 4.3): measured RMS 0.030, mean +0.002.
    # Guards a systematic drift the per-point bounds could each absorb.
    errs = []
    for beta in (30.0, 45.0):
        errs += [slope_d(beta, s) - v for s, v in FAIRED_30_45.items()]
    errs += [slope_d(60.0, s) - v for s, v in B60.items()]
    errs += [slope_d(70.0, s) - v for s, v in B70.items()]
    e = np.asarray(errs)
    assert float(np.sqrt(np.mean(e * e))) < 0.045
    assert abs(float(np.mean(e))) < 0.02


def test_fig107_shape_slope_rises_with_solidity():
    # Structural check independent of absolute level: at every measured
    # beta1 the design turning slope RISES with solidity over the chart's
    # span, as the data shows (more guidance -> turning follows the blade).
    for beta in (30.0, 45.0, 60.0, 70.0):
        vals = [slope_d(beta, s) for s in (0.5, 0.8, 1.1, 1.5)]
        assert all(b > a for a, b in zip(vals, vals[1:]))
