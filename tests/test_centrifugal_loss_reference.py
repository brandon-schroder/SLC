"""Reference-verified centrifugal internal-loss forms.

Pins the two base loss forms confirmed against Galvas NASA TN D-7487 / Aungier
/ Braembussche in ``docs/references/CENT-LOSS.md`` (extracted 2026-07-09,
citation-backed). Both are form + leading-coefficient checks (no clean single
constant beyond 0.5 and the 2*Cf leading factor).

The two modeling conventions (incidence ``f_inc``; the skin-friction mean
velocity) were RESOLVED to Aungier (2000) in 2026-07 and are pinned below.
"""
import numpy as np
import pytest

from slcflow.closures.centrifugal.loss import (
    CentrifugalLoss, incidence_loss, skin_friction_loss)


@pytest.mark.parametrize("wf,wb", [(120.0, 90.0), (60.0, 95.0), (0.0, 40.0)])
def test_incidence_loss_is_half_delta_wtheta_squared(wf, wb):
    # Galvas Eq 5.6: dh = 1/2 (dW_theta)^2 (full NASA KE, f_inc = 1).
    assert float(incidence_loss(wf, wb)) == pytest.approx(
        0.5 * (wf - wb) ** 2, rel=1e-12)


@pytest.mark.parametrize("w_avg,cf,ld", [(200.0, 0.005, 4.0),
                                         (150.0, 0.008, 3.0)])
def test_skin_friction_leading_factor_is_2cf(w_avg, cf, ld):
    # Galvas: dh = 4 Cf (L/D)(W^2/2) = 2 Cf (L/D) W^2.
    assert float(skin_friction_loss(w_avg, cf, ld)) == pytest.approx(
        2.0 * cf * ld * w_avg ** 2, rel=1e-12)


def test_incidence_zero_when_congruent():
    # No tangential mismatch -> no incidence loss.
    assert float(incidence_loss(85.0, 85.0)) == pytest.approx(0.0, abs=1e-12)


def test_f_inc_default_is_aungier_0p8():
    # Resolved 2026-07 (CENT-LOSS.md): the incidence-KE fraction actually lost
    # is a genuine 0.5-1.0 family (Conrad 0.5-0.7 / Aungier 0.8 / Galvas 1.0);
    # the default adopts Aungier 0.8 (coherent with mean-of-squares friction),
    # exposed as a tunable field.
    assert CentrifugalLoss().f_inc == pytest.approx(0.8)


@pytest.mark.parametrize("w1,w2", [(260.0, 150.0), (200.0, 200.0), (90.0, 180.0)])
def test_skin_friction_mean_of_squares_convention(w1, w2):
    # Resolved 2026-07: CentrifugalLoss passes the RMS velocity so the squared
    # term is Aungier's mean-of-squares 1/2(W1^2+W2^2), NOT the square-of-mean
    # [1/2(W1+W2)]^2. Pin the identity and the convexity gap (mean-of-squares
    # >= square-of-mean, strict when W1 != W2).
    cf, ld = 0.005, 4.0
    w_rms = np.sqrt(0.5 * (w1 ** 2 + w2 ** 2))
    got = float(skin_friction_loss(w_rms, cf, ld))
    assert got == pytest.approx(2.0 * cf * ld * 0.5 * (w1 ** 2 + w2 ** 2),
                                rel=1e-12)
    sq_of_mean = float(skin_friction_loss(0.5 * (w1 + w2), cf, ld))
    if w1 == w2:
        assert got == pytest.approx(sq_of_mean, rel=1e-12)
    else:
        assert got > sq_of_mean          # convexity: mean-of-squares is larger
