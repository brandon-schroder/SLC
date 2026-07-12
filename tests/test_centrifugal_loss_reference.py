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
    CentrifugalLoss, blade_loading_loss, incidence_loss, skin_friction_loss)


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


# --------------------------------------------------------------------------
# Blade-loading (diffusion) loss -- Coppage/Aungier Eq 5.15 (added 2026-07)
# --------------------------------------------------------------------------
def test_blade_loading_matches_coppage_oh1997():
    # dh = 0.05 D_f^2 U2^2 with the radial diffusion factor VERBATIM from
    # Oh-Yoon-Chung (1997) Table 6 / the equation above Table 5:
    #   D_f = 1 - W2/W1 + 0.75 (dh_Euler/U2^2) / { (W1/W2)[(Z/pi)(1-r1/r2)
    #                                                       + 2 r1/r2] }
    # i.e. the loading term carries W2/W1 (W1/W2 in the DENOMINATOR). Values
    # chosen so the D_f ceiling (2.5) and the 1 m/s velocity floors are inactive,
    # so the code must equal the plain formula.
    w1, w2, u2, dh_euler, z, rr = 240.0, 110.0, 340.0, 78000.0, 17, 0.55
    geom = (z / np.pi) * (1.0 - rr) + 2.0 * rr
    d_f = 1.0 - w2 / w1 + 0.75 * (dh_euler / u2 ** 2) * (w2 / w1) / geom
    assert d_f < 2.5                                   # ceiling inactive
    expected = 0.05 * d_f ** 2 * u2 ** 2
    assert float(blade_loading_loss(w1, w2, u2, dh_euler, z, rr)) == \
        pytest.approx(expected, rel=2e-3)


def test_blade_loading_uses_w2_over_w1_not_w1_over_w2():
    # Regression guard for the 2026-07 ratio-inversion fix (CENT-LOSS.md):
    # the Oh-1997 loading term is W2/W1 (<1 under diffusion), NOT the inverted
    # W1/W2 (>1) the closure used before. At a diffusing point (W2 < W1) the
    # correct form gives a STRICTLY SMALLER loss than the inverted one.
    w1, w2, u2, dh_euler, z, rr = 240.0, 110.0, 340.0, 78000.0, 17, 0.55
    geom = (z / np.pi) * (1.0 - rr) + 2.0 * rr
    dq = dh_euler / u2 ** 2
    df_correct = 1.0 - w2 / w1 + 0.75 * dq * (w2 / w1) / geom
    df_inverted = 1.0 - w2 / w1 + 0.75 * dq * (w1 / w2) / geom
    got = float(blade_loading_loss(w1, w2, u2, dh_euler, z, rr))
    assert got == pytest.approx(0.05 * df_correct ** 2 * u2 ** 2, rel=2e-3)
    assert got < 0.05 * df_inverted ** 2 * u2 ** 2       # correct < inverted


def test_blade_loading_grows_with_loading():
    # The blade-loading loss is a LOADING loss: at fixed velocities it grows
    # with the aerodynamic loading (Euler work dh_euler). Diffusion itself is
    # carried by the leading 1 - W2/W1 term; the W2/W1 loading term is a
    # positive correction proportional to dh_euler (Oh 1997).
    w1, w2, u2, z, rr = 240.0, 110.0, 340.0, 17, 0.55
    losses = [float(blade_loading_loss(w1, w2, u2, dh, z, rr))
              for dh in (40000.0, 60000.0, 80000.0, 100000.0)]
    assert all(a < b for a, b in zip(losses, losses[1:]))


def test_blade_loading_ceiling_bounds_tiny_w2():
    # The smooth D_f ceiling (2.5) keeps a transient tiny W2 from blowing the
    # loss up (the axial omega-bar-ceiling analogue, section 7.3.2): finite and
    # bounded by ~0.05 * 2.5^2 * U2^2.
    u2 = 340.0
    big = float(blade_loading_loss(240.0, 1e-3, u2, 78000.0, 17, 0.55))
    assert np.isfinite(big)
    assert big <= 0.05 * 2.6 ** 2 * u2 ** 2
