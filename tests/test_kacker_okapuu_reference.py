"""Reference-verified constants for the Kacker-Okapuu loss chain.

Unlike ``test_kacker_okapuu.py`` (structural bands/trends/C1), these pin the
*exact scalar constants* of the K-O / Dunham-Came / Ainley-Mathieson method
that were confirmed term-by-term against the source in
``docs/references/KO82.md`` (extracted 2026-07-09 from the cleaned NotebookLM
loss-model library, citation-backed). If a future refactor drifts one of
these numbers, that is a regression against the paper, and this file goes red.

Scope: the confirmed *formula* constants, plus (2026-07) the nozzle/impulse
``yp1``/``yp2`` reference curves now that they are calibrated to digitized
points off Ainley-Mathieson R&M 2974 Fig. 4 (``tools/digitize_am_fig4.py``,
KO82.md). The TE ``phi2`` curve remains a surrogate NOT pinned here (its
source is the paywalled K-O TE figure, not in the library — still [VERIFY]).
"""
import numpy as np
import pytest

from slcflow.closures.axial_turbine.kacker_okapuu import (
    _aspect_ratio_factor, mach_profile_correction, profile_loss_am,
    reynolds_correction, secondary_loss, shock_loss, trailing_edge_zeta)

DEG = np.pi / 180.0

# Digitized minima off Ainley-Mathieson R&M 2974 Fig. 4 (t/c=20%, Re=2e5,
# M<0.6): (alpha2_deg, s/c at minimum, Y_p,min). These ARE the reference the
# yp1/yp2 surrogate is calibrated to (tools/digitize_am_fig4.py fits the same
# points); read to ~+/-0.005 off the 1951 raster chart, cross-checked against
# an automated column scan and the canonical Dixon / CRS values. See KO82.md.
NOZZLE = [(40, 0.86, 0.021), (50, 0.83, 0.023), (60, 0.79, 0.026),
          (65, 0.76, 0.030), (70, 0.72, 0.035), (75, 0.68, 0.042),
          (80, 0.63, 0.049)]
IMPULSE = [(40, 0.75, 0.067), (50, 0.72, 0.075), (55, 0.70, 0.086),
           (60, 0.65, 0.102), (65, 0.62, 0.115), (70, 0.58, 0.135)]


@pytest.mark.parametrize("a2,sc,ymin", NOZZLE)
def test_nozzle_curve_matches_am_fig4a(a2, sc, ymin):
    # yp1 (beta1=0, r=0) reproduces the digitized R&M 2974 Fig. 4a nozzle
    # minima. t/c=0.20 (the chart condition) makes the thickness factor unity.
    # Tolerance = chart read precision (~+/-0.005) + fit residual; a drift in
    # the u^4 level law or the optimum-pitch line turns this red.
    got = float(profile_loss_am(sc, 0.0, float(a2), 0.20)[0])
    assert got == pytest.approx(ymin, abs=6e-3)


@pytest.mark.parametrize("a2,sc,ymin", IMPULSE)
def test_impulse_curve_matches_am_fig4b(a2, sc, ymin):
    # yp2 (beta1=beta2, r=1) reproduces the digitized Fig. 4b impulse minima.
    got = float(profile_loss_am(sc, float(a2), float(a2), 0.20)[0])
    assert got == pytest.approx(ymin, abs=6e-3)


def test_profile_loss_floor_width_is_loss_scaled_not_angle_scaled():
    # Regression for the 2026-07 fix: the AD-10 positivity floor's smooth_max
    # transition width must be LOSS-scaled (Y_p ~ 0.02-0.15), not the
    # angle-ratio _R_W=0.1. smooth_max overestimates by up to width*ln2, so a
    # 0.1 width inflated every nozzle/impulse loss by ~0.06. Guard it: the
    # nozzle minimum must sit near the digitized ~0.021 (Fig. 4a, a2=40), NOT
    # ~0.084 as the mis-scaled width produced.
    y = float(profile_loss_am(0.86, 0.0, 40.0, 0.20)[0])
    assert y < 0.030          # would be ~0.084 with the 0.1 angle-ratio width
    assert y == pytest.approx(0.021, abs=6e-3)


def test_reynolds_exponents_match_ko82():
    # KO82: f_Re = (Re/2e5)^-0.4 (Re<=2e5), (Re/1e6)^-0.2 (Re>1e6).
    # Evaluate deep in each branch (past the C1 blend) so the exponent shows.
    assert float(reynolds_correction(2.0e4)) == pytest.approx(
        (2.0e4 / 2.0e5) ** (-0.4), rel=1e-4)          # 10^0.4
    assert float(reynolds_correction(1.0e7)) == pytest.approx(
        (1.0e7 / 1.0e6) ** (-0.2), rel=1e-4)          # 10^-0.2
    # Unity flat band between the knees (KO82).
    assert float(reynolds_correction(5.0e5)) == pytest.approx(1.0, abs=1e-6)


def test_mach_K1_ramp_slope_match_ko82():
    # With M1=M2 -> K2=(M1/M2)^2=1 -> K_p = K1(M2). KO82 endpoints: K1=1 at
    # M2=0.2, K1=0 at M2=1.0; the coded linear surrogate has slope 1.25.
    assert float(mach_profile_correction(0.6, 0.6)) == pytest.approx(
        0.5, abs=2e-3)                                  # 1 - 1.25*(0.6-0.2)
    assert float(mach_profile_correction(0.4, 0.4)) == pytest.approx(
        0.75, abs=5e-3)                                 # 1 - 1.25*(0.4-0.2)
    assert float(mach_profile_correction(0.1, 0.1)) == pytest.approx(
        1.0, abs=1e-2)     # M2<0.2 -> K1=1 (C1-smoothed near the 0.2 knee)


def test_aspect_ratio_factor_match_ko82():
    # KO82: f_AR = (1 - 0.25*sqrt(2-AR))/AR for AR<=2, 1/AR for AR>2.
    assert float(_aspect_ratio_factor(1.0)) == pytest.approx(
        (1.0 - 0.25 * np.sqrt(2.0 - 1.0)) / 1.0, rel=1e-3)   # 0.75
    assert float(_aspect_ratio_factor(4.0)) == pytest.approx(0.25, rel=1e-6)
    # Continuous at AR=2, both branches -> 0.5 (blend puts it right at ~0.5).
    assert float(_aspect_ratio_factor(2.0)) == pytest.approx(0.5, abs=0.02)


def test_shock_coefficient_and_exponent_match_ko82():
    # KO82 hub shock: proportional to 0.75*(M1-0.4)^1.75.
    assert float(shock_loss(1.4)[0]) == pytest.approx(0.75, rel=1e-4)
    # Exponent 1.75 from the ratio at excess 2 vs 1 (softplus ~ identity here).
    ratio = float(shock_loss(2.4)[0]) / float(shock_loss(1.4)[0])
    assert ratio == pytest.approx(2.0 ** 1.75, rel=1e-3)


def test_te_zeta_to_Y_is_incompressible_limit_of_ko_relation():
    # axial_turbine/loss.py maps the TE energy coefficient zeta to Y via
    # Y_TE = zeta/(1-zeta). KO82's exact compressible relation (verbatim,
    # KO82.md) is Y = {[1-(g-1)/2 M2^2(1/phi^2-1)]^(-g/(g-1)) - 1} /
    # {1-(1+(g-1)/2 M2^2)^(-g/(g-1))} with phi^2 = 1-zeta. Its M2->0 limit
    # must equal zeta/(1-zeta).
    g, M2 = 1.4, 1e-4
    for zeta in (0.02, 0.05, 0.10):
        phi2 = 1.0 - zeta
        num = (1.0 - (g - 1) / 2 * M2 ** 2 * (1.0 / phi2 - 1.0)) ** (
            -g / (g - 1)) - 1.0
        den = 1.0 - (1.0 + (g - 1) / 2 * M2 ** 2) ** (-g / (g - 1))
        assert num / den == pytest.approx(zeta / (1.0 - zeta), rel=1e-3)


def test_profile_weight_is_ko82_signed_not_am_symmetric():
    # KO82 finding 1 (resolved 2026-07): the nozzle->impulse interpolation
    # weight is the SIGNED |r|*r (r = b1/b2), not AM-1957's symmetric r^2.
    # Consequences, pinned physics-anchored (not the surrogate curve values):
    #   (a) identical for r >= 0  -> behavior-preserving in-domain (V6: r>0);
    #   (b) for r < 0 the signed form gives STRICTLY LESS profile loss than
    #       the symmetric r^2 form would (the negative-incidence bucket);
    #   (c) loss stays strictly positive even at the deep-negative clip
    #       extreme (the AD-10 positivity floor), where naive |r|*r on the
    #       surrogate curves would go negative.
    # Isolate the WEIGHT: t/c = 0.20 makes the thickness factor
    # (t/c/0.2)^r == 1 for all r, so only the nozzle->impulse interpolation
    # shows through.
    s_c, tc, a2 = 0.80, 0.20, 55.0

    # (a) r >= 0: monotone nozzle -> impulse (this branch is |r|r == r^2, i.e.
    # byte-identical to the old AM symmetric form, so every in-domain case is
    # unaffected -- V6 runs r in [0.04, 0.72]).
    y_noz = float(profile_loss_am(s_c, 0.0, a2, tc)[0])     # r = 0  (nozzle)
    y_imp = float(profile_loss_am(s_c, a2, a2, tc)[0])      # r = 1  (impulse)
    assert y_imp > y_noz > 0.0                              # impulse loss higher

    # (b) r < 0 pulls loss DOWN (the signed weight is negative), whereas the
    # symmetric r^2 form is even and would give the SAME as the mirrored +|r|.
    # So y(-r) < y(+r): the negative-incidence bucket, the KO82 correction.
    y_neg = float(profile_loss_am(s_c, -20.0, a2, tc)[0])   # r = -0.36
    y_pos = float(profile_loss_am(s_c, +20.0, a2, tc)[0])   # r = +0.36
    assert y_neg < y_pos            # signed weight: negative r pulls loss DOWN
    assert y_neg > 0.0              # ...but stays physical

    # (c) deep-negative extrapolation stays strictly positive (floor active),
    # where a naive |r|r on the surrogate curves would go negative.
    y_extreme = float(profile_loss_am(s_c, -60.0, a2, tc)[0])   # r -> clip
    assert y_extreme > 0.0


def test_mach_K1_is_the_exact_ko82_fig8_equation():
    # KO82 Fig. 8 prints the equation ON the chart: K1 = 1 - 1.25(M2 - 0.2)
    # for M2 > 0.2 (confirmed 2026-07 from the paper, kacker_mean_1980.pdf).
    # The coded ramp IS this equation, not a surrogate. With M1=M2 -> K2=1 ->
    # K_p = K1, so mach_profile_correction(M2,M2) == 1 - 1.25(M2-0.2).
    # (abs 5e-3 absorbs the C1 smoothing near the M2->1 knee where K1->0.)
    for m2 in (0.4, 0.6, 0.8):
        assert float(mach_profile_correction(m2, m2)) == pytest.approx(
            1.0 - 1.25 * (m2 - 0.2), abs=5e-3)


# Digitized KO82 Fig. 14 (dphi^2_TET vs t_TE/o): (t/o, nozzle, impulse).
# The axial-entry NOZZLE is the UPPER curve; IMPULSE the lower (the code had
# them swapped + ~3x too high before 2026-07). tools/digitize_ko82_fig14.py.
TE_FIG14 = [(0.10, 0.0143, 0.0073), (0.20, 0.0506, 0.0283),
            (0.30, 0.100, 0.051), (0.40, 0.140, 0.0739)]


@pytest.mark.parametrize("t_o,noz,imp", TE_FIG14)
def test_trailing_edge_curves_match_ko82_fig14(t_o, noz, imp):
    # Nozzle via alpha1=0 (r=0 -> phi2_ax); impulse via alpha1=alpha2 (r=1 ->
    # phi2_imp). Reproduces the digitized Fig. 14 within reading precision, and
    # nozzle > impulse (the corrected ordering).
    got_noz = float(trailing_edge_zeta(0.0, 60.0, t_o)[0])
    got_imp = float(trailing_edge_zeta(60.0, 60.0, t_o)[0])
    assert got_noz == pytest.approx(noz, abs=0.012)
    assert got_imp == pytest.approx(imp, abs=0.012)
    assert got_noz > got_imp                      # Fig. 14 ordering


def test_secondary_loss_constants_match_ko82():
    # KO82/DC: Y_s = 1.2 * 0.0334 * f_AR * (cos a2/cos a1) * (C_L/(s/c))^2 *
    # cos^2 a2 / cos^3 a_m, with C_L/(s/c) = 2|d(tan a)|*cos a_m (signed
    # frame) and tan a_m = 1/2 (tan a1 + tan a2). Reimplement independently
    # and require the code to reproduce it (pins 1.2 and 0.0334 and the
    # loading structure). Angles well inside the a1 soft-clip.
    a1_deg, a2_deg, ar = 20.0, 55.0, 3.0
    a1, a2 = a1_deg * DEG, a2_deg * DEG
    t1, t2 = np.tan(a1), np.tan(a2)
    tan_m = 0.5 * (t1 + t2)
    cos_m = 1.0 / np.sqrt(1.0 + tan_m * tan_m)
    load = 2.0 * abs(t2 - t1) * cos_m
    z = load * load * np.cos(a2) ** 2 / cos_m ** 3
    ys_ref = 1.2 * 0.0334 * float(_aspect_ratio_factor(ar)) \
        * (np.cos(a2) / np.cos(a1)) * z
    ys_code = float(secondary_loss(a1_deg, a2_deg, ar)[0])
    assert ys_code == pytest.approx(ys_ref, rel=1e-4)
