"""Reference-verified constants for the Kacker-Okapuu loss chain.

Unlike ``test_kacker_okapuu.py`` (structural bands/trends/C1), these pin the
*exact scalar constants* of the K-O / Dunham-Came / Ainley-Mathieson method
that were confirmed term-by-term against the source in
``docs/references/KO82.md`` (extracted 2026-07-09 from the cleaned NotebookLM
loss-model library, citation-backed). If a future refactor drifts one of
these numbers, that is a regression against the paper, and this file goes red.

Scope: only the confirmed *formula* constants are pinned. The nozzle/impulse
``yp1``/``yp2`` and TE ``phi2`` reference curves are surrogate fits to the
AM/K-O charts and are NOT pinned here (they need figure-point digitization —
the residual [VERIFY], see KO82.md).
"""
import numpy as np
import pytest

from slcflow.closures.axial_turbine.kacker_okapuu import (
    _aspect_ratio_factor, mach_profile_correction, profile_loss_am,
    reynolds_correction, secondary_loss, shock_loss)

DEG = np.pi / 180.0


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
