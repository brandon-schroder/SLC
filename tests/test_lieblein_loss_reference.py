"""Reference-verified Lieblein (1959) profile-loss constants.

Pins the equivalent-diffusion-ratio and wake-momentum-thickness constants
confirmed term-by-term against Aungier ch.6 / Cumpsty / Dixon in
``docs/references/LIEB59.md`` (extracted 2026-07-09, citation-backed).

Scope: D_eq (1.12/0.61), theta/c (0.004/1.17), and the omega_bar assembly
(2 (theta/c)(sigma/cos b2)(W2/W1)^2, Aungier 6-27 / Cumpsty 1.32 -- the
velocity-ratio inversion bug found in this pass is now FIXED, 2026-07).

Also pins the chart-OUTPUT validation of ``wake_momentum_thickness`` against
Lieblein's own Fig. 6 (``tools/digitize_lieblein_loss.py``): the coded curve
was digitized-verified to ride the published dashed EQUATION-[8] line and the
cascade data cloud (max |coded - chart| = 0.0003; clean validation, no bug).
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.loss import (
    blade_loading_coefficient, endwall_clearance_loss, equivalent_diffusion,
    normal_shock_pt_ratio, off_design_bucket, profile_loss_coefficient,
    shock_loss, stall_choke_ranges, wake_momentum_thickness)

DEG = np.pi / 180.0


@pytest.mark.parametrize("b1d,b2d,s,w1,w2", [(45., 20., 1.2, 1.0, 0.75),
                                             (55., 30., 1.0, 1.0, 0.70),
                                             (35., 15., 1.5, 1.0, 0.82)])
def test_equivalent_diffusion_1p12_0p61(b1d, b2d, s, w1, w2):
    # Aungier 6-36 / Dixon 3.40: D_eq = (W1/W2)[1.12 + 0.61(cos^2 b1/s)(tan
    # b1 - tan b2)]. (The W1/W2 factor here IS correct -- the inversion bug
    # is only in the omega_bar assembly, not in D_eq.)
    b1, b2 = b1d * DEG, b2d * DEG
    ref = (w1 / w2) * (1.12 + 0.61 * np.cos(b1) ** 2 / s
                       * (np.tan(b1) - np.tan(b2)))
    assert float(equivalent_diffusion(w1, w2, b1, b2, s)) == pytest.approx(
        ref, rel=1e-6)


@pytest.mark.parametrize("d_eq", [1.3, 1.5, 1.8, 2.0])
def test_wake_momentum_thickness_0p004_1p17(d_eq):
    # Dixon 3.37: theta*/c = 0.004 / (1 - 1.17 ln D_eq). Deep in-domain, the
    # input soft-saturation is ~identity; rtol absorbs the residual.
    ref = 0.004 / (1.0 - 1.17 * np.log(d_eq))
    assert float(wake_momentum_thickness(d_eq)[0]) == pytest.approx(
        ref, rel=3e-3)


# Digitized Lieblein (1959) Fig. 6 dashed-curve readings (DR, (theta/c)_2);
# see tools/digitize_lieblein_loss.py for provenance + the overlay check.
_FIG6 = [(1.10, 0.0046), (1.15, 0.0049), (1.20, 0.0049), (1.40, 0.0063),
         (1.60, 0.0089), (1.80, 0.0128), (2.05, 0.0247), (2.10, 0.0298)]


@pytest.mark.parametrize("d_eq,chart", _FIG6)
def test_wake_momentum_thickness_matches_lieblein_fig6(d_eq, chart):
    # Lieblein 1959 Fig. 6: the theta/c = 0.004/(1 - 1.17 ln D_eq) curve
    # digitized off the primary paper (not just the textbook algebra). The
    # coded output must reproduce the published dashed EQUATION-[8] curve to
    # chart reading precision (~0.0006 in theta/c).
    got = float(wake_momentum_thickness(d_eq)[0])
    assert got == pytest.approx(chart, abs=1.5e-3)


def test_wake_momentum_thickness_diverges_at_lieblein_2p35_limit():
    # Lieblein 1959 (p.5): the k_s = 1.17 fit diverges at the "limit
    # V_max,s/V2 = 2.35" -- exactly the denominator zero e^(1/1.17). The code
    # saturates D_eq below this (ceiling 2.2) so the output stays finite and
    # rises steeply toward it, never crossing.
    assert np.exp(1.0 / 1.17) == pytest.approx(2.35, abs=5e-3)
    below = float(wake_momentum_thickness(2.30)[0])   # near the limit
    mid = float(wake_momentum_thickness(1.80)[0])
    assert below > mid > 0.0                          # monotone rise, finite
    assert np.isfinite(below)


@pytest.mark.parametrize("b2d,theta_c,sigma,w1,w2", [(25.0, 0.01, 1.2, 1.0, 0.72),
                                                    (35.0, 0.015, 1.0, 1.0, 0.65)])
def test_omega_bar_uses_W2_over_W1_squared(b2d, theta_c, sigma, w1, w2):
    # Aungier 6-27 / Cumpsty 1.32: omega_bar = 2(theta/c)(sigma/cos b2)(W2/W1)^2
    # (the inversion bug is fixed). W1 > W2 for a compressor -> factor < 1.
    b2 = b2d * DEG
    expect = 2.0 * theta_c * sigma / np.cos(b2) * (w2 / w1) ** 2
    got = float(profile_loss_coefficient(theta_c, sigma, b2, w1, w2))
    assert got == pytest.approx(expect, rel=1e-9)
    # Guard against a silent regression to the inverted form.
    inverted = 2.0 * theta_c * sigma / np.cos(b2) * (w1 / w2) ** 2
    assert got < inverted           # (W2/W1)^2 < (W1/W2)^2 since W2 < W1


@pytest.mark.parametrize("b1d,b2d,s", [(50.0, 40.0, 1.2), (55.0, 30.0, 1.0),
                                       (45.0, 20.0, 1.5)])
def test_blade_loading_coefficient_matches_dixon(b1d, b2d, s):
    # Dixon 3.15 / 3.26a (Saravanamuttoo 5.32/5.33): tan(b_m) = (tan b1 +
    # tan b2)/2 and C_L = (2/sigma) cos(b_m)(tan b1 - tan b2) (the -C_D tan b_m
    # term dropped). Moderate angles -> the AD-10 soft-clip is ~identity.
    b1, b2 = b1d * DEG, b2d * DEG
    bm_ref = np.arctan(0.5 * (np.tan(b1) + np.tan(b2)))
    cl_ref = 2.0 / s * np.cos(bm_ref) * (np.tan(b1) - np.tan(b2))
    cl, bm = blade_loading_coefficient(b1, b2, s)
    assert float(cl) == pytest.approx(cl_ref, rel=1e-3)
    assert float(bm) == pytest.approx(bm_ref, rel=1e-3)


@pytest.mark.parametrize("b1d,b2d,s,ar,th", [(50.0, 40.0, 1.2, 2.5, 0.0),
                                             (55.0, 35.0, 1.0, 2.0, 0.02),
                                             (48.0, 30.0, 1.3, 3.0, 0.03)])
def test_endwall_clearance_loss_matches_howell(b1d, b2d, s, ar, th):
    # Howell p.451 (Saravanamuttoo 5.35/5.36) + Lakshminarayana (via Cumpsty):
    #   C_Ds = 0.018 C_L^2 ; C_Da = 0.020/(sigma*AR) ; C_Dk = 0.7 C_L^2 (t/h)
    # converted to omega_bar via the inverse of Cumpsty 4.9
    #   omega = sigma (cos^2 b1 / cos^3 b_m)(C_Ds + C_Da + C_Dk).
    b1, b2 = b1d * DEG, b2d * DEG
    bm = np.arctan(0.5 * (np.tan(b1) + np.tan(b2)))
    cl = 2.0 / s * np.cos(bm) * (np.tan(b1) - np.tan(b2))
    cds, cda, cdk = 0.018 * cl * cl, 0.020 / (s * ar), 0.7 * cl * cl * th
    ref = s * np.cos(b1) ** 2 / np.cos(bm) ** 3 * (cds + cda + cdk)
    om, _ = endwall_clearance_loss(b1, b2, s, ar, th)
    assert float(om) == pytest.approx(ref, rel=2e-3)


def test_endwall_clearance_term_is_inert_without_clearance():
    # The tip-clearance drag C_Dk = 0.7 C_L^2 (t/h) vanishes at zero clearance,
    # so a zero-clearance row sees only secondary + annulus endwall loss (the
    # existing V5 cases, which set no clearance, are unaffected by the term).
    a = float(endwall_clearance_loss(50 * DEG, 40 * DEG, 1.2, 2.5, 0.0)[0])
    b = float(endwall_clearance_loss(50 * DEG, 40 * DEG, 1.2, 2.5, 0.02)[0])
    assert b > a > 0.0                       # clearance adds loss, monotone


def test_endwall_validity_drops_at_high_loading():
    # Howell's drag data is moderate-loading; the compact-support validity
    # ceiling on C_L saturates (-> 0) at very high loading.
    _, v_lo = endwall_clearance_loss(50 * DEG, 40 * DEG, 1.2, 2.5, 0.0)
    _, v_hi = endwall_clearance_loss(65 * DEG, 30 * DEG, 0.8, 2.5, 0.0)
    assert float(v_lo) == pytest.approx(1.0, abs=1e-3)
    assert float(v_hi) < 0.1


# Standard normal-shock stagnation-pressure ratios (gas tables, gamma=1.4).
@pytest.mark.parametrize("mach,pt_ratio", [(1.0, 1.0000), (1.5, 0.9298),
                                           (2.0, 0.7209), (2.5, 0.4990)])
def test_normal_shock_pt_ratio_matches_gas_tables(mach, pt_ratio):
    # The Rayleigh supersonic-pitot / normal-shock relation (Aungier 6.7 uses a
    # real-gas solve of 6-72..74; this is the perfect-gas closed form). Pinned
    # against standard compressible-flow tables.
    assert float(normal_shock_pt_ratio(mach, 1.4)) == pytest.approx(
        pt_ratio, abs=5e-4)


def test_shock_loss_is_inert_subsonic():
    # M_shock = M1 sqrt(ratio); well below the M_shock=1 onset the loss is ~0,
    # so subsonic compressor rows (all current V5 cases) are unaffected.
    om, v = shock_loss(0.6, 1.3, 1.4)       # M_shock = 0.6*sqrt(1.3) = 0.684
    assert float(om) < 1e-4
    assert float(v) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("m1,ratio", [(1.3, 1.25), (1.4, 1.20), (1.25, 1.35)])
def test_shock_loss_matches_aungier_formula(m1, ratio):
    # Aungier 6.7: M_shock = sqrt(M1 * M_ss), M_ss = M1*ratio (Eq 6-71); the
    # normal-shock Pt loss referenced to inlet dynamic head (Eq 2-68). Recompute
    # the documented closed form independently and require the code to match.
    # Chosen deep-supersonic (M_shock > 1.45) where the C1 softplus onset floor
    # is identity; the near-onset supercritical behaviour is a separate test.
    g = 1.4
    m_shock = np.sqrt(m1 * (m1 * ratio))
    pr = float(normal_shock_pt_ratio(m_shock, g))
    p_pt1 = (1.0 + 0.5 * (g - 1.0) * m1 * m1) ** (-g / (g - 1.0))
    ref = (1.0 - pr) / (1.0 - p_pt1)
    om, _ = shock_loss(m1, ratio, g)
    assert float(om) == pytest.approx(ref, rel=1e-2)


def test_shock_loss_supercritical_onset_and_growth():
    # The geometric-mean Mach can exceed 1 while the inlet M1 is still subsonic
    # (Aungier's supercritical regime), and the loss grows monotonically with M1.
    oms = [float(shock_loss(m, 1.4, 1.4)[0])
           for m in [0.85, 0.95, 1.05, 1.15]]
    assert oms[0] < oms[1] < oms[2] < oms[3]          # monotone growth
    assert oms[1] > 1e-3                              # supercritical: M1<1 loss>0


@pytest.mark.parametrize("theta,b1", [(12.0, 52.0), (25.0, 45.0), (8.0, 60.0)])
def test_stall_choke_ranges_match_aungier(theta, b1):
    # Aungier ch.6 low-speed stall/choke incidence ranges (LIEB59.md):
    #   R_s = 10.3 + (2.92 - b1/15.6) theta/8.2
    #   R_c = 9.0  - (1 - (30/b1)^0.48) theta/4.176
    # Chosen well inside so the positivity floors do not bind.
    rs_ref = 10.3 + (2.92 - b1 / 15.6) * theta / 8.2
    rc_ref = 9.0 - (1.0 - (30.0 / b1) ** 0.48) * theta / 4.176
    rs, rc = stall_choke_ranges(np.array([theta]), np.array([b1]))
    assert float(rs[0]) == pytest.approx(rs_ref, rel=1e-6)
    assert float(rc[0]) == pytest.approx(rc_ref, rel=1e-6)


def test_off_design_bucket_is_aungier_piecewise():
    # Aungier ch.6 normalized-incidence multiplier (w_s = 0 subsonic):
    #   f = 1 + xi^2       for -2 <= xi <= 1
    #   f = 2 + 2(xi - 1)  for xi > 1        (deep positive stall)
    #   f = 5 - 4(xi + 2)  for xi < -2       (deep negative stall / choke)
    # xi = (i - i_ref)/R_s for i>=i_ref, /R_c below. Pin at points clear of the
    # C1 blend transitions (breakpoints xi = 1, -2).
    r_s, r_c = np.array([10.0]), np.array([8.0])

    def f(di):
        return float(off_design_bucket(np.array([di]), np.array([0.0]),
                                       r_s, r_c)[0])

    assert f(0.0) == pytest.approx(1.0, abs=1e-9)        # min-loss at reference
    assert f(5.0) == pytest.approx(1.0 + 0.5 ** 2, rel=2e-3)   # xi=0.5 core
    assert f(-4.0) == pytest.approx(1.0 + 0.5 ** 2, rel=2e-3)  # xi=-0.5 core
    # xi = 1 boundary: both core and linear branch give 2 (C1-matched).
    assert f(10.0) == pytest.approx(2.0, rel=1e-2)
    # xi = 2 deep positive stall (well past the blend): 2 + 2(2-1) = 4.
    assert f(20.0) == pytest.approx(4.0, rel=2e-2)
    # xi = -3 deep negative stall: 5 - 4(-3+2) = 9.
    assert f(-24.0) == pytest.approx(9.0, rel=2e-2)
    # Asymmetry: positive side uses R_s, negative uses R_c (R_s != R_c).
    assert f(8.0) != pytest.approx(f(-8.0), rel=1e-3)
