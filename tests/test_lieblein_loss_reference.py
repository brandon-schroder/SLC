"""Reference-verified Lieblein (1959) profile-loss constants.

Pins the equivalent-diffusion-ratio and wake-momentum-thickness constants
confirmed term-by-term against Aungier ch.6 / Cumpsty / Dixon in
``docs/references/LIEB59.md`` (extracted 2026-07-09, citation-backed).

Scope: D_eq (1.12/0.61), theta/c (0.004/1.17), and the omega_bar assembly
(2 (theta/c)(sigma/cos b2)(W2/W1)^2, Aungier 6-27 / Cumpsty 1.32 -- the
velocity-ratio inversion bug found in this pass is now FIXED, 2026-07).
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.loss import (
    equivalent_diffusion, off_design_bucket, profile_loss_coefficient,
    stall_choke_ranges, wake_momentum_thickness)

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
