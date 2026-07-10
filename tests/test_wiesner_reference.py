"""Reference-verified constants for the Wiesner (1967) slip factor.

Pins the exact form/exponent confirmed term-by-term against the source in
``docs/references/WIE67.md`` (extracted 2026-07-09 from the NotebookLM theory
library; cross-agreeing across Aungier/Braembussche/Cumpsty/Dixon/
Lakshminarayana/Whitfield-Baines). A drift in the numerator, the Z exponent,
or the radial angle reference turns this red.

Also pins the radius-ratio limit correction (RESOLVED 2026-07, WIE67.md
finding 1): the limit exponent 8.16, the Braembussche cubic (adopted over
Aungier's beta2/10 outlier on 3-source consensus), and the below-limit
no-op / above-limit reduction behavior.
"""
import numpy as np
import pytest

from slcflow.closures.centrifugal.wiesner import wiesner_slip

DEG = np.pi / 180.0


def _ref_sigma(b2b_deg, Z):
    # Wiesner: sigma = 1 - sqrt(cos(beta2b)) / Z**0.7, beta2b from radial.
    return 1.0 - np.sqrt(np.cos(b2b_deg * DEG)) / Z ** 0.7


@pytest.mark.parametrize("b2b_deg,Z", [(0.0, 15), (30.0, 15),
                                       (45.0, 20), (30.0, 8), (20.0, 12)])
def test_wiesner_form_matches_source(b2b_deg, Z):
    got = float(wiesner_slip(b2b_deg * DEG, Z)[0])
    assert got == pytest.approx(_ref_sigma(b2b_deg, Z), rel=1e-4)


def test_wiesner_radial_blades_give_max_slip():
    # beta2b = 0 (radial blades): cos(0)=1 -> sigma = 1 - 1/Z**0.7, the
    # deepest slip (confirmed reference direction: from radial, not tangent).
    Z = 15
    assert float(wiesner_slip(0.0, Z)[0]) == pytest.approx(
        1.0 - 1.0 / Z ** 0.7, rel=1e-4)


def test_wiesner_z_exponent_is_0p7():
    # Ratio of (1 - sigma) between two blade counts isolates the exponent:
    # (1-sigma1)/(1-sigma2) = (Z2/Z1)**0.7 at fixed backsweep.
    b = 30.0 * DEG
    s1 = 1.0 - float(wiesner_slip(b, 10)[0])
    s2 = 1.0 - float(wiesner_slip(b, 20)[0])
    assert s1 / s2 == pytest.approx((20.0 / 10.0) ** 0.7, rel=1e-3)


# --- radius-ratio limit correction (WIE67.md finding 1, resolved 2026-07) ---

def _eps_lim(b2b_deg, Z):
    # Limit exponent 8.16, from-radial (cos) form (Cumpsty/Dixon 7.35c).
    return np.exp(-8.16 * np.cos(b2b_deg * DEG) / Z)


def test_limit_correction_inactive_below_eps_lim():
    # r1/r2 comfortably below the limit -> the base form is untouched
    # (softplus positive part is ~0). This is why in-limit designs (e.g. V7
    # hub/mean streamlines) see no change.
    b2b_deg, Z = 30.0, 18
    base = float(wiesner_slip(b2b_deg * DEG, Z)[0])
    rr = _eps_lim(b2b_deg, Z) - 0.15
    corrected = float(wiesner_slip(b2b_deg * DEG, Z, rr)[0])
    assert corrected == pytest.approx(base, rel=1e-4)


def test_limit_correction_is_braembussche_cubic():
    # Above the limit sigma is reduced by the cubic factor 1 - ((r1/r2 -
    # eps_lim)/(1 - eps_lim))^3 (Braembussche 3.84, exponent 3 confirmed;
    # the 8.16 limit exponent likewise). Well above the limit the softplus
    # smoothing is negligible, so the closed form must match tightly.
    b2b_deg, Z = 30.0, 18
    base = float(wiesner_slip(b2b_deg * DEG, Z)[0])
    eps = _eps_lim(b2b_deg, Z)
    for rr in (0.80, 0.90):
        cube = ((rr - eps) / (1.0 - eps)) ** 3
        expected = base * (1.0 - cube)
        got = float(wiesner_slip(b2b_deg * DEG, Z, rr)[0])
        assert got == pytest.approx(expected, rel=2e-3)


def test_limit_correction_reduces_slip_factor_and_stays_nonnegative():
    # The correction only ever reduces sigma (more slip at high radius ratio),
    # and sigma stays >= 0 through the degenerate r1 -> r2 (AD-10).
    b2b_deg, Z = 20.0, 12
    base = float(wiesner_slip(b2b_deg * DEG, Z)[0])
    prev = base
    for rr in (0.70, 0.80, 0.90, 0.999):
        s = float(wiesner_slip(b2b_deg * DEG, Z, rr)[0])
        assert s <= prev + 1e-9      # monotone non-increasing in r1/r2
        assert s >= 0.0              # never negative
        prev = s


def test_cubic_beats_aungier_beta2_10_which_is_not_adopted():
    # Guard the fork resolution: the implemented exponent is the cube (3),
    # NOT Aungier 4-10's beta2/10. At beta2b=30 deg, beta2/10 = 3.0 too, so
    # pick an angle where they diverge (beta2b=60 -> beta2/10 = 6.0).
    b2b_deg, Z = 60.0, 15
    base = float(wiesner_slip(b2b_deg * DEG, Z)[0])
    eps = _eps_lim(b2b_deg, Z)
    rr = 0.90
    frac = (rr - eps) / (1.0 - eps)
    cubic = base * (1.0 - frac ** 3)
    aungier = base * (1.0 - frac ** (b2b_deg / 10.0))
    got = float(wiesner_slip(b2b_deg * DEG, Z, rr)[0])
    assert got == pytest.approx(cubic, rel=2e-3)
    assert abs(got - aungier) > 1e-3     # demonstrably not the beta2/10 form
