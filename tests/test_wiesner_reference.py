"""Reference-verified constants for the Wiesner (1967) slip factor.

Pins the exact form/exponent confirmed term-by-term against the source in
``docs/references/WIE67.md`` (extracted 2026-07-09 from the NotebookLM theory
library; cross-agreeing across Aungier/Braembussche/Cumpsty/Dixon/
Lakshminarayana/Whitfield-Baines). A drift in the numerator, the Z exponent,
or the radial angle reference turns this red.

Not pinned: the radius-ratio limit correction (omitted in code by design —
the closure doesn't read the inducer radius; see WIE67.md finding 1).
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
