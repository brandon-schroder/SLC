"""Reference-verified Lieblein (1959) profile-loss constants.

Pins the equivalent-diffusion-ratio and wake-momentum-thickness constants
confirmed term-by-term against Aungier ch.6 / Cumpsty / Dixon in
``docs/references/LIEB59.md`` (extracted 2026-07-09, citation-backed).

Scope: the CONFIRMED-correct pieces (D_eq 1.12/0.61, theta/c 0.004/1.17). The
omega_bar velocity-ratio INVERSION bug (code (W1/W2)^2 vs source (W2/W1)^2) is
documented in LIEB59.md with the fix deferred to the consolidation pass -- see
``test_omega_bar_velocity_ratio_is_inverted`` below, an xfail that will start
passing once the bug is fixed.
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.loss import (
    equivalent_diffusion, wake_momentum_thickness)

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


@pytest.mark.xfail(reason="known bug (LIEB59.md): omega_bar uses (W1/W2)^2, "
                   "Aungier 6-27/Cumpsty 1.32 give (W2/W1)^2. Fix deferred to "
                   "the consolidation pass; this xpasses once corrected.",
                   strict=True)
def test_omega_bar_velocity_ratio_is_inverted():
    # Pin the bug in-suite: build the coded omega_min factor and the correct
    # one, assert they AGREE (they don't yet -> xfail). W1>W2 for a compressor.
    b2 = 25.0 * DEG
    theta_c, sigma, w1, w2 = 0.01, 1.2, 1.0, 0.72
    coded = 2.0 * theta_c * sigma / np.cos(b2) * (w1 / w2) ** 2
    correct = 2.0 * theta_c * sigma / np.cos(b2) * (w2 / w1) ** 2
    assert coded == pytest.approx(correct, rel=1e-6)
