"""Reference-verified centrifugal internal-loss forms.

Pins the two base loss forms confirmed against Galvas NASA TN D-7487 / Aungier
/ Braembussche in ``docs/references/CENT-LOSS.md`` (extracted 2026-07-09,
citation-backed). Both are form + leading-coefficient checks (no clean single
constant beyond 0.5 and the 2*Cf leading factor).

The [DECIDE] modeling variants (incidence f_inc<1; Aungier's mean-of-squares
W_avg) are documented in CENT-LOSS.md, not pinned here.
"""
import numpy as np
import pytest

from slcflow.closures.centrifugal.loss import (
    incidence_loss, skin_friction_loss)


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
