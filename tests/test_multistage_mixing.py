"""Multistage-compressor mixing revisit (Theory Manual sections 9.5, 3.6;
Appendix C.5m; M8 sub-step 3, REVISED repeatedly through 2026-07).

Section 3.6's stated motivation is that multistage machines "develop
unrealistic spanwise stratification of h0, s and rVt without a mixing model."
This case exercises the operator on a two-stage axial compressor: mixing ON
vs OFF, at the Gallimore-Cumpsty-calibrated coefficient, now with the case
geometry retuned so the Lieblein loss runs INSIDE its validity window
(closure validity 1.0, not the earlier saturated 0 -- so the stratification is
a meaningful measurement, not saturated-loss garbage).

**What this measures (the honest, in-window result).** At the G-C-calibrated
``c_mix = 5e-4`` spanwise mixing is a MODEST damping of the exit entropy
spread -- measured ~24% on two stages (s_base 0.495 -> s_mix 0.377 J/(kg.K))
with the Howell endwall + tip-clearance loss now in the set (it was ~18%
before, s_base 0.267; the endwall loss adds its own spanwise-varying entropy).
It does NOT catch up with the stratification production, i.e. it is not a
homogenizer. That is what this file pins (by ratio, not absolute value).

The claim has been wrong three times before, each a traced artifact (kept as a
warning): (1) M8-3 called mixing a *convergence prerequisite* -- a driver
stale-split bug (fixed at the 2026-07 Tier-3 stabilization; un-mixed converges
fine). (2) The revision called it a *dramatic homogenizer* (~25x) -- the
compound artifact of the Lieblein omega_bar inversion (~4x too much loss) and
a ~20x-strong c_mix. (3) After both were fixed the case still ran at closure
validity 0 (over-loaded, untwisted blade over a wide annulus -> D_eq out of
window), so the loss was saturated. Retuning the annulus (hub/tip 0.64 -> 0.73)
put it in-window; the surviving claim is the modest damping above.

Structural gate (bands, not V5 validation tolerances -- [VERIFY], as the
single-stage V5). Provenance: M8 sub-step 3, written with the case.
"""
import numpy as np
import pytest

from slcflow.transport import GallimoreMixing
from slcflow.types import FidelityConfig, MassFlowSpec
from slcflow.verification.v5_axial_compressor import V5MultistageCompressor


@pytest.fixture(scope="module")
def case():
    return V5MultistageCompressor(n_stages=2)


@pytest.fixture(scope="module")
def with_mixing(case):
    # Default evaluate(): Tier 3, mixing_term=1, the shipped Gallimore default.
    return case.evaluate(n_sl=9)


@pytest.fixture(scope="module")
def without_mixing(case):
    from slcflow.drivers.classical import ClassicalConfig
    return case.machine().evaluate(
        MassFlowSpec(case.mdot), FidelityConfig.tier3(), n_sl=9,
        config=ClassicalConfig(max_outer=300))


# --------------------------------------------------------------------------
# Section 3.6: mixing lets the multistage converge and compress, in-window
# --------------------------------------------------------------------------
def test_multistage_with_mixing_converges_and_compresses(with_mixing, case):
    r = with_mixing
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
    assert r.pressure_ratio > 1.0                    # net compression
    elo, ehi = case.eta_band
    assert elo < r.efficiency < ehi
    # The retune's whole point: the loss runs IN the Lieblein validity window.
    val = float(np.asarray(r.result.frozen.closures.validity).min())
    assert val > 0.5


def test_multistage_with_mixing_deswirls(with_mixing):
    r = with_mixing
    tr = r.result.frozen.transported
    # Repeating stage: the last stator returns the flow near axial.
    vtheta_ex = tr.rvt[:, -1] / r.r
    alpha_ex = np.degrees(np.arctan2(vtheta_ex, r.vm))
    assert np.all(np.abs(alpha_ex) < 20.0)
    # The exit entropy spread stays bounded (loose sanity, not a homogenization
    # claim -- see the module docstring and the comparison test below).
    assert np.ptp(tr.s[:, -1]) < 5.0


# --------------------------------------------------------------------------
# Section 3.6: at the G-C-calibrated coefficient mixing is a MODEST damping
# --------------------------------------------------------------------------
def test_mixing_modestly_reduces_stratification_at_gc_calibration(
        without_mixing, with_mixing):
    # At the honest c_mix (5e-4) AND in-window loss (the 2026-07 retune), the
    # mixed exit spread is only slightly below the un-mixed one -- measured
    # ~24% on two stages (s_base 0.495 vs s_mix 0.377 J/(kg.K), with the Howell
    # endwall loss now in the set; ~18% before). Both converge
    # in-window; the direction (mixing REDUCES the spread) is a guaranteed
    # property of the diffusion operator; the SMALLNESS is the finding this
    # pins (refuting the old "dramatic homogenizer" claim).
    base, mix = without_mixing, with_mixing
    assert base.converged
    assert float(np.asarray(base.result.frozen.closures.validity).min()) > 0.5
    s_base = float(np.ptp(base.result.frozen.transported.s[:, -1]))
    s_mix = float(np.ptp(mix.result.frozen.transported.s[:, -1]))
    assert s_base > s_mix                            # mixing reduces the spread
    assert s_base > 1.05 * s_mix                     # by a real (>5%) amount
    assert s_base < 1.5 * s_mix                      # but only modestly (<50%)
    assert s_base > 0.15                             # spread is non-trivial
