"""Multistage-compressor mixing revisit (Theory Manual sections 9.5, 3.6;
Appendix C.5m; M8 sub-step 3, REVISED TWICE in 2026-07).

Section 3.6's stated motivation is that multistage machines "develop
unrealistic spanwise stratification of h0, s and rVt without a mixing model."
This case exercises the operator on a two-stage axial compressor: mixing ON
vs OFF, at the Gallimore-Cumpsty-calibrated coefficient.

**What this actually measures (the honest, twice-corrected result).** At the
G-C-calibrated ``c_mix = 5e-4`` (docs/references/GC86.md, resolution pass
2026-07) spanwise mixing is a MODEST damping of the exit entropy spread --
measured ~11% on two stages (s_base 1.88 -> s_mix 1.68 J/(kg.K)), holding at
~8-11% up to four stages while the absolute spread itself grows
(2/3/4 stages: 1.9/6.8/10.3 J/(kg.K)). It is NOT a homogenizer: the mixing
does not catch up with the stratification production. That is what this file
pins now.

Two historical over-claims, both since traced to artifacts (kept as a warning):
  * M8-3 recorded mixing as a *convergence prerequisite* (mixing-off
    NUMERICAL_FAILURE at 800 iters). The 2026-07 Tier-3 stabilization showed
    that was the driver's stale-split / spurious-branch bug, not physics --
    un-mixed now converges fine.
  * The same revision then recorded a *dramatic* stratification difference
    (~6-25x). That was the over-strong ``c_mix = 0.01`` (~20x the G-C
    calibration); at the corrected coefficient the effect is the modest ~11%
    above. The "mixing flattens multistage stratification" narrative does not
    survive an honestly-calibrated coefficient on this representative case.

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
# Section 3.6: mixing lets the multistage converge and compress
# --------------------------------------------------------------------------
def test_multistage_with_mixing_converges_and_compresses(with_mixing, case):
    r = with_mixing
    assert r.converged
    lo, hi = case.pr_band
    assert lo < r.pressure_ratio < hi
    assert r.pressure_ratio > 1.0                    # net compression
    elo, ehi = case.eta_band
    assert elo < r.efficiency < ehi


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
    # RESOLUTION PASS 2026-07 (c_mix 0.01 -> 5e-4, G-C-calibrated): the earlier
    # "dramatic (>4x) stratification difference" was the over-strong
    # coefficient. At the honest coefficient the mixed exit spread is only
    # slightly below the un-mixed one -- measured ~11% (s_base 1.88 vs s_mix
    # 1.68 J/(kg.K)). Both converge; the direction (mixing REDUCES the spread)
    # is a guaranteed property of the diffusion operator; the SMALLNESS is the
    # finding this pins (refuting the old homogenization claim).
    base, mix = without_mixing, with_mixing
    assert base.converged
    s_base = float(np.ptp(base.result.frozen.transported.s[:, -1]))
    s_mix = float(np.ptp(mix.result.frozen.transported.s[:, -1]))
    assert s_base > s_mix                            # mixing reduces the spread
    assert s_base > 1.05 * s_mix                     # by a real (>5%) amount
    assert s_base < 1.5 * s_mix                      # but only modestly (<50%)
    assert s_base > 1.0                              # spread is non-trivial
