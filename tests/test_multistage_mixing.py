"""Multistage-compressor mixing revisit (Theory Manual sections 9.5, 3.6;
Appendix C.5m; M8 sub-step 3, REVISED at the 2026-07 Tier-3 stabilization).

Section 3.6's stated motivation is that multistage machines "develop
unrealistic spanwise stratification of h0, s and rVt without a mixing model."
This case makes that concrete on a two-stage axial compressor: with the
default Gallimore mixing the exit entropy profile is nearly uniform; with
mixing OFF the machine converges to a HEAVILY STRATIFIED state (~25x the
spanwise entropy spread) -- section 3.6's "unrealistic stratification" made
measurable.

Historical note: M8-3 originally recorded mixing as a *convergence
prerequisite* (mixing-off NUMERICAL_FAILURE at 800 iterations). The 2026-07
diagnosis showed that non-convergence was the driver's stale-split /
spurious-branch artifact, not physics; post-stabilization the un-mixed case
converges fine. The surviving (physical) claim is the stratification, and
that is what this file pins.

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


def test_multistage_with_mixing_deswirls_and_is_unstratified(with_mixing):
    r = with_mixing
    tr = r.result.frozen.transported
    # Repeating stage: the last stator returns the flow near axial.
    vtheta_ex = tr.rvt[:, -1] / r.r
    alpha_ex = np.degrees(np.arctan2(vtheta_ex, r.vm))
    assert np.all(np.abs(alpha_ex) < 20.0)
    # Mixing keeps the exit entropy profile nearly uniform across the span.
    assert np.ptp(tr.s[:, -1]) < 5.0


# --------------------------------------------------------------------------
# Section 3.6: WITHOUT mixing the machine converges but heavily stratified
# --------------------------------------------------------------------------
def test_without_mixing_converges_heavily_stratified(without_mixing,
                                                     with_mixing):
    # REVISED 2026-07 (Tier-3 stabilization): the original M8-3 claim that
    # mixing is a *convergence prerequisite* was the driver's stale-split
    # artifact -- un-mixed, the two-stage now converges. The PHYSICAL
    # section 3.6 claim stands and is what this pins: without mixing the
    # exit entropy profile is dramatically stratified (measured 17.6 vs
    # 0.69 J/(kg K), ~25x; asserted at >10x with a >5 J/(kg K) floor).
    base, mix = without_mixing, with_mixing
    assert base.converged
    s_base = np.ptp(base.result.frozen.transported.s[:, -1])
    s_mix = np.ptp(mix.result.frozen.transported.s[:, -1])
    assert s_base > 10.0 * s_mix                     # dramatically worse
    assert s_base > 5.0                              # and absolutely large
