"""Multistage-compressor mixing revisit (Theory Manual sections 9.5, 3.6;
Appendix C.5m; M8 sub-step 3).

Section 3.6's stated motivation is that multistage machines "develop
unrealistic spanwise stratification of h0, s and rVt without a mixing model."
This case makes that concrete on a two-stage axial compressor: with the
default Gallimore mixing it converges and compresses; with mixing OFF the
spanwise entropy stratification runs away and the solve fails outright. So
mixing here is not cosmetic -- it is a convergence prerequisite for the
multistage configuration.

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
        config=ClassicalConfig(max_outer=800))


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
# Section 3.6: WITHOUT mixing the stratification runs away
# --------------------------------------------------------------------------
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # runaway is the point
def test_without_mixing_stratification_runs_away(without_mixing, with_mixing):
    base, mix = without_mixing, with_mixing
    # Even at 800 outer iterations the un-mixed two-stage does not converge:
    # the hub/tip entropy split diverges (the section 3.6 failure mode).
    assert not base.converged
    s_base = np.ptp(base.result.frozen.transported.s[:, -1])
    s_mix = np.ptp(mix.result.frozen.transported.s[:, -1])
    assert s_base > 10.0 * s_mix                     # dramatically worse
