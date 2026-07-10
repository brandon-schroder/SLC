"""SP-36 chart-OUTPUT validation for the Lieblein incidence/deviation set.

Distinct from ``test_lieblein_reference.py`` (which pins the Aungier fit
COEFFICIENTS): this pins the fit OUTPUTS against points digitized off the
original NASA SP-36 (Lieblein) design charts -- the end-to-end check that the
whole chain SP-36 data -> Aungier fit -> our code agrees.

Charts (NASA SP-36 / Johnsen & Bullock 1965, NTRS 19650013744, Chapter VI;
tools/digitize_sp36.py, docs/references/AUN-C.md):
  * Fig. 137 -- (i0)_10  reference zero-camber incidence  (Eq 6-13)
  * Fig. 161 -- (delta0)_10 reference zero-camber deviation (Eq 6-20)
  * Fig. 138 / 162 -- camber slopes n / m (validated by overlay; spot-pinned).

``reference_incidence(b1, sigma, 0.10, camber=0)`` returns exactly K_ti*(i0)_10
= (i0)_10 (K_ti = 1 at t/c = 0.10), and likewise ``reference_deviation`` returns
(delta0)_10 -- so camber=0, t/c=0.10 isolates the chart quantity.

Tolerance = the ~+/-0.15 deg reading precision of the 1965 raster (fine grid)
plus the small closure saturation; measured RMS was ~0.10 deg (i0) / ~0.17 deg
(delta0). A drift in the fit forms beyond that turns this red.
"""
import numpy as np
import pytest

from slcflow.closures.axial_compressor.lieblein import (
    reference_deviation, reference_incidence)

SIGMA = [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]

# Fig. 137 (i0)_10 [deg] digitized at beta1 = 40, 50 deg (SIGMA order).
I0_10 = {40: [6.32, 5.69, 5.02, 4.39, 3.77, 3.09, 2.47, 1.85, 1.17],
         50: [7.85, 7.02, 6.23, 5.47, 4.66, 3.86, 3.06, 2.25, 1.46]}
# Fig. 161 (delta0)_10 [deg] digitized at beta1 = 50 deg (sigma 2.0..0.6).
D0_10 = {50: [1.84, 1.71, 1.57, 1.41, 1.22, 1.05, 0.82, 0.58]}

I0_CASES = [(b, s, v) for b, row in I0_10.items() for s, v in zip(SIGMA, row)]
D0_CASES = [(b, s, v) for b, row in D0_10.items()
            for s, v in zip(SIGMA, row)]


@pytest.mark.parametrize("b1,sigma,chart", I0_CASES)
def test_reference_incidence_matches_sp36_fig137(b1, sigma, chart):
    # (i0)_10: code reproduces the digitized Fig. 137 point within the raster
    # reading precision. camber=0, t/c=0.10 -> reference_incidence == (i0)_10.
    got = float(reference_incidence(float(b1), sigma, 0.10, 0.0)[0])
    assert got == pytest.approx(chart, abs=0.4)


@pytest.mark.parametrize("b1,sigma,chart", D0_CASES)
def test_reference_deviation_matches_sp36_fig161(b1, sigma, chart):
    # (delta0)_10: code reproduces the digitized Fig. 161 point.
    got = float(reference_deviation(float(b1), sigma, 0.10, 0.0)[0])
    assert got == pytest.approx(chart, abs=0.5)


def test_sp36_incidence_aggregate_agreement():
    # The whole (i0)_10 set agrees in aggregate to well under the reading
    # precision (measured RMS ~0.10 deg) -- guards a systematic fit drift that
    # per-point tolerances could each individually absorb.
    errs = [float(reference_incidence(float(b), s, 0.10, 0.0)[0]) - v
            for b, s, v in I0_CASES]
    rms = float(np.sqrt(np.mean(np.square(errs))))
    assert rms < 0.20
    assert abs(float(np.mean(errs))) < 0.15      # no systematic bias


def test_camber_slopes_match_sp36_fig138_and_162():
    # n (Fig. 138) and m (Fig. 162) extracted from the code as d/d(camber),
    # spot-pinned at the well-separated high-beta1 / extreme-sigma corners the
    # overlay confirmed (chart reads: n(70,0.4)~-0.46, n(70,2.0)~-0.20;
    # m(70,0.4)~0.52, m(70,2.0)~0.24).
    def n(b, s):
        return float(reference_incidence(b, s, 0.10, 1.0)[0]
                     - reference_incidence(b, s, 0.10, 0.0)[0])

    def m(b, s):
        return float(reference_deviation(b, s, 0.10, 1.0)[0]
                     - reference_deviation(b, s, 0.10, 0.0)[0])

    assert n(70, 0.4) == pytest.approx(-0.46, abs=0.05)
    assert n(70, 2.0) == pytest.approx(-0.20, abs=0.05)
    assert m(70, 0.4) == pytest.approx(0.52, abs=0.05)
    assert m(70, 2.0) == pytest.approx(0.24, abs=0.05)
