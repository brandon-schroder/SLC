"""Reference-verified Ainley-Mathieson turbine exit-angle rule.

Pins the throat cosine rule confirmed against AM R&M 2974 / Kacker-Okapuu in
``docs/references/AM-ANGLE.md`` (extracted 2026-07-09): at M2=1 the gas exit
angle is ``arccos(o/s)`` (AM Eq 2 gauge angle). The low-speed ``-4(s/e)`` term
and linear M2 blend are deferred (M6-transonic) and not exercised here.
"""
import numpy as np
import pytest

from slcflow.closures.axial_turbine.ainley import throat_exit_angle

DEG = np.pi / 180.0


@pytest.mark.parametrize("os_ratio", [0.3, 0.45, 0.6, 0.75])
def test_throat_exit_angle_is_arccos_os(os_ratio):
    # AM Eq 2 (M2=1): alpha2 = arccos(o/s). Deep inside the arccos domain the
    # soft-clip and the 85-deg cap are inactive, so it is exact.
    expect_deg = np.degrees(np.arccos(os_ratio))
    assert float(throat_exit_angle(os_ratio)[0]) == pytest.approx(
        expect_deg, rel=1e-4)


def test_exit_angle_monotone_in_os():
    # Tighter throat (smaller o/s) -> larger exit angle (more turning).
    a_tight = float(throat_exit_angle(0.35)[0])
    a_open = float(throat_exit_angle(0.70)[0])
    assert a_tight > a_open


def test_cos_of_exit_angle_recovers_os():
    # Round-trip: cos(alpha2) == o/s in the calibrated band.
    for os_ratio in (0.4, 0.55, 0.7):
        a = float(throat_exit_angle(os_ratio)[0])
        assert np.cos(a * DEG) == pytest.approx(os_ratio, rel=1e-4)
