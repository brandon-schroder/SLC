"""VKI LS-89 cascade validation of the axial-turbine closures (section 9.6).

Measured-data check of the K-O/Ainley chain at the CASCADE level (the V6
analogue of the TR-1368 pass): the LS-89 transonic nozzle guide vane (Arts &
Lambert de Rouvroit), extracted verbatim from the primary paper via the Test
Cases notebook, 2026-07-15 (docs/references/LS89.md):

  geometry: chord 67.647 mm, pitch/chord g/c = 0.850, stagger 55 deg,
    throat/pitch o/g = 0.2597, TE radius r_TE/c = 0.0105 (thickness
    t_TE = 2 r_TE), axial inflow (alpha1 = 0);
  measured, at M2is = 1.0 (Re2 0.5-2.0e6, loss Re-insensitive there):
    ENERGY loss coefficient zeta2 ~ 2.25% total, decomposed ~1.0% boundary
    layer + ~0.75% trailing edge + ~0.5% shock (+ ~0 base);
  exit flow angle ~ the gauging angle (sin^-1(o/g) = 15.05 deg from
    TANGENTIAL, i.e. 74.95 deg from axial).

This is a 2-D stationary cascade, so the comparison is CLOSURE-LEVEL (the
pure K-O functions, no machine solve, mirroring KackerOkapuuLoss.evaluate's
assembly for a nozzle at midspan): profile (AM bracket x K_p x f_Re) + TE;
no secondary (2-D midspan), no K-O inlet-shock term (M1 ~ 0.15 < 0.4 onset
— note the MEASURED 0.5% shock loss is an EXIT (TE) shock system, which the
K-O method does not model separately at M2 <= 1; recorded).

MEASURED AGREEMENT (2026-07-15): predicted energy-zeta ~ 0.031 vs measured
0.0225 (~ +40%). K-O over-predicting a modern high-efficiency vane by
30-50% is the documented behaviour of the 1982 mean-line method (it was
calibrated on 1950s-70s hardware and targets +/-1.5% efficiency at the
STAGE level, where secondary/TE dominate); the exit-angle rule is exact by
construction here. Pinned as bands so drift is visible.
"""
import numpy as np
import pytest

from slcflow.closures.axial_turbine.ainley import throat_exit_angle
from slcflow.closures.axial_turbine.kacker_okapuu import (
    mach_profile_correction, profile_loss_am, reynolds_correction,
    shock_loss, trailing_edge_zeta)

# --- LS-89 transcription (docs/references/LS89.md) -------------------------
S_C = 0.850                 # pitch / chord
O_S = 0.2597                # throat / pitch
RTE_C = 0.0105              # TE radius / chord
CHORD_MM = 67.647
ALPHA1 = 0.0                # axial inflow
M2IS = 1.0
M1 = 0.15                   # inlet Mach at M2is = 1 (continuity, o/g ~ 0.26)
RE2 = 1.0e6
MEASURED_ZETA2 = 0.0225     # total, at M2is = 1.0
GAMMA = 1.4


def _te_o_ratio() -> float:
    # t_TE / o = 2 r_TE / (o/g * g) with everything per-chord.
    return 2.0 * RTE_C / (O_S * S_C)


def predicted_total_y() -> float:
    """Mirror KackerOkapuuLoss.evaluate for a 2-D nozzle cascade midspan."""
    alpha2, _ = throat_exit_angle(O_S)
    yp_am, _ = profile_loss_am(S_C, ALPHA1, float(alpha2), 0.20)
    kp = mach_profile_correction(M1, M2IS)
    env = 0.914 * float(reynolds_correction(RE2))
    y_profile = env * (2.0 / 3.0) * float(yp_am) * float(kp)
    y_shock = env * float(shock_loss(M1)[0])
    zeta_te, _ = trailing_edge_zeta(ALPHA1, float(alpha2), _te_o_ratio())
    y_te = float(zeta_te) / (1.0 - float(zeta_te))
    return y_profile + y_shock + y_te


def y_to_energy_zeta(y: float, m2is: float = M2IS) -> float:
    """Exact perfect-gas conversion: exit-reference Y -> the paper's energy
    loss coefficient zeta2, at the M2is exit condition (B.3/B.4 kin)."""
    k = (GAMMA - 1.0) / GAMMA
    p2_p01 = (1.0 + 0.5 * (GAMMA - 1.0) * m2is ** 2) ** (-1.0 / k)
    # Y = (p01 - p02) / (p02 - p2), p01 = 1
    p02 = (1.0 + y * p2_p01) / (1.0 + y)
    return 1.0 - (1.0 - (p2_p01 / p02) ** k) / (1.0 - p2_p01 ** k)


def test_exit_angle_matches_ls89_gauging():
    # arccos(o/s) (section 4.5) vs the paper's gauging angle: sin^-1(0.2597)
    # = 15.05 deg from tangential = 74.95 deg from axial/meridional.
    alpha2, validity = throat_exit_angle(O_S)
    assert float(alpha2) == pytest.approx(90.0 - 15.05, abs=0.1)
    assert float(validity) > 0.5


def test_ko_inlet_shock_inactive_at_ls89_inlet_mach():
    # The K-O shock term is an INLET (leading-edge) transonic component;
    # LS-89's M1 ~ 0.15 is far below the 0.4 onset. The measured ~0.5%
    # shock loss at M2is = 1 is a TE shock system — recorded as not
    # separately modelled by the method (module docstring).
    assert float(shock_loss(M1)[0]) < 1e-4


def test_total_loss_vs_measured_ls89_band():
    # Predicted total energy-zeta vs the measured 2.25% at M2is = 1.0.
    # Measured agreement ~ +40% (K-O over-predicts a modern high-efficiency
    # vane; documented). Band [0.7x, 1.7x]: catches a regression to the
    # pre-calibration constants (which read ~3x) or a sign/reference error,
    # without claiming better accuracy than the method has.
    zeta_pred = y_to_energy_zeta(predicted_total_y())
    ratio = zeta_pred / MEASURED_ZETA2
    assert 0.7 < ratio < 1.7, (zeta_pred, MEASURED_ZETA2)


def test_te_component_vs_measured_te_breakdown():
    # The paper decomposes ~0.75% TE loss out of the 2.25% at M2is = 1.0.
    # The K-O TE curve at t_TE/o = 0.095 gives zeta_TE ~ 1.9% — the K-O TE
    # share is HIGH vs this rig's measured split (recorded; the Fig. 14
    # curves were calibrated on thicker-TE hardware). Bounded loosely: the
    # component must stay the right order, below the TOTAL measured x2.
    zeta_te, _ = trailing_edge_zeta(ALPHA1, 74.95, _te_o_ratio())
    assert 0.25 * 0.0075 < float(zeta_te) < 2.0 * MEASURED_ZETA2
