"""Wiesner slip-factor closure for centrifugal (radial) impeller rows
(Theory Manual sections 3.4, 7.1, 7.3; Wiesner 1967).

Provenance: the radial-impeller analogue of the axial deviation closure --
the exit tangential velocity falls short of the blade-congruent value by the
slip velocity ``C_slip = (1 - sigma) U2``, with the Wiesner slip factor

    sigma = 1 - sqrt(cos(beta2b)) / Z^0.7

(``beta2b`` the exit backsweep from the meridional/radial direction, ``Z``
the blade count). The exit velocity triangle at the radial exit (phi -> 90
deg, so the meridional velocity IS the radial velocity) is

    W_theta2 = -Vm2 tan(beta2b),   V_theta2 = sigma U2 + W_theta2,

i.e. ``V_theta2 = sigma U2 - Vm2 tan(beta2b)`` -- backsweep and slip both
reduce the exit swirl (and the Euler work). The closure is phi-agnostic in
form: it reads the meridional (radial) velocity and the exit radius from the
flow view, so it works at any exit orientation the grid provides.

Verification status (see ``docs/references/WIE67.md``, 2026-07-09, cross-
checked against Aungier/Braembussche/Cumpsty/Dixon/Lakshminarayana/Whitfield-
Baines): the base form is CONFIRMED -- ``sqrt(cos)``, the ``Z**0.7`` exponent,
no leading coefficient, and ``beta2b`` referenced from the RADIAL direction
(the standard Wiesner/American convention; Aungier's ``sin`` form is the same
quantity referenced from the tangent). Pinned in ``test_wiesner_reference.py``.

**Radius-ratio limit correction (RESOLVED 2026-07, WIE67.md finding 1).**
Wiesner's slip factor is reduced once ``r1/r2`` exceeds a limit
``eps_lim = exp(-8.16 cos(beta2b)/Z)`` (from-radial form -- NB ``cos`` not the
``sin`` of Aungier's tangent convention; constant 8.16 confirmed). Above it,
``sigma_corr = sigma*[1 - ((r1/r2 - eps_lim)/(1 - eps_lim))**3]`` (Braembussche
3.84; the cubic is the 3-source consensus Cumpsty/Dixon/Braembussche, adopted
over Aungier 4-10's ``beta2/10`` outlier). ``wiesner_slip`` now applies it when
given the ``radius_ratio`` argument, which ``exit_rvt`` supplies as
``r1/r2 = flow.r / flow.r_te`` from the flow view -- NO section 4.1 geometry-
contract addition was needed (the inducer/exit radii are already in the lagged
flow). Inactive when ``r1/r2 < eps_lim`` (behavior-preserving for in-limit
designs like V7). The von Backstrom single-parameter alternative is NOT in the
library, so it stays an unresourced option.

Smoothness (section 7.3): the backsweep is soft-saturated into ``[0, 85 deg)``
so ``sqrt(cos)`` and ``tan`` stay bounded, and every evaluation returns a
compact-support validity measure. No raw clamps, no exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._namespace import get_xp
from ..interfaces import RowFlowView, RowView, SwirlResult
from ..smoothmath import abs_smooth, blend, smooth_min, soft_clip, softplus

__all__ = ["wiesner_slip", "WiesnerSlip", "BACKSWEEP_CAL"]

_DEG = 3.141592653589793 / 180.0
# Backsweep magnitude cap (keep sqrt(cos) > 0 and tan bounded) and the
# calibration window (lo, hi, width) in radians for the validity measure.
_B2B_CAP, _B2B_W = 85.0 * _DEG, 2.0 * _DEG
BACKSWEEP_CAL = (0.0, 70.0 * _DEG, 3.0 * _DEG)   # calibrated backsweep band
_Z_CAL = (3.0, 40.0, 1.0)                        # calibrated blade-count band
# Radius-ratio limit correction (WIE67.md finding 1): the limit exponent
# ``eps_lim = exp(-8.16 cos(beta2b)/Z)`` (Cumpsty/Dixon 7.35c; constant 8.16
# confirmed) and the softplus width for the (r1/r2 - eps_lim) positive part.
_EPS_LIM_C = 8.16
_RR_W = 0.02


def wiesner_slip(beta2b_rad, blade_count, radius_ratio=None, *, xp=None):
    """Wiesner slip factor ``sigma = 1 - sqrt(cos(beta2b)) / Z^0.7``
    (section 3.4; form/exponent CONFIRMED, docs/references/WIE67.md).

    ``beta2b`` is the exit backsweep from the meridional/radial direction
    (sign-agnostic; the magnitude enters through ``cos``). When the inlet/exit
    radius ratio ``radius_ratio = r1/r2`` is supplied, the radius-ratio limit
    correction is applied: above ``eps_lim = exp(-8.16 cos(beta2b)/Z)`` the base
    slip over-predicts and is reduced by the Braembussche 3.84 cubic
    ``sigma *= 1 - ((r1/r2 - eps_lim)/(1 - eps_lim))^3`` (3-source consensus
    Cumpsty/Dixon/Braembussche over Aungier's ``beta2/10`` outlier;
    docs/references/WIE67.md finding 1). Below the limit the smooth positive
    part is ~0, so the base form is untouched. ``radius_ratio=None`` skips the
    correction entirely (the base-form contract callers/reference tests rely on).

    Returns ``(sigma, validity)`` with the compact-support :data:`BACKSWEEP_CAL`
    / blade-count calibration windows."""
    xp = get_xp(xp)
    b = smooth_min(abs_smooth(beta2b_rad, 1.0e-4, xp=xp), _B2B_CAP, _B2B_W,
                   xp=xp)
    sigma = 1.0 - xp.sqrt(xp.cos(b)) / blade_count ** 0.7
    if radius_ratio is not None:
        # C1 (section 7.3): softplus positive part -> ~0 below the limit, a
        # smooth ramp above it; smooth_min caps the normalized excess at 1 so
        # the multiplicative factor stays in [0, 1] (sigma never goes negative,
        # AD-10) even at the degenerate r1 -> r2.
        eps_lim = xp.exp(-_EPS_LIM_C * xp.cos(b) / blade_count)
        excess = smooth_min(
            softplus(radius_ratio - eps_lim, _RR_W, xp=xp) / (1.0 - eps_lim),
            1.0, _RR_W, xp=xp)
        sigma = sigma * (1.0 - excess * excess * excess)
    lo, hi, w = BACKSWEEP_CAL
    v_b = 1.0 - blend(b, hi, w, xp=xp)
    zlo, zhi, zw = _Z_CAL
    v_z = blend(blade_count, zlo, zw, xp=xp) \
        * (1.0 - blend(blade_count, zhi, zw, xp=xp))
    return sigma, v_b * v_z


@dataclass(frozen=True)
class WiesnerSlip:
    """SwirlClosure (section 7.1): centrifugal impeller exit rVt from the
    Wiesner slip factor.

    Reads the exit backsweep ``beta2_blade`` and blade count from
    ``row.geometry`` (the section 4.1 contract; blade orientation gives the
    swirl sign, ARCH-4.2) and the exit radius / meridional velocity from the
    lagged TE flow (``r_te``/``vm_te``, section 7.2). Blade speed
    ``U2 = omega r_te``; exit ``V_theta = sigma U2 - Vm2 tan(beta2b)``.

    Requires ``row.geometry`` with a nonzero ``blade_count``."""

    def exit_rvt(self, row: RowView, flow: RowFlowView) -> SwirlResult:
        xp = get_xp(None)
        g = row.geometry
        y = flow.psi                       # span fraction ~ mass fraction
        sgn = g.orientation                # geometry constant (ARCH-4.2)

        b2b = sgn * g.beta2_blade(y)       # signed backsweep [rad]
        # r1/r2 from the flow view (LE inducer radius / TE exit radius) drives
        # the radius-ratio limit correction -- no section 4.1 geometry addition.
        sigma, v = wiesner_slip(b2b, g.blade_count, flow.r / flow.r_te, xp=xp)
        u2 = row.omega * flow.r_te
        w_theta_2 = -flow.vm_te * xp.tan(
            soft_clip(b2b, -_B2B_CAP, _B2B_CAP, _B2B_W, xp=xp))
        vtheta_2 = sigma * u2 + w_theta_2
        return SwirlResult(rvt=flow.r_te * vtheta_2, validity=float(xp.min(v)))
