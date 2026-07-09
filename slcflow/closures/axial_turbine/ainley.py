"""Ainley-Mathieson throat-based exit-angle closure for axial-turbine rows
(Theory Manual sections 4.5, 3.4, 7.1, 7.3; the M2 -> 1 limit of the
Kacker-Okapuu 1982 exit-angle rule).

Provenance: the turbine gas exit angle is set primarily by the passage
throat via the cosine rule ``cos(alpha2) = o/s`` (throat opening ``o``,
pitch ``s``) — the classical Ainley-Mathieson result, and the transonic
limit of Kacker-Okapuu. The subsonic covered-passage / unguided-turning
deviation correction and the Mach interpolation between the low-speed
correlation and this sonic value are **deferred to the M6 transonic step**
(they need the exit Mach the M6-4 shock-loss work introduces); this module
ships the throat cosine rule alone, which is the dominant term and the
correct sonic asymptote.

Verified 2026-07-09 (docs/references/AM-ANGLE.md, vs AM R&M 2974 / K-O):
``cos(alpha2) = o/s`` is CONFIRMED as AM's M2=1 value (Eq 2,
``alpha2 = -cos^-1(A_t/A_n)`` -> gauge angle for straight backs; pinned in
test_ainley_reference.py). The DEFERRED correction is now precisely: the
low-speed AM Eq 1 ``alpha2 = alpha2* - 4(s/e)`` (constant 4; ``e`` = convex
back-surface radius of curvature throat->TE; ``alpha2* = f(cos^-1 o/s)`` Fig 5),
then a LINEAR interpolation of alpha2 in M2 over [0.5, 1.0] to the sonic value
(K-O/Dunham-Came adopt this). Needs exit Mach + back-surface ``e`` (not yet in
the section 4.1 contract). **[VERIFY when M6-transonic lands.]**

Frame convention (section 2.4 + the [VERIFY per correlation] remap duty):
the exit-angle magnitude ``arccos(o/s)`` is measured from the meridional;
it is signed by the blade's EXIT turning direction — the sign of the TE
metal angle (``geometry.orientation_te``; geometry data, constant per
solve, so branching on it is topology branching, ARCH-4.2) — and mapped
through the relative->absolute velocity triangle
(``V_theta = W_theta + omega r``), which degenerates to the absolute frame
for stators (``omega = 0``). The TE sign matters: a reaction rotor with
co-rotating relative inflow has LE and TE metal angles of OPPOSITE sign,
and signing the exit angle by the LE orientation flips the exit swirl and
turns work extraction into work input (2026-07 audit finding; the
regression is ``test_ainley.py::test_reaction_rotor_corotating_inflow``).

Smoothness (section 7.3): the ``o/s`` ratio is soft-clipped into the
arccos domain (``arccos'`` blows up at +-1, so the argument is kept
strictly inside), the exit-angle magnitude is smoothly capped, and every
evaluation returns a compact-support validity measure (1 inside the
calibrated throat/pitch band, C1-smoothly to 0 outside).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._namespace import get_xp
from ..interfaces import RowFlowView, RowView, SwirlResult
from ..smoothmath import blend, smooth_min, soft_clip

__all__ = ["throat_exit_angle", "AinleyTurbineSwirl", "COS_ARG"]

# Calibrated throat/pitch band (lo, hi, transition width) for the validity
# window: typical turbine cascades run o/s ~ 0.3-0.75 (exit angle ~ 41-73
# deg); the window flags only near-degenerate throats. [VERIFY range]
COS_ARG = (0.10, 0.95, 0.03)
# arccos MATHEMATICAL domain guard: soft-clip o/s strictly inside (0, 1) so
# the fractional-derivative endpoints are never reached (bounded derivative,
# section 7.3.2). Corner width kept well below the calibration window.
_ARG_LO, _ARG_HI, _ARG_W = 0.02, 0.999, 0.01
# Smooth cap on the exit-angle magnitude [deg]: arccos(0.02) ~ 88.9 deg
# would give a runaway tangent; cap below that (the low end is left exact so
# near-axial exits are undistorted). [VERIFY ceiling choice]
_ALPHA_CEIL, _ALPHA_W = 85.0, 2.0


def throat_exit_angle(os_ratio, *, xp=None):
    """Gas exit-angle magnitude [deg] from the throat/pitch ratio via the
    cosine rule ``alpha2 = arccos(o/s)`` (section 4.5).

    ``o/s`` is soft-clipped into the arccos domain (bounded derivative,
    section 7.3.2) and the result smoothly capped below 90 deg. Returns
    ``(alpha2_deg, validity)`` with the :data:`COS_ARG` compact-support
    calibration window. The ``arccos(o/s)`` sonic value is CONFIRMED (AM Eq 2,
    AM-ANGLE.md); the low-speed ``-4(s/e)`` term + linear M2 blend are deferred
    to M6-transonic (need exit Mach + back-surface ``e``)."""
    xp = get_xp(xp)
    arg = soft_clip(os_ratio, _ARG_LO, _ARG_HI, _ARG_W, xp=xp)
    alpha2 = xp.rad2deg(xp.arccos(arg))
    alpha2 = smooth_min(alpha2, _ALPHA_CEIL, _ALPHA_W, xp=xp)
    lo, hi, w = COS_ARG
    v = blend(os_ratio, lo, w, xp=xp) * (1.0 - blend(os_ratio, hi, w, xp=xp))
    return alpha2, v


@dataclass(frozen=True)
class AinleyTurbineSwirl:
    """SwirlClosure (section 7.1): exit rVt from the throat-based gas exit
    angle.

    Pitch ``s = 2 pi r_te / N`` from the lagged TE radius and the blade
    count (a config constant, taken from ``row.geometry.blade_count`` which
    is validated >= 1 at construction); throat ``o`` from
    ``row.geometry.throat`` (section 4.5). The exit gas-angle magnitude is
    ``arccos(o/s)``, signed by the TE turning direction
    (``geometry.orientation_te`` — NOT the LE orientation, see module
    docstring), applied to the lagged TE meridional velocity (section 7.2).
    Relative frame for rotors (``omega != 0``), absolute for stators,
    unified via ``V_theta = W_theta + omega r`` (section 2.4).

    Requires ``row.geometry`` implementing the section 4.1 contract with a
    ``throat`` opening, and the view's ``r_te``/``vm_te`` fields
    (driver-provided, lagged)."""

    def exit_rvt(self, row: RowView, flow: RowFlowView) -> SwirlResult:
        xp = get_xp(None)
        g = row.geometry
        y = flow.psi                      # span fraction ~ mass fraction
        sgn = g.orientation_te            # TE turning direction (ARCH-4.2)

        pitch = 2.0 * xp.pi * flow.r_te / g.blade_count
        os_ratio = g.throat(y) / pitch
        alpha2_deg, v = throat_exit_angle(os_ratio, xp=xp)

        w_theta_2 = flow.vm_te * xp.tan(sgn * xp.deg2rad(alpha2_deg))
        vtheta_2 = w_theta_2 + row.omega * flow.r_te
        return SwirlResult(rvt=flow.r_te * vtheta_2, validity=float(xp.min(v)))
