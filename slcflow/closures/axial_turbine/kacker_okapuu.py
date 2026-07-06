"""Kacker-Okapuu (1982) profile-loss chain for axial-turbine rows (Theory
Manual sections 4.3, 4.4, 7.1, 7.3; Appendix B.3 conversion).

Provenance: the Kacker-Okapuu recalibration of the Ainley-Mathieson /
Dunham-Came profile loss:

    Y_p,AM = [ Y_p(b1=0) + (b1/b2)^2 (Y_p(b1=b2) - Y_p(b1=0)) ] (t/c / 0.2)^(b1/b2)
    Y_profile = 0.914 (2/3 Y_p,AM K_p + Y_shock) f_Re

where ``b1`` is the inlet flow angle and ``b2`` the exit gas angle (both from
the meridional), ``K_p`` the compressible profile-loss correction (K-O found
the incompressible AM charts over-predict as M2 rises), and ``f_Re`` the
Reynolds correction. The two reference curves ``Y_p(b1=0)`` (nozzle) and
``Y_p(b1=b2)`` (impulse) are smooth surrogate fits to the AM charts (minimum
near the loading-optimal pitch/chord, rising with exit angle; impulse loss
higher, its optimum at tighter spacing). **[VERIFY: every coefficient and
the two reference curves against the library copies (Ainley-Mathieson 1951;
Dunham-Came 1970; Kacker-Okapuu 1982) — encoded from general knowledge
pending the library pass, as for the compressor set.]**

The shock component ``Y_shock`` lands at M6-4 (the V5 choke-knee is waiting
on it); this module ships the subsonic chain. Per-node Reynolds number from
a viscosity/transport backend is deferred (ARCH-9); the loss model carries a
design Reynolds number, defaulting into the flat band so ``f_Re = 1``.

Smoothness (section 7.3): inputs soft-saturate into the fits' calibrated
domain over documented widths, and each fit returns a compact-support
validity measure. No raw clamps, no exceptions.
"""
from __future__ import annotations

from ..._namespace import get_xp
from ..smoothmath import (blend, blend_between, smooth_max, smooth_min,
                          soft_clip, softplus)

__all__ = ["profile_loss_am", "mach_profile_correction",
           "reynolds_correction", "CALIBRATED"]

# Calibrated input domain (lo, hi, transition width, mathematical floor) for
# the validity windows + the hard floor the saturated value may not cross.
# [VERIFY ranges against the AM/K-O charts.]
CALIBRATED = {
    "s_c": (0.4, 1.1, 0.05, 0.15),        # pitch/chord
    "beta2_deg": (20.0, 80.0, 2.0, 5.0),  # exit gas angle
    "tc": (0.05, 0.30, 0.01, 0.02),       # max thickness / chord
}
# Angle ratio b1/b2 soft-clip range (nozzle 0 -> impulse 1; a little slack
# for negative incidence and super-impulse). [VERIFY]
_R_LO, _R_HI, _R_W = -1.0, 1.2, 0.1
# Mach-correction constants (K-O): K1 ramps 1 -> 0 over M2 in [0.2, 1.0].
_M2_FLOOR, _M_W, _KP_FLOOR = 0.05, 0.05, 0.10


def _saturate(x, key, *, xp=None):
    """Smoothly saturate into a fit's safe domain and return the
    compact-support in-range validity (section 7.3.2), identical in spirit
    to the compressor set's ``_saturate``."""
    lo, hi, w, floor = CALIBRATED[key]
    x_sat = floor + softplus(smooth_min(x, hi, w, xp=xp) - floor, w, xp=xp)
    v = blend(x, lo, w, xp=xp) * (1.0 - blend(x, hi, w, xp=xp))
    return x_sat, v


def profile_loss_am(s_c, beta1_deg, beta2_deg, tc, *, xp=None):
    """Ainley-Mathieson profile-loss coefficient ``Y_p,AM`` (section 4.3;
    [VERIFY coefficients]).

    Interpolates on the squared angle ratio between the nozzle (``b1=0``)
    and impulse (``b1=b2``) reference curves, with the K-O thickness
    correction ``(t/c / 0.2)^(b1/b2)``. Returns ``(Y_p_AM, validity)``.
    """
    xp = get_xp(xp)
    sc, v1 = _saturate(s_c, "s_c", xp=xp)
    a2, v2 = _saturate(beta2_deg, "beta2_deg", xp=xp)
    t, v3 = _saturate(tc, "tc", xp=xp)
    u = a2 / 70.0                                   # normalized exit angle

    # Nozzle and impulse reference curves: parabola in s/c about a
    # loading-optimal pitch/chord that tightens with exit angle. [VERIFY]
    s_opt_n = 0.80 - 0.08 * u
    yp1 = 0.025 + 0.020 * u * u + 0.035 * ((sc - s_opt_n) / 0.35) ** 2
    s_opt_i = 0.62 - 0.10 * u
    yp2 = 0.045 + 0.055 * u * u + 0.070 * ((sc - s_opt_i) / 0.35) ** 2

    r = soft_clip(beta1_deg / a2, _R_LO, _R_HI, _R_W, xp=xp)
    tfac = (t / 0.20) ** r
    yp_am = (yp1 + r * r * (yp2 - yp1)) * tfac
    return yp_am, v1 * v2 * v3


def mach_profile_correction(m1, m2, *, xp=None):
    """Kacker-Okapuu compressible profile-loss correction ``K_p`` (the AM
    charts over-predict as the exit Mach rises):

    ``K1 = 1`` for ``M2 < 0.2``, ramping to 0 across ``M2 in [0.2, 1.0]``;
    ``K2 = (M1/M2)^2``; ``K_p = 1 - K2 (1 - K1)``. Built C1 with a smooth
    ramp and a positive floor. [VERIFY the ramp constants.]"""
    xp = get_xp(xp)
    m2s = smooth_max(m2, _M2_FLOOR, _M_W, xp=xp)
    k1 = 1.0 - 1.25 * (soft_clip(m2, 0.2, 1.0, _M_W, xp=xp) - 0.2)
    k2 = (m1 / m2s) ** 2
    kp = 1.0 - k2 * (1.0 - k1)
    return smooth_max(kp, _KP_FLOOR, _M_W, xp=xp)


def reynolds_correction(re, *, xp=None):
    """Kacker-Okapuu Reynolds correction ``f_Re`` (chord Reynolds number):
    ``(Re/2e5)^-0.4`` for ``Re < 2e5``, unity in ``[2e5, 1e6]``, mild
    ``(Re/1e6)^-0.2`` for ``Re > 1e6`` — the three branches blended C1 in
    ``log10(Re)`` over documented widths. [VERIFY exponents/knees.]"""
    xp = get_xp(xp)
    lre = xp.log10(re)
    lo = (re / 2.0e5) ** (-0.4)          # low-Re penalty (>= 1)
    hi = (re / 1.0e6) ** (-0.2)          # high-Re mild rise (<= 1)
    flat = blend_between(lre, lo, 1.0, xp.log10(2.0e5), 0.3, xp=xp)
    return blend_between(lre, flat, hi, xp.log10(1.0e6), 0.3, xp=xp)
