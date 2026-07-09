"""Kacker-Okapuu (1982) profile-loss chain for axial-turbine rows (Theory
Manual sections 4.3, 4.4, 7.1, 7.3; Appendix B.3 conversion).

Provenance: the Kacker-Okapuu recalibration of the Ainley-Mathieson /
Dunham-Came profile loss:

    Y_p,AM = [ Y_p(b1=0) + |b1/b2|(b1/b2) (Y_p(b1=b2) - Y_p(b1=0)) ] (t/c / 0.2)^(b1/b2)
    Y_profile = 0.914 (2/3 Y_p,AM K_p + Y_shock) f_Re

where ``b1`` is the inlet flow angle and ``b2`` the exit gas angle (both from
the meridional), ``K_p`` the compressible profile-loss correction (K-O found
the incompressible AM charts over-predict as M2 rises), and ``f_Re`` the
Reynolds correction. The two reference curves ``Y_p(b1=0)`` (nozzle) and
``Y_p(b1=b2)`` (impulse) are smooth surrogate fits to the AM charts (minimum
near the loading-optimal pitch/chord, rising with exit angle; impulse loss
higher, its optimum at tighter spacing).

Verification status (see ``docs/references/KO82.md`` for the term-by-term
transcription + citations, 2026-07-09): the **scalar formula constants** are
CONFIRMED against KO82/DC/AM — the 0.914 and 2/3 bracket factors, K_p form +
Mach endpoints + K2, f_Re knees/exponents, the secondary 0.0334 (DC) x 1.2
(K-O) + f_AR, the shock 0.75/(M-0.4)^1.75, and the (t/c/0.2)^(b1/b2) thickness
exponent. The **loading sign convention** is confirmed frame-consistent (KO82
uses sum-of-tangents for load / difference for mean angle; our signed frame
swaps them — same physics, see the secondary_loss note). **Still [VERIFY]:**
the two nozzle/impulse reference curves ``yp1``/``yp2`` and the TE ``phi2``
curves are surrogate fits to the AM/K-O *charts* — they need reference-figure
points (digitization), not a formula lookup. The profile interpolation weight
uses KO82's signed ``|b1/b2|(b1/b2)`` (**resolved 2026-07** from the prior
AM-1957 symmetric ``(b1/b2)^2``; identical for ``b1>=0`` so behavior-preserving
for every in-domain case, differs only at negative incidence) — see
``profile_loss_am`` and ``docs/references/KO82.md``.

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
from ..smoothmath import (abs_smooth, blend, blend_between, smooth_max,
                          smooth_min, soft_clip, softplus)

__all__ = ["profile_loss_am", "mach_profile_correction",
           "reynolds_correction", "secondary_loss", "trailing_edge_zeta",
           "shock_loss", "CALIBRATED"]

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
# KO82 signed interpolation weight |r|*r: abs_smooth eps (small vs the r
# scale, so the weight ~ r^2 away from 0 and C1 through it), and a positivity
# floor (fraction of the nozzle loss) the signed weight may not drive the
# bracket below -- an AD-10 safety on deep negative-incidence EXTRAPOLATION
# only (no in-domain case reaches it; V6 runs r in [0.04, 0.72]).
_WEIGHT_EPS, _YP_FLOOR_FRAC = 0.05, 0.5
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

    Interpolates on KO82's signed angle-ratio weight ``|r|r`` (``r=b1/b2``)
    between the nozzle (``b1=0``) and impulse (``b1=b2``) reference curves,
    with the K-O thickness correction ``(t/c / 0.2)^(b1/b2)``. Returns
    ``(Y_p_AM, validity)``.
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
    # KO82's SIGNED interpolation weight |r|*r (Kacker-Okapuu modified
    # AM-1957's symmetric r^2 "to account for negative inlet angles",
    # docs/references/KO82.md finding 1, resolved 2026-07). Identical for
    # r>=0 -- so behavior-preserving for every in-domain case (V6 runs
    # r in [0.04, 0.72]); for r<0 it lets the loss dip below nozzle (the
    # negative-incidence bucket) instead of AM's symmetric rise. The bracket
    # is floored to a fraction of the nozzle loss so the surrogate curves
    # cannot return negative loss under deep negative-r extrapolation (AD-10;
    # off-design safety, not a calibration -- no in-domain case reaches it).
    weight = abs_smooth(r, _WEIGHT_EPS, xp=xp) * r
    bracket = smooth_max(yp1 + weight * (yp2 - yp1), _YP_FLOOR_FRAC * yp1,
                         _R_W, xp=xp)
    yp_am = bracket * tfac
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


# --- secondary loss (Kacker-Okapuu / Ainley-Mathieson) --------------------
_AR_FLOOR, _AR_W = 0.3, 0.1        # blade aspect-ratio floor / blend width
_TAN_EPS = 1.0e-3                  # abs_smooth epsilon on the loading term
_A1_CLIP = (-85.0, 85.0, 2.0)      # inlet-angle soft-clip (cos not -> 0)


def _aspect_ratio_factor(ar, *, xp=None):
    """Ainley-Mathieson aspect-ratio factor ``f_AR``:
    ``(1 - 0.25 sqrt(2 - AR)) / AR`` for ``AR < 2``, ``1/AR`` for ``AR >= 2``
    — the two branches blended C1 across ``AR = 2`` (continuous there, both
    give ``0.5``). Low aspect ratio => larger secondary loss. [VERIFY]"""
    xp = get_xp(xp)
    a = smooth_max(ar, _AR_FLOOR, _AR_W, xp=xp)
    lo = (1.0 - 0.25 * xp.sqrt(smooth_max(2.0 - a, 0.0, _AR_W, xp=xp))) / a
    hi = 1.0 / a
    return blend_between(a, lo, hi, 2.0, 0.1, xp=xp)


def secondary_loss(alpha1_deg, alpha2_deg, aspect_ratio, *, xp=None):
    """Kacker-Okapuu secondary (endwall) loss coefficient ``Y_s`` (B.3
    exit-dynamic-head reference; [VERIFY coefficients]).

    ``Y_s = 1.2 * 0.0334 * f_AR * (cos a2 / cos a1) * (C_L/(s/c))^2 *
    cos^2 a2 / cos^3 a_m`` with the tangential loading
    ``C_L/(s/c) = 2 |tan a2 - tan a1| cos a_m`` and mean angle
    ``tan a_m = 1/2 (tan a1 + tan a2)`` — frame-safe in signed cascade
    angles (the loading is the whirl change ``d(Vtheta)/Vm``). ``a1`` is
    soft-clipped away from +-90 deg (cos not -> 0). The K-O secondary Mach
    factor ``K_s`` is **[VERIFY]-deferred** (second order; K_s = 1 here).
    Returns ``(Y_s, validity)``."""
    xp = get_xp(xp)
    a1 = xp.deg2rad(soft_clip(alpha1_deg, *_A1_CLIP, xp=xp))
    a2 = xp.deg2rad(alpha2_deg)
    t1, t2 = xp.tan(a1), xp.tan(a2)
    tan_m = 0.5 * (t1 + t2)
    cos_m = 1.0 / xp.sqrt(1.0 + tan_m * tan_m)
    load = 2.0 * abs_smooth(t2 - t1, _TAN_EPS, xp=xp) * cos_m   # C_L/(s/c)
    cos_a1, cos_a2 = xp.cos(a1), xp.cos(a2)
    z = load * load * cos_a2 * cos_a2 / cos_m ** 3
    ys = 1.2 * 0.0334 * _aspect_ratio_factor(aspect_ratio, xp=xp) \
        * (cos_a2 / cos_a1) * z
    v = blend(aspect_ratio, 0.8, 0.2, xp=xp) \
        * (1.0 - blend(aspect_ratio, 8.0, 0.5, xp=xp))
    return ys, v


# --- trailing-edge loss (Kacker-Okapuu energy coefficient) ----------------
_TE_CEIL, _TE_W = 0.3, 0.02       # kinetic-energy coefficient ceiling


def trailing_edge_zeta(alpha1_deg, alpha2_deg, te_o_ratio, *, xp=None):
    """Kacker-Okapuu trailing-edge kinetic-energy loss coefficient
    ``dPhi^2_TE`` (interpolated between axial-entry and impulse blades on the
    squared angle ratio, each a smooth function of TE-thickness/throat
    ``t_TE/o``; [VERIFY coefficients]).

    Returned as a kinetic-energy coefficient ``zeta`` (the caller maps it to
    an exit-reference ``Y`` before summing with the profile/secondary terms).
    Smoothly ceilinged; returns ``(zeta, validity)``."""
    xp = get_xp(xp)
    x = smooth_max(te_o_ratio, 0.0, 0.005, xp=xp)          # t_TE/o >= 0
    phi2_ax = 0.4 * x + 2.0 * x * x
    phi2_imp = 0.7 * x + 4.0 * x * x
    a2 = smooth_max(alpha2_deg, 5.0, 1.0, xp=xp)           # avoid /0
    r = soft_clip(alpha1_deg / a2, _R_LO, _R_HI, _R_W, xp=xp)
    zeta = smooth_min(phi2_ax + r * r * (phi2_imp - phi2_ax), _TE_CEIL,
                      _TE_W, xp=xp)
    v = 1.0 - blend(x, 0.15, 0.05, xp=xp)
    return zeta, v


# --- inlet shock loss (Kacker-Okapuu transonic component) -----------------
_M_SHOCK, _SHOCK_W = 0.4, 0.05    # leading-edge shock onset Mach + knee width
_SHOCK_C, _SHOCK_EXP = 0.75, 1.75  # K-O coefficient + exponent [VERIFY]


def shock_loss(m1, *, xp=None):
    """Kacker-Okapuu leading-edge / inlet shock-loss coefficient
    ``Y_shock = 0.75 (M1 - 0.4)^1.75`` (B.3 exit-dynamic-head reference),
    entering the K-O profile bracket ``0.914 (2/3 Y_p,AM K_p + Y_shock)``.

    Written C1 via ``softplus`` on the Mach excess (the raw ``(M1 - 0.4)``
    kinks at onset): near zero below ``M1 ~ 0.4``, growing smoothly above.
    This is the transonic term the M5/V9 choke-knee note is about; per the
    original K-O correlation it is strongest at the hub (highest ``U``), here
    evaluated at the local relative Mach per streamtube. The K-O geometric
    ``(r_hub/r_tip)`` and pressure-ratio factors are **[VERIFY]-deferred**
    (they reduce the bare coefficient). Returns ``(Y_shock, validity)`` with
    the calibration fading above ``M1 ~ 1.6``."""
    xp = get_xp(xp)
    excess = softplus(m1 - _M_SHOCK, _SHOCK_W, xp=xp)      # >= 0, smooth onset
    y = _SHOCK_C * excess ** _SHOCK_EXP
    v = 1.0 - blend(m1, 1.6, 0.2, xp=xp)
    return y, v
