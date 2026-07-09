"""Lieblein diffusion-factor profile loss for axial-compressor rows
(Theory Manual sections 4.3, 4.4, 7.1, 7.3; Appendix B.2 conversion).

Provenance: Lieblein's 1959 equivalent-diffusion-ratio correlation chain as
reproduced in the standard texts (Cumpsty; Aungier; Dixon):

    D_eq   = (W1/W2) [1.12 + 0.61 (cos^2 beta1 / sigma)(tan beta1 - tan beta2)]
    theta/c = 0.004 / (1 - 1.17 ln D_eq)
    omega_bar_min = 2 (theta/c) (sigma / cos beta2) (W2/W1)^2

Verification status (docs/references/LIEB59.md, 2026-07-09, vs Aungier ch.6 /
Cumpsty / Dixon): D_eq (1.12, 0.61), theta/c (0.004, 1.17), the inlet-relative
reference dynamic head, and the ``omega_bar`` assembly are CONFIRMED (pinned in
tests/test_lieblein_loss_reference.py).

**FIXED 2026-07 (was a confirmed [BUG]).** The ``omega_bar`` velocity ratio was
coded inverted as ``(W1/W2)^2``; Aungier Eq 6-27 and Cumpsty Eq 1.32 give
``(W2/W1)^2`` (``profile_loss_coefficient``). For a compressor W2 < W1, so the
old form overestimated profile loss by ~(W1/W2)^4 (~4x at DF~0.45); the fix
lowers profile loss / raises efficiency. V5 is structural bands so it stayed in
range; the M4 _WBAR_CEIL / 10-deg bucket were NOT retuned against the old value
(verified by the full-suite re-run).

**Off-design model (resolved 2026-07):** the fixed-10-deg quadratic bucket
(an unverified substitution) is replaced by Aungier's ch.6 normalized-incidence
bucket -- a quadratic ``f = 1 + xi^2`` with PHYSICALLY-DERIVED asymmetric stall
ranges (``xi`` normalized by the positive-stall ``R_s`` above the min-loss
incidence and the choke ``R_c`` below), C1-matched to linear extrapolations in
deep stall (``stall_choke_ranges`` / ``off_design_bucket``; the Aungier shape
is itself a ``1 + xi^2`` bucket, so this is the *physical* width, not a new
shape). ``f(0) = 1`` leaves the design-incidence loss unchanged. Deferred
refinements ([VERIFY]): Mach-number adjustment of ``R_s``/``R_c``, and the
``(i-i*)^1.43`` D_eq surface-velocity term (which feeds max surface velocity,
not the loss bucket). The theta/c fit's validity ends near D_eq ~ 2.35
(denominator zero); saturation reflects that.

The native coefficient is the B.2 relative-total-pressure loss
``omega_bar`` referenced to the INLET relative dynamic head; conversion to
entropy happens HERE (section 4.4: converted at row exit, B.1
re-referencing included) so ``delta_s`` is what crosses the interface.
Exit state comes from the same Lieblein deviation set — the two closures
ship as one consistent CorrelationSet (section 7.1).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._namespace import get_xp
from ..conversions import delta_s_compressor_omega_bar, relative_stagnation
from ..interfaces import LossBreakdown, RowFlowView, RowView
from ..smoothmath import (blend, blend_between, smooth_min, soft_clip,
                          softplus)
from .lieblein import reference_deviation, reference_incidence

__all__ = ["equivalent_diffusion", "wake_momentum_thickness",
           "profile_loss_coefficient", "stall_choke_ranges",
           "off_design_bucket", "LieblienLoss"]

# theta/c fit domain: the denominator 1 - 1.17 ln(D_eq) vanishes at
# D_eq = e^(1/1.17) ~ 2.35; saturate D_eq below that with a smooth ceiling,
# calibration window [1.0, 2.0] for validity. [VERIFY range]
_DEQ_CEIL, _DEQ_W = 2.2, 0.05
_DEQ_CAL = (1.0, 2.0, 0.05)
_DEQ_FLOOR = 0.3
# Conservative asymptote for the final coefficient (section 7.3.2: losses
# level off smoothly, never grow to where the B.2 conversion loses its
# domain -- omega_bar * q_dyn must stay well below p0r_2,ideal). Measured
# at M4-4: violent mid-transient velocity triangles (vm_te collapse) push
# the raw chain to omega_bar ~ 40 and NaN the conversion without this.
_WBAR_CEIL, _WBAR_W = 0.5, 0.05   # [VERIFY ceiling choice]

# Aungier ch.6 OFF-DESIGN bucket (docs/references/LIEB59.md): the loss follows
# a normalized-incidence bucket xi with PHYSICALLY-DERIVED asymmetric stall
# ranges (positive-stall R_s, negative-stall/choke R_c), quadratic near design
# and C1-matched to linear extrapolations in deep stall. Replaces the fixed
# 10-deg quadratic bucket (an unverified substitution). [VERIFY constants]
_RS_C = (10.3, 2.92, 15.6, 8.2)      # R_s = a0 + (a1 - b1/a2) theta/a3
_RC_C = (9.0, 30.0, 0.48, 4.176)     # R_c = c0 - (1 - (c1/b1)^c2) theta/c3
_R_FLOOR, _R_FW = 2.0, 0.5           # stall-range positivity floor [deg]
_B1_FLOOR, _B1_FW = 5.0, 1.0         # beta1 floor for (30/b1)^0.48 [deg]
_DI_W = 1.0                          # incidence blend width for the asym. xi
_XI_POS, _XI_NEG, _XI_W = 1.0, -2.0, 0.3   # deep-stall breakpoints + blend


def equivalent_diffusion(w1, w2, beta1_rad, beta2_rad, sigma, *, xp=None):
    """Lieblein equivalent diffusion ratio ``D_eq`` (cascade frame,
    positive angles; [VERIFY coefficients])."""
    xp = get_xp(xp)
    turning = xp.tan(beta1_rad) - xp.tan(beta2_rad)
    return (w1 / w2) * (1.12 + 0.61 * xp.cos(beta1_rad) ** 2 / sigma
                        * turning)


def wake_momentum_thickness(d_eq, *, xp=None):
    """``theta*/c`` from ``D_eq`` (Lieblein 1959 fit; [VERIFY]).

    ``D_eq`` is smoothly saturated into the fit's mathematical domain
    (floor + ceiling below the denominator zero) per section 7.3.2;
    returns ``(theta_c, validity)`` with the compact-support calibration
    window [1.0, 2.0]."""
    xp = get_xp(xp)
    d = _DEQ_FLOOR + softplus(
        smooth_min(d_eq, _DEQ_CEIL, _DEQ_W, xp=xp) - _DEQ_FLOOR,
        _DEQ_W, xp=xp)
    lo, hi, w = _DEQ_CAL
    v = blend(d_eq, lo, w, xp=xp) * (1.0 - blend(d_eq, hi, w, xp=xp))
    return 0.004 / (1.0 - 1.17 * xp.log(d)), v


def profile_loss_coefficient(theta_c, sigma, beta2_rad, w1, w2, *, xp=None):
    """Lieblein profile total-pressure loss coefficient (inlet-relative
    reference dynamic head):

        omega_bar_min = 2 (theta*/c) (sigma / cos beta2) (W2/W1)^2

    Aungier Eq 6-27 / Cumpsty Eq 1.32 (CONFIRMED, docs/references/LIEB59.md).
    The velocity ratio is ``(W2/W1)^2`` -- for a compressor ``W2 < W1`` so this
    is < 1. (This was previously coded inverted as ``(W1/W2)^2``, a ~4x
    overestimate; fixed 2026-07.)"""
    xp = get_xp(xp)
    return 2.0 * theta_c * sigma / xp.cos(beta2_rad) * (w2 / w1) ** 2


def stall_choke_ranges(camber_deg, beta1_deg, *, xp=None):
    """Aungier ch.6 low-speed positive-stall (``R_s``) and negative-stall /
    choke (``R_c``) incidence ranges [deg] (docs/references/LIEB59.md;
    [VERIFY constants]):

        R_s = 10.3 + (2.92 - beta1/15.6) * theta / 8.2
        R_c = 9.0  - (1 - (30/beta1)^0.48) * theta / 4.176

    ``theta`` is the camber and ``beta1`` the (reference) inlet flow angle,
    both in the cascade frame [deg]. These are the bucket half-widths, so they
    are floored strictly positive (denominators downstream); ``beta1`` is
    floored before the ``(30/beta1)`` power. Mach-number adjustment of the
    ranges (Aungier's ``1 + 0.5 M^2`` / ``1 + 0.5 (K_sh M')^3`` factors) is a
    deferred refinement -- these are the low-speed values."""
    xp = get_xp(xp)
    theta = camber_deg
    b1 = _B1_FLOOR + softplus(beta1_deg - _B1_FLOOR, _B1_FW, xp=xp)
    a0, a1, a2, a3 = _RS_C
    r_s = a0 + (a1 - b1 / a2) * theta / a3
    c0, c1, c2, c3 = _RC_C
    r_c = c0 - (1.0 - (c1 / b1) ** c2) * theta / c3
    r_s = _R_FLOOR + softplus(r_s - _R_FLOOR, _R_FW, xp=xp)
    r_c = _R_FLOOR + softplus(r_c - _R_FLOOR, _R_FW, xp=xp)
    return r_s, r_c


def off_design_bucket(i, i_ref, r_s, r_c, *, xp=None):
    """Aungier ch.6 normalized-incidence loss multiplier ``f(xi) =
    omega/omega_min`` (docs/references/LIEB59.md; [VERIFY]).

    The normalized incidence is asymmetric about the min-loss incidence
    ``i_m`` (= ``i_ref`` at low speed):

        xi = (i - i_ref) / R_s   for i >= i_ref   (positive-stall side)
        xi = (i - i_ref) / R_c   for i <  i_ref   (choke side)

    and the multiplier is a quadratic bucket C1-matched to linear
    extrapolations in deep stall (``w_s`` shock term is 0 for the subsonic
    Lieblein set, so ``omega = omega_min * f``):

        f = 1 + xi^2          for -2 <= xi <= 1
        f = 2 + 2 (xi - 1)    for xi > 1     (deep positive stall)
        f = 5 - 4 (xi + 2)    for xi < -2    (deep negative stall / choke)

    ``f(0) = 1`` so the design-incidence loss is unchanged; the branches meet
    C1 at ``xi = 1, -2`` by construction (Aungier), so the smooth blends below
    are C1 and near-exact. The asymmetric reciprocal half-width is itself
    C1-blended across ``i_ref`` -- and since ``xi = 0`` there, ``f`` is C1
    through it regardless of ``R_s != R_c``."""
    xp = get_xp(xp)
    di = i - i_ref
    inv_w = blend_between(di, 1.0 / r_c, 1.0 / r_s, 0.0, _DI_W, xp=xp)
    xi = di * inv_w
    core = 1.0 + xi * xi
    pos = 2.0 + 2.0 * (xi - 1.0)                # xi > 1 linear branch
    neg = 5.0 - 4.0 * (xi + 2.0)                # xi < -2 linear branch
    upper = blend_between(xi, core, pos, _XI_POS, _XI_W, xp=xp)
    return blend_between(xi, neg, upper, _XI_NEG, _XI_W, xp=xp)


@dataclass(frozen=True)
class LieblienLoss:
    """LossModel (section 7.1): reference profile loss from the
    equivalent-diffusion chain, Aungier off-design incidence bucket (physical
    stall/choke ranges), converted to entropy per B.2 at the B.1-re-referenced
    exit state.

    Requires ``row.geometry`` (section 4.1 contract) and the view's lagged
    TE fields."""

    k_sh: float = 1.0

    def evaluate(self, row: RowView, flow: RowFlowView) -> LossBreakdown:
        xp = get_xp(None)
        g = row.geometry
        y = flow.psi
        sgn = g.orientation              # geometry constant (ARCH-4.2)

        # Cascade-frame angles [deg] and the shared deviation chain.
        b1_blade = xp.rad2deg(sgn * g.beta1_blade(y))
        b2_blade = xp.rad2deg(sgn * g.beta2_blade(y))
        camber = b1_blade - b2_blade
        sigma = g.solidity(y)
        tc = g.thickness_ratio(y)
        b1_flow = xp.rad2deg(sgn * flow.beta)
        i = b1_flow - b1_blade
        i_ref, v_i = reference_incidence(b1_flow, sigma, tc, camber, xp=xp)
        d_ref, _ = reference_deviation(b1_flow, sigma, tc, camber, xp=xp)

        # Actual inlet relative angle -> W1 for the B.2 conversion reference.
        b1r = xp.deg2rad(soft_clip(b1_flow, -80.0, 80.0, 2.0, xp=xp))
        w1 = flow.vm / xp.cos(b1r)

        # REFERENCE (min-loss) velocity triangle -> omega_min. Aungier: the
        # off-design incidence bucket is the SOLE off-design mechanism, so the
        # minimum-loss coefficient is evaluated at the reference incidence
        # (b1_blade + i_ref, exit b2_blade + d_ref), not the actual off-design
        # triangle -- otherwise D_eq and the bucket double-count incidence.
        b1_ref = xp.deg2rad(soft_clip(b1_blade + i_ref, -80.0, 80.0, 2.0, xp=xp))
        b2_ref = xp.deg2rad(soft_clip(b2_blade + d_ref, -80.0, 80.0, 2.0, xp=xp))
        w1_ref = flow.vm / xp.cos(b1_ref)
        w2_ref = flow.vm_te / xp.cos(b2_ref)
        d_eq = equivalent_diffusion(w1_ref, w2_ref, b1_ref, b2_ref, sigma, xp=xp)
        theta_c, v_d = wake_momentum_thickness(d_eq, xp=xp)
        omega_min = profile_loss_coefficient(theta_c, sigma, b2_ref, w1_ref,
                                             w2_ref, xp=xp)
        # Aungier off-design bucket: physically-derived asymmetric stall/choke
        # ranges from the reference inlet flow angle, replacing the old fixed
        # 10-deg width (section 4.3; docs/references/LIEB59.md).
        r_s, r_c = stall_choke_ranges(camber, b1_blade + i_ref, xp=xp)
        omega_raw = omega_min * off_design_bucket(i, i_ref, r_s, r_c, xp=xp)
        # Section 7.3.2 conservative asymptote: level off smoothly well
        # inside the B.2 conversion's domain; validity reflects binding.
        omega_bar = smooth_min(omega_raw, _WBAR_CEIL, _WBAR_W, xp=xp)
        v_w = 1.0 - blend(omega_raw, _WBAR_CEIL, 10.0 * _WBAR_W, xp=xp)

        # Section 4.4 / B.2: convert at row exit with B.1 re-referencing.
        fluid = flow.fluid
        p1 = fluid.p(flow.h, flow.s)
        T0r_1, p0r_1 = relative_stagnation(fluid, flow.T, p1, w1, xp=xp)
        u1 = row.omega * flow.r
        u2 = row.omega * flow.r_te
        delta_s, _ = delta_s_compressor_omega_bar(
            fluid, omega_bar, T0r_1, p0r_1, p1, u1, u2, xp=xp)
        return LossBreakdown(components={"profile_omega_bar": omega_bar},
                             delta_s=delta_s,
                             validity=float(xp.min(v_i * v_d * v_w)))
