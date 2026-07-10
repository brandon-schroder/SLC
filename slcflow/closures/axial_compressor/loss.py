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

**Endwall + clearance loss (2026-07, docs/references/HOWELL.md).** The profile
loss above is only ~half of a real stage's loss; the Howell endwall model
(secondary drag ``C_Ds = 0.018 C_L^2`` + annulus drag ``C_Da = 0.020 s/h``)
and the Lakshminarayana tip-clearance drag (``C_Dk = 0.7 C_L^2 t/h``) are
added as inlet-referenced ``omega_bar`` (Cumpsty 4.9 drag->loss conversion),
so a single B.2 conversion covers profile + endwall + clearance. Aungier's own
endwall method (K1/K2 factors folding into the profile correlation, Eq 6-46 +
charts Fig 6-11/6-12) was *not* clean-additive, so Howell's additive
drag-coefficient form was used instead (§7.1 permits Koch-Smith **or** Aungier
-- Howell is the closed-form, library-verifiable choice).

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
           "profile_loss_coefficient", "blade_loading_coefficient",
           "endwall_clearance_loss", "normal_shock_pt_ratio", "shock_loss",
           "stall_choke_ranges", "off_design_bucket", "LieblienLoss"]

# theta/c fit domain: the denominator 1 - 1.17 ln(D_eq) vanishes at
# D_eq = e^(1/1.17) ~ 2.35 -- Lieblein's stated "limit V_max,s/V2 = 2.35"
# (1959 paper p.5); saturate D_eq below that with a smooth ceiling. The
# calibration window [1.0, 2.0] and the theta/c curve were digitized-verified
# against Lieblein Fig. 6 (docs/references/LIEB59.md, 2026-07-10): the data
# span DR ~ 1.15-2.25, so [1.0, 2.0] is sound (conservative on the upper end);
# the ceiling 2.2 sits at the data edge, safely below the 2.35 divergence.
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

# Howell/Dixon endwall (secondary + annulus) drag + Lakshminarayana tip
# clearance (docs/references/HOWELL.md). Additive drag coefficients converted
# to the inlet-referenced loss omega_bar (Cumpsty 4.9), summed with the
# Lieblein profile omega_bar under one B.2 conversion (section 4.4):
#   tan(b_m) = (tan b1 + tan b2)/2                       (Dixon 3.15)
#   C_L      = (2/sigma) cos(b_m)(tan b1 - tan b2)       (Dixon 3.26a; -C_D
#              tan b_m dropped as negligible)
#   C_Ds     = 0.018 C_L^2            (secondary / trailing vortex, Howell p.451)
#   C_Da     = 0.020 (s/h) = 0.020/(sigma*AR)   (annulus friction, Howell p.451)
#   C_Dk     = 0.7 C_L^2 (t/h)        (tip clearance, Lakshminarayana via Cumpsty)
#   omega_ew = sigma (cos^2 b1 / cos^3 b_m)(C_Ds + C_Da + C_Dk)   (zeta from
#              C_D = zeta (s/l)(cos^3 b_m/cos^2 b1), inverted)
_CDS_C = 0.018       # Howell secondary constant
_CDA_C = 0.020       # Howell annulus constant
_CDK_C = 0.7         # Lakshminarayana tip-clearance constant
_B1_CLIP = 80.0      # inlet-angle soft clip [deg] (AD-10 / C1, as profile)
_CL_CEIL, _CL_W = 1.6, 0.15   # loading validity ceiling (compact support)

# Aungier (2003) section 6.7 transonic shock loss: a normal shock at the
# geometric-mean Mach M_shock = sqrt(M1 * M_ss) (Eq 6-71), with the suction-
# surface Mach M_ss = M1 (W_max,s/W1) and the surface velocity ratio W_max,s/W1
# taken as the equivalent-diffusion bracket (Aungier's own estimate of W_max,s
# = D_eq * W2; used in lieu of the 6-69/70 Prandtl-Meyer surface-curvature
# expansion, whose suction-surface radius of curvature is not in the section-4.1
# contract). Normal-shock Pt loss referenced to inlet dynamic head (Eq 2-68),
# added to the profile+endwall omega_bar (one B.2 conversion).
_MSH_W = 0.08        # C1 onset width on (M_shock - 1) [VERIFY]
_DEN_FLOOR, _DEN_FW = 0.05, 0.02   # inlet dynamic-head denominator floor (AD-10)
_MSH_CEIL, _MSH_VW = 1.7, 0.15     # validity fade far supersonic [VERIFY]


def equivalent_diffusion(w1, w2, beta1_rad, beta2_rad, sigma, *, xp=None):
    """Lieblein equivalent diffusion ratio ``D_eq`` (cascade frame,
    positive angles; [VERIFY coefficients])."""
    xp = get_xp(xp)
    turning = xp.tan(beta1_rad) - xp.tan(beta2_rad)
    return (w1 / w2) * (1.12 + 0.61 * xp.cos(beta1_rad) ** 2 / sigma
                        * turning)


def wake_momentum_thickness(d_eq, *, xp=None):
    """``theta*/c`` from ``D_eq`` (Lieblein 1959 Eq. 8, digitized-verified
    against the paper's Fig. 6 -- docs/references/LIEB59.md).

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


def blade_loading_coefficient(beta1_rad, beta2_rad, sigma, *, xp=None):
    """Howell tangential lift/loading coefficient ``C_L`` and the mean vector
    angle ``beta_m`` (Dixon 3.15 / 3.26a, Saravanamuttoo 5.32/5.33):

        tan(beta_m) = (tan beta1 + tan beta2)/2
        C_L = (2/sigma) cos(beta_m)(tan beta1 - tan beta2)

    The published ``- C_D tan(beta_m)`` term is dropped (negligibly small in
    the normal range -- the standard "theoretical" C_L). Cascade-frame
    relative angles [rad]; ``sigma`` = solidity = chord/pitch = l/s. Returns
    ``(C_L, beta_m)``. ``beta_m = arctan(...)`` is in (-pi/2, pi/2) so
    ``cos(beta_m) > 0`` always (the downstream ``cos^3 beta_m`` never
    vanishes)."""
    xp = get_xp(xp)
    b1 = xp.deg2rad(soft_clip(xp.rad2deg(beta1_rad), -_B1_CLIP, _B1_CLIP,
                              2.0, xp=xp))
    b2 = xp.deg2rad(soft_clip(xp.rad2deg(beta2_rad), -_B1_CLIP, _B1_CLIP,
                              2.0, xp=xp))
    beta_m = xp.arctan(0.5 * (xp.tan(b1) + xp.tan(b2)))
    c_l = 2.0 / sigma * xp.cos(beta_m) * (xp.tan(b1) - xp.tan(b2))
    return c_l, beta_m


def endwall_clearance_loss(beta1_rad, beta2_rad, sigma, aspect_ratio,
                           clearance_ratio, *, xp=None):
    """Howell secondary + annulus drag + Lakshminarayana tip-clearance drag,
    summed and converted to the inlet-referenced loss coefficient
    ``omega_bar`` (docs/references/HOWELL.md):

        C_Ds = 0.018 C_L^2            (secondary / trailing vortex)
        C_Da = 0.020 (s/h) = 0.020/(sigma*AR)   (annulus endwall friction)
        C_Dk = 0.7 C_L^2 (t/h)       (tip clearance; 0 when clearance = 0)
        omega_ew = sigma (cos^2 beta1 / cos^3 beta_m)(C_Ds + C_Da + C_Dk)

    where ``s/h = 1/(sigma*AR)`` (AR = blade height/chord) and ``t/h =
    clearance_ratio``. The conversion is the inverse of Cumpsty 4.9
    ``C_D = zeta (s/l)(cos^3 beta_m/cos^2 beta1)`` -- all inlet-referenced, so
    ``omega_ew`` ADDS directly to the Lieblein profile ``omega_bar`` under one
    B.2 conversion (section 4.4). Returns ``(omega_ew, validity)`` with a
    compact-support ceiling on the loading ``C_L`` (Howell's data is moderate-
    loading). Evaluated at the reference (design) triangle in
    :class:`LieblienLoss`; off-design growth of the secondary loss (``C_L`` at
    the actual triangle) is a recorded refinement."""
    xp = get_xp(xp)
    c_l, beta_m = blade_loading_coefficient(beta1_rad, beta2_rad, sigma, xp=xp)
    b1 = xp.deg2rad(soft_clip(xp.rad2deg(beta1_rad), -_B1_CLIP, _B1_CLIP,
                              2.0, xp=xp))
    cds = _CDS_C * c_l * c_l
    cda = _CDA_C / (sigma * aspect_ratio)
    cdk = _CDK_C * c_l * c_l * clearance_ratio
    c_d = cds + cda + cdk
    omega_ew = sigma * xp.cos(b1) ** 2 / xp.cos(beta_m) ** 3 * c_d
    v = 1.0 - blend(c_l, _CL_CEIL, _CL_W, xp=xp)
    return omega_ew, v


def normal_shock_pt_ratio(mach, gamma, *, xp=None):
    """Normal-shock stagnation-pressure ratio ``p02/p01`` at upstream Mach
    ``mach >= 1`` (perfect gas -- the Rayleigh supersonic-pitot relation):

        p02/p01 = [ (g+1)M^2 / ((g-1)M^2 + 2) ]^(g/(g-1))
                * [ (g+1) / (2 g M^2 - (g-1)) ]^(1/(g-1))

    Unity at M=1 and falling as ~(M-1)^3 for weak shocks (Cumpsty), so C1 at
    onset. Aungier 6.7 uses a real-gas conservation solve (Eq 6-72..6-74); this
    is the perfect-gas closed form, exact for :class:`PerfectGas`. Callers pass
    ``mach >= 1`` (the denominator ``2 g M^2 - (g-1) >= g+1 > 0``)."""
    xp = get_xp(xp)
    g = gamma
    m2 = mach * mach
    t1 = ((g + 1.0) * m2 / ((g - 1.0) * m2 + 2.0)) ** (g / (g - 1.0))
    t2 = ((g + 1.0) / (2.0 * g * m2 - (g - 1.0))) ** (1.0 / (g - 1.0))
    return t1 * t2


def shock_loss(m1, surface_velocity_ratio, gamma, *, xp=None):
    """Aungier (2003) section 6.7 transonic shock-loss coefficient
    ``omega_shock`` (inlet-relative reference; docs/references/AUN-C.md):

        M_ss     = M1 (W_max,s / W1)                      (surface Mach)
        M_shock  = sqrt(M1 * M_ss)                        (Aungier 6-71)
        omega_shock = (1 - p02/p01|M_shock) / (1 - p1/p01|M1)   (Aungier 2-68)

    ``surface_velocity_ratio`` = ``W_max,s/W1`` is the equivalent-diffusion
    bracket (Aungier's own estimate of ``W_max,s = D_eq W2``), used in lieu of
    the 6-69/70 Prandtl-Meyer surface-curvature expansion the section-4.1
    contract lacks the geometry for (recorded ``[VERIFY]``). Because
    ``M_ss = M1 * ratio``, ``M_shock = M1 sqrt(ratio) > M1`` -- the shock can
    turn on while the inlet itself is subsonic (Aungier's supercritical regime).
    C1 at the ``M_shock = 1`` onset via softplus; the loss is 0 (to reading
    precision) well below onset, so subsonic rows are unaffected. Returns
    ``(omega_shock, validity)`` referenced to the inlet dynamic head, so it ADDS
    to the profile+endwall ``omega_bar`` under one B.2 conversion (section 4.4)."""
    xp = get_xp(xp)
    m_ss = m1 * surface_velocity_ratio
    m_shock = xp.sqrt(m1 * m_ss)                        # geometric mean (6-71)
    m_eff = 1.0 + softplus(m_shock - 1.0, _MSH_W, xp=xp)   # C1 onset, >= 1
    pr = normal_shock_pt_ratio(m_eff, gamma, xp=xp)
    g = gamma
    p_pt1 = (1.0 + 0.5 * (g - 1.0) * m1 * m1) ** (-g / (g - 1.0))
    denom = _DEN_FLOOR + softplus((1.0 - p_pt1) - _DEN_FLOOR, _DEN_FW, xp=xp)
    omega = (1.0 - pr) / denom
    v = 1.0 - blend(m_shock, _MSH_CEIL, _MSH_VW, xp=xp)
    return omega, v


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
    equivalent-diffusion chain (Aungier off-design incidence bucket, physical
    stall/choke ranges) PLUS Howell endwall (secondary + annulus) and
    Lakshminarayana tip-clearance loss PLUS the Aungier section-6.7 transonic
    shock loss, summed as one inlet-referenced ``omega_bar`` and converted to
    entropy per B.2 at the B.1-re-referenced exit state. The shock term is 0
    (to reading precision) for subsonic rows, so it only engages transonic
    cases -- all current subsonic V5 cases are unaffected.

    ``aspect_ratio`` (blade height/chord) is a row-scalar design input for the
    endwall/clearance drag (``s/h`` and ``t/h``; deriving it from the annulus
    span is a future refinement, as for the turbine set). Tip clearance comes
    from the geometry contract (``tip_clearance()``); it is 0 by default, so
    the clearance term is inert unless a clearance is set -- existing
    zero-clearance cases see only the secondary + annulus endwall loss.

    Requires ``row.geometry`` (section 4.1 contract) and the view's lagged
    TE fields."""

    k_sh: float = 1.0
    aspect_ratio: float = 2.5

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
        omega_profile = omega_min * off_design_bucket(i, i_ref, r_s, r_c, xp=xp)
        # Howell endwall (secondary + annulus) + Lakshminarayana tip-clearance
        # loss (section 4.4; docs/references/HOWELL.md). Evaluated at the
        # reference (design) loading and ADDED to the bucketed profile loss --
        # all inlet-referenced omega_bar, so one B.2 conversion covers the sum.
        # t/h from the geometry clearance (0 by default -> clearance term inert).
        chord = g.chord(y)
        clearance_ratio = g.tip_clearance() / (self.aspect_ratio * chord)
        omega_ew, v_e = endwall_clearance_loss(
            b1_ref, b2_ref, sigma, self.aspect_ratio, clearance_ratio, xp=xp)
        # Aungier section 6.7 transonic shock loss (docs/references/AUN-C.md):
        # normal shock at the geometric-mean Mach, using the ACTUAL inlet
        # relative Mach and the equivalent-diffusion bracket W_max,s/W1 as the
        # suction-surface velocity ratio. Zero (to reading precision) for
        # subsonic rows, so all existing V5 cases are unaffected.
        m1_rel = w1 / flow.a
        surf_ratio = d_eq * w2_ref / w1_ref            # W_max,s/W1 (D_eq bracket)
        omega_shock, v_sh = shock_loss(m1_rel, surf_ratio, flow.fluid.gamma,
                                       xp=xp)
        omega_raw = omega_profile + omega_ew + omega_shock
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
        # Components recorded individually for auditability (B.5.3).
        return LossBreakdown(
            components={"profile_omega_bar": omega_profile,
                        "endwall_omega_bar": omega_ew,
                        "shock_omega_bar": omega_shock,
                        "omega_bar_total": omega_bar},
            delta_s=delta_s,
            validity=float(xp.min(v_i * v_d * v_w * v_e * v_sh)))
