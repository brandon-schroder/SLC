"""Lieblein diffusion-factor profile loss for axial-compressor rows
(Theory Manual sections 4.3, 4.4, 7.1, 7.3; Appendix B.2 conversion).

Provenance: Lieblein's 1959 equivalent-diffusion-ratio correlation chain as
reproduced in the standard texts (Cumpsty; Aungier; Dixon):

    D_eq   = (W1/W2) [1.12 + 0.61 (cos^2 beta1 / sigma)(tan beta1 - tan beta2)]
    theta/c = 0.004 / (1 - 1.17 ln D_eq)
    omega_bar_min = 2 (theta/c) (sigma / cos beta2) (W2/W1)^2   # see [BUG]

Verification status (docs/references/LIEB59.md, 2026-07-09, vs Aungier ch.6 /
Cumpsty / Dixon): D_eq (1.12, 0.61), theta/c (0.004, 1.17), and the inlet-
relative reference dynamic head are CONFIRMED (pinned in
tests/test_lieblein_loss_reference.py).

**[BUG -- confirmed, fix DEFERRED to the consolidated resolution pass.]** The
omega_bar velocity-ratio factor is INVERTED. Aungier Eq 6-27 and Cumpsty Eq
1.32 both give ``omega_bar = 2 (theta/c)(sigma/cos beta2)(W2/W1)^2``, but the
code below (``:evaluate``) uses ``(W1/W2)^2``. For a compressor W2 < W1 so this
overestimates profile loss by ~(W1/W2)^4 (~4x at DF~0.45). Not fixed in
isolation because it shifts every V4/V5 result and may interact with the
M4-tuned _WBAR_CEIL and 10-deg bucket; needs a paired loss-calibration re-look
+ V5 re-measurement.

**[DECIDE] off-design model:** the quadratic bucket (``w_bucket``, 10 deg
default) is our substitution; Lieblein's published off-design extends D_eq by
``+k(i-i*)^1.43`` (k=0.0117 NACA-65; Aungier 6-38 / Dixon 3.41). The theta/c
fit's validity ends near D_eq ~ 2.35 (denominator zero); saturation reflects
that.

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
from ..smoothmath import blend, smooth_min, soft_clip, softplus
from .lieblein import (deviation_slope, reference_deviation,
                       reference_incidence)

__all__ = ["equivalent_diffusion", "wake_momentum_thickness",
           "LieblienLoss"]

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


@dataclass(frozen=True)
class LieblienLoss:
    """LossModel (section 7.1): reference profile loss from the
    equivalent-diffusion chain, quadratic off-design bucket, converted to
    entropy per B.2 at the B.1-re-referenced exit state.

    Requires ``row.geometry`` (section 4.1 contract) and the view's lagged
    TE fields. ``bucket_width_deg`` is the incidence half-width at which
    the loss doubles [VERIFY]."""

    k_sh: float = 1.0
    bucket_width_deg: float = 10.0

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
        dev = d_ref + deviation_slope(b1_flow, sigma, xp=xp) * (i - i_ref)
        b2_deg = soft_clip(b2_blade + dev, -80.0, 80.0, 2.0, xp=xp)

        # Velocity triangle magnitudes (relative frame): W = Vm / cos(beta).
        b1r = xp.deg2rad(soft_clip(b1_flow, -80.0, 80.0, 2.0, xp=xp))
        b2r = xp.deg2rad(b2_deg)
        w1 = flow.vm / xp.cos(b1r)
        w2 = flow.vm_te / xp.cos(b2r)

        d_eq = equivalent_diffusion(w1, w2, b1r, b2r, sigma, xp=xp)
        theta_c, v_d = wake_momentum_thickness(d_eq, xp=xp)
        # [BUG] velocity ratio is INVERTED: Aungier 6-27 / Cumpsty 1.32 give
        # (W2/W1)^2, not (W1/W2)^2 -- ~4x loss overestimate for a compressor.
        # Fix deferred to the consolidated resolution pass (docs/references/
        # LIEB59.md): touches every V4/V5 result + the M4 loss calibration.
        omega_min = (2.0 * theta_c * sigma / xp.cos(b2r)
                     * (w1 / w2) ** 2)
        omega_raw = omega_min * (1.0 + ((i - i_ref)
                                        / self.bucket_width_deg) ** 2)
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
