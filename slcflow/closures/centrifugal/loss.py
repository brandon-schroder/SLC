"""Centrifugal-impeller internal loss model (Theory Manual sections 4.3,
4.4, 7.1, 7.3; Appendix B enthalpy-loss conversion).

Provenance: a representative subset of the Aungier / Galvas / Oh centrifugal
meanline loss set -- the three internal components that dominate a well-designed
impeller across its range:

  * **incidence loss** (inducer): the tangential kinetic energy lost when the
    inlet relative flow does not enter blade-congruent,
    ``dh_inc = 1/2 (W_theta1 - W_theta1,blade)^2``;
  * **skin-friction / passage loss**: the pipe-friction analogy over the mean
    relative velocity, ``dh_sf = 2 Cf (L/D_hyd) W_avg^2`` with
    ``W_avg = 1/2 (W1 + W2)``;
  * **blade-loading (diffusion) loss** (Coppage/Aungier 5.15):
    ``dh_bl = 0.05 D_f^2 U2^2`` with the radial diffusion factor ``D_f`` --
    the secondary-flow loss driven by the blade-to-blade pressure gradient,
    added 2026-07 (was deferred). See :func:`blade_loading_loss`.

Each internal loss is an *enthalpy* rise, converted to entropy INDIVIDUALLY
(B.5) at the exit static temperature via ``delta_s_enthalpy_loss``. Exit-state
quantities (the relative exit velocity ``W2`` and ``T2``) are built from the
shared Wiesner slip (same CorrelationSet) and B.1 rothalpy re-referencing
across the radius change -- no addition to the flow-view contract.

Verification status (docs/references/CENT-LOSS.md, 2026-07-09, vs Galvas
NASA TN D-7487 / Aungier / Conrad / Braembussche): both base forms CONFIRMED --
the incidence ``1/2 (dW_theta)^2`` is Galvas Eq 5.6 (W_x sin(dbeta) = dW_theta),
the skin-friction leading ``2 Cf`` is Galvas ``4 Cf W^2/2``, and Cf=0.005 is
Braembussche's typical wall value. Pinned in test_centrifugal_loss_reference.py.

**Two conventions resolved 2026-07 (both to Aungier 2000; CENT-LOSS.md):**
(a) incidence loss now applies ``f_inc = 0.8`` (Aungier) to the full NASA KE --
a genuinely design-dependent 0.5-1.0 factor (Conrad 0.5-0.7 / Aungier 0.8 /
Galvas full-KE 1.0), exposed as a tunable ``CentrifugalLoss.f_inc`` field.
(b) skin friction uses Aungier's mean-of-squares passage velocity
``1/2(W1^2+W2^2)`` (was the square-of-mean ``[1/2(W1+W2)]^2``) -- the physical
passage average since friction ~ local W^2.

**Blade-loading added 2026-07** (Coppage/Aungier 5.15; Oh-Yoon-Chung 1997), the
dominant remaining internal component. Still deferred and **[VERIFY]** -- a
per-streamtube closure does not cleanly see the inputs they need: **tip-
clearance** (Jansen 1967 needs the exit blade width ``b2`` and hub/tip radii,
absent from the section 4.1 contract) and **disk-friction/windage** (a
machine-level parasitic ``~ rho2 U2^3 r2^2 / mdot`` needs ``mdot``, which a
per-streamtube loss model is not given); recirculation/leakage likewise.
``Cf``/``l_over_dhyd`` are row-scalar design inputs (geometry-derived hydraulic
length is a later refinement).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._namespace import get_xp
from ..conversions import delta_s_enthalpy_loss
from ..interfaces import LossBreakdown, RowFlowView, RowView
from ..smoothmath import smooth_min, soft_clip, softplus
from .wiesner import wiesner_slip

__all__ = ["incidence_loss", "skin_friction_loss", "blade_loading_loss",
           "CentrifugalLoss"]

_PI = 3.141592653589793
_DEG = _PI / 180.0
_ANG_CAP, _ANG_W = 85.0 * _DEG, 2.0 * _DEG   # metal-angle soft-clip (tan bound)
_T_FLOOR, _T_W = 20.0, 5.0                   # exit static-T floor (a2 real)
_BL_C = 0.05                                 # Coppage blade-loading constant
_BL_LOAD = 0.75                              # Coppage loading-term constant
_V_FLOOR, _V_W = 1.0, 1.0                    # C1 velocity floor (m/s) for ratios
_BL_DF_CEIL, _BL_DF_W = 2.5, 0.2             # D_f ceiling (stalled/out-of-range)


def incidence_loss(w_theta_flow, w_theta_blade, *, xp=None):
    """Inducer incidence loss KINETIC ENERGY ``1/2 (W_theta_flow -
    W_theta_blade)^2`` [J/kg] (section 4.3; Galvas/NASA Eq 5.6, CONFIRMED --
    CENT-LOSS.md). This is the FULL tangential-KE form; the fraction actually
    lost (``f_inc``, a 0.5-1.0 family) is applied by :class:`CentrifugalLoss`."""
    xp = get_xp(xp)
    d = w_theta_flow - w_theta_blade
    return 0.5 * d * d


def blade_loading_loss(w1, w2, u2, dh_euler, blade_count, r_ratio, *, xp=None):
    """Coppage/Jansen blade-loading (diffusion) loss ``dh = 0.05 D_f^2 U2^2``
    [J/kg] (Aungier 2000 Eq 5.15; Oh-Yoon-Chung 1997 optimum set;
    docs/references/CENT-LOSS.md). The loss of the secondary flows driven by
    the blade-to-blade pressure gradient, scaled by an equivalent radial
    *diffusion factor*

        D_f = 1 - W2/W1 + 0.75 (dh_euler/U2^2) (W1/W2)
                          / [ (Z/pi)(1 - r1/r2) + 2 (r1/r2) ]

    with ``r_ratio = r1/r2`` and ``dh_euler`` the Euler work
    ``U2 Vtheta2 - U1 Vtheta1`` [J/kg]. The loading term uses ``W1/W2`` (> 1
    under diffusion) so the loss GROWS with diffusion (W2 << W1) -- the
    physically-required direction, and the Oh-Yoon-Chung / Galvas consensus
    (the MathML source render is ambiguous on this fraction; pinned by
    ``test_blade_loading_grows_with_diffusion``). ``W1`` is the local
    streamtube inlet relative velocity (the Coppage shroud value ``W1s`` at the
    meanline; a spanwise run supplies the per-streamtube value).

    C1 in every flow input (section 7.3): ``W2`` and ``U2`` enter a division and
    are softplus-floored; the geometric bracket is a positive constant
    (``Z >= 1``, ``r1 < r2``). ``D_f`` may dip slightly negative at very low
    loading -- harmless, since only ``D_f^2`` enters."""
    xp = get_xp(xp)
    w1f = _V_FLOOR + softplus(w1 - _V_FLOOR, _V_W, xp=xp)
    w2f = _V_FLOOR + softplus(w2 - _V_FLOOR, _V_W, xp=xp)
    u2f = _V_FLOOR + softplus(u2 - _V_FLOOR, _V_W, xp=xp)
    dq = dh_euler / (u2f * u2f)                       # dimensionless Euler work
    geom = (blade_count / _PI) * (1.0 - r_ratio) + 2.0 * r_ratio
    d_f = 1.0 - w2 / w1f + _BL_LOAD * dq * (w1 / w2f) / geom
    # Smoothly cap D_f at a stalled/out-of-correlation ceiling (Coppage is a
    # design-range fit; real impeller D_f ~ 0.4-2). This leaves the design
    # point untouched (V7 D_f ~ 1.1) and bounds the transient blow-up when a
    # lagged W2 goes small mid-solve -- the same role the axial omega-bar
    # ceiling plays (section 7.3.2). Only the upper side is capped; low/negative
    # D_f is harmless since only D_f^2 enters.
    d_f = smooth_min(d_f, _BL_DF_CEIL, _BL_DF_W, xp=xp)
    return _BL_C * d_f * d_f * u2 * u2


def skin_friction_loss(w_rep, cf, l_over_dhyd, *, xp=None):
    """Passage skin-friction loss ``dh = 2 Cf (L/D_hyd) W_rep^2`` [J/kg]
    (section 4.3; Galvas ``4 Cf W^2/2`` leading factor CONFIRMED -- CENT-LOSS.md).

    ``W_rep`` is the representative passage velocity the caller squares in.
    :class:`CentrifugalLoss` passes the RMS ``sqrt(1/2 (W1^2 + W2^2))`` so the
    squared term is Aungier's mean-of-squares ``1/2 (W1^2 + W2^2)`` (resolved
    2026-07 from the earlier square-of-mean ``[1/2(W1+W2)]^2``; the mean of the
    squares is the physical passage average since friction ~ local W^2)."""
    xp = get_xp(xp)
    return 2.0 * cf * l_over_dhyd * w_rep * w_rep


@dataclass(frozen=True)
class CentrifugalLoss:
    """LossModel (section 7.1): representative centrifugal internal loss
    (incidence + skin friction), each converted to entropy per the Appendix-B
    enthalpy-loss form at the B.1-re-referenced exit static state.

    Requires ``row.geometry`` (section 4.1 contract) and the view's lagged TE
    fields. ``cf``/``l_over_dhyd`` are row-scalar design inputs.

    ``f_inc`` is the fraction of the inducer incidence kinetic energy actually
    lost (resolved 2026-07; CENT-LOSS.md). It is genuinely design-dependent
    with no single source value -- Conrad et al. (1980) fit 0.5-0.7, Aungier
    (2000) uses 0.8, and the raw Galvas/NASA shock-loss form is the full KE
    (1.0). Default 0.8 adopts Aungier (coherent with the mean-of-squares
    skin-friction convention); tune per design. The skin-friction loss uses
    Aungier's mean-of-squares passage velocity (see :func:`skin_friction_loss`).
    """

    cf: float = 0.005
    l_over_dhyd: float = 4.0
    f_inc: float = 0.8

    def evaluate(self, row: RowView, flow: RowFlowView) -> LossBreakdown:
        xp = get_xp(None)
        g = row.geometry
        y = flow.psi
        sgn = g.orientation                  # geometry constant (ARCH-4.2)
        fluid = flow.fluid

        # Inducer incidence loss (LE): blade-congruent vs actual relative
        # tangential velocity (both in the section 2.4 relative convention).
        b1 = soft_clip(g.beta1_blade(y), -_ANG_CAP, _ANG_CAP, _ANG_W, xp=xp)
        w_theta_blade = flow.vm * xp.tan(b1)
        dh_inc = self.f_inc * incidence_loss(flow.w_theta, w_theta_blade,
                                             xp=xp)

        # Exit relative velocity via the shared Wiesner slip (same set):
        # W_theta2 = (sigma - 1) U2 - Vm2 tan(beta2b).
        b2b = soft_clip(sgn * g.beta2_blade(y), -_ANG_CAP, _ANG_CAP, _ANG_W,
                        xp=xp)
        # r1/r2 drives the same radius-ratio limit correction as the swirl
        # closure (shared Wiesner slip, consistent exit velocity).
        sigma, v_s = wiesner_slip(b2b, g.blade_count, flow.r / flow.r_te, xp=xp)
        u1 = row.omega * flow.r
        u2 = row.omega * flow.r_te
        w_theta_2 = (sigma - 1.0) * u2 - flow.vm_te * xp.tan(b2b)
        w1 = xp.sqrt(flow.vm * flow.vm + flow.w_theta * flow.w_theta)
        w2 = xp.sqrt(flow.vm_te * flow.vm_te + w_theta_2 * w_theta_2)
        # Aungier mean-of-squares passage velocity: pass the RMS so the
        # skin-friction term squares to 1/2 (W1^2 + W2^2).
        w_rms = xp.sqrt(0.5 * (w1 * w1 + w2 * w2))
        dh_sf = skin_friction_loss(w_rms, self.cf, self.l_over_dhyd, xp=xp)

        # Blade-loading (diffusion) loss (Coppage/Aungier 5.15): the Euler work
        # is U2 Vtheta2 - U1 Vtheta1 with Vtheta2 = U2 + Wtheta2 (relative
        # convention, section 2.4) and Vtheta1 the LE absolute swirl.
        vtheta_2 = u2 + w_theta_2
        dh_euler = u2 * vtheta_2 - u1 * flow.vtheta
        dh_bl = blade_loading_loss(w1, w2, u2, dh_euler, g.blade_count,
                                   flow.r / flow.r_te, xp=xp)

        # Exit static temperature via B.1 rothalpy re-referencing (charging
        # temperature for the entropy conversion).
        T0r_1 = flow.T + 0.5 * w1 * w1 / fluid.cp
        T0r_2 = T0r_1 + 0.5 * (u2 * u2 - u1 * u1) / fluid.cp
        T2 = _T_FLOOR + softplus(T0r_2 - 0.5 * w2 * w2 / fluid.cp - _T_FLOOR,
                                 _T_W, xp=xp)

        ds_inc = delta_s_enthalpy_loss(fluid, dh_inc, T2, xp=xp)
        ds_sf = delta_s_enthalpy_loss(fluid, dh_sf, T2, xp=xp)
        ds_bl = delta_s_enthalpy_loss(fluid, dh_bl, T2, xp=xp)
        return LossBreakdown(
            components={"incidence_dh": dh_inc, "friction_dh": dh_sf,
                        "blade_loading_dh": dh_bl},
            delta_s=ds_inc + ds_sf + ds_bl, validity=float(xp.min(v_s)))
