"""Centrifugal-impeller internal loss model (Theory Manual sections 4.3,
4.4, 7.1, 7.3; Appendix B enthalpy-loss conversion).

Provenance: a representative subset of the Aungier / Galvas / Oh centrifugal
meanline loss set -- the two components that dominate a well-designed
impeller across its range:

  * **incidence loss** (inducer): the tangential kinetic energy lost when the
    inlet relative flow does not enter blade-congruent,
    ``dh_inc = 1/2 (W_theta1 - W_theta1,blade)^2``;
  * **skin-friction / passage loss**: the pipe-friction analogy over the mean
    relative velocity, ``dh_sf = 2 Cf (L/D_hyd) W_avg^2`` with
    ``W_avg = 1/2 (W1 + W2)``.

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

**[DECIDE] two modeling choices (documented, not changed):** (a) incidence
uses the FULL NASA KE (f_inc=1); Conrad applies 0.5-0.7 and Aungier 0.8 -- the
coded value is the conservative upper bound. (b) skin friction squares the mean
velocity, ``[1/2(W1+W2)]^2``; Aungier specifies the mean of the squares,
``1/2(W1^2+W2^2)``. **[VERIFY]** the deferred components (blade-loading
diffusion, tip-clearance, disk-friction/windage, recirculation, leakage; Oh-
Yoon-Chung 1997 set) at V7 calibration time. ``Cf``/``l_over_dhyd`` are
row-scalar design inputs (geometry-derived hydraulic length is a later
refinement).
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._namespace import get_xp
from ..conversions import delta_s_enthalpy_loss
from ..interfaces import LossBreakdown, RowFlowView, RowView
from ..smoothmath import soft_clip, softplus
from .wiesner import wiesner_slip

__all__ = ["incidence_loss", "skin_friction_loss", "CentrifugalLoss"]

_DEG = 3.141592653589793 / 180.0
_ANG_CAP, _ANG_W = 85.0 * _DEG, 2.0 * _DEG   # metal-angle soft-clip (tan bound)
_T_FLOOR, _T_W = 20.0, 5.0                   # exit static-T floor (a2 real)


def incidence_loss(w_theta_flow, w_theta_blade, *, xp=None):
    """Inducer incidence loss ``dh = 1/2 (W_theta_flow - W_theta_blade)^2``
    [J/kg] (section 4.3; Galvas/NASA Eq 5.6, CONFIRMED -- CENT-LOSS.md; a
    Conrad/Aungier f_inc=0.5-0.8 reducing factor is a [DECIDE] refinement)."""
    xp = get_xp(xp)
    d = w_theta_flow - w_theta_blade
    return 0.5 * d * d


def skin_friction_loss(w_avg, cf, l_over_dhyd, *, xp=None):
    """Passage skin-friction loss ``dh = 2 Cf (L/D_hyd) W_avg^2`` [J/kg]
    (section 4.3; Galvas ``4 Cf W^2/2`` leading factor CONFIRMED -- CENT-LOSS.md.
    Caller passes ``W_avg = 1/2(W1+W2)``; Aungier's mean-of-squares
    ``1/2(W1^2+W2^2)`` is the [DECIDE] alternative)."""
    xp = get_xp(xp)
    return 2.0 * cf * l_over_dhyd * w_avg * w_avg


@dataclass(frozen=True)
class CentrifugalLoss:
    """LossModel (section 7.1): representative centrifugal internal loss
    (incidence + skin friction), each converted to entropy per the Appendix-B
    enthalpy-loss form at the B.1-re-referenced exit static state.

    Requires ``row.geometry`` (section 4.1 contract) and the view's lagged TE
    fields. ``cf``/``l_over_dhyd`` are row-scalar design inputs."""

    cf: float = 0.005
    l_over_dhyd: float = 4.0

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
        dh_inc = incidence_loss(flow.w_theta, w_theta_blade, xp=xp)

        # Exit relative velocity via the shared Wiesner slip (same set):
        # W_theta2 = (sigma - 1) U2 - Vm2 tan(beta2b).
        b2b = soft_clip(sgn * g.beta2_blade(y), -_ANG_CAP, _ANG_CAP, _ANG_W,
                        xp=xp)
        sigma, v_s = wiesner_slip(b2b, g.blade_count, xp=xp)
        u1 = row.omega * flow.r
        u2 = row.omega * flow.r_te
        w_theta_2 = (sigma - 1.0) * u2 - flow.vm_te * xp.tan(b2b)
        w1 = xp.sqrt(flow.vm * flow.vm + flow.w_theta * flow.w_theta)
        w2 = xp.sqrt(flow.vm_te * flow.vm_te + w_theta_2 * w_theta_2)
        dh_sf = skin_friction_loss(0.5 * (w1 + w2), self.cf,
                                   self.l_over_dhyd, xp=xp)

        # Exit static temperature via B.1 rothalpy re-referencing (charging
        # temperature for the entropy conversion).
        T0r_1 = flow.T + 0.5 * w1 * w1 / fluid.cp
        T0r_2 = T0r_1 + 0.5 * (u2 * u2 - u1 * u1) / fluid.cp
        T2 = _T_FLOOR + softplus(T0r_2 - 0.5 * w2 * w2 / fluid.cp - _T_FLOOR,
                                 _T_W, xp=xp)

        ds_inc = delta_s_enthalpy_loss(fluid, dh_inc, T2, xp=xp)
        ds_sf = delta_s_enthalpy_loss(fluid, dh_sf, T2, xp=xp)
        return LossBreakdown(
            components={"incidence_dh": dh_inc, "friction_dh": dh_sf},
            delta_s=ds_inc + ds_sf, validity=float(xp.min(v_s)))
