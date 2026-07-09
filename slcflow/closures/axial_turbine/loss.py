"""Kacker-Okapuu profile-loss model for axial-turbine rows (Theory Manual
sections 4.3, 4.4, 7.1, 7.3; Appendix B.3 conversion).

The native coefficient is the B.3 turbine total-pressure loss ``Y``
(EXIT-dynamic-head reference); conversion to entropy happens HERE (section
4.4: converted at row exit, B.1 rothalpy re-referencing included) so
``delta_s`` is what crosses the interface. The exit gas angle comes from the
same throat chain as :class:`AinleyTurbineSwirl` — the two closures ship as
one consistent CorrelationSet (section 7.1).

Exit-state evaluation: the loss references the exit static pressure ``p2``
and the exit relative Mach, both taken at the loss-free (ideal) exit state
built from B.1 rothalpy conservation across the radius change. The loss then
perturbs ``p0r,2`` below its ideal value (B.3). This needs no addition to the
flow-view contract — only relative speeds, the LE static state, and the
blade speeds at LE/TE.

Subsonic components only (profile, with Mach ``K_p`` and Reynolds ``f_Re``
corrections); shock, secondary, and trailing-edge loss land at M6-3..M6-4.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._namespace import get_xp
from ..conversions import (delta_s_turbine_Y, ideal_exit_relative_stagnation,
                          relative_stagnation)
from ..interfaces import LossBreakdown, RowFlowView, RowView
from ..smoothmath import blend, smooth_min, soft_clip, softplus
from .ainley import throat_exit_angle
from .kacker_okapuu import (mach_profile_correction, profile_loss_am,
                           reynolds_correction, secondary_loss, shock_loss,
                           trailing_edge_zeta)

__all__ = ["KackerOkapuuLoss"]

# Ideal exit static-temperature floor (section 7.3.2): keep the exit speed
# of sound real through violent transients (the loss-free T2 dipping toward 0
# only happens far out of the subsonic calibration, where validity is 0).
_T_FLOOR, _T_W = 20.0, 5.0
# Conservative asymptote for Y (section 7.3.2): level off well inside the B.3
# conversion's domain; validity reflects binding. [VERIFY ceiling choice]
_Y_CEIL, _Y_W = 0.5, 0.05


@dataclass(frozen=True)
class KackerOkapuuLoss:
    """LossModel (section 7.1): Kacker-Okapuu subsonic loss (profile +
    secondary + trailing-edge), converted to entropy per B.3 at the
    B.1-re-referenced exit state.

    All three components share the B.3 exit-dynamic-head reference: the
    trailing-edge kinetic-energy coefficient is mapped to an equivalent
    ``Y`` (``Y_TE = zeta/(1-zeta)`` -- CONFIRMED as the M2->0 incompressible
    limit of the exact compressible K-O relation, docs/references/KO82.md) so
    the family sums to one exit-reference ``Y`` and one B.3 conversion (K-O
    standard; B.5-compliant within the single-reference turbine family). Using B.3 rather than B.4
    for the TE term keeps the residual path exception-free (AD-10) — B.4's
    ``assert`` guard cannot bind.

    Requires ``row.geometry`` (section 4.1 contract, with a ``throat``) and
    the view's lagged TE fields. ``reynolds`` is the design chord Reynolds
    number (per-node Re deferred to a transport backend, ARCH-9; default in
    the flat band so ``f_Re = 1``). ``aspect_ratio`` (blade height/chord)
    and ``te_o_ratio`` (TE thickness/throat) are row-scalar design inputs;
    deriving aspect ratio from the annulus span is a future refinement."""

    reynolds: float = 5.0e5
    aspect_ratio: float = 3.0
    te_o_ratio: float = 0.02

    def evaluate(self, row: RowView, flow: RowFlowView) -> LossBreakdown:
        xp = get_xp(None)
        g = row.geometry
        y = flow.psi
        # Cascade frame = the blade's EXIT turning direction (the frame in
        # which the throat exit angle is positive), so the inlet flow angle
        # is signed consistently with the exit-angle chain below. A reaction
        # rotor with co-rotating relative inflow maps to a NEGATIVE cascade
        # inlet angle (beyond-nozzle; the AM fits soft-clip b1/b2 >= -1).
        sgn = g.orientation_te           # geometry constant (ARCH-4.2)

        # Exit gas angle from the shared throat chain (same CorrelationSet).
        pitch = 2.0 * xp.pi * flow.r_te / g.blade_count
        alpha2_deg, v_a = throat_exit_angle(g.throat(y) / pitch, xp=xp)
        b2r = xp.deg2rad(alpha2_deg)

        # Inlet flow angle (cascade frame, signed), pitch/chord, thickness.
        b1_deg = xp.rad2deg(sgn * flow.beta)
        s_c = 1.0 / g.solidity(y)
        tc = g.thickness_ratio(y)

        # Velocity-triangle magnitudes (relative frame): W = Vm / cos(angle).
        b1r = xp.deg2rad(soft_clip(b1_deg, -85.0, 85.0, 2.0, xp=xp))
        w1 = flow.vm / xp.cos(b1r)
        w2 = flow.vm_te / xp.cos(b2r)

        # Relative stagnation + ideal (loss-free) exit static state via B.1
        # rothalpy re-referencing across the radius change.
        fluid = flow.fluid
        p1 = fluid.p(flow.h, flow.s)
        T0r_1, p0r_1 = relative_stagnation(fluid, flow.T, p1, w1, xp=xp)
        u1 = row.omega * flow.r
        u2 = row.omega * flow.r_te
        T0r_2, p0r_2_id = ideal_exit_relative_stagnation(
            fluid, T0r_1, p0r_1, u1, u2, xp=xp)
        T2 = _T_FLOOR + softplus(T0r_2 - 0.5 * w2 * w2 / fluid.cp - _T_FLOOR,
                                 _T_W, xp=xp)
        a2 = xp.sqrt(fluid.gamma * fluid.R * T2)
        m1 = w1 / flow.a
        m2 = w2 / a2
        p2 = p0r_2_id * (T2 / T0r_2) ** (fluid.gamma / (fluid.gamma - 1.0))

        # Kacker-Okapuu loss components -- all exit-reference Y (B.3); the TE
        # kinetic-energy coefficient is mapped to an equivalent Y before
        # summing (see class docstring). Profile + inlet shock share the K-O
        # bracket 0.914 (2/3 Y_p,AM K_p + Y_shock) f_Re; recorded separately.
        yp_am, v_p = profile_loss_am(s_c, b1_deg, alpha2_deg, tc, xp=xp)
        kp = mach_profile_correction(m1, m2, xp=xp)
        y_shock_raw, v_sh = shock_loss(m1, xp=xp)
        env = 0.914 * reynolds_correction(self.reynolds, xp=xp)
        y_profile = env * (2.0 / 3.0) * yp_am * kp
        y_shock = env * y_shock_raw
        y_secondary, v_s = secondary_loss(b1_deg, alpha2_deg,
                                          self.aspect_ratio, xp=xp)
        zeta_te, v_te = trailing_edge_zeta(b1_deg, alpha2_deg,
                                           self.te_o_ratio, xp=xp)
        y_te = zeta_te / (1.0 - zeta_te)

        y_raw = y_profile + y_shock + y_secondary + y_te
        Y = smooth_min(y_raw, _Y_CEIL, _Y_W, xp=xp)
        v_y = 1.0 - blend(y_raw, _Y_CEIL, 10.0 * _Y_W, xp=xp)

        delta_s, _ = delta_s_turbine_Y(fluid, Y, T0r_1, p0r_1, p2, u1, u2,
                                       xp=xp)
        # Components recorded individually for auditability (B.5.3).
        return LossBreakdown(
            components={"profile_Y": y_profile, "shock_Y": y_shock,
                        "secondary_Y": y_secondary, "te_Y": y_te,
                        "te_zeta": zeta_te, "Y_total": Y},
            delta_s=delta_s,
            validity=float(xp.min(v_a * v_p * v_y * v_s * v_sh * v_te)))
