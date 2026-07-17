"""Centrifugal parasitic (shaft-side) losses: disk friction, blade-clearance
leakage, recirculation (Theory Manual section 4.4 scope note; the deferred
gate-#3 components).

Provenance — Aungier, *Centrifugal Compressors* (2000), ch. 4, extracted
verbatim via the theory notebook (2026-07-16; docs/references/CENT-LOSS.md
"parasitic" section):

  * Disk friction  (Eqs 4-21..4-25, 4-31): ``dh_DF = C_M rho r2^2 U2^3 /
    (2 mdot)``, ``Re = rho omega r2^2 / mu``, ``C_M`` = the LARGEST of the
    four Daily-Nece regime correlations, times the 0.75 experience factor.
  * Leakage        (Eqs 4-17..4-19, 4-40): ``dp_CL = mdot (r2 CU2 - r1 CU1)
    / (z rbar bbar L)``; ``U_CL = 0.816 sqrt(2 dp_CL / rho2)``;
    ``mdot_CL = rho2 z s L U_CL``; ``dh_L = mdot_CL U_CL U2 / (2 mdot)``.
  * Recirculation  (Eqs 4-41..4-43): ``dh_RC = I_R U2^2``,
    ``I_R = (D_eq/2 - 1)(W_U2/C_m2 - 2 cot(beta2b)) >= 0`` with Aungier's
    impeller equivalent diffusion ``D_eq = W_max/W2``, ``W_max = (W1 + W2 +
    dW)/2``, ``dW = 2 pi d2 U2 I_B / (z L_B) = 4 pi (r2 CU2 - r1 CU1) /
    (z L_B)`` (triangular loading). ``beta2b`` is the blade exit angle
    FROM TANGENTIAL (radial blades: 90 deg, cot = 0).

These are PARASITIC in Aungier's accounting: they add to the shaft work
(the impeller re-energizes leakage/recirculated/windage fluid) without
contributing pressure rise — so they debit the STAGE efficiency
``eta = dh_ideal(PR) / (dh0_flow + sum dh_parasitic)`` and never enter the
flow solution. Accordingly these are POST-SOLVE SCALAR diagnostics
(facade/case level, like the drivers' capacity checks): they never touch
flow-state arrays on the solve path, so the section 7.3 C1 duty does not
bind (plain float arithmetic; the ``max``/branches below are on converged
scalars, ARCH-4.2). Machine-level ``mdot`` is exactly why they cannot be
per-streamtube LossModel components (the recorded M7 deferral).
"""
from __future__ import annotations

__all__ = ["disk_friction_work", "leakage_work", "recirculation_work",
           "vaneless_diffuser_loss"]

_TWO_PI = 6.283185307179586


def disk_friction_work(mdot, rho, u2, r2, mu=1.81e-5, gap_ratio=0.02,
                       experience=0.75):
    """Disk-friction parasitic work [J/kg] (Aungier Eqs 4-21..25, 4-31).

    ``rho`` is the disk-cavity density (impeller-exit static is the
    customary estimate), ``gap_ratio`` the backface axial gap ``s/r2``
    (rig value rarely published — recorded assumption), ``experience``
    the 0.75 factor of Eq 4-31. The regime is the LARGEST of the four
    Daily-Nece torque coefficients (Aungier's stated selection rule).
    """
    re = rho * u2 * r2 / mu                 # rho omega r2^2 / mu, u2 = omega r2
    s_r = gap_ratio
    cm1 = _TWO_PI / (s_r * re)
    cm2 = 3.7 * s_r ** 0.1 / re ** 0.5
    cm3 = 0.08 / (s_r ** (1.0 / 6.0) * re ** 0.25)
    cm4 = 0.102 * s_r ** 0.1 / re ** 0.2
    cm = experience * max(cm1, cm2, cm3, cm4)
    return cm * rho * r2 ** 2 * u2 ** 3 / (2.0 * mdot)


def leakage_work(mdot, rho2, u2, r1, r2, b1, b2, cu1, cu2, blade_count,
                 clearance, blade_length):
    """Blade-clearance leakage parasitic work [J/kg] (Aungier Eqs
    4-17..4-19, 4-40; open impellers, half the leakage re-energized).

    ``clearance`` is the blade tip gap ``s`` [m]; ``blade_length`` the
    mean blade (meridional) length ``L`` [m]; ``b1``/``b2`` the passage
    widths at inlet/exit; ``cu1``/``cu2`` absolute tangential velocities.
    """
    r_bar = 0.5 * (r1 + r2)
    b_bar = 0.5 * (b1 + b2)
    dp_cl = mdot * (r2 * cu2 - r1 * cu1) / (
        blade_count * r_bar * b_bar * blade_length)
    if dp_cl <= 0.0:
        return 0.0                          # no positive blade loading
    u_cl = 0.816 * (2.0 * dp_cl / rho2) ** 0.5
    mdot_cl = rho2 * blade_count * clearance * blade_length * u_cl
    return mdot_cl * u_cl * u2 / (2.0 * mdot)


def vaneless_diffuser_loss(cf, r_in, r_out, b_in, c_in, cu_in, u_ref):
    """Vaneless-diffuser skin-friction loss [J/kg] — an INTERNAL (p0) loss,
    included here as the stage-level post-solve companion of the parasitic
    set (same accounting seam: evaluated on the converged impeller-exit
    state, never inside the solve).

    Closed-form Coppage et al. (1956)/Stanitz (1952) simplification as
    quoted by Whitfield & Baines (1990) Eq. [30] (extracted verbatim via
    the theory notebook, 2026-07-17 — docs/references/CENT-LOSS.md):

        delta_q = cf r_x (1 - (r_x/r_y)^1.5) (C_x/U_T)^2
                  / (1.5 b_x cos(alpha_x))

    a loss in work-coefficient units (returned here times ``u_ref**2``).
    ``alpha_x`` is the flow angle from TANGENT (cos = Cu/C). Aungier's own
    treatment is the full 5-45/5-46 radial marching integration — recorded
    as the refinement; the closed form assumes a log-spiral path at the
    inlet flow angle with constant width, adequate for a constant-area
    vaneless space (the Eckardt configuration).
    """
    if c_in <= 0.0 or cu_in <= 0.0:
        return 0.0                       # no swirl path, negligible loss
    cos_alpha = cu_in / c_in
    dq = cf * r_in * (1.0 - (r_in / r_out) ** 1.5) * (c_in / u_ref) ** 2 \
        / (1.5 * b_in * cos_alpha)
    return dq * u_ref ** 2


def recirculation_work(u2, w1, w2, cm2, wu2, cot_beta2_blade, r1cu1, r2cu2,
                       blade_count, blade_length):
    """Recirculation parasitic work [J/kg] (Aungier Eqs 4-41..4-43).

    ``cot_beta2_blade`` is cot of the blade EXIT angle from TANGENTIAL
    (radial-ending blades: 0). Zero at low loading via the published
    ``I_R >= 0`` constraint (a design-point impeller typically recirculates
    nothing; this is the high-loading/off-design term).
    """
    dw = 4.0 * 3.141592653589793 * (r2cu2 - r1cu1) / (
        blade_count * blade_length)
    d_eq = (w1 + w2 + dw) / (2.0 * w2)
    i_r = (d_eq / 2.0 - 1.0) * (wu2 / cm2 - 2.0 * cot_beta2_blade)
    return max(0.0, i_r) * u2 ** 2
