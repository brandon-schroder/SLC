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
           "vaneless_diffuser_loss", "tip_distortion_loss",
           "jansen_clearance_loss", "johnston_dean_mixing_loss",
           "supercritical_loss"]

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


def vaneless_diffuser_march(fluid, r_in, r_out, b_in, h0, s_in, rcu_in,
                            cm_in, cf=0.005, area_law="width", n=200):
    """One-dimensional vaneless-diffuser radial marching (Aungier Eqs
    5-45/5-46 family; theory notebook 2026-07-17) — the refinement of the
    Coppage/Stanitz closed form, needed because the width law matters:
    Eckardt's space is constant-WIDTH (b = b2) while Krain's is
    constant-AREA (r b = r2 b2 — the papers' own attribution for the
    0.95 impeller -> 0.84 stage efficiency drop).

        d(r Cu)/dr = - cf r C Cu / (b Cm)          (Eq 5-45, dm = dr)
        T ds/dr    =   cf C^3 / (b Cm)             (wall dissipation)
        Cm         =   mdot / (rho 2 pi r b),  rho = rho(h0 - C^2/2, s)

    marched RK2 from ``r_in`` to ``r_out``; ``area_law`` selects
    ``b(r) = b_in`` ("width") or ``b(r) = b_in r_in/r`` ("area").
    Returns ``(delta_s, rcu_out, cm_out)`` — the entropy rise is the p0
    debit (``exp(-ds/R)``), swirl decays per 5-45. Post-solve scalar
    marching at the case level (never on the solve path); the plain
    Python loop is the ARCH-4.2-sanctioned driver-style iteration.
    """
    mdot_per = None                      # rho2 Cm 2 pi r b, set at inlet
    r = r_in
    b = b_in
    rho = float(fluid.rho(h0 - 0.5 * (cm_in ** 2 + (rcu_in / r_in) ** 2),
                          s_in))
    mdot_per = rho * cm_in * r_in * b_in
    rcu, s = float(rcu_in), float(s_in)
    dr = (r_out - r_in) / n

    def rates(r, rcu, s):
        b = b_in if area_law == "width" else b_in * r_in / r
        cu = rcu / r
        # meridional velocity from continuity at the local state
        # (one fixed-point pass on rho suffices at these Mach numbers)
        cm = cm_in
        for _ in range(3):
            c2 = cm * cm + cu * cu
            rho = float(fluid.rho(h0 - 0.5 * c2, s))
            cm = mdot_per / (rho * r * b)
        c = (cm * cm + cu * cu) ** 0.5
        T = float(fluid.T(h0 - 0.5 * c * c, s))
        drcu = -cf * r * c * cu / (b * cm)
        ds = cf * c ** 3 / (b * cm * T)
        return drcu, ds, cm

    for _ in range(n):
        k1_rcu, k1_s, _ = rates(r, rcu, s)
        k2_rcu, k2_s, _ = rates(r + dr, rcu + dr * k1_rcu, s + dr * k1_s)
        rcu += 0.5 * dr * (k1_rcu + k2_rcu)
        s += 0.5 * dr * (k1_s + k2_s)
        r += dr
    _, _, cm_out = rates(r_out, rcu, s)
    return s - float(s_in), rcu, cm_out


def supercritical_loss(m1_rel, w1, w2, dw_loading, w_star):
    """Aungier supercritical Mach-number INTERNAL loss [J/kg] (shock /
    shock-induced separation once the suction-surface peak goes sonic;
    theory notebook 2026-07-17, verbatim):

        W_max  = (W1 + W2 + dW)/2                (Eq 4-41, dW = loading)
        M'_cr  = M1' W*/W_max                    (Eq 5-41)
        omega_bar_cr = 0.4 [ (M1' - M'_cr) W_max/W1 ]^2   (Eq 5-42)

    active only for ``M1' > M'_cr`` (floored otherwise); returned on the
    ch.-5 inlet-relative reference (x 1/2 W1^2). ``w_star`` is the local
    sonic velocity at the relative total conditions. The loading dW is
    the Eq 4-42 blade-loading velocity difference the recirculation term
    already uses. The mechanism that discriminates the Krain (M1'~0.85)
    from the Eckardt (M1'~0.67) inducer."""
    w_max = 0.5 * (w1 + w2 + dw_loading)
    m_cr = m1_rel * w_star / w_max
    dm = m1_rel - m_cr
    if dm <= 0.0:
        return 0.0
    return 0.4 * (dm * w_max / w1) ** 2 * 0.5 * w1 * w1


def jansen_clearance_loss(clearance, b2, cu2, cm1, r1h, r1t, r2,
                          rho1, rho2, blade_count):
    """Jansen (1967) tip-clearance INTERNAL loss [J/kg] — the Oh-native
    clearance component (sudden contraction/expansion through the gap),
    verbatim as quoted by Whitfield & Baines (theory notebook, 2026-07-17;
    docs/references/CENT-LOSS.md "Oh-native" section):

        dh_cl = 0.6 (s/b2) Cu2 sqrt( (4 pi/(b2 Z))
                * (r1t^2 - r1h^2)/((r2 - r1t)(1 + rho2/rho1))
                * Cu2 Cm1 )

    Mutually exclusive with the Aungier lambda tip-distortion accounting
    (same physics family) — the caller selects one (see the case's
    ``accounting`` option)."""
    if cu2 <= 0.0 or cm1 <= 0.0:
        return 0.0
    inner = (4.0 * 3.141592653589793 / (b2 * blade_count)
             * (r1t ** 2 - r1h ** 2)
             / ((r2 - r1t) * (1.0 + rho2 / rho1)) * cu2 * cm1)
    if inner <= 0.0:
        return 0.0
    return 0.6 * (clearance / b2) * cu2 * inner ** 0.5


def johnston_dean_mixing_loss(cm2, wake_fraction=0.2, blockage=0.05):
    """Johnston & Dean (1966) wake MIXING internal loss [J/kg] — the
    Oh-native mixing component (sudden expansion of the jet-wake pattern),
    involving ONLY the meridional velocity (Aungier, quoted verbatim:
    "the tangential velocity is controlled by conservation of angular
    momentum ... the wake mixing loss involves only the meridional
    component"; theory notebook 2026-07-17):

        dh_mix = 1/(1 + eps_w) * ((1 - eps_w - b*)/(1 - eps_w))^2
                 * Cm2^2 / 2

    ``wake_fraction`` (eps_w) typical 0.15-0.25 for high-performance
    impellers; ``blockage`` (b*) typical 0.02-0.12 — both documented
    ranges, defaults at the customary mid-values. Mutually exclusive with
    the Aungier lambda accounting (same physics family)."""
    e, b = wake_fraction, blockage
    return (1.0 / (1.0 + e)) * ((1.0 - e - b) / (1.0 - e)) ** 2 \
        * 0.5 * cm2 * cm2


def tip_distortion_loss(omega_sf, pv1, pv2, w1, w2, cm2, d_hyd, b2, l_b,
                        area_ratio, rho1, rho2, clearance):
    """Aungier tip-distortion (wake-mixing) INTERNAL loss [J/kg] — the
    stage-level companion for the clearance/blockage effect Aungier folds
    into the impeller tip distortion factor rather than a separate
    clearance-loss coefficient.

    Verbatim chain (theory notebook, 2026-07-17 —
    docs/references/CENT-LOSS.md):

        B2 = omega_SF (pv1/pv2) sqrt(W1 d_H/(W2 b2))
             + [0.3 + (b2/L_B)^2] A_R^2 rho2 b2/(rho1 L_B)
             + s_CL/(2 b2)                                     (Eq 4-12)
        lambda = 1/(1 - B2)                                    (Eq 120)
        omega_bar_lambda = [(lambda - 1) C_m2/W2]^2            (Eq 5-36)

    with the mean hydraulic diameter d_H = (d_H1 + d_H2)/2,
    ``d_Hi = 2 b_i w_i/(b_i + w_i)``, blade-to-blade width
    ``w = 2 pi r sin(beta)/Z`` (beta from tangent; Eqs 111/113), and the
    passage area ratio ``A_R = A2 sin(beta2)/(A1 sin(beta_th))``
    (Eq 4-13) — the caller supplies those geometric composites.
    ``omega_SF`` is the impeller skin-friction loss coefficient on the
    inlet relative velocity pressure (the internal set's own convention);
    ``pv1/pv2`` the inlet/exit relative velocity-pressure ratio. Returned
    as ``omega_bar * 0.5 W1^2`` (the ch.-5 inlet-relative reference).
    ``B2`` is guarded below the lambda pole (0.9 ceiling — far outside
    the correlation's range; post-solve scalar guard, ARCH-4.2).

    NOTE: lambda has a SECOND role in Aungier's method — the work-input
    exit triangle (Eq 4-3: ``C_U2inf/U2 = 1 - lambda*phi2*cot(beta2)``, so
    the jet's blocked ``lambda*Cm2`` REDUCES the backswept work). Grounded
    and characterized 2026-07-19, then NOT adopted: it is exactly zero for
    a radial impeller (Eckardt, beta2b=0) and ~6% of work for backswept
    Krain — the WRONG direction there (measured Krain work is already at
    the high end, Wiesner-unblocked only -2% under), and the shared
    diffuser calibration (anchored on radial Eckardt) cannot rebalance it.
    See docs/references/CENT-LOSS.md; pinned in test_parasitic_reference.
    """
    b2_blk = (omega_sf * (pv1 / pv2) * (w1 * d_hyd / (w2 * b2)) ** 0.5
              + (0.3 + (b2 / l_b) ** 2) * area_ratio ** 2 * rho2 * b2
              / (rho1 * l_b)
              + clearance / (2.0 * b2))
    b2_blk = min(b2_blk, 0.9)
    lam = 1.0 / (1.0 - b2_blk)
    return ((lam - 1.0) * cm2 / w2) ** 2 * 0.5 * w1 * w1


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
