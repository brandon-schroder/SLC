"""Loss-coefficient -> entropy conversions (Theory Manual section 4.4,
Appendix B — normative working forms).

Internal loss currency is specific entropy (section 1, principle 3); every
correlation-native coefficient is converted HERE, at row-exit conditions,
before it crosses the closure interface. Perfect-gas closed forms per the
Appendix B preamble — the fast path is explicitly permitted inside
conversions (section 4.4); a real-gas backend later replaces these closed
forms with ``WorkingFluid`` inversions while keeping the same coefficient
*definitions*. The ``fluid`` argument must expose ``cp``, ``gamma``, ``R``
(the :class:`~slcflow.fluid.perfectgas.PerfectGas` backend does).

Bookkeeping rules (B.5): convert loss components INDIVIDUALLY, then sum the
resulting delta_s — never sum pressure-loss coefficients with different
reference dynamic heads. Each source correlation's exact coefficient
definition (reference dynamic head, frame) is **[VERIFY per correlation]**
against its paper — the single most common implementation bug class in
throughflow codes (section 4.4).

All functions vectorized over streamtube arrays; smooth (log/exp) in their
coefficient arguments, so section 7.3 C1 discipline is inherited from the
upstream correlation's smoothness. Out-of-range saturation is the
CORRELATION's job (coefficients are saturated before conversion, B.4 note);
these conversions assume physically admissible inputs.
"""
from __future__ import annotations

from .._namespace import get_xp

__all__ = ["relative_stagnation", "ideal_exit_relative_stagnation",
           "delta_s_from_p0_deficit", "delta_s_compressor_omega_bar",
           "delta_s_turbine_Y", "delta_s_kinetic_energy_zeta"]


def relative_stagnation(fluid, T, p, W, *, xp=None):
    """Relative-frame stagnation state ``(T0r, p0r)`` from a static state
    and relative speed ``W`` (isentropic recovery, perfect gas).

    Absolute-frame usage: pass the absolute speed ``V`` (the formulas are
    frame-blind; only the speed differs).
    """
    xp = get_xp(xp)
    T0r = T + W * W / (2.0 * fluid.cp)
    p0r = p * (T0r / T) ** (fluid.gamma / (fluid.gamma - 1.0))
    return T0r, p0r


def ideal_exit_relative_stagnation(fluid, T0r_1, p0r_1, u1, u2, *, xp=None):
    """Loss-free exit relative stagnation state (B.1).

    Rothalpy conservation across the radius change: ``T0r_2 = T0r_1 +
    (U2^2 - U1^2) / (2 cp)``, with the isentropic re-referencing
    ``p0r_2,id = p0r_1 (T0r_2 / T0r_1)^(gamma/(gamma-1))``. This radius
    correction is what makes the conversion correct for radial and mixed
    rotors; omitting it (valid only when U2 ~ U1) is a known axial-code
    habit that must not survive here (B.1). Stator/duct degeneracy:
    ``u1 = u2 = 0`` gives ``T0_2,id = T0_1``, ``p0_2,id = p0_1`` exactly.
    """
    xp = get_xp(xp)
    T0r_2 = T0r_1 + (u2 * u2 - u1 * u1) / (2.0 * fluid.cp)
    p0r_2_id = p0r_1 * (T0r_2 / T0r_1) ** (fluid.gamma / (fluid.gamma - 1.0))
    return T0r_2, p0r_2_id


def delta_s_from_p0_deficit(fluid, p0_actual, p0_ideal, *, xp=None):
    """Master relation (Appendix B): ``delta_s = -R ln(p0_actual /
    p0_ideal)`` at common stagnation temperature."""
    xp = get_xp(xp)
    return -fluid.R * xp.log(p0_actual / p0_ideal)


def delta_s_compressor_omega_bar(fluid, omega_bar, T0r_1, p0r_1, p1, u1, u2,
                                 *, xp=None):
    """Compressor-style relative total-pressure loss coefficient (B.2):
    ``omega_bar = (p0r_2,id - p0r_2) / (p0r_1 - p1)`` with the INLET
    relative dynamic head as reference **[VERIFY per correlation]**.

    Returns ``(delta_s, p0r_2)``; charged at exit conditions via the B.1
    re-referenced ideal state. Stator case: absolute-frame inputs with
    ``u1 = u2 = 0``.
    """
    xp = get_xp(xp)
    _, p0r_2_id = ideal_exit_relative_stagnation(fluid, T0r_1, p0r_1,
                                                 u1, u2, xp=xp)
    p0r_2 = p0r_2_id - omega_bar * (p0r_1 - p1)
    return delta_s_from_p0_deficit(fluid, p0r_2, p0r_2_id, xp=xp), p0r_2


def delta_s_turbine_Y(fluid, Y, T0r_1, p0r_1, p2, u1, u2, *, xp=None):
    """Turbine total-pressure loss coefficient, Ainley / Kacker-Okapuu
    style (B.3): ``Y = (p0(r),1 - p0(r),2) / (p0(r),2 - p2)`` — EXIT
    dynamic head reference **[VERIFY per correlation]**.

    Rotor form with B.1 re-referencing per the manual's rearrangement:
    ``p0r_2 = (p0r_2,id + Y p2) / (1 + Y)``. Returns ``(delta_s, p0r_2)``.
    Stator: absolute quantities, ``u1 = u2 = 0``.
    """
    xp = get_xp(xp)
    _, p0r_2_id = ideal_exit_relative_stagnation(fluid, T0r_1, p0r_1,
                                                 u1, u2, xp=xp)
    p0r_2 = (p0r_2_id + Y * p2) / (1.0 + Y)
    return delta_s_from_p0_deficit(fluid, p0r_2, p0r_2_id, xp=xp), p0r_2


def delta_s_kinetic_energy_zeta(fluid, zeta, T2, V2, *, xp=None):
    """Kinetic-energy / enthalpy loss coefficient, Craig-Cox style (B.4):
    ``zeta = (h2 - h2s) / (V2^2 / 2)`` (or relative-frame ``W2`` — per
    source **[VERIFY per correlation]**), ``h2s`` the isentropic exit
    static enthalpy at ``p2``.

    ``delta_s = cp ln(T2 / T2s)`` with ``T2s = T2 - zeta V2^2 / (2 cp)``
    (entropy difference between actual and isentropic states at the same
    p2). The guard ``zeta V2^2 / (2 cp T2) < 1`` must hold; the section
    7.3 saturation rule applies to the COEFFICIENT before conversion, so
    it never binds in a converged solve — assert, don't clamp (B.4).
    """
    xp = get_xp(xp)
    ratio = zeta * V2 * V2 / (2.0 * fluid.cp * T2)
    assert bool(xp.all(ratio < 1.0)), (
        "B.4 guard violated: zeta V2^2 / (2 cp T2) >= 1 -- the upstream "
        "correlation failed to saturate its coefficient (section 7.3)")
    T2s = T2 - zeta * V2 * V2 / (2.0 * fluid.cp)
    return fluid.cp * xp.log(T2 / T2s)
