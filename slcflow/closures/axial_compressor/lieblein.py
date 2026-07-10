"""Lieblein incidence/deviation correlation set for axial-compressor rows
(Theory Manual sections 4.3, 3.4, 7.1, 7.3; NASA SP-36 via Aungier's
analytic curve fits).

Provenance: the working equations below are Aungier's published fits
(*Axial-Flow Compressors*, ch. 6) to Lieblein's NASA SP-36 low-speed
cascade charts for NACA-65 profiles (shape factor ``K_sh = 1``).

Verification status (see ``docs/references/AUN-C.md``, 2026-07-09): every
incidence/deviation/off-design-slope **fit coefficient** is CONFIRMED against
Aungier ch. 6 (Eqs 6-13/6-14/6-15/6-20/6-21/6-22/6-24/6-25/6-76), pinned in
``tests/test_lieblein_reference.py``. One transcription bug was found and
FIXED: the ``K_ti`` exponent used ``(10 t/c)^0.3`` where Aungier Eq 6-11 has
``(t/c)^0.3``. The fit **outputs** are also now validated end-to-end against
the digitized NASA SP-36 (Lieblein) design charts (Figs 137/138/161/162):
(i0)_10 to RMS 0.10 deg, (delta0)_10 to RMS 0.17 deg, and the n/m camber slopes
overlay-coincident -- no discrepancy (``tools/digitize_sp36.py``,
``tests/test_lieblein_sp36_charts.py``, docs/references/AUN-C.md). Residual
``[VERIFY]``: ``K_sh``/ranges for non-NACA-65 profiles.

Frame convention (section 2.4 + the [VERIFY per correlation] remap duty):
the fits are written in the *cascade frame* — angles in DEGREES, measured
from the meridional, positive in the blade's tangential orientation, so
inlet flow angle, metal angles, and camber are all normally positive for a
decelerating cascade. The closure maps our signed-radian convention in and
out using the BLADE orientation (geometry data, constant per solve —
branching on it is topology branching, ARCH-4.2).

Smoothness (section 7.3): inputs are soft-clipped into the calibrated
domain (documented widths) and every evaluation returns a validity measure
built from compact-support blend windows — 1 strictly inside calibration,
smoothly to 0 outside. No raw clamps, no exceptions.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..._namespace import get_xp
from ..interfaces import RowFlowView, RowView, SwirlResult
from ..smoothmath import blend, smooth_min, soft_clip, softplus

__all__ = ["reference_incidence", "reference_deviation", "deviation_slope",
           "LieblienSwirl", "CALIBRATED"]

# Calibrated input domain of the SP-36 charts (cascade frame, degrees):
# (lo, hi, transition width) for the validity windows, plus the hard
# MATHEMATICAL floor the saturated value may never cross (the fits contain
# fractional powers and divisions whose domain ends there). [VERIFY ranges]
CALIBRATED = {
    "beta1_deg": (0.0, 70.0, 2.0, 0.0),
    "sigma": (0.4, 2.0, 0.05, 0.1),
    "tc": (0.02, 0.12, 0.005, 0.005),
}


def _saturate(x, key, *, xp=None):
    """Smoothly saturate into the fits' safe domain and return the
    compact-support in-range validity (1 strictly inside calibration,
    C1-smoothly to 0 outside; section 7.3.2/3).

    The value channel is ``floor + softplus(smooth_min(x, hi) - floor)``:
    exact (to fp) deep inside the range, C-infinity, ceilinged near ``hi``
    and provably >= ``floor`` — fractional powers and divisions downstream
    can never see an out-of-domain base, no matter how wild the input."""
    lo, hi, w, floor = CALIBRATED[key]
    x_sat = floor + softplus(smooth_min(x, hi, w, xp=xp) - floor, w, xp=xp)
    v = blend(x, lo, w, xp=xp) * (1.0 - blend(x, hi, w, xp=xp))
    return x_sat, v


def reference_incidence(beta1_deg, sigma, tc, camber_deg, *, xp=None):
    """Minimum-loss reference incidence [deg] (section 4.3; SP-36 fits).

    ``i_ref = K_sh K_ti (i0)_10 + n * camber`` with ``K_sh = 1`` (NACA-65)
    and Aungier's fits for the 10%-thickness zero-camber incidence
    ``(i0)_10``, the thickness correction ``K_ti``, and the camber slope
    ``n``. Returns ``(i_ref_deg, validity)``.
    """
    xp = get_xp(xp)
    b, v1 = _saturate(beta1_deg, "beta1_deg", xp=xp)
    s, v2 = _saturate(sigma, "sigma", xp=xp)
    t, v3 = _saturate(tc, "tc", xp=xp)
    p = 0.914 + s ** 3 / 160.0
    i0_10 = (b ** p / (5.0 + 46.0 * xp.exp(-2.3 * s))
             - 0.1 * s ** 3 * xp.exp((b - 70.0) / 4.0))
    # Aungier Eq 6-10/6-11: K_ti = (10 t/c)^q, q = 0.28/[0.1 + (t/c)^0.3].
    # The 0.3-power base is t/c (NOT 10 t/c) -- verified, docs/references/
    # AUN-C.md; corrected here (was an extra x10 in the exponent denominator).
    k_ti = (10.0 * t) ** (0.28 / (0.1 + t ** 0.3))
    n = (0.025 * s - 0.06
         - (b / 90.0) ** (1.0 + 1.2 * s) / (1.5 + 0.43 * s))
    return k_ti * i0_10 + n * camber_deg, v1 * v2 * v3


def reference_deviation(beta1_deg, sigma, tc, camber_deg, *, xp=None):
    """Minimum-loss reference deviation [deg] (section 3.4; SP-36 fits).

    ``dev_ref = K_sh K_td (d0)_10 + m * camber`` with Aungier's fits for
    the zero-camber deviation ``(d0)_10``, thickness correction ``K_td``,
    and Carter-style camber slope ``m = m_1.0 / sigma^b``. Returns
    ``(dev_ref_deg, validity)``.
    """
    xp = get_xp(xp)
    b, v1 = _saturate(beta1_deg, "beta1_deg", xp=xp)
    s, v2 = _saturate(sigma, "sigma", xp=xp)
    t, v3 = _saturate(tc, "tc", xp=xp)
    d0_10 = (0.01 * s * b
             + (0.74 * s ** 1.9 + 3.0 * s) * (b / 90.0) ** (1.67 + 1.09 * s))
    k_td = 6.25 * t + 37.5 * t * t   # = 1 at t/c = 0.10 by construction
    m10 = 0.17 - 0.0333 * (b / 100.0) + 0.333 * (b / 100.0) ** 2
    bexp = 0.9625 - 0.17 * (b / 100.0) - 0.85 * (b / 100.0) ** 3
    m = m10 / s ** bexp
    return k_td * d0_10 + m * camber_deg, v1 * v2 * v3


def deviation_slope(beta1_deg, sigma, *, xp=None):
    """Off-design deviation slope ``(d delta / d i)`` at the reference
    point (Aungier's fit); dimensionless, applied as
    ``delta = dev_ref + slope * (i - i_ref)`` (section 4.3)."""
    xp = get_xp(xp)
    b, _ = _saturate(beta1_deg, "beta1_deg", xp=xp)
    s, _ = _saturate(sigma, "sigma", xp=xp)
    return (1.0 + (s + 0.25 * s ** 4) * (b / 53.0) ** 2.5) / xp.exp(3.1 * s)


@dataclass(frozen=True)
class LieblienSwirl:
    """SwirlClosure (section 7.1): exit rVt from Lieblein deviation.

    Exit relative angle ``beta2 = beta2_blade + deviation`` (section 3.4),
    with deviation evaluated at the actual incidence via the reference
    point and the off-design slope (section 4.3). The TE meridional
    velocity is taken from the lagged TE flow in the view (``vm_te``,
    section 7.2 "and TE where iterative"); the exit relative angle is
    soft-limited to ``+-80 deg`` in the cascade frame (smooth saturation,
    section 7.3.2) before the tangent map.

    Requires ``row.geometry`` implementing the section 4.1 contract and
    the view's ``r_te``/``vm_te`` fields (driver-provided, lagged).
    """

    k_sh: float = 1.0    # blade shape factor; NACA-65 = 1.0 [VERIFY others]

    def exit_rvt(self, row: RowView, flow: RowFlowView) -> SwirlResult:
        xp = get_xp(None)
        g = row.geometry
        y = flow.psi                      # span fraction ~ mass fraction
        sgn = g.orientation               # geometry constant (ARCH-4.2)

        b1_blade = xp.rad2deg(sgn * g.beta1_blade(y))
        b2_blade = xp.rad2deg(sgn * g.beta2_blade(y))
        camber = b1_blade - b2_blade
        sigma = g.solidity(y)
        tc = g.thickness_ratio(y)
        b1_flow = xp.rad2deg(sgn * flow.beta)

        i = b1_flow - b1_blade            # incidence, cascade frame (4.3)
        i_ref, v_i = reference_incidence(b1_flow, sigma, tc, camber, xp=xp)
        d_ref, v_d = reference_deviation(b1_flow, sigma, tc, camber, xp=xp)
        slope = deviation_slope(b1_flow, sigma, xp=xp)
        dev = d_ref + slope * (i - i_ref)

        b2_deg = soft_clip(b2_blade + dev, -80.0, 80.0, 2.0, xp=xp)
        w_theta_2 = flow.vm_te * xp.tan(sgn * xp.deg2rad(b2_deg))
        vtheta_2 = w_theta_2 + row.omega * flow.r_te
        return SwirlResult(rvt=flow.r_te * vtheta_2,
                           validity=float(xp.min(v_i * v_d)))
