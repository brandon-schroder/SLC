"""Spanwise mixing operator (Theory Manual section 3.6).

Multistage machines develop unphysical spanwise stratification of
``h0``, ``s`` and ``rV_theta`` unless a turbulent-mixing model redistributes
them. Section 3.6 adopts the diffusive form (Gallimore-Cumpsty): after each
row's transport step, march the parabolic operator

    d(chi)/dm = 1/(r (1-B) rho Vm) * d/dq ( mu_mix r d(chi)/dq ),
    chi in {h0, s, rV_theta},

one implicit step per station interval. Implicit (backward-Euler in ``m``,
tridiagonal in ``q``) gives unconditional stability so the marching step is
never limited by the mixing coefficient (section 3.6). Discretized in
finite-volume form with **zero-flux walls**, the step conserves the
mass-flux-weighted total ``sum_i w_i dq_i chi_i`` exactly (``w = r(1-B)rho Vm``
is the section 3.2 mass-flux weight) -- mixing redistributes h0/s/rVt across
the span, it does not create or destroy their fluxes.

This runs in the driver's **lagged field refresh** (section 6.2.2.4 step 4,
AD-4), on the frozen transported fields between outer iterates -- never on the
pure residual path (AD-3). The coefficient field ``mu_mix`` comes from a
:class:`~slcflow.closures.interfaces.MixingModel`; the default Gallimore form
and its constant are ``[VERIFY]`` (calibration deferred, ARCH-9).

Off by default in all tiers (``FidelityConfig.mixing_term = 0``); multistage
axial Tier-3 cases opt in.
"""
from __future__ import annotations

from dataclasses import dataclass

from .._namespace import get_xp
from ..fluid.base import Array
from .streamwise import TransportFields

__all__ = ["GallimoreMixing", "spanwise_diffusion_step", "mix_transported"]


# --------------------------------------------------------------------------
# The implicit diffusion step (one station interval)
# --------------------------------------------------------------------------
def spanwise_diffusion_step(chi, q, dm, weight, mu_r, *, xp=None):
    """One backward-Euler ``m``-step of the section 3.6 operator across the
    span, for one or more fields sharing the same operator.

    Parameters
    ----------
    chi : ``(n_sl,)`` or ``(k, n_sl)`` field value(s) at the station.
    q : ``(n_sl,)`` spanwise arc positions (strictly increasing).
    dm : scalar meridional step of the interval [m].
    weight : ``(n_sl,)`` mass-flux weight ``w = r (1-B) rho Vm`` (section 3.2).
    mu_r : ``(n_sl,)`` nodal ``mu_mix * r``; averaged to faces internally.

    Returns the diffused ``chi`` (same shape). Zero-flux at both walls, so the
    mass-flux-weighted total ``sum w_i dq_i chi_i`` is conserved to machine
    precision. A single node (``n_sl = 1``, Tier-1 meanline) has no spanwise
    neighbour and is returned unchanged -- a topology branch, not a flow one.
    """
    xp = get_xp(xp)
    n = q.shape[0]
    if n == 1:
        return chi

    dq_face = q[1:] - q[:-1]                       # (n-1,) node gaps
    # Control-volume widths (dual mesh): half-gaps summed at each node.
    dqi = xp.concatenate([
        0.5 * dq_face[:1],
        0.5 * (dq_face[1:] + dq_face[:-1]),
        0.5 * dq_face[-1:]])
    # Face diffusivity kappa = mu_mix r, harmonic-free arithmetic face mean.
    kappa_face = 0.5 * (mu_r[1:] + mu_r[:-1])
    cf = kappa_face / dq_face                      # (n-1,) face conductances
    cap = weight * dqi / dm                         # (n,) time-capacity

    zero = xp.zeros((1,), dtype=cf.dtype)
    cf_left = xp.concatenate([zero, cf])            # C_{i-1/2}, 0 at wall 0
    cf_right = xp.concatenate([cf, zero])           # C_{i+1/2}, 0 at wall 1
    main = cap + cf_left + cf_right
    # Symmetric tridiagonal A = diag(main) - offdiag(cf); implicit Euler.
    a = xp.diag(main) - xp.diag(cf, 1) - xp.diag(cf, -1)

    rhs = cap * chi                                 # broadcasts over (k, n)
    sol = xp.linalg.solve(a, xp.swapaxes(rhs, -1, 0) if rhs.ndim > 1 else rhs)
    return xp.swapaxes(sol, -1, 0) if rhs.ndim > 1 else sol


# --------------------------------------------------------------------------
# Marching the operator over a whole solve (lagged field refresh)
# --------------------------------------------------------------------------
def mix_transported(transported: TransportFields, *, m, r, blockage, rho, vm,
                    mu_mix, strength, xp=None) -> TransportFields:
    """Apply section 3.6 mixing to swept fields, marching station by station.

    Per interval ``j -> j+1``: take the transport increment the sweep already
    produced (``chi[:, j+1] - chi[:, j]``), add it to the *mixed* upstream
    profile, then diffuse across the span at ``j+1`` (backward Euler, one
    step). This is "transport then mix" per interval (section 3.6) without
    disturbing the pure sweep. ``strength`` is ``FidelityConfig.mixing_term``
    (0 returns the input unchanged -- Tier 1/2 and un-opted Tier 3).

    All coefficient arrays are ``(n_sl, n_qo)`` from the current iterate
    (lagged, AD-4). Returns new :class:`TransportFields`; inputs are untouched.
    """
    xp = get_xp(xp)
    n_sl, n_qo = transported.h0.shape
    if strength == 0.0 or n_sl == 1:
        return transported

    weight = r * (1.0 - blockage) * rho * vm        # section 3.2 mass flux
    mu_r = strength * mu_mix * r
    fields = (transported.h0, transported.s, transported.rvt)
    cols = [[f[:, 0] for f in fields]]              # station 0: inlet, unmixed
    for j in range(n_qo - 1):
        dm = xp.mean(m[:, j + 1] - m[:, j])         # interval meridional step
        incr = [f[:, j + 1] - f[:, j] for f in fields]
        upstream = xp.stack([c + d for c, d in zip(cols[-1], incr)])
        mixed = spanwise_diffusion_step(
            upstream, r[:, j + 1], dm, weight[:, j + 1], mu_r[:, j + 1], xp=xp)
        cols.append([mixed[k] for k in range(len(fields))])

    stacked = [xp.stack([c[k] for c in cols], axis=1) for k in range(3)]
    return TransportFields(h0=stacked[0], s=stacked[1], rvt=stacked[2])


# --------------------------------------------------------------------------
# Default mixing coefficient (Gallimore form)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GallimoreMixing:
    """Default section 3.6 coefficient (Gallimore-Cumpsty turbulent mixing).

    ``mu_mix = c_mix * rho * Vm * r`` -- a dynamic eddy-diffusivity whose
    effective per-metre spanwise diffusion length is ``c_mix * r / (1-B)``
    (the operator's ``mu_mix r / (r(1-B)rho Vm)``). ``c_mix`` sets the mixing
    intensity; the Gallimore value corresponds to a few-percent non-dimensional
    diffusivity. **[VERIFY the form and c_mix against a library calibration]**;
    a spanwise-velocity (Adkins-Smith) alternative is the recorded refinement.
    """

    c_mix: float = 0.01

    def mu_mix(self, flow) -> Array:
        """Nodal ``mu_mix`` from a flow view exposing ``rho``, ``vm``, ``r``."""
        return self.c_mix * flow.rho * flow.vm * flow.r
