"""Streamwise transport of the per-streamtube fields h0, s, rVt
(Theory Manual sections 3.3-3.5; ARCH-2 ``transport`` scope).

One universal station-to-station update covers ducts, stators, and rotors
without code branching (AD-6, and the section 8 degeneracy requirement):

    rVt_out = rVt_in                     (duct: angular momentum, section 3.4)
              or the closure/schedule-prescribed value (blade rows)
    h0_out  = h0_in + omega * (rVt_out - rVt_in)   (rothalpy, section 3.3)
    s_out   = s_in + delta_s                       (section 3.5)

With ``omega = 0`` the h0 update degenerates to the stationary-row/duct rule
``h0 = const``; with ``rVt_out = rVt_in`` both work terms vanish. Rotor work
therefore enters exclusively through the swirl target and rothalpy
conservation, per section 4.2.

Spanwise mixing (section 3.6) is deferred to M8 (ARCH-8/ARCH-9). Its
per-interval increment ``delta_s_mix`` enters additively through
``TransportStep.delta_s`` when it lands; the section 3.5 update above is
already written as the sum of row and mixing contributions.

All functions here are pure (AD-3 discipline): arrays in, arrays out, no
mutation, no exceptions on flow arrays (AD-10) -- ``ConfigError`` is raised
only on frozen-topology inputs at construction boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass

from .._namespace import get_xp
from ..errors import ConfigError
from ..fluid.base import Array
from .schedules import SmoothRampSchedule

__all__ = ["TransportStep", "TransportFields", "rothalpy", "apply_step",
           "sweep", "row_steps"]


def rothalpy(h0, rvt, omega):
    """Rothalpy ``I = h0 - omega * rVt`` (section 3.3).

    The quantity conserved per streamtube through a rotor row for adiabatic
    flow with casing windage neglected; the verification currency for the
    section 3.3 transport step. ``omega = 0`` reduces I to h0.
    """
    return h0 - omega * rvt


@dataclass(frozen=True)
class TransportStep:
    """Transport of (h0, s, rVt) across one station interval, into the
    downstream station (sections 3.3-3.5).

    The default-constructed step is the adiabatic, frictionless duct:
    ``omega = 0``, angular momentum conserved, no entropy source.

    Parameters
    ----------
    omega : shaft speed of the row governing this interval [rad/s]; 0 for
        ducts and stationary rows (section 3.3).
    rvt : target rVt at the downstream station [m^2/s] -- the lagged closure
        output (AD-4) or a scheduled in-blade value (section 3.4). ``None``
        conserves rVt (duct rule; this is topology data, not a flow branch).
    delta_s : entropy increment charged over the interval [J/(kg K)]: the
        scheduled row loss plus, later, the mixing increment (section 3.5).
    """

    omega: float = 0.0
    rvt: Array | None = None
    delta_s: Array = 0.0


@dataclass(frozen=True)
class TransportFields:
    """Swept transported fields, all ``(n_sl, n_qo)`` (AD-2 struct-of-arrays)."""

    h0: Array
    s: Array
    rvt: Array


def apply_step(h0, s, rvt, step: TransportStep):
    """One station-to-station update of (h0, s, rvt) per sections 3.3-3.5.

    Returns the downstream ``(h0, s, rvt)`` tuple. The h0 update is rothalpy
    conservation in Euler-work form, ``h0_out = h0_in + omega * (rVt_out -
    rVt_in)`` (section 3.3); the entropy update is ``s_out = s_in + delta_s``
    (section 3.5). Pure: inputs are never mutated, so returning the input
    ``rvt`` object for a duct step is safe (AD-6).
    """
    rvt_out = rvt if step.rvt is None else step.rvt
    h0_out = h0 + step.omega * (rvt_out - rvt)
    s_out = s + step.delta_s
    return h0_out, s_out, rvt_out


def sweep(h0_inlet, s_inlet, rvt_inlet, steps, *, xp=None) -> TransportFields:
    """March the transport relations through all station intervals
    (the section 6.2.1 initialization sweep and the per-outer-iterate
    field update of section 6.2.2).

    Parameters
    ----------
    h0_inlet, s_inlet, rvt_inlet : fields at station 0, ``(n_sl,)`` or
        broadcastable scalars.
    steps : sequence of ``len = n_qo - 1`` :class:`TransportStep`, one per
        station interval, upstream to downstream.

    Returns
    -------
    :class:`TransportFields` with ``n_qo = len(steps) + 1`` columns; column
    ``j`` holds the fields at station ``j``. The spanwise shape is the joint
    broadcast of all inputs; if everything is scalar the result is
    ``(1, n_qo)`` (the Tier-1 single-streamline degenerate shape).
    """
    xp = get_xp(xp)
    h0_cols, s_cols, rvt_cols = [h0_inlet], [s_inlet], [rvt_inlet]
    for step in steps:
        h0, s, rvt = apply_step(h0_cols[-1], s_cols[-1], rvt_cols[-1], step)
        h0_cols.append(h0)
        s_cols.append(s)
        rvt_cols.append(rvt)

    # The spanwise shape is determined jointly across the three fields: a
    # field may stay scalar through the whole sweep (e.g. uniform s with
    # scalar delta_s) while another is per-streamline, so broadcast every
    # column of every field to the common shape before stacking on axis 1.
    n_st = len(h0_cols)
    cols = [xp.atleast_1d(c)
            for c in xp.broadcast_arrays(*h0_cols, *s_cols, *rvt_cols)]
    return TransportFields(
        h0=xp.stack(cols[:n_st], axis=1),
        s=xp.stack(cols[n_st:2 * n_st], axis=1),
        rvt=xp.stack(cols[2 * n_st:], axis=1),
    )


def row_steps(*, omega, rvt_le, rvt_te, delta_s_row, t_stations=(1.0,),
              work_schedule=None, loss_schedule=None):
    """Build the :class:`TransportStep` sequence for one blade row
    (sections 3.4, 3.5, B.5.1).

    Parameters
    ----------
    omega : row shaft speed [rad/s]; 0 for stators.
    rvt_le : rVt arriving at EDGE_LE, ``(n_sl,)``. Must equal the swept field
        at the LE station for the scheduled in-blade profile to be
        consistent; the TE value is exact regardless (the work schedule
        reaches exactly 1 there).
    rvt_te : exit rVt at EDGE_TE from the swirl closure (section 3.4, AD-4).
    delta_s_row : per-streamtube row entropy rise from the loss closure,
        already converted per Appendix B (section 3.5).
    t_stations : meridional fractions of the stations downstream of EDGE_LE
        through EDGE_TE, strictly increasing in (0, 1] with the last exactly
        1.0. The default ``(1.0,)`` is the edge-only row (single LE -> TE
        step). Frozen topology (AD-8) -- validated here with ConfigError.
    work_schedule, loss_schedule : :class:`DistributionSchedule` for the
        in-blade rVt ramp and delta_s distribution. Defaults: smooth ramp
        for work (section 3.4); loss follows the work schedule (B.5.1).

    Returns
    -------
    tuple of :class:`TransportStep`, one per interval from EDGE_LE onward.
    """
    ts = tuple(float(t) for t in t_stations)
    if not ts or ts[-1] != 1.0:
        raise ConfigError(f"t_stations must end exactly at 1.0, got {ts}")
    if any(t <= 0.0 for t in ts) or any(b <= a for a, b in zip(ts, ts[1:])):
        raise ConfigError(
            f"t_stations must be strictly increasing in (0, 1], got {ts}")
    if work_schedule is None:
        work_schedule = SmoothRampSchedule()
    if loss_schedule is None:
        loss_schedule = work_schedule  # loss follows the work schedule (B.5.1)

    steps = []
    g_prev = 0.0
    for t in ts:
        f = work_schedule(t)
        g = loss_schedule(t)
        # (1-f)/f form hits rvt_le / rvt_te exactly at f = 0 / 1 (section 3.4).
        steps.append(TransportStep(
            omega=omega,
            rvt=(1.0 - f) * rvt_le + f * rvt_te,
            delta_s=(g - g_prev) * delta_s_row,
        ))
        g_prev = g
    return tuple(steps)
