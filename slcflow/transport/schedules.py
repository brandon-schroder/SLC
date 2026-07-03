"""In-blade distribution schedules (Theory Manual sections 3.4, 3.5, B.5.1).

A blade row's work (rVt change) and loss (delta_s) are distributed along its
meridional chord by a *cumulative fraction schedule* ``f(t)``, with ``t`` the
normalized meridional fraction through the row (0 at EDGE_LE, 1 at EDGE_TE).
Section 3.4 requires the resulting rVt(m) to be C1 in ``m``. Because the
transported fields are constant upstream of the LE and downstream of the TE,
C1 continuity of the *composite* field additionally requires the schedule's
end slopes to vanish; that stronger condition is part of the contract here.

Loss distribution defaults to the work schedule itself (B.5.1) unless the
loss source is explicitly local -- callers override ``loss_schedule`` in
:func:`slcflow.transport.streamwise.row_steps` for that case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..closures.smoothmath import smoothstep
from ..fluid.base import Array

__all__ = ["DistributionSchedule", "SmoothRampSchedule"]


@runtime_checkable
class DistributionSchedule(Protocol):
    """Cumulative distribution fraction ``f(t)`` over a blade row (section 3.4).

    Contract (mandatory, tested per correlation-style domain sweep):

      * ``f(0) = 0`` and ``f(1) = 1`` exactly;
      * monotone non-decreasing on [0, 1];
      * at least C1 in ``t`` over the whole real line, with ``f'(0) = f'(1)
        = 0`` so the composite field stays C1 across the LE/TE joins to the
        constant upstream/downstream regions;
      * constant (0 / 1) outside [0, 1] -- evaluation off-range must saturate,
        never extrapolate (section 7.3 discipline applied to schedules).
    """

    def __call__(self, t: Array, *, xp=None) -> Array: ...


@dataclass(frozen=True)
class SmoothRampSchedule:
    """Default section 3.4 schedule: smooth monotone ramp over the row chord.

    Quintic smoothstep on [0, 1] (C2 including the joins, zero first and
    second derivatives at both edges -- see
    :func:`slcflow.closures.smoothmath.smoothstep`), which satisfies every
    clause of the :class:`DistributionSchedule` contract with no parameters.
    """

    def __call__(self, t, *, xp=None):
        return smoothstep(t, 0.0, 1.0, xp=xp)
