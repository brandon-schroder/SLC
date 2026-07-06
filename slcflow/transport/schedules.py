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

__all__ = ["DistributionSchedule", "SmoothRampSchedule",
           "assert_valid_schedule"]


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


def assert_valid_schedule(schedule, *, n=2001, slope_tol=1e-3, tol=1e-9):
    """Contract check for a :class:`DistributionSchedule` (section 3.4 /
    7.3.4 -- the smoothness-sweep gate applied to schedules).

    Verifies, by a domain sweep: finiteness; ``f(0) = 0`` and ``f(1) = 1``
    exactly; monotone non-decreasing on ``[0, 1]``; **zero end slopes**
    (``f'(0) = f'(1) = 0`` so the composite in-blade field stays C1 across
    the LE/TE joins to the constant upstream/downstream regions); and
    off-range saturation (constant 0 below 0, 1 above 1 -- never extrapolate).
    Raises ``AssertionError`` on violation. Config-boundary helper (M4
    carryover): call it on any non-default schedule before a solve."""
    import numpy as _np  # ad6: allow -- config-boundary contract helper, not the residual path

    t = _np.linspace(0.0, 1.0, n)
    f = _np.asarray(schedule(t), dtype=float)
    dt = t[1] - t[0]
    assert _np.all(_np.isfinite(f)), "schedule produced non-finite values"
    assert abs(float(f[0])) <= tol, f"f(0) must be 0, got {float(f[0])}"
    assert abs(float(f[-1]) - 1.0) <= tol, f"f(1) must be 1, got {float(f[-1])}"
    assert _np.all(_np.diff(f) >= -tol), "schedule must be non-decreasing"
    assert abs(float(f[1] - f[0]) / dt) <= slope_tol, "f'(0) must be 0"
    assert abs(float(f[-1] - f[-2]) / dt) <= slope_tol, "f'(1) must be 0"
    lo = float(_np.asarray(schedule(_np.array([-0.5])), dtype=float)[0])
    hi = float(_np.asarray(schedule(_np.array([1.5])), dtype=float)[0])
    assert abs(lo) <= tol, f"schedule must saturate to 0 below 0, got {lo}"
    assert abs(hi - 1.0) <= tol, f"schedule must saturate to 1 above 1, got {hi}"
