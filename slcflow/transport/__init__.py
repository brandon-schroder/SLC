"""Streamwise transport of h0, s, rVt: conservation relations and work/loss
distribution schedules (Theory Manual sections 3.3-3.5; ARCH-2).

The spanwise mixing operator (section 3.6) also belongs to this package but
is deliberately deferred to milestone M8 (ARCH-8, ARCH-9); its entropy
increment slots additively into ``TransportStep.delta_s`` and its own field
updates will reuse the same sweep contract.
"""
from .schedules import DistributionSchedule, SmoothRampSchedule
from .streamwise import (TransportFields, TransportStep, apply_step,
                         rothalpy, row_steps, sweep)

__all__ = [
    "DistributionSchedule",
    "SmoothRampSchedule",
    "TransportFields",
    "TransportStep",
    "apply_step",
    "rothalpy",
    "row_steps",
    "sweep",
]
