"""Streamwise transport of h0, s, rVt: conservation relations and work/loss
distribution schedules (Theory Manual sections 3.3-3.5; ARCH-2).

The spanwise mixing operator (section 3.6) also lives here (``mixing.py``,
M8): an implicit spanwise diffusion of h0/s/rVt applied in the driver's
lagged field refresh, off the pure residual path.
"""
from .mixing import (GallimoreMixing, mix_transported,
                     spanwise_diffusion_step)
from .schedules import (DistributionSchedule, SmoothRampSchedule,
                        assert_valid_schedule)
from .streamwise import (TransportFields, TransportStep, apply_step,
                         rothalpy, row_steps, sweep)

__all__ = [
    "DistributionSchedule",
    "GallimoreMixing",
    "SmoothRampSchedule",
    "assert_valid_schedule",
    "mix_transported",
    "spanwise_diffusion_step",
    "TransportFields",
    "TransportStep",
    "apply_step",
    "rothalpy",
    "row_steps",
    "sweep",
]
