"""Axial-turbine correlation set (Theory Manual section 7.1): the
Kacker-Okapuu / Ainley-Mathieson family. M6 lands it in reviewed steps —
the throat-based exit-angle closure first (M6-1), then the profile /
secondary / trailing-edge / shock loss components and the bundled
``CorrelationSet`` (M6-2..M6-4)."""
from .ainley import AinleyTurbineSwirl, throat_exit_angle

__all__ = ["AinleyTurbineSwirl", "throat_exit_angle"]
