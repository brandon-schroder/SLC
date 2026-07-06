"""Centrifugal (radial) correlation set (Theory Manual section 7.1): the
Aungier / Galvas loss family + Wiesner / von Backstrom slip. Lands in
reviewed M7 steps -- the Wiesner slip closure first (M7-1), then the loss
components and the bundled ``CorrelationSet`` (M7-2), INBLADE support and the
in-blade force model (M7-3), and V7 (M7-4)."""
from .wiesner import WiesnerSlip, wiesner_slip

__all__ = ["WiesnerSlip", "wiesner_slip"]
