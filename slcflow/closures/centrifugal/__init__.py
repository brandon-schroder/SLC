"""Centrifugal (radial) correlation set (Theory Manual section 7.1): the
Aungier / Galvas loss family + Wiesner / von Backstrom slip, bundled as a
named CorrelationSet with provenance. Lands in reviewed M7 steps -- Wiesner
slip (M7-1) + representative internal loss (M7-2) here; INBLADE support and
the in-blade force model (M7-3) and V7 (M7-4) follow."""
from ..interfaces import CorrelationSet
from .loss import CentrifugalLoss, incidence_loss, skin_friction_loss
from .wiesner import WiesnerSlip, wiesner_slip

CENTRIFUGAL = CorrelationSet(
    name="centrifugal-wiesner-aungier",
    swirl=WiesnerSlip(),
    loss=CentrifugalLoss(),
    provenance="Wiesner slip + representative centrifugal internal loss "
               "(incidence + skin friction); blade-loading/clearance/"
               "disk-friction components extend the set at V7 calibration "
               "[VERIFY against library copies]",
)

__all__ = ["CENTRIFUGAL", "WiesnerSlip", "CentrifugalLoss", "wiesner_slip",
           "incidence_loss", "skin_friction_loss"]
