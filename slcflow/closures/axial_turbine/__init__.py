"""Axial-turbine correlation set (Theory Manual section 7.1): the
Kacker-Okapuu / Ainley-Mathieson family, bundled as a named CorrelationSet
with provenance. Lands in reviewed M6 steps — throat-based exit angle
(M6-1) + subsonic profile loss (M6-2) here; secondary / trailing-edge
(M6-3) and shock (M6-4) loss components extend the set."""
from ..interfaces import CorrelationSet
from .ainley import AinleyTurbineSwirl, throat_exit_angle
from .kacker_okapuu import (mach_profile_correction, profile_loss_am,
                           reynolds_correction, secondary_loss,
                           trailing_edge_zeta)
from .loss import KackerOkapuuLoss

KACKER_OKAPUU = CorrelationSet(
    name="kacker-okapuu",
    swirl=AinleyTurbineSwirl(),
    loss=KackerOkapuuLoss(),
    provenance="Ainley-Mathieson throat exit angle + Kacker-Okapuu 1982 "
               "subsonic profile loss (Mach Kp + Reynolds fRe); secondary/"
               "trailing-edge/shock components land at M6-3..M6-4 "
               "[VERIFY against library copies]",
)

__all__ = ["KACKER_OKAPUU", "AinleyTurbineSwirl", "KackerOkapuuLoss",
           "mach_profile_correction", "profile_loss_am",
           "reynolds_correction", "secondary_loss", "trailing_edge_zeta",
           "throat_exit_angle"]
