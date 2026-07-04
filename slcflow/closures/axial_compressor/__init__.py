"""Axial-compressor correlation set (Theory Manual section 7.1): Lieblein
incidence/deviation + diffusion-factor profile loss, bundled as a named
CorrelationSet with provenance. Koch-Smith / Aungier endwall & shock loss
components extend the set at V5 calibration time (recorded deferral)."""
from ..interfaces import CorrelationSet
from .lieblein import (LieblienSwirl, deviation_slope, reference_deviation,
                       reference_incidence)
from .loss import LieblienLoss, equivalent_diffusion, wake_momentum_thickness

LIEBLEIN_NACA65 = CorrelationSet(
    name="lieblein-naca65",
    swirl=LieblienSwirl(),
    loss=LieblienLoss(),
    provenance="Aungier analytic fits to NASA SP-36 (incidence/deviation) "
               "+ Lieblein 1959 equivalent-diffusion profile loss "
               "[VERIFY against library copies]",
)

__all__ = ["LIEBLEIN_NACA65", "LieblienLoss", "LieblienSwirl",
           "deviation_slope", "equivalent_diffusion", "reference_deviation",
           "reference_incidence", "wake_momentum_thickness"]
