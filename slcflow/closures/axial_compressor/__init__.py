"""Axial-compressor correlation set (Theory Manual section 7.1): Lieblein
incidence/deviation + diffusion-factor profile loss + Howell endwall
(secondary + annulus) and Lakshminarayana tip-clearance loss, bundled as a
named CorrelationSet with provenance. The endwall/clearance components (added
2026-07, docs/references/HOWELL.md) are the deferral this docstring previously
named; a compressor SHOCK loss component (for transonic V5 cases) remains
deferred."""
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
