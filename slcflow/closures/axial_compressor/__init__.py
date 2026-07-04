"""Axial-compressor correlation set (Theory Manual section 7.1): Lieblein
incidence/deviation now; the loss set (Koch-Smith / Aungier) is M4-4."""
from .lieblein import (LieblienSwirl, deviation_slope, reference_deviation,
                       reference_incidence)

__all__ = ["LieblienSwirl", "deviation_slope", "reference_deviation",
           "reference_incidence"]
