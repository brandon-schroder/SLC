"""Verification-ladder problem definitions V1..V9 (Theory Manual section 9;
ARCH-7). Importable cases with independent reference solutions; tests/ binds
them to the Appendix C tolerances as pytest regressions. May import anything;
nothing in the kernel imports this package."""
from .v1_analytic_ree import (V1Exact, V1ForcedVortex, V1FreeVortex,
                              annulus_topology)
from .v2_curved_annulus import V2CurvedAnnulus, V2Exact
from .v3_tier_consistency import (mass_averaged_vm, v3_case_pair,
                                  v3_tier1_pair)
from .v5_axial_compressor import V5AxialRotor, V5MultistageCompressor
from .v6_axial_turbine import V6AxialTurbine
from .v7_centrifugal import V7Centrifugal
from .v9_operability import V9Operability

__all__ = ["V1Exact", "V1ForcedVortex", "V1FreeVortex", "V2CurvedAnnulus",
           "V2Exact", "V5AxialRotor", "V5MultistageCompressor",
           "V6AxialTurbine", "V7Centrifugal", "V9Operability",
           "annulus_topology", "mass_averaged_vm", "v3_case_pair",
           "v3_tier1_pair"]
