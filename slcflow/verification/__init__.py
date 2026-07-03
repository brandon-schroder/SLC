"""Verification-ladder problem definitions V1..V9 (Theory Manual section 9;
ARCH-7). Importable cases with independent reference solutions; tests/ binds
them to the Appendix C tolerances as pytest regressions. May import anything;
nothing in the kernel imports this package."""
from .v1_analytic_ree import (V1Exact, V1ForcedVortex, V1FreeVortex,
                              annulus_topology)

__all__ = ["V1Exact", "V1ForcedVortex", "V1FreeVortex", "annulus_topology"]
