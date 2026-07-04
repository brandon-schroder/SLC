"""V3 — Multi-fidelity tier consistency (Theory Manual sections 8, 9.3;
AD-1; ARCH-7).

The section 8 consistency requirement, run as a standing regression: for a
free-vortex, uniform-inlet case, Tier 2 and Tier 3 must agree to
discretization tolerance. On the straight annulus every Tier-3-exclusive
term carries a factor cos(eps), sin(eps), or kappa_m that is exactly zero
at the fixed point (A.5 check 3), so agreement is expected at solver — not
merely discretization — tolerance; anything looser indicates hidden tier
branching, which AD-1 forbids by construction.

The Tier-1 clause ("Tier 1 must equal their mass-averaged result to
closure-evaluation error") requires the n_sl = 1 machine facade and
closures, and joins this case at M4 (recorded deferral; see the
ResidualAssembler n_sl >= 2 constraint).

Problem definitions are the V1 cases; this module just pairs them with the
tier configurations.
"""
from __future__ import annotations

from ..types import FidelityConfig
from .v1_analytic_ree import V1ForcedVortex, V1FreeVortex

__all__ = ["v3_case_pair"]


def v3_case_pair(case=None, n_sl=9):
    """Solve one V1-class case at Tier 2 and Tier 3 on the same grid.

    Default case: the compressible free vortex (the section 8 wording);
    callers may pass e.g. ``V1ForcedVortex()`` — any straight-annulus case
    where the Tier-3 terms vanish at the fixed point qualifies.
    """
    if case is None:
        case = V1FreeVortex.compressible()
    res2 = case.solve(n_sl, fidelity=FidelityConfig.tier2())
    res3 = case.solve(n_sl, fidelity=FidelityConfig.tier3())
    return res2, res3


# Re-exported so the V3 test reads as a ladder entry without reaching into
# the V1 module for its cases.
FreeVortex = V1FreeVortex
ForcedVortex = V1ForcedVortex
