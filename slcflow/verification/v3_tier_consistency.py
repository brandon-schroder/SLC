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
closure-evaluation error") is bound at M4 via :func:`v3_tier1_pair` and
:func:`mass_averaged_vm`: with the n_sl = 1 meanline now assembling (the
one-point area rule, section 8), the meanline Vm must equal the mass-flux-
weighted span average of the Tier-2 field. On the prescribed (closure-free)
V1 cases the closure-evaluation error vanishes, so what remains is the
meanline quadrature residue (Appendix C.4).

Problem definitions are the V1 cases; this module just pairs them with the
tier configurations.
"""
from __future__ import annotations

import numpy as np  # verification layer: mass-average reduction  # ad6: allow

from ..types import FidelityConfig
from .v1_analytic_ree import V1ForcedVortex, V1FreeVortex

__all__ = ["v3_case_pair", "v3_tier1_pair", "mass_averaged_vm"]


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


def v3_tier1_pair(case=None, n_sl_ref=9):
    """Solve one V1-class case at Tier 1 (meanline, ``n_sl = 1``) and Tier 2
    (``n_sl = n_sl_ref``) — the section 8 Tier-1 consistency clause.

    ``FidelityConfig.tier1()`` is the Tier-2 flag set (types.py); the tier
    distinction is entirely the single mid-``psi`` streamline (section 8),
    which is exactly what this pairing exercises.
    """
    if case is None:
        case = V1FreeVortex.compressible()
    res1 = case.solve(1, fidelity=FidelityConfig.tier1())
    res_ref = case.solve(n_sl_ref, fidelity=FidelityConfig.tier2())
    return res1, res_ref


def mass_averaged_vm(res):
    """Per-station mass-flux-weighted span average of ``Vm`` (weight
    ``rho Vm cos(eps) r``, section 3.2) — the "mass-averaged result" the
    Tier-1 meanline is required to match. A single-node (Tier-1) field
    returns itself."""
    f = res.fields
    out = []
    for j in range(res.frozen.n_qo):
        if f.vm[:, j].size == 1:
            out.append(float(f.vm[0, j]))
            continue
        w = (f.rho[:, j] * f.vm[:, j] * np.cos(f.metrics.eps[:, j])
             * f.metrics.r[:, j])
        out.append(float(np.trapezoid(w * f.vm[:, j], f.q[:, j])
                         / np.trapezoid(w, f.q[:, j])))
    return np.array(out)


# Re-exported so the V3 test reads as a ladder entry without reaching into
# the V1 module for its cases.
FreeVortex = V1FreeVortex
ForcedVortex = V1ForcedVortex
