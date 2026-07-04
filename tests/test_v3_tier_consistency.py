"""V3 regression: multi-fidelity tier consistency (Theory Manual sections 8,
9.3; AD-1; tolerances in Appendix C.4).

Standing regression per section 8: Tier 2 and Tier 3 solve the free-vortex
(and forced-vortex) uniform-inlet straight-annulus cases on the same grid
and must agree. Measured at M3-4: agreement is BIT-FOR-BIT (the Tier-3
exclusive terms multiply exactly-zero kappa/eps through the collinear-node
spline, and the relaxation paths coincide at the cap) — asserted here at
1e-10, far below discretization, so any hidden tier branching or term
leakage fails loudly. The Tier-1 mass-average clause joins at M4 (machine
facade + closures; see the V3 module docstring).

Provenance: M3 sub-step 4, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.verification import v3_case_pair
from slcflow.verification.v3_tier_consistency import ForcedVortex, FreeVortex


@pytest.mark.parametrize("case", [None, ForcedVortex()],
                         ids=["free_vortex", "forced_vortex"])
def test_v3_tier2_equals_tier3_on_straight_annulus(case):
    # Section 8 consistency requirement / A.5 check 3 (AD-1 by
    # construction): identical fields, positions, and thermodynamics.
    res2, res3 = v3_case_pair(case)
    assert res2.converged and res3.converged
    np.testing.assert_allclose(res3.fields.vm, res2.fields.vm, rtol=1e-10)
    np.testing.assert_allclose(res3.fields.q, res2.fields.q, atol=1e-10)
    np.testing.assert_allclose(res3.fields.rho, res2.fields.rho, rtol=1e-10)
    np.testing.assert_allclose(res3.x, res2.x, rtol=1e-10, atol=1e-12)


def test_v3_is_not_vacuous():
    # Guard against the consistency test passing because the tiers were
    # never different: on a CURVED path the two tiers must disagree
    # substantially (the Tier-3 curvature physics is real; see
    # test_drivers_tier3 for the full contrast).
    from slcflow.verification import V2CurvedAnnulus
    from slcflow.types import FidelityConfig
    from slcflow.drivers import solve_classical
    from slcflow.transport import TransportFields
    from slcflow.types import MassFlowSpec

    case = V2CurvedAnnulus()
    exact = case.exact()
    topo = case.topology(9)
    inlet = TransportFields(h0=np.full(9, case.h0), s=np.full(9, case.s),
                            rvt=np.zeros(9))
    res2 = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(exact.mdot), inlet)
    assert res2.converged
    # Tier 2 on the bend: spanwise-uniform; the Tier-3 solution (V2 tests)
    # has a ~2.5x spanwise ratio. Uniformity here proves the flags gate
    # real physics rather than dead code.
    np.testing.assert_allclose(
        res2.fields.vm,
        np.broadcast_to(res2.fields.vm[0:1, :], res2.fields.vm.shape),
        rtol=1e-8)
