"""V2 regression: swirl-free curved annulus, full Tier 3 with repositioning
(Theory Manual section 9.2; tolerances and caveats in Appendix C.2).

Reference: planar-limit concentric solution (slcflow.verification). Two
measured facts shape the assertions (M3-2, recorded in C.2):

  * The comparison window is the CENTRAL THIRD of the bend: the spline
    end-condition error at the inflow/outflow stations contaminates a fixed
    *physical* length, so a fixed station-count exclusion anti-converges
    under refinement.
  * The central-third disagreement floors at ~1e-2 in Vm, independent of
    grid (5,5)->(17,13) AND of rc (400 vs 4000): it is the
    boundary-development difference between the solved problem (flow
    boundaries at the bend ends) and the fully-developed reference vortex,
    not solver error. Discretization-order evidence for the machinery lives
    in the M1 frozen-streamline gate and V1d.

Provenance: M3 sub-step 2, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.assembly import ResidualAssembler
from slcflow.verification import V2CurvedAnnulus

SPAN = 0.3


def central_errors(case, res, exact):
    """(pos_err/span, vm_rel_err) over the central third of the bend."""
    psi = res.frozen.topology.psi
    q_star = exact.q_of_psi(psi)
    n_qo = res.frozen.topology.n_qo
    theta = np.linspace(0.0, 1.0, n_qo)
    central = [j for j, t in enumerate(theta) if 1 / 3 <= t <= 2 / 3]
    e_pos = max(np.max(np.abs(res.fields.q[:, j] - q_star))
                for j in central) / SPAN
    e_vm = max(np.max(np.abs(res.fields.vm[:, j] / exact.vm_of_q(q_star)
                             - 1.0)) for j in central)
    return e_pos, e_vm


@pytest.fixture(scope="module")
def v2_result():
    case = V2CurvedAnnulus()
    return case, case.exact(), case.solve(n_sl=9)


def test_v2a_central_agreement_and_orientation(v2_result):
    # Section 9.2: Tier-3 coupled solution vs. the planar-limit concentric
    # reference over the central third (Appendix C.2 tolerances; measured
    # vm ~1.0e-2, pos ~2.6e-3).
    case, exact, res = v2_result
    assert res.converged
    e_pos, e_vm = central_errors(case, res, exact)
    assert e_vm < 2e-2, f"central Vm err {e_vm:.2e}"
    assert e_pos < 6e-3, f"central position err {e_pos:.2e}"
    # Vm increases toward the bend center (q = 0 is the OUTER-bend wall:
    # the A.1.1 orientation flip exercised end-to-end, AD-9).
    assert res.frozen.topology.flowpath.q_origin_wall == 1
    assert np.all(np.diff(res.fields.vm, axis=0) > 0.0)


def test_v2a_residual_contract(v2_result):
    # The driver's answer satisfies the assembler's residual vector
    # (integration + continuity + repositioning verified together).
    case, exact, res = v2_result
    r = ResidualAssembler(res.frozen).residual(res.x)
    assert np.max(np.abs(r)) / (exact.mdot / (2 * np.pi)) < 1e-7


def test_v2b_planar_limit_family(v2_result):
    # The solver must approach the planar-limit reference as rc grows:
    # measured central Vm err 8.9e-2 (rc=4) -> 1.4e-2 (rc=40) -> 1.0e-2
    # (rc=400): the axisymmetric machinery reproduces the O(R/rc) physics.
    case400, exact400, res400 = v2_result
    _, e400 = central_errors(case400, res400, exact400)
    errs = {400.0: e400}
    for rc in (4.0, 40.0):
        case = V2CurvedAnnulus(rc=rc)
        errs[rc] = central_errors(case, case.solve(n_sl=9), case.exact())[1]
    assert errs[4.0] > 4.0 * errs[400.0]
    assert errs[40.0] < errs[4.0] / 3.0


def test_v2c_reference_floor_is_grid_independent():
    # Appendix C.2: the residual central-third disagreement is a reference
    # modeling floor, not discretization -- coarse and fine grids must both
    # sit at or below the same ~2e-2 bound (measured 8.2e-3 / 8.6e-3).
    for n_sl, n_st in ((5, 5), (17, 13)):
        case = V2CurvedAnnulus()
        res = case.solve(n_sl=n_sl, n_stations=n_st)
        assert res.converged, f"({n_sl},{n_st}): {res.status}"
        _, e_vm = central_errors(case, res, case.exact())
        assert e_vm < 2e-2, f"({n_sl},{n_st}): central Vm err {e_vm:.2e}"
