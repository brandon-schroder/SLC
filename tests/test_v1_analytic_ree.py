"""V1 regression: analytic REE end-to-end through the classical driver
(Theory Manual section 9.1; tolerances recorded in Appendix C.1).

The M2 acceptance gate: Tier 2 passing V1 (free/forced vortex, exact Vm(r))
plus the grid-convergence order check (expect 2nd). Reference solutions come
from slcflow.verification (dense 1-D machinery independent of the kernel's
nodal quadrature). Written in the same session as the implementation
(provenance: M2 sub-step 4).
"""
import numpy as np
import pytest

from slcflow.assembly import ResidualAssembler
from slcflow.verification import V1ForcedVortex, V1FreeVortex

SPAN = 0.3  # r1 - r0 of the default V1 annulus


def residual_scale(res):
    return res.frozen.spec.mdot / (2.0 * np.pi)


def position_radii(res):
    """Solver nodal radii on the first q-o (cylinder: r = r0 + q)."""
    return res.fields.metrics.r[:, 0]


# --------------------------------------------------------------------------
# V1a: incompressible-limit free vortex
# --------------------------------------------------------------------------
def test_v1a_incompressible_free_vortex():
    case = V1FreeVortex.incompressible()
    res = case.solve(n_sl=9)
    assert res.converged
    # Section A.5 case 1: uniform Vm to solver tolerance.
    np.testing.assert_allclose(res.fields.vm, res.fields.vm[0, 0], rtol=1e-8)
    # Incompressible limit: closed-form area-rule radii (Appendix C.1
    # tolerance covers the Mm^2 ~ 5e-4 compressibility residue).
    topo_psi = res.frozen.topology.psi
    r_exact = np.sqrt(topo_psi * (case.r1**2 - case.r0**2) + case.r0**2)
    np.testing.assert_allclose(position_radii(res), r_exact,
                               atol=1e-3 * SPAN)
    # And the tight check against the dense reference.
    exact = case.exact()
    assert res.fields.vm[0, 0] == pytest.approx(exact.vm0, rel=1e-6)
    np.testing.assert_allclose(position_radii(res),
                               exact.r_of_psi(topo_psi), atol=1e-5 * SPAN)


# --------------------------------------------------------------------------
# V1b: compressible free vortex
# --------------------------------------------------------------------------
def test_v1b_compressible_free_vortex():
    case = V1FreeVortex.compressible()
    exact = case.exact()
    res = case.solve(n_sl=9)
    assert res.converged
    np.testing.assert_allclose(res.fields.vm, res.fields.vm[0, 0], rtol=1e-8)
    # Vm level set by compressible continuity: dense-reference agreement to
    # the n_sl = 9 nodal-trapezoid discretization error (measured 1.2e-5;
    # Appendix C.1) -- NOT machine precision, the reference quadrature is
    # deliberately independent of the solver's.
    assert res.fields.vm[0, 0] == pytest.approx(exact.vm0, rel=5e-5)
    np.testing.assert_allclose(position_radii(res),
                               exact.r_of_psi(res.frozen.topology.psi),
                               atol=2e-4 * SPAN)
    # The full residual vector vanishes at the answer (integration,
    # continuity, and repositioning verified together, section 9.1).
    r = ResidualAssembler(res.frozen).residual(res.x)
    assert np.max(np.abs(r)) / residual_scale(res) < 1e-7


# --------------------------------------------------------------------------
# V1c: forced vortex
# --------------------------------------------------------------------------
def test_v1c_forced_vortex_profile_and_positions():
    case = V1ForcedVortex()
    exact = case.exact()
    res = case.solve(n_sl=17)
    assert res.converged
    psi = res.frozen.topology.psi
    r_star = exact.r_of_psi(psi)
    # Positions: mass-fraction radii against the dense reference.
    np.testing.assert_allclose(position_radii(res), r_star, atol=2e-3 * SPAN)
    # Profile: exact Vm(r) family evaluated at the exact radii.
    np.testing.assert_allclose(res.fields.vm[:, 0],
                               case.vm_family(r_star, exact.vm0), rtol=2e-3)
    # Hub level from continuity.
    assert res.fields.vm[0, 0] == pytest.approx(exact.vm0, rel=2e-3)
    r = ResidualAssembler(res.frozen).residual(res.x)
    assert np.max(np.abs(r)) / residual_scale(res) < 1e-7


# --------------------------------------------------------------------------
# V1d: grid-convergence order (section 9.1: expect 2nd)
# --------------------------------------------------------------------------
def test_v1d_grid_convergence_second_order():
    case = V1ForcedVortex()
    exact = case.exact()
    errs = []
    for n_sl in (5, 9, 17):
        res = case.solve(n_sl=n_sl)
        assert res.converged
        psi = res.frozen.topology.psi
        e_pos = np.max(np.abs(position_radii(res) - exact.r_of_psi(psi)))
        e_vm = np.max(np.abs(res.fields.vm[:, 0]
                             / case.vm_family(exact.r_of_psi(psi), exact.vm0)
                             - 1.0))
        errs.append(max(e_pos / SPAN, e_vm))
    order = np.log2(errs[0] / errs[2]) / 2.0
    assert order > 1.7, f"observed order {order:.2f}, errors {errs}"
