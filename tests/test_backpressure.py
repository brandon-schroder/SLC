"""Back-pressure residual form (Theory Manual section 6.6; ARCH-4.3; M5-3a).

The choke-proximal boundary condition: exit static pressure is specified at a
throttling station and mdot joins the state with one matching residual. These
bind the residual FORM (the state layout, the added row, and its consistency
with the normal-mode solution); the hysteretic choke<->normal SWITCH in the
continuation driver is the next unit (M5-3b).

The anchor is a round trip against the already-verified normal mode: solve at
mdot0, read the exit static pressure the machine produces, feed it back as a
BackPressureSpec, and require the back-pressure solve to recover mdot0 and the
same state — from a DIFFERENT warm start, so mdot is genuinely solved for.

Provenance: M5 sub-step 3a, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.assembly.assembler import ResidualAssembler
from slcflow.assembly.inputs import ClosureFields, FrozenInputs
from slcflow.assembly.pack import n_unknowns, pack, unpack
from slcflow.diagnostics.record import SolveStatus
from slcflow.drivers import solve_classical, solve_newton
from slcflow.transport import TransportFields
from slcflow.types import BackPressureSpec, FidelityConfig, MassFlowSpec
from slcflow.verification.v1_analytic_ree import V1ForcedVortex, annulus_topology


def _setup(n_sl=9):
    case = V1ForcedVortex()
    topo = annulus_topology(case.r0, case.r1, case.length, n_sl,
                            case.n_stations)
    exact = case.exact()
    inlet = TransportFields(h0=np.full(n_sl, case.h0), s=np.full(n_sl, case.s),
                            rvt=case.inlet_rvt(topo.psi, exact))
    return case, topo, inlet


def _exit_static_pressure(result, station):
    """Static pressure at the ``q = 0`` node of ``station`` (the back-pressure
    handle) from a converged result."""
    f, tr = result.fields, result.frozen.transported
    r0 = f.metrics.r[0, station]
    vt0 = tr.rvt[0, station] / r0
    h = tr.h0[0, station] - 0.5 * (f.vm[0, station] ** 2 + vt0 ** 2)
    return float(result.frozen.fluid.p(h, tr.s[0, station]))


# --------------------------------------------------------------------------
# State-vector layout (section 6.6 / ARCH-3.2)
# --------------------------------------------------------------------------
def test_pack_unpack_carries_mdot_in_backpressure_mode():
    n_sl, n_qo = 5, 4
    assert n_unknowns(n_sl, n_qo, backpressure=True) == \
        n_unknowns(n_sl, n_qo) + 1
    vm = np.arange(n_qo, dtype=float)
    q = np.arange((n_sl - 2) * n_qo, dtype=float).reshape(n_sl - 2, n_qo)
    x = pack(vm, q, 42.0)
    vm2, q2, mdot = unpack(x, n_sl, n_qo, backpressure=True)
    np.testing.assert_array_equal(vm2, vm)
    np.testing.assert_array_equal(q2, q)
    assert mdot == 42.0
    with pytest.raises(Exception):        # normal-mode length rejects the +1
        unpack(x, n_sl, n_qo, backpressure=False)


# --------------------------------------------------------------------------
# Section 6.6: the added residual is zero at the consistent state
# --------------------------------------------------------------------------
def test_backpressure_row_vanishes_at_normal_solution():
    case, topo, inlet = _setup()
    mdot0 = 100.0
    res = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                          MassFlowSpec(mdot0), inlet)
    assert res.converged
    st = topo.n_qo - 1
    p_exit = _exit_static_pressure(res, st)

    # Same state + the true mdot, but under a BackPressureSpec matching the
    # produced pressure: every residual row (continuity, position, and the
    # appended back-pressure row) must be ~0.
    frozen = FrozenInputs(
        topology=topo, fluid=case.gas, fidelity=FidelityConfig.tier2(),
        spec=BackPressureSpec(p_exit=p_exit, station=st),
        transported=res.frozen.transported, closures=res.frozen.closures,
        vm_lagged=res.frozen.vm_lagged)
    asm = ResidualAssembler(frozen)
    vm_q0, q_int, _ = unpack(res.x, topo.n_sl, topo.n_qo)
    x_bp = pack(vm_q0, q_int, mdot0)
    r = asm.residual(x_bp)
    assert r.size == n_unknowns(topo.n_sl, topo.n_qo, backpressure=True)
    assert abs(r[-1]) < 1e-3 * p_exit             # back-pressure row
    assert float(np.max(np.abs(r[:topo.n_qo]))) < 1e-6 * mdot0  # continuity


# --------------------------------------------------------------------------
# Round trip against normal mode (the consistency anchor)
# --------------------------------------------------------------------------
def test_backpressure_solve_recovers_mdot_from_a_different_seed():
    case, topo, inlet = _setup()
    mdot0 = 100.0
    st = topo.n_qo - 1
    target = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                             MassFlowSpec(mdot0), inlet)
    p_exit = _exit_static_pressure(target, st)

    # Warm start from a DIFFERENT operating point so mdot is genuinely solved.
    seed = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(115.0), inlet)
    res_bp = solve_newton(topo, case.gas, FidelityConfig.tier2(),
                          BackPressureSpec(p_exit=p_exit, station=st), inlet,
                          warm_start=seed)
    assert res_bp.status is SolveStatus.CONVERGED
    _, _, mdot_rec = unpack(res_bp.x, topo.n_sl, topo.n_qo, backpressure=True)
    assert mdot_rec == pytest.approx(mdot0, rel=1e-4)
    # And the recovered state matches the normal-mode solution at mdot0.
    np.testing.assert_allclose(res_bp.x[:-1], target.x, atol=1e-5)


def test_higher_backpressure_throttles_mass_flow():
    # Physical monotonicity: raising the specified exit static pressure
    # reduces the mass flow (throttling toward choke/stall).
    case, topo, inlet = _setup()
    st = topo.n_qo - 1
    seed = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(100.0), inlet)
    p_ref = _exit_static_pressure(seed, st)

    def mdot_at(p_exit):
        r = solve_newton(topo, case.gas, FidelityConfig.tier2(),
                         BackPressureSpec(p_exit=p_exit, station=st), inlet,
                         warm_start=seed)
        assert r.converged
        return unpack(r.x, topo.n_sl, topo.n_qo, backpressure=True)[2]

    assert mdot_at(1.03 * p_ref) < mdot_at(p_ref) < mdot_at(0.97 * p_ref)
