"""Global Newton driver tests (Theory Manual section 6.3; ARCH-5.3; M5-1).

Anchors the Newton driver against the *independent* references already in the
ladder: it must reach the same fixed point the classical driver does (they
solve the same R(x) = 0) and the V1 analytic solution, and it must do so from
a warm start under line-search globalization — including rejecting the
crossing-streamline trial steps the classical scheme structurally cannot
produce but Newton can (the M2/M3 carryover).

Provenance: M5 sub-step 1, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.assembly.assembler import ResidualAssembler
from slcflow.assembly.pack import pack, unpack
from slcflow.diagnostics.record import SolveStatus
from slcflow.drivers import (ClassicalConfig, newton_solve, solve_classical,
                             solve_newton)
from slcflow.drivers.newton import (NewtonConfig, _is_feasible_q,
                                    _residual_scale, _safe_residual)
from slcflow.errors import ConfigError
from slcflow.grid import initialize_positions
from slcflow.transport import TransportFields
from slcflow.types import FidelityConfig, MassFlowSpec
from slcflow.verification.v1_analytic_ree import (V1ForcedVortex, V1FreeVortex,
                                                  annulus_topology)


def _setup(case, n_sl):
    topo = annulus_topology(case.r0, case.r1, case.length, n_sl,
                            case.n_stations)
    exact = case.exact()
    inlet = TransportFields(h0=np.full(n_sl, case.h0),
                            s=np.full(n_sl, case.s),
                            rvt=case.inlet_rvt(topo.psi, exact))
    return topo, inlet, exact


def _tier2(topo, case, inlet, **kw):
    return solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(case.mdot), inlet, **kw)


# --------------------------------------------------------------------------
# Section 6.3: the Newton core solves R(x) = 0
# --------------------------------------------------------------------------
def test_newton_core_matches_classical_and_analytic():
    # newton_solve from a perturbed area-rule warm start reaches BOTH the
    # classical fixed point and the V1c analytic hub Vm; the scaled residual
    # is driven below tolerance. Quadratic locally -> far fewer iterations
    # than the classical relaxation.
    case = V1ForcedVortex()
    topo, inlet, exact = _setup(case, 17)
    res_c = _tier2(topo, case, inlet)
    assert res_c.converged

    asm = ResidualAssembler(res_c.frozen)
    q0 = initialize_positions(topo)
    x0 = pack(res_c.x[:topo.n_qo] * 0.9, q0[1:-1, :])   # 10% off in Vm
    x, status, recs = newton_solve(asm, x0)

    assert status is SolveStatus.CONVERGED
    np.testing.assert_allclose(x, res_c.x, atol=1e-6)
    assert x[0] == pytest.approx(exact.vm0, rel=2e-3)   # analytic anchor
    assert recs[-1].cont_norm < NewtonConfig().tol_res
    assert len(recs) < len(res_c.record.iterations)     # beats relaxation


@pytest.mark.parametrize("case", [V1FreeVortex.compressible(),
                                  V1ForcedVortex()],
                         ids=["free_vortex", "forced_vortex"])
def test_solve_newton_outer_matches_classical(case):
    # The outer driver, warm-started from a DELIBERATELY unconverged classical
    # run (3 iterations), converges to the fully-converged classical solution.
    n_sl = 9 if isinstance(case, V1FreeVortex) else 17
    topo, inlet, _ = _setup(case, n_sl)
    warm = _tier2(topo, case, inlet, config=ClassicalConfig(max_outer=3))
    assert not warm.converged                            # genuinely partial
    res_n = solve_newton(topo, case.gas, FidelityConfig.tier2(),
                         MassFlowSpec(case.mdot), inlet, warm_start=warm)
    res_c = _tier2(topo, case, inlet)
    assert res_n.converged and res_c.converged
    np.testing.assert_allclose(res_n.x, res_c.x, atol=1e-6)


def test_newton_residual_is_below_tol_at_solution():
    # The converged Newton state has a genuinely small scaled residual (not
    # just a small step): the section 6.2.5 continuity/position norms combined.
    case = V1FreeVortex.compressible()
    topo, inlet, _ = _setup(case, 9)
    warm = _tier2(topo, case, inlet, config=ClassicalConfig(max_outer=3))
    res_n = solve_newton(topo, case.gas, FidelityConfig.tier2(),
                         MassFlowSpec(case.mdot), inlet, warm_start=warm)
    asm = ResidualAssembler(res_n.frozen)
    r = _safe_residual(asm, res_n.x, _residual_scale(res_n.frozen))
    assert r is not None
    assert float(np.max(np.abs(r))) < NewtonConfig().tol_res


# --------------------------------------------------------------------------
# ARCH-5.3: warm start mandatory
# --------------------------------------------------------------------------
def test_solve_newton_requires_warm_start():
    case = V1FreeVortex.compressible()
    topo, inlet, _ = _setup(case, 9)
    with pytest.raises(ConfigError, match="warm_start"):
        solve_newton(topo, case.gas, FidelityConfig.tier2(),
                     MassFlowSpec(case.mdot), inlet, warm_start=None)


# --------------------------------------------------------------------------
# Section 6.3 globalization: crossing-streamline rejection
# --------------------------------------------------------------------------
def test_crossing_streamline_is_infeasible_and_rejected():
    # A non-monotone q-o (swapped interior streamlines) must be flagged
    # infeasible so the line search backtracks instead of the assembler
    # raising on its PCHIP construction (AD-10 letter for the Newton path).
    case = V1ForcedVortex()
    topo, inlet, _ = _setup(case, 17)
    res_c = _tier2(topo, case, inlet)
    vm_q0, q_int, _ = unpack(res_c.x, 17, topo.n_qo)
    q_bad = q_int.copy()
    q_bad[[0, 5]] = q_bad[[5, 0]]                        # cross two streamlines
    x_bad = pack(vm_q0, q_bad)

    assert not _is_feasible_q(x_bad, res_c.frozen)
    asm = ResidualAssembler(res_c.frozen)
    assert _safe_residual(asm, x_bad, _residual_scale(res_c.frozen)) is None
    # The converged Newton solution itself is strictly monotone in q.
    q_full = np.concatenate([np.zeros((1, topo.n_qo)),
                             unpack(res_c.x, 17, topo.n_qo)[1],
                             np.array([[qo.length for qo
                                        in topo.flowpath.qo_curves]])], axis=0)
    assert np.all(np.diff(q_full, axis=0) > 0.0)


def test_newton_line_search_recovers_from_a_crossing_full_step():
    # From a valid warm start whose UNDAMPED Newton step would cross
    # streamlines, the Armijo + feasibility line search still converges
    # (alpha < 1 recorded on at least one early iterate).
    case = V1ForcedVortex()
    topo, inlet, _ = _setup(case, 17)
    res_c = _tier2(topo, case, inlet)
    asm = ResidualAssembler(res_c.frozen)
    # Warm start: compress all interior streamlines toward one wall so the
    # first Newton correction is large and position-heavy.
    vm_q0, q_int, _ = unpack(res_c.x, 17, topo.n_qo)
    lengths = np.array([qo.length for qo in topo.flowpath.qo_curves])
    q_squeezed = 0.03 * lengths[None, :] + 0.94 * q_int
    x0 = pack(vm_q0 * 1.1, q_squeezed)
    x, status, recs = newton_solve(asm, x0)
    assert status is SolveStatus.CONVERGED
    assert _is_feasible_q(x, res_c.frozen)
    np.testing.assert_allclose(x, res_c.x, atol=1e-6)


def test_negative_vm_trial_is_infeasible_despite_finite_residual():
    # 2026-07 Tier-3 stabilization follow-up (Newton side of the classical
    # positive-branch rule): the master ODE is singular at Vm = 0, and a
    # trial x whose integrated Vm crosses zero sits on a spurious branch
    # whose residual can be FINITE (mass balancing by sign cancellation) --
    # the old screen (finite residual + monotone q) accepted it, letting
    # garbage poison the line-search merit. Scaling one station's Vm_q0 to
    # 0.1x on the converged V1c state produces exactly that: negative
    # interior Vm, finite residual. The guard must reject it.
    case = V1ForcedVortex()
    topo, inlet, _ = _setup(case, 17)
    res_c = _tier2(topo, case, inlet)
    assert res_c.converged
    asm = ResidualAssembler(res_c.frozen)
    scale = _residual_scale(res_c.frozen)

    x_bad = np.array(res_c.x)
    x_bad[1] *= 0.1
    assert _is_feasible_q(x_bad, res_c.frozen)      # positions untouched
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        fields = asm.split(x_bad)
        r = asm.residual(x_bad)
    assert np.all(np.isfinite(r))                   # old rule would ACCEPT
    assert np.any(fields.vm < 0.0)                  # ...a spurious branch
    assert _safe_residual(asm, x_bad, scale) is None  # new rule rejects

    # Healthy states are untouched: the converged x still screens clean and
    # residual_from(split(x), x) is exactly residual(x) (the refactor seam).
    r_ok = _safe_residual(asm, res_c.x, scale)
    assert r_ok is not None
    np.testing.assert_array_equal(
        asm.residual_from(asm.split(res_c.x), res_c.x), asm.residual(res_c.x))


# --------------------------------------------------------------------------
# ARCH-5.3: colored-FD Jacobian, validated column-for-column against dense
# --------------------------------------------------------------------------
def test_colored_jacobian_matches_dense_straight_annulus():
    # Straight annulus, Tier 2: the groupable regime (curvature/lean off,
    # |sin(eps)| ~ 0) -- q columns share stride-1 colors and the colored
    # Jacobian must match the dense baseline to FD noise (measured 1.8e-8;
    # asserted at 1e-6 of the Jacobian scale).
    from slcflow.drivers.newton import (_fd_jacobian, _fd_jacobian_colored,
                                        _q_columns_groupable)
    case = V1ForcedVortex()
    topo, inlet, _ = _setup(case, 17)
    res = _tier2(topo, case, inlet)
    assert res.converged
    asm = ResidualAssembler(res.frozen)
    scale = _residual_scale(res.frozen)
    x = np.array(res.x, dtype=float)
    r0 = _safe_residual(asm, x, scale)
    assert _q_columns_groupable(asm, x)          # the exact grouped regime
    jd = _fd_jacobian(asm, x, r0, scale, NewtonConfig())
    jc = _fd_jacobian_colored(asm, x, r0, scale, NewtonConfig(), True)
    assert np.max(np.abs(jc - jd)) < 1e-6 * np.max(np.abs(jd))


def test_colored_jacobian_curved_path_detects_and_stays_exact():
    # Curved annulus: grouping q columns is NOT safe (the eps coupling is
    # first-order on a bend -- measured 38% error when forced), so the
    # groupability check must refuse, and the ungrouped colored Jacobian
    # (vm color + per-column q) must equal dense BIT-EXACTLY: the vm
    # cross-station terms are true zeros, so the grouped vm evaluation
    # reproduces each dense column's arithmetic identically.
    from slcflow.drivers.newton import (_fd_jacobian, _fd_jacobian_colored,
                                        _q_columns_groupable)
    from slcflow.verification import V2CurvedAnnulus

    case = V2CurvedAnnulus()
    exact = case.exact()
    topo = case.topology(5)
    inlet = TransportFields(h0=np.full(5, case.h0), s=np.full(5, case.s),
                            rvt=np.zeros(5))
    warm = solve_classical(topo, case.gas, FidelityConfig.tier2(),
                           MassFlowSpec(exact.mdot), inlet,
                           config=ClassicalConfig(max_outer=5))
    asm = ResidualAssembler(warm.frozen)
    scale = _residual_scale(warm.frozen)
    x = np.array(warm.x, dtype=float)
    r0 = _safe_residual(asm, x, scale)
    assert not _q_columns_groupable(asm, x)      # bend: refuse grouping
    jd = _fd_jacobian(asm, x, r0, scale, NewtonConfig())
    jc = _fd_jacobian_colored(asm, x, r0, scale, NewtonConfig(), False)
    np.testing.assert_array_equal(jc, jd)


def test_colored_and_dense_newton_reach_the_same_solution():
    # End-to-end: the default (colored) and the dense baseline drive the
    # same warm start to the same fixed point.
    case = V1ForcedVortex()
    topo, inlet, _ = _setup(case, 17)
    res_c = _tier2(topo, case, inlet)
    asm = ResidualAssembler(res_c.frozen)
    x0 = pack(res_c.x[:topo.n_qo] * 0.9,
              unpack(res_c.x, 17, topo.n_qo)[1])
    xa, sa, _ = newton_solve(asm, x0, NewtonConfig(jacobian="colored"))
    xb, sb, _ = newton_solve(asm, x0, NewtonConfig(jacobian="dense"))
    assert sa is SolveStatus.CONVERGED and sb is SolveStatus.CONVERGED
    np.testing.assert_allclose(xa, xb, atol=1e-8)


def test_newton_config_rejects_unknown_jacobian_mode():
    with pytest.raises(ConfigError, match="jacobian"):
        NewtonConfig(jacobian="banded")


def test_newton_is_deterministic():
    case = V1FreeVortex.compressible()
    topo, inlet, _ = _setup(case, 9)
    warm = _tier2(topo, case, inlet, config=ClassicalConfig(max_outer=3))
    a = solve_newton(topo, case.gas, FidelityConfig.tier2(),
                     MassFlowSpec(case.mdot), inlet, warm_start=warm)
    b = solve_newton(topo, case.gas, FidelityConfig.tier2(),
                     MassFlowSpec(case.mdot), inlet, warm_start=warm)
    np.testing.assert_array_equal(a.x, b.x)
