"""Tests for slcflow.drivers.classical (Theory Manual section 6.2, 6.4-6.6;
ARCH-5.2, ARCH-6).

Each test cites the spec clause it verifies. Written in the same session as
the implementation (no adjudication needed; provenance: M2 sub-step 3).
"""
import numpy as np
import pytest

from slcflow.assembly import ResidualAssembler
from slcflow.diagnostics import SolveStatus
from slcflow.drivers import ClassicalConfig, solve_classical
from slcflow.errors import ConfigError
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import FlowPath, StationDef, StationType, WallCurve
from slcflow.grid import GridTopology
from slcflow.transport import TransportFields, TransportStep
from slcflow.types import FidelityConfig, MassFlowSpec

GAS = PerfectGas()
H0, S0 = 3.0e5, 0.0
R0, R1 = 0.3, 0.6


def cylinder_topology(n_sl=9, n_stations=6):
    z = np.linspace(0.0, 1.0, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R0)]))
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R1)]))
    fracs = np.linspace(0.0, 1.0, n_stations)
    fp = FlowPath(w0, w1, [StationDef(StationType.DUCT, f, f) for f in fracs])
    return GridTopology(fp, n_sl=n_sl)


def inlet_fields(n_sl, h0=H0, s=S0, rvt=0.0):
    full = lambda v: np.full(n_sl, float(v))
    return TransportFields(h0=full(h0), s=full(s), rvt=full(rvt))


def uniform_mdot(vm, rvt=0.0):
    """Analytic annulus mass flow for a spanwise-uniform Vm (swirl-free)."""
    rho = GAS.rho(H0 - 0.5 * vm**2, S0)
    return float(np.pi * rho * vm * (R1**2 - R0**2))


def solve(topo, mdot, rvt=0.0, fidelity=None, **kw):
    return solve_classical(
        topo, GAS, FidelityConfig.tier2() if fidelity is None else fidelity,
        MassFlowSpec(mdot), inlet_fields(topo.n_sl, rvt=rvt), **kw)


# --------------------------------------------------------------------------
# Section 6.2: convergence on analytic duct flows
# --------------------------------------------------------------------------
def test_uniform_flow_converges_to_analytic_vm():
    # Section 6.2: swirl-free uniform annulus flow. The converged Vm must
    # invert the analytic continuity relation on every q-o.
    vm_target = 120.0
    res = solve(cylinder_topology(), uniform_mdot(vm_target))
    assert res.status is SolveStatus.CONVERGED and res.converged
    np.testing.assert_allclose(res.fields.vm, vm_target, rtol=1e-8)


def test_free_vortex_v1_residual_consistency():
    # Sections 6.1/6.2 + A.5 check 1: the classical driver and the residual
    # assembler must agree -- at the driver's converged state the assembled
    # residual vector vanishes, and Vm is spanwise-uniform (free vortex,
    # uniform h0/s, Tier 2).
    topo = cylinder_topology()
    res = solve(topo, 100.0, rvt=12.0)
    assert res.converged
    r = ResidualAssembler(res.frozen).residual(res.x)
    scale = res.frozen.spec.mdot / (2.0 * np.pi)
    assert np.max(np.abs(r)) / scale < 1e-7
    np.testing.assert_allclose(res.fields.vm, res.fields.vm[0, 0], rtol=1e-8)


def test_forced_vortex_profile_matches_analytic_family():
    # Section 9 V1 seed through the full driver: converged forced-vortex Vm
    # obeys Vm^2(r) = Vm_q0^2 - 2 Omega_f^2 (r^2 - r0^2) on every q-o.
    omega_f = 60.0
    topo = cylinder_topology(n_sl=17)
    n_sl = topo.n_sl

    # Prescribing rVt = Omega_f * r^2 needs the streamline radii, which are
    # themselves part of the solution -- so re-prescribe at the converged
    # radii once (removes the sampling error to second order; the remaining
    # profile error is the section 5.3 discretization).
    r_rows = np.sqrt(np.linspace(0.0, 1.0, n_sl) * (R1**2 - R0**2) + R0**2)
    res = None
    for _ in range(2):
        inlet = TransportFields(h0=np.full(n_sl, H0), s=np.full(n_sl, S0),
                                rvt=omega_f * r_rows**2)
        res = solve_classical(topo, GAS, FidelityConfig.tier2(),
                              MassFlowSpec(100.0), inlet)
        assert res.converged
        r_rows = res.fields.metrics.r[:, 0]
    for j in range(topo.n_qo):
        r = res.fields.metrics.r[:, j]
        vm = res.fields.vm[:, j]
        vm_exact = np.sqrt(vm[0]**2 - 2.0 * omega_f**2 * (r**2 - r[0]**2))
        np.testing.assert_allclose(vm, vm_exact, rtol=2e-3)
    # And the state satisfies the residual contract.
    r_vec = ResidualAssembler(res.frozen).residual(res.x)
    assert np.max(np.abs(r_vec)) / (100.0 / (2 * np.pi)) < 1e-6


def test_rotor_step_euler_work_through_driver():
    # Sections 3.3/6.2.2: a rotor transport step raises h0 by the Euler work
    # in the driver's swept fields (partial cover of the rvt_le-consistency
    # carryover; the closure-fed path lands with M4).
    topo = cylinder_topology(n_stations=6)
    omega, rvt_in, rvt_out = 500.0, 12.0, 30.0
    steps = [TransportStep(), TransportStep(),
             TransportStep(omega=omega, rvt=np.full(topo.n_sl, rvt_out)),
             TransportStep(), TransportStep()]
    res = solve(topo, 100.0, rvt=rvt_in, steps=steps)
    assert res.converged
    h0_expect = H0 + omega * (rvt_out - rvt_in)
    np.testing.assert_allclose(res.frozen.transported.h0[:, -1], h0_expect,
                               rtol=1e-12)
    np.testing.assert_allclose(res.frozen.transported.rvt[:, -1], rvt_out,
                               rtol=1e-12)


# --------------------------------------------------------------------------
# Sections 6.4-6.6: relaxation, statuses (ARCH-6: returned, never raised)
# --------------------------------------------------------------------------
def test_choke_limited_status():
    # Section 6.6: mdot above the annulus capacity must return a typed
    # CHOKE_LIMITED status, never raise.
    res = solve(cylinder_topology(), 400.0)
    assert res.status is SolveStatus.CHOKE_LIMITED
    assert not res.converged
    assert "capacity" in res.record.reason


def test_numerical_failure_status():
    # AD-10/ARCH-6 boundary check: unphysical inputs (negative h0) yield a
    # typed NUMERICAL_FAILURE with the reason recorded, never an exception.
    topo = cylinder_topology()
    with np.errstate(invalid="ignore"):
        res = solve_classical(topo, GAS, FidelityConfig.tier2(),
                              MassFlowSpec(50.0),
                              inlet_fields(topo.n_sl, h0=-1.0e5))
    assert res.status is SolveStatus.NUMERICAL_FAILURE
    assert "non-finite" in res.record.reason


def test_max_iter_status_and_record():
    # Sections 6.2.5/ARCH-6: unconverged-by-budget returns MAX_ITER with the
    # full iteration history; every record carries all three norms plus the
    # section 6.4 relaxation factor within the configured cap.
    cfg = ClassicalConfig(max_outer=1, tol_pos=1e-16)
    res = solve(cylinder_topology(), 100.0, rvt=12.0, config=cfg)
    assert res.status is SolveStatus.MAX_ITER
    assert res.record.n_iterations == 1
    rec = res.record.iterations[0]
    assert rec.pos_norm > 1e-16          # free vortex moves streamlines
    assert rec.closure_norm == 0.0       # closures static in M2
    assert np.isfinite(rec.cont_norm)
    assert 0.0 < rec.omega_sl <= cfg.omega_sl_max


def test_all_norms_reported_each_iteration():
    # Section 6.2.5: "Report all three."
    res = solve(cylinder_topology(), 100.0, rvt=12.0)
    assert res.record.n_iterations >= 1
    for rec in res.record.iterations:
        for attr in ("cont_norm", "pos_norm", "closure_norm", "omega_sl"):
            assert np.isfinite(getattr(rec, attr))


# --------------------------------------------------------------------------
# Config boundary (AD-10)
# --------------------------------------------------------------------------
def test_wrong_step_count_raises():
    topo = cylinder_topology(n_stations=6)
    with pytest.raises(ConfigError, match="transport steps"):
        solve(topo, 100.0, steps=[TransportStep()] * 2)


def test_config_validation():
    with pytest.raises(ConfigError):
        ClassicalConfig(max_outer=0)
    with pytest.raises(ConfigError):
        ClassicalConfig(omega_sl_max=0.0)
