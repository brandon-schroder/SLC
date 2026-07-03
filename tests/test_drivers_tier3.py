"""Tier-3-through-the-driver tests (M3 sub-step 1): curvature terms active
with MOVING streamlines (Theory Manual sections 5.5, 6.2, 6.4; ARCH-8 M3).

Covers: section 5.5 curvature under-relaxation wiring (analytic check on the
frozen bend), the full Tier-3 classical solve on the curved annulus, and the
structural no-crossing guarantee of the classical repositioning (a convex
blend of monotone position vectors stays monotone -- see the assembler's
known-limitations note; the Newton-side guard is M5).

Written in the same session as the implementation (provenance: M3 sub-step 1).
"""
import numpy as np
import pytest

from slcflow.assembly import (ClosureFields, FrozenInputs, ResidualAssembler,
                              pack)
from slcflow.diagnostics import SolveStatus
from slcflow.drivers import ClassicalConfig, solve_classical
from slcflow.errors import ConfigError
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import FlowPath, StationDef, StationType, WallCurve
from slcflow.grid import GridTopology
from slcflow.transport import TransportFields
from slcflow.types import FidelityConfig, MassFlowSpec

GAS = PerfectGas()
H0, S0 = 3.0e5, 0.0
BEND_CENTER = (0.0, 0.8)
R_INNERBEND, R_OUTERBEND = 0.2, 0.5


def bend_topology(n_sl=9, n_stations=13):
    zc, rc = BEND_CENTER

    def wall(R):
        return lambda u: (zc + R * np.sin(0.5 * np.pi * u),
                          rc - R * np.cos(0.5 * np.pi * u))

    w0 = WallCurve.from_callable(wall(R_INNERBEND), n=201)
    w1 = WallCurve.from_callable(wall(R_OUTERBEND), n=201)
    fracs = np.linspace(0.0, 1.0, n_stations)
    fp = FlowPath(w0, w1, [StationDef(StationType.DUCT, f, f) for f in fracs])
    return GridTopology(fp, n_sl=n_sl)


def frozen_bend(topo, q, kappa_lagged=None, kappa_relax=1.0):
    n_sl, n_qo = topo.n_sl, topo.n_qo
    full = lambda v: np.full((n_sl, n_qo), float(v))
    return FrozenInputs(
        topology=topo, fluid=GAS, fidelity=FidelityConfig.tier3(),
        spec=MassFlowSpec(50.0),
        transported=TransportFields(h0=full(H0), s=full(S0), rvt=full(0.0)),
        closures=ClosureFields(np.zeros((n_sl, n_qo))),
        kappa_lagged=kappa_lagged, kappa_relax=kappa_relax)


# --------------------------------------------------------------------------
# Section 5.5: curvature under-relaxation wiring
# --------------------------------------------------------------------------
def test_kappa_blend_halves_curvature_effect():
    # On the frozen concentric bend the curvature term gives Vm ~ 1/R_c
    # (A.5 case 2). Blending kappa 50/50 with a zero lagged field halves the
    # exponent: Vm ~ (1/R_c)^0.5, checked against the analytic profile at
    # the M1-gate tolerance class (discretization-limited, not exact).
    topo = bend_topology()
    R_sl = np.linspace(R_OUTERBEND, R_INNERBEND, topo.n_sl)
    q = np.tile((R_OUTERBEND - R_sl)[:, None], (1, topo.n_qo))
    x = pack(np.full(topo.n_qo, 100.0), q[1:-1, :])

    fz_half = frozen_bend(topo, q,
                          kappa_lagged=np.zeros((topo.n_sl, topo.n_qo)),
                          kappa_relax=0.5)
    vm_half = ResidualAssembler(fz_half).split(x).vm
    ratio_half = vm_half / vm_half[0:1, :]
    exact_half = np.sqrt(R_sl[0] / R_sl)[:, None]
    np.testing.assert_allclose(ratio_half,
                               np.broadcast_to(exact_half, ratio_half.shape),
                               rtol=2e-2)


def test_kappa_relax_one_or_no_lag_is_identity():
    # kappa_relax = 1 and kappa_lagged = None must both reproduce the
    # unblended integration bit-for-bit.
    topo = bend_topology()
    R_sl = np.linspace(R_OUTERBEND, R_INNERBEND, topo.n_sl)
    q = np.tile((R_OUTERBEND - R_sl)[:, None], (1, topo.n_qo))
    x = pack(np.full(topo.n_qo, 100.0), q[1:-1, :])
    base = ResidualAssembler(frozen_bend(topo, q)).split(x).vm
    lagged_junk = np.full((topo.n_sl, topo.n_qo), 99.0)
    same = ResidualAssembler(
        frozen_bend(topo, q, kappa_lagged=lagged_junk, kappa_relax=1.0)
    ).split(x).vm
    np.testing.assert_array_equal(base, same)


def test_kappa_config_validation():
    topo = bend_topology(n_sl=3, n_stations=4)
    q = np.tile(np.linspace(0.0, 0.3, 3)[:, None], (1, topo.n_qo))
    with pytest.raises(ConfigError, match="kappa_relax"):
        frozen_bend(topo, q, kappa_relax=0.0)
    with pytest.raises(ConfigError, match="kappa_lagged shape"):
        frozen_bend(topo, q, kappa_lagged=np.zeros((2, 2)))
    with pytest.raises(ConfigError, match="kappa_relax"):
        ClassicalConfig(kappa_relax=1.5)


# --------------------------------------------------------------------------
# Tier 3 end-to-end: curvature + repositioning coupled (section 6.2/6.4)
# --------------------------------------------------------------------------
def tier3_bend_solve(n_sl=9, n_stations=7, **cfg_kw):
    topo = bend_topology(n_sl=n_sl, n_stations=n_stations)
    inlet = TransportFields(h0=np.full(n_sl, H0), s=np.full(n_sl, S0),
                            rvt=np.zeros(n_sl))
    config = ClassicalConfig(**cfg_kw) if cfg_kw else ClassicalConfig()
    return topo, solve_classical(topo, GAS, FidelityConfig.tier3(),
                                 MassFlowSpec(50.0), inlet, config=config)


@pytest.fixture(scope="module")
def tier3_result():
    return tier3_bend_solve()


def test_tier3_bend_converges_with_repositioning(tier3_result):
    # ARCH-8 M3 core: the curvature feedback loop (kappa -> Vm -> mass
    # distribution -> positions -> kappa) reaches a fixed point under the
    # section 6.4 adaptive relaxation + section 5.5 curvature lag defaults.
    # Without the lag this case diverges via the streamwise odd-even mode
    # at ANY relaxation factor (M3-1 measurement).
    _, res = tier3_result
    assert res.status is SolveStatus.CONVERGED
    # The converged state satisfies the assembler's residual contract.
    r = ResidualAssembler(res.frozen).residual(res.x)
    assert np.max(np.abs(r)) / (50.0 / (2 * np.pi)) < 1e-7
    # Physics sanity (V2 formal comparison is M3-2): Vm rises monotonically
    # from the outer-bend wall (q = 0) toward the bend center on every q-o,
    # per the A.5 case 2 mechanism.
    assert np.all(np.diff(res.fields.vm, axis=0) > 0.0)
    # Spanwise ratio in the ballpark of the concentric free-vortex estimate
    # (positions shift off concentric, so this is deliberately loose).
    ratio = res.fields.vm[-1, :] / res.fields.vm[0, :]
    assert np.all(ratio > 1.8) and np.all(ratio < 3.2)  # concentric: 2.5


def test_tier3_positions_stay_monotone(tier3_result):
    # Structural no-crossing guarantee of classical repositioning: nodal q
    # strictly increasing on every q-o of the converged state (and the solve
    # never tripped the PCHIP monotonicity requirement on the way).
    _, res = tier3_result
    assert res.converged
    assert np.all(np.diff(res.fields.q, axis=0) > 0.0)


def test_tier3_kappa_relax_setting_does_not_move_the_answer(tier3_result):
    # Section 5.5: the lag changes the ITERATION, never the fixed point.
    _, res_base = tier3_result
    _, res_lag = tier3_bend_solve(kappa_relax=0.5)
    assert res_lag.status is SolveStatus.CONVERGED
    np.testing.assert_allclose(res_lag.fields.vm, res_base.fields.vm,
                               rtol=1e-6)
    np.testing.assert_allclose(res_lag.fields.q, res_base.fields.q,
                               atol=1e-7)


def test_tier2_on_bend_is_spanwise_uniform(tier3_result):
    # AD-1 flags as data, driver level: the same bend solved at Tier 2 has
    # no curvature term, so Vm is spanwise-uniform per station (levels vary
    # station to station with flow area); Tier 3 is strongly sheared.
    topo = bend_topology(n_stations=7)
    inlet = TransportFields(h0=np.full(9, H0), s=np.full(9, S0),
                            rvt=np.zeros(9))
    res2 = solve_classical(topo, GAS, FidelityConfig.tier2(),
                           MassFlowSpec(50.0), inlet)
    assert res2.converged
    np.testing.assert_allclose(
        res2.fields.vm, np.broadcast_to(res2.fields.vm[0:1, :],
                                        res2.fields.vm.shape), rtol=1e-8)
    _, res3 = tier3_result
    assert np.max(res3.fields.vm[-1, :] / res3.fields.vm[0, :]) > 1.8
