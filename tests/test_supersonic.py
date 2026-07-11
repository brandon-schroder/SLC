"""Meridional-supersonic-branch driver tests (Theory Manual sections 6.6, C.9).

Purpose-designed test: a **meanline converging-diverging duct (nozzle)**, whose
supersonic-meridional throat root the isentropic area-Mach relation gives in
closed form -- so this is a real analytic verification, not a structural gate.
The nozzle is isentropic (no rows, prescribed duct transport), so each station's
continuity is exactly the quasi-1D area rule ``mdot = rho Vm A`` with the classic
subsonic/supersonic root pair folded at ``M_m = 1``.

The tests bind: (a) the fold that defeats natural-parameter continuation (the
classical mass-flow driver chokes above the throat capacity); (b) the
pseudo-arclength driver crossing that fold onto the supersonic branch; (c) the
landed supersonic throat Mach matching the isentropic area-Mach supersonic root;
(d) the two-branch structure (same ``mdot``, sub- vs supersonic root); and (e)
config guards / typed statuses.

Provenance: written with the driver (drivers/supersonic.py), against the
independent isentropic area-Mach reference below.
"""
import numpy as np
import pytest

from slcflow.drivers import (ArclengthConfig, MeridionalBranchResult,
                             solve_classical, solve_supersonic_branch)
from slcflow.diagnostics.record import SolveStatus
from slcflow.errors import ConfigError
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import FlowPath, StationDef, StationType, WallCurve
from slcflow.grid import GridTopology
from slcflow.transport import TransportFields
from slcflow.types import FidelityConfig, MassFlowSpec

_R0 = 0.30                         # constant hub radius
_H0, _S0 = 3.0e5, 0.0              # inlet stagnation state


def _shroud(z):
    # Converging-diverging: minimum gap (the throat) at z = 0.5.
    return 0.50 - 0.14 * np.exp(-((z - 0.5) / 0.28) ** 2)


def _nozzle():
    """Meanline converging-diverging duct + isentropic inlet."""
    gas = PerfectGas()
    z = np.linspace(0.0, 1.0, 21)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, _R0)]))
    w1 = WallCurve.from_points(np.column_stack([z, _shroud(z)]))
    stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                StationDef(StationType.DUCT, 0.5, 0.5),     # throat (station 1)
                StationDef(StationType.DUCT, 1.0, 1.0)]
    fp = FlowPath(w0, w1, stations)
    topo = GridTopology(fp, n_sl=1)
    inlet = TransportFields(h0=np.array([_H0]), s=np.array([_S0]),
                            rvt=np.array([0.0]))
    return gas, topo, inlet


def _throat_area():
    return np.pi * (_shroud(0.5) ** 2 - _R0 ** 2)


def _throat_capacity(gas):
    """Isentropic sonic mass flow through the throat: ``rho* a* A*``."""
    T0 = _H0 / gas.cp
    p0 = float(gas.p(np.array([_H0]), _S0)[0])
    rho0 = p0 / (gas.R * T0)
    astar = np.sqrt(gas.gamma * gas.R * T0 * 2.0 / (gas.gamma + 1.0))
    rho_star = rho0 * (2.0 / (gas.gamma + 1.0)) ** (1.0 / (gas.gamma - 1.0))
    return rho_star * astar * _throat_area()


def _isentropic_supersonic_mach(area_ratio, gamma):
    """Supersonic root of the isentropic area-Mach relation ``A/A* = f(M)``."""
    g = gamma

    def f(M):
        return ((1.0 / M) * ((2.0 / (g + 1.0))
                * (1.0 + (g - 1.0) / 2.0 * M * M)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
                - area_ratio)

    M = 2.0
    for _ in range(100):
        d = (f(M + 1e-7) - f(M)) / 1e-7
        M -= f(M) / d
    return M


def _subsonic_seed(gas, topo, inlet, mdot):
    seed = solve_classical(topo, gas, FidelityConfig.tier1(),
                           MassFlowSpec(mdot), inlet)
    assert seed.converged
    return seed


# --------------------------------------------------------------------------
# (a) the fold that defeats natural-parameter continuation (section 6.6 / C.9)
# --------------------------------------------------------------------------
def test_classical_chokes_above_throat_capacity_the_fold():
    # The classical mass-flow driver takes the subsonic root and cannot cross
    # the throat capacity peak: below capacity it converges (subsonic, M_m < 1),
    # above it reports CHOKE_LIMITED -- the natural-parameter fold the
    # pseudo-arclength driver exists to cross.
    gas, topo, inlet = _nozzle()
    cap = _throat_capacity(gas)
    sub = solve_classical(topo, gas, FidelityConfig.tier1(),
                          MassFlowSpec(0.9 * cap), inlet)
    assert sub.converged
    assert sub.fields.mach_m[0, 1] < 1.0                 # subsonic throat root
    over = solve_classical(topo, gas, FidelityConfig.tier1(),
                           MassFlowSpec(1.05 * cap), inlet)
    assert over.status is SolveStatus.CHOKE_LIMITED


# --------------------------------------------------------------------------
# (b)+(c) cross the fold, land on the supersonic branch, match the analytic
# area-Mach supersonic root (section 6.6 / C.9)
# --------------------------------------------------------------------------
def test_crosses_fold_onto_supersonic_branch():
    gas, topo, inlet = _nozzle()
    cap = _throat_capacity(gas)
    seed = _subsonic_seed(gas, topo, inlet, 0.9 * cap)
    res = solve_supersonic_branch(topo, gas, FidelityConfig.tier1(), inlet,
                                  subsonic_seed=seed, target_mdot=0.9 * cap)
    assert isinstance(res, MeridionalBranchResult)
    assert res.converged
    assert res.fold_crossed
    # The turning point is the throat capacity (A.7) to discretization.
    assert res.fold_mdot == pytest.approx(cap, rel=5e-3)
    # Landed on the SUPERSONIC branch at the target mass flow.
    assert res.result.fields.mach_m[0, 1] > 1.0


def test_supersonic_throat_matches_isentropic_area_mach():
    # The real verification: the landed throat Mach equals the isentropic
    # area-Mach supersonic root for the achieved mass flow, and the large-area
    # inlet/exit stay subsonic (only the throat crossed -- a rank-1 fold).
    gas, topo, inlet = _nozzle()
    cap = _throat_capacity(gas)
    target = 0.88 * cap
    seed = _subsonic_seed(gas, topo, inlet, 0.9 * cap)
    res = solve_supersonic_branch(topo, gas, FidelityConfig.tier1(), inlet,
                                  subsonic_seed=seed, target_mdot=target)
    assert res.converged
    mm = res.result.fields.mach_m[0]
    # A_throat/A* = capacity/mdot for an isentropic throat.
    area_ratio = cap / target
    m_analytic = _isentropic_supersonic_mach(area_ratio, gas.gamma)
    assert mm[1] == pytest.approx(m_analytic, rel=3e-3)   # supersonic throat
    assert mm[0] < 1.0 and mm[2] < 1.0                    # inlet/exit subsonic


# --------------------------------------------------------------------------
# (d) two branches at one mass flow (section C.9)
# --------------------------------------------------------------------------
def test_same_mdot_has_sub_and_supersonic_roots():
    gas, topo, inlet = _nozzle()
    cap = _throat_capacity(gas)
    mdot = 0.85 * cap
    subsonic = solve_classical(topo, gas, FidelityConfig.tier1(),
                               MassFlowSpec(mdot), inlet)
    seed = _subsonic_seed(gas, topo, inlet, 0.92 * cap)
    supersonic = solve_supersonic_branch(topo, gas, FidelityConfig.tier1(),
                                         inlet, subsonic_seed=seed,
                                         target_mdot=mdot)
    assert subsonic.converged and supersonic.converged
    # Same mass flow, two throat roots straddling M_m = 1.
    assert subsonic.fields.mach_m[0, 1] < 1.0
    assert supersonic.result.fields.mach_m[0, 1] > 1.0
    # Both satisfy continuity at the same mdot (the achieved mass flow matches).
    assert supersonic.result.frozen.spec.mdot == pytest.approx(mdot)


def test_deterministic_replay():
    gas, topo, inlet = _nozzle()
    cap = _throat_capacity(gas)
    seed = _subsonic_seed(gas, topo, inlet, 0.9 * cap)
    kw = dict(subsonic_seed=seed, target_mdot=0.9 * cap)
    a = solve_supersonic_branch(topo, gas, FidelityConfig.tier1(), inlet, **kw)
    b = solve_supersonic_branch(topo, gas, FidelityConfig.tier1(), inlet, **kw)
    np.testing.assert_array_equal(a.result.x, b.result.x)


# --------------------------------------------------------------------------
# (e) config guards / typed statuses (ARCH-6, AD-10 at the config boundary)
# --------------------------------------------------------------------------
def test_requires_a_frozen_seed():
    gas, topo, inlet = _nozzle()
    with pytest.raises(ConfigError):
        solve_supersonic_branch(topo, gas, FidelityConfig.tier1(), inlet,
                                subsonic_seed=None, target_mdot=30.0)


def test_rejects_nonpositive_target():
    gas, topo, inlet = _nozzle()
    seed = _subsonic_seed(gas, topo, inlet, 30.0)
    with pytest.raises(ConfigError):
        solve_supersonic_branch(topo, gas, FidelityConfig.tier1(), inlet,
                                subsonic_seed=seed, target_mdot=-1.0)


def test_arclength_config_validates():
    with pytest.raises(ConfigError):
        ArclengthConfig(ds0=1.0, ds_max=0.5)          # ds0 > ds_max
    with pytest.raises(ConfigError):
        ArclengthConfig(max_steps=0)


def test_max_steps_exhaustion_is_typed_not_raised():
    # A too-small step budget cannot reach the fold+target: a typed MAX_ITER,
    # never an exception (ARCH-6).
    gas, topo, inlet = _nozzle()
    cap = _throat_capacity(gas)
    seed = _subsonic_seed(gas, topo, inlet, 0.9 * cap)
    res = solve_supersonic_branch(topo, gas, FidelityConfig.tier1(), inlet,
                                  subsonic_seed=seed, target_mdot=0.9 * cap,
                                  config=ArclengthConfig(max_steps=2))
    assert res.status is SolveStatus.MAX_ITER
    assert not res.converged
