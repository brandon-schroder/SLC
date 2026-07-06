"""Tests for slcflow.assembly and slcflow.types (Theory Manual sections 5.3,
5.4, 6.1, A.5, A.7; ARCH-3.2/3.3, ARCH-5.1).

Each test cites the spec clause it verifies. Written in the same session as
the implementation (no adjudication needed; provenance: M2 sub-step 2).
"""
import numpy as np
import pytest

from slcflow.assembly import (
    ClosureFields,
    FrozenInputs,
    ResidualAssembler,
    n_unknowns,
    pack,
    unpack,
)
from slcflow.errors import ConfigError
from slcflow.fluid.perfectgas import PerfectGas
from slcflow.geometry import FlowPath, StationDef, StationType, WallCurve
from slcflow.grid import GridTopology, initialize_positions
from slcflow.transport import TransportFields
from slcflow.types import BackPressureSpec, FidelityConfig, MassFlowSpec

GAS = PerfectGas()
H0, S0 = 3.0e5, 0.0
R0, R1 = 0.3, 0.6


# --------------------------------------------------------------------------
# Fixtures (same analytic flow paths as test_grid.py)
# --------------------------------------------------------------------------
def cylinder_path(n_stations=6, lean=0.0):
    z = np.linspace(0.0, 1.0, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R0)]))
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, R1)]))
    f0 = np.linspace(0.0, 1.0 - lean, n_stations)
    stations = [StationDef(StationType.DUCT, f, f + lean) for f in f0]
    return FlowPath(w0, w1, stations)


BEND_CENTER = (0.0, 0.8)
R_INNERBEND, R_OUTERBEND = 0.2, 0.5


def bend_setup(n_sl=9, n_stations=13):
    """90-degree concentric bend with frozen concentric streamlines (the M1
    acceptance-gate geometry; q runs from the OUTER-bend wall per A.1.1)."""
    zc, rc = BEND_CENTER

    def wall(R):
        return lambda u: (zc + R * np.sin(0.5 * np.pi * u),
                          rc - R * np.cos(0.5 * np.pi * u))

    w0 = WallCurve.from_callable(wall(R_INNERBEND), n=201)
    w1 = WallCurve.from_callable(wall(R_OUTERBEND), n=201)
    fracs = np.linspace(0.0, 1.0, n_stations)
    fp = FlowPath(w0, w1, [StationDef(StationType.DUCT, f, f) for f in fracs])
    topo = GridTopology(fp, n_sl=n_sl)
    R_sl = np.linspace(R_OUTERBEND, R_INNERBEND, n_sl)
    q = np.tile((R_OUTERBEND - R_sl)[:, None], (1, topo.n_qo))
    return topo, q, R_sl


def uniform_transport(n_sl, n_qo, h0=H0, s=S0, rvt=0.0):
    full = lambda v: np.full((n_sl, n_qo), float(v))
    return TransportFields(h0=full(h0), s=full(s), rvt=full(rvt))


def make_frozen(topo, transported=None, fidelity=None, mdot=10.0,
                blockage=0.0, vm_lagged=None):
    n_sl, n_qo = topo.n_sl, topo.n_qo
    if transported is None:
        transported = uniform_transport(n_sl, n_qo)
    return FrozenInputs(
        topology=topo, fluid=GAS,
        fidelity=FidelityConfig.tier2() if fidelity is None else fidelity,
        spec=MassFlowSpec(mdot), transported=transported,
        closures=ClosureFields(np.full((n_sl, n_qo), float(blockage))),
        vm_lagged=vm_lagged)


def cylinder_state(topo, vm_q0, q_rows=None):
    """State vector for frozen uniform-q streamlines on the cylinder."""
    n_sl, n_qo = topo.n_sl, topo.n_qo
    if q_rows is None:
        q_rows = np.linspace(0.0, R1 - R0, n_sl)
    q_int = np.tile(q_rows[1:-1, None], (1, n_qo))
    return pack(np.full(n_qo, float(vm_q0)), q_int), q_rows


# --------------------------------------------------------------------------
# pack / unpack (ARCH-3.2)
# --------------------------------------------------------------------------
def test_pack_unpack_round_trip():
    # ARCH-3.2: x = [Vm_q0[j] all j] + [interior q, C-order].
    rng = np.random.default_rng(7)
    vm, qi = rng.uniform(10, 100, 5), rng.uniform(0, 1, (3, 5))
    x = pack(vm, qi)
    assert x.shape == (n_unknowns(5, 5),)
    vm2, qi2, _ = unpack(x, n_sl=5, n_qo=5)
    assert np.array_equal(vm2, vm) and np.array_equal(qi2, qi)
    # Ordering is normative: vm block first.
    assert np.array_equal(x[:5], vm)


def test_unpack_rejects_wrong_length():
    with pytest.raises(ConfigError, match="expected"):
        unpack(np.zeros(7), n_sl=5, n_qo=5)


def test_n_unknowns_degenerate_wall_only():
    # n_sl = 2 (walls only): no interior positions, continuity rows only.
    assert n_unknowns(2, 4) == 4


# --------------------------------------------------------------------------
# Config-boundary validation (AD-10, ARCH-3.3)
# --------------------------------------------------------------------------
def test_fidelity_and_spec_validation():
    with pytest.raises(ConfigError):
        FidelityConfig(curvature_term=1.5)
    with pytest.raises(ConfigError):
        MassFlowSpec(0.0)
    with pytest.raises(ConfigError):
        BackPressureSpec(p_exit=-1.0, station=0)
    assert FidelityConfig.tier2().curvature_term == 0.0
    assert FidelityConfig.tier3().lean_term == 1.0


def test_frozen_inputs_validation():
    topo = GridTopology(cylinder_path(), n_sl=5)
    with pytest.raises(ConfigError, match="shape"):
        make_frozen(topo, transported=uniform_transport(4, topo.n_qo))
    with pytest.raises(ConfigError, match="0 <= B < 1"):
        make_frozen(topo, blockage=1.0)

    def _bp(station):
        return FrozenInputs(
            topology=topo, fluid=GAS, fidelity=FidelityConfig.tier2(),
            spec=BackPressureSpec(p_exit=1e5, station=station),
            transported=uniform_transport(5, topo.n_qo),
            closures=ClosureFields(np.zeros((5, topo.n_qo))))

    # BackPressureSpec is accepted since M5 (choke-proximal mode, section 6.6);
    # an out-of-range throttling station is a config-boundary error (AD-10).
    assert _bp(0).n_qo == topo.n_qo
    with pytest.raises(ConfigError, match="station"):
        _bp(topo.n_qo)


def test_tier1_meanline_requires_q_fixed():
    # Tier 1 (n_sl = 1, section 8) is now supported, but the fixed mean-line
    # position is frozen data (repositioning off), validated at the config
    # boundary (AD-10): omitting q_fixed raises.
    topo = GridTopology(cylinder_path(), n_sl=1)
    with pytest.raises(ConfigError, match="q_fixed"):
        make_frozen(topo)


def test_tier1_meanline_assembles_single_node():
    # With q_fixed supplied, the assembler builds the single mid-psi node and
    # the master ODE is trivial: Vm along the (one-node) q-o is exactly the
    # boundary value, and the residual has n_qo continuity rows only (no
    # interior position rows).
    topo = GridTopology(cylinder_path(), n_sl=1)
    q_fixed = initialize_positions(topo)           # area-rule mean line
    fz = FrozenInputs(
        topology=topo, fluid=GAS, fidelity=FidelityConfig.tier2(),
        spec=MassFlowSpec(10.0),
        transported=uniform_transport(1, topo.n_qo, rvt=8.0),
        closures=ClosureFields(np.zeros((1, topo.n_qo))), q_fixed=q_fixed)
    asm = ResidualAssembler(fz)
    x = pack(np.full(topo.n_qo, 90.0), np.zeros((0, topo.n_qo)))
    fields = asm.split(x)
    assert fields.vm.shape == (1, topo.n_qo)
    np.testing.assert_allclose(fields.vm[0, :], 90.0)   # trivial one-node ODE
    r = asm.residual(x)
    assert r.shape == (topo.n_qo,)                        # no position rows


# --------------------------------------------------------------------------
# Master-equation integration (section 5.3, A.5 special cases)
# --------------------------------------------------------------------------
def test_a5_free_vortex_gives_uniform_vm():
    # A.5 check 1: uniform h0, s and rVt = const in a straight annulus give
    # dVm/dq = 0 exactly (all PCHIP derivatives of constant data vanish).
    topo = GridTopology(cylinder_path(), n_sl=9)
    fz = make_frozen(topo, transported=uniform_transport(9, topo.n_qo,
                                                         rvt=12.0))
    asm = ResidualAssembler(fz)
    x, _ = cylinder_state(topo, vm_q0=120.0)
    fields = asm.split(x)
    np.testing.assert_allclose(fields.vm, 120.0, rtol=1e-12)


def test_v1_forced_vortex_analytic():
    # Section 9 V1 seed: forced vortex Vtheta = Omega_f * r with uniform h0,
    # s. Master equation (Tier 2): Vm^2(r) = Vm0^2 - 2 Omega_f^2 (r^2-r0^2),
    # exact for any fluid since the T ds and dh0 terms vanish.
    omega_f, vm0 = 80.0, 150.0
    topo = GridTopology(cylinder_path(), n_sl=17)
    q_rows = np.linspace(0.0, R1 - R0, 17)
    r_rows = R0 + q_rows
    rvt = np.tile((omega_f * r_rows**2)[:, None], (1, topo.n_qo))
    fz = make_frozen(topo, transported=TransportFields(
        h0=np.full_like(rvt, H0), s=np.full_like(rvt, S0), rvt=rvt))
    asm = ResidualAssembler(fz)
    x, _ = cylinder_state(topo, vm_q0=vm0)
    fields = asm.split(x)
    vm_exact = np.sqrt(vm0**2 - 2.0 * omega_f**2 * (r_rows**2 - R0**2))
    for j in range(topo.n_qo):
        np.testing.assert_allclose(fields.vm[:, j], vm_exact, rtol=1e-3)


def test_a5_curvature_term_bend_free_vortex():
    # A.5 check 2 / V2 (M1 acceptance-gate case through the assembler):
    # swirl-free curved annulus, Tier 3 -> Vm ~ 1/R_c. Also exercises the
    # A.1.1 orientation for real: q = 0 is the OUTER-bend wall (AD-9).
    topo, q, R_sl = bend_setup()
    fz = make_frozen(topo, fidelity=FidelityConfig.tier3())
    asm = ResidualAssembler(fz)
    x = pack(np.full(topo.n_qo, 100.0), q[1:-1, :])
    fields = asm.split(x)
    exact = 100.0 * R_sl[0] / R_sl
    for j in range(topo.n_qo):
        np.testing.assert_allclose(fields.vm[:, j], exact, rtol=2e-2)


def test_tier2_flags_kill_curvature_as_data():
    # Section 8 / AD-1: identical bend inputs with Tier-2 flags give uniform
    # Vm -- term deactivation is data, not a code path.
    topo, q, _ = bend_setup()
    fz = make_frozen(topo, fidelity=FidelityConfig.tier2())
    asm = ResidualAssembler(fz)
    x = pack(np.full(topo.n_qo, 100.0), q[1:-1, :])
    fields = asm.split(x)
    np.testing.assert_allclose(fields.vm, 100.0, rtol=1e-12)


def test_lean_term_wiring():
    # Section 3.1 lean term: on leaned q-o's (eps != 0) with a lagged
    # meridional Vm gradient, the lean flag must change the integration.
    topo = GridTopology(cylinder_path(lean=0.15), n_sl=7)
    n_sl, n_qo = topo.n_sl, topo.n_qo
    vm_lag = np.tile(np.linspace(100.0, 140.0, n_qo)[None, :], (n_sl, 1))
    x, _ = cylinder_state(topo, vm_q0=120.0)
    vm_on = ResidualAssembler(make_frozen(
        topo, fidelity=FidelityConfig(curvature_term=0.0, lean_term=1.0),
        vm_lagged=vm_lag)).split(x).vm
    vm_off = ResidualAssembler(make_frozen(
        topo, fidelity=FidelityConfig.tier2(),
        vm_lagged=vm_lag)).split(x).vm
    assert np.max(np.abs(vm_on - vm_off)) > 1e-3


# --------------------------------------------------------------------------
# Continuity (sections 3.2, 5.4) and capacity (A.7)
# --------------------------------------------------------------------------
def _uniform_mdot(vm, blockage=0.0):
    rho = GAS.rho(H0 - 0.5 * vm**2, S0)
    return float(np.pi * rho * vm * (1.0 - blockage) * (R1**2 - R0**2))


def test_continuity_F_uniform_analytic():
    # Section 5.4: F_j = 2*pi*integral(rho Vm cos(eps) (1-B) r dq) - mdot.
    # Uniform swirl-free state, radial q-o's: integrand linear in q, so THE
    # trapezoid rule is exact and F vanishes at the analytic mdot.
    vm, b = 120.0, 0.05
    topo = GridTopology(cylinder_path(), n_sl=9)
    fz = make_frozen(topo, mdot=_uniform_mdot(vm, b), blockage=b)
    asm = ResidualAssembler(fz)
    x, _ = cylinder_state(topo, vm_q0=vm)
    fields = asm.split(x)
    for j in range(topo.n_qo):
        F = asm.continuity_F(j, vm, fields)
        assert abs(F) / fz.spec.mdot < 1e-10
    # Subsonic branch: F increases with vm_q0 (A.7).
    assert asm.continuity_F(0, 140.0, fields) > asm.continuity_F(0, 120.0,
                                                                 fields)


def test_qo_capacity_matches_1d_choking():
    # A.7: capacity = max over Vm of the continuity integral; for a uniform
    # swirl-free q-o this is the 1-D choking flow, found here by an
    # independent dense scan of the fluid relations.
    topo = GridTopology(cylinder_path(), n_sl=9)
    fz = make_frozen(topo, mdot=50.0)
    asm = ResidualAssembler(fz)
    x, _ = cylinder_state(topo, vm_q0=100.0)
    fields = asm.split(x)
    vm_scan = np.linspace(1.0, 0.999 * np.sqrt(2 * H0), 20000)
    ref = np.max([_uniform_mdot(v) for v in vm_scan])
    cap = asm.qo_capacity(0, fields)
    assert cap == pytest.approx(ref, rel=1e-3)
    assert cap > fz.spec.mdot  # benign operating point has margin


# --------------------------------------------------------------------------
# Residual vector (section 6.1)
# --------------------------------------------------------------------------
def test_residual_zero_at_consistent_state():
    # Section 6.1: at mass-consistent positions (area rule = mass rule for a
    # uniform state) and the analytic mdot, both residual blocks vanish.
    vm = 120.0
    topo = GridTopology(cylinder_path(), n_sl=9)
    fz = make_frozen(topo, mdot=_uniform_mdot(vm))
    asm = ResidualAssembler(fz)
    q_init = initialize_positions(topo)
    x = pack(np.full(topo.n_qo, vm), q_init[1:-1, :])
    r = asm.residual(x)
    assert r.shape == (n_unknowns(topo.n_sl, topo.n_qo),)
    scale = fz.spec.mdot / (2.0 * np.pi)
    assert np.max(np.abs(r)) / scale < 1e-4


def test_residual_ordering_matches_pack():
    # Section 6.1 / ARCH-3.2: rows are [R_cont_j all j] then interior R_pos
    # in C-order. Perturbing mdot shifts every continuity row by -d(mdot)
    # and every position row by -psi_i*d(mdot)/(2*pi).
    vm = 120.0
    topo = GridTopology(cylinder_path(), n_sl=5)
    n_qo = topo.n_qo
    q_init = initialize_positions(topo)
    x = pack(np.full(n_qo, vm), q_init[1:-1, :])
    mdot = _uniform_mdot(vm)
    d = 1.0
    r_a = ResidualAssembler(make_frozen(topo, mdot=mdot)).residual(x)
    r_b = ResidualAssembler(make_frozen(topo, mdot=mdot + d)).residual(x)
    diff = r_a - r_b
    np.testing.assert_allclose(diff[:n_qo], d, rtol=1e-12)
    psi_int = topo.psi[1:-1]
    expect_pos = np.tile(psi_int[:, None] * d / (2 * np.pi), (1, n_qo))
    np.testing.assert_allclose(diff[n_qo:], expect_pos.ravel(), rtol=1e-12)


def test_residual_pure_and_deterministic():
    # AD-3: same (x, FrozenInputs) -> identical residual; x never mutated.
    topo = GridTopology(cylinder_path(), n_sl=7)
    asm = ResidualAssembler(make_frozen(topo, mdot=_uniform_mdot(120.0)))
    x, _ = cylinder_state(topo, vm_q0=120.0)
    x_copy = x.copy()
    r1, r2 = asm.residual(x), asm.residual(x)
    assert np.array_equal(r1, r2)
    assert np.array_equal(x, x_copy)


def test_split_thermo_consistent():
    # ARCH-5.1 split: thermodynamics from the (h, s) pair (section 3.7);
    # spot-check h = h0 - V^2/2 and the meridional Mach number.
    topo = GridTopology(cylinder_path(), n_sl=5)
    rvt_val = 15.0
    fz = make_frozen(topo, transported=uniform_transport(5, topo.n_qo,
                                                         rvt=rvt_val))
    asm = ResidualAssembler(fz)
    x, q_rows = cylinder_state(topo, vm_q0=100.0)
    fields = asm.split(x)
    r = fields.metrics.r
    h_expect = H0 - 0.5 * (fields.vm**2 + (rvt_val / r) ** 2)
    np.testing.assert_allclose(fields.h, h_expect, rtol=1e-12)
    np.testing.assert_allclose(
        fields.mach_m, fields.vm / GAS.a(fields.h, S0), rtol=1e-12)
