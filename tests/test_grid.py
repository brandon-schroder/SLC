"""FlowPath, topology, initialization, and metric tests (G-8.1/4/6/7), closing
with the M1 acceptance gate: the frozen-streamline V2 curved-annulus case.

The V2 case is deliberately chosen to exercise the A.1.1 orientation rule for
real: in a 90-degree axial->radial bend, the streamline normal e_n points
toward the bend center, so the q-o parameter must run from the *outer-bend*
wall inward -- q = 0 is NOT the "hub". The analytic solution of the master
equation's curvature term is the meridional free vortex Vm ~ 1/R_c.
"""
import numpy as np
import pytest

from slcflow.errors import ConfigError
from slcflow.geometry import FlowPath, StationDef, StationType, WallCurve
from slcflow.grid import (
    GridTopology,
    cumulative,
    evaluate_metrics,
    initialize_positions,
)


# --------------------------------------------------------------------------
# Fixtures: analytic flow paths
# --------------------------------------------------------------------------
def cylinder_path(n_stations=6, r0=0.3, r1=0.6, length=1.0):
    z = np.linspace(0.0, length, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, r0)]), name="hub")
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, r1)]), name="casing")
    fracs = np.linspace(0.0, 1.0, n_stations)
    stations = [StationDef(StationType.DUCT, f, f) for f in fracs]
    return FlowPath(w0, w1, stations)


BEND_CENTER = (0.0, 0.8)
R_INNERBEND, R_OUTERBEND = 0.2, 0.5  # bend radii of the two walls


def bend_path(n_stations=13):
    """Concentric 90-degree bend (axial -> radial). wall_0 is the small-bend-
    radius wall (closer to the bend center, larger r at inlet)."""
    zc, rc = BEND_CENTER

    def wall(R):
        return lambda u: (zc + R * np.sin(0.5 * np.pi * u),
                          rc - R * np.cos(0.5 * np.pi * u))

    w0 = WallCurve.from_callable(wall(R_INNERBEND), n=201, name="innerbend")
    w1 = WallCurve.from_callable(wall(R_OUTERBEND), n=201, name="outerbend")
    fracs = np.linspace(0.0, 1.0, n_stations)
    stations = [StationDef(StationType.DUCT, f, f) for f in fracs]
    return FlowPath(w0, w1, stations)


# --------------------------------------------------------------------------
# StationDef / FlowPath validation
# --------------------------------------------------------------------------
def test_station_validation():
    with pytest.raises(ConfigError, match="anchor"):
        StationDef(StationType.DUCT, -0.1, 0.5)
    with pytest.raises(ConfigError, match="row_id"):
        StationDef(StationType.EDGE_LE, 0.2, 0.2)
    with pytest.raises(ConfigError, match="row_id"):
        StationDef(StationType.DUCT, 0.2, 0.2, row_id="r1")
    ok = StationDef(StationType.EDGE_TE, 0.4, 0.45, row_id="rotor1")
    assert ok.row_id == "rotor1"


def test_nonmonotone_anchors_raise():
    z = np.linspace(0, 1, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, 0.3)]))
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, 0.6)]))
    stations = [StationDef(StationType.DUCT, 0.5, 0.5),
                StationDef(StationType.DUCT, 0.3, 0.7)]
    with pytest.raises(ConfigError, match="strictly increasing"):
        FlowPath(w0, w1, stations)


def test_crossing_predicate_directly():
    """The G-4.3 non-crossing guard, unit-tested at the predicate level.

    Rationale: with straight q-o's and *strictly monotone anchors* (already
    enforced), adjacent segments can only cross for concave wall geometry --
    the anchor-monotonicity check fires first in simple layouts, so the
    predicate is tested directly here and stays armed for curved-wall cases.
    """
    from slcflow.geometry.flowpath import _segments_cross

    # proper crossing
    assert _segments_cross((0, 0), (1, 1), (0, 1), (1, 0))
    # disjoint
    assert not _segments_cross((0, 0), (1, 0), (0, 1), (1, 1))
    # shared endpoint is not a "crossing" (adjacent q-o's may meet at a wall)
    assert not _segments_cross((0, 0), (1, 1), (1, 1), (2, 0))
    # collinear overlap: excluded by the proper-intersection definition
    assert not _segments_cross((0, 0), (2, 0), (1, 0), (3, 0))


def test_cylinder_orientation_no_flip():
    fp = cylinder_path()
    assert fp.q_origin_wall == 0  # e_n = +r; wall_0 -> wall_1 is +r already
    for qo in fp.qo_curves:
        assert qo.unit_tangent[1] == pytest.approx(1.0)  # radial q-o's


def test_bend_orientation_flips():
    """A.1.1 in action: e_n points toward the bend center, so q must run from
    the outer-bend wall (wall_1) inward. q = 0 is not wall_0 here."""
    fp = bend_path()
    assert fp.q_origin_wall == 1
    for qo in fp.qo_curves:
        assert qo.origin_wall == 1
        assert qo.length == pytest.approx(R_OUTERBEND - R_INNERBEND, rel=1e-4)


# --------------------------------------------------------------------------
# Initialization (G-5)
# --------------------------------------------------------------------------
def test_initialize_positions_cylinder_analytic():
    """Equal integral of r dq on a radial q-o: r_i = sqrt(psi*(r1^2-r0^2)+r0^2)."""
    fp = cylinder_path()
    topo = GridTopology(fp, n_sl=7)
    q = initialize_positions(topo)
    r0, r1 = 0.3, 0.6
    r_exact = np.sqrt(topo.psi * (r1**2 - r0**2) + r0**2)
    assert q.shape == (7, len(fp.qo_curves))
    for j in range(q.shape[1]):
        assert np.allclose(q[:, j] + r0, r_exact, atol=2e-6)


def test_single_streamline_topology():
    """Tier-1 degeneration: n_sl = 1 sits at the mid mass fraction (G-6.4)."""
    fp = cylinder_path()
    topo = GridTopology(fp, n_sl=1)
    assert topo.psi.tolist() == [0.5]
    q = initialize_positions(topo)
    assert q.shape == (1, len(fp.qo_curves))


# --------------------------------------------------------------------------
# Metrics (G-6)
# --------------------------------------------------------------------------
def test_metrics_cylinder_exact():
    fp = cylinder_path()
    topo = GridTopology(fp, n_sl=5)
    q = initialize_positions(topo)
    met = evaluate_metrics(topo, q)
    assert np.allclose(met.phi, 0.0, atol=1e-10)
    assert np.allclose(met.kappa_m, 0.0, atol=1e-8)
    assert np.allclose(met.eps, 0.0, atol=1e-10)
    assert met.cos_eps_ok.all()
    # meridional arc length equals z for axial streamlines
    assert np.allclose(met.m, met.z - met.z[:, [0]], atol=1e-9)


def test_metrics_purity_and_shape_validation():
    fp = cylinder_path()
    topo = GridTopology(fp, n_sl=5)
    q = initialize_positions(topo)
    q_copy = q.copy()
    m1 = evaluate_metrics(topo, q)
    m2 = evaluate_metrics(topo, q)
    assert np.array_equal(q, q_copy)                      # no mutation
    assert np.array_equal(m1.kappa_m, m2.kappa_m)         # deterministic
    with pytest.raises(ConfigError, match="shape"):
        evaluate_metrics(topo, q[:, :-1])


def test_metrics_degrade_below_four_stations():
    """2-3 stations: chord tangents, kappa reported zero (documented G-6.4/
    core docstring behavior for edge-only meanline layouts)."""
    fp = cylinder_path(n_stations=3)
    topo = GridTopology(fp, n_sl=3)
    q = initialize_positions(topo)
    met = evaluate_metrics(topo, q)
    assert np.allclose(met.kappa_m, 0.0)
    assert np.allclose(met.phi, 0.0, atol=1e-12)


# --------------------------------------------------------------------------
# M1 ACCEPTANCE GATE: V2 frozen-streamline curved annulus
# --------------------------------------------------------------------------
def _bend_frozen_setup(n_sl=9, n_stations=13):
    fp = bend_path(n_stations)
    topo = GridTopology(fp, n_sl=n_sl)
    # Frozen concentric streamlines at prescribed bend radii; q measured from
    # the outer-bend wall (orientation-flipped origin): q_i = R_outer - R_i.
    R_sl = np.linspace(R_OUTERBEND, R_INNERBEND, n_sl)
    q = np.tile((R_OUTERBEND - R_sl)[:, None], (1, len(fp.qo_curves)))
    return topo, q, R_sl


def test_v2_metrics_on_frozen_bend():
    topo, q, R_sl = _bend_frozen_setup()
    met = evaluate_metrics(topo, q)
    theta = np.linspace(0.0, 0.5 * np.pi, topo.n_qo)

    assert met.cos_eps_ok.all()
    assert np.allclose(met.eps, 0.0, atol=2e-3)           # radial rays
    interior = slice(2, -2)
    # phi = theta along each streamline
    assert np.allclose(met.phi[:, interior],
                       np.broadcast_to(theta, met.phi.shape)[:, interior],
                       atol=2e-3)
    # kappa = +1/R_c, sign per A.1 (turning axial->radial)
    kappa_exact = (1.0 / R_sl)[:, None]
    rel = np.abs(met.kappa_m[:, interior] / kappa_exact - 1.0)
    assert np.max(rel) < 5e-3, f"max kappa rel err {np.max(rel):.2e}"
    # meridional arc length: m = R_c * theta
    m_exact = R_sl[:, None] * theta[None, :]
    assert np.allclose(met.m, m_exact, rtol=2e-3, atol=1e-5)


def test_v2_master_ode_curvature_term_gives_free_vortex():
    """A.5 special case 2 on real machinery: integrating
    d(ln Vm)/dq = kappa_m cos(eps) along each q-o using *computed metrics*
    (and THE shared quadrature) must recover Vm ~ 1/R_c."""
    topo, q, R_sl = _bend_frozen_setup()
    met = evaluate_metrics(topo, q)
    for j in range(topo.n_qo):
        integrand = met.kappa_m[:, j] * np.cos(met.eps[:, j])
        ln_ratio = cumulative(integrand, q[:, j])          # ln(Vm_i / Vm_0)
        vm_ratio = np.exp(ln_ratio)
        exact = R_sl[0] / R_sl                             # 1/R_c profile
        # measured worst-station error 1.35e-2 at (n_sl=9, n_st=13);
        # converges at 2nd order under coupled refinement (see next test)
        assert np.allclose(vm_ratio, exact, rtol=2e-2), (
            f"station {j}: max rel err "
            f"{np.max(np.abs(vm_ratio / exact - 1)):.2e}"
        )


def test_v2_refinement_second_order():
    """V2 error must converge at ~2nd order under COUPLED refinement.

    Spanwise (n_sl) and streamwise (n_stations) resolution must be refined
    together: the total error is trapezoid quadrature (scales with spanwise
    spacing^2) plus streamline-fit curvature error (scales with streamwise
    spacing^2). Refining only one direction plateaus on the other's floor --
    measured: quadrature 9.7e-3 -> 6.1e-4 over 4x n_sl while kappa error sat
    fixed at 1.4e-3 with frozen n_stations. Coupled refinement measured:
    5.29e-2 / 1.35e-2 / 3.38e-3 (ratios 3.92, 3.99 = clean 2nd order)."""
    errs = []
    for n_sl, n_st in ((5, 7), (9, 13), (17, 25)):
        topo, q, R_sl = _bend_frozen_setup(n_sl=n_sl, n_stations=n_st)
        met = evaluate_metrics(topo, q)
        worst = 0.0
        for j in range(topo.n_qo):
            integrand = met.kappa_m[:, j] * np.cos(met.eps[:, j])
            vm_ratio = np.exp(cumulative(integrand, q[:, j]))
            worst = max(worst, np.max(np.abs(vm_ratio / (R_sl[0] / R_sl) - 1.0)))
        errs.append(worst)
    order = np.log2(errs[0] / errs[2]) / 2.0
    assert order > 1.7, f"observed order {order:.2f}, errors {errs}"