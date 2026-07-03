"""WallCurve tests against analytic geometries (Grid & Geometry Spec G-8.1,
G-8.3) plus construction validation and parameterization quality.

Sign convention pin (G-6.1.3): a circular arc turning axial -> radial with
increasing phi must report kappa = +1/R under the A.1 convention. This test is
the permanent guard on the curvature sign; if it ever needs "fixing", the bug
is elsewhere.
"""
import numpy as np
import pytest

from slcflow.errors import ConfigError
from slcflow.geometry import WallCurve


# --------------------------------------------------------------------------
# Analytic cases
# --------------------------------------------------------------------------
def test_cylindrical_wall_exact():
    """Straight cylinder: phi = 0, kappa = 0 to near machine precision
    (cubic splines reproduce straight lines exactly)."""
    z = np.linspace(0.0, 1.0, 15)
    wall = WallCurve.from_points(np.column_stack([z, np.full_like(z, 0.5)]))
    s = np.linspace(0.0, wall.arclength, 40)
    assert wall.arclength == pytest.approx(1.0, rel=1e-9)
    assert np.allclose(wall.slope_phi(s), 0.0, atol=1e-12)
    assert np.allclose(wall.curvature(s), 0.0, atol=1e-9)
    zs, rs = wall.point(s)
    assert np.allclose(rs, 0.5, atol=1e-12)
    assert np.allclose(zs, s, atol=1e-9)  # arc length == z for a cylinder


def test_conical_wall_exact():
    """Cone: phi = const = atan(dr/dz), kappa = 0."""
    slope = 0.35
    z = np.linspace(0.0, 2.0, 12)
    wall = WallCurve.from_points(np.column_stack([z, 0.3 + slope * z]))
    s = np.linspace(0.0, wall.arclength, 30)
    assert np.allclose(wall.slope_phi(s), np.arctan(slope), atol=1e-12)
    assert np.allclose(wall.curvature(s), 0.0, atol=1e-9)
    assert wall.arclength == pytest.approx(2.0 * np.hypot(1.0, slope), rel=1e-9)


def test_circular_arc_sign_and_magnitude():
    """Quarter circle turning axial -> radial (phi: 0 -> 90 deg).

    With theta in [0, pi/2]: z = zc + R sin(theta), r = rc - R cos(theta);
    phi = theta increases along the curve, so kappa = dphi/dm = +1/R exactly.
    THE SIGN HERE IS NORMATIVE (A.1 / G-6.1.3).
    """
    R, zc, rc = 0.4, 0.0, 0.6

    def arc(u):
        th = 0.5 * np.pi * u
        return zc + R * np.sin(th), rc - R * np.cos(th)

    wall = WallCurve.from_callable(arc, n=161)
    s = np.linspace(0.02, wall.arclength - 0.02, 50)  # avoid natural-BC ends
    assert wall.arclength == pytest.approx(0.5 * np.pi * R, rel=1e-6)
    assert np.allclose(wall.curvature(s), 1.0 / R, rtol=2e-4)
    # phi(sigma) = sigma / R for this arc
    assert np.allclose(wall.slope_phi(s), s / R, atol=1e-5)


def test_reversed_arc_gives_negative_curvature():
    """Same arc traversed radial -> axial: phi decreases, kappa = -1/R."""
    R = 0.4

    def arc(u):
        th = 0.5 * np.pi * (1.0 - u)  # reverse direction
        return R * np.sin(th), 0.6 - R * np.cos(th)

    wall = WallCurve.from_callable(arc, n=161)
    s = np.linspace(0.02, wall.arclength - 0.02, 50)
    assert np.allclose(wall.curvature(s), -1.0 / R, rtol=2e-4)


def test_ninety_degree_bend_through_vertical():
    """Axial->radial 90-degree bend: phi passes through +pi/2 without
    representation failure -- the reason parametric form is mandatory (G-6.1.2).
    Checks phi is monotone through the pole and reaches ~90 deg."""
    R = 0.5

    def bend(u):
        th = 0.5 * np.pi * u
        return R * np.sin(th), 0.2 + R * (1.0 - np.cos(th))

    wall = WallCurve.from_callable(bend, n=201)
    s = np.linspace(0.0, wall.arclength, 200)
    phi = wall.slope_phi(s)
    assert np.all(np.diff(phi) > 0)                       # monotone turning
    assert phi[-1] == pytest.approx(0.5 * np.pi, abs=1e-3)  # reaches radial
    ez, er = wall.unit_tangent(wall.arclength)
    assert abs(ez) < 1e-3 and er == pytest.approx(1.0, abs=1e-4)


# --------------------------------------------------------------------------
# Convergence order (G-8.3)
# --------------------------------------------------------------------------
def test_curvature_convergence_order_on_arc():
    """Interpolating-fit curvature error must shrink at ~2nd order or better
    with point count (cubic spline second-derivative accuracy)."""
    R = 0.4

    def arc(u):
        th = 0.5 * np.pi * u
        return R * np.sin(th), 0.6 - R * np.cos(th)

    errs = []
    for n in (21, 41, 81):
        wall = WallCurve.from_callable(arc, n=n)
        s = np.linspace(0.1, wall.arclength - 0.1, 25)  # interior only
        errs.append(np.max(np.abs(wall.curvature(s) - 1.0 / R)))
    order1 = np.log2(errs[0] / errs[1])
    order2 = np.log2(errs[1] / errs[2])
    assert order1 > 1.5 and order2 > 1.5, f"orders {order1:.2f}, {order2:.2f}"


# --------------------------------------------------------------------------
# Arc-length parameterization quality (G-3.2)
# --------------------------------------------------------------------------
def test_unit_speed_parameterization():
    """|dP/dsigma| must be ~1: sigma is true arc length."""
    def bend(u):
        th = 0.5 * np.pi * u
        return 0.5 * np.sin(th), 0.2 + 0.5 * (1.0 - np.cos(th))

    wall = WallCurve.from_callable(bend, n=201)
    s = np.linspace(0.01, wall.arclength - 0.01, 60)
    ds = 1e-6 * wall.arclength
    z1, r1 = wall.point(s - ds)
    z2, r2 = wall.point(s + ds)
    speed = np.hypot(z2 - z1, r2 - r1) / (2 * ds)
    assert np.allclose(speed, 1.0, atol=2e-4)


def test_nonuniform_input_spacing_handled():
    """Badly clustered input points must still yield arc-length access."""
    u = np.linspace(0.0, 1.0, 30) ** 3  # heavy clustering at start
    z = 2.0 * u
    r = 0.4 + 0.1 * u
    wall = WallCurve.from_points(np.column_stack([z, r]))
    assert wall.arclength == pytest.approx(2.0 * np.hypot(1.0, 0.05), rel=1e-6)
    zs, _ = wall.point(0.5 * wall.arclength)
    assert zs == pytest.approx(1.0, abs=1e-3)  # midpoint in arc length


# --------------------------------------------------------------------------
# Smoothing path
# --------------------------------------------------------------------------
def test_smoothing_suppresses_noise_curvature():
    """Noisy cylinder: interpolating fit produces large spurious curvature;
    smoothing fit must reduce it by orders of magnitude (G-6.2 rationale)."""
    rng = np.random.default_rng(7)
    z = np.linspace(0.0, 1.0, 60)
    r = 0.5 + rng.normal(0.0, 2e-4, size=z.size)
    pts = np.column_stack([z, r])
    interp = WallCurve.from_points(pts)
    smooth = WallCurve.from_points(pts, smoothing=1e-4)
    s = np.linspace(0.05, 0.95, 40)
    k_interp = np.max(np.abs(interp.curvature(s)))
    k_smooth = np.max(np.abs(smooth.curvature(s)))
    assert k_smooth < 0.05 * k_interp
    assert k_smooth < 0.5  # sane absolute level for a nominally straight wall


# --------------------------------------------------------------------------
# Validation (config boundary, AD-10)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("pts,msg", [
    (np.array([[0, 1], [1, 1], [2, 1]]), "at least 4"),
    (np.array([[0, 1], [0, 1], [1, 1], [2, 1]]), "repeated"),
    (np.array([[0, 1], [1, np.nan], [2, 1], [3, 1]]), "non-finite"),
    (np.array([[0, 1], [1, -0.1], [2, 1], [3, 1]]), "negative radius"),
    (np.array([0.0, 1.0, 2.0, 3.0]), "shape"),
])
def test_invalid_points_raise(pts, msg):
    with pytest.raises(ConfigError, match=msg):
        WallCurve.from_points(pts)


def test_invalid_smoothing_raises():
    z = np.linspace(0, 1, 6)
    with pytest.raises(ConfigError, match="smoothing"):
        WallCurve.from_points(np.column_stack([z, np.ones_like(z)]), smoothing=-1.0)


def test_sigma_out_of_range_raises():
    z = np.linspace(0, 1, 6)
    wall = WallCurve.from_points(np.column_stack([z, np.ones_like(z)]))
    with pytest.raises(ValueError, match="sigma outside"):
        wall.point(wall.arclength * 1.5)


def test_natural_bc_endpoint_curvature_pollution_documented():
    """Pin the G-3.4 finding: natural BCs force kappa -> 0 at curve ends.
    This is a *characterization* test -- it documents behavior we chose to
    route around (from_callable uses not-a-knot), so a future change to end
    handling is a conscious decision, not an accident."""
    R = 0.4

    def arc(u):
        th = 0.5 * np.pi * u
        return R * np.sin(th), 0.6 - R * np.cos(th)

    natural = WallCurve.from_callable(arc, n=161, bc_type="natural")
    # exactly at the endpoint, natural BC forces zero curvature:
    assert abs(natural.curvature(0.0)) < 1e-8
    # while not-a-knot recovers the true value there:
    nak = WallCurve.from_callable(arc, n=161)
    assert nak.curvature(0.0) == pytest.approx(1.0 / R, rel=1e-3)