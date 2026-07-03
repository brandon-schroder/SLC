"""Quasi-orthogonal curves, station definitions, and the FlowPath assembly
(Grid & Geometry Spec G-2, G-4; Theory Manual sections 2.3, 2.5, A.1.1).

The FlowPath is the configuration boundary for meridional topology: it owns
the two labeled walls, the ordered station definitions, and the constructed
q-o curves -- including the normative A.1.1 orientation decision (the q-o
parameter must run such that ``e_q . e_n >= 0`` against the expected flow
direction, which is NOT always wall_0 -> wall_1).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np  # geometry layer binds numpy directly  # ad6: allow

from ..errors import ConfigError
from .wallcurve import WallCurve

__all__ = ["StationType", "StationDef", "StraightQO", "FlowPath"]


class StationType(enum.Enum):
    DUCT = "duct"
    EDGE_LE = "edge_le"
    EDGE_TE = "edge_te"
    INBLADE = "inblade"


@dataclass(frozen=True)
class StationDef:
    """One computing station: type, per-wall anchors as *fractional wall arc
    length* in [0, 1] (G-4 primary form), and the owning row id (blade
    stations only)."""

    stype: StationType
    anchor_w0: float
    anchor_w1: float
    row_id: str | None = None

    def __post_init__(self):
        for name, a in (("anchor_w0", self.anchor_w0), ("anchor_w1", self.anchor_w1)):
            if not (0.0 <= a <= 1.0):
                raise ConfigError(f"{name} must be in [0, 1], got {a}")
        is_blade = self.stype in (StationType.EDGE_LE, StationType.EDGE_TE,
                                  StationType.INBLADE)
        if is_blade and self.row_id is None:
            raise ConfigError(f"{self.stype.value} station requires a row_id")
        if not is_blade and self.row_id is not None:
            raise ConfigError("DUCT station must not carry a row_id")


@dataclass(frozen=True)
class StraightQO:
    """Straight quasi-orthogonal segment, parameterized by arc length
    ``q in [0, length]`` from its origin end (A.1.1-oriented by FlowPath)."""

    p_origin: tuple[float, float]   # (z, r) at q = 0
    p_end: tuple[float, float]      # (z, r) at q = length
    origin_wall: int                # 0 or 1: which physical wall sits at q = 0

    @property
    def length(self) -> float:
        dz = self.p_end[0] - self.p_origin[0]
        dr = self.p_end[1] - self.p_origin[1]
        return float(np.hypot(dz, dr))

    @property
    def unit_tangent(self) -> tuple[float, float]:
        """Constant unit tangent ``e_q = (e_z, e_r)`` (straight segment)."""
        L = self.length
        return ((self.p_end[0] - self.p_origin[0]) / L,
                (self.p_end[1] - self.p_origin[1]) / L)

    def point(self, q):
        """Meridional coordinates ``(z, r)`` at arc length(s) ``q``."""
        q = np.asarray(q, dtype=float)
        ez, er = self.unit_tangent
        return self.p_origin[0] + q * ez, self.p_origin[1] + q * er


def _segments_cross(a0, a1, b0, b1) -> bool:
    """Proper intersection test for 2-D segments (shared endpoints excluded)."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = orient(b0, b1, a0), orient(b0, b1, a1)
    d3, d4 = orient(a0, a1, b0), orient(a0, a1, b1)
    return (d1 * d2 < 0.0) and (d3 * d4 < 0.0)


class FlowPath:
    """Meridional flow-path topology: two labeled walls + ordered stations.

    Construction performs all G-2/G-4 validation and the A.1.1 orientation
    decision; the result exposes the oriented q-o curves via ``.qo_curves``.

    Parameters
    ----------
    wall_0, wall_1 : the two annulus walls. Labels are *physical* (AD-9): the
        machine definition maps them to hub/shroud; this class never assumes
        which is which, and ``q = 0`` may sit on either (A.1.1).
    stations : ordered station definitions, upstream to downstream.
    """

    def __init__(self, wall_0: WallCurve, wall_1: WallCurve, stations):
        self.wall_0, self.wall_1 = wall_0, wall_1
        self.stations = tuple(stations)
        self._validate_station_ordering()
        self.qo_curves, self.q_origin_wall = self._build_oriented_qos()
        self._validate_non_crossing()

    # ------------------------------------------------------------------
    def _validate_station_ordering(self):
        if len(self.stations) < 2:
            raise ConfigError("need at least 2 stations")
        for w, attr in ((0, "anchor_w0"), (1, "anchor_w1")):
            a = np.array([getattr(s, attr) for s in self.stations])
            if np.any(np.diff(a) <= 0.0):
                raise ConfigError(
                    f"station anchors on wall_{w} must be strictly increasing "
                    f"(upstream -> downstream); got {a.tolist()}"
                )

    def _anchor_points(self, station: StationDef):
        p0 = self.wall_0.point(station.anchor_w0 * self.wall_0.arclength)
        p1 = self.wall_1.point(station.anchor_w1 * self.wall_1.arclength)
        return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))

    def _build_oriented_qos(self):
        """Construct q-o's with the A.1.1 orientation: choose the q direction
        so that ``e_q . e_n >= 0``, where ``e_n`` is the +90-degree rotation of
        the *expected* meridional flow direction ``e_m`` (mean of the two wall
        tangents at the anchors). All stations must agree on the direction;
        a mixed result indicates a pathological station layout."""
        flips = []
        for s in self.stations:
            p0, p1 = self._anchor_points(s)
            t0 = self.wall_0.unit_tangent(s.anchor_w0 * self.wall_0.arclength)
            t1 = self.wall_1.unit_tangent(s.anchor_w1 * self.wall_1.arclength)
            em = np.array([t0[0] + t1[0], t0[1] + t1[1]])
            em = em / np.hypot(*em)
            en = np.array([-em[1], em[0]])          # +90-degree rotation (A.1)
            eq_01 = np.array([p1[0] - p0[0], p1[1] - p0[1]])
            flips.append(float(eq_01 @ en) < 0.0)
        if all(flips):
            origin = 1
        elif not any(flips):
            origin = 0
        else:
            raise ConfigError(
                "inconsistent q-o orientation across stations (A.1.1): "
                f"per-station flip flags {flips}; check station anchors"
            )
        qos = []
        for s in self.stations:
            p0, p1 = self._anchor_points(s)
            if origin == 0:
                qos.append(StraightQO(p_origin=p0, p_end=p1, origin_wall=0))
            else:
                qos.append(StraightQO(p_origin=p1, p_end=p0, origin_wall=1))
        return tuple(qos), origin

    def _validate_non_crossing(self):
        for a, b in zip(self.qo_curves[:-1], self.qo_curves[1:]):
            if _segments_cross(a.p_origin, a.p_end, b.p_origin, b.p_end):
                raise ConfigError(
                    "adjacent quasi-orthogonals intersect within the annulus "
                    "(G-4.3); check station anchor definitions"
                )