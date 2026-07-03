"""Grid topology, streamline initialization, and metric evaluation
(Grid & Geometry Spec G-5, G-6; Theory Manual sections 2.5, 5.1-5.2).

``evaluate_metrics`` is the pure function of ARCH-3.2: given the frozen
topology and nodal q-positions, return the metric bundle. No retained state,
no mutation of inputs (AD-3/AD-6 discipline on the residual path).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # grid layer is numpy-bound via scipy splines  # ad6: allow
from scipy.interpolate import CubicSpline, make_smoothing_spline

from ..errors import ConfigError
from ..geometry.flowpath import FlowPath
from .quadrature import cumulative, invert_cumulative

__all__ = ["GridTopology", "MetricsConfig", "GridMetrics",
           "initialize_positions", "evaluate_metrics"]

_DENSE_INIT = 400        # dense samples per q-o for the area-rule inversion
_DENSE_ARC = 12          # dense samples per streamline interval for arc length


@dataclass(frozen=True)
class GridTopology:
    """Immutable grid topology (AD-8): the flow path with its oriented q-o's,
    the streamline count, and the fixed mass fractions ``psi`` (endpoints 0
    and 1 are the walls)."""

    flowpath: FlowPath
    n_sl: int
    psi: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.n_sl < 1:
            raise ConfigError(f"n_sl must be >= 1, got {self.n_sl}")
        if self.psi is None:
            # Uniform mass fractions including walls; a single streamline sits
            # at the mid mass fraction (Tier 1, G-6.4).
            psi = (np.array([0.5]) if self.n_sl == 1
                   else np.linspace(0.0, 1.0, self.n_sl))
            object.__setattr__(self, "psi", psi)
        psi = np.asarray(self.psi, dtype=float)
        if psi.shape != (self.n_sl,) or np.any(np.diff(psi) <= 0.0) \
                or np.any(psi < 0.0) or np.any(psi > 1.0):
            raise ConfigError("psi must be strictly increasing in [0, 1] with length n_sl")

    @property
    def n_qo(self) -> int:
        return len(self.flowpath.qo_curves)


@dataclass(frozen=True)
class MetricsConfig:
    """Streamline-fit settings (G-6.1, G-6.2).

    ``bc_type='not-a-knot'`` default per the measured G-3.4 finding (natural
    end conditions force zero end curvature). ``smoothing`` is the [DECIDE]
    default pending the G-8.5 noise sweep: ``None`` (interpolating) for now.
    """

    bc_type: str = "not-a-knot"
    smoothing: float | None = None


@dataclass(frozen=True)
class GridMetrics:
    """Nodal metric bundle, all arrays shaped ``(n_sl, n_qo)`` (G-6.3)."""

    z: np.ndarray
    r: np.ndarray
    phi: np.ndarray          # streamline slope from axial (rad)
    kappa_m: np.ndarray      # signed meridional curvature d(phi)/dm (1/m)
    eps: np.ndarray          # q-o lean from streamline normal (rad)
    m: np.ndarray            # meridional arc length along each streamline
    qo_length: np.ndarray    # (n_qo,) arc length of each q-o
    cos_eps_ok: np.ndarray   # A.1.1 diagnostic: cos(eps) >= 0 per node


# ---------------------------------------------------------------------------
# Initialization (G-5)
# ---------------------------------------------------------------------------
def initialize_positions(topology: GridTopology) -> np.ndarray:
    """Initial nodal q-positions by the annulus-area rule (G-5): on each q-o,
    place streamline ``i`` where the cumulative ``integral of r dq`` reaches
    fraction ``psi_i`` -- exact for uniform ``rho Vm cos(eps)``. Uses THE
    shared quadrature rule (G-5 consistency requirement)."""
    cols = []
    for qo in topology.flowpath.qo_curves:
        q_dense = np.linspace(0.0, qo.length, _DENSE_INIT)
        _, r_dense = qo.point(q_dense)
        area = cumulative(r_dense, q_dense)
        cols.append(invert_cumulative(q_dense, area, topology.psi * area[-1]))
    return np.stack(cols, axis=1)  # (n_sl, n_qo)


# ---------------------------------------------------------------------------
# Metric evaluation (G-6) -- pure function of (topology, q_positions)
# ---------------------------------------------------------------------------
def _fit_streamline(z_nodes, r_nodes, config: MetricsConfig):
    """Parametric C2 fit through one streamline's nodes; returns per-node
    (phi, kappa, m). Chord-length parameter; parametric forms only (G-6.1.2).

    Graceful degradation (documented): with fewer than 4 stations a cubic fit
    is impossible -- tangents come from chord directions and curvature is
    reported as zero. Tiers that need curvature must supply >= 4 stations.
    """
    chord = np.hypot(np.diff(z_nodes), np.diff(r_nodes))
    if np.any(chord == 0.0):
        raise ConfigError("coincident adjacent streamline nodes")
    t = np.concatenate(([0.0], np.cumsum(chord)))
    t = t / t[-1]
    n = z_nodes.size

    if n < 4:
        dz = np.gradient(z_nodes, t)
        dr = np.gradient(r_nodes, t)
        phi = np.arctan2(dr, dz)
        kappa = np.zeros(n)
        m = np.concatenate(([0.0], np.cumsum(chord)))
        return phi, kappa, m

    if config.smoothing is None:
        sz = CubicSpline(t, z_nodes, bc_type=config.bc_type)
        sr = CubicSpline(t, r_nodes, bc_type=config.bc_type)
    else:
        sz = make_smoothing_spline(t, z_nodes, lam=config.smoothing)
        sr = make_smoothing_spline(t, r_nodes, lam=config.smoothing)

    zp, rp = sz(t, 1), sr(t, 1)
    zpp, rpp = sz(t, 2), sr(t, 2)
    phi = np.arctan2(rp, zp)
    speed2 = zp * zp + rp * rp
    kappa = (zp * rpp - rp * zpp) / speed2**1.5

    # Meridional arc length at the nodes via dense quadrature of the fit.
    t_dense = np.linspace(0.0, 1.0, _DENSE_ARC * (n - 1) + 1)
    speed_dense = np.hypot(sz(t_dense, 1), sr(t_dense, 1))
    m_dense = cumulative(speed_dense, t_dense)
    m = np.interp(t, t_dense, m_dense)
    return phi, kappa, m


def evaluate_metrics(topology: GridTopology, q_positions,
                     config: MetricsConfig = MetricsConfig()) -> GridMetrics:
    """Metric fields from nodal positions (pure; ARCH-3.2 contract).

    Parameters
    ----------
    q_positions : (n_sl, n_qo) arc-length positions along each q-o.
    """
    q = np.asarray(q_positions, dtype=float)
    n_sl, n_qo = topology.n_sl, topology.n_qo
    if q.shape != (n_sl, n_qo):
        raise ConfigError(f"q_positions shape {q.shape} != ({n_sl}, {n_qo})")

    qos = topology.flowpath.qo_curves
    zr = [qo.point(q[:, j]) for j, qo in enumerate(qos)]
    z = np.stack([c[0] for c in zr], axis=1)
    r = np.stack([c[1] for c in zr], axis=1)

    fits = [_fit_streamline(z[i, :], r[i, :], config) for i in range(n_sl)]
    phi = np.stack([f[0] for f in fits], axis=0)
    kappa = np.stack([f[1] for f in fits], axis=0)
    m = np.stack([f[2] for f in fits], axis=0)

    # Lean angle from tangent dot products, never angle arithmetic (G-9).
    et = np.array([qo.unit_tangent for qo in qos])          # (n_qo, 2)
    em_z, em_r = np.cos(phi), np.sin(phi)
    en_z, en_r = -np.sin(phi), np.cos(phi)                  # +90-deg rotation
    sin_eps = et[None, :, 0] * em_z + et[None, :, 1] * em_r
    cos_eps = et[None, :, 0] * en_z + et[None, :, 1] * en_r
    eps = np.arctan2(sin_eps, cos_eps)

    return GridMetrics(
        z=z, r=r, phi=phi, kappa_m=kappa, eps=eps, m=m,
        qo_length=np.array([qo.length for qo in qos]),
        cos_eps_ok=(cos_eps >= 0.0),
    )