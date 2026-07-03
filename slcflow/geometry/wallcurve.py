"""Wall-curve representation (Grid & Geometry Spec G-3; Theory Manual §5.1).

A :class:`WallCurve` is a parametric C² curve ``(z(σ), r(σ))`` in the
meridional plane, parameterized by arc length ``σ ∈ [0, L]``. It is the
representation for annulus walls and (in M1) the reference implementation the
streamline fit will mirror.

Normative requirements implemented here (G-3):
  * parametric always -- valid through ``φ = ±90°`` (no ``r(z)`` anywhere);
  * arc-length parameterization by iterative re-parameterization (G-3.2);
  * interpolating or smoothing cubic fit, both C² (G-3.3);
  * natural end conditions by default, clamped available (G-3.4);
  * exact-through-the-poles slope and curvature via ``atan2`` and the
    parametric curvature formula (G-6.1.2 -- the only permitted forms).

Construction is a configuration boundary: validation errors raise
:class:`~slcflow.errors.ConfigError` (AD-10). Evaluation methods are pure and
vectorized over ``σ``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # geometry layer binds numpy directly  # ad6: allow
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline, make_smoothing_spline

from ..errors import ConfigError

__all__ = ["WallCurve"]

_DENSE_PER_INTERVAL = 20  # dense-sampling factor for arc-length tables
_REPARAM_TOL = 1e-10
_REPARAM_MAX_ITERS = 8


def _fit_1d(t, y, smoothing, bc_type):
    """Cubic C² fit of y(t): interpolating (smoothing=None) or smoothing."""
    if smoothing is None:
        return CubicSpline(t, y, bc_type=bc_type)
    # make_smoothing_spline: natural boundary behavior by construction; the
    # bc_type argument does not apply. lam=smoothing controls the penalty.
    return make_smoothing_spline(t, y, lam=smoothing)


@dataclass(frozen=True)
class WallCurve:
    """Parametric meridional curve with arc-length parameter access.

    Do not construct directly -- use :meth:`from_points` or
    :meth:`from_callable`.
    """

    _sz: object = field(repr=False)          # spline z(t), t in [0, 1]
    _sr: object = field(repr=False)          # spline r(t)
    _t_dense: np.ndarray = field(repr=False)
    _sigma_dense: np.ndarray = field(repr=False)  # cumulative arc length at t_dense
    name: str = "wall"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_points(
        cls,
        points,
        *,
        smoothing: float | None = None,
        bc_type: str = "natural",
        name: str = "wall",
    ) -> "WallCurve":
        """Fit a wall curve to ordered meridional points ``[(z, r), ...]``.

        Parameters
        ----------
        points : (n, 2) array-like of ordered ``(z, r)`` coordinates,
            n >= 4, finite, no repeated consecutive points.
        smoothing : ``None`` for an interpolating spline (verification inputs);
            a positive ``lam`` penalty for a smoothing spline (measured data).
            Default heuristic is an open [DECIDE] in the G-spec: callers of
            real geometry should pass a value scaled to their noise level.
        bc_type : end condition for the interpolating fit, ``"natural"``
            (default, G-3.4) or ``"clamped"``/``"not-a-knot"`` per SciPy.
        """
        pts = np.asarray(points, dtype=float)
        cls._validate_points(pts)
        if smoothing is not None and not smoothing > 0.0:
            raise ConfigError(f"smoothing must be None or > 0, got {smoothing!r}")

        z, r = pts[:, 0], pts[:, 1]

        # Initial parameter: normalized cumulative chord length (G-3.2).
        chord = np.hypot(np.diff(z), np.diff(r))
        t = np.concatenate(([0.0], np.cumsum(chord)))
        t /= t[-1]

        # Iterative arc-length re-parameterization: fit, measure arc length,
        # re-space parameters proportionally, refit; converges in 2-3 passes.
        for _ in range(_REPARAM_MAX_ITERS):
            sz = _fit_1d(t, z, smoothing, bc_type)
            sr = _fit_1d(t, r, smoothing, bc_type)
            t_dense, sigma_dense = cls._arclength_table(sz, sr, t)
            # Re-space data-site parameters to equal fractional arc length.
            t_new = np.interp(t, t_dense, sigma_dense) / sigma_dense[-1]
            delta = float(np.max(np.abs(t_new - t)))
            t = t_new
            if delta < _REPARAM_TOL:
                break

        sz = _fit_1d(t, z, smoothing, bc_type)
        sr = _fit_1d(t, r, smoothing, bc_type)
        t_dense, sigma_dense = cls._arclength_table(sz, sr, t)
        return cls(_sz=sz, _sr=sr, _t_dense=t_dense, _sigma_dense=sigma_dense, name=name)

    @classmethod
    def from_callable(cls, fn, *, n: int = 201, name: str = "wall", **kwargs) -> "WallCurve":
        """Convenience for analytic verification curves: sample ``fn(u)`` for
        ``u ∈ [0, 1]`` returning ``(z, r)``, then :meth:`from_points` with an
        interpolating fit.

        Defaults to ``bc_type="not-a-knot"``: natural end conditions force
        ``κ = 0`` at the endpoints (zero second derivative), polluting
        end-region curvature of analytic curves by ~1e-3 relative -- the
        G-3.4 end-condition sensitivity, measured. Not-a-knot imposes no
        artificial endpoint curvature and is correct for smooth noise-free
        data; ``natural`` remains the :meth:`from_points` default for
        measured geometry, where it is the more noise-robust choice.
        """
        kwargs.setdefault("bc_type", "not-a-knot")
        u = np.linspace(0.0, 1.0, n)
        pts = np.column_stack(fn(u))
        return cls.from_points(pts, name=name, **kwargs)

    # ------------------------------------------------------------------
    # Evaluation (pure, vectorized over sigma)
    # ------------------------------------------------------------------
    @property
    def arclength(self) -> float:
        """Total arc length L; valid parameter domain is [0, L]."""
        return float(self._sigma_dense[-1])

    def point(self, sigma):
        """Meridional coordinates ``(z, r)`` at arc length ``sigma``."""
        t = self._t_of_sigma(sigma)
        return self._sz(t), self._sr(t)

    def unit_tangent(self, sigma):
        """Unit tangent ``(e_z, e_r)`` -- parameter-invariant via normalization."""
        t = self._t_of_sigma(sigma)
        zp, rp = self._sz(t, 1), self._sr(t, 1)
        speed = np.hypot(zp, rp)
        return zp / speed, rp / speed

    def slope_phi(self, sigma):
        """Streamwise slope ``φ = atan2(dr, dz)`` (§2.2). Exact through ±90°."""
        t = self._t_of_sigma(sigma)
        return np.arctan2(self._sr(t, 1), self._sz(t, 1))

    def curvature(self, sigma):
        """Signed meridional curvature ``κ = dφ/dm`` (A.1 convention):
        ``(z' r'' - r' z'') / (z'^2 + r'^2)^{3/2}`` -- parameter-invariant
        (G-6.1.2, the only permitted form)."""
        t = self._t_of_sigma(sigma)
        zp, rp = self._sz(t, 1), self._sr(t, 1)
        zpp, rpp = self._sz(t, 2), self._sr(t, 2)
        speed2 = zp * zp + rp * rp
        return (zp * rpp - rp * zpp) / speed2**1.5

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _arclength_table(sz, sr, t_sites):
        n_dense = _DENSE_PER_INTERVAL * (len(t_sites) - 1) + 1
        t_dense = np.linspace(0.0, 1.0, n_dense)
        speed = np.hypot(sz(t_dense, 1), sr(t_dense, 1))
        sigma_dense = cumulative_trapezoid(speed, t_dense, initial=0.0)
        return t_dense, sigma_dense

    def _t_of_sigma(self, sigma):
        sigma = np.asarray(sigma, dtype=float)
        L = self.arclength
        tol = 1e-9 * L
        if np.any(sigma < -tol) or np.any(sigma > L + tol):
            raise ValueError(
                f"sigma outside [0, {L}] for wall {self.name!r}: "
                f"[{float(np.min(sigma))}, {float(np.max(sigma))}]"
            )
        return np.interp(np.clip(sigma, 0.0, L), self._sigma_dense, self._t_dense)

    @staticmethod
    def _validate_points(pts):
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ConfigError(f"points must have shape (n, 2), got {pts.shape}")
        if pts.shape[0] < 4:
            raise ConfigError(f"need at least 4 points for a C2 fit, got {pts.shape[0]}")
        if not np.all(np.isfinite(pts)):
            raise ConfigError("points contain non-finite values")
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        if np.any(seg == 0.0):
            raise ConfigError("repeated consecutive points")
        if np.any(pts[:, 1] < 0.0):
            raise ConfigError("negative radius in wall points")