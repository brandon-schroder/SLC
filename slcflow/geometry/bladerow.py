"""Blade-row geometry: the section 4.1 row data contract (ARCH-3.1).

Correlations consume this contract only — never raw CAD. Everything is a
callable of span fraction ``y in [0, 1]`` (wall_0 side to wall_1 side),
guaranteed C1 in ``y`` (ARCH-3.1: section 7.3 smoothness propagates
upstream to geometry). Angles follow the section 2.4 convention: measured
from the meridional direction, positive toward rotor rotation, radians
(AD-7).

``ParamRowGeometry`` is the design-parameter implementation: each quantity
is a scalar (constant over span) or an array of values at uniform span
nodes, interpolated monotone-cubic (PCHIP — C1). ``TabulatedRowGeometry``
(from existing hardware geometry) is deferred until a real dataset needs it
(ARCH-3.1; record: deliberate).

Contract slots join with their consumers, not before (adding them early
would be untestable speculation). ``throat`` (the throat opening ``o``,
section 4.5) landed at M6 with its first consumer, the axial-turbine
exit-angle closure; it is optional (compressor rows never set it) and
raises loudly if a turbine closure asks for a throat that was not provided.
Still deferred until their own consumers exist: lean/sweep of the mean
stream surface, tangential thickness (sections A.8, 3.2 blade blockage; M7).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np  # geometry layer binds numpy directly  # ad6: allow
from scipy.interpolate import PchipInterpolator

from ..errors import ConfigError

__all__ = ["BladeRowGeometry", "ParamRowGeometry"]


@runtime_checkable
class BladeRowGeometry(Protocol):
    """Section 4.1 row data contract (the M4 subset; see module docstring
    for the deliberately deferred slots)."""

    blade_count: int

    def beta1_blade(self, y): ...      # LE metal angle [rad]
    def beta2_blade(self, y): ...      # TE metal angle [rad]
    def chord(self, y): ...            # [m]
    def stagger(self, y): ...          # [rad]
    def solidity(self, y): ...         # chord / pitch
    def thickness_ratio(self, y): ...  # max thickness / chord
    def throat(self, y): ...           # throat opening o [m] (section 4.5)
    def tip_clearance(self) -> float: ...  # [m]


def _spanwise(value, name):
    """Scalar -> constant callable; array -> PCHIP over uniform span nodes
    (C1, monotone-cubic; ARCH-3.1). Config boundary: validates here."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        v = float(arr)
        return lambda y: np.broadcast_to(v, np.shape(y)).copy() \
            if np.ndim(y) else v
    if arr.ndim != 1 or arr.size < 2:
        raise ConfigError(f"{name}: expected scalar or 1-D array of >= 2 "
                          f"span values, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ConfigError(f"{name}: non-finite span values")
    return PchipInterpolator(np.linspace(0.0, 1.0, arr.size), arr,
                             extrapolate=True)


@dataclass(frozen=True)
class ParamRowGeometry:
    """Design-parameter row geometry (ARCH-3.1 ``ParamRowGeometry``).

    Every spanwise quantity is a scalar or an array at uniform span nodes.
    The LE metal angle must be nonzero and single-signed across span
    (:attr:`orientation`, the inlet-keyed cascade-frame sign, is derived
    from it — rejected here, loudly, at construction). The TE metal angle
    carries no construction-time sign constraint: compressor rows
    legitimately turn to (or slightly past) axial. Closures that key off
    the TE turning direction instead use :attr:`orientation_te`, which
    validates lazily on first use (the ``throat`` precedent) — a turbine
    row's beta1 and beta2 routinely have OPPOSITE signs (reaction rotor
    with co-rotating relative inflow), which is in scope.
    """

    blade_count: int
    beta1: object               # LE metal angle(s) [rad]
    beta2: object               # TE metal angle(s) [rad]
    chord_len: object           # [m]
    solidity_val: object
    thickness: object = 0.10    # max t/c
    stagger_val: object = 0.0   # [rad]
    throat_val: object = None   # throat opening o [m]; None = not provided
    clearance: float = 0.0      # [m]
    _f: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.blade_count < 1:
            raise ConfigError(f"blade_count must be >= 1, got "
                              f"{self.blade_count}")
        f = {"beta1": _spanwise(self.beta1, "beta1"),
             "beta2": _spanwise(self.beta2, "beta2"),
             "chord": _spanwise(self.chord_len, "chord_len"),
             "solidity": _spanwise(self.solidity_val, "solidity_val"),
             "thickness": _spanwise(self.thickness, "thickness"),
             "stagger": _spanwise(self.stagger_val, "stagger_val")}
        y = np.linspace(0.0, 1.0, 33)
        b1 = np.asarray(f["beta1"](y), dtype=float)
        if np.any(b1 == 0.0) or (np.any(b1 > 0.0) and np.any(b1 < 0.0)):
            raise ConfigError(
                "beta1 metal angle must be nonzero and single-signed across "
                "span (cascade-frame mapping; see class docstring)")
        if np.any(np.asarray(f["solidity"](y), dtype=float) <= 0.0):
            raise ConfigError("solidity must be > 0 across span")
        if np.any(np.asarray(f["thickness"](y), dtype=float) <= 0.0):
            raise ConfigError("thickness ratio must be > 0 across span")
        if self.throat_val is not None:
            f["throat"] = _spanwise(self.throat_val, "throat_val")
            if np.any(np.asarray(f["throat"](y), dtype=float) <= 0.0):
                raise ConfigError("throat must be > 0 across span")
        object.__setattr__(self, "_f", f)

    # --- section 4.1 contract ---------------------------------------------
    def beta1_blade(self, y):
        return self._f["beta1"](y)

    def beta2_blade(self, y):
        return self._f["beta2"](y)

    def chord(self, y):
        return self._f["chord"](y)

    def stagger(self, y):
        return self._f["stagger"](y)

    def solidity(self, y):
        return self._f["solidity"](y)

    def thickness_ratio(self, y):
        return self._f["thickness"](y)

    def throat(self, y):
        """Throat opening ``o`` [m] (section 4.5). Config boundary (AD-10):
        raises if no throat was provided for this row — a turbine
        exit-angle closure asked for a throat a compressor row never set."""
        if "throat" not in self._f:
            raise ConfigError(
                "throat opening not provided for this row (required by the "
                "axial-turbine exit-angle closure; set throat_val)")
        return self._f["throat"](y)

    def tip_clearance(self) -> float:
        return self.clearance

    @property
    def orientation(self) -> float:
        """Sign of the blade's INLET tangential orientation (+1 or -1),
        from the LE metal angle — geometry data, constant per solve, safe
        to branch on (ARCH-4.2). This is the frame sign for inlet-keyed
        closures (Lieblein incidence/deviation, Wiesner slip, incidence
        loss). It is NOT the exit turning direction: use
        :attr:`orientation_te` to sign exit-angle quantities — a turbine
        rotor's LE and TE angles routinely have opposite signs."""
        return 1.0 if float(self._f["beta1"](0.5)) > 0.0 else -1.0

    @property
    def orientation_te(self) -> float:
        """Sign of the blade's EXIT turning direction (+1 or -1), from the
        TE metal angle — the sign for throat-based exit angles (section 4.5)
        and the turbine cascade frame. Validated lazily like ``throat``
        (AD-10 config boundary): compressor rows may carry a near-axial or
        sign-crossing TE angle and never ask; a closure that keys off the
        exit direction needs it well-defined, so an axial or span-sign-
        crossing TE angle raises loudly here."""
        y = np.linspace(0.0, 1.0, 33)
        b2 = np.asarray(self._f["beta2"](y), dtype=float)
        if np.any(b2 == 0.0) or (np.any(b2 > 0.0) and np.any(b2 < 0.0)):
            raise ConfigError(
                "beta2 metal angle must be nonzero and single-signed across "
                "span to define the exit turning direction (orientation_te, "
                "required by throat-based turbine exit-angle/loss closures); "
                "got a zero or sign-crossing TE angle")
        return 1.0 if float(b2[0]) > 0.0 else -1.0
