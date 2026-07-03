"""Working-fluid interface (Theory Manual section 3.7 / 4.6, ARCH-4.1).

The kernel accesses thermodynamics *only* through :class:`WorkingFluid`. The
primary state pair is ``(h, s)`` -- static enthalpy and specific entropy --
chosen because entropy is the internal loss currency (section 1, principle 3)
and enthalpy/rothalpy are the conserved streamwise quantities (section 3.3).

No perfect-gas shortcut (bare ``gamma``, ``cp``) may appear in kernel code; it
lives behind a backend implementing this Protocol. All methods are vectorized
over NumPy-compatible arrays and accept scalars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union, runtime_checkable

import numpy.typing as npt

# Scalars and NumPy-compatible arrays are both accepted; all backends must
# broadcast. (Becomes a broader union when a second array backend lands.)
Array = Union[float, npt.NDArray]


@dataclass(frozen=True)
class StagState:
    """Stagnation state reached by isentropic deceleration from a static state.

    ``s`` is unchanged from the originating static state (isentropic), retained
    here for convenience so a StagState is a self-contained thermodynamic point.
    """

    h0: Array
    s: Array
    T0: Array
    p0: Array


@runtime_checkable
class WorkingFluid(Protocol):
    """Thermodynamic backend contract. All methods vectorized and pure."""

    # --- primary evaluations from the (h, s) state pair -------------------
    def rho(self, h: Array, s: Array) -> Array: ...
    def T(self, h: Array, s: Array) -> Array: ...
    def p(self, h: Array, s: Array) -> Array: ...
    def a(self, h: Array, s: Array) -> Array: ...  # speed of sound

    # --- conversions from a (T, p) description ----------------------------
    def h_from_Tp(self, T: Array, p: Array) -> Array: ...
    def s_from_Tp(self, T: Array, p: Array) -> Array: ...

    # --- stagnation <-> static (isentropic) -------------------------------
    def stag_from_static(self, h: Array, s: Array, V: Array) -> StagState: ...
    def static_h_from_stag(self, h0: Array, V: Array) -> Array: ...