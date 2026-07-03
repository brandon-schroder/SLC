"""Thermally + calorically perfect-gas backend (Theory Manual section 3.7).

Reference implementation of :class:`~slcflow.fluid.base.WorkingFluid` and the
analytic oracle against which any future real-gas backend is checked at its
perfect-gas limit.

Thermodynamic model
-------------------
Constant ``cp``, ``cv``; ``R = cp - cv``; ``gamma = cp/cv``. With enthalpy
referenced so that ``h = cp * T`` (i.e. ``h = 0`` at ``T = 0``):

    T(h)       = h / cp
    s(T, p)    = cp ln(T/Tref) - R ln(p/pref)          (s = 0 at ref state)
    p(h, s)    = pref exp[(cp ln(T/Tref) - s) / R]
    rho        = p / (R T)
    a          = sqrt(gamma R T)

Stagnation is isentropic (``s0 = s``), so ``p0/p = (T0/T)**(gamma/(gamma-1))``
follows from the entropy relation -- this identity is used as a regression
check rather than hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .._namespace import get_xp
from .base import StagState

# Sea-level standard reference state (arbitrary; only differences matter).
_T_REF = 288.15  # K
_P_REF = 101325.0  # Pa


@dataclass(frozen=True)
class PerfectGas:
    """Perfect-gas backend. Defaults to dry air.

    Parameters
    ----------
    gamma : ratio of specific heats.
    R : specific gas constant [J/(kg K)].
    T_ref, p_ref : entropy reference state (``s = 0`` there).
    xp : array namespace (AD-6); defaults to NumPy.

    Domain
    ------
    Valid for ``h > 0`` (i.e. ``T > 0`` with the ``h = cp*T`` reference).
    Out-of-domain inputs produce NaN/Inf, detected at the assembler boundary
    per AD-10 -- this backend performs no per-call domain checks on the hot
    path by design.
    """

    gamma: float = 1.4
    R: float = 287.05
    T_ref: float = _T_REF
    p_ref: float = _P_REF
    # Array namespace (AD-6). Excluded from equality/repr because a module
    # object must not participate in provenance hashing (ARCH-6); leave as
    # ``None`` (NumPy default) for picklable/serializable configurations.
    xp: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # Config-boundary validation (AD-10): exceptions allowed here.
        if not self.gamma > 1.0:
            raise ValueError(f"gamma must be > 1, got {self.gamma}")
        if not self.R > 0.0:
            raise ValueError(f"R must be > 0, got {self.R}")
        if not (self.T_ref > 0.0 and self.p_ref > 0.0):
            raise ValueError("reference state must have T_ref > 0 and p_ref > 0")

    # --- derived constants ------------------------------------------------
    @property
    def cp(self) -> float:
        return self.gamma * self.R / (self.gamma - 1.0)

    @property
    def cv(self) -> float:
        return self.R / (self.gamma - 1.0)

    # --- primary evaluations from (h, s) ----------------------------------
    # NOTE on broadcasting: several perfect-gas relations are mathematically
    # independent of one argument (T of s, h of p). They must still broadcast
    # against it so this backend is shape-indistinguishable from a real-gas
    # backend (conformance test 5); the degeneracy must not leak via shapes.
    def T(self, h, s):
        xp = get_xp(self.xp)
        h_b, _ = xp.broadcast_arrays(h, s)
        return h_b / self.cp

    def p(self, h, s):
        xp = get_xp(self.xp)
        T = self.T(h, s)
        return self.p_ref * xp.exp((self.cp * xp.log(T / self.T_ref) - s) / self.R)

    def rho(self, h, s):
        T = self.T(h, s)
        return self.p(h, s) / (self.R * T)

    def a(self, h, s):
        xp = get_xp(self.xp)
        return xp.sqrt(self.gamma * self.R * self.T(h, s))

    # --- conversions from (T, p) ------------------------------------------
    def h_from_Tp(self, T, p):
        xp = get_xp(self.xp)
        T_b, _ = xp.broadcast_arrays(T, p)  # h is p-independent; shape is not
        return self.cp * T_b

    def s_from_Tp(self, T, p):
        xp = get_xp(self.xp)
        return self.cp * xp.log(T / self.T_ref) - self.R * xp.log(p / self.p_ref)

    # --- stagnation <-> static (isentropic) -------------------------------
    def stag_from_static(self, h, s, V):
        h0 = h + 0.5 * V * V
        T0 = self.T(h0, s)
        p0 = self.p(h0, s)  # same entropy -> isentropic stagnation
        return StagState(h0=h0, s=s, T0=T0, p0=p0)

    def static_h_from_stag(self, h0, V):
        return h0 - 0.5 * V * V