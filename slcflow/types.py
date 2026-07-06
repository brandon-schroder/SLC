"""Shared value types: fidelity configuration and operating specification
(Theory Manual section 8; Architecture Spec ARCH-4.3, AD-1).

Layer-0 module (no imports from the package) so that ``assembly`` can consume
these types without reaching up to the ``machine`` facade, which re-exports
them for user-facing composition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .errors import ConfigError

__all__ = ["FidelityConfig", "MassFlowSpec", "BackPressureSpec",
           "OperatingSpec"]


@dataclass(frozen=True)
class FidelityConfig:
    """Fidelity tier as data, never as a code path (AD-1, section 8).

    The Tier-3-exclusive master-equation terms carry multiplicative flags
    (ARCH-5.1): the residual assembler always evaluates every term and
    multiplies by these floats, so tier switching involves no branching and
    the section 8 degeneracy (Tier 2 = Tier 3 with flags zeroed) holds by
    construction. The in-blade force term (section 3.1, A.8) is gated by
    INBLADE station topology and lands with M7, not by a flag here.

    Parameters
    ----------
    curvature_term : multiplies ``Vm^2 kappa_m cos(eps)`` (section 3.1).
    lean_term : multiplies ``Vm (dVm/dm) sin(eps)`` (section 3.1).
    mixing_term : scales the section 3.6 spanwise-mixing operator in the
        lagged field refresh (0 = off, the Tier 1/2 default and the Tier 3
        default too; multistage-axial cases opt in explicitly). It is NOT a
        master-equation term -- it acts on the transported fields between
        outer iterates (AD-4), not on the residual -- so it never breaks the
        section 8 tier degeneracy or the V3 Tier 2 == Tier 3 identity.
    """

    curvature_term: float = 1.0
    lean_term: float = 1.0
    mixing_term: float = 0.0

    def __post_init__(self):
        for name in ("curvature_term", "lean_term", "mixing_term"):
            v = getattr(self, name)
            if not (0.0 <= v <= 1.0):
                raise ConfigError(f"{name} must be in [0, 1], got {v}")

    @classmethod
    def tier2(cls) -> "FidelityConfig":
        """Streamline-REE: REE terms only, curvature/lean off (section 8)."""
        return cls(curvature_term=0.0, lean_term=0.0)

    @classmethod
    def tier3(cls, *, mixing_term: float = 0.0) -> "FidelityConfig":
        """Full SLC: all master-equation terms active (section 8). Spanwise
        mixing is opt-in (``mixing_term``) -- on for multistage axial, off by
        default so the V3 Tier-consistency identity holds."""
        return cls(curvature_term=1.0, lean_term=1.0, mixing_term=mixing_term)

    # Tier 1 (meanline) shares the Tier-2 flag set; its degeneration is the
    # single mid-psi streamline (n_sl = 1) plus repositioning-off, which are
    # grid topology and driver settings respectively (section 8).
    tier1 = tier2


@dataclass(frozen=True)
class MassFlowSpec:
    """Normal-mode operating point: mass flow specified (section 6.6)."""

    mdot: float  # kg/s

    def __post_init__(self):
        if not self.mdot > 0.0:
            raise ConfigError(f"mdot must be > 0, got {self.mdot}")


@dataclass(frozen=True)
class BackPressureSpec:
    """Choke-proximal operating point: exit static pressure specified at a
    throttling station; mdot joins the state vector (section 6.6, ARCH-4.3).

    Constructible now for interface stability; the BC-switched residual form
    lands with the continuation driver milestone (M5, ARCH-8).
    """

    p_exit: float   # Pa
    station: int    # q-o index of the throttling station

    def __post_init__(self):
        if not self.p_exit > 0.0:
            raise ConfigError(f"p_exit must be > 0, got {self.p_exit}")
        if self.station < 0:
            raise ConfigError(f"station must be >= 0, got {self.station}")


OperatingSpec = Union[MassFlowSpec, BackPressureSpec]
