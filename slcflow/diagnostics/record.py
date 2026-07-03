"""Solve status and convergence provenance (ARCH-6).

Solver failures are *returned* as typed status values, never raised (AD-10),
so map drivers and optimizers can react programmatically. The
``ConvergenceRecord`` answers "how did we get here" for every solve: all
three section 6.2.5 norms per outer iteration, the relaxation factor used,
and what stopped the iteration.

The structured-logging / reproducer-bundle plumbing ARCH-6 also describes is
deferred until the continuation driver needs it (M5); the record here is the
in-memory core it will serialize.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

__all__ = ["SolveStatus", "IterationRecord", "ConvergenceRecord"]


class SolveStatus(enum.Enum):
    """Typed solve outcomes (ARCH-6). ``STALLED`` is reserved for the
    continuation driver's stall flagging (section 6.7, M5)."""

    CONVERGED = "converged"
    MAX_ITER = "max_iter"
    CHOKE_LIMITED = "choke_limited"
    NUMERICAL_FAILURE = "numerical_failure"
    STALLED = "stalled"


@dataclass(frozen=True)
class IterationRecord:
    """One outer iteration's norms (section 6.2.5: report all three) plus
    the streamline relaxation factor actually used (section 6.4)."""

    iteration: int
    cont_norm: float      # max_j |F_j| / mdot
    pos_norm: float       # max |delta q| / q-o length
    closure_norm: float   # closure-update norm (0.0 while closures static)
    omega_sl: float


@dataclass(frozen=True)
class ConvergenceRecord:
    """Full per-solve iteration history with the terminating status and,
    on numerical failure, a human-readable reason."""

    status: SolveStatus
    iterations: tuple
    reason: str = ""

    @property
    def n_iterations(self) -> int:
        return len(self.iterations)
