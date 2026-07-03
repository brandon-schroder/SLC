"""Array-namespace injection (AD-6).

The kernel is written against a NumPy-compatible array namespace so that a
future swap to ``jax.numpy`` (for algorithmic differentiation through the
solver) is a namespace substitution rather than a rewrite. Every numerical
function/class accepts an optional ``xp`` argument defaulting to NumPy.

Rules enforced downstream of this module (see Architecture Spec AD-6):
  * no in-place array mutation on the residual path,
  * no data-dependent *Python* branching on array values (elementwise ufunc
    selection such as ``xp.where`` / ``xp.maximum`` is permitted),
  * all math routed through the injected ``xp``.
"""
from __future__ import annotations

import numpy as _np

# The default namespace. Kept as a module attribute so tests and future
# backends can reference a single canonical default.
default_xp = _np


def get_xp(xp=None):
    """Return the array namespace to use (NumPy if unspecified)."""
    return default_xp if xp is None else xp