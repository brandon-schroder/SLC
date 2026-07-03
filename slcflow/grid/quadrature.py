"""The single quadrature + cumulative-inversion utility (Grid Spec G-5).

Normative rule (Theory Manual section 5.4): streamline initialization and the
future repositioning residual MUST use the *same* integration rule for
continuity-type integrals, or mass-fraction targets drift between the two.
That rule lives here and only here: composite trapezoid on the given nodes,
with monotone inversion by linear interpolation of the cumulative.

Residual-path module (AD-6): functional style, no in-place mutation.
"""
from __future__ import annotations

import numpy as np  # grid layer is numpy-bound via scipy splines  # ad6: allow
from scipy.integrate import cumulative_trapezoid

__all__ = ["cumulative", "invert_cumulative"]


def cumulative(integrand, coord):
    """Cumulative trapezoid of ``integrand`` over ``coord`` (same shape),
    starting at 0. THE continuity quadrature rule -- do not duplicate."""
    return cumulative_trapezoid(np.asarray(integrand, dtype=float),
                                np.asarray(coord, dtype=float), initial=0.0)


def invert_cumulative(coord, cumulative_values, targets):
    """Coordinates at which the cumulative integral reaches ``targets``.

    Requires ``cumulative_values`` non-decreasing (monotone integrand >= 0);
    violated inputs indicate a programming error upstream (negative radius or
    negative mass flux) and raise ``ValueError``.
    """
    cum = np.asarray(cumulative_values, dtype=float)
    if np.any(np.diff(cum) < 0.0):
        raise ValueError("cumulative integral is decreasing; integrand < 0 upstream")
    return np.interp(np.asarray(targets, dtype=float), cum, np.asarray(coord, dtype=float))