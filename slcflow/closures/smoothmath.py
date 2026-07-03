"""C1 smooth-math toolbox (Theory Manual section 7.3, Architecture Spec ARCH-4.2).

Every closure/correlation in ``slcflow.closures`` MUST be built from these
primitives instead of raw ``if`` branches or hard ``min``/``max``/``clip`` on
flow quantities. The contract each primitive satisfies:

  * **At least C1** over the whole real line (value and first derivative
    continuous). Most are C-infinity; where only C1/C2 holds it is documented.
  * **Mandatory, explicit ``width``** (or edge span) controlling the transition
    scale -- there is no hidden default smoothing length. Callers choose it and
    document it, so smoothness is auditable.
  * **Vectorized and pure**: arrays in, arrays out, no mutation, routed through
    the injected array namespace ``xp`` (AD-6).

Design notes
------------
``smooth_max``/``smooth_min`` use the log-sum-exp form via ``xp.logaddexp``,
which is numerically stable and C-infinity. The overestimate of ``smooth_max``
over the true maximum is bounded by ``width * ln 2`` (attained when the two
arguments are equal); ``smooth_min`` symmetrically underestimates.
"""
from __future__ import annotations

from .._namespace import get_xp

__all__ = [
    "smoothstep",
    "softplus",
    "smooth_max",
    "smooth_min",
    "soft_clip",
    "logistic",
    "blend",
    "blend_between",
    "abs_smooth",
]


def _check_positive(value, name):
    """Config-boundary validation (AD-10): scalar smoothing parameters must be
    strictly positive. Raises ``ValueError`` -- exceptions are permitted here
    because widths/edges are configuration constants, not flow-state arrays."""
    import numpy as _np

    if not _np.all(_np.asarray(value) > 0.0):
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _check_ordered(lo, hi, names):
    import numpy as _np

    if not _np.all(_np.asarray(hi) > _np.asarray(lo)):
        raise ValueError(f"require {names[1]} > {names[0]}, got {lo!r}, {hi!r}")


def smoothstep(x, edge0, edge1, *, xp=None):
    """Quintic smoothstep: 0 for ``x <= edge0``, 1 for ``x >= edge1``.

    Uses Perlin's quintic ``6t^5 - 15t^4 + 10t^3`` whose first *and* second
    derivatives vanish at both edges, so the result is C2 including the joins
    to the flat regions. Requires ``edge1 > edge0``.
    """
    xp = get_xp(xp)
    _check_ordered(edge0, edge1, ("edge0", "edge1"))
    t = (x - edge0) / (edge1 - edge0)
    t = xp.clip(t, 0.0, 1.0)  # C0 clip, but the polynomial's 1st & 2nd
    # derivatives are zero at t in {0,1}, so the assembled function is C2.
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def smooth_max(a, b, width, *, xp=None):
    """C-infinity approximation to ``max(a, b)``.

    ``width * logaddexp(a/width, b/width)``. Overestimates the true max by at
    most ``width * ln 2``. ``width`` must be > 0.
    """
    xp = get_xp(xp)
    _check_positive(width, "width")
    return width * xp.logaddexp(a / width, b / width)


def smooth_min(a, b, width, *, xp=None):
    """C-infinity approximation to ``min(a, b)`` (dual of :func:`smooth_max`)."""
    xp = get_xp(xp)
    _check_positive(width, "width")
    return -width * xp.logaddexp(-a / width, -b / width)


def softplus(x, width, *, xp=None):
    """C-infinity one-sided saturation: ``max(x, 0)`` with a smooth knee.

    ``width * logaddexp(0, x/width)``. Tends to ``x`` for ``x >> width`` and to
    0 for ``x << -width``; value at ``x = 0`` is ``width * ln 2``. This is the
    canonical building block for section 7.3.2 smooth loss saturation, e.g.
    ``loss = base + slope * softplus(incidence - i_stall, width)`` for smooth
    post-stall growth from an exact pre-stall baseline.
    """
    xp = get_xp(xp)
    _check_positive(width, "width")
    return width * xp.logaddexp(0.0, x / width)


def soft_clip(x, lo, hi, width, *, xp=None):
    """Smooth (C-infinity) clip of ``x`` into ``[lo, hi]`` with rounded corners.

    Built as ``smooth_min(smooth_max(x, lo), hi)``. In the interior
    (``lo + width << x << hi - width``) the output equals ``x`` to within
    floating point; corners are rounded over the scale ``width``. Requires
    ``hi - lo`` comfortably larger than ``width`` for a clean plateau.
    """
    xp = get_xp(xp)
    _check_ordered(lo, hi, ("lo", "hi"))
    return smooth_min(smooth_max(x, lo, width, xp=xp), hi, width, xp=xp)


def logistic(x, x0, width, *, xp=None):
    """C-infinity logistic weight in (0, 1), passing through 0.5 at ``x = x0``.

    ``width`` is the e-folding scale of the transition. Monotone increasing.
    Never reaches exactly 0 or 1 (use :func:`blend` for compact support).

    Implemented via the identity ``sigma(z) = (1 + tanh(z/2)) / 2``, which is
    overflow-free in both tails (the naive ``1/(1+exp(-z))`` overflows for
    large negative ``z``).
    """
    xp = get_xp(xp)
    _check_positive(width, "width")
    z = (x - x0) / width
    return 0.5 * (1.0 + xp.tanh(0.5 * z))


def blend(x, x0, width, *, xp=None):
    """Compact-support C2 blend weight: 0 for ``x <= x0 - width``, 1 for
    ``x >= x0 + width``, smoothstep transition across ``[x0-width, x0+width]``.

    Preferred over :func:`logistic` when correlations must return *exact*
    regime values away from the transition band.
    """
    return smoothstep(x, x0 - width, x0 + width, xp=xp)


def blend_between(x, val_lo, val_hi, x0, width, *, xp=None):
    """Smoothly interpolate ``val_lo`` (regime ``x < x0``) into ``val_hi``
    (regime ``x > x0``) using the compact-support :func:`blend` weight.

    The practical regime-switch combiner for correlations: e.g. blending a
    subsonic loss branch into a supersonic one across a Mach band.
    """
    w = blend(x, x0, width, xp=xp)
    return val_lo + (val_hi - val_lo) * w


def abs_smooth(x, eps, *, xp=None):
    """C-infinity approximation to ``|x|``: ``sqrt(x^2 + eps^2)``.

    Overestimates ``|x|`` by at most ``eps`` (at ``x = 0``). ``eps`` must be > 0.
    """
    xp = get_xp(xp)
    _check_positive(eps, "eps")
    return xp.sqrt(x * x + eps * eps)