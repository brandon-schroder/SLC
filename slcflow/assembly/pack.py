"""State-vector packing (ARCH-3.2, section 6.1).

The Newton state vector for the mass-flow-specified form is

    x = [Vm_q0[j] for all j]  +  [q[i, j] for interior i, C-order]

where ``Vm_q0[j]`` is the meridional velocity at the ``q = 0`` end of q-o
``j`` (ARCH-3.2 writes "Vm_wall0"; per A.1.1/AD-9 the ``q = 0`` wall is
machine-dependent — see ``FlowPath.q_origin_wall`` for the physical mapping)
and interior streamlines are ``i = 1 .. n_sl - 2`` (walls fixed, section
6.1).

In choke-proximal (back-pressure) mode (section 6.6) the mass flow ``mdot``
becomes an unknown and is appended as the final component; the assembler adds
one matching back-pressure residual. ``backpressure=True`` selects that
layout throughout pack/unpack/n_unknowns.

Residual-path module (AD-6): pure functions, no mutation.
"""
from __future__ import annotations

from .._namespace import get_xp
from ..errors import ConfigError

__all__ = ["n_unknowns", "pack", "unpack"]


def n_unknowns(n_sl: int, n_qo: int, *, backpressure: bool = False) -> int:
    """Length of ``x`` (ARCH-3.2); ``backpressure`` appends the ``mdot``
    unknown (section 6.6)."""
    return n_qo + max(n_sl - 2, 0) * n_qo + (1 if backpressure else 0)


def pack(vm_q0, q_interior, mdot=None, *, xp=None):
    """Pack ``(vm_q0 (n_qo,), q_interior (n_sl-2, n_qo))`` into ``x``,
    appending the scalar ``mdot`` unknown when given (back-pressure mode)."""
    xp = get_xp(xp)
    parts = [xp.ravel(vm_q0), xp.ravel(q_interior)]
    if mdot is not None:
        parts.append(xp.reshape(xp.asarray(mdot, dtype=float), (-1,)))
    return xp.concatenate(parts)


def unpack(x, n_sl: int, n_qo: int, *, backpressure: bool = False, xp=None):
    """Split ``x`` into ``(vm_q0 (n_qo,), q_interior (n_sl-2, n_qo), mdot)``.

    ``mdot`` is the trailing back-pressure-mode unknown, or ``None`` in the
    normal (mass-flow-specified) form. A wrong-length ``x`` is a programming
    error at the driver/assembler seam, not out-of-domain physics — it raises
    ``ConfigError`` (grid-layer precedent for shape validation on the residual
    path).
    """
    xp = get_xp(xp)
    n_int = max(n_sl - 2, 0)
    base = n_qo + n_int * n_qo
    expected = base + (1 if backpressure else 0)
    if xp.shape(x) != (expected,):
        raise ConfigError(
            f"x has shape {xp.shape(x)}; expected ({expected},) for "
            f"(n_sl, n_qo) = ({n_sl}, {n_qo}), backpressure={backpressure}")
    vm_q0 = x[:n_qo]
    q_int = xp.reshape(x[n_qo:base], (n_int, n_qo))
    mdot = x[base] if backpressure else None
    return vm_q0, q_int, mdot
