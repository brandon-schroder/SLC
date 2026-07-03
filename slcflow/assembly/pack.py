"""State-vector packing (ARCH-3.2, section 6.1).

The Newton state vector for the mass-flow-specified form is

    x = [Vm_q0[j] for all j]  +  [q[i, j] for interior i, C-order]

where ``Vm_q0[j]`` is the meridional velocity at the ``q = 0`` end of q-o
``j`` (ARCH-3.2 writes "Vm_wall0"; per A.1.1/AD-9 the ``q = 0`` wall is
machine-dependent — see ``FlowPath.q_origin_wall`` for the physical mapping)
and interior streamlines are ``i = 1 .. n_sl - 2`` (walls fixed, section
6.1). In choke-proximal mode ``mdot`` joins ``x`` (M5, not packed here).

Residual-path module (AD-6): pure functions, no mutation.
"""
from __future__ import annotations

from .._namespace import get_xp
from ..errors import ConfigError

__all__ = ["n_unknowns", "pack", "unpack"]


def n_unknowns(n_sl: int, n_qo: int) -> int:
    """Length of ``x`` for the MassFlowSpec form (ARCH-3.2)."""
    return n_qo + max(n_sl - 2, 0) * n_qo


def pack(vm_q0, q_interior, *, xp=None):
    """Pack ``(vm_q0 (n_qo,), q_interior (n_sl-2, n_qo))`` into ``x``."""
    xp = get_xp(xp)
    return xp.concatenate([xp.ravel(vm_q0), xp.ravel(q_interior)])


def unpack(x, n_sl: int, n_qo: int, *, xp=None):
    """Split ``x`` into ``(vm_q0 (n_qo,), q_interior (n_sl-2, n_qo))``.

    A wrong-length ``x`` is a programming error at the driver/assembler
    seam, not out-of-domain physics — it raises (grid-layer precedent for
    shape validation on the residual path).
    """
    xp = get_xp(xp)
    n_int = max(n_sl - 2, 0)
    if xp.shape(x) != (n_qo + n_int * n_qo,):
        raise ConfigError(
            f"x has shape {xp.shape(x)}; expected ({n_unknowns(n_sl, n_qo)},) "
            f"for (n_sl, n_qo) = ({n_sl}, {n_qo})")
    return x[:n_qo], xp.reshape(x[n_qo:], (n_int, n_qo))
