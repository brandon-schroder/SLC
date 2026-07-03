"""Immutable inputs to residual assembly (ARCH-3.3; AD-3, AD-4, AD-10).

``FrozenInputs`` is THE configuration boundary of the solver core: everything
is validated here, loudly, with ``ConfigError`` — so that the residual path
downstream never needs to raise (AD-10). Construction happens once per outer
iterate (the lagged fields change), which is cheap relative to a residual
evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # config-boundary validation + defaults only  # ad6: allow

from ..errors import ConfigError
from ..fluid.base import WorkingFluid
from ..grid.core import GridTopology, MetricsConfig
from ..transport.streamwise import TransportFields
from ..types import FidelityConfig, MassFlowSpec, OperatingSpec

__all__ = ["ClosureFields", "FrozenInputs"]


@dataclass(frozen=True)
class ClosureFields:
    """Lagged closure outputs entering assembly as data (AD-4, ARCH-3.3).

    M2 scope: nodal blockage only (consumed by the continuity integrand,
    section 3.2). The per-row fields ARCH-3.3 also assigns here (delta_s_row,
    exit rVt, in-blade schedules, mixing coefficients) are consumed by the
    driver's transport sweep, not by the assembler, and land with the
    correlation milestones (M4+) — recorded so their absence is deliberate.

    Parameters
    ----------
    blockage : total aerodynamic blockage ``B(i, j)`` in [0, 1) (section 3.2).
    validity : aggregate closure validity in [0, 1] (section 7.3.3).
    iteration_tag : outer-iterate counter that produced these fields, for
        convergence-record attribution (ARCH-3.3).
    """

    blockage: np.ndarray
    validity: float = 1.0
    iteration_tag: int = 0

    def __post_init__(self):
        b = np.asarray(self.blockage, dtype=float)
        object.__setattr__(self, "blockage", b)
        if np.any(b < 0.0) or np.any(b >= 1.0):
            raise ConfigError("blockage must satisfy 0 <= B < 1 everywhere")
        if not (0.0 <= self.validity <= 1.0):
            raise ConfigError(f"validity must be in [0, 1], got {self.validity}")


@dataclass(frozen=True)
class FrozenInputs:
    """The immutable bundle entering residual assembly (AD-3/AD-4, ARCH-3.3).

    Residual assembly is a pure function of ``(x, FrozenInputs)``; everything
    lagged (transported fields per section 6.1, closure outputs per AD-4) or
    frozen (topology per AD-8, fluid backend, fidelity flags, operating spec)
    lives here as data.

    Parameters
    ----------
    topology : immutable grid topology (AD-8).
    fluid : thermodynamic backend (section 3.7).
    fidelity : tier term-flags (section 8, AD-1).
    spec : operating specification. M2 supports ``MassFlowSpec``; the
        BC-switched ``BackPressureSpec`` residual form lands with M5
        (ARCH-4.3, ARCH-8) and is rejected here until then.
    transported : lagged nodal (h0, s, rVt), each ``(n_sl, n_qo)``
        (section 6.1: transported fields are functions of the state updated
        by lagged sweeps).
    closures : lagged closure outputs (AD-4).
    vm_lagged : previous-iterate ``Vm(i, j)`` used for the dVm/dm lean term
        (section 5.2 "using the current iterate"). Defaults to zeros, which
        together with Tier-2 flags reproduces the REE limit exactly.
    kappa_lagged, kappa_relax : optional curvature under-relaxation
        (section 5.5): the master-ODE curvature distribution uses
        ``kappa_relax * kappa_new + (1 - kappa_relax) * kappa_lagged``
        when a lagged field is supplied. ``kappa_relax = 1`` (default) or
        ``kappa_lagged = None`` disables the blend. Curvature is the noise
        amplifier of SLC; drivers pass the previous iterate's field here.
    metrics_config : streamline-fit settings forwarded to the grid layer.
    """

    topology: GridTopology
    fluid: WorkingFluid
    fidelity: FidelityConfig
    spec: OperatingSpec
    transported: TransportFields
    closures: ClosureFields
    vm_lagged: np.ndarray = None
    kappa_lagged: np.ndarray = None
    kappa_relax: float = 1.0
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)

    def __post_init__(self):
        shape = (self.topology.n_sl, self.topology.n_qo)
        for name, arr in (("transported.h0", self.transported.h0),
                          ("transported.s", self.transported.s),
                          ("transported.rvt", self.transported.rvt),
                          ("closures.blockage", self.closures.blockage)):
            if np.shape(arr) != shape:
                raise ConfigError(
                    f"{name} shape {np.shape(arr)} != (n_sl, n_qo) = {shape}")
        if not isinstance(self.spec, MassFlowSpec):
            raise ConfigError(
                "only MassFlowSpec is supported until the M5 BC-switching "
                f"milestone (ARCH-8), got {type(self.spec).__name__}")
        if self.vm_lagged is None:
            object.__setattr__(self, "vm_lagged", np.zeros(shape))
        else:
            vm = np.asarray(self.vm_lagged, dtype=float)
            if vm.shape != shape:
                raise ConfigError(
                    f"vm_lagged shape {vm.shape} != (n_sl, n_qo) = {shape}")
            object.__setattr__(self, "vm_lagged", vm)
        if not (0.0 < self.kappa_relax <= 1.0):
            raise ConfigError(
                f"kappa_relax must be in (0, 1], got {self.kappa_relax}")
        if self.kappa_lagged is not None:
            kl = np.asarray(self.kappa_lagged, dtype=float)
            if kl.shape != shape:
                raise ConfigError(
                    f"kappa_lagged shape {kl.shape} != (n_sl, n_qo) = {shape}")
            object.__setattr__(self, "kappa_lagged", kl)

    @property
    def n_sl(self) -> int:
        return self.topology.n_sl

    @property
    def n_qo(self) -> int:
        return self.topology.n_qo
