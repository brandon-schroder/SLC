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
from ..types import (BackPressureSpec, FidelityConfig, MassFlowSpec,
                     OperatingSpec)

__all__ = ["ClosureFields", "FrozenInputs"]


@dataclass(frozen=True)
class ClosureFields:
    """Lagged closure outputs entering assembly/transport as data (AD-4,
    ARCH-3.3).

    ``blockage`` is consumed by the continuity integrand (section 3.2);
    the per-row swirl/loss outputs are consumed by the driver's transport
    sweep (sections 3.3-3.5), keyed by row id. In-blade schedules and
    mixing coefficient fields join at M7/M8 (recorded deferral).

    Parameters
    ----------
    blockage : total aerodynamic blockage ``B(i, j)`` in [0, 1) (section 3.2).
    row_exit_rvt : per-row lagged exit rVt, ``{row_id: (n_sl,)}``
        (section 3.4, from the swirl closure).
    row_delta_s : per-row lagged entropy rise, ``{row_id: (n_sl,)}``
        (section 3.5, Appendix-B-converted by the loss model).
    validity : aggregate closure validity in [0, 1] (section 7.3.3).
    iteration_tag : outer-iterate counter that produced these fields, for
        convergence-record attribution (ARCH-3.3).
    """

    blockage: np.ndarray
    row_exit_rvt: dict = field(default_factory=dict)
    row_delta_s: dict = field(default_factory=dict)
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
    spec : operating specification. ``MassFlowSpec`` (normal mode) or, since
        M5, ``BackPressureSpec`` (choke-proximal: ``mdot`` joins the state and
        the assembler appends the section 6.6 back-pressure residual at the
        throttling station).
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
    q_fixed : the ``(1, n_qo)`` fixed mean-line q-positions for the Tier-1
        meanline (``n_sl = 1``, section 8): repositioning is off, so the
        single mid-``psi`` streamline sits at the area-rule position, which
        is *not* a state variable and therefore enters as frozen data. The
        driver supplies ``initialize_positions(topology)``. Required when
        ``n_sl == 1`` and ignored otherwise (walls-plus-interior tiers rebuild
        their positions from ``x``).
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
    q_fixed: np.ndarray = None
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
        if isinstance(self.spec, BackPressureSpec):
            if not (0 <= self.spec.station < self.topology.n_qo):
                raise ConfigError(
                    f"BackPressureSpec.station {self.spec.station} out of "
                    f"range [0, {self.topology.n_qo})")
        elif not isinstance(self.spec, MassFlowSpec):
            raise ConfigError(
                "spec must be MassFlowSpec or BackPressureSpec, got "
                f"{type(self.spec).__name__}")
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
        # Tier-1 meanline (section 8): the fixed mean-line position is frozen
        # data, not state. Validate it here at the config boundary (AD-10) so
        # the assembler's split() never has to.
        if self.topology.n_sl == 1:
            if self.q_fixed is None:
                raise ConfigError(
                    "Tier-1 meanline (n_sl = 1) requires q_fixed, the "
                    "area-rule mean-line position (section 8, repositioning "
                    "off); the driver supplies initialize_positions(topology)")
            qf = np.asarray(self.q_fixed, dtype=float)
            if qf.shape != shape:
                raise ConfigError(
                    f"q_fixed shape {qf.shape} != (n_sl, n_qo) = {shape}")
            object.__setattr__(self, "q_fixed", qf)

    @property
    def n_sl(self) -> int:
        return self.topology.n_sl

    @property
    def n_qo(self) -> int:
        return self.topology.n_qo
