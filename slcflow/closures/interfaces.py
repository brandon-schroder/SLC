"""Closure interfaces and the views closures see (Theory Manual sections
7.1-7.2, 4.1; Architecture Spec ARCH-4.2).

Machine-type knowledge lives behind these Protocols (AD-5): the kernel and
drivers import THIS module, never correlation implementations. ``RowView``
and ``RowFlowView`` are constructed by the driver/assembler and expose only
the section 4.1 row data contract and the local circumferentially averaged
flow (section 7.2) — closures physically cannot reach solver internals.

Every closure returns a validity measure in [0, 1] (section 7.3.3) and must
obey the C1 smoothness rules (section 7.3) — built from
``closures.smoothmath``, enforced by review + the AD-6 lint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from ..errors import ConfigError
from ..fluid.base import Array, WorkingFluid

__all__ = ["RowView", "RowFlowView", "LossBreakdown", "SwirlResult",
           "LossModel", "SwirlClosure", "BlockageModel", "MixingModel",
           "CorrelationSet"]


@dataclass(frozen=True)
class RowView:
    """The section 4.1 row data contract as seen by closures.

    ``geometry`` carries the BladeRowGeometry callables-of-span-fraction
    (metal angles, chord, solidity, thicknesses, throat, ...); it is typed
    loosely until the first geometry-consuming correlation lands (M4-3) —
    prescribed closures ignore it.
    """

    row_id: str
    omega: float                 # shaft speed [rad/s]; 0 for stators
    blade_count: int = 0         # 0 = unspecified (prescribed closures)
    geometry: object = None      # BladeRowGeometry (section 4.1), M4-3


@dataclass(frozen=True)
class RowFlowView:
    """Local circumferentially averaged flow at a row station (section 7.2),
    one value per streamtube crossing the row. All SI, angles in radians
    (AD-7); relative quantities use the section 2.4 sign convention
    (positive toward rotor rotation)."""

    psi: Array          # streamtube mass fractions
    r: Array            # radius [m]
    vm: Array           # meridional velocity [m/s]
    vtheta: Array       # absolute tangential velocity [m/s]
    w_theta: Array      # relative tangential velocity [m/s]
    alpha: Array        # absolute swirl angle [rad]
    beta: Array         # relative flow angle [rad]
    h: Array            # static enthalpy [J/kg]
    s: Array            # entropy [J/(kg K)]
    T: Array
    rho: Array
    a: Array            # speed of sound [m/s]
    fluid: WorkingFluid = field(repr=False, default=None)


@dataclass(frozen=True)
class LossBreakdown:
    """Loss-model output (ARCH-4.2): source-native component coefficients
    preserved for diagnostics (section 4.4), the CONVERTED per-streamtube
    entropy rise (Appendix B applied inside the model), and validity."""

    components: Mapping[str, Array]
    delta_s: Array
    validity: float = 1.0

    def __post_init__(self):
        if not (0.0 <= float(self.validity) <= 1.0):
            raise ConfigError(f"validity must be in [0, 1], got {self.validity}")


@dataclass(frozen=True)
class SwirlResult:
    """Swirl-closure output: exit rVt per streamtube (section 3.4) and
    validity."""

    rvt: Array
    validity: float = 1.0

    def __post_init__(self):
        if not (0.0 <= float(self.validity) <= 1.0):
            raise ConfigError(f"validity must be in [0, 1], got {self.validity}")


@runtime_checkable
class LossModel(Protocol):
    """Row loss evaluation (section 7.1): components + converted delta_s."""

    def evaluate(self, row: RowView, flow: RowFlowView) -> LossBreakdown: ...


@runtime_checkable
class SwirlClosure(Protocol):
    """Row exit swirl (section 7.1): deviation- or slip-based internally,
    unified as exit rVt given LE flow."""

    def exit_rvt(self, row: RowView, flow: RowFlowView) -> SwirlResult: ...


@runtime_checkable
class BlockageModel(Protocol):
    """Endwall blockage field B(i, j) (sections 3.2, 7.2)."""

    def blockage(self, topology, metrics) -> Array: ...


@runtime_checkable
class MixingModel(Protocol):
    """Spanwise mixing coefficient field (section 3.6; lands M8)."""

    def mu_mix(self, flow) -> Array: ...


@dataclass(frozen=True)
class CorrelationSet:
    """Named, versioned closure bundle per machine type (section 7.1,
    ARCH-4.2) with a documented provenance string per member. Mixing sets
    across rows of one machine is allowed but warned (driver's job)."""

    name: str
    swirl: SwirlClosure
    loss: LossModel
    blockage: BlockageModel = None
    mixing: MixingModel = None
    provenance: str = ""
