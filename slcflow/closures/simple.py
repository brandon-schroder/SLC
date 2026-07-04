"""Prescribed (machine-agnostic) closures (Theory Manual section 7.1).

Design-mode building blocks and the reference implementations of the
closure Protocols: the user prescribes the exit swirl / row loss directly
instead of correlating it from geometry. Constant prescriptions are trivially
C1 in every flow input (section 7.3) with validity 1 everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..fluid.base import Array
from .interfaces import LossBreakdown, RowFlowView, RowView, SwirlResult

__all__ = ["PrescribedSwirl", "PrescribedLoss"]


@dataclass(frozen=True)
class PrescribedSwirl:
    """Exit rVt prescribed per streamtube (scalar or ``(n_sl,)`` array,
    broadcast by the transport sweep). The direct design-mode swirl input
    of section 3.4."""

    rvt: Array

    def exit_rvt(self, row: RowView, flow: RowFlowView) -> SwirlResult:
        return SwirlResult(rvt=self.rvt, validity=1.0)


@dataclass(frozen=True)
class PrescribedLoss:
    """Row entropy rise prescribed per streamtube [J/(kg K)] — already in
    the internal loss currency, so no Appendix B conversion applies."""

    delta_s: Array = 0.0

    def evaluate(self, row: RowView, flow: RowFlowView) -> LossBreakdown:
        return LossBreakdown(components={"prescribed": self.delta_s},
                             delta_s=self.delta_s, validity=1.0)
