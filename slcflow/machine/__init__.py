"""Machine composition facade and MDO surface (Architecture Spec ARCH-5.5;
Theory Manual section 8).

``Machine.evaluate(spec, fidelity, n_sl, ...) -> PerformanceResult`` is the
single user-facing entry point: it composes a flow path, its blade rows, the
working fluid, and an inlet condition into a grid at the requested spanwise
resolution, runs the classical driver, and reduces the converged field to the
scalars an outer loop (cycle deck, MDO optimizer) needs — mass flow, pressure
ratio, efficiency, aggregate closure validity — plus the spanwise exit
profiles and the raw :class:`ClassicalResult` for replay/warm-starting.

Tiers are data (AD-1): the same call serves every fidelity. Tier 1 (meanline,
section 8) is simply ``n_sl = 1`` — the driver and assembler degenerate to the
single mid-``psi`` streamline with a one-point area rule; Tier 2/3 use a
spanwise grid. Nothing here branches on tier beyond forwarding the
``FidelityConfig`` flags and the caller's ``n_sl``.

This is the facade layer of ARCH-2 (``drivers -> machine -> io``); it imports
the driver, never the reverse. ``FidelityConfig`` and the operating specs are
re-exported here for one-import user composition.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # facade layer: orchestration + reporting, not residual path  # ad6: allow

from ..drivers.classical import (ClassicalConfig, ClassicalResult, RowSpec,
                                  solve_classical)
from ..errors import ConfigError
from ..fluid.base import WorkingFluid
from ..geometry.flowpath import FlowPath
from ..grid.core import GridTopology, MetricsConfig
from ..transport.streamwise import TransportFields
from ..types import (BackPressureSpec, FidelityConfig, MassFlowSpec,
                     OperatingSpec)

__all__ = ["InletCondition", "PerformanceResult", "Machine", "RowSpec",
           "FidelityConfig", "MassFlowSpec", "BackPressureSpec",
           "OperatingSpec"]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class InletCondition:
    """Stagnation inlet state at station 0 (section 6.2.1 boundary data).

    ``h0`` and ``s`` are the inlet stagnation enthalpy and entropy; ``rvt`` is
    the inlet ``r*V_theta``. Each is a scalar (uniform inflow, the usual case)
    or a callable of the mass-fraction array ``psi`` returning an ``(n_sl,)``
    profile — evaluated at solve time so one machine serves every ``n_sl``
    (including the ``psi = [0.5]`` meanline)."""

    h0: float
    s: float = 0.0
    rvt: object = 0.0          # float or callable(psi) -> (n_sl,)

    def fields(self, psi) -> TransportFields:
        psi = np.asarray(psi, dtype=float)
        rvt = self.rvt(psi) if callable(self.rvt) else self.rvt
        return TransportFields(
            h0=np.full(psi.shape, float(self.h0)),
            s=np.full(psi.shape, float(self.s)),
            rvt=np.broadcast_to(np.asarray(rvt, dtype=float),
                                psi.shape).astype(float))


@dataclass(frozen=True)
class PerformanceResult:
    """Reduced performance picture for coupled/MDO use (ARCH-5.5).

    Scalars are total-to-total, mass-averaged over the exit span with the
    section 3.2 mass-flux weight (a single point at Tier 1). ``result`` is the
    full :class:`ClassicalResult` — the deterministic replay/warm-start handle
    (AD-3). ``choke_margin``/``stall_margin`` are reserved for the continuation
    /operability milestone (M5/V9, ARCH-8) and are ``None`` here.

    Parameters
    ----------
    converged : whether the driver reached the section 6.2.5 tolerances.
    mdot : achieved mass flow [kg/s] (the target at convergence).
    pressure_ratio : total-to-total stagnation pressure ratio (exit/inlet).
    efficiency : total-to-total isentropic efficiency (perfect-gas backend;
        real-gas lands with the CoolProp wrapper, ARCH-9). A compressor value
        in (0, 1); ``> 1`` inverts the isentropic reference for a turbine.
    validity : aggregate closure validity in [0, 1] (section 7.3.3).
    r, vm, alpha, p0, T0 : spanwise exit profiles (absolute flow angle
        ``alpha`` in radians from meridional, section 2.4).
    """

    status: object
    converged: bool
    mdot: float
    pressure_ratio: float
    efficiency: float
    validity: float
    r: np.ndarray
    vm: np.ndarray
    alpha: np.ndarray
    p0: np.ndarray
    T0: np.ndarray
    result: ClassicalResult = field(repr=False)


class Machine:
    """A composed turbomachine ready to evaluate (ARCH-5.5).

    Parameters
    ----------
    flowpath : the annulus + station topology (walls, EDGE/DUCT stations).
    fluid : working-fluid backend (section 3.7).
    inlet : :class:`InletCondition` at station 0.
    rows : blade-row :class:`RowSpec` sequence (closure-fed); empty for a
        pure duct (V1/V2-style). Must match the flow path's declared rows.
    blockage : optional prescribed ``B(psi)`` callable or scalar (section
        7.2); default zero.
    """

    def __init__(self, flowpath: FlowPath, fluid: WorkingFluid,
                 inlet: InletCondition, rows=(), blockage=0.0):
        if not isinstance(inlet, InletCondition):
            raise ConfigError("inlet must be an InletCondition")
        self.flowpath = flowpath
        self.fluid = fluid
        self.inlet = inlet
        self.rows = tuple(rows)
        self.blockage = blockage

    # ------------------------------------------------------------------
    def evaluate(self, spec: OperatingSpec, fidelity: FidelityConfig,
                 n_sl: int, *, warm_start: PerformanceResult = None,
                 config: ClassicalConfig = None,
                 metrics_config: MetricsConfig = None,
                 mixing=None) -> PerformanceResult:
        """Solve one operating point and reduce it to a PerformanceResult.

        ``n_sl`` is the spanwise resolution and the sole tier-1/2/3 topology
        knob (section 8): ``n_sl = 1`` is the meanline, ``5-11`` streamline-
        REE, ``11-21+`` full SLC. ``warm_start`` is accepted for interface
        stability but only consumed once the Newton/continuation drivers land
        (M5, ARCH-8); the classical driver cold-starts from the area rule.
        """
        if not isinstance(spec, MassFlowSpec):
            raise ConfigError(
                "the classical driver supports MassFlowSpec only; "
                "BackPressureSpec continuation lands at M5 (ARCH-8)")
        topo = GridTopology(self.flowpath, n_sl=n_sl)
        inlet = self.inlet.fields(topo.psi)
        blk = self.blockage(topo.psi) if callable(self.blockage) \
            else self.blockage
        blockage = np.broadcast_to(np.asarray(blk, dtype=float),
                                   (topo.n_sl, topo.n_qo)).astype(float)
        kwargs = {} if config is None else {"config": config}
        if metrics_config is not None:
            kwargs["metrics_config"] = metrics_config
        if mixing is not None:
            kwargs["mixing"] = mixing
        res = solve_classical(topo, self.fluid, fidelity, spec, inlet,
                              rows=self.rows, blockage=blockage, **kwargs)
        return self._reduce(res, spec)

    # ------------------------------------------------------------------
    def _reduce(self, res: ClassicalResult, spec: MassFlowSpec
                ) -> PerformanceResult:
        """Mass-averaged total-to-total scalars + exit profiles from a solve."""
        fluid = self.fluid
        if res.fields is None:            # boundary check fired pre-assembly
            nan = float("nan")
            empty = np.array([])
            return PerformanceResult(
                status=res.status, converged=res.converged, mdot=spec.mdot,
                pressure_ratio=nan, efficiency=nan,
                validity=(res.frozen.closures.validity
                          if res.frozen is not None else 0.0),
                r=empty, vm=empty, alpha=empty, p0=empty, T0=empty,
                result=res)

        tr = res.frozen.transported
        j_in, j_ex = 0, res.frozen.n_qo - 1
        h0_in = self._mavg(tr.h0[:, j_in], res, j_in)
        h0_ex = self._mavg(tr.h0[:, j_ex], res, j_ex)
        p0_in = self._mavg(fluid.p(tr.h0[:, j_in], tr.s[:, j_in]), res, j_in)
        p0_ex = self._mavg(fluid.p(tr.h0[:, j_ex], tr.s[:, j_ex]), res, j_ex)
        pr = p0_ex / p0_in
        # Total-to-total isentropic efficiency. Isentropic exit stagnation
        # enthalpy on the inlet isentrope at the exit stagnation pressure:
        # h0s = h0_in * PR^((gamma-1)/gamma) (perfect gas; ARCH-9 defers the
        # real-gas form). eta_c = (h0s - h0_in)/(h0_ex - h0_in); the same
        # ratio reads > 1 for a turbine (dh0 < 0), a documented sign, not a
        # branch.
        kappa = (fluid.gamma - 1.0) / fluid.gamma
        h0s = h0_in * pr ** kappa
        dh0 = h0_ex - h0_in
        eta = (h0s - h0_in) / dh0 if abs(dh0) > 1e-9 * abs(h0_in) \
            else float("nan")

        r = res.fields.metrics.r[:, j_ex]
        vm = res.fields.vm[:, j_ex]
        vtheta = tr.rvt[:, j_ex] / r
        p0 = fluid.p(tr.h0[:, j_ex], tr.s[:, j_ex])
        T0 = fluid.T(tr.h0[:, j_ex], tr.s[:, j_ex])
        return PerformanceResult(
            status=res.status, converged=res.converged, mdot=spec.mdot,
            pressure_ratio=float(pr), efficiency=float(eta),
            validity=float(res.frozen.closures.validity),
            r=r, vm=vm, alpha=np.arctan2(vtheta, vm), p0=p0, T0=T0,
            result=res)

    @staticmethod
    def _mavg(phi, res: ClassicalResult, j) -> float:
        """Mass-flux-weighted span average of ``phi`` at station ``j``
        (weight ``rho Vm cos(eps) r``, section 3.2). A single point at Tier 1
        returns itself — the meanline value IS the mass average by
        construction."""
        phi = np.asarray(phi, dtype=float)
        if phi.size == 1:
            return float(phi[0])
        f = res.fields
        w = (f.rho[:, j] * f.vm[:, j] * np.cos(f.metrics.eps[:, j])
             * f.metrics.r[:, j])
        q = f.q[:, j]
        return float(np.trapezoid(w * phi, q) / np.trapezoid(w, q))
