"""Residual assembler (Theory Manual sections 5.3, 5.4, 6.1, A.7; ARCH-5.1).

``ResidualAssembler.residual`` is a pure function of ``(x, FrozenInputs)``
(AD-3): unpack the state, rebuild metrics from it, integrate the master
equation along every q-o (the section 6.1 *elimination* — momentum is
satisfied by construction and contributes no residual rows), and evaluate
the continuity and streamtube-position residuals with THE shared quadrature
rule (section 5.4 consistency requirement).

Numpy/scipy-bound like the grid layer (PCHIP interpolation is a section 5.3
requirement); the AD-6 xp-injection story for this module follows whatever
resolution the grid layer's spline dependency eventually gets.

Known limitations, deliberate for M2 (recorded per ARCH-9 discipline):
  * A crossing-streamline iterate (non-monotone q on a q-o) raises from
    scipy's PCHIP constructor rather than saturating (AD-10 letter). The
    repositioning relaxation that prevents such iterates is M3 machinery;
    revisit the guard there.
  * The single NaN/Inf boundary check with diagnostic dump (ARCH-6) lands
    with the driver sub-step, which owns the reproducer-bundle plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # assembly is numpy/scipy-bound via PCHIP (5.3)  # ad6: allow
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from ..errors import ConfigError
from ..grid.core import GridMetrics, evaluate_metrics
from ..grid.quadrature import cumulative
from .inputs import FrozenInputs
from .pack import unpack

__all__ = ["AssembledFields", "ResidualAssembler"]

_TWO_PI = 2.0 * np.pi
_CAPACITY_SCAN = 96      # coarse-scan points for the A.7 capacity search


@dataclass(frozen=True)
class _QoDistributions:
    """Monotone-cubic (PCHIP, section 5.3) interpolants of the master-ODE
    right-hand-side distributions along one q-o, plus the exact wall-to-wall
    radius map of the (straight) q-o curve."""

    q_nodes: np.ndarray
    h0: PchipInterpolator
    dh0: PchipInterpolator
    s: PchipInterpolator
    ds: PchipInterpolator
    rvt: PchipInterpolator
    drvt: PchipInterpolator
    kcos: PchipInterpolator        # kappa_m * cos(eps)
    leansin: PchipInterpolator     # (dVm/dm, lagged) * sin(eps)
    r_of: object = field(repr=False)  # q -> r, exact from the q-o curve


@dataclass(frozen=True)
class AssembledFields:
    """Per-``x`` assembled picture (ARCH-5.1 ``split``): nodal positions,
    metrics, the eliminated ``Vm`` field, and thermodynamics. ``dists``
    carries the section 5.3 interpolants so the reusable pieces
    (``integrate_master_ode``, ``continuity_F``, ``qo_capacity``) can be
    driven at arbitrary trial ``Vm_q0`` without re-preparation."""

    q: np.ndarray                  # (n_sl, n_qo) incl. wall rows
    metrics: GridMetrics
    vm: np.ndarray
    h: np.ndarray
    T: np.ndarray
    rho: np.ndarray
    mach_m: np.ndarray
    dists: tuple = field(repr=False)


class ResidualAssembler:
    """Assembles the section 6.1 residual vector for the MassFlowSpec form.

    Residual ordering matches ``assembly.pack``: ``n_qo`` continuity rows
    (section 5.4), then the interior streamtube mass-fraction rows
    (section 6.1) in C-order — so the Jacobian pairs unknowns and residuals
    consistently.
    """

    def __init__(self, frozen: FrozenInputs):
        if frozen.n_sl < 2:
            raise ConfigError(
                "ResidualAssembler needs n_sl >= 2 (wall streamlines fixed "
                "to the annulus, section 6.1); the Tier-1 n_sl = 1 mode "
                "arrives with the machine facade (M4)")
        self.frozen = frozen

    # ------------------------------------------------------------------
    # Preparation: x -> geometry, metrics, interpolants
    # ------------------------------------------------------------------
    def _full_q(self, q_interior):
        """Nodal q with wall rows attached: walls sit at ``q = 0`` and
        ``q = qo_length`` (section 6.1; which physical wall is which is the
        A.1.1 orientation, never assumed here — AD-9)."""
        topo = self.frozen.topology
        lengths = np.array([qo.length for qo in topo.flowpath.qo_curves])
        n_qo = topo.n_qo
        return np.concatenate([np.zeros((1, n_qo)), q_interior,
                               lengths.reshape(1, n_qo)], axis=0)

    def _build_dists(self, q_full, metrics: GridMetrics):
        fz = self.frozen
        # Lagged dVm/dm along each streamline: non-uniform second-order
        # differences in meridional arc length (section 5.2).
        dvm_dm = np.stack(
            [np.gradient(fz.vm_lagged[i, :], metrics.m[i, :])
             for i in range(fz.n_sl)], axis=0)
        kcos = metrics.kappa_m * np.cos(metrics.eps)
        leansin = dvm_dm * np.sin(metrics.eps)

        dists = []
        for j, qo in enumerate(fz.topology.flowpath.qo_curves):
            qn = q_full[:, j]

            def interp(vals):
                return PchipInterpolator(qn, vals, extrapolate=True)

            p_h0 = interp(fz.transported.h0[:, j])
            p_s = interp(fz.transported.s[:, j])
            p_rvt = interp(fz.transported.rvt[:, j])
            dists.append(_QoDistributions(
                q_nodes=qn,
                h0=p_h0, dh0=p_h0.derivative(),
                s=p_s, ds=p_s.derivative(),
                rvt=p_rvt, drvt=p_rvt.derivative(),
                kcos=interp(kcos[:, j]),
                leansin=interp(leansin[:, j]),
                r_of=(lambda qq, _qo=qo: _qo.point(qq)[1]),
            ))
        return tuple(dists)

    # ------------------------------------------------------------------
    # Master-equation integration (section 5.3) — the 6.1 elimination
    # ------------------------------------------------------------------
    def _rhs(self, d: _QoDistributions, qq, vm):
        """dVm/dq at (q, Vm): the boxed section 3.1 equation divided by Vm.
        Tier-3-exclusive terms multiply FidelityConfig flags — data, not
        branches (AD-1/AD-6, ARCH-5.1). The in-blade force term is M7."""
        fid = self.frozen.fidelity
        r = d.r_of(qq)
        rvt = d.rvt(qq)
        vt = rvt / r
        h = d.h0(qq) - 0.5 * (vm * vm + vt * vt)
        T = self.frozen.fluid.T(h, d.s(qq))
        core = d.dh0(qq) - T * d.ds(qq) - (rvt / (r * r)) * d.drvt(qq)
        return (core / vm
                + fid.curvature_term * vm * d.kcos(qq)
                + fid.lean_term * d.leansin(qq))

    def _integrate(self, d: _QoDistributions, vm_q0):
        """RK2 (midpoint) node-to-node integration of the master ODE from
        the ``q = 0`` wall (section 5.3; second-order)."""
        vms = [float(vm_q0)]
        qn = d.q_nodes
        for i in range(qn.size - 1):
            dq = qn[i + 1] - qn[i]
            k1 = self._rhs(d, qn[i], vms[-1])
            vm_mid = vms[-1] + 0.5 * dq * k1
            k2 = self._rhs(d, qn[i] + 0.5 * dq, vm_mid)
            vms.append(vms[-1] + dq * k2)
        return np.array(vms)

    def integrate_master_ode(self, j, vm_q0, fields: AssembledFields):
        """Nodal ``Vm`` along q-o ``j`` for boundary value ``vm_q0``
        (ARCH-5.1 reusable piece; the one-parameter family of section 5.3)."""
        return self._integrate(fields.dists[j], vm_q0)

    # ------------------------------------------------------------------
    # Continuity (sections 3.2, 5.4) and capacity (A.7, section 6.6)
    # ------------------------------------------------------------------
    def _mass_cumulative(self, j, vm, fields: AssembledFields):
        """Cumulative of ``rho Vm cos(eps) (1 - B) r`` over the nodal
        partition with THE shared quadrature rule (section 5.4)."""
        fz = self.frozen
        r = fields.metrics.r[:, j]
        vt = fz.transported.rvt[:, j] / r
        h = fz.transported.h0[:, j] - 0.5 * (vm * vm + vt * vt)
        rho = fz.fluid.rho(h, fz.transported.s[:, j])
        integrand = (rho * vm * np.cos(fields.metrics.eps[:, j])
                     * (1.0 - fz.closures.blockage[:, j]) * r)
        return cumulative(integrand, fields.q[:, j])

    def continuity_F(self, j, vm_q0, fields: AssembledFields) -> float:
        """Q-o continuity residual ``F_j(Vm_q0)`` (section 5.4)."""
        vm = self._integrate(fields.dists[j], vm_q0)
        return float(_TWO_PI * self._mass_cumulative(j, vm, fields)[-1]
                     - self.frozen.spec.mdot)

    def qo_capacity(self, j, fields: AssembledFields) -> float:
        """Q-o mass-flow capacity ``mdot_max_j = max(F_j + mdot)`` (A.7,
        section 6.6): coarse scan + bounded refine over ``Vm_q0``.

        Scalar-optimization machinery for the drivers' operability logic —
        not part of the pure residual, so the iteration inside is acceptable
        (ARCH-4.3 keeps the switching *decision* in drivers)."""
        fz = self.frozen
        d = fields.dists[j]
        r0 = d.r_of(0.0)
        vt0 = fz.transported.rvt[0, j] / r0
        # Static enthalpy at the q = 0 node must stay positive: hard upper
        # bound on the scan window (perfect-gas domain, section 3.7).
        vm_hi = 0.999 * np.sqrt(np.maximum(
            2.0 * fz.transported.h0[0, j] - vt0 * vt0, 1e-12))

        def neg_mdot(v):
            val = _TWO_PI * self._mass_cumulative(
                j, self._integrate(d, v), fields)[-1]
            return float(np.nan_to_num(-val, nan=np.inf, posinf=np.inf,
                                       neginf=np.inf))

        grid = np.linspace(vm_hi / _CAPACITY_SCAN, vm_hi, _CAPACITY_SCAN)
        scan = np.array([neg_mdot(v) for v in grid])
        k = int(np.nanargmin(scan))
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        res = minimize_scalar(neg_mdot, bounds=(lo, hi), method="bounded")
        return -min(float(res.fun), float(scan[k]))

    # ------------------------------------------------------------------
    # ARCH-5.1 public assembly
    # ------------------------------------------------------------------
    def split(self, x) -> AssembledFields:
        """Assembled per-q-o picture from the state vector: metrics, the
        eliminated ``Vm`` field, and thermodynamics (ARCH-5.1)."""
        fz = self.frozen
        vm_q0, q_int = unpack(x, fz.n_sl, fz.n_qo)
        q_full = self._full_q(q_int)
        metrics = evaluate_metrics(fz.topology, q_full, fz.metrics_config)
        dists = self._build_dists(q_full, metrics)
        vm = np.stack([self._integrate(dists[j], vm_q0[j])
                       for j in range(fz.n_qo)], axis=1)
        vt = fz.transported.rvt / metrics.r
        h = fz.transported.h0 - 0.5 * (vm * vm + vt * vt)
        T = fz.fluid.T(h, fz.transported.s)
        rho = fz.fluid.rho(h, fz.transported.s)
        mach_m = vm / fz.fluid.a(h, fz.transported.s)
        return AssembledFields(q=q_full, metrics=metrics, vm=vm, h=h, T=T,
                               rho=rho, mach_m=mach_m, dists=dists)

    def residual(self, x):
        """The section 6.1 residual vector — pure in ``(x, FrozenInputs)``
        (AD-3). Rows: ``R_cont_j`` (section 5.4) for all j, then
        ``R_pos_ij`` (streamtube mass-fraction error) for interior i in
        C-order, matching ``assembly.pack``."""
        fz = self.frozen
        fields = self.split(x)
        mdot = fz.spec.mdot
        cums = [self._mass_cumulative(j, fields.vm[:, j], fields)
                for j in range(fz.n_qo)]
        r_cont = np.array([_TWO_PI * c[-1] - mdot for c in cums])
        psi_int = fz.topology.psi[1:-1]
        r_pos = np.stack(
            [c[1:-1] - psi_int * (mdot / _TWO_PI) for c in cums],
            axis=1) if fz.n_sl > 2 else np.zeros((0, fz.n_qo))
        return np.concatenate([r_cont, np.ravel(r_pos)])
