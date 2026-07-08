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

Known limitations (recorded per ARCH-9 discipline):
  * A crossing-streamline iterate (non-monotone q on a q-o) raises from
    scipy's PCHIP constructor rather than saturating (AD-10 letter). The
    classical driver structurally cannot produce one — its repositioning is
    a convex blend of monotone position vectors (tested at M3) — so this is
    reachable only through externally supplied ``x``; the Newton driver's
    line-search globalization must reject such trial steps (M5).
  * The NaN/Inf boundary check lives in the classical driver (input side +
    assembled fields); the ARCH-6 reproducer-bundle serialization is M5.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # assembly is numpy/scipy-bound via PCHIP (5.3)  # ad6: allow
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

from ..errors import ConfigError
from ..grid.core import GridMetrics, evaluate_metrics
from ..grid.quadrature import cumulative
from ..types import BackPressureSpec
from .inputs import FrozenInputs
from .pack import unpack

__all__ = ["AssembledFields", "ResidualAssembler"]

_TWO_PI = 2.0 * np.pi
_CAPACITY_SCAN = 96      # coarse-scan points for the A.7 capacity search
_DENSE_AREA = 400        # dense samples for the Tier-1 area measure (G-5 rule)


class _Const:
    """A constant "interpolant" for the Tier-1 meanline (section 8): with a
    single mid-``psi`` node there is nothing to interpolate *across* span, and
    the master ODE is trivial (one node -> the integrator's node-to-node loop
    runs zero times and never evaluates these). Built anyway so the
    ``_QoDistributions`` shape is uniform across tiers (no parallel path,
    AD-1); ``derivative`` returns the zero constant."""

    def __init__(self, c):
        self._c = float(c)

    def __call__(self, q):
        return np.full(np.shape(q), self._c) if np.ndim(q) else self._c

    def derivative(self):
        return _Const(0.0)


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
        # n_sl == 1 is the Tier-1 meanline (section 8): a single mid-psi
        # streamline, walls entering only as the q-o LENGTH measure. n_sl >= 2
        # is the walls-plus-interior form (section 6.1). Both run THE SAME
        # assembly below (AD-1) — the only differences are the coarsest
        # quadrature and a trivial one-node ODE, selected by topology integer,
        # never by a flow value. FrozenInputs already validated q_fixed for
        # the n_sl == 1 case at the config boundary.
        if frozen.n_sl < 1:
            raise ConfigError(
                f"ResidualAssembler needs n_sl >= 1, got {frozen.n_sl}")
        self.frozen = frozen
        # Tier-1 meanline area measure: the geometric ``integral of r dq`` per
        # q-o (the annulus area / 2*pi), the frozen part of the mass integral
        # the meanline flux multiplies (section 8). Built with THE shared
        # quadrature rule on a dense geometry sample — the same rule (and the
        # same dense-r cumulative) that ``grid.initialize_positions`` places
        # the mean line with, so the flux point and its measure are consistent
        # (section 5.4). Geometry is frozen (AD-8), so this is computed once.
        self._area_measure = None
        if frozen.n_sl == 1:
            self._area_measure = np.array([
                float(cumulative(qo.point(
                    np.linspace(0.0, qo.length, _DENSE_AREA))[1],
                    np.linspace(0.0, qo.length, _DENSE_AREA))[-1])
                for qo in frozen.topology.flowpath.qo_curves])

    # ------------------------------------------------------------------
    # Preparation: x -> geometry, metrics, interpolants
    # ------------------------------------------------------------------
    def _full_q(self, q_interior):
        """Nodal q with wall rows attached: walls sit at ``q = 0`` and
        ``q = qo_length`` (section 6.1; which physical wall is which is the
        A.1.1 orientation, never assumed here — AD-9).

        Tier-1 exception (section 8): the meanline has no wall streamlines in
        the state — the single mid-``psi`` node sits at the fixed area-rule
        position (``frozen.q_fixed``), and the walls enter only through the
        q-o LENGTH measure in :meth:`mass_cumulative`. Returning the single
        node here is what makes the master ODE trivial."""
        topo = self.frozen.topology
        if topo.n_sl == 1:
            return self.frozen.q_fixed
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
        # Optional curvature under-relaxation (section 5.5): blend with the
        # lagged field. Branching on the *presence* of a lagged field is
        # config/topology branching, not a flow-value branch (AD-6).
        kappa = metrics.kappa_m
        if fz.kappa_lagged is not None:
            kappa = (fz.kappa_relax * metrics.kappa_m
                     + (1.0 - fz.kappa_relax) * fz.kappa_lagged)
        kcos = kappa * np.cos(metrics.eps)
        leansin = dvm_dm * np.sin(metrics.eps)

        dists = []
        for j, qo in enumerate(fz.topology.flowpath.qo_curves):
            qn = q_full[:, j]

            def interp(vals, _qn=qn):
                # Tier-1 meanline: a single span node -> no PCHIP is
                # constructible (needs >= 2 nodes) and none is needed (the
                # one-node ODE never evaluates it). A constant carries the
                # nodal value with a zero derivative (section 8).
                if _qn.size == 1:
                    return _Const(vals[0])
                return PchipInterpolator(_qn, vals, extrapolate=True)

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
    def mass_cumulative(self, j, vm, fields: AssembledFields):
        """Cumulative of ``rho Vm cos(eps) (1 - B) r`` over the nodal
        partition with THE shared quadrature rule (section 5.4).

        Public reusable piece (ARCH-5.2): the classical driver's streamline
        repositioning inverts exactly this cumulative, which is what keeps
        the repositioning targets and the position residual consistent
        (the section 5.4 same-rule requirement)."""
        fz = self.frozen
        r = fields.metrics.r[:, j]
        vt = fz.transported.rvt[:, j] / r
        h = fz.transported.h0[:, j] - 0.5 * (vm * vm + vt * vt)
        rho = fz.fluid.rho(h, fz.transported.s[:, j])
        if fz.n_sl == 1:
            # Tier-1 meanline (section 8): the wall-to-wall mass integral
            # factored as [flux]_mean * (integral of r dq) — the flux
            # (WITHOUT the r weight) evaluated at the single mid-psi node,
            # times the frozen geometric area measure. This is the textbook
            # meanline area rule and the coarsest instance of THE section 5.4
            # quadrature, not a separate rule; the length-1 cumulative keeps
            # continuity_F/qo_capacity/residual reading ``[-1]`` exactly as at
            # higher n_sl.
            flux = (rho * vm * np.cos(fields.metrics.eps[:, j])
                    * (1.0 - fz.closures.blockage[:, j]))
            return flux * self._area_measure[j]
        integrand = (rho * vm * np.cos(fields.metrics.eps[:, j])
                     * (1.0 - fz.closures.blockage[:, j]) * r)
        return cumulative(integrand, fields.q[:, j])

    def continuity_F(self, j, vm_q0, fields: AssembledFields) -> float:
        """Q-o continuity residual ``F_j(Vm_q0)`` (section 5.4)."""
        vm = self._integrate(fields.dists[j], vm_q0)
        return float(_TWO_PI * self.mass_cumulative(j, vm, fields)[-1]
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
            vm = self._integrate(d, v)
            # Positive-branch guard (matching the driver's _solve_qo
            # feasibility rule, V8 Tier-3 diagnosis 2026-07): a profile that
            # crossed the master ODE's Vm = 0 singularity is a spurious
            # branch and carries no physical capacity.
            if not bool(np.all(np.isfinite(vm)) and np.all(vm > 0.0)):
                return np.inf
            val = _TWO_PI * self.mass_cumulative(j, vm, fields)[-1]
            return float(np.nan_to_num(-val, nan=np.inf, posinf=np.inf,
                                       neginf=np.inf))

        grid = np.linspace(vm_hi / _CAPACITY_SCAN, vm_hi, _CAPACITY_SCAN)
        scan = np.array([neg_mdot(v) for v in grid])
        k = int(np.nanargmin(scan))
        if not np.isfinite(scan[k]):
            return 0.0      # no positive branch anywhere: zero capacity
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, grid.size - 1)]
        res = minimize_scalar(neg_mdot, bounds=(lo, hi), method="bounded")
        return -min(float(res.fun), float(scan[k]))

    # ------------------------------------------------------------------
    # ARCH-5.1 public assembly
    # ------------------------------------------------------------------
    @property
    def backpressure(self) -> bool:
        """Choke-proximal mode (section 6.6): the spec fixes exit pressure and
        ``mdot`` is a state unknown carrying one extra residual."""
        return isinstance(self.frozen.spec, BackPressureSpec)

    def split(self, x) -> AssembledFields:
        """Assembled per-q-o picture from the state vector: metrics, the
        eliminated ``Vm`` field, and thermodynamics (ARCH-5.1). The
        back-pressure-mode ``mdot`` unknown does not enter the field picture
        (it is a continuity *target*), so it is ignored here."""
        fz = self.frozen
        vm_q0, q_int, _ = unpack(x, fz.n_sl, fz.n_qo,
                                 backpressure=self.backpressure)
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
        C-order, matching ``assembly.pack``. In back-pressure mode (section
        6.6) ``mdot`` is read from the trailing state component and ONE more
        row is appended: the throttling-station back-pressure condition."""
        return self.residual_from(self.split(x), x)

    def residual_from(self, fields: AssembledFields, x):
        """The section 6.1 residual rows evaluated on an already-assembled
        ``fields`` (ARCH-5.1 reusable piece): lets a driver share ONE
        ``split`` between its own trial screening and the residual — the
        Newton positive-branch guard inspects ``fields.vm`` before paying
        for the rows (2026-07 stabilization follow-up). ``fields`` must be
        ``self.split(x)`` for the same ``x``; nothing revalidates that
        here. ``residual(x)`` is exactly this after its own split."""
        fz = self.frozen
        if self.backpressure:
            _, _, mdot = unpack(x, fz.n_sl, fz.n_qo, backpressure=True)
        else:
            mdot = fz.spec.mdot
        cums = [self.mass_cumulative(j, fields.vm[:, j], fields)
                for j in range(fz.n_qo)]
        r_cont = np.array([_TWO_PI * c[-1] - mdot for c in cums])
        psi_int = fz.topology.psi[1:-1]
        r_pos = np.stack(
            [c[1:-1] - psi_int * (mdot / _TWO_PI) for c in cums],
            axis=1) if fz.n_sl > 2 else np.zeros((0, fz.n_qo))
        rows = [r_cont, np.ravel(r_pos)]
        if self.backpressure:
            rows.append(self._backpressure_residual(fields))
        return np.concatenate(rows)

    def _backpressure_residual(self, fields: AssembledFields):
        """Section 6.6 back-pressure condition: the static pressure at the
        ``q = 0`` node of the throttling station equals the specified exit
        pressure. This is the "hub-velocity level at the throttling station"
        handle — one scalar residual balancing the added ``mdot`` unknown.
        (Which physical wall is ``q = 0`` is the A.1.1 orientation, AD-9.)"""
        fz = self.frozen
        st = fz.spec.station
        vm0 = fields.vm[0, st]
        r0 = fields.metrics.r[0, st]
        vt0 = fz.transported.rvt[0, st] / r0
        h = (fz.transported.h0[0, st]
             - 0.5 * (vm0 * vm0 + vt0 * vt0))
        p_static = fz.fluid.p(h, fz.transported.s[0, st])
        return np.array([p_static - fz.spec.p_exit])
