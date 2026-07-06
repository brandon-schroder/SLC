"""Spanwise-mixing operator tests (Theory Manual section 3.6; M8 sub-step 1).

The operator is an implicit (backward-Euler in m, tridiagonal in q) spanwise
diffusion of {h0, s, rVt}. The two properties that make it correct: it
**conserves** the mass-flux-weighted total of each field (zero-flux walls),
and it is **unconditionally stable** (any step size drives the profile toward
the weighted mean, never oscillates or diverges). Both are checked here
against the section 3.6 discretization, independent of the driver wiring.

Provenance: M8 sub-step 1, written with the implementation.
"""
import numpy as np
import pytest

from slcflow.closures.interfaces import MixingModel
from slcflow.transport import (GallimoreMixing, TransportFields,
                               mix_transported, spanwise_diffusion_step)


def _dqi(q):
    """Control-volume widths the operator uses (dual mesh) -- for the
    conservation invariant sum_i w_i dq_i chi_i."""
    d = np.diff(q)
    return np.concatenate([0.5 * d[:1], 0.5 * (d[1:] + d[:-1]), 0.5 * d[-1:]])


def _weighted_mean(chi, q, w):
    dq = _dqi(q)
    return float(np.sum(w * dq * chi) / np.sum(w * dq))


# --------------------------------------------------------------------------
# Section 3.6: conservation (zero-flux walls)
# --------------------------------------------------------------------------
def test_step_conserves_mass_flux_weighted_total():
    q = np.array([0.0, 0.1, 0.25, 0.45, 0.7, 1.0])   # nonuniform span
    w = np.array([1.2, 1.4, 1.5, 1.5, 1.3, 1.1])     # r(1-B)rho Vm
    mu_r = np.array([0.02, 0.03, 0.035, 0.03, 0.025, 0.02])
    chi = np.array([300.0, 305.0, 312.0, 318.0, 322.0, 320.0])
    before = np.sum(w * _dqi(q) * chi)
    out = spanwise_diffusion_step(chi, q, dm=0.05, weight=w, mu_r=mu_r)
    after = np.sum(w * _dqi(q) * out)
    np.testing.assert_allclose(after, before, rtol=1e-12)


def test_step_reduces_stratification():
    q = np.linspace(0.0, 1.0, 9)
    w = np.ones_like(q)
    mu_r = 0.04 * np.ones_like(q)
    chi = np.linspace(-10.0, 10.0, 9)                # strongly stratified
    out = spanwise_diffusion_step(chi, q, dm=0.05, weight=w, mu_r=mu_r)
    assert np.ptp(out) < np.ptp(chi)                 # spread shrinks
    # Interior gradients relax toward the mean; no new extrema (no overshoot).
    assert out.max() <= chi.max() + 1e-12
    assert out.min() >= chi.min() - 1e-12


# --------------------------------------------------------------------------
# Section 3.6: unconditional stability -> steady state is the weighted mean
# --------------------------------------------------------------------------
def test_unconditional_stability_large_step():
    q = np.linspace(0.0, 1.0, 11)
    w = 1.0 + 0.3 * q                                # nonuniform weight
    mu_r = 0.05 * np.ones_like(q)
    chi = np.cos(3.0 * q)                            # oscillatory profile
    target = _weighted_mean(chi, q, w)
    out = spanwise_diffusion_step(chi, q, dm=1.0e6, weight=w, mu_r=mu_r)
    assert np.all(np.isfinite(out))                  # never blows up
    # A huge step collapses to the conserved weighted mean, uniformly.
    np.testing.assert_allclose(out, target, atol=1e-3)


def test_many_small_steps_converge_to_weighted_mean():
    q = np.linspace(0.0, 1.0, 9)
    w = np.ones_like(q)
    mu_r = 0.03 * np.ones_like(q)
    chi0 = np.linspace(0.0, 1.0, 9) ** 2
    target = _weighted_mean(chi0, q, w)
    chi = chi0.copy()
    for _ in range(1500):
        chi = spanwise_diffusion_step(chi, q, dm=0.2, weight=w, mu_r=mu_r)
    # Converges to the (conserved) weighted mean: spread collapses > 99%.
    assert np.ptp(chi) < 0.01 * np.ptp(chi0)
    np.testing.assert_allclose(_weighted_mean(chi, q, w), target, rtol=1e-9)


def test_single_node_is_identity():
    q = np.array([0.3])
    chi = np.array([277.0])
    out = spanwise_diffusion_step(chi, q, dm=0.1, weight=np.array([1.0]),
                                  mu_r=np.array([0.02]))
    np.testing.assert_array_equal(out, chi)


def test_stacked_fields_share_operator():
    # h0/s/rVt diffused together (one matrix, three RHS) == separately.
    q = np.linspace(0.0, 1.0, 7)
    w = 1.0 + 0.2 * q
    mu_r = 0.03 * np.ones_like(q)
    stack = np.stack([np.linspace(300, 320, 7),
                      np.linspace(0.0, 2.0, 7),
                      np.linspace(5.0, 25.0, 7)])
    out = spanwise_diffusion_step(stack, q, dm=0.04, weight=w, mu_r=mu_r)
    for k in range(3):
        one = spanwise_diffusion_step(stack[k], q, dm=0.04, weight=w, mu_r=mu_r)
        np.testing.assert_allclose(out[k], one, rtol=1e-12)


# --------------------------------------------------------------------------
# mix_transported: marching wrapper + fidelity gating
# --------------------------------------------------------------------------
def _synthetic(n_sl=9, n_qo=5):
    q = np.linspace(0.0, 1.0, n_sl)
    m = np.broadcast_to(np.linspace(0.0, 0.4, n_qo), (n_sl, n_qo)).copy()
    r = np.broadcast_to(0.3 + 0.2 * q[:, None], (n_sl, n_qo)).copy()
    rho = np.full((n_sl, n_qo), 1.2)
    vm = np.full((n_sl, n_qo), 100.0)
    B = np.zeros((n_sl, n_qo))
    # Stratified but streamwise-constant fields (a duct): rVt varies in span.
    strat = np.linspace(-8.0, 8.0, n_sl)[:, None]
    tr = TransportFields(h0=3.0e5 + 50.0 * strat * np.ones((1, n_qo)),
                         s=np.zeros((n_sl, n_qo)),
                         rvt=20.0 + strat * np.ones((1, n_qo)))
    return tr, dict(m=m, r=r, blockage=B, rho=rho, vm=vm)


def test_mix_transported_off_is_identity():
    tr, co = _synthetic()
    mu = GallimoreMixing().mu_mix(
        type("F", (), {"rho": co["rho"], "vm": co["vm"], "r": co["r"]}))
    out = mix_transported(tr, mu_mix=mu, strength=0.0, **co)
    np.testing.assert_array_equal(out.rvt, tr.rvt)   # strength 0 -> untouched


def test_mix_transported_relaxes_stratification_downstream():
    tr, co = _synthetic()
    flow = type("F", (), {"rho": co["rho"], "vm": co["vm"], "r": co["r"]})
    mu = GallimoreMixing(c_mix=0.05).mu_mix(flow)
    out = mix_transported(tr, mu_mix=mu, strength=1.0, **co)
    # Inlet column untouched; exit column less stratified than inlet.
    np.testing.assert_allclose(out.rvt[:, 0], tr.rvt[:, 0], rtol=1e-12)
    assert np.ptp(out.rvt[:, -1]) < np.ptp(tr.rvt[:, -1])
    assert np.ptp(out.h0[:, -1]) < np.ptp(tr.h0[:, 0])
    # Mass-flux-weighted exit mean of rVt preserved (duct: no work/loss).
    w = (co["r"] * co["rho"] * co["vm"])[:, -1]
    q = np.linspace(0.0, 1.0, tr.rvt.shape[0])
    np.testing.assert_allclose(_weighted_mean(out.rvt[:, -1], q, w),
                               _weighted_mean(tr.rvt[:, -1], q, w), rtol=1e-6)


def test_gallimore_is_mixing_model():
    assert isinstance(GallimoreMixing(), MixingModel)


# --------------------------------------------------------------------------
# M8-2: wired through the classical driver (lagged field refresh)
# --------------------------------------------------------------------------
from slcflow.drivers import solve_classical                       # noqa: E402
from slcflow.fluid.perfectgas import PerfectGas                   # noqa: E402
from slcflow.geometry import (FlowPath, StationDef, StationType,  # noqa: E402
                              WallCurve)
from slcflow.grid import GridTopology                             # noqa: E402
from slcflow.types import FidelityConfig, MassFlowSpec            # noqa: E402

_GAS = PerfectGas()


def _duct_topology(n_sl=9, n_st=6):
    z = np.linspace(0.0, 1.0, 8)
    w0 = WallCurve.from_points(np.column_stack([z, np.full_like(z, 0.3)]))
    w1 = WallCurve.from_points(np.column_stack([z, np.full_like(z, 0.6)]))
    fr = np.linspace(0.0, 1.0, n_st)
    fp = FlowPath(w0, w1, [StationDef(StationType.DUCT, f, f) for f in fr])
    return GridTopology(fp, n_sl=n_sl)


def _stratified_inlet(n_sl):
    strat = np.linspace(-15.0, 15.0, n_sl)           # rVt varies across span
    return TransportFields(h0=np.full(n_sl, 3.0e5), s=np.zeros(n_sl),
                           rvt=20.0 + strat)


def test_driver_mixing_smooths_stratified_duct():
    topo = _duct_topology()
    inlet = _stratified_inlet(topo.n_sl)
    base = solve_classical(topo, _GAS, FidelityConfig.tier3(),
                           MassFlowSpec(80.0), inlet)
    mixed = solve_classical(topo, _GAS, FidelityConfig.tier3(mixing_term=1.0),
                            MassFlowSpec(80.0), inlet,
                            mixing=GallimoreMixing(c_mix=0.3))
    assert base.converged and mixed.converged
    b_rvt = base.frozen.transported.rvt[:, -1]
    m_rvt = mixed.frozen.transported.rvt[:, -1]
    # A duct conserves rVt per streamline, so the baseline exit stays fully
    # stratified; mixing relaxes it toward the (flux-weighted) mean.
    assert np.ptp(m_rvt) < 0.9 * np.ptp(b_rvt)


def test_driver_mixing_off_is_bit_identical():
    # mixing_term = 0 (tier3 default) disables the operator even if a model is
    # supplied -> protects the section 8 tier degeneracy / V3 identity.
    topo = _duct_topology()
    inlet = _stratified_inlet(topo.n_sl)
    a = solve_classical(topo, _GAS, FidelityConfig.tier3(),
                        MassFlowSpec(80.0), inlet)
    b = solve_classical(topo, _GAS, FidelityConfig.tier3(),
                        MassFlowSpec(80.0), inlet, mixing=GallimoreMixing())
    np.testing.assert_array_equal(a.frozen.transported.rvt,
                                  b.frozen.transported.rvt)
    np.testing.assert_array_equal(a.frozen.transported.h0,
                                  b.frozen.transported.h0)
