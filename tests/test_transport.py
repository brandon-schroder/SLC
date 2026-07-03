"""Tests for slcflow.transport (Theory Manual sections 3.3-3.5, B.5.1).

Each test cites the spec clause it verifies. Written in the same session as
the implementation (no adjudication needed; provenance: M2 sub-step 1).
"""
import numpy as np
import pytest

from slcflow.errors import ConfigError
from slcflow.transport import (
    SmoothRampSchedule,
    TransportStep,
    apply_step,
    rothalpy,
    row_steps,
    sweep,
)


# --------------------------------------------------------------------------
# Helpers (C1 checker: same refinement-scaling algorithm as
# test_smoothmath.py::assert_c1_continuous; see its docstring for rationale)
# --------------------------------------------------------------------------
def _assert_c1_continuous(f, x_lo, x_hi):
    def indicator(n):
        x = np.linspace(x_lo, x_hi, n)
        dx = x[1] - x[0]
        y = f(x)
        d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
        assert np.all(np.isfinite(y))
        return np.max(np.abs(d2)) / dx

    ratio = indicator(4001) / (indicator(2001) + 1e-300)
    assert ratio < 0.75, (
        f"first derivative appears discontinuous (refinement ratio {ratio:.3f})"
    )


def _spanwise(lo, hi, n=7):
    return np.linspace(lo, hi, n)


# --------------------------------------------------------------------------
# Section 3.3 -- energy and work
# --------------------------------------------------------------------------
def test_duct_and_stator_conserve_h0():
    # Section 3.3: stationary rows and ducts, h0 = const.
    h0 = _spanwise(4.0e5, 4.2e5)
    s = _spanwise(10.0, 30.0)
    rvt = _spanwise(20.0, 40.0)
    for step in [TransportStep(),  # duct
                 TransportStep(omega=0.0, rvt=rvt * 0.5, delta_s=2.0)]:  # stator
        h0_out, _, _ = apply_step(h0, s, rvt, step)
        assert np.array_equal(h0_out, h0)


def test_rotor_conserves_rothalpy():
    # Section 3.3: I = h0 - omega * rVt = const across a rotor step.
    omega = 1200.0
    h0 = _spanwise(4.0e5, 4.2e5)
    s = _spanwise(10.0, 30.0)
    rvt_in = _spanwise(5.0, 15.0)
    rvt_out = _spanwise(40.0, 60.0)
    h0_out, _, rvt_2 = apply_step(h0, s, rvt_in,
                                  TransportStep(omega=omega, rvt=rvt_out))
    np.testing.assert_allclose(rothalpy(h0_out, rvt_2, omega),
                               rothalpy(h0, rvt_in, omega), rtol=1e-14)


def test_rotor_euler_work():
    # Section 3.3: Euler work equation, delta h0 = omega * delta(rVt).
    omega = 800.0
    h0, s, rvt_in = 3.0e5, 50.0, 10.0
    rvt_out = 35.0
    h0_out, _, _ = apply_step(h0, s, rvt_in,
                              TransportStep(omega=omega, rvt=rvt_out))
    assert h0_out - h0 == pytest.approx(omega * (rvt_out - rvt_in))


def test_rothalpy_degenerates_to_h0_at_zero_omega():
    # Section 3.3: with omega = 0 rothalpy is h0 itself.
    assert rothalpy(4.0e5, 123.0, 0.0) == 4.0e5


# --------------------------------------------------------------------------
# Section 3.4 -- swirl
# --------------------------------------------------------------------------
def test_duct_conserves_angular_momentum():
    # Section 3.4: ducts, rVt = const per streamtube.
    rvt = _spanwise(20.0, 40.0)
    _, _, rvt_out = apply_step(0.0, 0.0, rvt, TransportStep())
    assert np.array_equal(rvt_out, rvt)


def test_row_exit_rvt_is_closure_value_exactly():
    # Section 3.4: blade-row exit rVt is set by the swirl closure -- the
    # swept TE value must equal the closure target bit-for-bit, including
    # via the row_steps schedule path (work fraction reaches exactly 1).
    rvt_le = _spanwise(5.0, 15.0)
    rvt_te = _spanwise(40.0, 60.0)
    steps = row_steps(omega=500.0, rvt_le=rvt_le, rvt_te=rvt_te,
                      delta_s_row=3.0, t_stations=(0.3, 0.7, 1.0))
    fields = sweep(_spanwise(4.0e5, 4.2e5), 0.0, rvt_le, steps)
    assert np.array_equal(fields.rvt[:, -1], rvt_te)


def test_inblade_rvt_follows_work_schedule():
    # Section 3.4: INBLADE rVt interpolated LE -> TE by the C1 schedule.
    rvt_le, rvt_te = 10.0, 50.0
    sched = SmoothRampSchedule()
    t_stations = (0.25, 0.5, 0.75, 1.0)
    steps = row_steps(omega=0.0, rvt_le=rvt_le, rvt_te=rvt_te,
                      delta_s_row=0.0, t_stations=t_stations)
    fields = sweep(3.0e5, 0.0, rvt_le, steps)
    expect = [(1.0 - sched(t)) * rvt_le + sched(t) * rvt_te
              for t in t_stations]
    np.testing.assert_allclose(fields.rvt[0, 1:], expect, rtol=1e-15)
    # Monotone LE -> TE for a monotone schedule.
    assert np.all(np.diff(fields.rvt[0, :]) >= 0.0)


def test_inblade_rothalpy_conserved_at_every_station():
    # Sections 3.3 + 3.4: the scheduled in-blade march conserves rothalpy
    # station-by-station, not just LE -> TE.
    omega = 900.0
    rvt_le = _spanwise(5.0, 15.0)
    rvt_te = _spanwise(40.0, 60.0)
    h0_le = _spanwise(4.0e5, 4.2e5)
    steps = row_steps(omega=omega, rvt_le=rvt_le, rvt_te=rvt_te,
                      delta_s_row=2.0, t_stations=(0.2, 0.5, 0.8, 1.0))
    fields = sweep(h0_le, 100.0, rvt_le, steps)
    I = rothalpy(fields.h0, fields.rvt, omega)
    np.testing.assert_allclose(I, np.broadcast_to(I[:, :1], I.shape),
                               rtol=1e-13)


# --------------------------------------------------------------------------
# Section 3.4 / 7.3 -- schedule contract (C1, endpoints, monotone)
# --------------------------------------------------------------------------
def test_schedule_endpoints_exact():
    # Section 3.4 contract: f(0) = 0 and f(1) = 1 exactly (TE exactness of
    # the closure value depends on this).
    sched = SmoothRampSchedule()
    assert sched(0.0) == 0.0
    assert sched(1.0) == 1.0


def test_schedule_saturates_outside_row():
    # Schedules must saturate, never extrapolate (section 7.3 discipline).
    sched = SmoothRampSchedule()
    assert sched(-0.5) == 0.0
    assert sched(1.5) == 1.0


def test_schedule_monotone():
    sched = SmoothRampSchedule()
    t = np.linspace(-0.2, 1.2, 500)
    assert np.all(np.diff(sched(t)) >= -1e-15)


def test_schedule_c1_including_le_te_joins():
    # Section 3.4: "The schedule must be C1 in m" -- checked by refinement
    # scaling across the full composite domain including the constant
    # regions upstream/downstream (zero end slopes required).
    _assert_c1_continuous(SmoothRampSchedule(), -0.5, 1.5)


def test_c1_checker_rejects_kinked_schedule():
    # Negative control (CLAUDE.md process rule): a C0 linear-clip ramp has
    # derivative jumps at t = 0, 1 and must FAIL the checker.
    def kinked_ramp(t):
        return np.clip(t, 0.0, 1.0)

    with pytest.raises(AssertionError):
        _assert_c1_continuous(kinked_ramp, -0.5, 1.5)


# --------------------------------------------------------------------------
# Section 3.5 / B.5.1 -- entropy
# --------------------------------------------------------------------------
def test_entropy_accumulates_row_and_mix_increments():
    # Section 3.5: s_{j+1} = s_j + delta_s_row + delta_s_mix (mixing enters
    # additively through the same delta_s slot; section 3.6 deferred).
    s = _spanwise(10.0, 30.0)
    ds = _spanwise(0.5, 1.5) + 0.1  # row + mix, summed by the caller
    _, s_out, _ = apply_step(0.0, s, 0.0, TransportStep(delta_s=ds))
    np.testing.assert_allclose(s_out, s + ds, rtol=1e-15)


def test_inblade_entropy_distribution_sums_to_row_total():
    # B.5.1: distributed in-blade delta_s increments telescope to the row
    # total delta_s_row at the TE.
    ds_row = _spanwise(0.8, 2.0)
    steps = row_steps(omega=0.0, rvt_le=0.0, rvt_te=10.0, delta_s_row=ds_row,
                      t_stations=(0.25, 0.5, 0.75, 1.0))
    total = sum(st.delta_s for st in steps)
    np.testing.assert_allclose(total, ds_row, rtol=1e-14)


def test_loss_distribution_defaults_to_work_schedule():
    # B.5.1: the loss schedule defaults to the work schedule itself.
    sched = SmoothRampSchedule()
    t_stations = (0.3, 0.6, 1.0)
    steps = row_steps(omega=0.0, rvt_le=0.0, rvt_te=1.0, delta_s_row=1.0,
                      t_stations=t_stations)
    cum = np.cumsum([st.delta_s for st in steps])
    np.testing.assert_allclose(cum, [sched(t) for t in t_stations],
                               rtol=1e-14)


def test_local_loss_schedule_override():
    # B.5.1: explicitly local loss sources may use their own schedule.
    def front_loaded(t, *, xp=None):
        return SmoothRampSchedule()(2.0 * t, xp=xp)
    steps = row_steps(omega=0.0, rvt_le=0.0, rvt_te=1.0, delta_s_row=1.0,
                      t_stations=(0.5, 1.0), loss_schedule=front_loaded)
    assert steps[0].delta_s == pytest.approx(1.0)  # all loss by mid-chord
    assert steps[1].delta_s == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Sweep: shapes, degeneracy, purity, xp injection
# --------------------------------------------------------------------------
def test_sweep_shapes_and_column_meaning():
    # AD-2: (n_sl, n_qo) struct-of-arrays output, column j = station j.
    n_sl = 5
    h0 = np.full(n_sl, 3.0e5)
    steps = [TransportStep(), TransportStep(omega=100.0, rvt=np.ones(n_sl)),
             TransportStep()]
    fields = sweep(h0, np.zeros(n_sl), np.zeros(n_sl), steps)
    for arr in (fields.h0, fields.s, fields.rvt):
        assert arr.shape == (n_sl, 4)
    np.testing.assert_array_equal(fields.h0[:, 0], h0)


def test_all_duct_sweep_is_constant():
    # Sections 3.3-3.5 duct rules jointly: a bladeless adiabatic sweep
    # transports every field unchanged (also the Tier-degeneracy limit of
    # the universal step, section 8 / AD-1).
    h0 = _spanwise(4.0e5, 4.2e5)
    s = _spanwise(10.0, 30.0)
    rvt = _spanwise(20.0, 40.0)
    fields = sweep(h0, s, rvt, [TransportStep()] * 4)
    for arr, inlet in ((fields.h0, h0), (fields.s, s), (fields.rvt, rvt)):
        assert np.array_equal(arr, inlet[:, None] * np.ones((1, 5)))


def test_sweep_broadcasts_scalar_inlet():
    # Scalar inlet against a per-streamline closure target must broadcast.
    fields = sweep(3.0e5, 0.0, 0.0,
                   [TransportStep(), TransportStep(omega=10.0,
                                                   rvt=_spanwise(1.0, 2.0, 4))])
    assert fields.rvt.shape == (4, 3)
    np.testing.assert_array_equal(fields.rvt[:, 0], np.zeros(4))


def test_purity_inputs_not_mutated():
    # AD-3/AD-6: transport never mutates its inputs.
    h0 = _spanwise(4.0e5, 4.2e5)
    s = _spanwise(10.0, 30.0)
    rvt = _spanwise(20.0, 40.0)
    copies = (h0.copy(), s.copy(), rvt.copy())
    steps = row_steps(omega=700.0, rvt_le=rvt, rvt_te=2.0 * rvt,
                      delta_s_row=_spanwise(0.1, 0.3), t_stations=(0.5, 1.0))
    sweep(h0, s, rvt, steps)
    for arr, ref in zip((h0, s, rvt), copies):
        assert np.array_equal(arr, ref)


def test_explicit_xp_injection_matches_default():
    # AD-6 namespace injection path.
    h0, s, rvt = 3.0e5, 0.0, _spanwise(1.0, 2.0)
    steps = [TransportStep(omega=5.0, rvt=2.0 * rvt, delta_s=0.5)]
    a = sweep(h0, s, rvt, steps, xp=np)
    b = sweep(h0, s, rvt, steps)
    for x, y in ((a.h0, b.h0), (a.s, b.s), (a.rvt, b.rvt)):
        assert np.array_equal(x, y)


# --------------------------------------------------------------------------
# Config-boundary validation (AD-10: raise loudly and early)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("t_stations", [
    (),                 # empty
    (0.5,),             # does not end at 1.0
    (0.5, 0.5, 1.0),    # not strictly increasing
    (0.0, 1.0),         # t = 0 is the LE itself, not a produced station
    (-0.2, 1.0),        # out of range
])
def test_row_steps_rejects_bad_t_stations(t_stations):
    with pytest.raises(ConfigError):
        row_steps(omega=0.0, rvt_le=0.0, rvt_te=1.0, delta_s_row=0.0,
                  t_stations=t_stations)
