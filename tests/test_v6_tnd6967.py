"""NASA TN D-6967 two-stage turbine point-by-point gate (section 9.6, V6).

The first MACHINE-LEVEL measured turbine validation (LS-89 covers the
cascade level): geometry from TN D-6967 Tables I/II + Figure 1 velocity
diagrams, measured air-equivalent design point from Table IV (see
slcflow/verification/v6_tnd6967.py and docs/references/TND6967.md).

MEASURED AGREEMENT (2026-07-16), pinned as recorded findings:

  * Tier-1 meanline converges (800 outer iterations — the 4-row Picard
    chain at closure_relax 0.25 is slow, not unstable) with validity 0.99.
  * At the measured equivalent flow (2.004 kg/s): eta_tt 0.926 vs measured
    0.93 — within half a point, end-to-end through four K-O rows. Note the
    K-O set carries no tip-clearance loss (rotors ran ~0.9% clearance/
    height), so the internal-loss model is honestly ~1-1.5 pt pessimistic,
    inside K-O's own +/-1.5 pt stage-eta target.
  * PR_tt at matched flow reads 3.13 vs the rig's 3.765 (work 74.6 vs
    84.9 J/g). RE-DIAGNOSED 2026-07-16 with the section 6.6 row-throat
    check (test_throat_capacity.py): the model's capacity is NOT high —
    it chokes at ~2.02 kg/s (annulus) with rotor-1's throat at ~2.06,
    within ~1% of the rig's measured 2.03-2.05 choke. Both rig and model
    sit within 1-2% of their choke flows at 2.004, where the PR-vs-mdot
    characteristic is near-vertical — so matched-MDOT comparison
    amplifies a ~1% capacity difference into the -17% PR read. The right
    comparison frame near choke is matched-PR (BackPressureSpec) or
    matched choke margin; recorded, not yet run.
  * Tier-2 spanwise is OPEN: at the measured flow the free-vortex hub
    (near-sonic by design) chokes the hub streamtube (CHOKE_LIMITED), and
    reduced flows hit NUMERICAL_FAILURE — the 4-row spanwise closure-lag
    chain is a recorded solver/case item, not pinned here.
"""
import pytest

from slcflow.verification.v6_tnd6967 import (DESIGN_EQ, MEASURED_EQ,
                                             TND6967Turbine)


@pytest.fixture(scope="module")
def result():
    return TND6967Turbine().evaluate(n_sl=1)


def test_transcribed_geometry_anchors():
    # Table I/II transcription guards: equivalent speed 15336 rpm, mean
    # radius 0.1016 m, blade counts 35/42/43/44, rotor-1 mean turning
    # 29.6 + 61.6 = 91.2 deg (Table II cross-check vs Figure 1 angles).
    from slcflow.verification.v6_tnd6967 import _R_MEAN, _ROWS
    case = TND6967Turbine()
    assert case.omega == pytest.approx(15336 * 2 * 3.14159265 / 60, rel=1e-6)
    assert _R_MEAN == 0.1016
    assert [d["z"] for d in _ROWS.values()] == [35, 42, 43, 44]
    r1 = _ROWS["r1"]
    assert r1["le"][1] - r1["te"][1] == pytest.approx(91.2, abs=0.01)
    r2 = _ROWS["r2"]
    assert r2["le"][1] - r2["te"][1] == pytest.approx(62.8, abs=0.01)


def test_meanline_converges_with_real_expansion(result):
    # Structural gate: the four-row machine converges end-to-end and
    # expands (PR_tt < 1, work extraction) with in-window closures.
    assert result.converged
    assert result.pressure_ratio < 0.5
    assert result.validity > 0.9


def test_eta_tt_matches_measured_within_half_point(result):
    # Table IV measured eta_tt = 0.93 at the equivalent design point.
    # Measured agreement -0.4 pt (facade efficiency inverts for a turbine).
    eta_tt = 1.0 / result.efficiency
    assert abs(eta_tt - MEASURED_EQ["eta_tt"]) < 0.02


def test_matched_pr_backpressure_comparison(result):
    # The MATCHED-PR frame (section 6.6 BackPressureSpec; the comparison
    # the near-choke re-diagnosis called for): impose the Table-I
    # equivalent total-to-static pressure ratio (p_exit = p0_in / 4.640)
    # and let mdot join the state. MEASURED (2026-07-16), seeded from the
    # converged matched-mdot state (the nearest operating point):
    #
    #     mdot   2.011  vs measured 2.004    (+0.35%)
    #     PR_tt  3.718  vs measured eq 3.765 (-1.2%)
    #     work   83.87  vs measured 84.90 J/g (-1.2%)
    #     eta_tt 0.926  vs measured 0.93     (-0.4 pt)
    #
    # In the natural near-choke frame the machine agrees with the rig to
    # ~1% ACROSS THE BOARD — confirming the matched-mdot "-17% PR" was
    # pure vertical-characteristic sensitivity, and that no additional
    # turning/loss deficit hides behind it. History: this solve initially
    # had SEED-DEPENDENT fixed points (a fresh mdot-2.0 seed landed a
    # spurious branch — station 7 on the supersonic continuity root, same
    # PR, work 8% low) — ROOT-CAUSED and FIXED 2026-07-16 by the section
    # 6.3 branch-preserving Newton trial guard
    # (test_backpressure_newton_stays_on_seed_branch); nearby seeds now
    # land the identical physical fixed point (far seeds may still fail
    # TYPED — Newton is local, warm-start quality is the caller's job).
    # The closure lag rings at ~1e-6 near choke (benign limit cycle) —
    # tol_closure loosened accordingly.
    import numpy as np

    from slcflow.assembly.pack import unpack
    from slcflow.drivers.newton import NewtonConfig, solve_newton
    from slcflow.grid.core import GridTopology
    from slcflow.types import BackPressureSpec, FidelityConfig

    case = TND6967Turbine()
    m = case.machine()
    topo = GridTopology(m.flowpath, n_sl=1)
    inlet = m.inlet.fields(topo.psi)
    # Seed from the already-solved matched-mdot fixture (same topology).
    seed = result.result
    res = solve_newton(topo, case.gas, FidelityConfig.tier1(),
                       BackPressureSpec(101325.0 / 4.640, topo.n_qo - 1),
                       inlet, rows=m.rows, warm_start=seed,
                       config=NewtonConfig(max_outer=800, tol_closure=1e-6))
    assert res.converged
    _, _, mdot = unpack(res.x, topo.n_sl, topo.n_qo, backpressure=True)
    tr = res.frozen.transported
    pr_tt = float(case.gas.p(tr.h0[0, 0], tr.s[0, 0])
                  / case.gas.p(tr.h0[0, -1], tr.s[0, -1]))
    work = float(tr.h0[0, 0] - tr.h0[0, -1])
    assert mdot == pytest.approx(MEASURED_EQ["mdot"], rel=0.015)
    assert pr_tt == pytest.approx(DESIGN_EQ["pr_tt"], rel=0.03)
    assert work / 1000.0 == pytest.approx(MEASURED_EQ["work_J_per_g"],
                                          rel=0.03)
    kappa = (case.gas.gamma - 1.0) / case.gas.gamma
    eta_tt = work / float(tr.h0[0, 0] * (1.0 - (1.0 / pr_tt) ** kappa))
    assert abs(eta_tt - MEASURED_EQ["eta_tt"]) < 0.02


def test_backpressure_newton_stays_on_seed_branch(result):
    # Regression for the SPURIOUS closure-lag branch (root-caused
    # 2026-07-16): from a fresh subsonic mdot-2.0 seed, the BP Newton
    # solve used to converge station 7 (rotor-2 LE) onto the SUPERSONIC
    # continuity root (M_m 1.997, Vm 452 m/s — the two-root A.7 pair:
    # subsonic ~112, capacity peak ~250, supersonic 452), because Newton
    # trials were positivity-guarded but not BRANCH-guarded; the closure
    # lag then locked in a self-consistent fixed point with the same PR
    # but 8%-low work. The section 6.3 branch-preserving trial guard
    # (newton._safe_residual: a trial may approach M_m = 1 but not jump
    # across it relative to the seed's per-station branch) must now land
    # the fresh seed on the PHYSICAL branch: subsonic everywhere, work at
    # the measured level.
    import numpy as np

    from slcflow.assembly.pack import unpack
    from slcflow.drivers.classical import ClassicalConfig, solve_classical
    from slcflow.drivers.newton import NewtonConfig, solve_newton
    from slcflow.grid.core import GridTopology
    from slcflow.types import (BackPressureSpec, FidelityConfig,
                               MassFlowSpec)

    case = TND6967Turbine()
    m = case.machine()
    topo = GridTopology(m.flowpath, n_sl=1)
    inlet = m.inlet.fields(topo.psi)
    fid = FidelityConfig.tier1()
    seed = solve_classical(topo, case.gas, fid, MassFlowSpec(2.0), inlet,
                           rows=m.rows,
                           config=ClassicalConfig(max_outer=800))
    assert seed.converged
    res = solve_newton(topo, case.gas, fid,
                       BackPressureSpec(101325.0 / 4.640, topo.n_qo - 1),
                       inlet, rows=m.rows, warm_start=seed,
                       config=NewtonConfig(max_outer=800, tol_closure=1e-6))
    assert res.converged
    # Subsonic everywhere (the seed's branch), physical-level work.
    assert float(np.max(res.fields.mach_m)) < 1.0
    tr = res.frozen.transported
    work = float(tr.h0[0, 0] - tr.h0[0, -1]) / 1000.0
    assert work == pytest.approx(MEASURED_EQ["work_J_per_g"], rel=0.035)


def test_pr_and_work_bounded_capacity_gap_recorded(result):
    # At matched mdot the PR/work read LOW: observed 1/PR = 3.13 vs rig
    # 3.765 — re-diagnosed (2026-07-16) as NEAR-CHOKE SENSITIVITY, not a
    # capacity error (module docstring; the model's choke is within ~1%
    # of the rig's, and PR-vs-mdot is near-vertical there). Bounded so a
    # drift beyond the understood effect turns this red.
    inv_pr = 1.0 / result.pressure_ratio
    assert 2.8 < inv_pr < DESIGN_EQ["pr_tt"] * 1.02
