"""Centrifugal parasitic-loss reference tests (section 4.4 gate-#3
components; Aungier 2000 ch. 4 verbatim — docs/references/CENT-LOSS.md
"parasitic" section, closures/centrifugal/parasitic.py).

Coefficient/chain pins with hand-computed references, plus the Eckardt O
measured-agreement integration (docs/references/ECKARDT.md).
"""
import pytest

from slcflow.closures.centrifugal.parasitic import (
    disk_friction_work, leakage_work, recirculation_work)


def test_disk_friction_daily_nece_regime_and_factor():
    # Hand reference: rho=1.8, u2=293.2, r2=0.2, mu=1.81e-5 -> Re=5.83e6,
    # gap 0.02 (regime III narrowly wins over IV there — the largest-C_M
    # selection rule, Eqs 4-22..25); x0.75 experience factor (Eq 4-31);
    # dh = C_M rho r2^2 u2^3 / (2 mdot) (Eq 4-21 chain).
    rho, u2, r2, mu, md = 1.8, 293.2, 0.2, 1.81e-5, 5.31
    re = rho * u2 * r2 / mu
    cms = [(2.0 * 3.141592653589793) / (0.02 * re),
           3.7 * 0.02 ** 0.1 / re ** 0.5,
           0.08 / (0.02 ** (1.0 / 6.0) * re ** 0.25),
           0.102 * 0.02 ** 0.1 / re ** 0.2]
    assert max(cms) == cms[2]          # regime III at this point
    want = 0.75 * max(cms) * rho * r2 ** 2 * u2 ** 3 / (2.0 * md)
    got = disk_friction_work(md, rho, u2, r2, mu=mu, gap_ratio=0.02)
    assert got == pytest.approx(want, rel=1e-12)
    # Largest-C_M selection rule at low Re (whichever regime wins):
    re_t = rho * 0.5 * 0.01 / mu
    cms = [(2.0 * 3.141592653589793) / (0.02 * re_t),
           3.7 * 0.02 ** 0.1 / re_t ** 0.5,
           0.08 / (0.02 ** (1.0 / 6.0) * re_t ** 0.25),
           0.102 * 0.02 ** 0.1 / re_t ** 0.2]
    want_t = 0.75 * max(cms) * rho * 0.01 ** 2 * 0.5 ** 3 / (2 * md)
    tiny = disk_friction_work(md, rho, 0.5, 0.01, mu=mu, gap_ratio=0.02)
    assert tiny == pytest.approx(want_t, rel=1e-9)


def test_leakage_chain_eq_4_17_to_4_40():
    # dp = mdot (r2 cu2 - r1 cu1)/(z rbar bbar L); U_CL = 0.816
    # sqrt(2 dp/rho2); mdot_CL = rho2 z s L U_CL; dh = mdot_CL U_CL u2/(2 mdot).
    md, rho2, u2 = 5.31, 1.9, 293.2
    r1, r2, b1, b2 = 0.0925, 0.2, 0.095, 0.026
    cu1, cu2, z, s, L = 0.0, 200.0, 20, 7e-4, 0.18
    dp = md * (r2 * cu2 - r1 * cu1) / (z * 0.5 * (r1 + r2)
                                       * 0.5 * (b1 + b2) * L)
    ucl = 0.816 * (2.0 * dp / rho2) ** 0.5
    want = rho2 * z * s * L * ucl * ucl * u2 / (2.0 * md)
    got = leakage_work(md, rho2, u2, r1, r2, b1, b2, cu1, cu2, z, s, L)
    assert got == pytest.approx(want, rel=1e-12)
    # Zero/negative loading -> no leakage work (guarded, not NaN):
    assert leakage_work(md, rho2, u2, r1, r2, b1, b2, 0.0, 0.0, z, s, L) \
        == 0.0


def test_recirculation_floor_and_formula():
    # I_R = (D_eq/2 - 1)(W_U2/C_m2 - 2 cot(beta2b)) >= 0 (Eq 4-43), D_eq
    # from W_max = (W1+W2+dW)/2, dW = 4 pi (r2cu2 - r1cu1)/(z L) (4-41/42).
    # Low diffusion -> floored to zero:
    assert recirculation_work(300.0, 200.0, 190.0, 100.0, 80.0, 0.0,
                              0.0, 5.0, 20, 0.18) == 0.0
    # Genuinely high diffusion (D_eq/2 > 1) -> the printed formula, exactly:
    u2, w1, w2, cm2, wu2 = 366.0, 250.0, 60.0, 100.0, 120.0
    r1cu1, r2cu2, z, L = 0.0, 14.0, 20, 0.18
    dw = 4.0 * 3.141592653589793 * (r2cu2 - r1cu1) / (z * L)
    deq = (w1 + w2 + dw) / (2.0 * w2)
    want = (deq / 2.0 - 1.0) * (wu2 / cm2) * u2 ** 2
    got = recirculation_work(u2, w1, w2, cm2, wu2, 0.0, r1cu1, r2cu2, z, L)
    assert got == pytest.approx(want, rel=1e-12)


def test_vaneless_diffuser_closed_form():
    # Coppage/Stanitz via Whitfield & Baines Eq [30] (CENT-LOSS.md):
    # delta_q = cf r_x (1-(r_x/r_y)^1.5)(C_x/U_T)^2 / (1.5 b_x cos a_x),
    # returned as delta_q * U_T^2; alpha from tangent (cos = Cu/C).
    from slcflow.closures.centrifugal.parasitic import vaneless_diffuser_loss
    cf, r_in, r_out, b, u = 0.005, 0.2, 0.4, 0.026, 293.2
    cu, cm = 257.0, 85.0
    c = (cu * cu + cm * cm) ** 0.5
    dq = cf * r_in * (1.0 - 0.5 ** 1.5) * (c / u) ** 2 \
        / (1.5 * b * (cu / c))
    want = dq * u ** 2
    got = vaneless_diffuser_loss(cf, r_in, r_out, b, c, cu, u)
    assert got == pytest.approx(want, rel=1e-12)
    # No swirl -> negligible-path guard returns 0:
    assert vaneless_diffuser_loss(cf, r_in, r_out, b, 100.0, 0.0, u) == 0.0


def test_eckardt_stage_performance_with_diffuser():
    # The full stage chain at the R/R2 = 2 measurement plane (ECKARDT.md):
    # internal 0.969 -> +parasitics 0.9265 -> +vaneless diffuser 0.9074 vs
    # measured stage 0.88 (+2.7 pt remaining: lambda tip-distortion, the
    # closed-form-vs-marching diffuser, cf level — recorded). PR_stage
    # 2.167 vs measured 2.1 (+3.2%, from +4.7% at the impeller exit).
    from slcflow.verification.v7_eckardt import EckardtO
    case = EckardtO()
    r = case.evaluate(n_sl=1)
    sp = case.stage_performance(r)
    assert sp["dh_vld"] == pytest.approx(1356.0, abs=150.0)
    assert sp["pr_stage"] == pytest.approx(2.167, abs=0.03)
    assert sp["eta_stage"] == pytest.approx(0.9074, abs=0.01)
    assert sp["eta_stage"] > 0.88          # the remaining gap stays positive
    assert sp["pr_stage"] < r.pressure_ratio


def test_eckardt_stage_efficiency_with_parasitics():
    # Integration (gate #3, ECKARDT.md): at the laser point the parasitic
    # set debits eta 0.969 -> 0.9265 (DF 370 + leakage 765 + recirculation
    # 2327 J/kg ~ 4.4% of work); the remaining ~4.6 pt to the measured
    # STAGE 0.88 is the unmodelled R/R2 = 2 vaneless diffuser (recorded).
    # At the 18000-rpm design point recirculation grows with loading and
    # eta-with-parasitics reads 0.877.
    from slcflow.verification.v7_eckardt import DESIGN_POINT, EckardtO
    case = EckardtO()
    r = case.evaluate(n_sl=1)
    par = case.parasitic_breakdown(r)
    assert 200.0 < par["disk_friction"] < 600.0
    assert 400.0 < par["leakage"] < 1200.0
    assert 1500.0 < par["recirculation"] < 3500.0
    eta = case.stage_efficiency(r)
    assert eta == pytest.approx(0.9265, abs=0.015)
    assert eta > 0.88            # the diffuser gap stays positive
    c18 = EckardtO(rpm=DESIGN_POINT["rpm"], mdot=DESIGN_POINT["mdot"])
    r18 = c18.evaluate(n_sl=1)
    assert c18.stage_efficiency(r18) == pytest.approx(0.877, abs=0.02)
