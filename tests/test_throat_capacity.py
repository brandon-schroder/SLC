"""Blade-row throat sonic capacity + the section 6.6 row-throat check.

``row_throat_capacity`` (drivers/classical.py) is the blade-passage sibling
of the A.7 annulus ``qo_capacity``: 1-D sonic capacity through Z*o(y)*dq in
the blade-relative frame (B.1 rothalpy re-referencing to the mid-passage
radius). The classical driver applies it POST-SOLVE to every row whose
geometry provides a throat (section 4.1 contract) and declares
CHOKE_LIMITED when mdot exceeds it — a converged annulus solution passing
more flow than a passage throat can swallow is not an operating point of
the real machine.

MEASURED anchors (2026-07-16, docs/references/TND6967.md / ROTOR37.md):

  * TN D-6967 (real throats from the design gauging): per-row capacities
    s1 2.189 / r1 2.062 / s2 2.138 / r2 2.195 kg/s eq; the machine chokes
    at min(annulus ~2.02, rotor-1 throat 2.06) ~ 2.02 vs the rig's
    measured 2.03-2.05 — within ~1% at cd = 1. (This SUPERSEDES the
    2026-07-16 morning "capacity ~2.19 vs rig 2.03" reading, which was a
    hand stator-gauge estimate, not a driver measurement.)
  * Rotor 37 (gauging-estimate throat o = s cos(KIC)): the rotor-relative
    throat capacity is ~24.6 kg/s > the ~22.25 annulus limit — a
    supersonic-inlet rotor chokes at its inlet swallowing limit, NOT the
    internal throat; the check is correctly inert and the case pins stand.
"""
import numpy as np
import pytest

from slcflow.drivers.classical import row_throat_capacity
from slcflow.fluid.perfectgas import PerfectGas


GAS = PerfectGas()


def _analytic_capacity(T0, p0, o, L, Z, cd=1.0):
    # Stationary row, uniform state: cd * Z * rho* a* * o * L with the
    # perfect-gas sonic state on the s=0 isentrope (section 3.7).
    g, R = GAS.gamma, GAS.R
    cp = g * R / (g - 1.0)
    T_st = T0 * 2.0 / (g + 1.0)
    p_st = p0 * (T_st / T0) ** (g / (g - 1.0))
    rho_st = p_st / (R * T_st)
    a_st = np.sqrt(g * R * T_st)
    return cd * Z * rho_st * a_st * o * L


def test_matches_analytic_stationary_uniform():
    # (section 6.6) exact against the closed-form 1-D sonic capacity.
    cp = GAS.gamma * GAS.R / (GAS.gamma - 1.0)
    h0 = np.full(5, cp * 288.15)
    s = np.zeros(5)
    q = np.linspace(0.0, 0.1, 5)             # nodes spanning the walls
    got = row_throat_capacity(GAS, h0, s, np.zeros(5), 0.0,
                              np.full(5, 0.5), np.full(5, 0.5),
                              q, (0.0, 0.1), np.full(5, 0.02), 30)
    want = _analytic_capacity(288.15, 101325.0, 0.02, 0.1, 30)
    assert got == pytest.approx(want, rel=1e-12)


def test_meanline_single_node_owns_full_span():
    # Tier-1 degenerate grid (n_sl = 1): the single node's ownership width
    # is the whole wall-to-wall span, not zero (the one-point area rule).
    cp = GAS.gamma * GAS.R / (GAS.gamma - 1.0)
    got = row_throat_capacity(GAS, [cp * 288.15], [0.0], [0.0], 0.0,
                              [0.5], [0.5], [0.05], (0.0, 0.1), [0.02], 30)
    want = _analytic_capacity(288.15, 101325.0, 0.02, 0.1, 30)
    assert got == pytest.approx(want, rel=1e-12)


def test_rothalpy_referencing_signs():
    # B.1 frame effects: co-rotating inlet swirl LOWERS the relative
    # stagnation state (turbine rotor: I = h0 - omega*rvt), cutting
    # capacity; wheel speed raises it back through +U^2/2 at the throat
    # radius (section 6.6).
    cp = GAS.gamma * GAS.R / (GAS.gamma - 1.0)
    base = dict(fluid=GAS, h0=[cp * 400.0], s=[0.0], omega=1000.0,
                r_le=[0.2], r_te=[0.2], q_nodes=[0.05],
                q_walls=(0.0, 0.1), throat_o=[0.02], blade_count=30)
    no_swirl = row_throat_capacity(rvt=[0.0], **base)
    co_swirl = row_throat_capacity(rvt=[20.0], **base)
    assert co_swirl < no_swirl
    slow = {**base, "omega": 100.0}
    assert row_throat_capacity(rvt=[0.0], **slow) < no_swirl


def test_cd_scales_linearly():
    cp = GAS.gamma * GAS.R / (GAS.gamma - 1.0)
    args = (GAS, [cp * 288.15], [0.0], [0.0], 0.0, [0.5], [0.5],
            [0.05], (0.0, 0.1), [0.02], 30)
    assert row_throat_capacity(*args, cd=0.95) == pytest.approx(
        0.95 * row_throat_capacity(*args), rel=1e-12)


def test_tnd6967_chokes_at_measured_flow():
    # Integration (section 6.6): the two-stage turbine converges at the
    # measured 2.004 kg/s eq and is choke-limited by ~2.05 — the rig's
    # measured choke is 2.03-2.05, so the model's capacity (annulus ~2.02,
    # rotor-1 throat ~2.06 at cd = 1) is within ~1%.
    from slcflow.verification.v6_tnd6967 import TND6967Turbine
    case = TND6967Turbine()
    ok = case.evaluate(n_sl=1)
    assert ok.converged
    choked = case.evaluate(n_sl=1, mdot=2.05)
    assert not choked.converged


def test_rotor37_gauging_throat_is_inert():
    # The supersonic-inlet compressor finding: the relative-frame throat
    # capacity (~24.6 kg/s, gauging estimate) exceeds the annulus
    # swallowing limit (~22.25), so providing the throat does NOT change
    # the case's operating points (pins stand; ROTOR37.md).
    from slcflow.verification.v5_rotor37 import Rotor37
    r = Rotor37().evaluate(n_sl=1)
    assert r.converged
    assert r.pressure_ratio == pytest.approx(2.135, abs=0.09)
