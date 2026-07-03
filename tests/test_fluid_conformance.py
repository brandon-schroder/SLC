"""Backend-agnostic conformance suite for the WorkingFluid contract (ARCH-4.1).

Any backend -- PerfectGas today, a real-gas/CoolProp backend later -- must pass
these tests unchanged: they verify *thermodynamic identities*, not
implementation details. Register new backends in ``BACKENDS`` below (or reuse
``FluidConformance`` from another test module).

Identities tested
-----------------
1. Round-trip closure:      (T,p) -> (h,s) -> (T,p)
2. Gibbs relation:          (dh/ds)|_p = T
3. Speed of sound:          a^2 = (dp/drho)|_s
4. Stagnation isentropy:    s0 = s and h0 = h + V^2/2
5. Vectorization/broadcast and purity (no input mutation)

Derivative identities are checked by *central finite differences using only
Protocol methods* (with a scalar root-find where an inverse is not part of the
contract), so the suite never assumes an analytic backend.

Property-style testing is done on seeded random state samples for
reproducibility; migrating to ``hypothesis`` for adversarial case generation
is a recorded upgrade path (ARCH-7).
"""
import numpy as np
import pytest
from scipy.optimize import brentq

from slcflow.fluid import PerfectGas

# --- registered backends and their valid sampling boxes --------------------
BACKENDS = [
    pytest.param(
        PerfectGas(),
        {"T": (210.0, 1900.0), "p": (2e4, 4e6)},  # gas-turbine-relevant box
        id="PerfectGas-air",
    ),
    pytest.param(
        PerfectGas(gamma=1.33, R=287.0),
        {"T": (400.0, 2000.0), "p": (5e4, 4e6)},
        id="PerfectGas-combustion",
    ),
]

RNG_SEED = 20260703  # fixed: these are regressions, not fuzzing
N_SAMPLES = 40


def _sample_states(box, n=N_SAMPLES):
    rng = np.random.default_rng(RNG_SEED)
    T = rng.uniform(*box["T"], size=n)
    p = np.exp(rng.uniform(np.log(box["p"][0]), np.log(box["p"][1]), size=n))
    return T, p


@pytest.mark.parametrize("fluid,box", BACKENDS)
class TestFluidConformance:
    # ---------------------------------------------------------------- 1
    def test_roundtrip_Tp_hs_Tp(self, fluid, box):
        T, p = _sample_states(box)
        h = fluid.h_from_Tp(T, p)
        s = fluid.s_from_Tp(T, p)
        assert np.allclose(fluid.T(h, s), T, rtol=1e-9)
        assert np.allclose(fluid.p(h, s), p, rtol=1e-9)

    # ---------------------------------------------------------------- 2
    def test_gibbs_relation_dh_ds_at_const_p(self, fluid, box):
        """(dh/ds)|_p = T, using only Protocol methods.

        At fixed p0, perturb s and recover the h that keeps p(h, s+ds) = p0 by
        a bracketed root-find; central difference in s then estimates
        (dh/ds)|_p.
        """
        T, p = _sample_states(box, n=8)  # root-finds: keep sample count modest
        for Ti, pi in zip(T, p):
            h0 = float(fluid.h_from_Tp(Ti, pi))
            s0 = float(fluid.s_from_Tp(Ti, pi))
            ds = max(abs(s0), 1.0) * 1e-6

            def h_at(s_val):
                f = lambda h: float(fluid.p(h, s_val)) - pi
                # bracket around h0 generously; p is monotone in h at fixed s
                return brentq(f, 0.2 * h0, 5.0 * h0, xtol=1e-10 * h0)

            dh_ds = (h_at(s0 + ds) - h_at(s0 - ds)) / (2 * ds)
            assert dh_ds == pytest.approx(Ti, rel=1e-5)

    # ---------------------------------------------------------------- 3
    def test_sound_speed_is_isentropic_dp_drho(self, fluid, box):
        """a^2 = (dp/drho)|_s by central differences along an isentrope."""
        T, p = _sample_states(box, n=12)
        h = fluid.h_from_Tp(T, p)
        s = fluid.s_from_Tp(T, p)
        dh = np.maximum(np.abs(h), 1.0) * 1e-6
        dp = fluid.p(h + dh, s) - fluid.p(h - dh, s)
        drho = fluid.rho(h + dh, s) - fluid.rho(h - dh, s)
        a2_fd = dp / drho
        assert np.allclose(np.sqrt(a2_fd), fluid.a(h, s), rtol=1e-5)

    # ---------------------------------------------------------------- 4
    def test_stagnation_isentropic_and_energy_consistent(self, fluid, box):
        T, p = _sample_states(box, n=12)
        h = fluid.h_from_Tp(T, p)
        s = fluid.s_from_Tp(T, p)
        V = 0.6 * fluid.a(h, s)  # representative M = 0.6
        stag = fluid.stag_from_static(h, s, V)
        assert np.allclose(stag.s, s)                       # isentropic
        assert np.allclose(stag.h0, h + 0.5 * V * V)        # energy
        assert np.allclose(fluid.T(stag.h0, stag.s), stag.T0)
        assert np.allclose(fluid.p(stag.h0, stag.s), stag.p0, rtol=1e-9)
        # inverse
        assert np.allclose(fluid.static_h_from_stag(stag.h0, V), h)

    # ---------------------------------------------------------------- 5
    def test_broadcast_and_purity(self, fluid, box):
        T, p = _sample_states(box, n=6)
        Tc, pc = T.copy(), p.copy()
        h = fluid.h_from_Tp(T[:, None], p[None, :])
        s = fluid.s_from_Tp(T[:, None], p[None, :])
        assert h.shape == s.shape == (6, 6)
        assert fluid.rho(h, s).shape == (6, 6)
        assert np.array_equal(T, Tc) and np.array_equal(p, pc)

    def test_physical_positivity(self, fluid, box):
        T, p = _sample_states(box)
        h = fluid.h_from_Tp(T, p)
        s = fluid.s_from_Tp(T, p)
        for q in (fluid.rho(h, s), fluid.T(h, s), fluid.p(h, s), fluid.a(h, s)):
            assert np.all(np.asarray(q) > 0)
            assert np.all(np.isfinite(np.asarray(q)))