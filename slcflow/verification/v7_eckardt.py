"""V7 — Eckardt rotor O, geometry-faithful endpoints (Theory Manual section
9.7; the point-by-point V7 validation case, ``model-readiness`` gate #1).

The DFVLR radial-discharge laser-anemometry impeller (Eckardt 1976), the
canonical centrifugal validation article. Upgrades the first-order
``tools/eckardt_anchor.py`` (concentric-bend approximation, assumed hub/width)
to the PRIMARY-PAPER geometry.

Provenance — Eckardt, "Detailed Flow Investigations Within a High-Speed
Centrifugal Compressor Impeller", ASME J. Fluids Eng. 98 (1976); extracted
verbatim from the paper via the "Turbomachinery: Test Cases" NotebookLM
notebook (2026-07-15; see docs/references/ECKARDT.md):

  * D2 = 400 mm (r2 = 0.200 m), **20 radially-ending blades** (Z = 20,
    beta2b = 0) — both previously "widely cited, unconfirmed".
  * Inducer LE: tip diameter 280 mm (r1t = 0.140 m), **hub diameter 90 mm
    (r1h = 0.045 m)** — the previously-missing hub radius.
  * Exit width **b2 = 26 mm**; axial length of impeller **130 mm**;
    relative tip clearance z_s/b2 = 0.027 (0.70 mm).
  * Laser operating point: **14 000 rpm, 5.31 kg/s — measured stage
    PR 2.1, stage eta_is 0.88** (near stage optimum). Design speed
    18 000 rpm at 7.16 kg/s (paper Fig. 16; Cumpsty quotes design PR 3.0).

Modelling choices (recorded):

  * Wall contours: quarter-ELLIPSES through the grounded endpoints — hub
    (z 0 -> 0.130 m, r 0.045 -> 0.200 m), shroud (z 0 -> 0.104 m,
    r 0.140 -> 0.200 m; 0.104 = 0.130 - b2) — axial-tangent at inlet,
    radial-tangent at exit, so the constant-radius inlet duct and the
    radial vaneless-diffuser stub join C1. The paper's Fig. 1 contours are
    a raster figure; the ellipse assumption is the recorded residual
    geometry infidelity (the blade camber lines are themselves elliptic).
  * Inducer metal angle: not stated numerically in the paper ("elliptic
    camber"); set for ZERO INCIDENCE at the 18 000-rpm design triangle,
    spanwise: beta1(y) = -atan(omega_design r1(y) / Vm1_design) with
    Vm1_design = 112 m/s (compressible one-D at 7.16 kg/s through the
    grounded inducer annulus). Recorded assumption, hub -37 deg / tip -67.
  * chord/solidity are friction-length representatives (meridional blade
    length ~ 0.18 m), not transcription — the skin-friction loss is the
    only consumer.
  * slcflow models the IMPELLER only: measured PR/eta are STAGE values
    (impeller + vaneless diffuser). PR comparison is nearly direct (the
    diffuser only loses a few % p0), so slcflow impeller-exit PR should sit
    AT-OR-SLIGHTLY-ABOVE the measured stage PR; stage efficiency is NOT
    directly comparable (deferred parasitic disk/recirculation/leakage +
    diffuser losses) — see docs/references/ECKARDT.md.

Status: measured-agreement record, pinned in ``tests/test_v7_eckardt.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np  # verification layer: case definitions  # ad6: allow

from ..closures.centrifugal import CENTRIFUGAL
from ..fluid.perfectgas import PerfectGas
from ..geometry import FlowPath, StationDef, StationType, WallCurve
from ..geometry.bladerow import ParamRowGeometry
from ..machine import (FidelityConfig, InletCondition, Machine, MassFlowSpec,
                       PerformanceResult, RowSpec)

__all__ = ["EckardtO", "KrainImpeller", "LASER_POINT", "DESIGN_POINT",
           "KRAIN_DESIGN"]

_DEG = np.pi / 180.0

# Measured anchors (stage values; impeller+vaneless diffuser — see docstring).
LASER_POINT = {"rpm": 14000.0, "mdot": 5.31, "stage_pr": 2.1,
               "stage_eta": 0.88}
DESIGN_POINT = {"rpm": 18000.0, "mdot": 7.16, "stage_pr": 3.0}


@dataclass(frozen=True)
class EckardtO:
    """Geometry-faithful-endpoints Eckardt rotor O (section 9.7)."""

    rpm: float = 14000.0          # laser-measurement speed
    mdot: float = 5.31            # kg/s
    T0_in: float = 288.15         # K (ambient rig inlet)
    p0_in: float = 101325.0       # Pa
    r1h: float = 0.045
    r1t: float = 0.140
    r2: float = 0.200
    b2: float = 0.026
    z_len: float = 0.130          # impeller axial length
    blade_count: int = 20
    beta2_blade_deg: float = 0.0  # radially-ending blades
    vm1_design: float = 112.0     # m/s, 18 000-rpm design (docstring)
    rpm_design: float = 18000.0
    chord: float = 0.18           # friction-length representative
    solidity: float = 2.5
    clearance: float = 7.0e-4     # z_s = 0.027 b2
    n_inblade: int = 6
    n_sl_rep: int = 7
    n_span_nodes: int = 21
    gas: PerfectGas = field(default_factory=PerfectGas)

    @property
    def omega(self) -> float:
        return self.rpm * 2.0 * np.pi / 60.0

    # ------------------------------------------------------------------
    def _walls(self):
        """Hub/shroud parametric callables: inlet straight + quarter-ellipse
        + radial diffuser stub, C1 at the joins (axial/radial tangents)."""
        z_in = -0.05
        r_ext = 0.230              # radial stub end (vaneless entry)

        def make(r_le, z_te):
            a_z, a_r = z_te, self.r2 - r_le
            # segment arc lengths (straight, ellipse ~ Ramanujan, straight)
            l1 = -z_in
            h = ((a_z - a_r) / (a_z + a_r)) ** 2
            l2 = 0.25 * np.pi * (a_z + a_r) * (
                1.0 + 3.0 * h / (10.0 + np.sqrt(4.0 - 3.0 * h)))
            l3 = r_ext - self.r2
            L = l1 + l2 + l3
            u1, u2 = l1 / L, (l1 + l2) / L

            def wall(u):
                u = np.asarray(u, dtype=float)
                th = (u - u1) / (u2 - u1) * 0.5 * np.pi
                z = np.where(u < u1, z_in + u * L,
                             np.where(u > u2, z_te,
                                      a_z * np.sin(th)))
                r = np.where(u < u1, r_le,
                             np.where(u > u2,
                                      self.r2 + (u - u2) * L,
                                      self.r2 - a_r * np.cos(th)))
                return z, r

            return wall, u1, u2

        hub = make(self.r1h, self.z_len)
        shd = make(self.r1t, self.z_len - self.b2)
        return hub, shd

    def _flowpath(self) -> FlowPath:
        (hub, h1, h2), (shd, s1, s2) = self._walls()
        w0 = WallCurve.from_callable(hub, n=301)
        w1 = WallCurve.from_callable(shd, n=301)
        stations = [StationDef(StationType.DUCT, 0.0, 0.0),
                    StationDef(StationType.EDGE_LE, h1, s1, row_id="imp")]
        for k in range(self.n_inblade):
            t = (k + 1) / (self.n_inblade + 1)
            stations.append(StationDef(
                StationType.INBLADE, h1 + t * (h2 - h1), s1 + t * (s2 - s1),
                row_id="imp"))
        stations += [StationDef(StationType.EDGE_TE, h2, s2, row_id="imp"),
                     StationDef(StationType.DUCT, 1.0, 1.0)]
        return FlowPath(w0, w1, stations)

    def _geometry(self) -> ParamRowGeometry:
        y = np.linspace(0.0, 1.0, self.n_span_nodes)
        r1 = self.r1h + y * (self.r1t - self.r1h)
        om_d = self.rpm_design * 2.0 * np.pi / 60.0
        beta1 = -np.arctan2(om_d * r1, self.vm1_design)
        return ParamRowGeometry(
            blade_count=self.blade_count,
            beta1=beta1,
            beta2=self.beta2_blade_deg * _DEG if self.beta2_blade_deg
            else 0.0,
            chord_len=self.chord, solidity_val=self.solidity,
            clearance=self.clearance)

    def machine(self) -> Machine:
        cp = self.gas.gamma * self.gas.R / (self.gas.gamma - 1.0)
        row = RowSpec(row_id="imp", omega=self.omega,
                      swirl=CENTRIFUGAL.swirl, loss=CENTRIFUGAL.loss,
                      blade_count=self.blade_count, geometry=self._geometry())
        return Machine(self._flowpath(), self.gas,
                       InletCondition(h0=cp * self.T0_in, s=0.0, rvt=0.0),
                       rows=[row])

    def evaluate(self, n_sl: int = 1, fidelity: FidelityConfig = None,
                 mdot: float = None) -> PerformanceResult:
        if fidelity is None:
            fidelity = FidelityConfig.tier1()
        target = self.mdot if mdot is None else mdot
        return self.machine().evaluate(MassFlowSpec(target), fidelity,
                                       n_sl=n_sl)

    # ------------------------------------------------------------------
    def parasitic_breakdown(self, result: PerformanceResult,
                            mdot: float = None) -> dict:
        """Aungier ch.-4 parasitic (shaft-side) works [J/kg] evaluated on
        the converged state (post-solve scalars; gate-#3 components —
        see closures/centrifugal/parasitic.py for provenance and why
        these are machine-level, not per-streamtube). Disk backface gap
        ratio 0.02 is a recorded assumption (not published for the rig).
        """
        from ..closures.centrifugal.parasitic import (
            disk_friction_work, leakage_work, recirculation_work)
        md = self.mdot if mdot is None else mdot
        res = result.result
        f, tr = res.fields, res.frozen.transported
        j_le, j_te = 1, 2 + self.n_inblade
        r1 = float(np.mean(f.metrics.r[:, j_le]))
        r2 = float(np.mean(f.metrics.r[:, j_te]))
        cu1 = float(np.mean(tr.rvt[:, j_le])) / r1
        cu2 = float(np.mean(tr.rvt[:, j_te])) / r2
        vm1 = float(np.mean(f.vm[:, j_le]))
        cm2 = float(np.mean(f.vm[:, j_te]))
        rho2 = float(np.mean(f.rho[:, j_te]))
        u1, u2 = self.omega * r1, self.omega * r2
        w1 = float(np.hypot(vm1, u1 - cu1))
        wu2 = u2 - cu2
        w2 = float(np.hypot(cm2, wu2))
        return {
            "disk_friction": disk_friction_work(md, rho2, u2, self.r2),
            "leakage": leakage_work(
                md, rho2, u2, r1, self.r2, self.r1t - self.r1h, self.b2,
                cu1, cu2, self.blade_count, self.clearance, self.chord),
            "recirculation": recirculation_work(
                u2, w1, w2, cm2, wu2,
                # cot of the blade exit angle FROM TANGENTIAL:
                # cot(90deg - |backsweep|) = tan(|backsweep|); radial
                # blades (Eckardt) give 0, Krain's 30-deg backsweep 0.577.
                float(np.tan(abs(self.beta2_blade_deg) * _DEG)),
                r1 * cu1, self.r2 * cu2, self.blade_count, self.chord),
        }

    def stage_efficiency(self, result: PerformanceResult,
                         mdot: float = None) -> float:
        """Total-to-total efficiency with the parasitic works debited from
        the shaft side: ``eta = dh_ideal(PR) / (dh0_flow + sum dh_par)``.
        Excludes the vaneless-diffuser p0 loss — see
        :meth:`stage_performance` for the R/R2 = 2 stage plane."""
        par = sum(self.parasitic_breakdown(result, mdot).values())
        cp = self.gas.gamma * self.gas.R / (self.gas.gamma - 1.0)
        kappa = (self.gas.gamma - 1.0) / self.gas.gamma
        dh_id = cp * self.T0_in * (result.pressure_ratio ** kappa - 1.0)
        dh0 = dh_id / result.efficiency
        return float(dh_id / (dh0 + par))

    def stage_performance(self, result: PerformanceResult,
                          mdot: float = None, cf: float = 0.005,
                          accounting: str = "aungier_lambda",
                          wake_fraction: float = 0.2,
                          wake_blockage: float = 0.05) -> dict:
        """Stage totals at the rig's R/R2 = 2 measurement plane: the
        impeller-exit result plus (a) the Aungier tip-distortion internal
        loss (``tip_distortion_loss``, the clearance/blockage effect),
        (b) the vaneless-diffuser skin-friction p0 loss
        (``vaneless_diffuser_loss``; the paper: constant flow area to
        R/R2 = 2), and (c) the parasitic shaft-work debits. ``cf = 0.005``
        is the Braembussche-typical value the internal set already uses.
        Geometric composites per Aungier's own definitions (Eqs 111/113,
        4-13; beta_th taken at the inlet — throat ~ LE for this inducer).
        Returns ``{"pr_stage", "eta_stage", "dh_vld", "dh_lambda"}``."""
        from ..closures.centrifugal.parasitic import (
            tip_distortion_loss, vaneless_diffuser_loss)
        res = result.result
        f, tr = res.fields, res.frozen.transported
        j_le, j_te = 1, 2 + self.n_inblade
        r1 = float(np.mean(f.metrics.r[:, j_le]))
        r2 = float(np.mean(f.metrics.r[:, j_te]))
        cu1 = float(np.mean(tr.rvt[:, j_le])) / r1
        cu2 = float(np.mean(tr.rvt[:, j_te])) / r2
        vm1 = float(np.mean(f.vm[:, j_le]))
        cm2 = float(np.mean(f.vm[:, j_te]))
        rho1 = float(np.mean(f.rho[:, j_le]))
        rho2 = float(np.mean(f.rho[:, j_te]))
        u1, u2 = self.omega * r1, self.omega * self.r2
        w1 = float(np.hypot(vm1, u1 - cu1))
        w2 = float(np.hypot(cm2, u2 - cu2))
        c2 = float(np.hypot(cm2, cu2))
        T2 = (float(np.mean(tr.h0[:, j_te])) - 0.5 * c2 * c2) / self.gas.cp

        # Aungier geometric composites (Eqs 111/113, 4-13): beta from
        # TANGENT = pi/2 - |beta_from_meridional|; radial exit -> sin = 1.
        b1_pass = self.r1t - self.r1h
        beta1_tan = 0.5 * np.pi - abs(float(
            self._geometry().beta1_blade(0.5)))
        w_bb1 = 2.0 * np.pi * r1 * np.sin(beta1_tan) / self.blade_count
        w_bb2 = 2.0 * np.pi * self.r2 / self.blade_count
        dh1 = 2.0 * b1_pass * w_bb1 / (b1_pass + w_bb1)
        dh2 = 2.0 * self.b2 * w_bb2 / (self.b2 + w_bb2)
        d_hyd = 0.5 * (dh1 + dh2)
        a1 = 2.0 * np.pi * r1 * b1_pass
        a2 = 2.0 * np.pi * self.r2 * self.b2
        area_ratio = a2 / (a1 * np.sin(beta1_tan))
        omega_sf = 2.0 * cf * (self.chord / d_hyd) * (1.0 + (w2 / w1) ** 2)
        pv1, pv2 = rho1 * w1 * w1, rho2 * w2 * w2
        # Clearance/wake-mixing accounting: the two grounded families model
        # the SAME physics and are mutually exclusive (CENT-LOSS.md):
        #   "aungier_lambda" — the Eq 4-12/120/5-36 distortion chain;
        #   "oh_native"      — Jansen clearance + Johnston-Dean mixing
        #                      (the Oh-1997 optimum-set components).
        if accounting == "aungier_lambda":
            dh_lambda = tip_distortion_loss(
                omega_sf, pv1, pv2, w1, w2, cm2, d_hyd, self.b2, self.chord,
                area_ratio, rho1, rho2, self.clearance)
        elif accounting == "oh_native":
            from ..closures.centrifugal.parasitic import (
                jansen_clearance_loss, johnston_dean_mixing_loss)
            dh_lambda = (jansen_clearance_loss(
                self.clearance, self.b2, cu2, vm1, self.r1h, self.r1t,
                self.r2, rho1, rho2, self.blade_count)
                + johnston_dean_mixing_loss(cm2, wake_fraction,
                                            wake_blockage))
        else:
            raise ValueError(f"unknown accounting {accounting!r}")

        # Aungier supercritical Mach loss (Eqs 5-41/42; 1-D convention:
        # mean-inlet values). Inert when the suction-surface peak stays
        # subsonic — the mechanism separating Krain (M1'~0.85 tip) from
        # Eckardt (M1'~0.67).
        from ..closures.centrifugal.parasitic import supercritical_loss
        T1 = (float(np.mean(tr.h0[:, j_le]))
              - 0.5 * (vm1 ** 2 + cu1 ** 2)) / self.gas.cp
        a1 = float(np.sqrt(self.gas.gamma * self.gas.R * T1))
        T0rel1 = T1 + 0.5 * w1 * w1 / self.gas.cp
        w_star = float(np.sqrt(2.0 * self.gas.gamma
                               / (self.gas.gamma + 1.0)
                               * self.gas.R * T0rel1))
        dw = 4.0 * np.pi * (self.r2 * cu2 - r1 * cu1) / (
            self.blade_count * self.chord)
        dh_cr = supercritical_loss(w1 / a1, w1, w2, dw, w_star)

        dh_vld = vaneless_diffuser_loss(cf, self.r2, 2.0 * self.r2,
                                        self.b2, c2, cu2, u2)
        # Losses -> entropy at the impeller-exit static state -> p0 debit.
        ds = (dh_vld + dh_lambda + dh_cr) / T2
        p0_fac = float(np.exp(-ds / self.gas.R))
        pr_stage = result.pressure_ratio * p0_fac
        par = sum(self.parasitic_breakdown(result, mdot).values())
        cp = self.gas.cp
        kappa = (self.gas.gamma - 1.0) / self.gas.gamma
        dh_id_stage = cp * self.T0_in * (pr_stage ** kappa - 1.0)
        dh_id_imp = cp * self.T0_in * (result.pressure_ratio ** kappa - 1.0)
        dh0 = dh_id_imp / result.efficiency
        return {"pr_stage": float(pr_stage),
                "eta_stage": float(dh_id_stage / (dh0 + par)),
                "dh_vld": float(dh_vld), "dh_lambda": float(dh_lambda),
                "dh_supercritical": float(dh_cr)}


# Krain design/measured anchors (Krain 1988 + Krain & Hoffmann 1989, via
# the Test Cases notebook 2026-07-17; see docs/references/ECKARDT.md
# "Krain" note): design rotor PR_tt 4.7 at 22 363 rpm / 4.0 kg/s; measured
# maxima: stage PR_tt ~4.5, impeller polytropic eta_tt 0.95, stage
# isentropic eta_tt 0.84.
KRAIN_DESIGN = {"rpm": 22363.0, "mdot": 4.0, "rotor_pr_design": 4.7,
                "stage_pr_max": 4.5, "impeller_eta_poly": 0.95,
                "stage_eta_max": 0.84}


@dataclass(frozen=True)
class KrainImpeller(EckardtO):
    """Krain 30-deg backswept impeller (section 9.7) — the second
    centrifugal validation point, cross-checking the Wiesner slip and the
    loss set beyond Eckardt at twice the pressure ratio.

    Geometry grounded from the PRIMARY papers via the Test Cases notebook
    (2026-07-17; the 1989 paper's Cartesian blade-coordinate table):
    r1h = 45.0 mm, r1t = 112.7 mm (LE, coordinate point 1); D2 ~ 400 mm
    (from U2 = 470 m/s at 22 363 rpm); b2 ~ 14.7 mm (TE hub-tip
    distance); 24 full blades, no splitters; exit backsweep 30 deg from
    radial; impeller axial length ~119.1 mm (hub LE -> TE X-extent).
    Tip clearance is NOT stated numerically in the papers (the 1989 CFD
    "modeled [it] by one grid line") — 0.5 mm is a recorded assumption.
    Same modelling frame as EckardtO (quarter-ellipse walls through the
    grounded endpoints; inducer metal angles set for zero incidence at
    the design triangle, Vm1_design ~ 102 m/s one-D compressible at
    4.0 kg/s through the grounded inducer annulus).
    """

    rpm: float = 22363.0
    mdot: float = 4.0
    r1h: float = 0.045
    r1t: float = 0.1127
    r2: float = 0.200
    b2: float = 0.0147
    z_len: float = 0.11909
    blade_count: int = 24
    beta2_blade_deg: float = -30.0   # backsweep, from radial (section 2.4)
    vm1_design: float = 102.0
    rpm_design: float = 22363.0
    chord: float = 0.16              # meridional blade-length representative
    clearance: float = 5.0e-4        # [VERIFY] not published; assumption
