# ECKARDT — Centrifugal impeller validation data (O / A / B): what is grounded

The canonical centrifugal-compressor validation set (DFVLR laser-anemometry
impellers O/A/B). Assembled 2026-07-12 for the V7 validation effort.

> **UPDATE 2026-07-15 — the primary Eckardt 1976 paper is now IN the
> library** ("Turbomachinery: Test Cases" NotebookLM notebook) and the
> missing geometry is grounded verbatim: **r1h = 45 mm, r1t = 140 mm
> (inducer LE), b2 = 26 mm, Z = 20 radially-ending blades, axial length
> 130 mm, z_s/b2 = 0.027**; laser point 14 000 rpm / 5.31 kg/s / stage
> PR 2.1 / stage η_is 0.88; design speed 18 000 rpm at 7.16 kg/s. The
> geometry-faithful-endpoints case is `slcflow/verification/v7_eckardt.py`
> (quarter-ellipse wall assumption recorded there), pinned by
> `tests/test_v7_eckardt.py`. **Results:** ALL THREE tiers converge with
> validity 1.0 and agree to ~0.1% (the synthetic-V7-testbed Tier-3 fold is
> a property of its tight 0.08 m bend, not of radial machines); laser-point
> impeller-exit PR 2.20 vs measured stage 2.1 (+4.7%, the unmodelled
> vaneless-diffuser p0 loss is the right size for the gap); design-point PR
> 3.38 vs 3.0 (+12.6%, gap grows with the deferred parasitic/clearance/
> diffuser losses); measured (PR, η) implies slip ~0.90 vs Wiesner 0.877
> (−3% work — a slip calibration observation). Stage η is NOT directly
> comparable (deferred parasitic + diffuser). The first-order
> `tools/eckardt_anchor.py` concentric-bend study is SUPERSEDED by this
> case. Remaining refinements: the true Fig. 1 wall contours (raster), the
> Oh-1997 map digitization for off-design points, Eckardt A/B backswept.

> **PARASITIC SET LANDED (2026-07-16, gate #3 —
> `closures/centrifugal/parasitic.py`, CENT-LOSS.md "parasitic"):** the
> deferred disk-friction + leakage + recirculation works (Aungier 2000
> ch. 4 verbatim) now debit the stage efficiency via
> `EckardtO.stage_efficiency`. Measured at the laser point: DF 370 +
> leakage 765 + recirculation 2327 J/kg (≈4.4% of work) → **η 0.969 →
> 0.9265** vs measured stage 0.88; at the 18 000-rpm design point
> recirculation grows with loading → η 0.877. The remaining ~4.6-pt
> laser-point gap is the unmodelled **R/R₂ = 2 vaneless diffuser** (+ the
> λ tip-distortion internal clearance effect) — the recorded next
> refinements for a full stage-η comparison. Assumptions recorded: disk
> backface gap s/r2 = 0.02, blade length = the friction-length chord.
>
> **VANELESS DIFFUSER ADDED (2026-07-17,
> `EckardtO.stage_performance`):** the Coppage/Stanitz closed form
> (Whitfield [30], CENT-LOSS.md) closes the chain to the rig's R/R₂ = 2
> plane: laser point Δh_vld ≈ 1.36 kJ/kg → **η 0.9074 vs measured 0.88
> (+2.7 pt)**, **PR_stage 2.167 vs 2.1 (+3.2%)**; design PR_stage 3.308
> vs 3.0 (+10.3%), η 0.859. The full stage comparison is now assembled
> end-to-end (internal → +parasitics → +diffuser); remaining-gap
> candidates: λ tip-distortion, marching-vs-closed-form diffuser, cf.

Historical context (2026-07-12, pre-primary-paper) below.

## Grounded from the library

**Cumpsty, *Compressor Aerodynamics* (1989)** — Drive
`cumpsty_compressor_1989_Combined.md` (44 Eckardt mentions, §2.4 and Ch. 6/7).
Verbatim facts:

- **Three impellers, common outer diameter** $D_2 = 400$ mm ($r_2 = 0.2$ m),
  same tip/inlet radii and same axial tip width $b_2$ at outlet; each run with
  a **vaneless diffuser** (Fig. 2.21, "From Eckardt, 1977"):
  - **O** — radial outlet (no backsweep, $\beta_{2b}=0$), *with* inducer.
  - **A** — *with* inducer, **30° backsweep** (vanes swept over the outer 20%
    of the radius; same shroud line + blade shape as O to 80% of $r_2$).
  - **B** — industrial type, **no inducer**, **40° backsweep**.
- **Eckardt O design point (Eckardt 1976, quoted by Cumpsty Ch. 6):** "tip
  diameter of 400 mm, designed to give a stagnation pressure ratio of **3.0**
  with a mass flow of **7.2 kg/s at 18 000 rev/min**." Peak **polytropic
  efficiency over 90%** (impeller) at the laser-measurement speed.
- **Eckardt O laser-measurement point (Fig. 6.2):** **14 000 rev/min** (78% of
  the 18 000 design speed), $\dot m = 5.31$ kg/s, **pressure ratio 2.1**.
- **A/B choke map (Fig. 2.21):** at 16 000 rev/min B (no inducer) chokes at
  ~6 kg/s; A (with inducer) passes >7 kg/s with no sign of choking.
- **Inducer sizing rule (Cumpsty §1/§2, representative):** inducer tip diameter
  ≈ **0.7–0.8 × outlet diameter** for best efficiency at $N_s\approx0.7$ → for
  Eckardt O, inducer tip $\approx 0.28$ m ($r_{1t}\approx0.14$ m).
- **NOT Eckardt (do not confuse):** Krain (1987) — 30° backsweep, $D_2=400$ mm,
  $U_2=468$ m/s at design, $N_s\approx0.62$, impeller PR ≈4 at ~4 kg/s. A
  *different* impeller with better shroud design.

**Oh, Yoon & Chung 1997** (Drive `oh_optimum_1997.md`) — the loss-model paper
whose blade-loading form slcflow uses; validates against Eckardt O/A/B with
**PR maps (Figs 2–4) and isentropic-efficiency maps (Figs 7–9)**. These are
**figures** (would need digitization). The paper tabulates only the KIMM
impeller geometry, **not** the Eckardt geometry.

## Still MISSING (needs the Eckardt primary papers, not in-library)

The exact geometry for a rigorous point-by-point case:
- inducer **hub** radius $r_{1h}$ (only the ~0.7·$D_2$ *tip* ratio is grounded);
- exit width **$b_2$** (Cumpsty says "same for all three" but gives no value;
  the widely-cited value is 26 mm — **unconfirmed from the library**);
- **blade count $Z$** (widely cited as 20 for O — **unconfirmed here**);
- inducer/exit **blade angles**, and the **meridional wall profiles** (Fig. 6.2
  is a figure; the real shroud/hub lines are not concentric arcs).

Primary sources to obtain: **Eckardt (1976)** *Trans. ASME J. Eng. Power* 98
(detailed velocity measurements), **Eckardt (1980)** ASME 80-GT (laser
velocimeter, backswept). Or a pedigree dataset (Japikse 1987, ref 19 in Oh).

## First-order design-point ANCHOR (2026-07-12, not a rigorous validation)

Using only the grounded numbers ($D_2=0.4$ m, radial, inducer tip $=0.7 r_2$,
$T_{01}\approx288$ K, ambient) with slcflow's V7 machinery (concentric-bend
meridional **approximation**, $Z=20$ assumed, impeller-internal loss only),
`tools/eckardt_anchor.py` (rerunnable):

| Operating point | slcflow PR | measured PR | Δ |
|---|---|---|---|
| 14 000 rpm, 5.31 kg/s | 1.95 | 2.1 | −7% |
| 18 000 rpm, 7.2 kg/s | 2.84 | 3.0 | −5% |

Right magnitude, right speed trend, a consistent ~5–7% under-prediction, and
robust to $Z$ (16→30 moves PR 2.78→2.93 at 18 000 rpm). slcflow η reads ~0.95
(**impeller-internal loss only**; the measured >90% is impeller polytropic, and
slcflow's would fall toward it once the **deferred** tip-clearance + disk-friction
losses are added). **This anchors the Euler-work + Wiesner-slip + corrected
blade-loading loss at the design point — it is NOT a geometry-faithful
point-by-point validation** (approximated meridional profile, unconfirmed
$b_2/Z/r_{1h}$, η not yet stage-comparable). The ~5–7% PR offset is the size of
the geometry-approximation error to close with the real profile.

## Path to a rigorous V7 validation (recorded, not done)

1. Obtain Eckardt 1976/1980 (or Japikse 1987) → exact $r_{1h}$, $b_2$, $Z$,
   blade angles, and the shroud/hub meridional profiles.
2. Fit the real meridional walls into a `WallCurve.from_callable` FlowPath
   (replace the concentric-arc approximation).
3. Digitize the PR + η maps (Oh 1997 Figs 2–4/7–9, or Eckardt's own) →
   `tools/digitize_eckardt.py` + a reference test (the `digitize_*` pattern).
4. For **efficiency**, either add the deferred parasitic (disk/recirc/leakage) +
   vaneless-diffuser losses (Oh 1997 Table 6 gives every formula) to compare
   **stage** η, or compare **impeller-exit** total conditions only. **PR** and
   **exit swirl** are already ~comparable (loss-insensitive).

See memory `centrifugal-validation-dataset`, `model-readiness`.
