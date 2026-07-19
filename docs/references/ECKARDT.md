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
> (−3% work — a slip calibration observation, **later REFUTED — see the
> slip disposition below**). Stage η is NOT directly comparable (deferred
> parasitic + diffuser). The first-order
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
> **VANELESS DIFFUSER + λ TIP-DISTORTION ADDED (2026-07-17,
> `EckardtO.stage_performance`) — the laser-point STAGE validation
> CLOSES:** the Coppage/Stanitz diffuser closed form (Whitfield [30])
> plus Aungier's λ tip-distortion internal loss (Eqs 4-12/120/5-36; all
> in CENT-LOSS.md) carry the comparison to the rig's R/R₂ = 2 plane:
> **PR_stage 2.121 vs measured 2.1 (+1.0%), η_stage 0.8796 vs 0.88
> (−0.04 pt)** — chain η 0.969 (internal) → 0.9265 (+parasitics) →
> 0.8796 (+diffuser 1.36 kJ/kg + λ 2.0 kJ/kg), every component grounded
> verbatim, zero locally fitted constants (agreement partly fortuitous
> given the recorded geometric estimates — β_th ≈ β1, L_B = chord, disk
> gap 0.02; each magnitude individually plausible). Design point:
> PR_stage 3.172 vs 3.0 (+5.7%), η 0.824. Remaining refinements: the
> full Aungier marching diffuser + λ work-input role, Oh-1997 map
> digitization for off-design points, Krain second impeller.

> **SLIP DISPOSITION (2026-07-19) — Wiesner 0.877 CONFIRMED for Eckardt O;
> the "implied ~0.90" observation REFUTED.** The `~0.90` recorded above was
> a stale `(PR, η)` inversion made **before** the parasitic+diffuser+λ loss
> stack closed the stage comparison (2026-07-15); `0.90` is precisely the
> **Stanitz** value (`σ = 1 − 0.63π/Z = 0.901` for Z=20 radial) vs Wiesner
> `0.877`. Grounding (test-cases + loss-models notebooks) refutes it:
> - the **Eckardt 1976 paper states no measured slip factor** — the exit is
>   a distorted jet/wake (wake ≈35% of channel area, ≈15% of mass flow), so
>   a single mid-passage "slip" is ill-defined and only a mass-averaged
>   effective value (what Wiesner represents) is meaningful;
> - the **literature calls Wiesner (~0.877) the *better* Eckardt-O match
>   than Stanitz (~0.90)** (the potential-theory Stanitz constant suits a
>   different blade count/flow).
>
> **Measured, decisive:** with the closed stage chain the Eckardt-O stage PR
> is **2.091 (−0.4%) with Wiesner** vs **2.122 (+1.1%) with Stanitz 0.90** —
> the higher slip *worsens* the comparison and flips it to over-prediction.
> So Wiesner is right for this rig and no recalibration is warranted (it
> would degrade the validated PR). Wiesner form was already CONFIRMED in
> WIE67.md; this is the case-level confirmation. **Constants unchanged.**
> Pinned: `test_v7_eckardt.py::test_wiesner_slip_is_the_better_match_for_eckardt_o`.

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

## Krain second impeller — ADDED 2026-07-17 (`KrainImpeller`)

Geometry grounded from the primaries via the Test Cases notebook (the
1989 paper's Cartesian blade-coordinate table): r1h 45.0 / r1t 112.7 mm
(LE), D2 ≈ 400 mm (U2 470 m/s @ 22 363 rpm), b2 ≈ 14.7 mm (TE hub-tip),
**24 full blades no splitters**, backsweep **30° from radial**, axial
length ≈ 119.1 mm; design 4.0 kg/s; design rotor PR_tt 4.7; measured
maxima: stage PR_tt ≈ 4.5, impeller η_polytropic 0.95 (≈ η_is 0.938),
stage η_is 0.84. Tip clearance NOT published → 0.5 mm recorded
assumption. Case = `EckardtO` subclass (same quarter-ellipse frame).

**Measured (2026-07-17, `test_krain_second_impeller_measured_agreement`):**
Tier 1 AND Tier 2 converge, validity 1.0, agreeing to 0.2% (impeller-exit
PR 5.00, internal η 0.972). Stage chain: **PR_stage 4.714 vs measured
stage max 4.5 (+4.8%)** — the PR side generalizes; **η_stage 0.905 vs
0.84 (+6.5 pt)** — the loss set that closes at Eckardt's PR 2.1 reads
LIGHT at PR 4.7 (≈3.4 pt internal at high loading, recirculation floors
to exactly 0 at design backsweep, clearance assumed). The two-point trend
is the finding: loss magnitudes calibrate adequately at moderate loading
and under-read at high loading — the quantified target for the next
centrifugal-loss calibration pass. A cot(β₂ᵦ) wiring bug in
`parasitic_breakdown` (hard-coded radial) was found and fixed by this
case — Eckardt unchanged (cot 0), Krain recirculation was reading
36.6 kJ/kg spuriously.

## CC3 — third centrifugal point, ADDED 2026-07-19 (`CC3Impeller`)

The **NASA CC3** (Allison Engine Company / McKain-Holbrook 4:1 stage) —
the third centrifugal validation point, a modern **transonic-inducer,
50°-backswept, splittered** design distinct from the radial Eckardt O and
the 30° Krain. **The recorded "CC3" acquisition item was mislabeled** (its
NTRS ID was the *Low-Speed* Centrifugal Compressor; see LSCC.md); the real
CC3 was found grounded in **Skoch, "Experimental Investigation of
Centrifugal Compressor Stabilization Techniques", J. Turbomach. 125 (2003)
704** (Drive `skoch_experimental_2003`, DOI 10.1115/1.1624846).

**Geometry (verbatim, Skoch 2003; `CC3_DESIGN`):** inlet tip radius 105 mm
(dia 210), inlet blade height 64 mm (r1h = 41 mm), exit radius 215.5 mm
(dia 431), b2 = 17 mm; **15 main + 15 splitter blades, 50° backsweep**, tip
clearance 2.4% b2 = 0.41 mm; PR **4:1** at 21 789 rpm / 4.54 kg/s, exit tip
speed 492 m/s, inlet relative Mach 0.9 tip / 0.45 hub. Recorded estimates
(not in Skoch — in the McKain-Holbrook coordinate report): axial length,
meridional chord, blade solidity. `blade_count = 30` is the exit-effective
count (main + splitter). CC3 uses a **vane-island diffuser**, not the
vaneless space of Eckardt/Krain, so the vaneless stage chain does not apply.

**Measured (2026-07-19, `test_v7_eckardt.py`):** all three tiers converge,
validity 1.0; **U2 = 491.7 m/s reproduces Skoch's 492 exactly** (geometry
validated). Impeller-exit PR **5.28** — above the measured stage 4:1
(the impeller over-compresses; the vane diffuser then recovers the large
exit dynamic head with a total-pressure loss the model doesn't carry).
Transonic inducer tip (W1t/a1 ≈ 0.83). **Point-by-point stage validation
is [VERIFY]** — blocked on a vane-island-diffuser model and a grounded
design η (Skoch Fig. 15 is a curve, not tabulated).

**CC3 corroborates the backsweep-dependent work over-prediction** (the λ
work-input finding, CENT-LOSS.md): the model's exit swirl Vθ2/U2 ≈ 0.75 vs
the ≈0.68 implied by stage PR 4:1 at a design η ≈ 0.86 → **~+10% work, the
largest of the three points** (Eckardt 0° exact, Krain 30°, CC3 50°) —
exactly the direction of the λ blockage term `−λ·Cm2·tan(β2b)`, which grows
with backsweep. So CC3 is the case that would most benefit from the λ
work-input role, and strengthens the case for it (or an equivalent) as the
missing high-backsweep physics — a recorded refinement (joint recalibration
per CENT-LOSS.md). Pinned:
`test_cc3_third_centrifugal_point` + `test_cc3_corroborates_backsweep_work_trend`.
