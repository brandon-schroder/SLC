# ROTOR37 — NASA Rotor/Stage 37 (Moore & Reid, TP-1659)

**Source:** Moore & Reid, *Performance of Single-Stage Axial-Flow Transonic
Compressor With Rotor and Stator Aspect Ratios of 1.19 and 1.26,
Respectively, and With Design Pressure Ratio of 2.05*, NASA TP-1659 (1980).
Public domain, NTRS **19800012840**. Also in the "Turbomachinery: Test
Cases" NotebookLM notebook (2026-07-15). The rotor blade coordinates + LDA
data of the 1994 IGTI blind test are in **AGARD-AR-355** (in the notebook;
not yet transcribed).

**Role:** the canonical transonic axial-compressor validation case — the V5
point-by-point dataset (`model-readiness` gate #1). Exercises the shock
loss, endwall/clearance loss, and the spanwise tiers on a real twisted
transonic rotor.

## Transcribed (2026-07-15 → `slcflow/verification/v5_rotor37.py`)

- **Table I** design intent: rotor PR 2.106, η_ad 0.889, ṁ 20.188 kg/s,
  17 188.7 rpm, tip speed 454.136 m/s, hub/tip 0.70, 36/46 blades.
- **Table III(a)** rotor geometry at 11 span elements (verified against a
  page render, not the noisy OCR): radii RI/RO, cone-plane metal angles
  KIC/KOC, max thickness TM, chord, solidity, setting angle, LE/TE axial
  positions ZI/ZO (swept edges).
- **Table IV(a)** measured 100%-speed rotor line (5 readings, choke→stall):
  ṁ [20.93, 20.83, 20.74, 20.43, 19.60] kg/s, rotor PR [1.785, 1.917,
  2.056, 2.157, 2.196], rotor η [0.842, 0.862, 0.876, 0.867, 0.852].
  (Stage rows and the 50–90% speed tables are in the report, untranscribed.)
- NOT yet transcribed: Table II design blade-element radial profiles
  (partially quoted in the case tests), Table III(b) stator geometry,
  Tables V/VI measured radial surveys, tip clearance (case uses 0.04 cm
  `[VERIFY]` — get the blind-test value from AGARD-AR-355).

## Measured agreement (2026-07-15; pinned in `tests/test_v5_rotor37.py`)

Both tiers **converge on the faithful geometry** across the measured flow
range — the structural half is real. Quantitatively, at the measured
peak-η point (20.74 kg/s; measured PR 2.056, η 0.876):

| run | PR | η | validity |
|-----|----|---|----------|
| Tier-1 meanline | 2.38 (+16%) | 0.872 (−0.4 pt) | 0.0 |
| Tier-2 REE n_sl=5..11 | 2.31 (+12%) | 0.865 (−1.1 pt) | 0.0 |
| Tier-2 + uniform 4% blockage (probe) | 2.20 (+7%) | 0.850 | 0.0 |

**Findings.**
1. **PR reads systematically high.** Decomposition: ~5 points from
   zero-blockage modelling (the rig runs substantial endwall blockage; the
   `Machine` blockage seam exists but the case ships 0 pending a grounded
   schedule), and ~7 points from **Lieblein NACA-65 deviation
   under-predicting on MCA transonic sections** (~3.5° at mid-span:
   β2_flow 44.2° vs the report's design-intent 47.7°) → Euler over-work
   (meanline T0-ratio 1.322 vs design 1.265). Out-of-pedigree by design;
   now a measured number.
2. **Efficiency lands close (−0.4..−1.1 pt)** — but validity reads 0: the
   equivalent-diffusion factor sits at/above the SP-36 ceiling (D_eq ≈ 2.0
   meanline), so the profile loss is ceiling-saturated; part of the η
   agreement is that ceiling, not a validated loss level.
3. **Speedline much shallower than measured** (code 2.36→2.49 vs measured
   1.785→2.196 over the same flows): the measured choke-side collapse is
   shock/choking-dominated — outside the subsonic off-design bucket. Only
   the slope SIGN is pinned.

## Measured blade-element data — Table V(c), reading 4182 (added 2026-07-16)

The peak-η reading's full radial survey is transcribed into
`MEASURED_BE_4182` (9 stations, 5–95% span from tip): measured relative
flow angles, **deviation**, incidence, Mach, per-span efficiency and loss
coefficients. Direct comparison of the coded Lieblein deviation chain at
the measured incidence (pinned:
`test_v5_rotor37.py::test_measured_deviation_gap_on_mca_sections`):

| span % (from tip) | 5 | 10 | 15 | 30 | 50 | 70 | 85 | 90 | 95 |
|---|---|---|---|---|---|---|---|---|---|
| dev measured [°] | 10.1 | 8.5 | 7.9 | 7.0 | 7.4 | 7.2 | 11.0 | 11.2 | 10.7 |
| dev predicted [°] | 4.6 | 4.1 | 4.1 | 4.3 | 4.8 | 5.6 | 6.6 | 7.0 | 7.6 |

**Mean error −3.6°, RMS 3.8°** (best −1.6° at 70%, worst −5.5° near tip;
the endwall stations' measured "deviation" carries secondary/tip-leakage
contamination). This quantifies, per span, the deviation gap behind the
+7-point PR excess — the concrete target for an MCA/transonic deviation
correction.

> **CORRECTION LANDED (2026-07-16, [`AGARD745.md`](AGARD745.md)):** the
> Çetin AGARD-R-745 Eq. 3.5 polynomial, applied as published (no local
> constant), takes this to **RMS 1.2°, mean ~0**; the Rotor 37 case now
> defaults it ON. End-to-end: **Tier-2 PR 2.051 vs measured 2.056
> (+0.2%)**, Tier-1 2.135 (+3.8%), validity 0 → ~0.8 at Tier 1.

## Capacity / near-choke findings (2026-07-16, second AGARD pass)

Grounded targets: rig **choke flow 20.93 ± 0.3 kg/s**, tip clearance
implied ≈ 0.41 mm (AGARD-AR-355 via the notebook — the case's 0.4 mm
assumption is consistent; no single blockage number exists, only inlet
profiles).

- **Capacity gap measured:** B=0 chokes the meanline at ~22.25 kg/s
  (+6.5%) and Tier-2 at ~21.65 (+3.5% — the spanwise tier resolves the
  endwall streamtubes and captures half the gap). A uniform **B = 0.033
  lands the Tier-2 choke inside the measured band** (pinned) but costs
  the mid-line PR ~7% (4 of 5 points degrade) → the deficit is NOT
  uniform blockage; it lives at the unmodelled blade-passage **throat**
  (a compressor throat/capacity station = the recorded model item).
  Case defaults stay parameter-free (B=0).
- **Matched-PR (vertical-characteristic) comparison RUN (2026-07-16,
  `test_matched_pr_traversal_down_the_vertical_characteristic`):**
  descending `BackPressureSpec` traversal from a near-choke seed
  CONVERGES down the steep side at both tiers (branch guard holding, no
  spurious fixed points; Tier 2 even rides past the classical
  patience-declared choke, 22.0 vs 21.65). Reading the model
  characteristic at the measured PRs (2.056/1.917/1.785, where the rig
  passes 20.74/20.83/20.93 kg/s): Tier 1 ~21.3/22.0/22.5 (+2.7→+7.5%),
  Tier 2 ~20.9/21.5/21.9 (+0.9→+4.9%). **In the correct frame the whole
  choke-side disagreement is a capacity/knee error, not a PR/loss
  error** — the rig's unique-incidence knee is razor-sharp (0.2 kg/s
  over that PR span) while the annulus model rounds over ~1 kg/s.
  Levers unchanged: capacity level (inlet swallowing physics) + knee
  sharpness; the speedline-shape story is now fully dispositioned.
- **Row-throat check landed (2026-07-16, `test_throat_capacity.py`):**
  with a gauging-estimate throat `o = s·cos(KIC)` the rotor-relative
  throat capacity computes ~24.6 kg/s — ABOVE the ~22.25 annulus limit,
  so the internal throat does not bind: a supersonic-inlet rotor chokes
  at its inlet swallowing limit. The check stays on (inert here; it is
  the binding constraint on the TN D-6967 turbine, where it lands within
  ~1% of the rig's choke).
- **AGARD Eq. 3.3 off-design loss measured in both variants, not
  adopted:** the full parabola collapses stall-side η (0.776 vs measured
  0.852 at 19.60); the choke-only hybrid is inert (this rotor, like the
  rig, never runs below reference incidence — the measured choke-side PR
  collapse is the **vertical characteristic**, i.e. a back-pressure-mode
  comparison, not a loss bucket). Swan Eq. 70 likewise measured inert →
  the three speedline hypotheses are now all dispositioned by
  measurement; the remaining levers are the throat/capacity station and
  a `BackPressureSpec` comparison for the vertical-characteristic points.

## Operability disposition — gate #5 (2026-07-18;
## `test_speedline_operability_criteria_measured_disposition`)

One §6.7 `solve_speedline` traversal at Tier 1 across and past the
measured 100% line: **mechanically robust** (every point converges
classical, no escalation/cut-backs, PR monotone rising), but **no stall
criterion fires anywhere near the measured stall** (19.60 kg/s):

- criterion (b) `pr_turnover` *cannot* fire on this machine — the rig's
  own measured line is stall-truncated while PR still rises (1.785→2.196);
- criterion (c) `validity_saturated` fires at ~15.5 kg/s, **21% below**
  the measured stall (Çetin-corrected Tier-1 validity stays 0.6–0.9 over
  the measured range); at Tier 2 the same criterion instead fires at the
  *first* point (endwall D_eq out-of-window at all flows — the recorded
  spanwise validity limitation, not stall physics);
- criterion (a) `solver_failure` never triggers.

**Honest disposition:** the operability machinery works on a real
transonic rotor, but the stall *line* is not predicted by
solver/validity signatures — it needs a grounded aerodynamic loading
criterion. Choke side already dispositioned (capacity/knee section
above).

**Follow-on MEASURED-VALIDATED (2026-07-18): the tip diffusion factor is
that criterion.** Characterizing the Lieblein/NACA RM E53D01 tip
`D = 1 − W₂/W₁ + |ΔV_θ|/(2σW₁)` (see LIEB59.md) along the measured lines
of both transonic siblings (Tier 2, `tip_diffusion_factor`): each
rotor's **measured stall sits at tip `D ≈ 0.6`** (R37 0.63 at 19.60; R38
0.595 at 20.44) — above the tip *design* limit 0.45 (η=0.90), at the
2-D-cascade sharp-loss value. A `D_tip = 0.60` stall threshold predicts
R37 within +3.0% and R38 within −0.6% of measured stall flow **and
orders the siblings the way the loss set cannot** — the high-AR R38
reaches the loading limit at higher flow → stalls earlier (measured
20.44 > 19.60), the AR-sensitivity differential the correlation set
misses (Rotor 38 section; loss reads PR +6.6% vs +0.2%). **Loading, not
loss, is the right variable for the stall line.** Zero tuning. Pinned:
`test_v5_rotor38.py::test_tip_diffusion_factor_predicts_the_sibling_stall_differential`.
Recorded next step (a feature, not this characterization): graduate `D`
to a C¹-safe closure-layer quantity and wire a `D_tip`-threshold
`StallFlag` criterion into `solve_speedline` — that turns gate #5
predictive.

**Next steps this dataset unlocks** (gate #2, in payoff order): a grounded
blockage schedule (report design values / AGARD); an MCA/transonic deviation
correction (the `MEASURED_BE_4182` deviation profile above is the target);
extending the loss validity window / transonic loss level vs the measured
per-span loss coefficients; stator + stage comparison; 50–90% speedlines.

## Rotor 38 — the high-AR sibling (ADDED 2026-07-17, `v5_rotor38.py`)

NASA TP-2001 (NTRS 19820014395; in the Test Cases notebook): same
annulus/speed/flow family (17 188.7 rpm, 20.188 kg/s, hub/tip 0.70) with
**48 short-chord blades** (AR 1.63 vs 1.19) — design rotor PR 2.105 /
η 0.878. Measured 100% line (Table IV(a), six readings): the rig
**stalled before design flow** (peak η 0.849 at PR 1.969 at the minimum
flow; never reached design PR — the documented high-AR shortfall).
Transcribed via the `TABLES` hook (Rotor 37 machinery unchanged).

**Measured (2026-07-17, `test_v5_rotor38.py`):** both tiers converge;
η level generalizes (T1 +1.1 / T2 +0.5 pt); **PR does NOT track the
high-AR shortfall** — T2 matched-flow +6.6% vs Rotor 37's +0.2%: the
measured Stage 37→38 degradation (early stall / endwall sensitivity at
high AR; no part-span damper mentioned) is not carried by the
correlation set (Howell's s/h term even moves slightly the wrong way
with the 48-blade pitch). **The axial two-point trend finding** —
the AR-sensitivity gap is the quantified axial calibration target,
recorded not tuned (differential +0.2% vs +6.6% is the frame-robust
statement; both measured lines sit near the vertical characteristic).
