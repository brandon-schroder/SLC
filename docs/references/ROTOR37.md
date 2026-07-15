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

**Next steps this dataset unlocks** (gate #2, in payoff order): a grounded
blockage schedule (report design values / AGARD); an MCA/transonic deviation
correction (Table V measured exit angles are the calibration target);
extending the loss validity window / transonic loss level vs the measured η
line; stator + stage comparison; 50–90% speedlines.
