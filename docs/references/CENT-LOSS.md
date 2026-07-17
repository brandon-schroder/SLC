# CENT-LOSS — Centrifugal impeller internal loss (Galvas/Aungier): verified

Sources: the NASA meanline loss form (**Galvas** NASA TN D-7487; Todd/Futral/
Jansen/Qvale lineage), **Conrad et al. (1980)**, **Aungier** *Centrifugal
Compressors* (2000), **Van den Braembussche** (2020) — all in the NotebookLM
"Staging Area (Theory)" notebook. Extracted 2026-07-09, source-grounded.
Cross-checked against `slcflow/closures/centrifugal/loss.py`.

## Confirmed — code matches a published form

| Loss | Source | Code | Status |
|------|--------|------|--------|
| Inducer incidence | Galvas/NASA Eq 5.6: `Δh = W_x²·sin²(β_x−β_opt)/2` — and `W_x·sin(Δβ) = ΔWθ`, so `= ½(ΔWθ)²` | `incidence_loss` `:50` | ✅ form + coefficient 0.5 (KE conversion) |
| Skin friction (leading factor) | Galvas: `Δq_sf = 4Cf·L·W̄²/(2D·U_T²)` → `Δh = 2Cf·(L/D)·W̄²` | `skin_friction_loss` `:57` | ✅ leading `2Cf` confirmed (= `4Cf·W²/2`) |
| `Cf` default | Braembussche: `0.005` typical for wall friction (Rodgers ≥0.003; Japikse ~0.01, range 0.005–0.02) | `cf = 0.005` `:69` | ✅ defensible / on the low-typical end |

Confirmed forms pinned in `tests/test_centrifugal_loss_reference.py`.

## Findings — modeling choices (both RESOLVED to Aungier 2000, 2026-07)

1. **Incidence factor `f_inc` — RESOLVED.** The code applied the *full* NASA
   kinetic energy `½(ΔWθ)²` (`f_inc = 1`). This is a genuine 0.5–1.0 family:
   Conrad et al. (1980) `k(ΔWθ²/2)` with `k = 0.5–0.7`; Aungier's
   total-pressure form (Eq 5-27) carries a leading `0.8`; Wasserbauer-Glassman
   (1975) use `sin³` for positive incidence. **Resolved:** exposed as a tunable
   `CentrifugalLoss.f_inc` field, default **0.8** (Aungier — coherent with the
   mean-of-squares friction choice below), applied as a multiplier on the
   confirmed Galvas Eq 5.6 KE form. Genuinely design-dependent, so tunable
   rather than a single "correct" constant. Pinned in
   `tests/test_centrifugal_loss_reference.py::test_f_inc_default_is_aungier_0p8`
   + `test_centrifugal_loss.py::test_f_inc_scales_incidence_only`.

2. **Mean-velocity definition in skin friction — RESOLVED.** The code formed
   `W_avg = ½(W1 + W2)` then squared → `[½(W1+W2)]²` (square of the mean).
   **Resolved to Aungier (2000)'s mean of the squares** `W̄² = ½(W1² + W2²)`
   (the physical passage average, since friction ∝ local `W²`; `≥` the
   square-of-mean by convexity, gap growing with `W1/W2` diffusion). The
   caller now passes the RMS velocity into the unchanged `skin_friction_loss`.
   Pinned in `test_skin_friction_mean_of_squares_convention`.

3. **`L/D_hyd` is geometry-derived**, not a universal constant. The code's
   `l_over_dhyd = 4.0` is a representative scalar design input; Aungier builds
   `d_H` from throat/tip areas and `L_H` from the mean-camberline path length.
   Fine as a design input; a geometry-derived value is the recorded refinement.

## Blade-loading (diffusion) loss — added 2026-07, ratio CORRECTED 2026-07-12

`Δh_bl = 0.05 D_f² U2²` (**Coppage et al. 1956** WADC 55-257; Oh-Yoon-Chung 1997
optimum-set Table 6), with the radial diffusion factor

```
D_f = 1 − W2/W1 + 0.75 (Δh_euler/U2²)(W2/W1) / [ (Z/π)(1 − r1/r2) + 2 (r1/r2) ]
```

Leading constant `0.05` and the `D_f` structure CONFIRMED. **The loading-term
ratio was `W1/W2` (numerator) and is now corrected to `W2/W1`** (equivalently
`W1/W2` in the *denominator* of that fraction).

- **Source (decisive):** Oh, Yoon & Chung (1997) — the exact paper the
  `0.05 D_f² U2²` form is cited from — prints, verbatim in clean text (Drive
  `oh_optimum_1997.md`, eqn above Table 5 + Table 6):
  `D_f = 1 − W2/W1t + 0.75(Δh_Euler/U2²) / { (W1t/W2)[(Z/π)(1 − D1t/D2)
  + 2 D1t/D2] }`. The `(W1t/W2)` is in the **denominator**, so the loading term
  carries `W2/W1`.
- **The earlier `W1/W2` was a bug**, on (i) an ambiguous NotebookLM MathML
  scrape and (ii) a mistaken "must grow with diffusion" argument — diffusion is
  carried by the leading `1 − W2/W1` term; the loading term is a positive
  correction proportional to the *loading* `Δh_Euler`, not a second diffusion
  term. Same category as the LIEB59 ω̄ velocity-ratio inversion.
- **The "Aungier 2000 Eq 5.15" attribution was also wrong.** Aungier's book
  (Drive `aungier_centrifugal_2000_part1.md`) uses a *different* blade-loading
  form entirely: `ω̄_BL = (ΔW/W1)²/24` (Eq **5-34**), a total-pressure coefficient
  based on the blade velocity difference `ΔW` — not `0.05 D_f² U2²`. Citation
  corrected to Coppage/Oh.

A smooth `D_f` ceiling (2.5) bounds transients (the axial ω̄-ceiling analogue);
with the corrected ratio `D_f → 1` as `W2 → 0` (the loading term vanishes), so
the transient is intrinsically better-behaved than the old blow-up.
`blade_loading_loss` + reference tests `test_blade_loading_matches_coppage_oh1997`,
`test_blade_loading_uses_w2_over_w1_not_w1_over_w2`,
`test_blade_loading_grows_with_loading` in `test_centrifugal_loss_reference.py`.

**Measured (2026-07-12, ratio fix).** At V7 design the loading term drops from
+0.400 to +0.062 → `D_f 1.005 → 0.668`, `Δh_bl 5609 → 2474 J/kg` (**2.27× less**,
still the dominant internal loss vs friction ~1.2, incidence ~0.2 kJ/kg).
Downstream (Tier 1/Tier 2, `n_sl=7`):

| Case | η before | η after | exit s-spread before → after |
|---|---|---|---|
| V7 T1 | 0.799 | **0.839** | — |
| V7 T2 | 0.803 | **0.828** | 69.6 → 50.5 J/(kg·K) (−27%) |
| V8 T1 | 0.897 | **0.930** | — |
| V8 T2 | 0.897 | **0.918** | 20.3 → 14.2 J/(kg·K) (−30%) |

Efficiency moves toward realistic and stratification falls ~27–30% ("less
stratified is key"). The radial/mixed **Tier-3 fragility is EASED but not
cracked**: the V7 T3 fold shifts (mdot=14 now reaches PR 2.28/η 0.91/s-spread 27
before failing, vs garbage before) but T3 still does not converge, and V8 T3
stays choke_limited at its mdot=12 — so the T3 `xfail` tripwires stand (see the
V7/V8 test dispositions and Appendix C.7/C.8). Multistage V5 is unaffected (axial
Lieblein set, AD-5).

Tip-clearance and disk-friction remain deferred (below); a full point-by-point
Eckardt *stage* validation additionally needs the parasitic (disk/recirc/leakage)
+ vaneless-diffuser losses Oh 1997 includes in its stage η — a separate item.

## Residual (tip-clearance / disk-friction)

Still deferred and **[VERIFY]** — a per-streamtube closure lacks their inputs:
**tip-clearance** (Jansen 1967 needs exit blade width `b2` + hub/tip radii, absent
from the §4.1 contract) and **disk-friction/windage** (machine-level parasitic
`~ρ2 U2³ r2²/ṁ`, no `ṁ` in a per-streamtube model); recirculation/leakage likewise.
Incidence + skin friction remain form-verified with both convention `[DECIDE]`s
resolved to Aungier (2000). No bug found here (contrast LIEB59).

## Parasitic (shaft-side) set — ADDED 2026-07-16 (gate #3)

Extracted verbatim from **Aungier 2000 ch. 4** via the theory notebook →
`closures/centrifugal/parasitic.py`, pinned by
`tests/test_parasitic_reference.py`:

- **Disk friction** (Eqs 4-21..4-25, 4-31): `dh_DF = C_M ρ r2² U2³/(2ṁ)`,
  `Re = ρωr2²/μ`, `C_M` = 0.75 × the LARGEST of the four Daily-Nece regime
  correlations (`2π/((s/r2)Re)`, `3.7(s/r2)^0.1/√Re`,
  `0.08/((s/r2)^{1/6}Re^{1/4})`, `0.102(s/r2)^0.1/Re^{0.2}`).
- **Leakage** (Eqs 4-17..4-19, 4-40): `Δp_CL = ṁ(r2C_U2−r1C_U1)/(z r̄ b̄ L)`;
  `U_CL = 0.816√(2Δp_CL/ρ2)`; `ṁ_CL = ρ2 z s L U_CL`;
  `dh_L = ṁ_CL U_CL U2/(2ṁ)`.
- **Recirculation** (Eqs 4-41..4-43): `dh_RC = I_R U2²`,
  `I_R = (D_eq/2−1)(W_U2/C_m2 − 2cot β2b) ≥ 0`, impeller
  `D_eq = W_max/W2`, `W_max = (W1+W2+ΔW)/2`,
  `ΔW = 2π d2 U2 I_B/(z L_B) = 4π(r2C_U2−r1C_U1)/(z L_B)`.

Parasitic accounting (Aungier): added to SHAFT work, no pressure rise →
post-solve scalar debits of stage efficiency at the case/facade level
(machine-level `ṁ` is why they cannot be per-streamtube LossModel
components — the recorded M7 deferral, now discharged). Recorded
assumptions: disk backface gap `s/r2 = 0.02`, `μ = 1.81e-5`, blade length
= the case friction-length chord. Aungier's internal clearance effect
(the λ tip-distortion factor, Eq 5-36) is a separate recorded refinement.

## Vaneless-diffuser loss — ADDED 2026-07-17 (the stage-plane companion)

`parasitic.vaneless_diffuser_loss`: the Coppage et al. (1956)/Stanitz
(1952) closed form as quoted by **Whitfield & Baines (1990) Eq. [30]**
(theory notebook, verbatim):

    delta_q = cf·r_x·(1−(r_x/r_y)^1.5)·(C_x/U_T)² / (1.5·b_x·cos α_x)

(work-coefficient units × U_T²; α from tangent). An INTERNAL p0 loss at
the stage level — applied post-solve as an entropy debit
`p0_factor = exp(−Δh/(T2·R))` in `EckardtO.stage_performance` (the rig's
constant-area vaneless space to R/R₂ = 2). `cf = 0.005` default (the
Braembussche-typical value the internal set already uses). Aungier's full
radial-marching treatment (Eqs 5-45/5-46 + his pipe-friction cf model,
also extracted) is the recorded refinement.

**Measured (Eckardt O laser point):** Δh_vld ≈ 1356 J/kg → stage chain
η 0.969 (internal) → 0.9265 (+parasitics) → **0.9074 (+diffuser)** vs
measured stage 0.88; PR_stage 2.167 vs 2.1 (+3.2%, from +4.7% at the
impeller exit). Design point: PR_stage 3.308 vs 3.0 (+10.3%), η 0.859.
Remaining +2.7 pt candidates: λ tip-distortion internal loss, closed-form
vs marching diffuser, cf level, η-definition subtleties.

## λ tip-distortion loss — ADDED 2026-07-17 (the chain closes)

`parasitic.tip_distortion_loss`: Aungier's clearance/blockage internal
loss, verbatim (theory notebook):

    B2 = ω̄_SF(pv1/pv2)√(W1 d_H/(W2 b2))
         + [0.3+(b2/L_B)²]A_R²ρ2 b2/(ρ1 L_B) + s_CL/(2b2)      (Eq 4-12)
    λ = 1/(1−B2)  (Eq 120);   ω̄_λ = [(λ−1)C_m2/W2]²           (Eq 5-36)

with d_H = mean of 2b·w/(b+w) at inlet/exit, blade-to-blade
w = 2πr·sinβ/Z (β from tangent; Eqs 111/113) and
A_R = A2 sinβ2/(A1 sinβ_th) (Eq 4-13). Returned on the ch.-5
inlet-relative reference (×½W1²); B2 guarded below the λ pole (0.9).
λ's second role (distorting the work-input triangle) = Aungier's full
analysis, recorded refinement. Recorded estimates: β_th ≈ β1 (throat ~
LE), L_B = the case chord.

**Measured — the Eckardt laser-point STAGE validation closes:**
Δh_λ ≈ 1996 J/kg → **PR_stage 2.121 vs measured 2.1 (+1.0%), η_stage
0.8796 vs 0.88 (−0.04 pt)** — chain: internal 0.969 → +parasitics
0.9265 → +diffuser+λ → 0.8796, every component grounded verbatim, zero
locally fitted constants. (Agreement partly fortuitous given the
recorded geometric estimates; component magnitudes each plausible.)
Design point: PR_stage 3.172 vs 3.0 (+5.7%), η 0.824.

## High-loading calibration pass — DISPOSITIONED 2026-07-17

Diagnosis (loss-budget probe, Eckardt control vs Krain): both rigs read
LIGHT on impeller-internal loss vs measured impeller η (Eckardt implied
~6.8 vs closure 2.3 kJ/kg; Krain ~10.8 vs 4.8); Eckardt's stage closes
because the stage-side stack compensates, Krain's (+6.5 pt) does not.
Three grounded mechanisms implemented + measured, none adopted:

1. **Oh-native accounting** (`jansen_clearance_loss` — verbatim Jansen
   1967 via Whitfield; `johnston_dean_mixing_loss` — J&D 1966 with the
   MERIDIONAL velocity only, Aungier quoted verbatim on why; ε_w
   0.15–0.25 / b* 0.02–0.12 typical ranges, defaults 0.2/0.05),
   selectable via `stage_performance(accounting="oh_native")`, mutually
   exclusive with the λ chain (same physics family). Measured 2×2:
   swings only ~2.5 kJ/kg — cannot close Krain's ~11 kJ/kg gap and
   overshoots Eckardt to −3.8 pt → **λ stays the default accounting**.
2. **Aungier supercritical Mach loss** (`supercritical_loss`, Eqs
   5-41/42 verbatim; onset = suction-surface peak sonic, W_max > W*):
   **inert at both rigs' 1-D mean inlet at design** — not the mechanism
   at this fidelity; a tip-resolved variant is the recorded follow-up
   (Krain M1t′ ≈ 0.85).
3. **The Krain +6.5 pt stands recorded** (~5–6% of work at PR 4.7).
   Measurement-narrowed suspects: the Krain stage measurement
   plane/η definition, the assumed 0.5 mm clearance, tip-resolved
   supercritical, loading-grown wake fraction.
