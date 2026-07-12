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
