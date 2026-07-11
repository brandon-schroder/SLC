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

## Blade-loading (diffusion) loss — added 2026-07

`Δh_bl = 0.05 D_f² U2²` (Coppage/Jansen; **Aungier 2000 Eq 5.15**;
Oh-Yoon-Chung 1997 optimum set), with the radial diffusion factor

```
D_f = 1 − W2/W1 + 0.75 (Δh_euler/U2²)(W1/W2) / [ (Z/π)(1 − r1/r2) + 2 (r1/r2) ]
```

Leading constant `0.05` and the `D_f` structure CONFIRMED via NotebookLM (Staging
Area Theory). **Ratio caveat:** the source render is ambiguous on the loading-term
fraction; it is `W1/W2` (>1 under diffusion), resolved by (i) the Oh-Yoon-Chung /
Galvas consensus and (ii) the physical requirement that the loss GROW with
diffusion (W2≪W1) — pinned by `test_blade_loading_grows_with_diffusion`. A smooth
`D_f` ceiling (2.5) bounds the transient blow-up (the axial ω̄-ceiling analogue).
`blade_loading_loss` + reference tests in `test_centrifugal_loss_reference.py`.

**Measured:** at V7 design `D_f ≈ 1.12`, `Δh_bl ≈ 6.9 kJ/kg` — the DOMINANT
internal loss (vs friction ~1.2, incidence ~0.2). Drops V7/V8 η 0.98 → a realistic
**~0.90**. **Landing note (memory `centrifugal-blade-loading-wip`):** the loss is
so dominant it drives the fragile radial/mixed **spanwise** solves into the
documented freeze-fallback wedge — V7's 90° bend at BOTH Tier 2 and Tier 3, V8's
mixed-flow bend at Tier 3. Landed with those tiers as `xfail` tripwires
(Tier-1 meanline + V8 Tier-2 carry the realistic-loss validation); cracking the
wedge (closure-in-Newton / compact-support streamline fit) is the standing #1
open item, separate from this loss.

## Residual (tip-clearance / disk-friction)

Still deferred and **[VERIFY]** — a per-streamtube closure lacks their inputs:
**tip-clearance** (Jansen 1967 needs exit blade width `b2` + hub/tip radii, absent
from the §4.1 contract) and **disk-friction/windage** (machine-level parasitic
`~ρ2 U2³ r2²/ṁ`, no `ṁ` in a per-streamtube model); recirculation/leakage likewise.
Incidence + skin friction remain form-verified with both convention `[DECIDE]`s
resolved to Aungier (2000). No bug found here (contrast LIEB59).
