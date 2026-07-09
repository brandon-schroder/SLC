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

## Findings — modeling choices (documented, not changed)

1. **Incidence factor `f_inc`.** The code uses the *full* NASA kinetic energy,
   `½(ΔWθ)²` (i.e. `f_inc = 1`). Several sets apply a reducing factor: Conrad
   et al. (1980) `k(ΔWθ²/2)` with `k = 0.5–0.7`; Aungier's total-pressure form
   (Eq 5-27) carries a leading `0.8`. Wasserbauer-Glassman (1975) use `sin³`
   for positive incidence (less loss there) and `sin²` for negative. So the
   coded value is the conservative upper bound of a 0.5–1.0 family. `[DECIDE]`
   whether to adopt an `f_inc`.

2. **Mean-velocity definition in skin friction.** The code forms
   `W_avg = ½(W1 + W2)` then squares → `[½(W1+W2)]²` (square of the mean).
   **Aungier (2000)** specifies the *mean of the squares*,
   `W̄² = ½(W1² + W2²)`. These differ (`[½(W1+W2)]² ≤ ½(W1²+W2²)`), the gap
   growing with the `W1/W2` diffusion. `[DECIDE]` — a one-line change if
   Aungier's convention is adopted.

3. **`L/D_hyd` is geometry-derived**, not a universal constant. The code's
   `l_over_dhyd = 4.0` is a representative scalar design input; Aungier builds
   `d_H` from throat/tip areas and `L_H` from the mean-camberline path length.
   Fine as a design input; a geometry-derived value is the recorded refinement.

## Residual

The two dominant components (incidence + skin friction) are form-verified.
The **deferred components** the docstring already lists — blade-loading
diffusion, tip-clearance, disk-friction/windage, recirculation, leakage — are
a separate V7-calibration extension (Oh-Yoon-Chung 1997 is the standard set),
not part of this pass. No bug found here (contrast LIEB59).
