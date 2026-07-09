# LIEB59 — Lieblein (1959) diffusion-factor profile loss: verified + one bug

Source: Lieblein's equivalent-diffusion-ratio profile-loss chain as given in
**Aungier** *Axial-Flow Compressors* ch. 6, **Cumpsty** *Compressor
Aerodynamics*, and **Dixon** *Fluid Mechanics and Thermodynamics of
Turbomachinery* — all in the NotebookLM "Staging Area (Theory)" notebook,
cross-agreeing with equation numbers. Extracted 2026-07-09, source-grounded.
Cross-checked against `slcflow/closures/axial_compressor/loss.py`.

## Confirmed — code matches the source

| Quantity | Source eq. | Code | Status |
|----------|-----------|------|--------|
| Equivalent diffusion ratio `D_eq` | Aungier 6-36 / Dixon 3.40: `(W1/W2)[1.12 + 0.61(cos²β1/σ)(tanβ1−tanβ2)]` | `equivalent_diffusion` `:58` | ✅ 1.12, 0.61; W1/W2 correct here |
| Wake momentum thickness `θ*/c` | Dixon 3.37: `0.004/(1 − 1.17 ln D_eq)` | `wake_momentum_thickness` `:75` | ✅ 0.004, 1.17; diverges at D_eq→2.35 (code ceils at 2.2) |
| Loss reference dynamic head | inlet relative dynamic head | docstring `:18`, `delta_s_compressor_omega_bar` | ✅ |
| ω̄ prefactor | `2·(θ*/c)·(σ/cos β2)·…` | `:118` | ✅ factor 2, σ/cos β2 |

Confirmed pieces pinned in `tests/test_lieblein_loss_reference.py`.

## BUG (confirmed, high priority; fix deferred to the consolidation pass)

**The ω̄ velocity-ratio factor is inverted.** Aungier Eq 6-27 and Cumpsty Eq
1.32 both give

    ω̄ = 2·(θ*/c)·(σ/cos β2)·(W2/W1)²

(NotebookLM, verbatim, two independent sources: *"the correct velocity-ratio
factor is (W2/W1)², which is the inverse of the term in your query"*; for
constant axial velocity `(W2/W1)² = (cos β1/cos β2)²`). The code (`loss.py:118-
119`) and its own docstring (`:9`) use **`(W1/W2)²`** — the reciprocal.

For a compressor the flow diffuses (`W2 < W1`, so `W1/W2 > 1`), so this is not
a small error: `code/correct = (W1/W2)⁴`. At a representative `W2/W1 ≈ 0.7`
(DF≈0.45) the coded profile loss is **~4× too high**. It is a transcription
error against the cited sources (docstring and code agree with each other but
not the source), not a modeling choice.

**Why not fixed here:** per the plan to consolidate all sources before
resolving, and because a ~4× loss change (i) shifts every V4/V5 result and
(ii) may interact with the M4-tuned `_WBAR_CEIL = 0.5` ceiling and the 10°
bucket width (which could have been calibrated against the inflated loss).
Fixing it wants a paired re-look at the whole compressor-loss calibration and
a V5 re-measurement, done in the resolution pass. Flagged `[BUG]` at the code
site.

## Findings — modeling differences (documented)

1. **Off-design model differs.** The code applies a **quadratic ω̄ bucket**
   `ω̄ = ω̄_min(1 + ((i−i_ref)/w_bucket)²)` with `w_bucket = 10°` (its own
   choice). Lieblein's published off-design instead extends **`D_eq`** with an
   incidence term `+k(i−i*)^1.43` inside the bracket (Aungier 6-38 / Dixon
   3.41; `k = 0.0117` NACA-65, `0.007` C.4; exponent 1.43), and the loss
   doubles at the positive-stall incidence `i_s` (half-width `i_s − i_m`). The
   quadratic bucket + 10° width are unverified modeling substitutions. `[DECIDE]`
2. **Alternative θ*/c fit.** Aungier 6-37 gives a direct polynomial
   `2σω̄*cosβ2*(W1/W2)² = 0.004[1 + 3.1(D_eq−1)² + 0.4(D_eq−1)⁸]` as an
   alternative to the Dixon 3.37 log form the code uses. Both are legitimate
   Lieblein forms; noting the alternative, no action.

## Residual

Constants confirmed; the ω̄ inversion and the off-design substitutions are the
open items, both deferred to the consolidated resolution pass.
