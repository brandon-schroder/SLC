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

## BUG — FIXED (resolution pass, 2026-07)

**The ω̄ velocity-ratio factor was inverted.** Aungier Eq 6-27 and Cumpsty Eq
1.32 both give

    ω̄ = 2·(θ*/c)·(σ/cos β2)·(W2/W1)²

(NotebookLM, verbatim, two independent sources: *"the correct velocity-ratio
factor is (W2/W1)², which is the inverse of the term in your query"*; for
constant axial velocity `(W2/W1)² = (cos β1/cos β2)²`). The code and its own
docstring had used **`(W1/W2)²`** — the reciprocal. For a compressor `W2 < W1`,
so it overestimated profile loss by `(W1/W2)⁴` (~4× at DF≈0.45).

**Fix.** The ω̄ assembly was extracted to `profile_loss_coefficient(theta_c,
sigma, beta2, w1, w2)` using `(W2/W1)²`, and pinned in
`tests/test_lieblein_loss_reference.py::test_omega_bar_uses_W2_over_W1_squared`
(with a guard against silent regression to the inverted form). The change
lowers profile loss / raises efficiency. **Measured:** the V5 structural bands
stayed in range and the full suite is green — i.e. the M4-tuned `_WBAR_CEIL`
ceiling and 10° bucket were **not** calibrated against the inflated loss, so no
paired recalibration was needed. (The point-by-point V5 speedline is still
`[VERIFY]` on the chart-digitization work, unaffected by this.)

## Off-design model — RESOLVED (2026-07), transcribed from Aungier ch.6

The old code used a **fixed-10° quadratic bucket** `ω̄ = ω̄_min(1 +
((i−i_ref)/10)²)` — an unverified substitution. A source-grounded NotebookLM
extraction of Aungier's **complete** off-design model shows the quadratic
*shape* is actually correct (Aungier's near-design law IS `1 + ξ²`); the
infidelity was the fixed width and the missing deep-stall branches. Adopted the
faithful model (`stall_choke_ranges` + `off_design_bucket`).

**Normalized incidence** (asymmetric about the min-loss incidence `i_m`; at low
speed `i_m = i*` = the reference incidence):

    ξ = (i − i_m)/R_s   for i ≥ i_m   (positive-stall side)
    ξ = (i − i_m)/R_c   for i <  i_m   (choke side)

**Loss multiplier** (`w_s` = upstream-shock loss = 0 for the subsonic set, so
`ω̄ = ω̄_min · f`):

    f = 1 + ξ²          for −2 ≤ ξ ≤ 1
    f = 2 + 2(ξ − 1)    for ξ > 1     (deep positive stall; C1 at ξ=1)
    f = 5 − 4(ξ + 2)    for ξ < −2    (deep negative stall/choke; C1 at ξ=−2)

**Low-speed stall/choke incidence ranges** (degrees; `θ` = camber, `β1` = inlet
flow angle):

    R_s = 10.3 + (2.92 − β1/15.6) · θ/8.2
    R_c = 9.0  − (1 − (30/β1)^0.48) · θ/4.176
    i_s = i* + R_s/(1 + 0.5(K_sh M1′)³)   ;   i_c = i* − R_c/(1 + 0.5 M1²)
    i_m = i_c + (i_s − i_c)·R_c/(R_c + R_s)   [→ i_m = i* at low speed]

**Design loss** stays the code's existing Dixon-3.37 `θ*/c` chain evaluated at
the **reference** velocity triangle (so the bucket is the *sole* off-design
mechanism — no D_eq/bucket double-count; Aungier evaluates `ω̄_min` at `i_m`).

Pinned in `tests/test_lieblein_loss_reference.py`
(`test_stall_choke_ranges_match_aungier`, `test_off_design_bucket_is_aungier_piecewise`).

**Deferred refinements** (`[VERIFY]`): the Mach-number adjustment of `R_s`/`R_c`
(the `1 + 0.5 M²` / `1 + 0.5(K_sh M′)³` factors) — code uses the low-speed
ranges; and the `(i − i*)^1.43` term (α = 0.0117 NACA-65 / 0.007 C.4), which
Aungier applies to the **maximum surface velocity** `W_max/W1`, **not** the
loss bucket — a common conflation; it is not needed for the off-design loss.

## Other findings (documented)

- **Alternative θ*/c fit.** Aungier 6-37 gives a direct polynomial
  `2σω̄*cosβ2*(W1/W2)² = 0.0073[K_2 + 3.1(D_eq−1)² + 0.4(D_eq−1)⁸]` (K_1 = 0.0073)
  as an alternative to the Dixon 3.37 log form the code uses. Both are legitimate
  Lieblein forms; noting the alternative, no action.

## θ*/c fit output — DIGITIZED vs Lieblein Fig. 6 (2026-07-10), clean

The `θ*/c` **chart output** (not just the textbook constants) was validated
against Lieblein's own figure. The primary paper — S. Lieblein, "Loss and Stall
Analysis of Compressor Cascades," ASME *J. Basic Eng.* **81** (1959) 387–400 —
is in the user's Google Drive (`lieblein_loss_1959.pdf`). Its **Fig. 6**,
"Experimental variation of wake momentum thickness with suction-surface
diffusion ratio at minimum loss," plots `(θ/c)₂` vs the diffusion ratio
`V_max,s/V₂` for NACA 65-(A₁₀) and C.4 cascade data, with the **dashed curve
labelled "EQUATION [8] WITH k_s = 1.17 AND ε = 0.004"** — i.e. the coded
`0.004/(1 − 1.17 ln D_eq)`.

- **Method** (`tools/digitize_lieblein_loss.py`, rerunnable): rendered at 600
  dpi, axes calibrated off the tick labels (DR 1.0→2.4; θ/c 0→.05), the dashed
  curve read by column scan where it separates from the data markers. The
  coded curve was **overlaid on the chart image** (the decisive check) — it
  lands on the dashed EQUATION-[8] line and through the centre of the data
  cloud across the whole range. **Max |coded − chart| = 0.0003** (reading
  precision ~0.0006). **Clean validation, no bug** (like the SP-36
  incidence/deviation pass). Pinned:
  `tests/test_lieblein_loss_reference.py::test_wake_momentum_thickness_matches_lieblein_fig6`.
- **Provenance note.** Lieblein's independent variable in Fig. 6 is the *actual*
  suction-surface diffusion ratio `V_max,s/V₂`; `D_eq` is his **computable
  estimate** of that ratio (Eq. for `D_eq` from inlet/outlet conditions). The
  code substitutes `D_eq` for `V_max,s/V₂` in Eq. 8 — the standard usage.
- **Validity window.** The data span DR ≈ **1.15 to ≈ 2.25**; the code's
  compact-support calibration window `_DEQ_CAL = (1.0, 2.0)` is sound and
  slightly conservative on the upper end (safe — the curve steepens sharply and
  the fit gets sensitive past 2.0). Lieblein states the fit diverges at the
  **"limit V_max,s/V₂ = 2.35"** — exactly the code's denominator zero
  `e^(1/1.17) = 2.351`; the ceiling at 2.2 sits safely at the data edge. Pinned:
  `test_wake_momentum_thickness_diverges_at_lieblein_2p35_limit`.

## Diffusion factor `D` and the tip stall limit — NACA RM E53D01
## (grounded 2026-07-18, staging-loss-models notebook)

Distinct from the 1959 `D_eq` above: the *original* blade-loading measure of
**Lieblein, Schwenk & Broderick, "Diffusion Factor for Estimating Losses and
Limiting Blade Loadings in Axial-Flow Compressor Blade Elements," NACA RM
E53D01 (1953)** (the basis of NASA SP-36), in the relative frame

    D = 1 − W₂/W₁ + |ΔV_θ| / (2 σ W₁).

Limiting values (NotebookLM, loss-models notebook, RM E53D01 "Summary of
Results"):

- **2-D cascade** (NACA 65-series, design α): loss rises only slightly with
  `D` up to ≈ **0.6**, then a sharp rise.
- **Rotor hub / mean**: minimum-loss coefficient roughly flat to `D ≈
  0.55–0.6`.
- **Rotor TIP — more severe**: "marked and practically linear" loss rise from
  `D` as low as **0.30**; to hold blade-element η = 0.90 the tip design limit
  is `D ≲ **0.45**` (RM E54A28 quotes a tip design `D ≈ 0.4` for stage
  η = 0.90).

**Used as a post-solve stall diagnostic** (verification layer,
`v5_rotor37.tip_diffusion_factor`; not a live closure). Measured on the two
transonic siblings (Rotor 37/38, gate #5 follow-on, 2026-07-18): each rotor's
**measured stall sits at tip `D ≈ 0.6`** (R37 0.63 at 19.60 kg/s; R38 0.595 at
20.44) — ~0.15 above the tip *design* limit 0.45, i.e. the rigs run past design
loading to stall, landing at the cascade sharp-loss value. A `D_tip = 0.60`
stall threshold predicts both measured stalls within ~3% and, unlike the loss
set, **orders the siblings correctly** (the high-AR R38 reaches the limit at
higher flow → earlier stall). Pinned:
`test_v5_rotor38.py::test_tip_diffusion_factor_predicts_the_sibling_stall_differential`.
The graduation path to a live `solve_speedline` criterion (C¹-safe closure-layer
`D`) is recorded in the ROTOR37 gate #5 disposition.

## Residual

Constants confirmed; the ω̄ inversion (fixed), the off-design model (resolved,
Aungier), and the θ*/c fit-output chart validation (Fig. 6, clean) are closed.
Open [VERIFY]: the two deferred off-design refinements above (Mach adjustment of
`R_s`/`R_c`; the `(i−i*)^1.43` max-surface-velocity term — not a loss-bucket
term).
