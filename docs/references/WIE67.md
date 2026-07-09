# WIE67 — Wiesner (1967) slip factor: verified form + constants

Source: Wiesner, F.J., "A Review of Slip Factors for Centrifugal Impellers,"
ASME J. Eng. Power **89** (1967). The 1967 paper itself is not in the
NotebookLM library, but its correlation is quoted verbatim by six standard
texts that are: **Aungier (2000)**, **Van den Braembussche (2020)**, **Cumpsty
(1989)**, **Dixon (2010)**, **Lakshminarayana (1996)**, **Whitfield & Baines
(1990)** — cross-agreeing, which is stronger than a single transcription.

**Provenance.** Extracted 2026-07-09 from the user's NotebookLM "Staging Area
(Theory)" notebook (source-grounded, citation-backed). Cross-checked term-by-
term against `slcflow/closures/centrifugal/wiesner.py`.

## Confirmed — code matches the source

| Quantity | Source | Code | Status |
|----------|--------|------|--------|
| Slip factor | `σ = 1 − √(cos β2b)/Z^0.7` (Braembussche 3.82; Cumpsty 6.10 `√cos χ2 /N^0.7`; Dixon 7.35b) | `:61` `1 − sqrt(cos(b))/Z**0.7` | ✅ √cos, exponent **0.7**, no leading coefficient. |
| Angle reference | Standard Wiesner/"American literature" references β2b **from the radial** direction; Aungier writes `sin` only because he references from *tangent* (sin(from-tangent) ≡ cos(from-radial)) | code's `beta2b` is from radial (`:10`, `:54`) → uses `cos` | ✅ convention consistent. |

Numerically pinned in `tests/test_wiesner_reference.py`.

## Findings — documented, not silently changed

1. **Radius-ratio limit correction is omitted in code.** Wiesner's
   correlation is only valid up to a limit inlet/outlet radius ratio; above it
   the slip factor is reduced. The library confirms both pieces:
   - Limit: `ε_lim = exp(−8.16·cos β2b / Z)` (from-radial form, **Cumpsty**,
     **Dixon 7.35c**). Aungier 4-8 writes `exp(−8.16·sin β2 / z)` — same thing
     in his tangent-referenced β. Constant **8.16** confirmed across sources.
   - Correction (Braembussche 3.84): `σ_corr = σ·[1 − ((r1/r2 − ε_lim)/
     (1 − ε_lim))³]` — cubic exponent **3** confirmed. (Aungier 4-10 uses a
     *different* exponent `β2/10`, not cubic — a source-to-source divergence to
     be aware of.)

   `wiesner.py` implements only the base `σ` and never applies the limit
   correction (the `WiesnerSlip.exit_rvt` closure doesn't read the inducer
   radius r1). Inactive when `r1/r2 < ε_lim` (typical designs sit near the
   limit — e.g. β2b=30°,Z=15 → ε_lim≈0.62; β2b=45°,Z=20 → ε_lim≈0.75), so it
   can bite at high radius ratio. Implementing it is a **geometry-contract
   addition** (needs r1) + behavior change → recorded here, `[DECIDE]` before
   adding, with the Braembussche cubic vs Aungier β2/10 choice to settle.

2. **Docstring sin/cos slip.** `wiesner.py:22–23`'s `[VERIFY]` note writes the
   limit as `exp(−8.16 sin(β2b)/Z)` while its β2b is defined from the *radial*
   (`:10`). With from-radial β2b the correct form is `cos` (Cumpsty/Dixon);
   the note borrowed Aungier's tangent-referenced `sin`. Corrected in the note
   text (doc-only; no runtime effect since the correction is unimplemented).

3. **von Backström alternative not in the library.** The code (`:24`) records
   the von Backström single-parameter model as a `[VERIFY]` option; it does not
   appear in these sources (they cover Stodola, Busemann, Stanitz, Wiesner).
   Not actionable from this library.

## Nothing else outstanding

The base Wiesner form is fully verified. The only residual is the deliberate
`[DECIDE]` on whether to add the radius-ratio limit correction (finding 1),
which is a modeling-scope choice, not an unverified constant.
