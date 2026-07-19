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

1. **Radius-ratio limit correction — RESOLVED 2026-07 (implemented).**
   Wiesner's correlation is only valid up to a limit inlet/outlet radius ratio;
   above it the slip factor is reduced. The library confirms both pieces:
   - Limit: `ε_lim = exp(−8.16·cos β2b / Z)` (from-radial form, **Cumpsty**,
     **Dixon 7.35c**). Aungier 4-8 writes `exp(−8.16·sin β2 / z)` — same thing
     in his tangent-referenced β. Constant **8.16** confirmed across sources.
   - Correction (Braembussche 3.84): `σ_corr = σ·[1 − ((r1/r2 − ε_lim)/
     (1 − ε_lim))³]` — cubic exponent **3** confirmed. (Aungier 4-10 uses a
     *different* exponent `β2/10`, not cubic — a source-to-source divergence.)

   **Fork resolved to the Braembussche cubic** on 3-source consensus
   (Cumpsty/Dixon/Braembussche all cube) over Aungier's lone `β2/10`.
   **The "geometry-contract addition" fear was wrong:** `r1/r2` is already in
   the lagged flow view as `flow.r / flow.r_te` (LE inducer radius / TE exit
   radius), so `wiesner_slip` gained an optional `radius_ratio` argument and
   both call sites (`WiesnerSlip.exit_rvt`, `CentrifugalLoss.evaluate`) supply
   it — **no section 4.1 change**. C¹ per §7.3: a softplus positive part (~0
   below the limit, smooth ramp above) with a `smooth_min` cap keeping the
   multiplicative factor in `[0, 1]` (σ never negative, AD-10). Inactive when
   `r1/r2 < ε_lim` so in-limit designs are behavior-preserving; e.g. V7's hub/
   mean streamlines (rr≈0.32–0.52 < ε_lim≈0.68) are unchanged while the tip
   (rr≈0.72) sees a ~0.4% σ reduction — V7 PR 2.432 / η 0.974, still in-band.
   Pinned in `tests/test_wiesner_reference.py` (8.16 exponent, cubic vs β2/10,
   below-limit no-op, above-limit closed form, non-negativity).

2. **Docstring sin/cos slip.** `wiesner.py:22–23`'s `[VERIFY]` note writes the
   limit as `exp(−8.16 sin(β2b)/Z)` while its β2b is defined from the *radial*
   (`:10`). With from-radial β2b the correct form is `cos` (Cumpsty/Dixon);
   the note borrowed Aungier's tangent-referenced `sin`. Corrected in the note
   text (doc-only; no runtime effect since the correction is unimplemented).

3. **von Backström alternative not in the library.** The code (`:24`) records
   the von Backström single-parameter model as a `[VERIFY]` option; it does not
   appear in these sources (they cover Stodola, Busemann, Stanitz, Wiesner).
   Not actionable from this library.

## Case-level confirmation — Eckardt O (2026-07-19)

Beyond the form verification, Wiesner `σ = 0.877` is confirmed at the CASE
level against the Eckardt O rig (Z=20 radial). The recorded "Eckardt implies
σ ~0.90" was a stale pre-loss-stack `(PR, η)` inversion (= the Stanitz value
`1 − 0.63π/Z = 0.901`), REFUTED by grounding: the Eckardt 1976 paper states no
measured slip (jet/wake exit, wake ≈35% area/≈15% mass), and the literature
calls Wiesner the *better* Eckardt-O match than Stanitz. Measured with the
closed stage chain: PR_stage **2.091 (−0.4%) with Wiesner** vs **2.122 (+1.1%)
with Stanitz** — the higher slip worsens the comparison. No recalibration;
constants unchanged. See `docs/references/ECKARDT.md` "Slip disposition",
pinned in `test_v7_eckardt.py::test_wiesner_slip_is_the_better_match_for_eckardt_o`.

## Nothing else outstanding

The base Wiesner form is fully verified, the radius-ratio limit correction
(finding 1) is implemented (Braembussche cubic), and the Eckardt-O case-level
check confirms `σ = 0.877` (above). Nothing from this source remains open —
findings 2 (the doc sin/cos slip, doc-only) and 3 (von Backström not in the
library) needed no code change.
