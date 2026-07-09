# Reference library index

Cross-links the Theory Manual §11 primary references to (a) an acquisition
status + source URL, and (b) the specific `[VERIFY]` tags in the code that
each reference is needed to discharge. This is the "map to library" §11 asks
for, grounded in what has actually been retrieved rather than a wish-list.

**Scope of a discharge.** A `[VERIFY]` tag is only cleared when a coded value
has been checked *term-by-term* against the authoritative source, and a test
anchor (reference-figure reproduction point, §-cited) records the check. None
are cleared yet — this index is the acquisition + planning layer. See the
"Verification status" column for how close each source gets.

## A hard split you must respect

The coded correlation **coefficients** and their **originating charts/tables**
often live in *different* documents:

- The axial-compressor incidence/deviation/loss fits in
  `closures/axial_compressor/` are **Aungier's analytic curve-fits** to the
  SP-36 cascade charts, not SP-36 formulas. **SP-36 can validate the fit
  _outputs_** at chart points (feed β1, σ, t/c → read (i0)10, (δ0)10, m off
  the figures → compare to the fit). **Only Aungier's book validates the fit
  _coefficients_** (0.914, s³/160, the K_ti/K_td forms, …). The tag at
  `lieblein.py:8` therefore needs *both* sources, and the two do different
  jobs. Same pattern for the K-O and Wiesner coded forms.

## Tooling note (why some of this is on you)

Retrieval here used web fetch + a small extraction model. It works for
machine-readable text but **fails on scanned PDFs with no text layer**
(Ainley R&M 2974 is CCITT raster; the SP-36 OCR is 526 pages and the
extractor only samples it). Chart *values* are in raster figures regardless
of OCR quality. Net: the public-domain **documents are acquired and their
identity/URLs verified**, but pulling exact coefficients/chart-points out of
them is an eyes-on-page (or book-in-hand) task, not something the fetch
pipeline can finish. Coefficient-level calibration of the paywalled-source
correlations needs you to supply a readable copy.

## NotebookLM source library (primary calibration route, 2026-07-09)

The reference corpus lives in the user's NotebookLM notebooks (many sources
marker-converted to cleaned markdown, manually structure-checked). Queried
via the `notebooklm` skill for **source-grounded, citation-backed** extraction
— this is what actually discharges coefficient-level tags (the web-fetch route
below cannot read scanned PDFs). Relevant notebooks:

| Notebook | Sources | Holds |
|----------|---------|-------|
| Staging Area (Loss Models) | 28 | K-O 1982, Dunham-Came, Ainley-Mathieson, Zhu-Sjolander, Benner, Aungier — **marker-cleaned** (use this over the older one) |
| Turbomachinery: Empirical Loss Models | 26 | same lineage, pre-cleanup (some garbled OCR) |
| Staging Area (Theory) / Turbomachinery: Fundamental Theory | 37 / 13 | SLC formulation, radial equilibrium, throughflow theory |
| Staging Area (Reduced Order) / Reduced-Order Aerodynamic Solvers | 36 / 30 | Novak, Wilkinson, Denton, method papers |
| OTAC | 6 | NASA object-oriented throughflow code |

**Discharged via this route:**
- **KO82 (Kacker-Okapuu) — scalar formula constants CONFIRMED**, see
  [`KO82.md`](KO82.md) and `tests/test_kacker_okapuu_reference.py`. Includes
  the TE `Y_TE = ζ/(1−ζ)` mapping (confirmed as the M2→0 limit of the exact
  compressible K-O relation). The negative-incidence interpolation weight is
  **fixed** (resolution pass): AM-1957's symmetric `(b1/b2)^2` → KO82's signed
  `|b1/b2|(b1/b2)` — behavior-preserving for `b1≥0` (all in-domain cases;
  V6 runs `r∈[0.04,0.72]`), with a C¹ positivity floor for deep-negative
  extrapolation. Residual `[VERIFY]`: the nozzle/impulse + TE **chart**
  reference curves (need figure digitization).
- **CONV-B (Appendix-B loss→entropy definitions) — all CONFIRMED**, see
  [`CONV-B.md`](CONV-B.md) and `tests/test_conversions_reference.py`. The
  foundational layer: master `Δs=−R ln(p02/p01)` (Denton 4a), compressor `ω̄`
  on the **inlet** dynamic head (Cumpsty/Dixon 3.5), turbine `Y` on the
  **exit** dynamic head (Aungier), KE `ζ` (Denton), and the
  compressor-inlet/turbine-exit convention — all verbatim. No bug. One benign
  nuance: B.4 `ζ` uses actual vs Denton's ideal exit KE (`ζ` is only the K-O
  TE term, mapped to `Y` before summing).
- **AM-ANGLE (Ainley turbine exit angle) — throat rule CONFIRMED**, see
  [`AM-ANGLE.md`](AM-ANGLE.md) and `tests/test_ainley_reference.py`. The
  coded `α2 = arccos(o/s)` is AM's M2=1 gauge angle (Eq 2) — the correct
  sonic asymptote. The deferred low-speed correction is now precisely pinned:
  `α2 = α2* − 4(s/e)` (AM Eq 1) + linear M2∈[0.5,1.0] blend (needs exit Mach
  + back-surface `e`). No bug; sign handling consistent with the `orientation_te`
  audit fix.
- **CENT-LOSS (centrifugal incidence + skin friction) — both forms
  CONFIRMED**, see [`CENT-LOSS.md`](CENT-LOSS.md) and
  `tests/test_centrifugal_loss_reference.py`. Incidence `½(ΔWθ)²` = Galvas
  Eq 5.6; skin-friction leading `2·Cf` = Galvas `4·Cf·W²/2`; `Cf=0.005` =
  Braembussche typical. No bug. Two `[DECIDE]`s: incidence `f_inc` (code uses
  1.0; Conrad 0.5–0.7 / Aungier 0.8) and `W_avg²` (code square-of-mean vs
  Aungier mean-of-squares).
- **LIEB59 (Lieblein compressor profile loss) — constants CONFIRMED, ω̄ bug
  FIXED**, see [`LIEB59.md`](LIEB59.md) and
  `tests/test_lieblein_loss_reference.py`. `D_eq` (1.12, 0.61) and `θ*/c`
  (0.004, 1.17) verified vs Aungier/Cumpsty/Dixon. The ω̄ velocity-ratio
  inversion (code `(W1/W2)²` vs source `(W2/W1)²`, ~4× overestimate) is
  **fixed** (resolution pass) — extracted to `profile_loss_coefficient`,
  V5 bands + full suite green. The **off-design model** is also **resolved**:
  the fixed-10° quadratic bucket → Aungier's normalized-incidence bucket
  (`1 + ξ²` with physically-derived asymmetric stall/choke ranges `R_s`/`R_c`,
  C1-matched deep-stall linear branches; min-loss `ω̄` now evaluated at the
  reference triangle so the bucket is the sole off-design mechanism — no D_eq
  double-count). Note: the `(i−i*)^1.43` term is Aungier's **surface-velocity**
  term, not the loss bucket (a common conflation). Deferred `[VERIFY]`: the
  Mach adjustment of `R_s`/`R_c`.
- **GC86 (Gallimore-Cumpsty mixing) — form CONFIRMED, `c_mix` RESOLVED**,
  see [`GC86.md`](GC86.md). The turbulent-diffusion form is right, but the old
  `c_mix=0.01` did NOT match G-C: they recommend `ε/(V_z·L_s) ≈ 1.8e-3` on the
  axial *stage length*, whereas the code nondimensionalizes on *radius* —
  reconciled, a G-C-consistent value is ~`5e-4` (the old default was ~10–50×
  too strong). **Fixed (resolution pass, option B): default `0.01 → 5e-4`**,
  keeping the r-based form (`stage` is ill-defined in the q-o march, so option
  A's `L_s` re-base was declined). **Paired V5 re-measure refutes the M8
  homogenization claim**: at the honest coefficient (and after the V5 annulus
  retune that puts the loss in-window) mixing shaves only ~14–18% off the exit
  `Δs` spread (2/3/4 stages) and does not catch up as stratification grows —
  the old "~25×" was the compounded artifact of the inflated Lieblein loss +
  the 20×-strong `c_mix`. C.5m + test revised.
  Primary G-C source is in the **"Reduced-Order Aerodynamic Solvers"**
  notebook; Wisler-1987 evaluation in the loss notebook.
- **AUN-C (Aungier compressor fits) — all incidence/deviation coefficients
  CONFIRMED**, see [`AUN-C.md`](AUN-C.md) and
  `tests/test_lieblein_reference.py`. Nine fits verified verbatim vs Aungier
  ch. 6 (the SP-36 *coefficient* half). Found + **FIXED a real bug**: the
  `K_ti` thickness exponent had an extra ×10 (`(10 t/c)^0.3` vs Aungier Eq
  6-11 `(t/c)^0.3`). Residual: SP-36 chart-point reproduction of the fit
  outputs; the `loss.py` (θ*/c, D_eq) side is a separate pass.
- **WIE67 (Wiesner slip) — base form CONFIRMED**, see [`WIE67.md`](WIE67.md)
  and `tests/test_wiesner_reference.py` (cross-agreeing across six texts).
  `σ = 1 − √(cos β2b)/Z^0.7`, β2b from radial, exponent 0.7 — verified. Found:
  the code omits the radius-ratio limit correction (`ε=exp(−8.16 cos β2b/Z)`
  + cubic reduction) — `[DECIDE]`, off-limit only; von Backström not in the
  library; a sin/cos docstring slip corrected. NB the Wiesner source lives in
  the **"Staging Area (Theory)"** notebook (centrifugal texts), not the loss
  notebook.

Query mechanics (for the next pass): NotebookLM persists chat server-side and
will anchor on it — **clear the chat between topics** (`scripts/clear_chat.py`,
added) or answers regurgitate the prior question. The skill's answer-scraper
was also patched to baseline answer-node count (else it returns the stale
persisted answer). Both fixes are in the skill dir, not this repo.

## Public-domain — acquired, URLs verified (2026-07-08)

| Key | Document | Source URL | Discharges (topic → tag sites) | Verification status |
|-----|----------|-----------|-------------------------------|---------------------|
| **SP-36** | Johnsen & Bullock (eds.), *Aerodynamic Design of Axial-Flow Compressors*, NASA SP-36 (1965) | [archive.org item](https://archive.org/details/NASA_NTRS_Archive_19650013744) · [PDF 252 MB](https://archive.org/download/NASA_NTRS_Archive_19650013744/NASA_NTRS_Archive_19650013744.pdf) · [OCR txt](https://archive.org/stream/NASA_NTRS_Archive_19650013744/NASA_NTRS_Archive_19650013744_djvu.txt) | Compressor incidence/deviation/loss **chart outputs** for `lieblein.py` (`:8`,`:130` `[VERIFY others]`), `axial_compressor/loss.py` D_eq/θ*/c and the `[VERIFY range]`s; the V5 reference-figure reproduction points (`v5_axial_compressor.py:14,18,73`). | **Outputs only** — validates fit predictions at chart points; cannot validate Aungier's fit coefficients. Chart points require reading the figures (raster; not fetch-extractable). |
| **AM51** | Ainley & Mathieson, *A Method of Performance Estimation for Axial-Flow Turbines*, ARC R&M 2974 (1951) | [Cranfield aerade PDF](https://reports.aerade.cranfield.ac.uk/bitstream/handle/1826.2/3538/arc-rm-2974.pdf) (link only — scanned raster, no text layer) | Turbine exit-angle law `α2 = arccos(o/s)` and its geometry contract in `axial_turbine/ainley.py`; the AM baseline the K-O profile/secondary losses build on (`kacker_okapuu.py`). | **VERIFIED via the notebook library** (the AM content is quoted verbatim by the theory texts) — see [`AM-ANGLE.md`](AM-ANGLE.md). The scanned PDF was not needed; not vendored. K-O *coefficients* are 1982-era, **not** in AM51 → see KO82. |

## Paywalled — need a readable copy from you (ASME / books)

Each of these is the authoritative source for a live `[VERIFY]` and cannot be
pulled from a public archive. Ordered by payoff (self-contained
correlation+data first).

| Key | Document | Discharges | Why it's the blocker |
|-----|----------|-----------|----------------------|
| **KO82** | Kacker & Okapuu, "A Mean Line Prediction Method for Axial Flow Turbine Efficiency," ASME J. Eng. Power 104 (1982) | Every coefficient in `axial_turbine/kacker_okapuu.py` (`:16`,`:42`,`:49`,`:67`,`:80`,`:98`,`:111`,`:130`,`:140`,`:148`(K_s),`:174`,`:193`(shock 0.75/1.75),`:206`); K-O ties into `loss.py` TE `[VERIFY]`. | Single self-contained paper: profile/secondary/TE/shock correlations **and** the reference turbine set — clears the whole V6 wall (`v6_axial_turbine.py:19,21,72`) at once. **Recommended first.** |
| **DC70** | Dunham & Came, "Improvements to the Ainley-Mathieson Method…," ASME J. Eng. Power 92 (1970) | The AM→K-O secondary-loss update K-O builds on; `kacker_okapuu.py` aspect-ratio/secondary forms. | The bridge between AM51 (public) and KO82; needed to confirm the secondary-loss lineage. |
| **WIE67** | Wiesner, "A Review of Slip Factors for Centrifugal Impellers," ASME J. Eng. Power 89 (1967) | `centrifugal/wiesner.py` coefficient/exponent (`:21`,`:24`,`:52`) — the `1 − √(cos β2b)/Z^0.7` form and its low-solidity limit. | Only source for the slip coefficient/exponent and the correction's validity range; core of V7 (`v7_centrifugal.py:19,20`). |
| **ECK** | Eckardt, centrifugal impeller measurements (rotor "O"/"A"/"B"; ASME J. Fluids Eng. 1976 / 1980) | V7 point-by-point reproduction (`v7_centrifugal.py`); the impeller loss-model calibration in `centrifugal/loss.py` (`:21`,`:47`,`:55`). | The standard radial-compressor validation dataset; without it V7 stays structural-only. |
| **AUN-C** | Aungier, *Axial-Flow Compressors* (2003), ch. 6 | The **fit coefficients** for `lieblein.py` + `axial_compressor/loss.py` (the other half of the SP-36 split above). | The literal source the coded fits were transcribed from; needed to check every constant. |
| **AUN-T / AUN-R** | Aungier, *Turbine Aerodynamics* / *Centrifugal Compressors* | Cross-checks for the turbine throat/loss and centrifugal forms; A.8 in-blade lean model (`theory_manual.md` A.6, A.8 `[VERIFY]`). | Aungier's unified SLC + in-blade treatment is one of the two named A.8 cross-checks (with Denton). |
| **GC86** | Gallimore & Cumpsty, "Spanwise Mixing in Multistage Axial Flow Compressors," ASME J. Turbomach. (1986) | `transport/mixing.py` diffusivity form + `c_mix` (`:23`,`:148`); the M8 `c_mix` calibration carry-over. | Source for the `μ_mix = c_mix·ρ·Vm·r` form and a defensible `c_mix` (coded 0.01, flagged `[VERIFY]`). |

## Workflow to actually discharge a tag

1. Get the source readable (public URL above, or your copy for paywalled).
2. Transcribe the relevant formula/chart-points into a short note under
   `docs/references/<key>.md` (verbatim + our-convention mapping per Appendix
   A.6 for anything sign-bearing).
3. Add/point a test anchor that reproduces a reference figure point, `§`-cited
   per CLAUDE.md process discipline.
4. Flip the code `[VERIFY]` to a citation of the note, and update the
   Theory-Manual §11 / Appendix-C provenance line.

No source binaries are vendored — every reference is a link (or, for the
paywalled set, a NotebookLM notebook). The AM51 scan was briefly vendored then
removed: its content is verified via the notebook library (AM-ANGLE.md), and a
32-page raster with no text layer was out of step with the markdown-note
format of the rest.
