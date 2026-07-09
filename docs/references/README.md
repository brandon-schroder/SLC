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

## Public-domain — acquired, URLs verified (2026-07-08)

| Key | Document | Source URL | Discharges (topic → tag sites) | Verification status |
|-----|----------|-----------|-------------------------------|---------------------|
| **SP-36** | Johnsen & Bullock (eds.), *Aerodynamic Design of Axial-Flow Compressors*, NASA SP-36 (1965) | [archive.org item](https://archive.org/details/NASA_NTRS_Archive_19650013744) · [PDF 252 MB](https://archive.org/download/NASA_NTRS_Archive_19650013744/NASA_NTRS_Archive_19650013744.pdf) · [OCR txt](https://archive.org/stream/NASA_NTRS_Archive_19650013744/NASA_NTRS_Archive_19650013744_djvu.txt) | Compressor incidence/deviation/loss **chart outputs** for `lieblein.py` (`:8`,`:130` `[VERIFY others]`), `axial_compressor/loss.py` D_eq/θ*/c and the `[VERIFY range]`s; the V5 reference-figure reproduction points (`v5_axial_compressor.py:14,18,73`). | **Outputs only** — validates fit predictions at chart points; cannot validate Aungier's fit coefficients. Chart points require reading the figures (raster; not fetch-extractable). |
| **AM51** | Ainley & Mathieson, *A Method of Performance Estimation for Axial-Flow Turbines*, ARC R&M 2974 (1951) | [Cranfield aerade PDF](https://reports.aerade.cranfield.ac.uk/bitstream/handle/1826.2/3538/arc-rm-2974.pdf) — vendored offline as [`AM51_arc-rm-2974.pdf`](AM51_arc-rm-2974.pdf) (2 MB, 32 pp) | Turbine exit-angle law `α2 = arccos(o/s)` and its geometry contract in `axial_turbine/ainley.py` (`:13` deviation correction, `:47`,`:55` `[VERIFY range]`/ceiling); the AM baseline the K-O profile/secondary losses build on (`kacker_okapuu.py`). | **Acquired but scanned** (no text layer; local copy in scratch `tool-results/`). Exit-angle law + loss framework are readable eyes-on-page; not yet transcribed. The K-O *coefficients* are 1982-era, **not** in AM51 → need K-O (below). |

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

Large PDFs (SP-36 is 252 MB) stay at their archive URLs. The 2 MB AM51 scan
is vendored offline (`AM51_arc-rm-2974.pdf`); everything else is a link.
