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
  SP-36 cascade charts, not SP-36 formulas. **SP-36 validates the fit
  _outputs_** at chart points (feed β1, σ, t/c → read (i0)10, (δ0)10, n, m off
  the figures → compare to the fit) — **DONE 2026-07** (Figs 137/138/161/162,
  RMS ~0.1–0.2°, no bug; see the AUN-C bullet below). **Only Aungier's book
  validates the fit _coefficients_** (0.914, s³/160, the K_ti/K_td forms, …),
  also done. The tag at `lieblein.py:8` needed *both* sources, doing different
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
  extrapolation. **The nozzle/impulse profile-loss curves are now DIGITIZED**
  (resolution pass): `yp1`/`yp2` calibrated to Ainley-Mathieson R&M 2974 Fig. 4
  points (`tools/digitize_am_fig4.py`; u⁴ level law, <0.003 in Y_p), which also
  **surfaced a real bug** — the positivity-floor `smooth_max` width was
  angle-scaled (`_R_W=0.1`), inflating every profile-Y by ~0.037; fixed to a
  loss-scaled 0.003. Residual `[VERIFY]`: the TE `φ²` + `K_p` `K1` Mach curves
  (their figures are in the paywalled K-O paper, not the library).
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
  Braembussche typical. No bug. Both `[DECIDE]`s **resolved to Aungier (2000)**
  (resolution pass): incidence `f_inc` → tunable field, default `0.8` (was full
  KE 1.0; a genuine 0.5–1.0 family, so tunable not fixed); skin-friction mean
  velocity → mean-of-squares `½(W1²+W2²)` (was square-of-mean; friction ∝ local
  `W²`). V7 stays in-band (PR 2.43, η 0.974).
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
  term, not the loss bucket (a common conflation). **`θ*/c` fit-output chart
  validation now DONE** (2026-07-10): the primary paper (Lieblein 1959, ASME,
  in the user's Drive) **Fig. 6** was digitized — the coded curve rides the
  published dashed EQUATION-[8] line and the data cloud, max |coded − chart|
  = 0.0003, **clean, no bug**; it also pins the validity window (data DR ≈
  1.15–2.25) and the 2.35 divergence limit (`tools/digitize_lieblein_loss.py`,
  `test_wake_momentum_thickness_matches_lieblein_fig6`). Deferred `[VERIFY]`:
  the Mach adjustment of `R_s`/`R_c`.
- **HOWELL (axial-compressor endwall + tip-clearance loss) — ADDED, verified**,
  see [`HOWELL.md`](HOWELL.md) and `tests/test_lieblein_loss_reference.py`. The
  profile-only axial-compressor set now carries the deferred endwall/clearance
  physics (the `__init__` "recorded deferral"). Howell's additive drag model was
  chosen over Aungier's (Aungier folds endwall into the profile correlation via
  K1/K2 + charts Fig 6-11/6-12 — not clean-additive; §7.1 permits either).
  Verified verbatim vs Dixon/Howell/Saravanamuttoo/Cumpsty: `tan β_m` (Dixon
  3.15), `C_L` (Dixon 3.26a), `C_Ds = 0.018 C_L²` + `C_Da = 0.020 s/h` (Howell
  p.451), `C_Dk = 0.7 C_L² t/h` (Lakshminarayana via Cumpsty), and the drag→loss
  conversion `ζ = σ(cos²β1/cos³β_m)C_D` (Cumpsty 4.9, derived from first
  principles to disambiguate the OCR). All inlet-referenced → one B.2 conversion.
  Measured: V5 rotor η ~0.96 → ~0.92 (realistic), PR ~unchanged. Deferred
  `[VERIFY]`/`[DECIDE]`: off-design `C_L` (actual triangle), secondary/clearance
  overlap, the `C_L` validity ceiling, compressor shock loss (transonic V5).
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
  6-11 `(t/c)^0.3`). **SP-36 fit-output reproduction now DONE** (resolution
  pass): the original NASA SP-36 was obtained (NTRS 19650013744) and Figs
  137/138/161/162 digitized — `(i0)_10` RMS 0.10°, `(δ0)_10` RMS 0.17°, n/m
  overlay-coincident, no bug (`tools/digitize_sp36.py`,
  `tests/test_lieblein_sp36_charts.py`). The `loss.py` (θ*/c, D_eq) side was a
  separate pass, now **also DONE** (Lieblein 1959 Fig. 6 — see LIEB59 above).
- **WIE67 (Wiesner slip) — base form CONFIRMED**, see [`WIE67.md`](WIE67.md)
  and `tests/test_wiesner_reference.py` (cross-agreeing across six texts).
  `σ = 1 − √(cos β2b)/Z^0.7`, β2b from radial, exponent 0.7 — verified. The
  radius-ratio limit correction (`ε=exp(−8.16 cos β2b/Z)` + Braembussche cubic)
  is now **implemented** (RESOLVED 2026-07): the cube adopted over Aungier's
  `β2/10` on 3-source consensus; `r1/r2 = flow.r/flow.r_te` from the flow view
  (no geometry-contract change); off-limit only, so V7 stays in-band. von
  Backström not in the library; a sin/cos docstring slip corrected. NB the
  Wiesner source lives in the **"Staging Area (Theory)"** notebook (centrifugal
  texts), not the loss notebook.

Query mechanics (for the next pass): NotebookLM persists chat server-side and
will anchor on it — **clear the chat between topics** (`scripts/clear_chat.py`,
added) or answers regurgitate the prior question. The skill's answer-scraper
was also patched to baseline answer-node count (else it returns the stale
persisted answer). Both fixes are in the skill dir, not this repo.

## Public-domain — acquired, URLs verified (2026-07-08)

| Key | Document | Source URL | Discharges (topic → tag sites) | Verification status |
|-----|----------|-----------|-------------------------------|---------------------|
| **SP-36** | Johnsen & Bullock (eds.), *Aerodynamic Design of Axial-Flow Compressors*, NASA SP-36 (1965) | [archive.org item](https://archive.org/details/NASA_NTRS_Archive_19650013744) · [PDF 252 MB](https://archive.org/download/NASA_NTRS_Archive_19650013744/NASA_NTRS_Archive_19650013744.pdf) · [OCR txt](https://archive.org/stream/NASA_NTRS_Archive_19650013744/NASA_NTRS_Archive_19650013744_djvu.txt) | Compressor incidence/deviation/loss **chart outputs** for `lieblein.py` (`:8`,`:130` `[VERIFY others]`), `axial_compressor/loss.py` D_eq/θ*/c and the `[VERIFY range]`s; the V5 reference-figure reproduction points (`v5_axial_compressor.py:14,18,73`). | **Outputs only** — validates fit predictions at chart points; cannot validate Aungier's fit coefficients. Chart points require reading the figures (raster; not fetch-extractable). |
| **TR1368** | Herrig, Emery & Erwin, *Systematic Two-Dimensional Cascade Tests of NACA 65-Series Compressor Blades at Low Speeds*, NACA RM L51G31 / TR-1368 (1951) | [NTRS 19930092353](https://ntrs.nasa.gov/api/citations/19930092353/downloads/19930092353.pdf) | The PRIMARY 65-series cascade data under the whole `lieblein.py` chain; V4 cascade-level validation (`model-readiness` gate #1). | **Fig. 107 (design turning slope) digitized + pinned 2026-07-15** — validates `deviation_slope` at the raw-data level, RMS 0.030, β₁=70 low-σ documented deviation region (`tools/digitize_tr1368_fig107.py`, `tests/test_lieblein_tr1368_fig107.py`, [`TR1368.md`](TR1368.md)). Fig. 111 design-point cross-plots = recorded extension. |
| **ROTOR37** | Moore & Reid, *Performance of Single-Stage Axial-Flow Transonic Compressor… Design Pressure Ratio of 2.05*, NASA TP-1659 (1980) | [NTRS 19800012840](https://ntrs.nasa.gov/api/citations/19800012840/downloads/19800012840.pdf) | The V5 point-by-point transonic dataset (gate #1): geometry-faithful `verification/v5_rotor37.py` + measured 100%-speed rotor line. | **Landed 2026-07-15** — both tiers converge on the digitised geometry; measured agreement pinned (PR +12–16% = blockage + MCA deviation gap, η within ~1 pt, validity 0 at Rotor-37 loading, shallow speedline) — see [`ROTOR37.md`](ROTOR37.md), `tests/test_v5_rotor37.py`. Tables II/III(b)/V/VI + AGARD-AR-355 coords untranscribed. |
| **LS89** | Arts & Lambert de Rouvroit, *Aero-Thermal Performance of a 2-D Highly Loaded Transonic Turbine NGV* (VKI LS-89, TN-174/1992) | Test Cases notebook (paywalled primary) | V6 cascade-level measured-data check of `throat_exit_angle` + the K-O loss chain. | **Landed 2026-07-15** — exit angle = gauging to 0.1°; predicted energy-ζ 0.0303 vs measured 0.0225 at M2is=1 (+35%, documented K-O behaviour; TE curve carries most of it); K-O inlet-shock correctly inert, the measured 0.5% TE-shock is a recorded model boundary. [`LS89.md`](LS89.md), `tests/test_v6_ls89.py`. Fig. 16 Mach-trend digitization + TN D-6967 stage case = recorded extensions. |
| **TND6967** | Kofskey & Nusbaum, *Design and Cold-Air Investigation of a Turbine for a Small Low-Cost Turbofan Engine*, NASA TN D-6967 (1972) | [NTRS 19720024422](https://ntrs.nasa.gov/api/citations/19720024422/downloads/19720024422.pdf) | Machine-level V6 measured case: 4-row two-stage turbine, `verification/v6_tnd6967.py`. | **Landed 2026-07-16** — meanline η_tt 0.926 vs measured 0.93 (−0.4 pt, no clearance loss modelled); PR/work −12..17% at matched flow = a capacity gap (geometric throat, no blockage; near-sonic-by-design stator hubs); Tier-2 spanwise open (hub chokes). [`TND6967.md`](TND6967.md), `tests/test_v6_tnd6967.py`. |
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
| **ECK** | Eckardt, centrifugal impeller measurements (rotor "O"/"A"/"B"; ASME J. Fluids Eng. 1976 / 1980) — **1976 primary ACQUIRED (Test Cases notebook), see [`ECKARDT.md`](ECKARDT.md)** | V7 point-by-point reproduction; the impeller loss-model calibration in `centrifugal/loss.py`. | **Landed 2026-07-15:** exact geometry grounded verbatim (r1h 45/r1t 140/b2 26 mm, Z=20 radial, 130 mm, z_s/b2 0.027) → geometry-faithful `verification/v7_eckardt.py`; all three tiers converge (validity 1, ~0.1% tier agreement); laser-point PR +4.7% vs stage (diffuser-sized gap), design +12.6% (deferred parasitic/clearance), implied slip ~0.90 vs Wiesner 0.877. `tests/test_v7_eckardt.py`. Remaining: true Fig. 1 contours, Oh-1997 map digitization, A/B backswept, Krain. |
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
