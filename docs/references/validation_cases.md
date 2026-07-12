# Validation test-case acquisition plan (the "Validation Cases" notebook manifest)

The **datasets** slcflow needs to move each verification case (V4–V8) from
*structural* to *point-by-point quantitative* (the `model-readiness` gate #1).
Distinct from `docs/references/` (which lists the *correlation-source* documents):
this lists **test rigs / measured cases** — geometry **+** measured performance —
to reproduce.

Assembled 2026-07-12. Each entry is grounded in a web-verified primary reference;
**public-domain** ones have a direct URL you can drop straight into NotebookLM.

## Consolidated source list (DOIs / report IDs)

DOIs verified via Crossref (2026-07-12). **NASA/NACA/AGARD/DFVLR items are
reports that predate DOIs** — they are identified by report number + NTRS/DTIC
accession, not a DOI. ✅ = public-domain direct URL; 🔒 = paywalled (ASME/AGARD).

| # | Document | DOI / report ID | Access |
|---|----------|-----------------|--------|
| 1 | Herrig, Emery, Erwin & Felix, *Systematic 2-D Cascade Tests of NACA 65-Series Compressor Blades at Low Speeds*, NACA TR-1368 (1957) | no DOI — NTRS **19930092353** | ✅ |
| 2 | Moore & Reid, *Performance of Single-Stage Axial-Flow Transonic Compressor… PR 2.05* (Stage 37), NASA **TP-1659** (1980) | no DOI — NTRS **19800012840** | ✅ |
| 3 | *CFD Validation for Propulsion System Components*, **AGARD-AR-355** (1998) — Rotor 37 blade coords + LDA | no DOI — NATO STO / DTIC | 🔒 |
| 4 | Arts & Lambert de Rouvroit, *Aero-Thermal Performance of a 2-D Highly Loaded Transonic Turbine NGV* (VKI **LS-89**), J. Turbomach. (1992) | **10.1115/1.2927978** (conf 1990: 10.1115/90-gt-358); primary data VKI **TN-174** (1990) | 🔒 |
| 5 | *Design and Cold-Air Investigation of a Turbine…* (single-stage), NASA **TN D-6967** (1972) | no DOI — NTRS **19720024422** | ✅ |
| 6 | Eckardt, *Detailed Flow Investigations Within a High-Speed Centrifugal Compressor Impeller* (rotor **O**, radial), J. Fluids Eng. 98 (1976) | **10.1115/1.3448334** | 🔒 |
| 7 | Eckardt, *Flow Field Analysis of Radial and Backswept Centrifugal Compressor Impellers, Part I* (rotors **A/B**), ASME 1980 (New Orleans, pp. 77–86) | no DOI — ASME 1980 conf volume | 🔒 |
| 8 | Krain, *Swirling Impeller Flow*, J. Turbomach. 110 (1988) | **10.1115/1.3262157** | 🔒 |
| 9 | Krain & Hoffmann, *Verification of an Impeller Design by Laser Measurements…*, ASME 89-GT-159 (1989) | **10.1115/89-gt-159** | 🔒 |
| 10 | Moore & Reid, *…Stage 35, PR 1.82* (second axial point), NASA TP (1982) | no DOI — NTRS **19820014395** | ✅ |
| 11 | NASA centrifugal CR (McKain & Holbrook CC3-class; confirm on NTRS) | no DOI — NTRS **19940012913** (verify) | ✅ |

**Correlation-source primaries** (already consulted via NotebookLM; DOIs here for
archival/citable copies — see `README.md` for which `[VERIFY]` each discharges):
Oh, Yoon & Chung 1997 **10.1243/0957650971537231**; Kacker & Okapuu 1982
**10.1115/1.3227240**; Wiesner 1967 **10.1115/1.3616734**; Dunham & Came 1970
**10.1115/1.3445349**; Gallimore & Cumpsty 1986 Part I **10.1115/1.3262019** /
Part II **10.1115/1.3262009**. (Lieblein 1959 + Coppage WADC-TR-55-257 + Galvas
NASA TN D-7487 + the Aungier books have no journal DOI / are already in the Drive.)

## How to use this with NotebookLM

The `notebooklm` skill can *query* notebooks and *register* a notebook URL in its
local library, but it **cannot create a notebook or upload sources** (that is a
web-UI action). To build the dedicated notebook:

1. In NotebookLM, create a new notebook, e.g. **"SLC — Validation Cases"**.
2. Add each source below. NotebookLM accepts a **URL** as a source directly, so
   the public-domain NTRS/NACA PDFs (✅ below) can be pasted in as-is; the
   paywalled ASME papers (🔒) you upload once you have a copy.
3. Send me the notebook URL and I'll register it in the skill library
   (`notebook_manager.py add`) so I can query it during calibration.

Priority = payoff for slcflow's *current* gaps (transonic loss stack, the
centrifugal blade-loading calibration, the subsonic-stage and turbine-stage
gaps). Tackle **P1** first.

---

## Axial compressor → V4 / V5

### ⭐ P1 — NACA Report 1368 (65-series cascade data) ✅ public
Herrig, Emery, Erwin & Felix, *Systematic Two-Dimensional Cascade Tests of NACA
65-Series Compressor Blades at Low Speeds*, NACA TR-1368 / RM-L51G31 (1951).
- **Why:** this is the **primary data the Lieblein/Aungier deviation + loss
  correlations were fit to** — the exact closures in `closures/axial_compressor/`.
  It validates V4 at the *cascade* level (deviation, turning, wake loss vs camber,
  solidity, inlet angle) with no confounding rig effects. The cleanest possible
  check of the coded correlations.
- **Provides:** 2-D cascade geometry (65-series, design lift 0–2.7) + measured
  turning/deviation/loss over the incidence range.
- NTRS: <https://ntrs.nasa.gov/citations/19930092353> ·
  PDF <https://ntrs.nasa.gov/api/citations/19930092353/downloads/19930092353.pdf>

### ⭐ P1 — NASA Rotor 37 / Stage 37 (transonic rotor) ✅ public
Moore & Reid, *Performance of Single-Stage Axial-Flow Transonic Compressor…
Design Pressure Ratio 2.05*, NASA TP-1659 (1980); blind-test coords + LDA data in
**AGARD-AR-355** (*CFD Validation for Propulsion System Components*, 1998).
- **Why:** the canonical transonic axial-compressor case — **exercises the exact
  transonic machinery just built**: the §6.7 shock loss, the meridional-
  supersonic-branch driver, and the endwall/clearance loss. Full **radial
  blade-element profiles** → the first real test of Tier-2/Tier-3 *spanwise*
  behaviour (not just meanline).
- **Design point (verified):** PR 2.106, η 88.9%, ṁ 20.19 kg/s, 17 188.7 rpm,
  M₁,rel ≈ 1.4 (supersonic inlet → the shock/supersonic-branch case).
- NASA TP-1659: <https://ntrs.nasa.gov/api/citations/19800012840/downloads/19800012840.pdf>
- AGARD-AR-355: NATO STO / DTIC (search "AGARD-AR-355"); the ASME/IGTI-1994 blind
  test geometry + laser data. 🔒 (report PDF, not on NTRS).

### P2 — NASA Stage 35 (milder transonic, PR 1.82) ✅ public
Same report family (Reid & Moore, NASA TP-1337 / TP-1659 series). A second,
less-loaded operating line for a trend/second-point check once Rotor 37 lands.
- <https://ntrs.nasa.gov/api/citations/19820014395/downloads/19820014395.pdf>

> **Note on a *subsonic* stage.** slcflow's V5 is set up subsonic, but the
> best-documented open NASA stages (35/37) are transonic. A truly low-speed
> subsonic stage (e.g. a low-speed research compressor) has no clean public rig
> dataset in-hand — **recorded gap.** Rotor 37 is the pragmatic first axial target
> because it also unlocks the transonic stack; a subsonic point can come from
> Stage 35's lower-speed lines.

---

## Axial turbine → V6

### ⭐ P1 — VKI LS-89 transonic nozzle cascade 🔒 (VKI, freely cited)
Arts, Lambert de Rouvroit & Rutherford, *Aero-Thermal Investigation of a Highly
Loaded Transonic Linear Turbine Guide Vane Cascade*, VKI TN-174 (1990).
- **Why:** the standard transonic HP-turbine cascade — validates V6's
  `α₂ = arccos(o/s)` exit angle, the K-O profile loss, the Mach factor `K_p`, and
  the §C.9 **shock loss**, all at measured surface-isentropic-Mach points.
- **Provides (verified):** chord 67.647 mm, gap/chord 0.850, design M₂,is = 0.9,
  Re₂ 0.5–2.2×10⁶, M₂ 0.7–1.1, inlet Tu 1–6%; blade coords + surface pressure +
  heat transfer.
- Data widely re-tabulated in open papers (e.g. MDPI *Energies* LS89 databases);
  VKI TN-174 itself is the primary. Grab a re-tabulation PDF for the notebook.

### P2 — NASA cold-air single-stage turbine ✅ public
e.g. NASA TN D-6967 (*Design and Cold-Air Investigation of a Turbine…*, single
stage) — a stage-level η-map + blade geometry to validate V6 **end-to-end** (not
just the cascade).
- <https://ntrs.nasa.gov/api/citations/19720024422/downloads/19720024422.pdf>
- (Goldman's supersonic-turbine series, NTRS 19720005134, is a further transonic
  turbine option.)

---

## Centrifugal → V7

### ⭐ P1 — Eckardt O / A / B 🔒 (ASME — the missing primary)
Eckardt, *Detailed Flow Investigations Within a High-Speed Centrifugal Compressor
Impeller*, ASME J. Eng. Power (1976) [rotor O, radial]; Eckardt, ASME 80-GT-8
(1980) [backswept]. See **`ECKARDT.md`** for what is already grounded (Cumpsty)
and what these primaries would add (exact `r₁ₕ/b₂/Z`/blade angles + meridional
profiles → the geometry-faithful V7 the ~5–7% anchor is 5–7% short of).
- **Get these two ASME papers** — the single highest-value centrifugal acquisition.

### P2 — Krain impeller (30° backsweep, LDV) 🔒 / partial ✅
Krain, *Swirling Impeller Flow*, ASME J. Turbomach. 110 (1988); Krain &
Hoffmann. D₂ = 400 mm, U₂ = 468 m/s, ~30° backsweep, PR ≈ 4, 3-component LDV.
- **Why:** a second, better-designed backswept impeller (Cumpsty's Fig 6.6) — a
  cross-check that V7's slip + blade-loading loss generalise beyond Eckardt.
- Some DLR/NTRS coverage; SRV2-O is the transonic 6:1 variant (50 000 rpm) for a
  transonic-centrifugal stretch goal.

### P3 — NASA low-flow centrifugal (CC3 / McKain–Holbrook) ✅ public
Full geometry + performance for a modern centrifugal stage; NTRS
(e.g. 19940012913). Good for a third independent centrifugal point.
- <https://ntrs.nasa.gov/api/citations/19940012913/downloads/19940012913.pdf>

---

## Mixed-flow → V8  (recorded gap)

No canonical **open** mixed-flow compressor rig with geometry + maps is in hand.
Options, none clean: a NASA mixed-flow study, or marine/turbocharger cases. **Keep
V8 structural-only** until a dataset surfaces, or validate its centrifugal set via
V7 (same closures) and treat the partial-φ bend as geometry-verified only.

---

## Priority order to populate the notebook

1. **NACA 1368** ✅ — validates the V4 compressor correlations at source (easy, public).
2. **NASA Rotor 37 / TP-1659 + AGARD-AR-355** — unlocks the transonic stack + spanwise V5.
3. **VKI LS-89** — the V6 turbine cascade (exit angle + K-O + shock).
4. **Eckardt 1976 & 1980 (ASME)** 🔒 — the geometry-faithful V7 (see `ECKARDT.md`).
5. **NASA cold-air turbine TN** ✅ + **Krain** / **CC3** — second points per type.

✅ = public-domain, direct URL above (paste into NotebookLM as a source).
🔒 = paywalled (ASME/AGARD) — needs a copy you supply, as with the correlation
sources in `README.md`.
