# LSCC — NASA Low-Speed Centrifugal Compressor (NOT the high-speed CC3)

**Status (2026-07-19): data-availability characterization; NO case built.**

## The naming correction

`docs/references/validation_cases.md` recorded a "third centrifugal point"
as **"CC3 / McKain–Holbrook", NTRS 19940012913**. That NTRS ID is in fact
**NASA TM-4481** — Hathaway, Chriss, Wood & Strazisar, *Experimental and
Computational Investigation of the NASA Low-Speed Centrifugal Compressor
Flow Field* (1993): the **LSCC**, a *different* machine from the high-speed
4:1 CC3 (McKain & Holbrook coordinates, NASA CR-204134). The two were
conflated. Corrected in `validation_cases.md`.

## LSCC — what it is, grounded from TM-4481

A large-scale, **low-speed** backswept centrifugal impeller + vaneless
diffuser, purpose-built for laser-anemometry **flow-field** studies (wake
development in unshrouded impellers), aerodynamically scaled to high-speed
subsonic centrifugals.

**Geometry (verbatim, TM-4481):**
- 20 **full** blades (no splitters), **55° backsweep**.
- Inlet diameter 0.870 m (r1t = 0.435 m); inlet blade height 0.218 m
  (r1h ≈ 0.217 m).
- Exit diameter 1.524 m (**r2 = 0.762 m**); exit blade height **b2 = 0.141 m**.
- Tip clearance 2.54 mm, constant (unshrouded).
- Test/design point: **30 kg/s, 1862 rpm** ("near peak efficiency"). At
  1862 rpm, **U2 ≈ 148.6 m/s** (M_U2 ≈ 0.44) → a *modest* compression
  (PR ≈ 1.1), not the transonic 4:1 of the real CC3.

## Why no case was built (the data gap)

TM-4481 is a **flow-field paper**: it has **no tables** — no tabulated
measured performance (PR, η) and no tabulated impeller-exit swirl/slip. All
quantitative results are velocity-field **figures** at the measurement
planes (a non-uniform jet/wake field). So there is **no ready measured
comparison quantity** for a meanline validation. Building an LSCC case would
be *structural-only* (converges, does 55° backswept work, slipped exit) with
no measured anchor — which is not the campaign's goal (Wiesner slip is
already CONFIRMED at Eckardt, WIE67.md). The model's prediction, for the
record when a measured comparison lands: **Wiesner slip σ ≈ 0.907** at 55°
backsweep / Z=20.

## Paths to a real third centrifugal MEASURED point (recorded, in effort order)

1. **Eckardt A / B (easiest, same rig family).** The Eckardt 1976 paper —
   already in the Test Cases notebook and used for the radial Eckardt O — also
   covers the **backswept A (30°) and B (30° with different blade loading)**
   impellers. A backswept Eckardt point reuses the grounded `EckardtO`
   machinery with new metal angles; the measured PR/η are in the same source.
   **This is the cleanest true third point** and does not need new
   acquisition.
2. **LSCC exit-swirl digitization.** Digitize the TM-4481 impeller-exit
   velocity-field figure to a mass-averaged Cθ2/U2 → a **55° slip** validation
   (a third backsweep after Eckardt 0° / Krain 30°). Moderate effort;
   mostly re-confirms Wiesner.
3. **The real high-speed CC3.** McKain & Holbrook, *Coordinates for a High
   Performance 4:1 Pressure Ratio Centrifugal Compressor* (NASA CR-204134,
   1997) for geometry + a Skoch et al. CC3 performance report for the map.
   A genuine transonic 4:1 third point, but a multi-source acquisition (blade
   coordinate tables → meanline geometry, plus a separate performance
   report); **neither is in the notebook.** These would need adding to the
   Test Cases notebook first.

**Recommendation:** take path 1 (Eckardt A/B) for a true third centrifugal
*measured* point with no new acquisition; treat the real CC3 (path 3) as a
transonic-centrifugal stretch goal pending notebook sources.

## RESOLVED (2026-07-19): the real high-speed CC3 was built via Skoch 2003

Path 3 turned out to be already accessible: the **real** CC3 geometry +
design point are grounded in **Skoch (2003)** (Drive
`skoch_experimental_2003`), not needing the McKain-Holbrook coordinate
report. Built as `CC3Impeller` (an `EckardtO` subclass) — 15 main + 15
splitter blades, 50° backsweep, r1t 105 / r1h 41 / r2 215.5 / b2 17 mm,
PR 4:1 at 21 789 rpm / 4.54 kg/s, U2 = 492 m/s (reproduced exactly),
transonic inducer. Converges all tiers; the point-by-point stage match
stays `[VERIFY]` (vane-island diffuser + design η not tabulated), and it
corroborates the backsweep work-over-prediction trend. See ECKARDT.md
"CC3". This LSCC note stays as the record of the naming correction; the
LSCC itself remains an unbuilt flow-field-only case.
