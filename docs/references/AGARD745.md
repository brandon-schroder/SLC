# AGARD745 — Çetin et al., transonic loss & deviation corrections

**Source:** Çetin, Üçer, Hirsch & Serovy, *Application of Modified Loss and
Deviation Correlations to Transonic Axial Compressors*, AGARD-R-745 (1987).
In the user's library (loss-models notebook); extracted verbatim 2026-07-16.

**Role:** the published correction set for exactly the gap measured on
Rotor 37 (`ROTOR37.md`): classical subsonic-cascade deviation rules applied
to MCA/DCA transonic rotors. Derived from 1970s-technology transonic stage
blade-element data (tip diameters 0.25–0.41 m, MCA/DCA sections, analysis
for M1 ≥ 0.5).

## Extracted verbatim

1. **Design-deviation correction (Eq. 3.5)** — Carter's rule
   "underestimates the deviation angle to some extent, especially at the
   higher deviation values"; the proposed second-order correction:

       delta*_cor = -1.099379 + 3.0186·delta*_CAR - 0.1988·(delta*_CAR)²

   (δ*_CAR = the standard Carter's-rule design deviation.) Attributed to
   "transonic and 3-D effects which can not be separated in the present
   analysis".
2. **Swan (1961) transonic off-design deviation** (App. II Eq. 70, M1>0.6):

       delta - delta* = [6.40 - 9.45(M1 - 0.60)]·(D_eq - D_eq*)

3. **Crouse (1974) modified m-factor** (App. II Eq. 53): m_md =
   (0.219 + 8.916e-4·ξ + 2.708e-5·ξ²)·(2a/c)^(2.175 − 0.03552·ξ +
   1.917e-4·ξ²), ξ = stagger [deg], a/c = max-camber location.
4. Shape factor note: "(K_δ)_sh = 1.1 (C series)" (NACA-65 base 1.0).

## Adoption (2026-07-16)

**Eq. 3.5 implemented** as `lieblein.cetin_deviation_correction` — opt-in
via `LieblienSwirl(transonic_correction="cetin_agard745")`, default off
(the NACA-65 pedigree is untouched). Recorded reading: the polynomial is
applied to the **SP-36/Aungier reference deviation** rather than Carter's
rule — both are the subsonic-cascade minimum-loss family the report
corrects, and against the Rotor 37 measured blade elements
(`MEASURED_BE_4182`) the as-published polynomial takes the per-span error
from **RMS 3.8° to 1.2° (mean −3.6° → ~0) with no locally fitted
constant**. C¹ saturation into the monotone fitted branch (vertex 7.59°;
window 0.5–7.5° with validity), pinned by `tests/test_cetin_correction.py`.

End-to-end (Rotor 37 case defaults the correction ON): Tier-2 PR 2.051 vs
measured 2.056 (+0.2%; was +12%), Tier-1 2.135 (+3.8%; was +16%); closure
validity 0 → ~0.8 at Tier 1.

## Swan Eq. 70 — implemented, measured, NOT default-adopted (2026-07-16)

`lieblein.swan_offdesign_deviation` + `LieblienSwirl(offdesign_rule=
"swan_agard745")`: the verbatim bracket, C¹-blended with the Aungier slope
across the stated M1 = 0.6 onset (no flow branch, AD-6), increment smoothly
ceilinged ±8°, validity ending at the data range (M1 ≈ 1.5); D_eq* from the
same reference-triangle convention as the loss chain. Pinned in
`tests/test_cetin_correction.py` (coefficients, transonic sign reversal at
M1 = 1.277, ceiling, C¹).

**Measured on Rotor 37 (hypothesis refuted):** the rule was expected to
steepen the choke-side speedline; in fact the measured 100%-speed line
spans only ~±3° of incidence about reference, so off-design deviation is
small under either rule — Swan shifts PR a uniform **+0.03 (slightly away
from measured)** via its negative Mach bracket at M1 ≈ 1.4 and does **not**
steepen the line (slope Δ0.163 → Δ0.167). The measured choke-side collapse
is **loss/choking physics** (the rig is choked at 20.93 kg/s; the meanline
still has capacity margin), not deviation. The rule stays available for
cases that genuinely traverse large incidence swings; the Rotor 37 case
keeps `offdesign_rule="aungier"` with the finding pinned
(`test_swan_offdesign_rule_runs_but_is_not_adopted`).

## Eq. 3.3 off-design loss — implemented, measured, NOT default-adopted (2026-07-16)

Extracted verbatim (Table 1): ω̄_T = ω̄*_T + c_m(i−i*)² with linear Mach
laws per family/side —

    MCA: choke (i<i*): c_m = 0.02845·M1 − 0.01741;  stall: 0.00363·M1 − 0.00065
    DCA: choke:        c_m = 0.05336·M1 − 0.02937;  stall: 0.00500·M1 − 0.00075

(valid M1 ≥ 0.5, 1970s MCA/DCA rotors; stall/choke = 2×min-loss at
i = i* ± √(ω̄*/c_m), Eq. 3.4. The report's min-loss recommendations —
Koch-Smith + LE-bluntness shock + Jansen-Moffatt Mach factor — are
recorded but NOT stacked on the Aungier §6.7 shock term already carried.)

Implemented as `loss.cetin_offdesign_loss` + `LieblienLoss(offdesign_loss=
"cetin_agard745" | "cetin_agard745_choke", blade_family="mca"|"dca")`:
C¹ M1-blend to the Aungier bucket at the 0.6 onset, softplus-floored
curvature, validity to M1 ≈ 1.5; the `_choke` variant applies the choke-side
line only and keeps Aungier on the stall side. Pinned in
`tests/test_cetin_correction.py`.

**Measured on Rotor 37 (with capacity calibrated): both variants
non-adopted.** The FULL parabola's stall-side line collapses the low-flow
efficiency (0.776 vs measured 0.852 at 19.60 kg/s — it over-penalizes the
meanline's incidence excursions). The CHOKE-ONLY hybrid is **inert**: with
capacity calibrated, this rotor (like the rig, whose measured incidence
stays positive at choke) never runs below reference incidence — its
choke-side PR collapse is the **vertical characteristic** (a
back-pressure-mode comparison where mdot is degenerate), not an off-design
loss bucket. Pinned: `test_agard_offdesign_loss_options_measured_not_adopted`.

## Capacity finding (same pass — see ROTOR37.md)

Rig choke 20.93 ± 0.3 kg/s (AGARD-AR-355). B=0: meanline chokes ~22.25
(+6.5%), Tier-2 ~21.65 (+3.5% — spanwise endwall streamtubes capture half
the gap). Uniform B = 0.033 lands the Tier-2 choke in the measured band but
costs the mid-line PR ~7% (4 of 5 measured points degrade) → **the capacity
deficit is not uniform blockage; it lives at the unmodelled blade-passage
THROAT** (a compressor throat/capacity station is the recorded model item).
Defaults stay parameter-free; the calibration is pinned as a test.

## Recorded, not adopted (deferred)

- **Crouse m-factor** (an alternative design-deviation route; Eq. 3.5
  chosen as the direct, validated correction).
- **Koch-Smith min-loss + LE-bluntness shock + Jansen-Moffatt** (would
  double-count the Aungier §6.7 shock term; revisit only with a min-loss
  level discrepancy in hand).
