# HOWELL — Axial-compressor endwall + tip-clearance loss (Howell / Dixon / Lakshminarayana)

Source: the classic additive drag-coefficient loss model for axial-compressor
blade rows — **Howell** (1945), as presented by **Dixon** (*Fluid Mechanics
and Thermodynamics of Turbomachinery*), **Saravanamuttoo** (*Gas Turbine
Theory*), and **Cumpsty** (*Compressor Aerodynamics*), plus the
**Lakshminarayana** tip-clearance drag (via Cumpsty). All in the NotebookLM
"Staging Area (Theory)" notebook; extracted 2026-07-10, source-grounded,
verbatim with equation numbers. Cross-checked against
`slcflow/closures/axial_compressor/loss.py` (`blade_loading_coefficient`,
`endwall_clearance_loss`).

## Why Howell (not Aungier) for the additive endwall term

§7.1 of the theory manual permits "Koch–Smith **or** Aungier" losses. A prior
extraction established that **Aungier's own endwall/secondary method is *not*
clean-additive**: end-wall and secondary losses are folded into the design
profile-loss correlation via the `K1`/`K2` factors (Aungier Eq 6-46), backed by
the charts Fig 6-11/6-12, and the Aungier tip-clearance term (Eq 6-89) is a
`ΔP_t` tied to blade torque `τ` and blade-row count `N_row` — awkward for a
per-row `omega_bar` closure. Howell's drag-coefficient model **is** clean,
closed-form, additive, and library-verifiable, so it was chosen (Koch–Smith is
chart-heavy — deferred). The annulus constant is common to both (Aungier 6-42 =
Howell p.451).

## Confirmed — code matches the source

| Quantity | Source eq. | Code | Status |
|----------|-----------|------|--------|
| Mean vector angle | `tan β_m = ½(tan β1 + tan β2)` | Dixon 3.15 / Sarav. 5.32 / Cumpsty 2.11 | `blade_loading_coefficient` | ✅ |
| Lift/loading coeff. | `C_L = 2(s/l)cos β_m(tan β1 − tan β2) [− C_D tan β_m]` | Dixon 3.26a / Sarav. 5.33 / Howell p.442 | `blade_loading_coefficient` | ✅ the `−C_D tan β_m` term dropped (source: "negligibly small", the standard "theoretical" `C_L`) |
| Secondary drag | `C_Ds = 0.018 C_L²` | Howell p.451 / Sarav. 5.35 / Cumpsty p.238 | `_CDS_C = 0.018` | ✅ 0.018 confirmed |
| Annulus drag | `C_Da = 0.020 (s/h)`, s=pitch, h=height | Howell p.451 / Sarav. 5.36 / Aungier 6-42 | `_CDA_C = 0.020` | ✅ 0.020 and `s/h` confirmed |
| Tip-clearance drag | `C_Dk = 0.7 C_L² (t/h)`, t=clearance, h=height | Lakshminarayana (1970) via Cumpsty | `_CDK_C = 0.7` | ✅ constant 0.7, `C_L²` dependence confirmed (no `C_L^1.5` in the sources) |
| Drag → loss | `C_D = ζ (s/l)(cos³β_m/cos²β1)` | Cumpsty 4.9 / Howell p.442 / Sarav. 5.32 | inverted in `endwall_clearance_loss` | ✅ (see note) |

Confirmed pieces pinned in `tests/test_lieblein_loss_reference.py`
(`test_blade_loading_coefficient_matches_dixon`,
`test_endwall_clearance_loss_matches_howell`,
`test_endwall_clearance_term_is_inert_without_clearance`,
`test_endwall_validity_drops_at_high_loading`).

## The drag → loss conversion (derived to disambiguate the OCR)

The NotebookLM render of the fraction in Cumpsty 4.9 was ambiguous, so it was
**derived from first principles** to fix the direction. With pitch `s`, chord
`l`, axial velocity `c_x` constant, drag `D = s·Δp0·cos β_m`, loss coefficient
`ζ = Δp0/(½ρc1²)`, and `C_D = D/(½ρc_m² l)`:

    C_D = ζ (s/l)(cos³β_m / cos²β1)      ⟹      ζ = σ (cos²β1 / cos³β_m) C_D

with `σ = l/s` (solidity). This matches the source's explicit clarification
("cos³β_m in the numerator for C_D, cos²β1 in the denominator, s/l multiplier").
So the coded endwall/clearance loss is

    ω_ew = σ (cos²β1 / cos³β_m)(C_Ds + C_Da + C_Dk)

referenced to the **inlet** relative dynamic head (confirmed: "for diffusing
components the inlet velocity pressure is the reference denominator") — the same
reference as the Lieblein profile `omega_bar`, so `ω_ew` **adds directly** and a
single B.2 conversion covers profile + endwall + clearance.

Note `s/h = 1/(σ·AR)` and `t/h = clearance/(AR·chord)` (AR = blade height/chord,
a row-scalar design input; the geometry supplies the tip clearance). The tip
clearance is 0 by default, so `C_Dk` is inert for zero-clearance rows.

## Modelling choices (recorded)

- **Evaluated at the reference (design) triangle.** `C_L` is taken at the
  min-loss velocity triangle (`β1_ref`, `β2_ref`), where the profile `ω_min` is
  already evaluated — the off-design incidence bucket multiplies only the
  profile loss, and the endwall/clearance loss is added flat. Off-design growth
  of the secondary loss (`C_L` at the *actual* triangle) is a recorded
  refinement — it would improve the near-stall speedline shape.
- **Secondary vs clearance overlap.** Howell's `C_Ds = 0.018 C_L²` was
  calibrated on data that *included* a typical tip clearance ("secondary losses
  — trailing vortices and tip clearance"). The explicit Lakshminarayana `C_Dk`
  is therefore best read as the clearance sensitivity *above* Howell's baseline;
  a mild double-count near typical clearances is a `[DECIDE]` refinement. It is
  inert (0) for the current zero-clearance verification cases.
- **`C_L` validity ceiling** `_CL_CEIL = 1.6` (compact support): Howell's drag
  data is moderate-loading; validity saturates to 0 at very high `C_L`. `[VERIFY]`.

## Measured effect

Adding the endwall loss to the profile-only set drops the V5 rotor efficiency
from ~0.96 to **~0.92** (Tier 1/2) — a realistic subsonic-stage level — with PR
essentially unchanged (loss affects η, not the Euler work). The multistage V5
(4 rows compounding) drops to η ≈ 0.64 at PR ≈ 1.09: honest for a lightly-loaded
matched-stage *mixing* testbed where the ideal work is small, not an efficiency
benchmark. This is the deferred loss physics the axial-compressor set's
`__init__` docstring named as required "at V5 calibration time" — it unblocks
the efficiency side of a future point-by-point V5 speedline reproduction (which
still needs a subsonic stage validation dataset, absent from the library).
