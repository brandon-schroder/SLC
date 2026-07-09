# CONV-B — Appendix-B loss→entropy conversion definitions: verified

The loss-coefficient definitions and reference dynamic heads in
`slcflow/closures/conversions.py` (Theory Manual Appendix B), checked against
Denton, Cumpsty, Aungier, Dixon, Lakshminarayana in the NotebookLM "Staging
Area (Theory)" notebook. Extracted 2026-07-09, source-grounded. **This is the
foundational layer — every loss set routes its native coefficient through one
of these**, so the `[VERIFY per correlation]` reference-head tags matter most
here.

## Confirmed — definitions and reference dynamic heads all match

| Conversion | Source definition | Code | Status |
|-----------|-------------------|------|--------|
| Master `Δs` from p0 deficit | `Δs = −R ln(p02/p01)` at common T0 (Denton 4a; Lakshminarayana 6.1) | `delta_s_from_p0_deficit` `:70` | ✅ verbatim |
| Compressor `ω̄` (B.2) | `ω̄ = Δp0 / (p01 − p1)` — **INLET** relative dynamic head (Cumpsty; Aungier 5-139; Dixon 3.5 `Yp=(p01−p02)/(p01−p1)`) | `delta_s_compressor_omega_bar` `:77` | ✅ inlet ref |
| Turbine `Y` (B.3) | `Y = Δp0 / (p02 − p2)` — **EXIT** ("discharge") dynamic head (Aungier "in terms of the discharge velocity pressure"; AM/K-O) | `delta_s_turbine_Y` `:93` | ✅ exit ref |
| KE coeff `ζ` (B.4) | `ζ = (h2 − h2s)/(½ c²)` (Denton 3.7/2b) | `delta_s_kinetic_energy_zeta` `:106` | ✅ (see note) |
| Reference-head convention | "diffusing components (compressors) → inlet KE; accelerating (turbines) → exit KE" (verbatim) | B.2 inlet / B.3 exit split | ✅ exactly this |

Rothalpy re-referencing (B.1, `:49`) — `T0r2 = T0r1 + (U2²−U1²)/(2cp)`,
`p0r2,id = p0r1 (T0r2/T0r1)^(γ/(γ−1))` — is textbook rothalpy conservation,
analytically correct. The per-correlation attachments were also cross-checked
in the individual passes: Lieblein `ω̄`→inlet ([`LIEB59.md`](LIEB59.md)), K-O
`Y`→exit ([`KO82.md`](KO82.md)), centrifugal→enthalpy loss
([`CENT-LOSS.md`](CENT-LOSS.md)).

Definitions pinned in `tests/test_conversions_reference.py`.

## Note — one benign convention nuance (ζ denominator)

Denton's `ζ` denominator is the **ideal** exit kinetic energy `½ c2is²`
(`= h01 − h2s`), whereas the code uses the **actual** exit KE `½ V2²`
(`delta_s_kinetic_energy_zeta(fluid, zeta, T2, V2)`). Equal at low loss, they
diverge slightly at high loss. The docstring already flags this `[VERIFY per
correlation]` ("or relative-frame W2 — per source"); it is a definitional
choice to match to whichever source a given `ζ`-based correlation used, not an
error. In this kernel `ζ` is only the K-O trailing-edge term, which is mapped
to a `Y` before summing (see KO82.md), so B.4 is not currently on a hot path.

## Residual

The Appendix-B definitions are verified — the `[VERIFY per correlation]` tags
are resolved at the definitional level (and per-set in LIEB59/KO82/CENT-LOSS).
No bug. The only open nuance is the ζ ideal-vs-actual denominator, benign and
documented.
