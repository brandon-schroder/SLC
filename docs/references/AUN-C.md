# AUN-C — Aungier *Axial-Flow Compressors* (2003): verified incidence/deviation fits

Source: Aungier, R.H., *Axial-Flow Compressors: A Strategy for Aerodynamic
Design and Analysis*, ASME Press (2003), ch. 6 — the analytic curve-fits to
the NASA SP-36 (Lieblein) minimum-loss cascade charts that
`slcflow/closures/axial_compressor/lieblein.py` implements. This is the
**coefficient** half of the SP-36 split (SP-36 = charts/outputs; Aungier =
the fit coefficients — see [`README.md`](README.md)).

**Provenance.** Extracted 2026-07-09 from the user's NotebookLM "Staging Area
(Theory)" notebook (source-grounded, citation-backed; equation numbers are
Aungier's own). Cross-checked term-by-term against `lieblein.py`.

## Confirmed — code matches Aungier verbatim

| Fit | Aungier eq. | Code | Status |
|-----|-------------|------|--------|
| Reference incidence `(i0)10` | 6-13: `β1^p/(5+46e^{−2.3σ}) − 0.1σ³e^{(β1−70)/4}` | `reference_incidence` `:75` | ✅ 46, 2.3, 5, 0.1 |
| exponent `p` | 6-14: `0.914 + σ³/160` | `:74` | ✅ |
| Camber slope `n` | 6-15: `0.025σ − 0.06 − (β1/90)^{1+1.2σ}/(1.5+0.43σ)` | `:78` | ✅ 0.025, 0.06, 1.2, 1.5, 0.43 |
| Reference deviation `(δ0*)10` | 6-20: `0.01σβ1 + [0.74σ^{1.9}+3σ](β1/90)^{1.67+1.09σ}` | `:95` | ✅ 0.01, 0.74, 1.9, 3, 1.67, 1.09 |
| Thickness corr `K_t,δ` | 6-25: `6.25(t/c) + 37.5(t/c)²` | `:97` | ✅ 6.25, 37.5 |
| Camber slope `m1.0` | 6-22: `0.17 − 0.0333x + 0.333x²`, `x=β1/100` | `:98` | ✅ 0.17, 0.0333, 0.333 |
| Carter exponent `b` | 6-24: `0.9625 − 0.17x − 0.85x³` | `:99` | ✅ 0.9625, 0.17, 0.85 |
| `m = m1.0/σ^b` | 6-21 | `:100` | ✅ |
| Off-design slope `[∂δ/∂i]*` | 6-76: `[1+(σ+0.25σ⁴)(β1/53)^{2.5}]/e^{3.1σ}` | `deviation_slope` `:111` | ✅ 0.25, 53, 2.5, 3.1 |

Nine fits confirmed verbatim, pinned in `tests/test_lieblein_reference.py`.

## Finding — a real transcription bug (FIXED)

**Incidence thickness correction `K_t,i`.** Aungier:
- 6-10: `K_t,i = (10·t_b/c)^q`
- 6-11: `q = 0.28 / [0.1 + (t_b/c)^0.3]`

The code (`:77`) had `q = 0.28 / [0.1 + (10·t_b/c)^0.3]` — an **extra ×10
inside the 0.3-power** that is not in Aungier. The base `(10 t/c)^q` was
correct. Both forms give `K_t,i = 1` at t/c=0.10 (base = 1), so the error is
invisible at the reference thickness and only bites off-design (e.g. t/c=0.08:
Aungier K_t,i ≈ 0.907 vs the old code's ≈ 0.941, ~3.5%). Because the docstring
explicitly claims these are "Aungier's published fits," this is a bug against
the cited source, not a modeling choice — **corrected** to `0.1 + t**0.3` and
pinned against Eq 6-11 in the reference test. (V4/V5 are structural-band checks
so the shift stays in-band; the full suite re-run confirmed green.)

## Residual

Every incidence/deviation/off-design-slope fit constant is now discharged.
The SP-36-side `[VERIFY]` (do these fit *outputs* reproduce the SP-36 chart
points?) remains a chart-digitization item, separate from the coefficient
check done here. The **loss** side (`axial_compressor/loss.py`: equivalent-
diffusion θ*/c, the D_eq form) is a separate Aungier/Lieblein-1959 pass, not
covered by this note.
