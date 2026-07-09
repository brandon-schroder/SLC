# AM-ANGLE — Ainley-Mathieson turbine exit-angle rule: verified

Sources: Ainley & Mathieson (ARC R&M 2974/2853, 1951 → [`AM51`](README.md))
and the Kacker-Okapuu (1982) adoption, as given in the NotebookLM "Staging
Area (Loss Models)" notebook. Extracted 2026-07-09, source-grounded, with AM's
own equation numbers. Cross-checked against
`slcflow/closures/axial_turbine/ainley.py`.

## Confirmed — the code ships the correct sonic asymptote

At outlet Mach `M2 = 1.0`, AM Eq 2 gives the exit angle from the throat area:

    α2 = −cos⁻¹(A_t / A_n,2)

which for straight-backed blades is the **gauge angle / throat cosine rule**
`cos⁻¹(o/s)` (throat opening `o`, pitch `s`). Kacker-Okapuu and Dunham-Came
use this as the `M2 → 1` value.

`ainley.py`'s `throat_exit_angle` returns exactly `arccos(o/s)` (magnitude),
signed by the TE turning direction (`orientation_te`) — matching AM's `−cos⁻¹`
sign convention. ✅ **Confirmed as the correct sonic value.** Pinned in
`tests/test_ainley_reference.py`. (The `orientation_te` signing was the
2026-07 audit fix — consistent with AM's signed exit angle.)

## The deferred correction — now precisely characterized

The code ships the sonic term alone and defers the low-speed correction + Mach
blend (docstring: "deferred to the M6 transonic step … need the exit Mach").
The source pins exactly what that deferred piece is:

1. **Low-speed value (M2 ≤ 0.5), AM Eq 1:**

       α2 = α2* − 4(s/e),   α2* = f(cos⁻¹(o/s))   [Fig. 5 relationship]

   - `4` is the numerical constant (confirmed verbatim).
   - `s` = blade pitch; `e` = mean radius of curvature of the convex (back/
     upper) surface between throat and trailing edge.
   - `α2*` is a near-identity function of the gauge angle `cos⁻¹(o/s)` read off
     Fig. 5. Worked example (stator): `s/e = 0.279 → 4(s/e) = 1.1°`;
     `α2* = −62.4° → α2 = −63.5°`.

2. **Mach interpolation (0.5 ≤ M2 ≤ 1.0):** "a linear variation of α2 may be
   assumed with reasonable accuracy" between the low-speed value (M2=0.5) and
   the sonic `cos⁻¹(o/s)` (M2=1.0); alternatively a smooth curve with an
   inflection at M2=0.75. Kacker-Okapuu and Dunham-Came adopt the linear form.

So the `[VERIFY]` deferral in `ainley.py` is **the `−4(s/e)` low-speed term
(with `α2* = f(gauge)` from Fig. 5) plus the linear M2∈[0.5,1.0] blend** — it
needs the exit Mach and the back-surface curvature `e` (a geometry input not
yet in the §4.1 contract). No bug; the shipped sonic asymptote is correct.

## Residual

Base throat rule verified. The deferred low-speed/Mach correction is now
precisely specified for the M6-transonic implementation; it stays deferred
(needs exit Mach + the back-surface radius `e`). The K-O *loss* side of the
same set was verified separately in [`KO82.md`](KO82.md).
