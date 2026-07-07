# Guide 0 — Reading Guide for the `docs/guide/` Series

> **What this document is.** The index and usage manual for the explanatory
> guide series. Like everything under `docs/guide/`, it is **descriptive, not
> normative**: the three frozen specs are the source of truth and code
> conforms to them, never to these guides. If a guide and a spec disagree,
> the spec wins and the guide has a bug.
>
> Written at commit `d7b7b27` (2026-07-07); suite 373 tests, both lint gates
> green, milestones M0–M8 all closed.

---

## 1. Why this series exists

The repository's documentation was deliberately split by *authority*:

| Layer | Documents | Character |
|---|---|---|
| **Normative** | `docs/theory_manual.md`, `docs/architecture_specification.md`, `docs/module_specification_geometry_and_grid.md` | Frozen. Terse. Code must conform to them. Open items are marked `[VERIFY]`/`[DECIDE]`. |
| **Charter** | `CLAUDE.md` | Working rules + the milestone-by-milestone status log. |
| **Descriptive** | `docs/overview.md`, this `docs/guide/` series | Explanatory. May be regenerated, expanded, or corrected freely — they carry no authority. |

The normative docs tell you *what is frozen*; `overview.md` tells you *what
exists* at a survey level. This series exists for **depth**: how the code
actually implements the equations, why the numerical machinery needed the
specific stabilizations it has, and what each module contributes — the
material you need to genuinely understand and audit the solver, not just
navigate it.

## 2. The series

| # | File | Contents | Status |
|---|---|---|---|
| 0 | `00_reading_guide.md` | This index. | current |
| 1 | `01_life_of_a_solve.md` | One real `Machine.evaluate` call traced end to end — every stage, every data structure, with actual numbers from two converged runs and one instructive failure. | current |
| 2 | `02_theory_companion.md` | Pedagogical twin of the theory manual: the master q-o ODE derived and explained term by term, the elimination form, Appendix B loss→entropy conversions with worked numbers. | planned |
| 3 | `03_numerics_and_stability.md` | The solution algorithms and the *measured* findings that shaped them: the curvature lag, the calibrated Wilkinson envelope, closure under-relaxation, mixing as a multistage convergence prerequisite, the INBLADE repositioning pockets, Newton/continuation. | planned |
| 4 | `04_module_reference.md` | Package-by-package narrative reference: purpose, public API and contracts, spec traceability, key invariants, pinning tests. | planned |
| 5 | `05_closures_compendium.md` | Per correlation set (Lieblein, Kacker–Okapuu, Centrifugal/Wiesner): physical basis, equations as implemented, coefficient provenance and `[VERIFY]` status, C¹-smoothing choices. | planned |
| 6 | `06_trust_and_verification.md` | The audit view: per V-case, what it proves and what would still pass if a correlation were wrong; the `[VERIFY]` map. | planned |
| 7 | `07_worked_tutorial.md` | Runnable: build a machine from scratch, run all three tiers, run a speedline, break it deliberately and read the typed failures. | planned |

## 3. Reading paths

- **"I want to understand how the solver works."** Read `overview.md` first
  (orientation), then Guide 1 (the concrete trace), then Guide 2 (why the
  equations), then Guide 3 (why the numerics).
- **"I want to modify or add a closure."** Theory Manual §7 + Appendix B
  (normative), then Guide 5, then the `CLAUDE.md` process rules (C¹
  discipline, AD-5). Guide 1 §D.6 shows exactly where closure outputs enter
  the solve.
- **"I want to know whether to trust a predicted number."** `overview.md`
  §10, then Guide 6. Short version: the machinery is verified; the
  correlation coefficients are not calibrated (`[VERIFY]`), by design.
- **"I want to work on the open numerical problems."** Guide 3, plus the
  status log at the bottom of `CLAUDE.md` (the top item is the radial/mixed
  Tier-3 repositioning stabilization, the V8 blocker).

## 4. Conventions used throughout the series

- **Every claim is anchored** to a spec section (`§6.4`, `A.7`, `B.1`,
  `G-5`), a code location (`slcflow/drivers/classical.py:439`), or a test
  (`tests/test_v3_tier_consistency.py`). An unanchored claim is a defect —
  report or fix it.
- **Line numbers are stamped to the header commit.** They rot as code moves;
  the function names beside them do not. If a line number misses, search for
  the named function and consider updating the guide's stamp.
- **Numbers shown are real.** Where a guide quotes iteration counts, norms,
  or performance values, they were produced by running the code at the
  header commit, not estimated. Rerun instructions are included so the
  numbers stay checkable.
- **Symbols follow the theory manual's nomenclature (§10)**: `Vm` meridional
  velocity, `rVθ` (code: `rvt`) angular momentum, `h0` stagnation enthalpy,
  `s` entropy, `q` arc-length position along a quasi-orthogonal, `ψ` (code:
  `psi`) target mass fraction, `κ_m` meridional curvature, `ε` q-o lean
  angle, `ω` shaft speed, `ω_sl` streamline relaxation factor.
- Each guide ends with a short **Check your understanding** section —
  questions with answers keyed back to the guide, since the point of the
  series is comprehension, not shelf-ware.

## 5. Maintenance rules

- Guides are updated in ordinary doc commits; no review gate beyond the
  usual one. They must never be cited as authority in code comments — cite
  the spec section instead.
- When a code change invalidates a guide passage, either fix the passage in
  the same PR or leave a `<!-- STALE: ... -->` marker so drift is visible
  rather than misleading.
- The status column in §2 above is the single place tracking which guides
  exist; update it when a planned guide lands.
