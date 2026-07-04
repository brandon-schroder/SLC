# CLAUDE.md — Project Instructions

## What this is

`slcflow`: a reduced-order, object-oriented, streamline-curvature (SLC)
throughflow solver for turbomachinery preliminary design — axial, radial, and
mixed-flow compressors and turbines, one kernel, multi-fidelity (meanline /
streamline-REE / full SLC) via grid collapse, not separate code paths.

## Read before touching related code

Three documents in `docs/` are the source of truth. Do not re-derive
formulation, architecture, or module-scope decisions from general
turbomachinery knowledge — they are frozen (with open items marked
`[VERIFY]`/`[DECIDE]`) and code must conform to them, not the other way round.

- `docs/theory_manual.md` — governing equations, sign conventions, the master
  q-o momentum equation and its full derivation (Appendix A, **normative for
  signs**), loss/entropy conversions (Appendix B), verification ladder (§9).
- `docs/architecture_specification.md` — package layout, the AD-1..AD-10
  binding decisions (below), interface contracts, solver drivers, milestones
  (ARCH-8).
- `docs/module_specification_geometry_and_grid.md` — scope and test plan for
  `slcflow/geometry` and `slcflow/grid` specifically.

Before writing code in a package, view that package's governing section(s)
above. Before writing code implementing an equation, check the theory manual
section number is cited in a comment or docstring.

## Binding architectural decisions (ARCH-1)

These are not suggestions; violating them is a bug even if tests pass.

- **AD-1** One kernel; fidelity tiers are data (`FidelityConfig`), never
  subclasses or parallel code paths.
- **AD-2** Struct-of-arrays state: nodal fields live in flat `(N_sl, N_qo)`
  arrays; objects describe topology, not data.
- **AD-3** Residual assembly is a **pure function** of `(x, FrozenInputs)` —
  no mutation, no hidden state, no I/O on the residual path.
- **AD-4** Closures are lagged into an immutable `ClosureFields` object per
  outer iterate; closure-in-Newton is a later upgrade behind the same
  interface.
- **AD-5** Machine-type knowledge (loss/deviation/blockage correlations)
  lives only in `slcflow/closures/`; the kernel imports interfaces, never
  implementations. Enforced by `tools/check_imports.py`.
- **AD-6** Numerical forward-compatibility on the residual path: array
  namespace injected via `xp=` (default NumPy, future JAX), no in-place
  mutation of state-derived arrays, no data-dependent Python branching on
  flow arrays (use `slcflow/closures/smoothmath.py` primitives instead).
  Partially enforced by `tools/check_ad6.py` (token-level grep lint — real
  but conservative; false positives get silenced with `# ad6: allow` +
  justification, not deleted).
- **AD-7** SI units, radians, everywhere in code. Degrees only at I/O
  boundaries.
- **AD-8** Q-o topology is immutable per solve; only streamline *positions*
  are state.
- **AD-9** Walls are labeled `wall_0`/`wall_1`, not `hub`/`shroud` — the
  physical mapping is machine-dependent (see theory manual A.1.1). Never
  assume `q=0` is the hub.
- **AD-10** No exceptions on the residual path. Out-of-domain physics
  saturates smoothly and returns validity metadata. Exceptions
  (`slcflow.errors.ConfigError`) are reserved for configuration/construction
  boundaries — raise loudly and early there.

## Process discipline

- **C¹ smoothness (Theory Manual §7.3) is mandatory** for anything in
  `closures/` and anything else touching flow-state arrays: no raw `if` /
  `clip` / `abs` / `min` / `max` on flow quantities. Build from
  `smoothmath.py`. A new primitive needs a test proving it's C¹ by
  refinement-scaling (see `tests/test_smoothmath.py::assert_c1_continuous`
  for the pattern — and its docstring for why naive spike-detection
  heuristics don't work), *and* a negative-control test proving the checker
  actually rejects a known-discontinuous function.
- **Adjudicate code of uncertain provenance before trusting it.** If code
  appears in the workspace that you did not just write this session (stray
  drafts, prior-session output, anything not traceable to a reviewed commit),
  do not silently adopt or extend it. Write an independent test suite derived
  from the relevant spec section *without reading the implementation first*,
  then check it against the code. Only adopt what passes. Record the
  provenance situation in the test file's docstring. This happened once
  already (`tests/test_grid_adjudication.py` is the precedent) — repeat the
  pattern, don't skip it because it's slower.
- **Every module implementing a numbered spec section gets a test tied to
  that section number**, not just to intuited correctness — e.g. a docstring
  or comment reading `(G-8.1)` or `(§5.4)` next to the assertion it verifies.
  This is what lets `docs/` and `tests/` stay synchronized as the ground
  truth instead of drifting apart.
- **Sign conventions are frozen in Theory Manual Appendix A.** If a
  derivation elsewhere in the literature (or in generated code) disagrees
  with Appendix A, the manual wins unless you are explicitly revising the
  manual itself as a reviewed, deliberate edit — never silently flip a sign
  to make a test pass.
- Before considering a module done, run both lints and the full suite:
  `python tools/check_imports.py && python tools/check_ad6.py && pytest -q`.
  All three are also CI gates (`.github/workflows/ci.yml`); don't rely on CI
  to catch what you can check locally first.

## Current status

- **M0** (scaffold, `smoothmath`, `PerfectGas`) — closed.
- **M1** (geometry/grid: `WallCurve`, `FlowPath`/q-o construction,
  streamline init, metric evaluation) — closed. Acceptance gate
  (frozen-streamline master-ODE vs. analytic bend solution) passing.
- **M0/M1 cleanup audit** — closed (6 commits on `main`, CI-green baseline).
  Covered: repo/environment sanity, naming/layout vs. ARCH-2, test-provenance
  re-check (`test_grid.py`/`test_grid_adjudication.py` split confirmed still
  valid), spec-traceability spot-check, dead-code sweep, and an independent
  `fluid`/`closures` adjudication suite
  (`tests/test_fluid_closures_adjudication.py`) matching the
  `test_grid_adjudication.py` precedent. No open items remain from this pass.
- **M2** (residual assembler + classical driver, Tier 2 vs. analytic radial
  equilibrium V1, grid-convergence check) — closed. Ran as four reviewed
  sub-steps on `main`: (1) `transport/` — §3.3–3.5 conservation relations +
  C¹ work/loss schedules (§3.6 mixing deferred to M8 per ARCH-8/ARCH-9);
  (2) `assembly/` + `slcflow/types.py` — `FrozenInputs` as the single config
  boundary, `ResidualAssembler` (RK2 master ODE over PCHIP distributions,
  §5.3–5.4/§6.1 elimination form, A.7 capacity); (3) `drivers/classical.py`
  + `diagnostics/` — §6.2 nested scheme, typed `SolveStatus` returns
  (ARCH-6), all three §6.2.5 norms recorded; (4) `verification/` — V1
  analytic-REE cases with independent dense references. Acceptance gate
  passing: V1a–V1c regressions + grid-order check, observed order 1.94
  (tolerances in Theory Manual Appendix C.1).
- **M3** (full Tier 3 repositioning, V2/V3 green, Wilkinson constant
  calibrated) — closed. Ran as four reviewed sub-steps on `main`:
  (1) Tier-3 stabilization — §5.5 curvature lag wired as lagged data;
  measured finding: the lag is *mandatory* (not optional) whenever the
  curvature term is active, else the streamwise odd-even mode diverges at
  any ω; crossing-streamline carryover resolved structurally (classical
  repositioning is a convex blend of monotone vectors; Newton-side guard
  is M5). (2) V2 curved annulus vs. planar-limit concentric reference
  (Appendix C.2; `[VERIFY]` remains for an external potential-flow
  cross-check with duct extensions). (3) §6.4 `[VERIFY]` resolved (manual
  v0.3): measured envelope is ω ≤ ~7.3·(Δm_min/L_qo)^1.5 — *not* the
  literature (Δm/Δq)² aspect form; shipped `wilkinson_c = 4.4` (0.6×
  margin); envelope + rerunnable study in Appendix C.3 /
  `tools/calibrate_wilkinson.py`. (4) V3 tier consistency: Tier 2 ≡ Tier 3
  bit-for-bit on straight-annulus vortex cases, asserted at 1e-10 with a
  curved-path non-vacuousness guard (Appendix C.4).
- **M4** (axial-compressor correlation set + Tier 1 mode, V4/V5) —
  **next**. First real content for `closures/` (Lieblein incidence/
  deviation, Koch–Smith or Aungier losses per §7.1) behind the ARCH-4.2
  interfaces, plus the `machine/` facade with the n_sl = 1 Tier-1 path
  (lifts the `ResidualAssembler` n_sl ≥ 2 constraint). Items that come due
  here, carried from M2/M3 reviews: the Tier-1 mass-average clause of V3
  (Appendix C.4); the `rvt_le` consistency integration test for the
  closure-fed transport path (§3.4); moving `t_stations` validation into
  `ClosureFields`/`FrozenInputs` when in-blade schedules become closure
  data; an `assert_valid_schedule` contract-test helper (§7.3.4) with the
  first non-default `DistributionSchedule`; Appendix B conversion checks
  charged at row exit (`[VERIFY]` per correlation, §4.4).

## Commands

```bash
pip install -e ".[test]"
pytest -q                          # full suite
python tools/check_imports.py      # ARCH-2 dependency-direction
python tools/check_ad6.py          # AD-6 / smoothness lint
```