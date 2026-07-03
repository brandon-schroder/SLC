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
- **M2** (residual assembler + classical driver, Tier 2 vs. analytic radial
  equilibrium V1, grid-convergence check) — **next**. See ARCH-8 for the full
  milestone list and ARCH-3.2/3.3 + §6.1 for the state/residual contracts M2
  must implement.

## Commands

```bash
pip install -e ".[test]"
pytest -q                          # full suite
python tools/check_imports.py      # ARCH-2 dependency-direction
python tools/check_ad6.py          # AD-6 / smoothness lint
```