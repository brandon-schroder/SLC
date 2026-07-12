# slcflow

A reduced-order, object-oriented, streamline-curvature (SLC) throughflow
solver for turbomachinery preliminary design — axial, radial, and mixed-flow
compressors and turbines. One kernel, multi-fidelity (meanline /
streamline-REE / full SLC) via grid collapse, not separate code paths.

## Status

All milestones on the ARCH-8 verification ladder (**M0–M8**) are closed:
geometry/grid, the pure residual assembler, the classical / Newton /
continuation drivers, the axial-compressor, axial-turbine, and centrifugal
correlation sets, all three fidelity tiers, and the §3.6 spanwise-mixing model.

Post-ladder work (see `CLAUDE.md` for the full log) includes an independent
audit + turbine-sign fix, a reference-library correlation-calibration pass
(coefficient level), the axial-compressor **endwall/clearance/shock** and
centrifugal **blade-loading** loss stacks, a **meridional-supersonic-branch
driver** (`drivers/supersonic.py`, pseudo-arclength continuation across the
per-station `M_m = 1` fold), and a 2026-07 diagnosis of the radial/mixed
Tier-3 fragility that separated it into distinct causes: V7 Tier 2 is a
solvable operating-point fold (cracked by an `ṁ` re-centre), V7 Tier 3 is a
physical feasibility fold at the current (high) loss, and V8 Tier 3 a narrow
convergent pocket. Suite: **535 passed / 2 xfailed** (the two `xfail`s are the
V7/V8 Tier-3 tripwires), both lint gates green.

The verification ladder is **structural** (convergence + trends + plausibility
bands), with pervasive `[VERIFY]` on the correlation coefficients — see
[`docs/overview.md` §10](docs/overview.md) for an honest
proven-vs-structural-vs-`[VERIFY]` breakdown. See `CLAUDE.md` for the
milestone-by-milestone log.

### Before use for quantitative design

The kernel, architecture, closure framework, and driver stack are **built and
structurally verified**, but the model has **not been calibrated or validated
against real turbomachinery data**. It is usable today as a **relative/trend
tool** (meanline Tier 1, REE Tier 2) — sizing, sensitivity, configuration
comparison. Before any *absolute* number (efficiency, PR, a speedline) can be
trusted, in priority order:

1. **Quantitative validation** — reproduce at least one published case per
   machine type (axial compressor stage, turbine cascade, Eckardt impeller)
   plus a speedline, point-by-point. Every V4–V8 case is structural-only today.
2. **Correlation calibration** against that data — the coefficients are
   representative fits (`[VERIFY]`); the **blade-loading loss magnitude** is the
   highest-leverage one (it drives the spanwise stratification behind the
   radial/mixed Tier-3 fragility).
3. **Deferred centrifugal losses** — tip-clearance and disk-friction (need
   exit width / hub-tip radii and a machine-level `ṁ`); centrifugal efficiency
   reads optimistic without them. The axial loss stack is complete.
4. **Tier-3 full-SLC reliability** on tight radial/mixed bends (V8 Tier-3
   pocket; V7 Tier-3 infeasible at the current loss — calibrating #2 down
   should help).
5. **Operability validation** — surge-line / choke-traversal quantitative
   match (the machinery exists and is demonstrated in V9; the match is
   `[VERIFY]`).

## Documentation

**Start here:**
[`docs/overview.md`](docs/overview.md) — a guided tour of what has been built
and how a solve runs end to end, with an honest account of what is and isn't
established.

The formulation, architecture, and module-level specs are the source of
truth for this codebase; code conforms to them, not the other way round.

- [`docs/theory_manual.md`](docs/theory_manual.md) — governing equations,
  sign conventions, master q-o momentum equation derivation, loss/entropy
  conversions, verification ladder.
- [`docs/architecture_specification.md`](docs/architecture_specification.md)
  — package layout, binding architectural decisions (AD-1..AD-10), interface
  contracts, solver drivers, milestones.
- [`docs/module_specification_geometry_and_grid.md`](docs/module_specification_geometry_and_grid.md)
  — scope and test plan for `slcflow/geometry` and `slcflow/grid`.

## Setup

```bash
pip install -e ".[test]"
```

Requires Python ≥ 3.14 (see `pyproject.toml`).

## Commands

```bash
pytest -q                          # full test suite
python tools/check_imports.py      # dependency-direction check (ARCH-2 / AD-5)
python tools/check_ad6.py          # AD-6 / smoothness lint
```

All three are CI gates (`.github/workflows/ci.yml`) and should be run
locally before pushing.
