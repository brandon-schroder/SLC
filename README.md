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
audit + turbine-sign fix, a reference-library correlation-calibration pass, the
axial-compressor endwall/clearance/shock loss stack, and a
**meridional-supersonic-branch driver** — pseudo-arclength continuation
(`drivers/supersonic.py`) that crosses the per-station `M_m = 1` continuity fold
onto the supersonic-meridional branch, with both a prescribed-transport (duct)
path and a closure-lagged blade-row path. Suite: 534 tests, both lint gates
green.

The verification ladder is largely **structural** (convergence + trends +
plausibility bands), with pervasive `[VERIFY]` on the correlation
coefficients — see [`docs/overview.md` §10](docs/overview.md) for an honest
proven-vs-structural-vs-`[VERIFY]` breakdown before trusting any predicted
performance number. See `CLAUDE.md` for the milestone-by-milestone log.

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
