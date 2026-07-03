# slcflow

A reduced-order, object-oriented, streamline-curvature (SLC) throughflow
solver for turbomachinery preliminary design — axial, radial, and mixed-flow
compressors and turbines. One kernel, multi-fidelity (meanline /
streamline-REE / full SLC) via grid collapse, not separate code paths.

## Status

- **M0** (scaffold, `smoothmath`, `PerfectGas`) — closed.
- **M1** (geometry/grid: `WallCurve`, `FlowPath`/q-o construction, streamline
  init, metric evaluation) — closed.
- **M2** (residual assembler + classical driver) — next.

See `CLAUDE.md` for the full milestone list and current focus.

## Documentation

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
