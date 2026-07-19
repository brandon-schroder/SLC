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

### Validation status

The kernel, architecture, closure framework, and driver stack are built and
structurally verified, and — as of 2026-07 — the model has been **validated
point-by-point against real turbomachinery data for every machine class
except mixed-flow**, landing within a few percent where a clean comparison
exists:

- **Axial compressor** — NASA Rotor 37 (TP-1659): with the library-grounded
  Çetin AGARD-R-745 transonic-deviation correction, Tier-2 pressure ratio
  2.051 vs measured 2.056 (**+0.2%**); matched-PR back-pressure traversal down
  the vertical characteristic. Rotor 38 (TP-2001) second point.
- **Axial turbine** — NASA TN D-6967 two-stage: agrees to **~1%** in the
  matched-PR frame, plus a digitized multi-speed map (work ±2.2% over
  50–100% speed) and a first-stage-only build. VKI LS-89 transonic cascade.
- **Centrifugal** — three points: Eckardt O (radial, stage PR +1.0%), Krain
  (30° backswept), and NASA CC3 (real high-speed 4:1, 50° backswept). The
  stage set closes to **PR ±2% / η ±2 pt** at both Eckardt/Krain loadings
  with one calibrated constant.
- **Operability** — a grounded tip-diffusion-factor stall criterion predicts
  the measured Rotor 37/38 stall within ~3%, wired opt-in into the speedline
  driver.

Calibration was done by **grounding every candidate correlation verbatim from
the reference library and dispositioning it by measurement** — only the Çetin
MCA correction was adopted; K-O TE, Zhu-Sjolander, Wiesner slip, and others
were confirmed, refuted, or found inert with **zero constants tuned to
individual data points**.

So for the validated classes the model is usable for **absolute numbers within
documented bounds** at Tier 1/2, not only trends. Known bounds and open items,
in rough priority:

1. **Mixed-flow (V8)** has no open measured rig — it stays structural-only.
2. **Backswept-centrifugal work** over-predicts with backsweep (the Aungier
   λ work-input role, grounded; a 3-point trend across Eckardt/Krain/CC3);
   adopting it needs a joint slip/blockage/diffuser recalibration.
3. **Tier-3 full-SLC on tight radial/mixed bends** — the interior is fast and
   robust (V8 accelerated 2.6×), but the pocket *edges* need closure-in-Newton.
4. **Off-design map depth** — some speedlines/maps digitized; more available.

Point-by-point speedline *shapes* on the choke side are a capacity/knee
matter (dispositioned for Rotor 37), and a handful of geometry inputs on the
newest cases are recorded estimates where the coordinate report wasn't in
hand. See `CLAUDE.md` and `docs/references/` for the per-case measured record.

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
