# SLC Throughflow Solver — Architecture Specification

**Status:** Draft v0.1 — companion to *Theory Manual v0.2*; all section references (§, A.x, B.x) point there unless prefixed "ARCH-".
**Scope:** Package structure, data model, interface contracts, solver drivers, numerical utility layer, diagnostics, and testing architecture for the Python implementation.

Working package name: `slcflow` (placeholder — rename before first release; check PyPI collision).

---

## ARCH-1. Design Principles and Decision Register

The architecture serves the five theory-manual principles (§1). Their software translations, recorded as binding architectural decisions (AD):

| ID | Decision | Rationale / consequence |
|---|---|---|
| AD-1 | One kernel; tiers are `FidelityConfig` values, never subclasses or parallel code paths | Enforces §8 consistency by construction; tier regression test V3 becomes trivial |
| AD-2 | Struct-of-arrays state: all nodal fields are `(N_sl, N_qo)` arrays owned by `SolutionState`; objects hold *views/indices*, never scalar copies | Vectorization, cheap Jacobians, serialization; per-station objects describe topology, not data |
| AD-3 | Residual assembly is a pure function of `(x, FrozenInputs)` — no mutation, no hidden state, no I/O | Required by §6.3; enables Newton, finite-difference/AD Jacobians, and deterministic replay |
| AD-4 | Closures are lagged: evaluated at the outer iterate into an immutable `ClosureFields` object that enters residual assembly as data | Keeps residual smooth and cheap; closure-in-Newton is a later upgrade behind the same interface |
| AD-5 | Machine-type knowledge lives only in `closures/` implementations; kernel imports closure *interfaces* only | §1 configuration agnosticism; import-linter rule enforces the dependency direction |
| AD-6 | AD-forward-compatibility rules on the residual path: array-namespace injection (`xp`), no in-place ops on state-derived arrays, no data-dependent Python branching (smooth switches instead) | Keeps a future `jax.numpy` swap tractable without committing to JAX now |
| AD-7 | SI units, angles in radians, everywhere in code; degrees only at I/O boundaries | Single-convention rule of §10; unit errors become I/O-layer bugs only |
| AD-8 | Q-o topology is immutable per solve; streamline *positions* are state | §2.3/§2.5; grid regeneration is a new problem, not a mutation |
| AD-9 | Walls are labeled `wall_0`/`wall_1` with a per-machine physical mapping to hub/shroud | Normative A.1.1 orientation convention |
| AD-10 | No exceptions raised on the residual path; out-of-domain physics returns smoothly saturated values + validity metadata (§7.3); NaN/Inf checked once at assembler boundary with diagnostic dump | Solver robustness near operability limits; exceptions reserved for programming/config errors |

Technology baseline: Python ≥ 3.11, NumPy, SciPy (splines, root-finding, sparse linear algebra), `dataclasses` (frozen where possible) for value objects, `typing.Protocol` for interfaces, `pytest` for testing. No heavyweight frameworks in the kernel; plotting and CoolProp are optional extras. Configuration files (YAML/JSON) are parsed at the I/O boundary into typed objects — the kernel never sees dictionaries.

---

## ARCH-2. Package Layout

```
slcflow/
├── fluid/            # WorkingFluid protocol; perfectgas.py backend (§3.7, §4.6)
├── geometry/         # FlowPath, BladeRowGeometry, station defs, meridional splines (§2, §5.1–5.2)
├── grid/             # StreamlineGrid topology + metric evaluation (phi, kappa_m, eps) (§2.5, §5.1)
├── closures/         # interfaces.py; smoothmath.py; axial_compressor/, axial_turbine/, centrifugal/ sets (§7)
├── transport/        # streamwise transport of h0, s, rVt; work/loss schedules; mixing operator (§3.3–3.6)
├── assembly/         # state packing, FrozenInputs, residual assembler, BC-switch forms (§6.1, §6.6)
├── drivers/          # classical.py, newton.py, continuation.py (§6.2–6.3, §6.7)
├── machine/          # Machine/Row/Duct composition; FidelityConfig; OperatingSpec (§8)
├── diagnostics/      # ConvergenceRecord, validity aggregation, structured logging
├── io/               # config schema in/out, result serialization, unit conversion boundary
└── verification/     # V1–V9 problem definitions as importable cases (§9)
tests/                # mirrors package; regression tolerances from Appendix C
```

Dependency direction (enforced): `fluid`, `geometry`, `closures.smoothmath` are leaves → `grid`, `transport`, `closures.*` → `assembly` → `drivers` → `machine` (facade) → `io`. `verification` may import anything; nothing imports it.

---

## ARCH-3. Core Data Model

### ARCH-3.1 Geometry layer

`FlowPath` holds the two wall curves as parametric C² splines (parametric in arc length — mandatory for $\phi$ through $\pm 90^\circ$, §5.1) plus the ordered station definitions. `StationDef` is topology: type (`DUCT | EDGE_LE | EDGE_TE | INBLADE`), owning row (if any), and the q-o curve (straight segment between wall anchor points by default; general curve allowed, §2.3). Q-o curves are fixed per AD-8.

`BladeRowGeometry` implements the row data contract (§4.1) as *callables of span fraction* (and meridional fraction where applicable): metal angles, chord, solidity, thicknesses, throat, lean/sweep, clearance, blade count, rotation flag. Implementations: `ParamRowGeometry` (design-parameter driven, for optimization) and `TabulatedRowGeometry` (from existing geometry). Both must guarantee C¹ output in span fraction (§7.3 propagates upstream to geometry).

### ARCH-3.2 Grid and state

`StreamlineGrid` (topology only): $N_{sl}$, $N_{qo}$, station list, fixed mass fractions $\psi_i$, wall labels (AD-9). `SolutionState` (data, SoA per AD-2):

```python
@dataclass
class SolutionState:
    # geometric (N_sl, N_qo)
    q: Array        # arc-length position of node along its q-o
    z: Array; r: Array          # derived from q via q-o curves (cached)
    phi: Array; kappa_m: Array; eps: Array   # metrics (cached, §5.1–5.2)
    # kinematic / thermodynamic (N_sl, N_qo)
    Vm: Array; rVt: Array; h0: Array; s: Array
    # operating scalars
    mdot: float; omega_shaft: dict[RowId, float]
    meta: SolveMeta   # fidelity config hash, convergence record ref, versions
```

The Newton **state vector** `x` (§6.1) is packed/unpacked by `assembly.pack`: `x = [Vm_wall0[j] for all j] + [q[i,j] for interior i] (+ [mdot] in choke-proximal mode)`. Everything else in `SolutionState` is either frozen input, cache, or reconstructed during assembly. Caches are invalidated by construction: assembly recomputes metrics from `x`; cached fields exist only for diagnostics/output.

### ARCH-3.3 Frozen inputs and closure fields

`FrozenInputs` is the immutable bundle entering residual assembly (AD-3/AD-4): grid topology, geometry callables sampled to arrays, fluid backend, `FidelityConfig`, `OperatingSpec`, and `ClosureFields` — the lagged closure outputs as nodal/row arrays: $\Delta s_{row}(i)$, exit $rV_\theta(i)$, blockage $B(i,j)$, mixing coefficient field, in-blade schedules, plus per-evaluation validity $v$ (§7.3.3). `ClosureFields` carries the iteration tag of the outer iterate that produced it, so convergence records can attribute movement to closure updates vs. flow updates.

---

## ARCH-4. Interface Contracts

All interfaces are `typing.Protocol` classes; implementations register through plain imports (no plugin machinery until needed). Signatures below are normative in shape; exact type aliases live in `slcflow/types.py`.

### ARCH-4.1 Working fluid (§3.7, §4.6)

```python
class WorkingFluid(Protocol):
    def rho(self, h: Array, s: Array) -> Array: ...
    def T(self, h: Array, s: Array) -> Array: ...
    def p(self, h: Array, s: Array) -> Array: ...
    def a(self, h: Array, s: Array) -> Array: ...          # speed of sound
    def h_from_Tp(self, T: Array, p: Array) -> Array: ...
    def s_from_Tp(self, T: Array, p: Array) -> Array: ...
    def stag_from_static(self, h, s, V) -> StagState: ...  # and inverse
```

Vectorized over arrays, no scalars-only paths (AD-2). `PerfectGas(gamma, R)` is the reference backend; its analytic forms are also the test oracle for any future real-gas backend at the perfect-gas limit.

### ARCH-4.2 Closures (§7.1–7.3)

```python
class LossModel(Protocol):
    def evaluate(self, row: RowView, flow: RowFlowView) -> LossBreakdown: ...
        # LossBreakdown: components in *source-native coefficients* + converted
        # delta_s per streamtube (Appendix B applied inside), validity v in [0,1]

class SwirlClosure(Protocol):
    def exit_rVt(self, row: RowView, flow: RowFlowView) -> SwirlResult: ...
        # deviation- or slip-based internally; returns rVt(i), validity

class BlockageModel(Protocol):
    def blockage(self, machine: MachineView, flow: FlowView) -> Array: ...  # B(i,j)

class MixingModel(Protocol):
    def mu_mix(self, flow: FlowView) -> Array: ...   # §3.6 coefficient field
```

`RowView` / `RowFlowView` expose *only* the §4.1 data contract and local circumferentially averaged flow (§7.2) — constructed by the assembler, so closures physically cannot reach solver internals. `CorrelationSet` is a named, versioned bundle `{loss, swirl, blockage, mixing}` with a documented provenance string per member (paper + library file ID); mixed sets across rows trigger a logged warning (§7.1).

`closures/smoothmath.py` is the C¹ toolbox every correlation must build from: `smoothstep`, `soft_clip(x, lo, hi, width)`, `smooth_max/min` (log-sum-exp), `blend(x, x0, width)` — each with mandatory, documented width parameters. Code review rule: **no raw `if` on flow quantities inside `closures/`**; branching on *geometry/topology* (row type, flags) is fine.

### ARCH-4.3 Operating specification and BC switching (§6.6)

```python
OperatingSpec = MassFlowSpec(mdot) | BackPressureSpec(p_exit, station)
```

The assembler builds the residual form matching the spec: `MassFlowSpec` → continuity residuals $R^{cont}_j$ close the system; `BackPressureSpec` → `mdot` joins `x`, one q-o continuity residual is replaced by the back-pressure match. The *switch decision* (hysteresis, thresholds $c_{sw}, \delta_{hys}$) lives in `drivers/continuation.py`, not in assembly — assembly only knows which form it was asked to build. This separation keeps the residual pure (AD-3) and the operability logic testable in isolation.

---

## ARCH-5. Assembly and Drivers

### ARCH-5.1 Residual assembler (§6.1)

```python
class ResidualAssembler:
    def __init__(self, frozen: FrozenInputs): ...
    def residual(self, x: Array) -> Array: ...            # pure (AD-3)
    def split(self, x) -> AssembledFields: ...            # Vm(q), metrics, thermo per q-o
    # reusable pieces (also used by the classical driver):
    def integrate_master_ode(self, j, Vm_wall0, fields) -> Array: ...   # §5.3
    def continuity_F(self, j, Vm_wall0, fields) -> float: ...           # §5.4
    def qo_capacity(self, j, fields) -> float: ...                      # A.7 / §6.6
```

Term switches (§8): curvature/lean terms multiply flags from `FidelityConfig` inside `integrate_master_ode` — data, not branches (AD-6). The same assembler instance therefore serves all tiers.

### ARCH-5.2 Classical driver (§6.2)

Implements the nested scheme as orchestration over assembler pieces: geometry pass → per-q-o scalar solves (`brentq` safeguarded Newton on `continuity_F`, bracketed on the subsonic branch per §6.5/A.7) → repositioning with adaptive $\omega_{sl}$ (§6.4; per-iteration value computed from worst-case local aspect ratio and $M_m$, logged) → lagged closure refresh. Convergence per §6.2.5 with all three norms reported.

### ARCH-5.3 Newton driver (§6.3)

`scipy.optimize`-style Newton with line search over `ResidualAssembler.residual`; Jacobian by colored finite differences initially (structure: near-block-tridiagonal in station index — exploit via `scipy.sparse`), AD later under AD-6. Warm start mandatory: from classical iterate or neighboring operating point. Closure lagging retained (quasi-Newton outer loop, §6.3).

### ARCH-5.4 Continuation / map driver (§6.7)

Owns speedline orchestration: point ordering choke→stall, warm starts, adaptive stepping with cut-back, driver escalation (classical→Newton), BC-switch state machine with hysteresis, stall flagging with recorded criterion. Emits `MapResult` (points, margins, flags, full per-point `SolutionState` refs).

### ARCH-5.5 Facade and MDO surface

`Machine.evaluate(spec, fidelity, warm_start=None) -> PerformanceResult` — the single entry point for coupled/MDO use: returns scalars (ṁ, PR, η, choke/stall margins, aggregate validity), spanwise profiles, and a reusable `SolutionState` for warm-starting the next optimizer iterate. Deterministic given identical inputs (AD-3 makes this checkable by replay). Map generation and single-point solves share this path.

---

## ARCH-6. Diagnostics, Provenance, Errors

`ConvergenceRecord`: per-iteration residual norms, streamline movement, $\omega_{sl}$ used, choke margins $c_j$/$c_{row}$, closure-update norms, aggregate validity — persisted alongside results; every converged solution can answer "how did we get here." Structured logging (JSON-lines) keyed by solve ID. Per AD-10, the assembler boundary performs the single NaN/Inf check and, on failure, dumps `x`, the offending field, and `FrozenInputs` hashes to a reproducer bundle. Solver failures return typed status objects (`Converged | Stalled | ChokeLimited | MaxIter | NumericalFailure`), never exceptions, so map drivers and optimizers can react programmatically.

---

## ARCH-7. Testing Architecture

Mirrors §9: `verification/` holds V1–V9 as importable problem definitions; `tests/` binds them to tolerances (Appendix C) as pytest regressions. Additional non-physics tests required by this spec: interface conformance tests per Protocol; smoothness sweeps per correlation (§7.3.4 — finiteness + bounded numerical derivative over the full domain, automated); AD-6 lint (no in-place ops / raw branches on the residual path, enforced with a custom flake8 plugin or grep-based CI check); dependency-direction check (import-linter); tier-consistency test (AD-1/V3) run on every commit. Property-based tests (hypothesis) for pack/unpack round-trips and fluid-backend thermodynamic identities.

---

## ARCH-8. Milestones (tied to the verification ladder)

M0 scaffolding + CI + smoothmath + PerfectGas (property tests green) → M1 geometry/grid/metrics + V2 curvature machinery on frozen streamlines → M2 assembler + classical driver, Tier 2 passing V1 (analytic REE) and grid-order check → M3 full Tier 3 repositioning, V2/V3 green, Wilkinson constant calibrated → M4 axial-compressor correlation set + Tier 1 mode, V4/V5 → M5 Newton driver + continuation + BC switching, V9 choke traversal on V5 case → M6 turbine set (V6) → M7 centrifugal: parametric-φ path, INBLADE stations, slip, V7 → M8 mixing model, multistage V5 revisit, mixed-flow V8. Each milestone ends with its verification cases as permanent regressions — no milestone is "done" on demo output.

---

## ARCH-9. Deliberately Deferred (recorded so they aren't accidental)

Real-gas backend (interface ready; tabulated CoolProp wrapper later). JAX/AD swap (rules enforced now, payoff later). Closure-in-Newton. Annulus boundary-layer *model* (prescribed blockage schedule first; Aungier/Jansen model behind `BlockageModel` later). Cooling/bleed flows (add as per-streamtube mass/enthalpy sources in transport — note the continuity integral already permits station-dependent ṁ; implement when needed). GUI/plotting beyond diagnostic matplotlib helpers.