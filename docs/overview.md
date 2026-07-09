# slcflow — System Overview & Guided Tour

> **What this document is.** A reader's guide to the whole codebase: what has
> been built, how the pieces fit, and — most importantly — **how a solve
> actually runs end to end**. It is descriptive, not normative: the three
> frozen specs in `docs/` are the source of truth, and code conforms to them.
> Where this guide states what is *proven* versus *structural* versus
> *unvalidated*, it tries to be scrupulously honest — see
> [§10 What is and isn't established](#10-what-is-and-isnt-established).
>
> Written after milestone M8 closed (the last on the ARCH-8 ladder). Suite at
> that point: **347 tests**, both lint gates green.

---

## 1. What slcflow is

`slcflow` is a **reduced-order streamline-curvature (SLC) throughflow solver**
for turbomachinery preliminary design — axial, radial (centrifugal), and
mixed-flow compressors and turbines. It computes the axisymmetric,
circumferentially-averaged meridional flow field through a machine: velocities,
pressures, temperatures, swirl, work, and loss along the hub-to-shroud span at
a series of streamwise stations.

The defining choice is **one kernel, multiple fidelities as data**:

- **Tier 1 — meanline**: a single mid-span streamline (`n_sl = 1`). The
  cheapest 0-D-ish estimate.
- **Tier 2 — streamline radial-equilibrium (REE)**: several streamlines,
  radial equilibrium without streamline curvature.
- **Tier 3 — full SLC**: many streamlines whose *positions* are solved for,
  including the curvature term.

These are **not** three code paths. They are the same equations and the same
solver with a different grid resolution and a few multiplier flags set to 0 or
1. That principle (binding decision **AD-1**) shapes the entire architecture.

## 2. The document map

Four documents, in order of authority:

| Document | Role |
|---|---|
| `docs/theory_manual.md` | **Normative** physics: governing equations, the master q-o momentum ODE and its full derivation (Appendix A, normative for signs), loss→entropy conversions (Appendix B), the verification ladder and its tolerances (Appendix C). |
| `docs/architecture_specification.md` | **Normative** structure: package layout, the AD-1…AD-10 binding decisions, interface contracts, the milestone ladder (ARCH-8). |
| `docs/module_specification_geometry_and_grid.md` | **Normative** scope/test plan for `geometry` and `grid`. |
| `docs/overview.md` (this file) | **Descriptive** guide: how the built code realizes those specs and how a solve runs. Read this first for orientation; read the specs before changing anything. |

`CLAUDE.md` is the working charter: the binding decisions restated, the process
discipline, and the milestone-by-milestone status log.

## 3. The one-kernel principle (why the code looks the way it does)

Everything downstream follows from a small set of **binding architectural
decisions** (`AD-*`). Violating one is a bug even if tests pass.

| # | Decision | Consequence in the code |
|---|---|---|
| **AD-1** | One kernel; fidelity tiers are *data* (`FidelityConfig`), never subclasses. | `Tier 1/2/3` differ by `n_sl` and by float flags `curvature_term`, `lean_term`, `mixing_term` ∈ {0,1}. No `if tier == …`. |
| **AD-2** | Struct-of-arrays state. | Nodal fields are flat `(N_sl, N_qo)` arrays; objects describe *topology*, not data. |
| **AD-3** | Residual assembly is a **pure function** of `(x, FrozenInputs)`. | `ResidualAssembler.residual(x)` has no mutation, no I/O, no hidden state. This is what lets the Newton driver differentiate it. |
| **AD-4** | Closures are **lagged** into an immutable `ClosureFields` per outer iterate. | Loss/deviation/mixing are evaluated from the *previous* iterate (Picard), frozen, then the residual sees constants. |
| **AD-5** | Machine-type knowledge lives **only** in `closures/`. | The kernel imports closure *interfaces*, never implementations. Enforced by `tools/check_imports.py`. |
| **AD-6** | Numerical forward-compatibility on the residual path. | Array namespace injected via `xp=` (NumPy today, JAX later); no in-place mutation of state arrays; no data-dependent Python branching on flow arrays (use `smoothmath`). Partially enforced by `tools/check_ad6.py`. |
| **AD-7** | SI units, radians everywhere in code. | Degrees only at I/O boundaries (verification case definitions convert on the way in). |
| **AD-8** | Q-o topology is immutable per solve. | Only streamline *positions* are state; the station/wall structure is fixed. |
| **AD-9** | Walls are `wall_0`/`wall_1`, never `hub`/`shroud`. | The physical mapping is machine-dependent; `q = 0` is *not* assumed to be the hub. |
| **AD-10** | No exceptions on the residual path. | Out-of-domain physics saturates smoothly and returns validity metadata + a typed `SolveStatus`. `ConfigError` is reserved for construction boundaries. |

A second rule, **process discipline C¹** (Theory Manual §7.3): anything
touching flow-state arrays must be **C¹-continuous** — no raw `if`/`clip`/`abs`
/`min`/`max` on flow quantities. Such functions are built from
`closures/smoothmath.py` primitives, each of which has a refinement-scaling C¹
test plus a negative control.

## 4. Architecture at a glance

Dependency direction flows **downward** (checked by `tools/check_imports.py`);
nothing low imports something high, and only `closures/` holds machine-type
knowledge.

```
machine/            Facade: Machine.evaluate(spec, fidelity, n_sl) -> PerformanceResult
   │
drivers/            Solution algorithms: classical (Picard), newton, continuation
   │
assembly/           The pure residual: ResidualAssembler, FrozenInputs, pack/unpack
   │
transport/          Streamwise march of h0/s/rVt; work/loss schedules; §3.6 mixing
grid/               Q-o topology, streamline init, metric evaluation, quadrature
geometry/           WallCurve, FlowPath (stations), blade-row geometry
   │
closures/           Machine knowledge: loss/swirl/mixing correlation sets + smoothmath
fluid/              Working-fluid backend (PerfectGas today)
diagnostics/        Convergence records
types.py, errors.py Shared value types (FidelityConfig, specs), exceptions
verification/       V1..V9 problem definitions with independent references
```

- `tools/check_imports.py` — enforces the dependency direction and AD-5.
- `tools/check_ad6.py` — token-level AD-6 / smoothness lint.
- `tools/calibrate_wilkinson.py` — rerunnable stability-envelope calibration.
- `.github/workflows/ci.yml` — runs both lints + `pytest` as gates.

## 5. The physics, briefly

The full derivation is Theory Manual §3 + Appendix A (normative for signs).
The essentials:

**Master q-o momentum ODE (§3.1 / A.5).** Along a quasi-orthogonal (a line
cutting across the span from `wall_0` to `wall_1`), the meridional velocity
`Vm` varies according to a first-order ODE in the span coordinate `q`:

```
dVm/dq = [ dh0/dq − T·ds/dq − (rVθ/r²)·d(rVθ)/dq ] / Vm
         + curvature_term · Vm·κ_m·cos(ε)          (Tier-3)
         + lean_term      · Vm·(dVm/dm)·sin(ε)      (Tier-3)
         + (in-blade force term — deferred; zero for radial stacking)
```

The bracketed part is the radial-equilibrium core (present at all tiers); the
`curvature_term`/`lean_term` are the Tier-3-exclusive streamline-curvature
contributions, multiplied by the `FidelityConfig` flags. This is integrated
across the span by RK2 from `Vm` at `q = 0`.

**Continuity (§3.2 / A.7).** Mass flux is `ṁ = 2π ∫ ρ Vm cos(ε)(1−B) r dq`
across the span. Given the ODE gives the whole `Vm(q)` profile from the single
value `Vm(q=0)`, continuity is one scalar equation per station that pins
`Vm(q=0)` to hit the target `ṁ`.

**Streamwise transport (§3.3–3.5).** Between stations, three per-streamtube
fields march downstream:
- `rVθ` (swirl): conserved in ducts; set by the row's swirl closure across a
  blade row; distributed along INBLADE stations by a C¹ schedule.
- `h0` (stagnation enthalpy): rothalpy conservation, `h0 += Ω·Δ(rVθ)` (Euler
  work).
- `s` (entropy): `s += Δs_row + Δs_mix`, where `Δs_row` comes from the loss
  closure (converted per Appendix B) and `Δs_mix` from mixing.

**Spanwise mixing (§3.6).** An implicit diffusion of `{h0, s, rVθ}` across the
span, applied per marching step. Off in Tiers 1–2; multistage axial Tier-3
opts in. See [§9](#9-spanwise-mixing-m8).

**Closures → entropy (Appendix B).** Loss models emit a native coefficient
(compressor `ω̄`, turbine `Y`, or enthalpy-loss `Δh`); Appendix B converts each
to an entropy increment at the correctly re-referenced exit state (B.1
rothalpy re-referencing across the radius change is what makes it correct for
radial rotors, not just axial).

## 6. The data model

- **State vector `x`** (`assembly/pack.py`): the *unknowns* — `Vm(q=0)` per
  station and the interior streamline `q`-positions, plus `ṁ` in back-pressure
  mode. Everything else is derived.
- **`FrozenInputs`** (`assembly/inputs.py`): the immutable single config
  boundary for one outer iterate — geometry, fluid, fidelity, spec, the
  *lagged* transported fields, lagged `Vm`, lagged curvature. The residual is a
  pure function of `(x, FrozenInputs)`.
- **`ClosureFields`**: the lagged closure outputs (row exit `rVθ`, row `Δs`,
  blockage, aggregate validity) for an iterate.
- **`AssembledFields`**: what the assembler builds from `x` — metrics, `Vm`,
  `ρ`, meridional Mach. Read-only downstream.
- **`SolveStatus`** / `PerformanceResult`: typed outcomes (`CONVERGED`,
  `CHOKE_LIMITED`, `NUMERICAL_FAILURE`, …) and the reduced performance picture
  (PR, efficiency, exit profiles, validity).

## 7. How a solve actually runs (the important section)

Entry point: `Machine.evaluate(spec, fidelity, n_sl)` → builds
`GridTopology(flowpath, n_sl)` and the inlet fields, calls `solve_classical`,
reduces the result to a `PerformanceResult`.

### 7.1 The classical nested driver (`drivers/classical.py`, Theory §6.2)

This is the default solver. Picard iteration with three nested concerns.

**Initialization.** Place streamlines by the area rule; make a 1-D-continuity
`Vm` guess; sweep the inlet fields downstream through the (initially duct)
transport steps.

**Outer loop** (up to `max_outer`), each iterate:

1. **Freeze** the geometry pass and lagged fields into `FrozenInputs`; build a
   `ResidualAssembler`.
2. **Station march (the inner solves).** For each q-o station `j`, solve the
   *scalar* continuity equation `F_j(Vm_{q=0}) = 0` with a safeguarded
   bracketed root find (`brentq`). Each evaluation integrates the master ODE
   across the span (RK2) to get `Vm(q)`, then integrates `ρVm` for the mass
   flux. Subsonic branch; if a station cannot pass `ṁ`, return
   `CHOKE_LIMITED`.
3. **Reposition streamlines** (Tier ≥ 2). Invert the mass cumulative so each
   interior streamline sits at its target mass fraction; move it only a
   fraction `ω_sl` of the way (Wilkinson relaxation, §6.4) for stability.
   (Tier 1 has one fixed streamline — nothing to reposition.)
4. **Lagged refresh (AD-4).** Re-evaluate the row closures (swirl + loss) from
   the current field, **under-relaxed** (the swirl↔continuity Picard loop
   diverges otherwise); rebuild the transport steps; re-sweep `h0/s/rVθ`;
   **apply spanwise mixing** if enabled; lag `Vm` for the next iterate's
   `dVm/dm` term.
5. **Norms.** Compute all three §6.2.5 residual norms (position, continuity,
   closure). Converged when all are below tolerance.

Returns a `ClassicalResult` (status, assembled fields, the `FrozenInputs`
actually used, and the iteration history).

### 7.2 The residual (`assembly/assembler.py`, §6.1)

The elimination form. State rows: continuity per station `R_cont_j`, plus the
interior-streamtube mass-fraction errors `R_pos`, plus (back-pressure mode) one
throttling-station pressure row. The master ODE is integrated per station from
the immutable lagged distributions (PCHIP interpolants of `h0/s/rVθ` and the
curvature terms). Pure and side-effect-free — the Newton driver depends on
that.

### 7.3 The Newton driver (`drivers/newton.py`, §6.3)

Global Newton over the *pure* `ResidualAssembler.residual`. **Warm start
mandatory** (Newton is local) — seeded from a classical solve on the same grid.
Dense forward-difference Jacobian (the correctness baseline; a colored-FD
version is a recorded optimization), Armijo line search with a
crossing-streamline monotonicity guard (an infeasible trial gets merit `+∞` so
the line search backtracks rather than the assembler raising). Converges
quadratically near the solution (measured: V1c in ~3 iterations vs. ~15
classical) — used near choke/stall where Picard stalls.

### 7.4 Continuation & operability (`drivers/continuation.py`, §6.6–6.7)

`solve_speedline` marches a sequence of operating points (choke→stall) with
per-point warm start, cut-back on failure, and classical→Newton escalation. It
records mass-averaged PR and stall flags with the criterion that fired
(`solver_failure` / `pr_turnover` / `validity_saturated`). Back-pressure mode
(`BackPressureSpec`) adds `ṁ` to the state and one back-pressure residual row;
a hysteretic choke↔normal BC switch is wired into the traversal.

## 8. Package-by-package tour

- **`fluid/`** — `PerfectGas` working-fluid backend (`h`, `s`, `ρ`, `T`, `a`,
  stagnation conversions). Real-gas (CoolProp) is deliberately deferred behind
  the same interface (ARCH-9).
- **`geometry/`** — `WallCurve` (parametric hub/shroud, exact through the
  axial↔radial 90° bend via `atan2(r', z')`), `FlowPath` (walls + typed
  stations: `DUCT`/`EDGE_LE`/`EDGE_TE`/`INBLADE`), `ParamRowGeometry`
  (blade-row metal angles, C¹-in-span).
- **`grid/`** — q-o construction, streamline initialization (area rule), metric
  evaluation (curvature `κ_m`, angle `ε`), and the mass-cumulative
  quadrature/inversion used for repositioning.
- **`transport/`** — the station-to-station update of `h0/s/rVθ`
  (`streamwise.py`), the C¹ in-blade distribution schedules (`schedules.py`),
  and the §3.6 spanwise mixing operator (`mixing.py`).
- **`assembly/`** — the pure residual: `ResidualAssembler`, `FrozenInputs`,
  `pack`/`unpack`.
- **`drivers/`** — `classical`, `newton`, `continuation`.
- **`closures/`** — see [§9](#9-closures-machine-type-knowledge).
- **`machine/`** — the `Machine` facade that composes a flowpath + fluid +
  inlet + rows and reduces a solve to a `PerformanceResult`.
- **`diagnostics/`** — per-iteration convergence records.
- **`verification/`** — the V1…V9 cases (see [§11](#11-the-verification-ladder)).

## 9. Closures (machine-type knowledge)

All machine-specific correlations live here (AD-5), behind protocol interfaces
(`LossModel`, `SwirlClosure`, `BlockageModel`, `MixingModel`), bundled as named
`CorrelationSet`s with provenance strings.

| Set | Machine | Contents |
|---|---|---|
| `LIEBLEIN_NACA65` | Axial compressor | Lieblein/SP-36 incidence & deviation (Aungier fits) + equivalent-diffusion profile loss. |
| `KACKER_OKAPUU` | Axial turbine | Throat-based exit angle (Ainley) + K-O profile/secondary/trailing-edge/shock loss. |
| `CENTRIFUGAL` | Centrifugal/mixed | Wiesner slip `σ = 1 − √(cos β₂ᵦ)/Z^0.7` + representative incidence & skin-friction internal loss. |

`smoothmath.py` holds the C¹ primitives (`smoothstep`, `softplus`,
`smooth_max/min`, `soft_clip`, `logistic`, `blend`, …) everything above is
built from. Loss→entropy conversions are in `closures/conversions.py`
(Appendix B).

### Spanwise mixing (M8)

`transport/mixing.py` implements §3.6: an **implicit** (backward-Euler in `m`,
tridiagonal in `q`) diffusion of `{h0, s, rVθ}`. It is finite-volume with
zero-flux walls, so it **conserves** the mass-flux-weighted total of each field
and is **unconditionally stable**. The default `GallimoreMixing` coefficient is
`μ_mix = c_mix·ρ·Vm·r` with `c_mix = 5×10⁻⁴` (Gallimore–Cumpsty-calibrated,
2026-07; was `0.01`, ~20× too strong). It runs in the
driver's *lagged* refresh (never the residual path) and is gated by
`FidelityConfig.mixing_term`, so it is a strict no-op unless a case opts in.

## 10. What is and isn't established

**This is the most important section for anyone auditing or trusting the
code.** The verification ladder mixes a few rigorous quantitative gates with
many *structural* ones. Be precise about which is which.

**Rigorously established (quantitative, independent references):**
- **V1** analytic radial-equilibrium: solver matches closed-form free/forced
  vortex solutions; **grid-convergence order 1.94** measured (target > 1.7).
  This is the one place the *numerics* are pinned to an exact answer.
- **V3** tier consistency: Tier 2 ≡ Tier 3 **bit-for-bit** on straight-annulus
  vortex cases (asserted at 1e-10), and Tier 1 tracks the mass-averaged Tier 2
  to ~1e-3. Proves the "fidelity is data" claim mechanically.
- **Structural invariants**: conservation and C¹ properties of the mixing
  operator, the smoothmath primitives, the transport relations, geometry
  metrics through the 90° bend.

**Structural only (trends, bands, convergence — NOT quantitative validation):**
- **V4/V5** (axial compressor), **V6** (axial turbine), **V7** (centrifugal),
  **V8** (mixed-flow): each *converges*, does physically correct work with
  loss, and lands PR/efficiency in **generous plausibility bands**. They do
  **not** reproduce any specific published/NASA/Eckardt case point-by-point.

**Explicitly unvalidated (`[VERIFY]`), by design:**
- **Every correlation coefficient** in every `CorrelationSet` is `[VERIFY]` —
  they are representative fits, not calibrated against the reference library.
  Efficiencies consequently read high (e.g. V7/V8 ≈ 0.98) because deferred loss
  components (blade-loading/clearance/disk-friction) are missing.
- **Speedline/choke-traversal validation** against data is `[VERIFY]`.
- The mixing constant `c_mix` and the mixing entropy-*production* term (the
  operator redistributes `s`; the irreversibility source beyond redistribution
  is a refinement) are `[VERIFY]`.

**Known limitations (measured, honest):**
- **Tier-3 radial/mixed is slow (stabilized 2026-07; the fragility is
  resolved).** The original "narrow, angle-specific pocket" story was
  diagnosed as a driver artifact — the master ODE's `Vm = 0` singularity
  reached from stale boundary values (chiefly the unrelaxed closure
  switch-on), a fatal boundary check on a repairable state, and continuity
  roots accepted on spurious negative-`Vm` branches — **not** the §6.4
  odd-even repositioning mode (Appendix C.7/C.8, revised). Post-
  stabilization V7 (edge-only *and* subdivided) and V8 all converge at
  Tier 3, pinned by the flipped tripwires. What remains is speed: the §6.4
  relaxation throttle holds ω_sl ≈ 0.07 on these bends (V8 Tier 3 ≈ 400
  iterations); Newton finishing and a §6.4 recalibration on
  blade-row-coupled bends (C.3 was duct-calibrated, possibly
  artifact-contaminated) are the follow-ups. The M8-3 "mixing is a
  convergence prerequisite" claim fell with the same artifact; the surviving
  §3.6 claim — spanwise stratification without mixing — was then re-measured
  by the 2026-07 reference-calibration pass down from "~25×" to a **modest
  ~18%** (the "25×" ran on inflated + saturated loss; fixed the Lieblein ω̄
  inversion, calibrated `c_mix`→5e-4, and retuned V5 into the validity
  window). Mixing is a modest damping, not a homogenizer.
- Real-gas backend, JAX/AD backend, closure-in-Newton, endwall
  boundary-layer *model*, cooling/bleed flows: all deliberately deferred
  (ARCH-9).

The upshot for a reviewer: **a green suite here proves the machinery is
internally consistent and the numerics converge at the right order — it does
not prove the predicted performance numbers are physically accurate.** The
plausibility bands are wide by intent. The right adversarial question is
"which tests would still pass if a correlation were wrong?"

## 11. The verification ladder

`verification/` holds importable case definitions with independent references;
`tests/` binds each to Appendix C tolerances as permanent regressions.

| Case | What it exercises | Status |
|---|---|---|
| **V1** | Analytic REE, grid order | **Quantitative** (order 1.94) |
| **V2** | Curved annulus, full Tier 3 vs planar-limit | Quantitative-ish (planar limit; external CFD cross-check `[VERIFY]`) |
| **V3** | Tier consistency (Tier2≡Tier3, Tier1 mass-avg) | **Quantitative** (bit-for-bit) |
| **V5** | Axial compressor (single + **multistage**) | Structural (in-window loss after 2026-07 retune); multistage shows mixing is a **modest damping** (~18%), not a homogenizer |
| **V6** | Axial turbine (K-O set) | Structural |
| **V7** | Centrifugal impeller (first radial end-to-end) | Structural; Tier-3 needs INBLADE subdivision |
| **V8** | Mixed-flow (partial-φ bend) | Structural at all tiers (Tier 3 since the 2026-07 stabilization) |
| **V9** | Operability: surge flag + BC-switching | Structural (behaviour demonstrated) |

The multistage-V5 result (M8-3) is worth highlighting: **without mixing, the
two-stage compressor's hub/tip entropy split runs away (~40 J/kg·K) and the
solve fails outright even at 800 iterations; the shipped default mixing bounds
it (~0.7 J/kg·K) and recovers convergence.** That is §3.6's stated motivation,
confirmed categorically.

## 12. Fidelity tiers

| Concern | Tier 1 (meanline) | Tier 2 (REE) | Tier 3 (full SLC) |
|---|---|---|---|
| `n_sl` | 1 | ~5–11 | ~11–21 |
| Repositioning | off (fixed streamline) | on | on |
| `curvature_term`, `lean_term` | 0 | 0 | 1 |
| Spanwise mixing | off | off | opt-in |

All one kernel; the tier is `n_sl` + three float flags (AD-1), verified by V3.

## 13. Status and open items

**Milestones M0…M8 are all closed** — the entire ARCH-8 verification ladder.
CI-green at 347 tests. Post-ladder open items (no numbered milestone drives
them; recorded in `CLAUDE.md` and the memory files):

1. **Robust radial/mixed repositioning stabilization** — the top item; the V8
   Tier-3 blocker.
2. `[VERIFY]` correlation calibration against the reference library (all sets);
   the deferred centrifugal loss components.
3. The A.8 in-blade meridional force (`f_b,q = f_b,θ·tanλ`; zero for radial
   stacking, needs lean geometry + a master-ODE streamwise-gradient term).
4. Mixing `c_mix` calibration + the `Δs_mix` entropy-production term.
5. Numerical: colored-FD Jacobian (over the dense baseline), reproducer-bundle
   serialization.

## 14. Running and extending

```bash
pip install -e ".[test]"           # requires Python >= 3.14
pytest -q                          # full suite
python tools/check_imports.py      # ARCH-2 dependency-direction + AD-5
python tools/check_ad6.py          # AD-6 / smoothness lint
```

Run all three before considering any change done; they are CI gates.

**To add a closure** (e.g. a new loss model): implement the relevant protocol
in `closures/<machine_type>/`, build it only from `smoothmath` (C¹), convert
loss to entropy via `closures/conversions.py`, bundle it into a
`CorrelationSet`, and add a section-numbered test. Never let machine knowledge
leak out of `closures/` (AD-5).

**To add a verification case**: add a case in `verification/`, with an
*independent* reference (not the kernel's own output), and bind it in `tests/`
to an Appendix C tolerance. If you are adjudicating pre-existing code of
uncertain provenance, write the test suite from the spec **before** reading the
implementation (the `tests/test_grid_adjudication.py` precedent).

**Process discipline** (from `CLAUDE.md`, non-negotiable): C¹ everywhere on
flow arrays; sign conventions frozen in Appendix A (the manual wins over any
generated derivation); every numbered-spec module gets a section-numbered test;
run both lints + the suite before "done."
