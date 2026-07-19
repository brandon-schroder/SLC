# slcflow — System Overview & Guided Tour

> **What this document is.** A reader's guide to the whole codebase: what has
> been built, how the pieces fit, and — most importantly — **how a solve
> actually runs end to end**. It is descriptive, not normative: the three
> frozen specs in `docs/` are the source of truth, and code conforms to them.
> Where this guide states what is *proven* versus *structural* versus
> *unvalidated*, it tries to be scrupulously honest — see
> [§10 What is and isn't established](#10-what-is-and-isnt-established).
>
> Written after milestone M8 closed (the last on the ARCH-8 ladder) and kept
> current through the post-ladder work — the independent audit, the
> reference-library calibration pass, the axial-compressor endwall/clearance/
> shock loss stack, and the post-ladder **meridional-supersonic-branch driver**
> ([§7.5](#75-the-meridional-supersonic-branch-driver-driverssupersonicpy-66--c9)).
> Suite: **534 tests**, both lint gates green.

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
drivers/            Solution algorithms: classical (Picard), newton, continuation,
   │                  supersonic (pseudo-arclength meridional-branch continuation)
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

### 7.5 The meridional-supersonic-branch driver (`drivers/supersonic.py`, §6.6 / C.9)

Each station's continuity is **folded at the sonic meridional condition
`M_m = 1`**: below the station capacity there are two `Vm(q=0)` roots — a
subsonic-meridional and a supersonic-meridional one — that coalesce at the
capacity peak, where the continuity Jacobian is singular. The classical driver
takes the subsonic root by construction, and *natural-parameter* continuation
(stepping `ṁ`, or `p_exit` in back-pressure mode) cannot cross the peak — it
chokes or pins at `M_m = 1`. Reaching the supersonic branch needs
**pseudo-arclength (Keller) continuation**: parametrise the solution curve in
`(state, ṁ)` by arclength so the *augmented* Jacobian stays non-singular **at**
the fold, and walk from the subsonic branch, through the sonic turning point,
onto the supersonic branch. `solve_supersonic_branch` then lands the exact
on-target supersonic root with a fixed-`ṁ` Newton (the branch is selected, so
that root is regular). The `ṁ` Jacobian column is analytic (`ṁ` enters
continuity linearly); the state columns are finite-difference with the Newton
positive-`Vm` guard; **variable scaling is mandatory** (an unscaled arclength
creeps in `ṁ` because `Vm` dominates the norm near the fold).

Two paths, selected by whether closure-fed `rows` are supplied:

- **Prescribed transport (duct)** — closures constant along the branch: one
  arclength crossing plus the landing Newton. Verified against the isentropic
  area–Mach relation on a purpose-designed converging–diverging **nozzle**: the
  classical driver chokes above the throat capacity, while this driver crosses
  it and lands the supersonic throat Mach to `<0.3%` of the analytic root
  (`M_m ≈ 1.40` at the sample), inlet/exit staying subsonic (a rank-1 fold).
- **Closure-lagged blade rows** — the flow-dependent swirl/loss closures must
  re-lag at the supersonic field. The driver bootstraps onto the branch by
  arclength *once*, then hands the supersonic seed to `solve_newton`, whose
  existing outer quasi-Newton closure-lag loop (§6.3) re-lags to
  self-consistency (the fold is behind the seed, so the positive-`Vm` guard
  keeps Newton on the supersonic branch). Verified on a Lieblein row upstream of
  the throat.

**Honest scope.** This handles a *single* dominant fold (the binding station's
`M_m = 1`). A fully supersonic **row inflow** that folds several stations at
once (measured on a transonic rotor, C.9) is the harder **multi-fold** regime
this does not claim. And it is **not** a V5 blocker — the transonic-V5 gate is
met on the *ordinary* branch (the in-window condition is blade loading `D_eq`,
not the meridional branch; see [§10](#10-what-is-and-isnt-established)); this
driver is a general, independently-verified capability.

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
- **`drivers/`** — `classical`, `newton`, `continuation`, `supersonic`
  (pseudo-arclength meridional-branch continuation, [§7.5](#75-the-meridional-supersonic-branch-driver-driverssupersonicpy-66--c9)).
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

**Point-by-point measured validation (2026-07 — landing within a few %):**
- **V4/V5** (axial compressor): NASA Rotor 37 (TP-1659) with the grounded
  Çetin transonic-deviation correction lands Tier-2 PR **+0.2%** of measured,
  with a matched-PR back-pressure traversal; Rotor 38 (TP-2001) second point.
- **V6** (axial turbine): NASA TN D-6967 two-stage agrees **~1%** in the
  matched-PR frame, with a digitized multi-speed map (±2.2%) and a
  first-stage build; VKI LS-89 cascade at the closure level.
- **V7** (centrifugal): THREE measured points — Eckardt O (stage PR +1.0%),
  Krain, and NASA CC3 (real high-speed 4:1) — the stage set closing to
  PR ±2% / η ±2 pt with one calibrated constant.
- **Operability** (gate #5): a grounded tip-diffusion-factor stall criterion
  predicts the measured Rotor 37/38 stall within ~3%.

**Structural only (trends, bands, convergence — no measured rig):**
- **V8** (mixed-flow): converges, does physically correct work with loss, and
  lands PR/efficiency in plausibility bands, but no open measured mixed-flow
  rig exists to reproduce point-by-point. (Tier-3 on tight radial/mixed bends
  is fast and robust in the interior — V8 accelerated 2.6× — but the pocket
  *edges* and V7's tight synthetic bend need closure-in-Newton; see the
  known-limitations note below.)

**Grounded and dispositioned (2026-07 reference-calibration + validation):**
- **The correlation coefficients were grounded verbatim against the reference
  library** (a systematic pass over every `CorrelationSet`), not left as
  representative fits — real bugs were found and fixed (Lieblein ω̄ inversion,
  K-O TE curves, the centrifugal `D_f` loading ratio), and each candidate
  refinement (K-O TE level, Zhu-Sjolander, Wiesner slip, tip-resolved
  supercritical, the λ work-input role) was **dispositioned by measurement**
  with zero constants tuned to individual points. Centrifugal losses are now
  complete through parasitic (disk/leakage/recirculation) + vaneless-diffuser
  marching + λ tip-distortion; efficiencies read realistic (centrifugal stage
  η ±2 pt vs measured).
- **Residual `[VERIFY]`s** are genuine refinements, not calibration gaps: the
  backswept-centrifugal work over-prediction (the λ work-input role, a
  recorded 3-point trend needing a joint recalibration), point-by-point
  choke-side speedline *shapes* (a capacity/knee matter, dispositioned for
  Rotor 37), and a few geometry inputs on the newest cases that are recorded
  estimates where the coordinate report wasn't in hand.
- The mixing constant `c_mix` and the mixing entropy-*production* term (the
  operator redistributes `s`; the irreversibility source beyond redistribution
  is a refinement) are `[VERIFY]`.

**Known limitations (measured, honest):**
- **Tier-3 radial/mixed convergence is the top open item; the 2026-07 diagnosis
  split the "wedge" into two distinct diseases.** Adding the dominant
  centrifugal blade-loading loss (~7 kJ/kg, the correct physics that makes η
  realistic) initially broke both V7 and V8 spanwise bends; characterization
  (`probe_cin_*`, memory `wedge-closure-in-newton`) then separated them:
  - **V7 Tier 2 = an operating-point stratification-capacity fold (CRACKED).**
    The stratified realistic loss drives an *interior* streamtube's `Vm` toward
    the master-ODE `Vm = 0` singularity, so below a mass-flow floor the
    **coupled** flow folds — even though each station's capacity individually
    stays `≫ ṁ` (the fold is coupled, not per-station: the old "no positive
    root at any `ṁ`" reading checked the wrong thing). It is **not** closure
    coupling (fixed prescribed stratified transport folds identically) and
    **not** the classical repositioning algorithm (a global Newton folds too);
    **closure-in-Newton was measured not to help**, and the fold terminates *at*
    the singularity so continuation has no far side. **Raising `ṁ` lifts every
    `Vm` off the singularity:** re-centring V7 to `ṁ = 17` (window ≈ [16, 20])
    makes Tier 2 a **passing** test with realistic loss (PR 2.03, η 0.83, with
    the corrected Coppage/Oh-1997 blade-loading loss — CENT-LOSS.md). Same
    category as the V5/transonic-V5 retunes.
  - **V7 Tier 3 = a physical feasibility fold at realistic loss (OPEN, NOT a
    solver gap).** Raising `ṁ` does *not* fix it. A damped-Newton +
    curvature-strength continuation (2026-07) showed the flow branch *folds*
    (interior `Vm→0`) at ~9% of full Tier-3 curvature at `ṁ`=17 (~26% at 20):
    the tight 0.08 m bend (κ~20) + realistic-loss stratification drive an
    interior streamtube to `Vm→0` (incipient reversal). **No positive-Vm root
    exists** → a stiff integrator / compact-support fit / damped Newton *cannot*
    help; this is **not** the robust-repositioning item. The case-side
    "calibrated/lower loss" lever was **tried** (the 2026-07-12 `D_f` ratio fix,
    ~2.3× less loss): it *eased* the fold (Tier 3 now fails at sane PR/η ~2.3/0.9
    rather than garbage) but did **not** crack it — Tier 3 still fails at every
    `ṁ` in 13…32. Stays an `xfail` tripwire.
  - **V8 Tier 3 = now converges at the re-centred `ṁ = 14` (2026-07-12).** With
    the earlier over-estimated loss this was a narrow choke/max-iter pocket
    (`xfail` at `ṁ=12`); the `D_f` ratio fix (~2.3× less loss → ~27–30% less
    stratification) LOWERED the pocket into a converging window `ṁ ∈ {13, 14}`
    (choke at 12, slow-max-iter at 15/16). At the re-centred `ṁ = 14` all three
    tiers converge (validity 1, Tier 3 PR agrees Tier 2 to ~2.5%) — still slow
    (`ω_sl ≈ 0.066`; Newton finishing / §6.4 recalibration are the recorded
    acceleration follow-ups). This is now a **passing** test; V7's tighter 90°
    bend is not liftable the same way.

  The V7 T3 `xfail(strict=True)` tripwire auto-flips if a fix makes it XPASS.
  Reasonable robustness patches (reposition-freeze, capacity-peak freeze) were
  re-measured non-curative and reverted. See Appendix C.7/C.8 and memory
  `wedge-closure-in-newton`, `centrifugal-blade-loading-wip`. The M8-3 "mixing is a
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
| **V7** | Centrifugal impeller (first radial end-to-end) | Structural at Tier 1+2 with realistic loss (re-centred `ṁ`=17, η≈0.80); **Tier 3** is a physical feasibility fold (interior `Vm→0`) at realistic loss — infeasible at every non-choked `ṁ`, not a solver gap (`xfail` tripwire) — see §10 |
| **V8** | Mixed-flow (partial-φ bend) | Structural at Tier 1+2 with realistic loss; **Tier 3** a narrow pocket (converges only at `ṁ`≈15, choke_limited at 12) — `xfail` tripwire — see §10 |
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
CI-green at 534 tests. Post-ladder *deliveries* (recorded in `CLAUDE.md`)
include the independent audit + turbine-sign fix, the reference-library
calibration pass, the endwall/clearance/shock loss stack, the colored-FD
Jacobian, and the **meridional-supersonic-branch driver**
([§7.5](#75-the-meridional-supersonic-branch-driver-driverssupersonicpy-66--c9)).
Post-ladder open items (no numbered milestone drives them; recorded in
`CLAUDE.md` and the memory files):

1. **Radial/mixed Tier-3 acceleration** (was "robust stabilization") — the
   2026-07 diagnosis + 2026-07-12 blade-loading calibration narrowed this to a
   *speed* problem on the cases that converge. **V7 Tier 3** is off this item: a
   physical feasibility fold (interior `Vm→0`, no positive-Vm root) that the
   ~2.3× loss reduction *eased but did not crack* — case-side levers only.
   **V8 Tier 3 now converges** at the re-centred `ṁ=14` (a **passing** test) but
   is **slow** (`ω_sl ≈ 0.066`). The open work is acceleration: Newton finishing,
   a compact-support streamline fit, or §6.4 recalibration. Closure-in-Newton was
   measured **not** to help the V7 folds.
2. `[VERIFY]` correlation calibration against the reference library (all sets);
   the deferred centrifugal loss components.
3. The A.8 in-blade meridional force (`f_b,q = f_b,θ·tanλ`; zero for radial
   stacking, needs lean geometry + a master-ODE streamwise-gradient term).
4. Mixing `c_mix` calibration + the `Δs_mix` entropy-production term.
5. Numerical: reproducer-bundle serialization; closure-in-Newton; a multi-fold
   continuation for a fully supersonic row inflow ([§7.5](#75-the-meridional-supersonic-branch-driver-driverssupersonicpy-66--c9)).

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
