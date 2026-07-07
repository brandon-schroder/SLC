# Guide 1 — Life of a Solve

> **What this document is.** One real call to `Machine.evaluate` traced end
> to end: every stage from composing the machine to the reduced
> `PerformanceResult`, with the actual data structures, their actual shapes,
> and actual numbers from runs at the header commit. Descriptive, not
> normative — the theory manual and architecture spec win on any
> disagreement. Section references like §6.2 are theory-manual sections;
> `G-5` is the geometry/grid module spec; `AD-n` are the binding
> architectural decisions (ARCH-1 / `CLAUDE.md`).
>
> Written at commit `d7b7b27` (2026-07-07); suite 373 tests green. Line
> numbers are stamped to that commit — the function names beside them are
> the durable anchors.

---

## 1. The runs we trace

Three runs of the V5 axial-compressor cases (`slcflow/verification/
v5_axial_compressor.py`), chosen because between them they exercise the
whole classical path:

| Run | Case | `n_sl` | Fidelity | Outcome |
|---|---|---|---|---|
| **A** | `V5AxialRotor` (single rotor) | 1 | Tier 1 meanline | `CONVERGED`, 63 outer iterations, PR = 1.1616, η = 0.8251 |
| **B** | `V5MultistageCompressor` (2 stages, 4 rows) | 9 | Tier 3 + mixing | `CONVERGED`, 90 outer iterations (~27 s), PR = 1.1799, η = 0.8752 |
| **C** | `V5AxialRotor` again | 9 | Tier 2 | `NUMERICAL_FAILURE` — deliberately included; see §8 |

Reproduce any of them in a REPL:

```python
from slcflow.verification.v5_axial_compressor import (
    V5AxialRotor, V5MultistageCompressor)
from slcflow.machine import FidelityConfig

perf_a = V5AxialRotor().evaluate(n_sl=1)                       # run A
perf_b = V5MultistageCompressor().evaluate()                   # run B
perf_c = V5AxialRotor().evaluate(n_sl=9,
                                 fidelity=FidelityConfig.tier2())  # run C
```

Run A is the primary thread of this guide; run B appears wherever the
spanwise machinery (repositioning, mixing) has something to do that a
single streamline doesn't.

## 2. Stage A — composing the machine

`V5AxialRotor.machine()` (`v5_axial_compressor.py:89`) assembles four
ingredients into a `Machine` (`slcflow/machine/__init__.py:106`). Nothing is
solved yet; this is pure description.

**The flow path** — two `WallCurve`s and a typed station list
(`v5_axial_compressor.py:77`):

- Cylindrical annulus: `wall_0` at r = 0.3 m, `wall_1` at r = 0.6 m, axial
  length 1.0 m. Note the AD-9 naming — the code never says "hub"; that
  `wall_0` happens to be the hub here is a property of *this* machine.
- Four stations, each a `StationDef(type, anchor_w0, anchor_w1)` where the
  anchors are arc-length fractions along each wall: `DUCT` at 0.0,
  `EDGE_LE` at 0.35 and `EDGE_TE` at 0.55 (both tagged `row_id="r1"`),
  `DUCT` at 1.0. `FlowPath` turns these into 4 oriented quasi-orthogonal
  (q-o) curves — straight span-wise cuts from `wall_0` to `wall_1`. This
  station set is the **immutable topology** (AD-8): it never changes during
  a solve; only where streamlines cross each q-o does.

**The blade row** — a `RowSpec` (`drivers/classical.py:115`) bundling
identity and physics sources: `row_id="r1"`, shaft speed ω = 400 rad/s,
`swirl=LIEBLEIN_NACA65.swirl`, `loss=LIEBLEIN_NACA65.loss`, and a
`ParamRowGeometry` with metal angles β₁ᵦ = −63°, β₂ᵦ = −45° (degrees at
this I/O boundary only, AD-7; radians inside), solidity 1.2, 31 blades.
The kernel never learns what "Lieblein" means — it holds two protocol
objects it will call through the §7.1 interfaces (AD-5).

**The inlet** — `InletCondition(h0=3.0e5, s=0.0, rvt=0.0)`
(`machine/__init__.py:45`): uniform stagnation enthalpy 300 kJ/kg, zero
entropy datum, axial (swirl-free) inflow. Scalars here; they are broadcast
to per-streamline profiles at solve time so the same machine serves any
`n_sl`.

**The fluid** — a `PerfectGas` (air-like defaults), behind the
`WorkingFluid` interface (§3.7): pure functions `T(h,s)`, `rho(h,s)`,
`p(h,s)`, `a(h,s)`.

**The operating point** is *not* part of the machine: it arrives per call
as a spec — here `MassFlowSpec(mdot=100.0)` (`types.py:70`), the
normal mode where mass flow is prescribed. (`BackPressureSpec` is the
choke-proximal alternative; §9.)

## 3. Stage B — `Machine.evaluate`: from description to grid

`Machine.evaluate(spec, fidelity, n_sl)` (`machine/__init__.py:131`) does
three small things and delegates:

1. **`GridTopology(flowpath, n_sl)`** (`grid/core.py:27`): fixes the target
   mass fractions `psi`. For run A, `psi = [0.5]` — the meanline is the
   streamline carrying the 50% mass-flow surface. For runs B/C,
   `psi = [0, 0.125, …, 1.0]` — nine surfaces including the walls. `psi`
   never changes during a solve: *"streamline i" means "the surface that
   should carry mass fraction ψᵢ"*, and repositioning (§D.5) is the act of
   moving it until it does.
2. **Inlet profiles**: `inlet.fields(topo.psi)` evaluates the scalars/
   callables into `(n_sl,)` arrays.
3. **Blockage** broadcast to `(n_sl, n_qo)` (zero here).

Then `solve_classical(...)` (`drivers/classical.py:335`) runs the actual
§6.2 scheme, and `_reduce` (§7) turns its result into a
`PerformanceResult`.

The fidelity argument deserves one pause. `FidelityConfig`
(`types.py:20`) is **three floats, not a mode switch**: `curvature_term`
and `lean_term` multiply the two Tier-3 terms of the master ODE, and
`mixing_term` scales the §3.6 spanwise mixing in the lagged refresh. Tier
2 is `(0, 0, 0)`; Tier 3 is `(1, 1, 0)`; Tier 1 *is* the Tier-2 flag set —
what makes it a meanline is `n_sl = 1`, a topology fact, not a flag
(AD-1). The assembler always evaluates every term and multiplies by these
floats; there is no `if tier == …` anywhere on the path you are about to
walk.

## 4. Stage C — `solve_classical` before the loop

### C.1 Resolving rows against the topology

`_resolve_rows` (`drivers/classical.py:129`) matches each `RowSpec` to its
stations by `row_id` and validates the shape loudly (`ConfigError`,
AD-10's "raise early at construction boundaries"): stations must run
`EDGE_LE, INBLADE*, EDGE_TE` on contiguous indices. For run A the row
resolves to `(j_le=1, j_te=2, t_stations=(1.0,))` — an edge-only row, one
LE→TE interval. (`t_stations` becomes interesting for centrifugal machines,
where M7 showed Tier-3 needs interior INBLADE stations; Guide 3.)

### C.2 Initialization (§6.2.1)

Three cheap guesses, one per unknown family:

- **Streamline positions**: `initialize_positions(topology)`
  (`grid/core.py:85`) places each streamline where the cumulative
  `∫ r dq` along each q-o reaches fraction ψᵢ — the **area rule** (G-5),
  exact for uniform `ρ Vm cos ε`. On run A's radial q-o's this puts the
  meanline at the RMS radius: r = √((0.3² + 0.6²)/2) = 0.4743 m — which is
  why every exit-radius printout in this guide reads 0.4743.
- **Transported fields**: one `sweep` (§C.3 below) of the inlet `(h0, s,
  rVθ)` through all-duct steps — the machine starts as an empty annulus;
  blade rows join from iterate 2, because closures need a flow field to
  evaluate against.
- **`Vm(q=0)` per station**: a 1-D continuity estimate
  `ṁ / (2π ρ₀ r̄ L_qo)` (`classical.py:400`).

### C.3 The transport sweep — how physics enters between stations

`sweep` (`transport/streamwise.py:97`) marches three per-streamtube fields
station to station through a list of `TransportStep`s, one per interval,
using the single universal update (`apply_step`, `streamwise.py:82`):

```
rVθ_out = rVθ_in            (duct)   or the step's target (blade row)
h0_out  = h0_in + ω · (rVθ_out − rVθ_in)      # Euler work / rothalpy, §3.3
s_out   = s_in + Δs_step                      # §3.5
```

There is no duct/stator/rotor branch: a duct is the default step
(`ω = 0`, conserve rVθ, `Δs = 0`), a stator is a step with a swirl target
and `ω = 0` (so no work), a rotor is a step with a swirl target and
`ω ≠ 0` (work enters *only* through the swirl change — §4.2). This is the
§8 degeneracy requirement in miniature.

Keep this picture: **the transported fields `(h0, s, rVθ)` are
`(n_sl, n_qo)` arrays that are *data* during a residual evaluation.** They
are re-swept once per outer iterate (§D.6), never inside the residual
(AD-3/AD-4).

## 5. Stage D — anatomy of one outer iterate (§6.2.2)

The outer loop (`classical.py:439`) runs up to `max_outer = 200` times.
Each iterate has five numbered sub-steps, matching §6.2.2.1–6.2.2.5. The
loop is bracketed by the two AD-10 boundary checks (`classical.py:441`,
`classical.py:471`): non-finite lagged inputs or assembled fields become a
typed `NUMERICAL_FAILURE` *before* they can crash scipy — run C dies at
exactly this fence.

### D.1 Freeze (§6.2.2.1)

Everything the residual may read is packed into an immutable
`FrozenInputs` (`assembly/inputs.py:62`): topology, fluid, fidelity
flags, the spec, the lagged transported fields, the lagged closure outputs
(`ClosureFields`), the lagged `Vm` field (for the lean term's `dVm/dm`),
and the lagged curvature field. Construction validates every shape loudly
(`ConfigError`) so that nothing downstream ever needs to raise — this
dataclass is *the* configuration boundary between "code that may throw"
and "code that may not" (AD-10).

Two of these members are worth flagging as *stabilizers*, both measured
necessities rather than design luxuries (full story in Guide 3):

- `kappa_lagged` / `kappa_relax` (§5.5): with the curvature term active,
  the curvature entering the ODE is an exponential moving average
  `0.3·κ_new + 0.7·κ_used_last` (`classical.py:465`). M3 measured that
  without this lag the streamwise odd-even mode diverges at *any*
  relaxation factor.
- `ClosureFields.row_exit_rvt` / `row_delta_s`: the closure outputs are
  not this iterate's — they are under-relaxed history (§D.6).

### D.2 The state vector

`pack(vm_q0, q_interior)` (`assembly/pack.py:34`) defines the unknowns:

```
x = [ Vm(q=0) at each station j ]  ++  [ q(i, j) for interior i, C-order ]
```

- Run A: `len(x) = 4` — four stations, and *no* position unknowns
  (`n_sl = 1` has no interior streamlines; the meanline sits at its frozen
  area-rule position, passed as `FrozenInputs.q_fixed`).
- Run B: `len(x) = 80` — 10 stations + 7 interior streamlines × 10.

Everything else — the whole `(n_sl, n_qo)` `Vm` field, densities,
temperatures — is *derived* from these unknowns. That compression is the
§6.1 **elimination form**, and the next sub-step is where it happens.

### D.3 `split(x)`: state → assembled picture

`ResidualAssembler.split` (`assembly/assembler.py:313`) is the pure
function (AD-3) that turns a state vector into an `AssembledFields`:

1. **Positions → metrics.** Attach the wall rows to the interior
   positions (`_full_q`), then `evaluate_metrics` (`grid/core.py:146`)
   fits a parametric C² cubic through each streamline's `(z, r)` nodes and
   reads off slope φ, curvature `κ_m`, lean ε (from tangent dot products,
   never angle arithmetic — G-9), and meridional arc length `m`.
2. **Lagged fields → interpolants.** For each q-o, PCHIP interpolants
   (§5.3 — monotone cubics, so no overshoot between nodes) of the frozen
   `h0(q)`, `s(q)`, `rVθ(q)`, plus the curvature and lean distributions
   (`_build_dists`, `assembler.py:157`). At `n_sl = 1` these degenerate to
   constants (`_Const`, `assembler.py:46`) — built anyway so the shape of
   the machinery is identical across tiers (AD-1).
3. **Master ODE → the `Vm` field.** For each station, integrate the §3.1
   master equation from the `q = 0` wall by RK2 (`_integrate`,
   `assembler.py:219`), with right-hand side (`_rhs`, `assembler.py:204`):

   ```
   dVm/dq = [ dh0/dq − T·ds/dq − (rVθ/r²)·d(rVθ)/dq ] / Vm     # REE core
          + curvature_term · Vm · κ_m cos ε                     # Tier 3
          + lean_term      · (dVm/dm)_lagged · sin ε            # Tier 3
   ```

   One boundary value `Vm(q=0)` therefore determines the whole spanwise
   profile — this is why momentum contributes **no residual rows**: it is
   satisfied by construction (§6.1). At `n_sl = 1` the node-to-node loop
   runs zero times and `Vm` *is* the boundary value.
4. **Thermodynamics.** `h = h0 − ½(Vm² + Vθ²)`, then `T`, `ρ`, and the
   meridional Mach from the fluid.

`split` is called at least once per iterate and — crucially — inside every
continuity evaluation below, at trial `Vm(q=0)` values. It is the
workhorse; its purity is what later lets the Newton driver
finite-difference the residual safely.

### D.4 Station march (§6.2.2.2): one scalar root-find per q-o

For each station j, the driver solves the scalar continuity equation

```
F_j(Vm_q0) = 2π ∫ ρ Vm cos ε (1−B) r dq  −  ṁ  =  0
```

with `brentq` (`_solve_qo`, `classical.py:260`), where every evaluation of
`F_j` integrates the master ODE across the span and then the mass flux
with **the** shared quadrature rule (`mass_cumulative`,
`assembler.py:240` — the same rule that repositioning inverts; §5.4's
consistency requirement is precisely this "the").

The physics shape of `F_j` (A.7): rising from −ṁ at `Vm_q0 → 0` up to a
**capacity peak**, then falling as compressibility wins. Three
consequences visible in the code:

- The root-finder deliberately brackets the **subsonic branch** — the scan
  looks for the first sign change *below* the peak (§6.5 branch selection).
- If the peak never reaches zero, this q-o physically cannot pass ṁ: the
  driver returns `CHOKE_LIMITED` naming the station (`classical.py:489`) —
  a typed result, not an error.
- Trial velocities that push static enthalpy negative produce non-finite
  F by design; `_solve_qo` maps them to a huge mass deficit so `brentq`
  bisects back toward the physical branch instead of raising (AD-10 on
  the driver side).

A warm bracket around the previous iterate's root (`±30%`) makes the happy
path cheap; the 64-point cold scan is the fallback.

Run A after convergence: `Vm(q=0) = [91.16, 91.16, 84.46, 84.46]` m/s —
identical within the duct pairs (same annulus, same transported fields on
both sides of a duct interval) and dropping across the rotor because
compression raised ρ.

### D.5 Repositioning (§6.2.2.3): move streamlines toward their ψ

For `n_sl > 1`: on each q-o, take the *same* mass cumulative used above,
and invert it — find the q where the cumulative reaches `ψᵢ · ṁ/(2π)`
(`invert_cumulative`; `classical.py:501`). That target is where streamline
i *should* sit. Then move only a fraction of the way:

```
q_next = q_full + ω_sl · (q_target − q_full)
```

`ω_sl` is the **Wilkinson relaxation factor** (`_omega_sl`,
`classical.py:305`), the single most safety-critical constant in the
driver. With the curvature term active it is throttled to

```
ω_sl = 4.4 · (1 − Mm²) · (Δm_min / L_qo)^1.5     (capped at 0.7)
```

— the form *measured* in the M3-3 calibration (Appendix C.3,
`tools/calibrate_wilkinson.py`), which deliberately disagrees with the
literature's (Δm/Δq)² aspect-ratio rule (both its failure directions were
measured; Guide 3). With curvature off (Tiers 1–2) repositioning is a
plain fixed point and runs at the user cap 0.7. In run B, ω_sl settles at
0.6031 — genuinely throttled below the cap by station density. Run A skips
this step entirely (`ω_sl = 0` in its records): one streamline, nothing to
move.

Why relax at all? Because positions feed curvature feeds the ODE feeds the
mass distribution feeds positions — a feedback loop whose gain grows with
station density. Under-relaxation is how the classical scheme keeps that
loop contractive; the failure mode when it isn't is the odd-even
streamwise zigzag (Guide 3).

### D.6 Lagged refresh (§6.2.2.4, AD-4): the closures speak

Only now — outside the residual, once per iterate — does machine-specific
physics run:

1. **Evaluate closures** (`_evaluate_rows`, `classical.py:203`). For each
   row, build a `RowFlowView` of the *current* LE flow (velocities, angles,
   thermodynamic state, plus lagged TE quantities for iterative
   closures) and make one call each to `swirl.exit_rvt(row, view)` and
   `loss.evaluate(row, view)`. For run A the Lieblein set returns the exit
   `rVθ` implied by incidence/deviation and an entropy rise converted per
   Appendix B, with a validity ∈ [0,1] (0.979 at convergence — slightly
   off the correlations' comfort zone, reported, not raised; §7.3.3).
2. **Under-relax the outputs** (`classical.py:531`). New outputs are
   blended into the lagged ones with `closure_relax = 0.25`: measured at
   M4, the swirl↔continuity Picard loop *diverges* at 0.5 on staggered
   rows (loop gain ~ tan β₂). This single constant is why run A takes 63
   iterations for a converged flow field that barely changes after
   iterate 10 — see §6.
3. **Rebuild transport steps and re-sweep.** `row_steps`
   (`streamwise.py:138`) turns the relaxed `(rVθ_te, Δs)` into the row's
   `TransportStep`s (C¹ in-blade schedules distribute them when INBLADE
   stations exist); one `sweep` refreshes `(h0, s, rVθ)` everywhere.
4. **Mixing** (run B only): `mix_transported` (`transport/mixing.py`)
   applies the §3.6 implicit spanwise diffusion to the re-swept fields,
   with coefficients from this iterate's flow. Never on the residual path;
   `mixing_term = 0` is bit-identical to a plain solve.
5. **Lag `Vm`** for the next iterate's lean term.

The AD-4 contract in one sentence: *the residual only ever sees constants;
everything that feeds back does so between iterates, under-relaxed.*

### D.7 Norms and the convergence test (§6.2.2.5)

Three relative norms, all recorded every iterate (`IterationRecord`):

- `cont_norm`: worst |F_j| / ṁ — essentially machine-zero every iterate,
  because the station march just solved it (≈10⁻¹³ throughout both runs).
- `pos_norm`: largest streamline move / span.
- `closure_norm`: largest relative change of the lagged closure outputs.

Converged means **all three** below 10⁻⁹. The three norms tell you *which*
coupling is the bottleneck — and the records say it plainly:

## 6. What convergence actually looked like

Run A (meanline rotor — no positions, so the closure loop is the whole
story):

| iterate | cont_norm | pos_norm | closure_norm |
|---|---|---|---|
| 1 | 2.0e−13 | 0 | 1.0 (closures just switched on) |
| 32 | 1.3e−13 | 0 | 9.3e−6 |
| 63 | 1.3e−13 | 0 | 9.1e−10 → **CONVERGED** |

The closure norm contracts by an almost perfectly constant factor ≈ 0.71
per iterate — the fixed-point contraction rate set by `closure_relax`
times the physical loop gain. That geometric tail is the visible cost of
Picard: the answer stops changing long before the norm crosses 10⁻⁹.
(The Newton driver exists exactly to replace this tail with quadratic
steps; §9.)

Run B (Tier 3, 9 streamlines, 4 rows, mixing on):

| iterate | cont_norm | pos_norm | closure_norm | ω_sl |
|---|---|---|---|---|
| 1 | 0 | 2.4e−7 | 1.0 | 0.6227 |
| 2 | 4.8e−14 | 6.9e−2 | 6.5e−1 | 0.5877 |
| 23 | 7.7e−15 | 7.9e−5 | 4.5e−3 | 0.6031 |
| 90 | 8.4e−15 | 4.3e−12 | 9.4e−10 | 0.6031 | **CONVERGED** |

Positions and closures converge together — they are one coupled fixed
point, which is the whole point of the nested scheme. The M8 headline
lives in this run's exit entropy: spanwise spread 0.69 J/(kg·K) *with*
mixing; the identical case with `mixing_term = 0` runs away to a
~40 J/(kg·K) hub/tip split and fails outright (Appendix C.5m,
`tests/test_multistage_mixing.py`) — mixing is a convergence
prerequisite for multistage axial, not a smoother.

**A consistency check you can do by eye.** Run A's exit: `rVθ = 39.74
m²/s`, `h0 = 315 896 J/kg`. Euler's work: `Δh0 = ω·ΔrVθ = 400 × 39.74 =
15 896 J/kg` — exactly the observed `315 896 − 300 000`. Not a coincidence
and not a test tolerance: the transport update *is* this identity (§C.3),
so it holds to round-off by construction. Similarly the exit flow angle:
`Vθ = 39.74/0.4743 = 83.8 m/s` against `Vm = 84.5` gives α = 44.8°, and in
the relative frame β₂ = atan2(83.8 − 400·0.4743, 84.5) = −51.4° versus the
−45° metal angle — about 6.4° of deviation, which is the Lieblein
deviation correlation doing its job (under-turning, the physically correct
sign).

## 7. Stage E — reduction to `PerformanceResult`

`Machine._reduce` (`machine/__init__.py:164`) compresses the converged
field into the scalars an MDO loop wants:

- Mass-flux-weighted span averages (`_mavg`, weight `ρ Vm cos ε r` — the
  §3.2 continuity weight, so "average" means "what the mass flow sees") of
  inlet/exit `p0` and `h0`. At Tier 1 the single point *is* the mass
  average by construction.
- `PR = p0_exit / p0_inlet`; total-to-total isentropic efficiency from
  the perfect-gas isentrope `h0s = h0_in · PR^((γ−1)/γ)`. The same
  expression reads > 1 for a turbine (Δh0 < 0) — a documented sign
  convention, not a code branch.
- Spanwise exit profiles (`r`, `Vm`, α, `p0`, `T0`) and the aggregate
  closure validity.
- The full `ClassicalResult` rides along (`result=`) as the deterministic
  replay/warm-start handle: status, final `x`, `AssembledFields`, the
  exact `FrozenInputs` of the last iterate, and the complete per-iteration
  record (ARCH-6).

Run A: `PR = 1.1616, η = 0.8251, validity = 0.979`. Run B:
`PR = 1.1799, η = 0.8752, validity = 0.9785`. Remember the standing
caveat (overview §10): these are *structurally* sane numbers from
uncalibrated (`[VERIFY]`) correlations, not validated predictions — η
reads high wherever deferred loss components are missing.

## 8. When it fails: run C, and the typed-status contract

The driver never throws on physics. Its outcomes are a closed set
(`SolveStatus`): `CONVERGED`, `MAX_ITER`, `CHOKE_LIMITED` (a station's
capacity peak sits below ṁ — an operability fact, §6.6), and
`NUMERICAL_FAILURE` (an AD-10 boundary check tripped; the record's
`reason` string says which and when).

Run C is the instructive one. The same rotor that converges as a meanline
fails spanwise in 11 iterates:

```
status: NUMERICAL_FAILURE
reason: non-finite assembled fields at outer iteration 11 (AD-10 boundary check)
it6 :  pos=3.4e-2  closure=2.0e-1        # already ragged, not contracting
it10:  pos=9.4e-2  closure=1.6           # diverging; validity -> 0.0
Vm(q=0) = [94.4, 93.0, 2.8, 2.8]         # mid-machine velocity collapse
```

Why: `V5AxialRotor`'s metal angles are **mid-span values** (its docstring
says so — a spanwise run should pass spanwise arrays via `geometry`).
Across the wide 0.3–0.6 m span the blade speed doubles, so the relative
inlet angle swings ≈ 52° (wall_0) to 69° (wall_1) against a uniform 63°
metal angle — roughly −11°/+6° of incidence at the walls. The loss
correlation, driven far off-design, charges enormous entropy near the
walls (17–19 J/(kg·K) at two nodes versus ~2 mid-span in the last
recorded field), density and Vm collapse there, and within a few
iterates the assembled fields go non-finite — at which point the boundary
check converts the mess into a typed status instead of a traceback.

Two durable lessons, both bought with real debugging time in this
project's history (see the memory note "verification case-design
gotchas"): **the convergence record is the first diagnostic** — a
closure norm that stops contracting around iterate 5–6 is a case-physics
smell, not a driver bug; and **check the case design before blaming the
solver.**

## 9. What changes off the happy path (forward pointers)

Everything above is the classical Picard driver — the default far from
operability limits. Three escalations reuse the same pure residual
(covered properly in Guide 3):

- **Newton** (`drivers/newton.py`, §6.3): stacks the *same* rows —
  `n_qo` continuity residuals + interior mass-fraction residuals
  (`ResidualAssembler.residual`, `assembler.py:334`) — into a global
  system, dense-FD Jacobian, Armijo line search with a
  crossing-streamline guard. Warm start mandatory; measured quadratic
  (V1c: ~3 iterations vs 15 classical). This is what the 0.71-per-iterate
  tail of §6 buys its way out of.
- **Back-pressure mode** (`BackPressureSpec`, §6.6): ṁ joins `x` as the
  trailing unknown and the assembler appends one row — exit static
  pressure at the throttling station — making the choke-proximal branch
  solvable where the ṁ-specified form's Jacobian goes singular.
- **Continuation** (`drivers/continuation.py`, §6.7): `solve_speedline`
  marches operating points choke→stall with per-point warm starts,
  classical→Newton escalation, and hysteretic BC switching.

(One seam to know about: `Machine.evaluate`'s `warm_start` parameter is
accepted for interface stability but not yet consumed — warm starting is
currently plumbed at the `solve_classical`/continuation level.)

## 10. Stage-by-stage map

| Stage | Code | Spec |
|---|---|---|
| Compose machine | `machine/__init__.py:106` `Machine` | ARCH-5.5 |
| Grid topology, ψ | `grid/core.py:27` `GridTopology` | §2.5, G-4 |
| Area-rule init | `grid/core.py:85` `initialize_positions` | G-5 |
| Transport sweep | `transport/streamwise.py:97` `sweep` | §3.3–3.5 |
| Freeze | `assembly/inputs.py:62` `FrozenInputs` | ARCH-3.3, AD-3/4/10 |
| State vector | `assembly/pack.py:34` `pack` | §6.1, ARCH-3.2 |
| Metrics | `grid/core.py:146` `evaluate_metrics` | G-6, §5.1–5.2 |
| Master ODE | `assembly/assembler.py:204` `_rhs` / `:219` `_integrate` | §3.1/§5.3, A.5 |
| Continuity / capacity | `assembler.py:240` `mass_cumulative`, `:269` `continuity_F`, `:275` `qo_capacity` | §3.2/§5.4, A.7 |
| Station march | `drivers/classical.py:260` `_solve_qo` | §6.2.2.2, §6.5 |
| Repositioning + ω_sl | `classical.py:305` `_omega_sl`, `:501` | §6.2.2.3, §6.4, C.3 |
| Closure refresh | `classical.py:203` `_evaluate_rows`, `:531` under-relax | §6.2.2.4/§6.2.4, AD-4 |
| Mixing | `transport/mixing.py` `mix_transported` | §3.6 |
| Norms | `classical.py:572` | §6.2.5 |
| Reduction | `machine/__init__.py:164` `_reduce` | ARCH-5.5, §3.2 weight |

## 11. Check your understanding

1. **Why does momentum contribute no rows to the residual vector?**
   Because the master ODE is integrated *exactly* (to RK2 order) from the
   single boundary value `Vm(q=0)` during assembly — momentum is satisfied
   by construction, and only continuity (per station) and streamtube mass
   fractions (per interior streamline) remain as equations (§6.1, §D.3).
2. **Run B has 9 streamlines and 10 stations. Why is `len(x) = 80` and
   not 90?** The two wall streamlines are not unknowns — walls are
   geometry (AD-8). 10 × `Vm(q=0)` + 7 interior × 10 positions = 80.
3. **The flow field of run A is visually converged by iterate ~10. Why
   does the driver run 63 iterates?** The closure↔continuity Picard loop
   contracts at ≈ 0.71/iterate (under-relaxation 0.25 × loop gain), and
   the gate is 10⁻⁹ on the *closure-update* norm — a geometric tail, the
   documented cost of lagged closures (AD-4) that Newton exists to remove.
4. **What, physically, is `CHOKE_LIMITED`?** At some station, the mass-flow
   capacity function F_j + ṁ peaks below the requested ṁ: no subsonic
   `Vm(q=0)` can push the demanded flow through that q-o (A.7). It is a
   statement about the operating point, not a solver malfunction.
5. **Where is the only place machine-type knowledge (Lieblein, K-O,
   Wiesner) executes, and when?** Inside `_evaluate_rows`, in the lagged
   refresh — once per outer iterate, through the §7.1 protocol interfaces,
   never inside the residual (AD-4, AD-5). The residual only ever sees the
   frozen, under-relaxed outputs.
6. **Why did run C fail while run A converged on the same machine?** Not a
   solver defect: the case supplies mid-span metal angles only, so a wide
   spanwise run puts the wall streamlines ~±10° off-design in incidence;
   the uncalibrated loss closure charges wall entropy spikes, `Vm`
   collapses mid-machine, and the AD-10 boundary check converts the
   resulting non-finite fields into a typed `NUMERICAL_FAILURE` with a
   reason string (§8).
