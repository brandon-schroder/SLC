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
- **M4** (axial-compressor correlation set + Tier 1 mode, V4/V5) — closed.
  Ran as five reviewed sub-steps on `main`: (1) `closures/interfaces.py`
  (ARCH-4.2 protocol layer) + `simple.py` prescribed closures + closure-fed
  blade-row wiring in the driver (AD-4 lagged eval, §6.2.2.4; the carried
  `rvt_le` consistency test landed here). (2) `closures/conversions.py` —
  Appendix B loss→entropy conversions (B.1 rotor re-referencing, B.2/B.3/B.4)
  with the assert-don't-clamp KE guard. (3) `geometry/bladerow.py`
  (`ParamRowGeometry`, C¹-in-span PCHIP) + `closures/axial_compressor/
  lieblein.py` — Lieblein/SP-36 incidence/deviation (Aungier fits), and the
  §6.2.4 closure under-relaxation discovered *necessary* here (the swirl↔
  continuity Picard loop diverges otherwise). (4) `axial_compressor/loss.py`
  — equivalent-diffusion profile loss + `LIEBLEIN_NACA65` CorrelationSet;
  measured `closure_relax = 0.25` (0.5 diverges), §7.3.2 ω̄ ceiling to keep
  B.2 in-domain, and an AD-10 gap closed (negative-Vm repositioning → typed
  `NUMERICAL_FAILURE`). (5) `machine/` facade (ARCH-5.5,
  `Machine.evaluate → PerformanceResult`) + the n_sl = 1 **Tier-1 meanline**
  (lifts the `ResidualAssembler` n_sl ≥ 2 constraint; one-point area rule as
  the coarsest §5.4 quadrature, repositioning off — one kernel, no tier
  branch per AD-1), the V3 Tier-1 mass-average clause (Appendix C.4), and the
  V5 axial-compressor entry point. **V4/V5 status is structural** (anchors,
  trends, bands, C¹ sweeps, end-to-end convergence): point-by-point
  published-figure / NASA-data reproduction and speedline/choke traversal
  are `[VERIFY]`, blocked on the reference-library correlation calibration
  (every Lieblein coefficient is `[VERIFY]`) and the M5 continuation driver.
  Still-open carryovers deferred past M4 (not needed until in-blade/mixing
  data exists): moving `t_stations` validation into `ClosureFields`/
  `FrozenInputs` for in-blade schedules; an `assert_valid_schedule`
  contract-test helper (§7.3.4) with the first non-default
  `DistributionSchedule`.
- **M5** (Newton driver + continuation + BC switching, V9 operability) —
  closed. Ran as four reviewed sub-steps on `main`: (1) `drivers/newton.py`
  — global Newton over the pure `ResidualAssembler.residual` (§6.3): **dense**
  forward-difference Jacobian (the correctness baseline the ARCH-5.3
  colored-FD version must match column-for-column — recorded as the next
  optimization, not a prerequisite), Armijo line search with the
  crossing-streamline monotonicity guard folded in (closes the M2/M3 AD-10
  carryover for the Newton path), warm start mandatory. Measured quadratic:
  V1c in ~3 iterations vs. 15 classical. (2) `drivers/continuation.py`
  `solve_speedline` (§6.7) + `solve_classical` `warm_start` seed — choke→stall
  traversal, per-point warm start, cut-back, classical→Newton escalation,
  §6.6 annulus choke margin + mass-averaged PR, stall flags with recorded
  criterion (`solver_failure`/`pr_turnover`/`validity_saturated`). (3)
  `BackPressureSpec` residual form (§6.6): `mdot` joins the state, the
  assembler appends the throttling-station back-pressure row, `FrozenInputs`
  accepts it (the M2 stub-and-reject lifted); Newton solves it — round-trip
  verified against normal mode. Then the hysteretic choke↔normal BC-switch
  wired into the traversal (auto + logged + `c_sw`/`δ_hys` band, no
  limit-cycling). (4) `verification/v9_operability.py` (V9): surge-flag
  behaviour demonstrated on the V5 rotor line (`pr_turnover`), stable
  BC-switching-across-choke on a swirling-duct testbed (Appendix C.9). Two
  honest `[VERIFY]`s remain, both closure-library boundaries not driver ones:
  the V5 point-by-point surge-line match (reference data), and the *V5*
  choke-knee traversal onto the supersonic-`mdot` branch (the single-node
  continuity Jacobian is singular at the capacity peak — *reclassified at
  M6-4*, see M6 below: a compressor-shock + continuation matter, not the
  turbine milestone). The `machine/` facade's `warm_start` argument and the
  ARCH-6 reproducer-bundle serialization remain seams for later.
- **M6** (axial-turbine correlation set, V6) — closed. Ran as five reviewed
  sub-steps on `main`, mirroring M4's deviation→loss→set ordering: (1)
  `closures/axial_turbine/ainley.py` — throat-based exit angle
  `α2 = arccos(o/s)` (§4.5; the M2→1 limit of K-O), `AinleyTurbineSwirl`;
  added the `throat` opening to the §4.1 geometry contract (its first
  consumer; optional, raises loudly if a turbine closure asks and it's
  unset). (2) `kacker_okapuu.py` + `loss.py` — K-O **profile loss** (AM
  nozzle/impulse interpolation + t/c + Mach `K_p` + Reynolds `f_Re`), native
  B.3 `Y` converted to entropy at the B.1-re-referenced exit state (the
  exit-Mach/`p2` reference taken at the loss-free ideal exit state, no
  flow-view contract change); `KACKER_OKAPUU` CorrelationSet. (3) **secondary
  + trailing-edge loss** — K-O endwall `Y_s` (frame-safe signed-cascade
  loading; aspect-ratio factor) + TE kinetic-energy coefficient mapped to an
  equivalent `Y`; all components share the B.3 exit reference and sum to one
  conversion (B.5-compliant; using B.3 not B.4 for TE keeps the residual path
  exception-free, AD-10). `aspect_ratio`/`te_o_ratio` are row-scalar design
  inputs (annulus-derived AR is a refinement). (4) **inlet shock loss** — K-O
  transonic term `Y_shock = 0.75(M1−0.4)^1.75` in the profile bracket, C¹ via
  softplus; **V5 choke-knee revisited and reclassified** (measured: V5
  meanline chokes at `mdot ≈ 175` kg/s, a continuity-capacity singularity no
  loss closure moves; by AD-5 the *turbine* shock term doesn't apply to the
  Lieblein *compressor* set — the V5 traversal needs a *compressor* shock
  closure + the M5-3 back-pressure mode, a compressor-set/M8 matter). (5) V6
  axial-turbine entry point (`verification/v6_axial_turbine.py`, a pre-swirled
  reaction rotor) — **structural** like V5: converges at all three tiers,
  extracts real work (Δh0<0), de-swirls to near-axial exit, PR/η in sane
  bands (Appendix C.6). Point-by-point K-O validation-case reproduction and
  speedline/choke traversal are `[VERIFY]`, blocked on the reference-library
  correlation calibration (every K-O coefficient is `[VERIFY]`), as for V5.
  The K-O secondary Mach factor `K_s`, the shock geometric/pressure factors,
  and per-node Reynolds (a design-Re parameter stands in) are `[VERIFY]`
  deferrals; the `(1−Mm²)` relaxation factor recalibration stays open (M6+).
- **M7** (centrifugal: parametric-φ path, INBLADE stations, slip, V7) —
  closed. Ran as four reviewed sub-steps on `main`: (1)
  `closures/centrifugal/wiesner.py` — Wiesner slip σ = 1 − √(cos β₂ᵦ)/Z^0.7 +
  `WiesnerSlip` swirl closure (σU₂ − Vₘtanβ₂ᵦ exit swirl; the inducer sgn/
  backsweep sign resolved by probing to give compression, not the turbine
  tangle). (2) `centrifugal/loss.py` — incidence + skin-friction internal
  loss, each converted individually to Δs at the B.1-re-referenced exit
  static T via the new `conversions.delta_s_enthalpy_loss` (cp·ln(1+Δh/cpT));
  `CENTRIFUGAL` CorrelationSet. Blade-loading/clearance/disk-friction
  deferred (why V7 η reads ~0.98). (3) **INBLADE stations** — the driver
  `EDGE_TE=EDGE_LE+1` M7-stub lifted: `_resolve_rows` accepts EDGE_LE,
  INBLADE*, EDGE_TE on contiguous indices and derives `t_stations` (topology-
  fixed mean-anchor meridional fractions, AD-8), wiring the existing §3.4/3.5
  `row_steps` distribution across sub-intervals (classical + Newton rebuild
  sites); no residual-path change. `transport.assert_valid_schedule` (§7.3.4
  contract gate) closes the M4 carryover. **A.8 in-blade force
  `f_b,q = f_b,θ·tanλ` deferred** (zero for radial stacking; needs lean
  geometry + a master-ODE streamwise-gradient term). (4) V7 centrifugal
  entry point (`verification/v7_centrifugal.py`, backswept impeller,
  U₂=362 m/s) — **structural** like V5/V6 and the **first radial end-to-end**:
  converges all three tiers on the φ→90° path, does centrifugal work
  (Δh0>0, PR≈2.46), slipped exit swirl (Vθ/U₂≈0.68), radial exit r=r₂
  (Appendix C.7). **Measured finding**: Tier-3 full-SLC repositioning on the
  90° bend *requires* the INBLADE subdivision (`n_inblade=6`) — edge-only
  diverges the §6.4 odd-even mode at any relaxation and Newton inherits the
  stiff seed; this is the concrete physical reason radial rows want in-blade
  stations. Point-by-point Eckardt reproduction stays `[VERIFY]` (reference
  library + deferred loss), as for V5/V6. Still open past M7: the A.8 force;
  a robust radial-repositioning stabilization (the stable `n_inblade` pocket
  is narrow); blade-loading/clearance/disk loss components.
- **M8** (spanwise mixing, multistage V5 revisit, mixed-flow V8) — closed.
  Ran as four reviewed sub-steps on `main`: (1) `transport/mixing.py` — the
  §3.6 spanwise-mixing operator: implicit (backward-Euler in m, tridiagonal
  in q) diffusion of {h0,s,rVθ}, finite-volume with zero-flux walls so it
  **conserves** the mass-flux-weighted total and is **unconditionally
  stable**; `GallimoreMixing` default (`μ_mix = c_mix·ρ·Vm·r`, `c_mix=0.01`
  `[VERIFY]`); `FidelityConfig.mixing_term` flag (default 0 in ALL tiers incl.
  tier3(), so the §8 degeneracy / V3 identity is untouched). (2) wired into
  the classical driver's lagged field refresh (§6.2.2.4, AD-4) — never the
  residual path (AD-3); `mixing_term=0` is bit-identical to a plain solve even
  with a model supplied. (3) `V5MultistageCompressor` (2 repeating
  rotor+stator stages) — **measured: mixing is a convergence prerequisite for
  multistage axial, not a smoother**: mixing-off runs away to a ~40 J/(kg·K)
  spanwise entropy split and NUMERICAL_FAILUREs even at 800 iters; the shipped
  default mixing converges it (PR≈1.18, 50× less stratified) (Appendix C.5m).
  (4) `V8MixedFlow` (partial φ→55° bend, centrifugal set) — structural at
  Tier 1+2 (converges, compresses PR≈1.56, exits mixed-flow with
  r_LE<r_exit<r_c), Appendix C.8. **Measured: the V7 90°-bend Tier-3 pocket
  does NOT transfer to intermediate angles** — Tier-3 mixed-flow repositioning
  fails across a wide (n_sl,n_inblade,Ω) grid; pinned as a tripwire test. Open
  past M8: the robust radial/mixed **repositioning stabilization** (now the
  V8 Tier-3 blocker), `c_mix` calibration, the mixing entropy-production term
  (the operator redistributes s; the Δs_mix irreversibility source is a
  refinement), plus the standing M7 carryovers (A.8 force, centrifugal loss
  components). Per ARCH-8 this was the last milestone on the ladder.
- **Post-M8 independent audit + turbine-sign fix (2026-07).** A cold audit
  per `docs/audit_charter.md` independently confirmed the kernel numerics
  (A.5 master ODE, RK2 step, A.7 choke identity, V1d order 1.94 all
  reproduced) and found one correctness bug: the axial-turbine closures
  signed the exit angle by `orientation` = sign(**LE** metal angle), which
  flips the exit swirl — work *input* instead of extraction — for a reaction
  rotor with co-rotating relative inflow (β1 > 0, β2 < 0; LE/TE metal-angle
  signs legitimately differ, that is what turning is). Fixed:
  `ParamRowGeometry.orientation_te` (the **TE** turning direction, validated
  lazily like `throat` — β2 must be nonzero and single-signed across span
  only when a TE-keyed closure asks) now signs `AinleyTurbineSwirl` and the
  `KackerOkapuuLoss` cascade frame. `orientation` (LE-keyed) stays for
  Lieblein/Wiesner/incidence loss — **do not swap them**: Wiesner keyed to
  the TE would mis-handle forward-sweep (audited correct as-is).
  Behavior-preserving for every existing case (V6 has both angles negative;
  all turbine test geometries same-signed). Physics-anchored regressions
  (assert work extraction, not the closure's own formula):
  `test_ainley.py::test_reaction_rotor_corotating_inflow`,
  `test_kacker_okapuu.py::test_loss_reaction_rotor_opposite_sign_metal_angles`,
  plus `orientation_te` guard tests. Suite 351, gates green.
- **Post-M8 checker hardening (2026-07, audit follow-up).** The audit found
  both CI lint gates weaker than the decisions they claim to enforce; no
  live violations existed (the discipline held by convention), but the gates
  are now made real. `tools/check_ad6.py`: R1 patterns now match the
  injected **`xp.` spelling** as well as `np.` (the old patterns never
  matched the namespace the kernel routes through); R1 now covers
  **`transport/`** (flow-array schedules/mixing, §7.3 discipline), not just
  `closures/`; and a bare `# ad6: allow` with no justification text is
  itself a violation (R0) — waivers stay auditable. `tools/check_imports.py`:
  new **AD-5 firewall** on top of the direction rule — outside `closures/`
  only `closures.interfaces` and `closures.smoothmath` are importable
  (direction alone let `assembly` import Lieblein constants). Both tools'
  scan cores are pure functions with negative controls in
  `tests/test_lint_tools.py` (the CLAUDE.md checker rule applied to the
  checkers themselves: each rule proven to reject a known violation, plus
  end-to-end positive controls). Suite 372, gates green. Known remaining
  limits (recorded, not enforced): raw `if` on flow arrays is not textually
  detectable (human review per ARCH-4.2); `qo_capacity`-style np use in
  `assembly/` stays outside R1 by design (driver-facing, not residual path).
- **Post-M8 V7 tripwire (2026-07, audit follow-up).** The C.7 "edge-only
  Tier-3 diverges without INBLADE" measured finding was narrative-only
  (asymmetric with the V8 tripwire); now pinned in-suite:
  `test_v7_centrifugal.py::test_tier3_edge_only_is_the_measured_inblade_necessity`
  (`n_inblade=0`, Tier 3 → must NOT converge; flip the assertion when a
  robust radial repositioning stabilization lands). C.7 updated to name it.
  Suite 373.
- **Post-M8 Tier-3 radial/mixed stabilization (2026-07).** The top post-ladder
  open item, closed via diagnosis-first (scratchpad probes, root cause in
  memory). **The M7-4/M8-4 failure attribution was wrong**: not the §6.4
  odd-even repositioning mode (streamlines barely moved before death; pure
  repositioning+curvature on the V8 bend is stable; density irrelevant). The
  real chain: master-ODE $V_m=0$ singularity when a q-o integrates from
  vm_q0 stale vs. the transported fields (the *unrelaxed closure switch-on*
  being the main producer, amplified by the REE swirl term ~ rVθ/r² at
  low-radius mid-bend stations) → the driver fatally boundary-checked the
  stale-guess split the solves were about to repair (killed states proven
  solvable: cold re-solve |F|/ṁ = 4e-15) → `_solve_qo` accepted roots on
  spurious negative-Vm branches / at the −1e30 cliff. The V7 "pocket" and V8
  "angle-specificity" were chaos in whether transient garbage stayed finite.
  **Fixes (drivers/classical.py + assembler capacity guard):** (1) AD-10
  flow-field check moved to the *solved* state (broken metrics stay fatal);
  (2) continuity roots validated onto the strictly-positive branch (root
  validation, not endpoint vetting — endpoint vetting was measured to
  destabilize the V7 closure-lag trajectory); (3) transiently root-less q-o's
  freeze their boundary value, CHOKE_LIMITED only after `choke_patience=15`
  consecutive deficient iterations (window ≈ 2/closure_relax, measured 8 on
  V8); (4) the **first closure application relaxes from the duct baseline**
  through the same §6.2.4 rule as later ones. **Results:** V8 Tier 3
  converges (396 it, PR 1.587, few-% off Tier 2); V7 edge-only converges
  (173 it, PR 2.4433 vs. pocket 2.4540) — *the INBLADE-necessity claim is
  refuted* (stations remain the in-blade resolution choice); V7 pocket
  reproduces its old fixed point (197 it, PR 2.4540); multistage V5
  mixing-off now *converges* — **the M8-3 "mixing is a convergence
  prerequisite" claim was the same artifact**; the surviving physical claim
  (spanwise stratification without mixing) was itself later re-measured down
  from "25×" to a **modest ~18%** by the 2026-07 reference-calibration pass
  (Lieblein ω̄-inversion fix + G–C `c_mix`→5e-4 + a V5 annulus retune putting
  the loss in-window — the "25×" ran on saturated, validity-0 loss); mixing is
  a modest damping, not a homogenizer, as `test_multistage_mixing` now pins.
  Both tripwires flipped; C.5m/C.7/
  C.8 revised; prescribed-closure exactness tests moved to closure-lag
  tolerance (ramp residual ~ tol_closure/closure_relax ≈ 4e-9 rel). Open
  follow-ups: Tier-3 radial/mixed is *slow* (ω_sl ≈ 0.066 throttle; Newton
  finishing / §6.4 recalibration on the blade-row-coupled family — C.3 was
  duct-calibrated and possibly artifact-contaminated); Newton path has no
  positive-branch guard yet (negative-Vm finite garbage passes its
  feasibility check); ln-Vm positivity-safe integration is the recorded
  principled root fix if new cases resurface it.
- **Post-M8 consolidation sprint (2026-07).** (1) **Newton positive-branch
  guard shipped**: `_safe_residual` now splits once, rejects trials whose
  integrated Vm is not strictly positive (spurious branches carry FINITE
  residuals — measured on plain V1c, not just bends), and evaluates the
  residual via the new `ResidualAssembler.residual_from(fields, x)` seam
  (no cost change); regression
  `test_newton.py::test_negative_vm_trial_is_infeasible_despite_finite_residual`.
  (2) **Newton finishing measured, not wired**: on V8 T3 the inner Newton is
  textbook (2-3 quadratic iterations/pass) but the quasi-Newton closure
  outer contracts at only ~0.73/pass after a ~4-pass hump → ~54 passes ×
  ~2.2 s (dense-FD Jacobian) ≈ 120 s vs classical's 75 s. Profitability
  gate = the ARCH-5.3 colored-FD Jacobian (~4× pass cost) or
  closure-in-Newton; don't wire escalation before one of those lands.
  (3) **§6.4 envelope headroom measured** (C.3 note): the duct-calibrated
  threshold is 2-3× conservative on blade-row bends — V8 identical answer at
  `wilkinson_c=13.2` in 152 iters (vs 396), diverges at 22; V7 halves at
  13.2. Default stays 4.4 pending a multi-family recalibration of
  `tools/calibrate_wilkinson.py`; per-case overrides safe to ~13 on
  V7/V8-class geometry.
- **Post-M8 colored-FD Jacobian (2026-07, ARCH-5.3 closed).** Default
  Newton Jacobian is now colored FD, **exact by construction** — validated
  column-for-column against the dense baseline in every configuration (the
  M5 bar). Measured structure (probes, recorded in the module docstring):
  `vm_q0` columns exactly block-diagonal at every tier → one color always;
  interior-q columns block-diagonal to FD noise ONLY with curvature/lean
  off AND `|sin(eps)|~0` (straight annulus — the cos ε sensitivity is
  second-order at ε=0), certified per solve by `_q_columns_groupable`; on
  bends the ε coupling is FIRST-order (measured 38% grouped error at Tier 2
  on V8!) and under curvature the spline couples stations globally
  (~0.27/station decay) — **the arch spec's "near-block-tridiagonal"
  premise is soft, not sparse**. A banded stride-6 approximate mode was
  built, measured (2.8% aliasing → ~1.7× more inner iterations → net
  LOSS end-to-end on V8 T3) and deliberately dropped. Shipped result:
  straight-annulus Tier-2/meanline Jacobians (the continuation/BP
  workhorse) 3.9× cheaper at 1.8e-8 agreement; curved/Tier-3 bit-exact
  with the free vm color (~1.2×). `jacobian="dense"` remains the escape
  hatch + automatic fallback. Newton-finishing profitability on radial/
  mixed Tier 3 therefore stays gated (the 4× hoped-for pass-cost cut is
  physically unavailable on curved paths without closure-in-Newton or a
  compact-support streamline fit — both recorded).
- **Post-M8 §6.4 multi-family recalibration (2026-07; C.3 revised,
  `tools/calibrate_wilkinson.py` extended with `[duct|bladerow|all]`).**
  Duct rerun post-stabilization **reproduces the C.3 fit exactly** (p=1.50,
  K=7.3; two near-threshold classifications softened x→slow-stable — the
  positive-branch guard removed the garbage-branch deaths at onset, mode
  unchanged). Blade-row family (V8 parametric bend, sweeping `wilkinson_c`
  itself — fixed-ω dies in the switch-on transient the adaptive (1−Mm²)
  factor rides out; `n_inblade` barely moves x, so points probe the
  constant, not the exponent): c* ∈ [8.8, ≥30] on converging points →
  **duct family binds, default 4.4 stands**; per-case overrides safe to
  ~13 on ib=6-class layouts. Two NEW measured open items at φ=55°
  (ib∈{2,12}, failing at every c, NOT envelope failures): the **freeze-
  fallback wedge** — a capacity-deficient exit-duct station frozen by the
  §6.6 patience fallback distorts its own repositioning targets and never
  gains the missing capacity (ib=2: false choke, 1200 patience-off iters
  sane-but-unconverged; ib=12: persistent ~3% deficiency with converged
  closures → Vm-singularity rupture at it 161). **Candidate fix
  (capacity-peak vm_q0) implemented, measured, REVERTED** — no-op on all
  passing cases, non-curative on the wedge. Deeper diagnosis (C.3,
  revised): these layouts settle into self-consistent lag states whose
  exit station has NO positive-branch root at mdot — ib=2 a stationary
  24.8% deficit that persists even with the ib=6 fixed point's closures
  PRESCRIBED (suspect: coarse-fit end-condition curvature at the exit);
  ib=12 a stationary surplus (lag-settled stratification, h0 span 22
  kJ/kg, s span 3→15 J/kgK, forces an REE shear whose MINIMUM feasible
  mass ≈ 28.8 > 12 kg/s at any boundary value). Recorded next attacks:
  closure-in-Newton on such states, an end-condition-aware/compact-support
  streamline fit, or documenting ib≈6 as the supported radial/mixed
  layout. Until then 55°-class bends with ib far from 6 are a
  known-unsupported region, honestly reported by typed statuses.
- **Reference-library calibration pass — COMPLETE (2026-07).** A systematic
  discharge of the `[VERIFY]`/`[DECIDE]` correlation backlog, checking every
  closure coefficient and every fitted chart output against authoritative
  library sources (the user's NotebookLM notebooks + Google Drive papers +
  public archives; transcription notes live in `docs/references/<KEY>.md`,
  index in `docs/references/README.md`). Coefficient level (9 sources
  discharged): confirmed the K-O (KO82), Wiesner (WIE67), Aungier compressor
  fits (AUN-C), Gallimore-Cumpsty mixing (GC86), Lieblein loss (LIEB59),
  centrifugal loss (CENT-LOSS), Ainley angle (AM-ANGLE), and Appendix-B
  conversions (CONV-B) constants verbatim. **Real bugs found + fixed:** the
  Lieblein ω̄ velocity-ratio inversion `(W1/W2)²`→`(W2/W1)²` (~4× profile-loss
  overestimate, `loss.py`); the `K_ti` thickness-exponent ×10 (`lieblein.py`);
  the AM-Fig4 positivity-floor width (angle-scaled on loss-scale values, ~0.06
  inflation of every profile-Y); the K-O TE φ² curves (nozzle/impulse swapped +
  ~3× high + symmetric weight, `kacker_okapuu.py`). **`[DECIDE]`s resolved:**
  GC `c_mix` 0.01→5e-4 (refuting the M8 "mixing homogenizes multistage" claim —
  it is a modest ~18% damping, re-measured after a V5 annulus retune that put
  the Lieblein loss in its validity window); Lieblein off-design bucket →
  Aungier ξ-model; centrifugal `f_inc`/`W_avg` → Aungier 2000; Wiesner
  radius-ratio limit → Braembussche cubic; KO82 interp weight → signed `|r|r`.
  **Chart-output digitizations (all clean-or-fixed, rerunnable
  `tools/digitize_*.py` + pinned reference tests):** SP-36 incidence/deviation
  (i0/n/δ0/m, RMS ≤0.17°, no bug), AM R&M2974 Fig.4 turbine profile loss
  (floor bug fixed), K-O Fig.8 K1 (confirmed exact — printed on the chart) +
  Fig.14 TE φ², and Lieblein 1959 Fig.6 θ*/c loss magnitude (clean, max
  |coded−chart|=0.0003; pinned the D_eq validity window + the 2.35 divergence
  limit). Method lesson (recorded in the tools' docstrings): on fine-grid
  rasters the frame is not distinguishable by ink weight — anchor to the
  uniform tick grid, and OVERLAY the coded fit on the chart image as the
  decisive check. **No known reference/chart work remains**; residual
  `[VERIFY]`s in the closures are genuine deferred *refinements* (e.g. the
  Lieblein off-design `R_s`/`R_c` Mach adjustment, the K-O `K_s`/shock geometric
  factors, per-node Reynolds), not calibration gaps. Point-by-point
  published-figure / NASA-data *case* reproduction (V4–V8 speedlines) stays
  `[VERIFY]` — a validation-dataset matter, separate from this coefficient pass.
- **Axial-compressor endwall + tip-clearance loss (2026-07).** The
  axial-compressor set modelled **profile loss only** (`profile_omega_bar`) —
  roughly half a real stage's loss — so it read efficiency systematically high,
  the physical blocker (beyond the absent validation dataset) for a
  point-by-point V5 speedline. Added the deferred endwall/clearance physics the
  set's `__init__` named "at V5 calibration time": **Howell's additive
  drag-coefficient model** (secondary `C_Ds = 0.018 C_L²` + annulus
  `C_Da = 0.020 s/h`) plus **Lakshminarayana tip clearance** (`C_Dk = 0.7 C_L²
  t/h`), converted to inlet-referenced `omega_bar` (Cumpsty 4.9, derived from
  first principles to disambiguate the OCR) and summed with the Lieblein
  profile `omega_bar` under one B.2 conversion (§4.4). **Howell over Aungier**:
  Aungier folds endwall into the profile correlation via K1/K2 + charts
  (Eq 6-46, Fig 6-11/6-12) — not clean-additive; §7.1 permits either, Howell is
  the closed-form library-verifiable one (`docs/references/HOWELL.md`, verified
  verbatim vs Dixon/Howell/Saravanamuttoo/Cumpsty). Evaluated at the reference
  (design) triangle and added flat to the bucketed profile loss; tip clearance
  from the geometry contract (0 by default → clearance term inert, existing
  zero-clearance cases see only secondary+annulus). **Measured:** V5 rotor η
  ~0.96 → ~0.92 (realistic subsonic level), PR ~unchanged (loss affects η, not
  Euler work); multistage V5 → η ~0.64 (honest for a low-PR mixing testbed —
  band widened, and the C.5m mixing damping re-measured ~18%→~24%, still modest,
  finding intact). Pure functions `blade_loading_coefficient` /
  `endwall_clearance_loss` with source-cited reference tests. Deferred
  `[VERIFY]`/`[DECIDE]`: off-design `C_L` (actual vs reference triangle),
  secondary/clearance overlap (Howell's 0.018 already includes a typical
  clearance), the `C_L` validity ceiling, and a compressor **shock** loss
  (transonic V5). Point-by-point V5 still needs a subsonic-stage validation
  dataset (absent from the library) — the efficiency-physics blocker is now
  lifted, the dataset one is not.
- **Axial-compressor transonic shock loss (2026-07).** Added the deferred
  compressor **shock** loss (the theory-manual §C.9 M6-4 deferral: the *turbine*
  K-O shock term doesn't apply to the Lieblein set by AD-5). **Aungier §6.7**,
  verified vs the library: a normal shock at the **geometric-mean Mach**
  `M_shock = √(M1·M_ss)` (Eq 6-71), with the suction-surface Mach
  `M_ss = M1·(W_max,s/W1)` taken from the **equivalent-diffusion bracket**
  (Aungier's own `D_eq = W_max,s/W2` estimate — used in lieu of the Eq 6-69/70
  Prandtl-Meyer surface-curvature expansion, whose `R_u` isn't in the §4.1
  contract). Normal-shock Pt loss (perfect-gas Rayleigh pitot, exact reduction
  of Aungier's real-gas 6-72..74) referenced to inlet dynamic head (Eq 2-68) and
  added to the profile+endwall `omega_bar`. **C¹ at the M=1 onset** (loss ∝
  (M−1)³ per Cumpsty, held C¹ by a softplus floor — pinned by a refinement-scaling
  test), and **inert subsonic** (0 to reading precision below onset → all current
  V5 cases unchanged; V5 rotor η 0.922 preserved). Because `M_shock > M1` it
  engages Aungier's *supercritical* regime (shock while the inlet is subsonic).
  New pure functions `normal_shock_pt_ratio`/`shock_loss`; reference tests pin
  the normal-shock ratio vs gas tables (M=2.0→0.7209), the formula, the
  supercritical onset, and end-to-end engagement (η falls with speed on a
  high-Ω rotor). Deferred `[VERIFY]`: the Prandtl-Meyer `M_ss` (needs `R_u`),
  the onset width, the far-supersonic validity ceiling. The remaining piece for
  the V5 supersonic-branch traversal is now a *transonic V5 case design* (an
  in-window supersonic-inlet rotor) + the back-pressure/continuation mode — a
  case + driver matter, no longer a missing closure.
- **Transonic V5 case — in-window meanline gate (2026-07; two-branch premise
  refuted).** `V5TransonicRotor` is a high-Ω rotor with supersonic *relative*
  inlet Mach (M1_rel≈1.14) exercising the §6.7 shock loss. The task "build the
  V5 supersonic-branch continuation driver" was **diagnosed and found
  unnecessary** — characterize-before-fixing (scratchpad probes) overturned the
  documented two-branch premise. **The in-window condition is set by blade
  *loading* (the equivalent-diffusion factor D_eq), NOT by which
  meridional-continuity branch the solve lands on.** The original geometry was
  simply over-diffused: on the ordinary subsonic-meridional branch the mass-flow
  driver already reaches, D_eq≈2.30 > the Lieblein window ceiling 2.0 → `v_d=0`
  (every other factor fine, incl. shock `v_sh=1` at M_shock=1.32<1.7); the
  supersonic-meridional branch is *worse* (higher Vm→higher W1→higher D_eq). The
  "β1≈50°, validity≈0.96 in-window branch" the old tripwire/docs claimed **never
  existed** for this geometry. **Fix = a case retune, not a driver:** β2 −52°→−58°
  (light relative turning, physical for a transonic rotor at the diffusion limit)
  drops D_eq in-window while keeping M1_rel>1 and the shock active → the plain
  mass-flow driver converges a genuine in-window transonic point (Tier-1 meanline,
  mdot=55: validity≈0.99, PR≈1.51, η≈0.86). The old `..._TRIPWIRE` test is
  replaced by a positive gate `test_transonic_meanline_is_in_window_and_shock_active`.
  **Two case-design bounds remain (not driver):** the in-window pocket is narrow
  in mdot (~55), and the **spanwise** tiers still read validity 0 (constant metal
  angle over the radius ratio swings β1 across span → endwall D_eq out of window;
  narrowing the annulus to fix it fights the high blade speed the transonic
  condition needs → an all-tier in-window transonic case is a deferred
  case-design refinement). The genuine **meridional-supersonic-branch traversal
  driver** is a real but *separate* capability (now built — next bullet), needed
  only for design points deliberately on the supersonic-meridional branch, not a
  V5 blocker.
- **Meridional-supersonic-branch driver (2026-07).** Built as a general,
  reusable capability: `drivers/supersonic.py` `solve_supersonic_branch` —
  **pseudo-arclength (Keller) continuation** in `(state, mdot)` that crosses the
  per-station `M_m=1` continuity fold (where the classical mass-flow driver
  chokes and natural-parameter/back-pressure continuation pins — see §C.9 branch
  map) onto the meridional-supersonic branch, then a fixed-`mdot` Newton lands
  the exact on-target supersonic root (branch selected → regular root). The
  augmented Jacobian's `mdot` column is analytic (mdot enters continuity
  linearly); state columns are FD with the Newton positive-branch guard;
  **variable scaling is mandatory** (measured: an unscaled arclength creeps in
  mdot because Vm dominates the norm near the fold). Added a DRY seam
  `ResidualAssembler.continuity_position_rows(fields, mdot)` (single-sources the
  continuity/position rows for mdot-as-variable; `residual_from` now calls it —
  behavior-preserving). **Verified** against an independent isentropic area–Mach
  reference on a purpose-designed meanline converging–diverging **nozzle**
  (`tests/test_supersonic.py`): classical chokes above throat capacity
  (the fold); the driver crosses it (turning point = analytic capacity to <0.2%)
  and lands the supersonic throat Mach = isentropic area–Mach supersonic root to
  <0.3% (e.g. M_m=1.397), inlet/exit staying subsonic (rank-1 fold); same mdot,
  two roots (sub vs supersonic). This closes the "V5 supersonic-branch traversal"
  driver item as a standalone method — decoupled from V5, whose gate is met on
  the ordinary branch (previous bullet).
- **Closure-lagged blade-row supersonic-branch extension (2026-07).** The
  `rows` path of `solve_supersonic_branch`: for closure-fed rows the swirl/loss
  closures are flow-dependent and lagged (AD-4), and the supersonic-branch field
  differs from the subsonic seed's, so the closures must re-lag at the landed
  supersonic state. Structure: **bootstrap onto the supersonic branch by
  arclength ONCE**, then **hand the supersonic seed to `solve_newton`** at
  `target_mdot` — which runs the SAME outer quasi-Newton closure-lag loop it uses
  everywhere (§6.3); the fold is behind the seed, so the Newton inner's
  positive-Vm guard keeps it supersonic while the outer loop re-lags. Reusing
  `solve_newton` (not a bespoke loop) is deliberate — the extension adds almost
  no numerical surface. Verified (`tests/test_supersonic.py`) on a **Lieblein row
  UPSTREAM of the throat** (inflow subsonic → closures in-window) with
  `target_mdot != seed_mdot` so the flow-dependent closure genuinely re-lags:
  throat crosses to M_m≈1.5, row inflow stays subsonic + Lieblein-valid, and the
  lagged closure is **self-consistent at the landed field** (fresh eval agrees to
  <0.1%) vs. a several-% inconsistency if the seed's closures are frozen. The lag
  is Picard at rate (1−closure_relax): the conservative default relax (kept safe
  for stiff M4-4 swirl-continuity loops) needs ~60 outer passes (ArclengthConfig
  bumps the default `newton.max_outer` to 120 so benign cases converge out of the
  box); weakly-coupled cases tolerate a larger relax. **Known limit** (honest): a
  fully supersonic ROW inflow that folds several stations at once (measured on a
  transonic rotor, §C.9) is the harder **multi-fold** regime this single-fold
  arclength does not claim — a deflated/multi-parameter continuation would be the
  next step, unneeded by any current case.
- **Centrifugal blade-loading (diffusion) loss (2026-07).** Added the dominant
  deferred centrifugal internal loss (`closures/centrifugal/loss.py`
  `blade_loading_loss`): Coppage/Jansen, **Aungier 2000 Eq 5.15**,
  `Δh_bl = 0.05 D_f² U2²` with the radial diffusion factor `D_f = 1 − W2/W1 +
  0.75(Δh_euler/U2²)(W1/W2)/[(Z/π)(1−r1/r2)+2(r1/r2)]`. Constant + structure
  verified via NotebookLM; the loading-term ratio `W1/W2` (the source's MathML
  render is ambiguous) resolved by Oh-Yoon-Chung/Galvas consensus **and** the
  physical grows-with-diffusion constraint (pinned by a reference test). Smooth
  `D_f` ceiling (2.5) bounds transients (the axial ω̄-ceiling analogue). Measured:
  V7 design `D_f≈1.12`, `Δh_bl≈6.9 kJ/kg` — the dominant internal loss — drops
  V7/V8 η 0.98→realistic **~0.90**. **Landing = option (b), documented wedge:**
  the loss is so dominant it drives the fragile radial/mixed **spanwise** solves
  into the documented freeze-fallback wedge (V7's 90° bend at BOTH Tier 2 and
  Tier 3, V8's mixed bend at Tier 3; lowering mdot makes it worse, the wedge
  signature; V7 Tier-2 retune to mdot 10 still only slow-max-iters). Attempted
  the Tier-3 stabilization first (per "stabilize then land"): reposition-freeze +
  capacity-peak-freeze robustness patches got the bends further but did NOT crack
  the wedge (capacity-peak was already measured+reverted post-M8) — and changed
  no test outcome, so **reverted** (clean driver, zero regression risk). Landed
  the correct physics with the spanwise-bend tiers as `xfail(strict=True)`
  tripwires (Tier-1 meanline + V8 Tier-2 carry the realistic-loss validation);
  reference tests pin the form + the ratio direction + the ceiling. Cracking the
  wedge (closure-in-Newton / compact-support streamline fit) is the standing #1
  open item, **separate from this loss**. Tip-clearance (needs exit width `b2` +
  hub/tip in the §4.1 contract) and disk-friction (machine-level parasitic, no
  `ṁ` in a per-streamtube closure) remain deferred. Memory:
  `centrifugal-blade-loading-wip`, `docs/references/CENT-LOSS.md`.
- **Wedge re-characterization + V7 operating-point crack (2026-07).** The
  "start on the wedge: closure-in-Newton" task was **diagnosed and the premise
  OVERTURNED** (characterize-first, `probe_cin_*` scratchpad probes on V7 Tier 2
  — the fastest repro): closure-in-Newton AND loss-continuation/pseudo-arclength
  are both the **wrong tools**. What had been called one "freeze-fallback wedge"
  is TWO distinct diseases. **(1) V7 Tier 2 = an operating-point
  stratification-capacity fold, CRACKED.** The realistic blade-loading loss
  stratifies the exit (rVθ, Δs) profile and drives an INTERIOR streamtube's `Vm`
  toward the master-ODE `Vm=0` singularity, so BELOW a mass-flow floor the
  *coupled* flow folds — even though each station's A.7 capacity individually
  stays ≫ ṁ (the old "no positive-branch root at any ṁ" reading checked the
  wrong, per-station thing). It is NOT closure coupling (FIXED prescribed
  stratified transport folds identically — built the coupled residual X=(x_flow,
  c) with consistency rows and measured it) and NOT the classical repositioning
  algorithm (a global Newton folds too); the fold terminates AT the singularity,
  so continuation has no far side. **Raising ṁ lifts every `Vm` off the
  singularity** — V7 T2 converges with realistic loss at ṁ∈[16,20]. Fix = a
  case-design **re-centre ṁ 12→17** (mid-window; validity 1, PR 1.97, η 0.80),
  same category as the V5 validity-0 and transonic-V5 retunes. V7 `test_tier2_
  converges_with_realistic_loss` is now a PASSING structural test (was an xfail
  tripwire). **(2) V7 Tier 3 = a physical FEASIBILITY FOLD at realistic loss
  (OPEN, not a solver gap)** — later diagnosed (damped-Newton + curvature-strength
  continuation, `probe_v7t3_*`): the flow branch folds (interior Vm→0, master-ODE
  singularity) at ~9% of full Tier-3 curvature at ṁ=17 (~26% at ṁ=20). The tight
  0.08 m bend (κ~20) swings spanwise Vm by +200..477 m/s; with the realistic-loss
  stratification it drives an interior streamtube to Vm→0 before full curvature.
  The fold IS ṁ-liftable (like the T2 wedge) but reaching c=1 needs ṁ~32 ≫ the T2
  choke ceiling ~22 → no operating point admits full radial equilibrium. **No
  positive-Vm root** → NOT the "robust repositioning" item; stiff integrator /
  compact-support fit / damped Newton can't help ("repositioning failed" is
  downstream of the fold). Case-side levers only: calibrated/lower loss ([VERIFY],
  likely high), gentler bend, or beyond-model-validity. Stays xfail
  (`test_tier3_infeasible_fold_at_realistic_loss`); memory `v7-tier3-root-cause`.
  **(3) V8 Tier 3 = a narrow pocket (OPEN, distinct from V7 T3)** — choke_limited
  at ṁ=12, converges only at ṁ≈15 (593 iters, agrees T2 ~3%), choke ≤14 /
  slow-max-iter ≥16; too narrow/slow to pin robustly, so V8 stays ṁ=12 with T3
  xfail; blockers are the Tier-3 radial slowness (`ω_sl≈0.066`, Newton finishing
  / §6.4 recalibration) + operating point. C.7/C.8, overview §10/§11, and memory
  `wedge-closure-in-newton` revised; `centrifugal-blade-loading-wip` "V7
  meanline-only" claim superseded. No new solver code (the wedge crack was an
  operating-point retune, not machinery); the coupled-residual probes stayed in
  scratchpad. The robust Tier-3 radial/mixed stabilization is now scoped to **V8
  T3 only** (narrow/slow pocket); V7 T3 was reclassified as a physical
  feasibility fold, off that item (case-side levers only).
- **Centrifugal blade-loading D_f ratio fix + validation-effort kickoff
  (2026-07-12).** First action of the validation+calibration gate
  (`model-readiness` #1/#2): characterize-before-changing the highest-leverage
  `[VERIFY]`, the centrifugal **blade-loading loss magnitude** ("reads high;
  less-stratified is key"). **Found + fixed a ratio-inversion bug** in the
  Coppage/Jansen diffusion factor `D_f` (`closures/centrifugal/loss.py`): the
  loading term used `(W1/W2)` in the NUMERATOR; the authoritative **Oh-Yoon-Chung
  1997** (the exact paper the `0.05 D_f² U2²` form is cited from — Drive
  `oh_optimum_1997.md`, clean verbatim text) prints `(W1t/W2)` in the
  DENOMINATOR → the loading term carries `W2/W1`. The old `W1/W2` rested on an
  ambiguous NotebookLM MathML scrape + a mistaken "must grow with diffusion"
  argument (diffusion is the leading `1−W2/W1` term; the loading term scales with
  LOADING Δh_Euler). Same category as the LIEB59 ω̄ inversion. **Also the "Aungier
  2000 Eq 5.15" citation was wrong** — Aungier uses a different form
  `ω̄_BL=(ΔW/W1)²/24` (Eq 5-34, Drive `aungier_centrifugal_2000_part1.md`); the
  `0.05 D_f² U2²` form is **Coppage et al. 1956**. Fixed `(w1/w2f)`→`(w2/w1f)`;
  citation corrected; docstrings/CENT-LOSS.md/theory C.7-C.8/overview synced.
  Reference tests: `test_blade_loading_matches_coppage_oh1997`,
  `test_blade_loading_uses_w2_over_w1_not_w1_over_w2` (regression guard),
  `test_blade_loading_grows_with_loading` (replaced the wrong
  `_grows_with_diffusion`). **Measured (V7 design):** loading term +0.400→+0.062,
  `D_f 1.005→0.668`, `Δh_bl 5609→2474 J/kg` (2.27× less). **Downstream:** V7 η
  0.799→0.839 (T1) / 0.803→0.828 (T2); V8 η 0.897→0.930 / →0.918; exit
  entropy-spread V7 T2 −27%, V8 T2 −30% ("less stratified"). **Fragility eased,
  partly cracked:** V7 T3 fold shifted (now fails at sane PR/η, not garbage) but
  still infeasible at every mdot → xfail stands (calibrated-loss lever tried,
  didn't crack the tight 90° bend); **V8 T3 pocket LOWERED into a converging
  window mdot∈{13,14}** — V8 re-centred 12→14, its Tier-3 xfail **flipped to a
  passing test** (`test_tier3_converges_at_recentred_mdot`, all three tiers,
  Tier 3 agrees Tier 2 ~2.5%). Multistage V5 unaffected (axial Lieblein set,
  AD-5). **Validation-dataset status surfaced** (`centrifugal-validation-dataset`
  memory): no primary Eckardt/Krain paper in Drive; Oh-1997 has Eckardt O/A/B
  PR+η MAPS (figures → need digitization) + only the KIMM impeller fully
  tabulated; full point-by-point STAGE η validation is confounded (Oh's η
  includes deferred parasitic + vaneless-diffuser losses) — the coefficient-level
  ratio fix did NOT need the dataset, but the point-by-point map reproduction
  still does. Memory: `centrifugal-blade-loading-wip`, `reference-calibration`,
  `centrifugal-validation-dataset`, `v7-tier3-root-cause`, `model-readiness`.

- **Measured-data validation campaign — first pass (2026-07-15).** The
  `model-readiness` gate #1 opened: the user assembled the "Turbomachinery:
  Test Cases" NotebookLM notebook (9 sources incl. the paywalled AGARD/
  Eckardt/Krain/LS-89 primaries), and four validation deliverables landed on
  `main`, each with a `docs/references/<KEY>.md` note + pinned
  measured-agreement tests (bands encode MEASURED agreement, not success
  claims): **(V4/TR1368)** NACA TR-1368 Fig. 107 (measured design turning
  slope) digitized (grid-anchored + seeded trace + overlay) → validates
  `deviation_slope` at the raw-data level, RMS 0.030/39 pts, β₁=70 low-σ a
  documented deviation region (+0.097); the 65-series equivalence
  θ_eq = 4·arctan(0.1103·C_l0) grounded (Fig. 111 cross-plots = recorded
  extension). **(V5/ROTOR37)** geometry-faithful NASA Rotor 37
  (`verification/v5_rotor37.py`, TP-1659 Table III(a) + measured 100%-speed
  line): both tiers converge; η within ~1 pt; PR +12–16% decomposed by probe
  into ~5 pts zero-blockage + ~7 pts Lieblein-deviation-on-MCA under-
  prediction (β2_flow 44.2° vs design-intent 47.7°); validity 0 at Rotor-37
  loading (D_eq ≥ SP-36 ceiling); speedline far shallower than the measured
  choke-side collapse (sign-only pinned). The calibration targets (blockage
  schedule, MCA deviation correction vs Table V, transonic loss level) are
  now measured numbers with data attached. **(V7/ECKARDT)** the primary
  Eckardt 1976 paper grounded the missing geometry (r1h 45/r1t 140/b2 26 mm,
  Z=20 radial, 130 mm, z_s/b2 0.027) → geometry-faithful
  `verification/v7_eckardt.py` (quarter-ellipse walls recorded): **all three
  tiers converge, validity 1.0, ~0.1% tier agreement — the V7-testbed Tier-3
  infeasibility fold is a property of its synthetic tight 0.08 m bend, not
  of radial machines**; laser-point PR +4.7% (vaneless-diffuser-sized gap),
  design +12.6% (deferred parasitic/clearance grow with speed); implied
  measured slip ~0.90 vs Wiesner 0.877 (recorded). `eckardt_anchor.py`
  superseded. **(V6/LS89)** VKI LS-89 cascade-level: `throat_exit_angle` =
  the paper's gauging angle to 0.1° at M2is=1; predicted energy-ζ 0.0303 vs
  measured 0.0225 (+35%, documented K-O behaviour; the TE curve carries most
  of it; the rig's 0.5% exit/TE-shock is a recorded model boundary).
  **(V6 machine-level, 2026-07-16 / TND6967)** NASA TN D-6967 two-stage
  cold-air turbine (`verification/v6_tnd6967.py`, Tables I/II + Fig. 1
  velocity diagrams, throats = design gauging): meanline η_tt 0.926 vs
  measured 0.93 (−0.4 pt, tip-clearance loss unmodelled — inside K-O's own
  ±1.5 pt target); PR/work −17/−12% at matched flow = a flow-CAPACITY gap
  (no-blockage geometric throat vs near-sonic-by-design stator hubs), the
  recorded levers being effective-throat blockage + the deferred AM
  low-speed exit-angle correction; the 4-row Picard chain needs
  `max_outer=800` (slow, not unstable; the case defaults it); Tier-2
  spanwise OPEN (free-vortex hub chokes its streamtube at measured flow) —
  joins the multi-row spanwise robustness items. Remaining from the
  campaign (recorded in the notes + memory): Stage 37 stator/stage + other
  speeds + Table V/VI surveys (the MCA-deviation calibration target) +
  AGARD coords/tip clearance, Fig. 111/Fig. 16/Oh-1997-map digitizations,
  Krain + Stage 38 + CC3 second points, TN D-6967 maps/first-stage config,
  mixed-flow V8 (still no open rig dataset).

- **MCA/transonic deviation correction (2026-07-16, gate #2's first
  data-validated calibration).** The Rotor 37 deviation gap is closed by a
  **library-grounded published correction, not a local fit**: AGARD-R-745
  (Çetin et al. 1987 — found IN the loss-models notebook) Eq. 3.5,
  `δ*cor = −1.099379 + 3.0186·δ* − 0.1988·δ*²`, fitted on exactly this
  blade family (1970s MCA/DCA transonic rotors, where Carter-family rules
  "underestimate the deviation angle"). Implemented verbatim as
  `lieblein.cetin_deviation_correction` — **opt-in**
  (`LieblienSwirl(transonic_correction="cetin_agard745")`, default `"none"`
  so the NACA-65 pedigree and every existing case are untouched),
  C¹-saturated into the polynomial's monotone fitted branch (window
  0.5–7.5°, vertex 7.59°) with compact-support validity, ConfigError on
  unknown options at construction (AD-10). Applied to the SP-36/Aungier
  reference deviation (recorded reading; AGARD's baseline is Carter — same
  subsonic-minimum-loss family). **Measured, with zero locally fitted
  constants**: Rotor 37 per-span deviation error RMS 3.8° → 1.2° (mean
  −3.6° → ~0) vs `MEASURED_BE_4182`; end-to-end with the correction ON
  (the Rotor 37 case's new default) **Tier-2 PR 2.051 vs measured 2.056
  (+0.2%, was +12%)**, Tier-1 2.135 (+3.8%, was +16%), closure validity
  0 → ~0.8 at Tier 1. The choke-side speedline collapse remains, as does
  the blockage schedule. Tests: `test_cetin_correction.py` (Eq. 3.5
  coefficient pins, C¹ refinement check across the saturation knees,
  validity compact support, config boundary, default-off preservation) +
  re-pinned `test_v5_rotor37.py` (uncorrected-gap record kept alongside
  the corrected pin). **Swan Eq. 70 (same source) was then implemented,
  measured, and NOT default-adopted** (2026-07-16): opt-in
  `LieblienSwirl(offdesign_rule="swan_agard745")` — verbatim bracket,
  C¹-blended with the Aungier slope across the stated M1=0.6 onset (AD-6,
  no flow branch), ±8° smooth increment ceiling, validity to the M1≈1.5
  data edge, D_eq* via the loss chain's reference-triangle convention
  (`equivalent_diffusion` moved to `lieblein.py`, re-exported via `loss`).
  Measured on Rotor 37 the choke-side-steepening hypothesis was REFUTED:
  the measured 100%-speed line spans only ~±3° of incidence about
  reference, so off-design deviation is small under either rule — Swan
  shifts PR a uniform +0.03 (slightly away from measured, its negative
  Mach bracket at M1≈1.4 cutting deviation) and does not steepen the line;
  the measured collapse is loss/choking physics (the rig is choked at
  20.93 kg/s where the meanline still has capacity margin). Finding pinned
  (`test_swan_offdesign_rule_runs_but_is_not_adopted`). **The remaining
  two speedline hypotheses were then also implemented and dispositioned
  by measurement (same day):** (a) the AGARD Eq. 3.3 Mach-dependent
  off-design loss parabola (`loss.cetin_offdesign_loss`,
  `LieblienLoss(offdesign_loss="cetin_agard745"|"cetin_agard745_choke",
  blade_family="mca"|"dca")`, C¹ M1-blend at the 0.6 onset,
  softplus-floored Table-1 curvature lines, choke-only hybrid variant) —
  NOT adopted: the full variant's stall-side line collapses low-flow η
  (0.776 vs measured 0.852 at 19.60 kg/s) and the choke-only hybrid is
  INERT (with capacity calibrated the rotor, like the rig, never runs
  below reference incidence; the measured choke-side PR collapse is the
  VERTICAL CHARACTERISTIC — a BackPressureSpec comparison, not a loss
  bucket); (b) a uniform-blockage capacity model via the existing
  `Machine` blockage seam (`Rotor37(blockage=...)`) calibrated to the
  AGARD-AR-355 measured choke flow 20.93 ± 0.3 kg/s — B=0 chokes the
  meanline +6.5% high and Tier-2 +3.5% (the spanwise tier resolves
  endwall streamtubes, capturing half the gap); **B = 0.033 lands the
  Tier-2 choke inside the measured band but costs the mid-line PR ~7%
  → the capacity deficit is NOT uniform blockage, it lives at the
  unmodelled blade-passage THROAT** (a compressor throat/capacity
  station is the recorded model item). Case defaults stay parameter-free
  (B=0, Aungier bucket) so the Çetin-corrected pins stand; the
  calibration + non-adoption findings are pinned
  (`test_capacity_gap_and_blockage_calibration`,
  `test_agard_offdesign_loss_options_measured_not_adopted`). AGARD-AR-355
  also grounds the tip clearance ≈ 0.41 mm (the case's 0.4 mm assumption
  is consistent — `[VERIFY]` softened).

- **Row-throat capacity check (2026-07-16, §6.6 `c_row` implemented).** The
  "compressor throat/capacity station" item, landed as the manual's
  anticipated row-throat margin: `drivers/classical.py
  row_throat_capacity` — 1-D blade-passage sonic capacity
  `cd·Z·Σρ*a*·o·dq` on the mid-passage q-o in the RELATIVE frame (B.1
  rothalpy at r_th=(r_LE+r_TE)/2), applied POST-SOLVE (solved-state-check
  pattern) to every row whose §4.1 geometry provides a throat; converged
  solutions with `mdot > capacity` are declared CHOKE_LIMITED with the
  capacity in the reason. `RowSpec.throat_cd` (default 1.0) folds
  passage-BL blockage. **Measured:** TN D-6967 — per-row capacities
  s1 2.19 / r1 2.06 / s2 2.14 / r2 2.20 kg/s eq; machine choke =
  min(annulus ~2.02, rotor-1 throat) ≈ 2.02 vs the rig's measured
  2.03–2.05 (**within ~1% at cd=1**), which RE-DIAGNOSES the −17%
  matched-mdot PR gap as NEAR-CHOKE SENSITIVITY (both rig and model sit
  1–2% from choke at 2.004 kg/s where PR-vs-mdot is near-vertical; the
  right frame there is matched-PR/BackPressureSpec — recorded, not run).
  Rotor 37 — with a gauging-estimate throat `o = s·cos(KIC)` the
  rotor-relative capacity is ~24.6 kg/s > the ~22.25 annulus limit: a
  supersonic-inlet rotor chokes at its inlet swallowing limit, the check
  is correctly inert, all case pins stand. Tests
  (`test_throat_capacity.py`): closed-form analytic match, meanline
  degenerate span, rothalpy sign effects, cd linearity, both machine
  integrations. Recorded extensions: flow-responding throat physics,
  `c_row` into the §6.7 continuation margins + Newton driver, and a
  first-stage-only TN D-6967 back-pressure comparison.
- **Matched-PR BackPressureSpec comparison (2026-07-16) — the TN D-6967
  validation closes at ~1%.** Imposing the Table-I equivalent
  total-to-static PR (p_exit = p0_in/4.640) via the M5-3 `BackPressureSpec`
  + `solve_newton` (mdot joins the state), seeded from the converged
  matched-mdot solution: **ṁ +0.35%, PR_tt −1.2%, work −1.2%, η_tt 0.926
  vs measured 0.93** — the machine agrees with the rig to ~1% across the
  board in the natural near-choke frame, confirming the matched-mdot
  "−17% PR" was pure vertical-characteristic sensitivity with no hidden
  turning/loss deficit (an intermediate "work −8.7% ⇒ AM under-turning"
  reading was an artifact of a spurious closure-lag branch and is
  retracted). Two measured solver findings recorded: the near-choke
  closure lag has **seed-dependent fixed points** (fresh mdot-2.0 seed →
  same PR but work 8% low; 1.95 seed → runaway NUMERICAL_FAILURE;
  warm-starting from the nearest converged operating point — the
  continuation driver's own discipline — selects the physical branch),
  and it limit-cycles at ~1e-6 closure norm (benign; `tol_closure`
  loosened in the pinned test). Pinned:
  `test_v6_tnd6967.py::test_matched_pr_backpressure_comparison`.
  Follow-ups recorded: root-cause the spurious branch (which station
  flips continuity root), Rotor 37 vertical-characteristic points in the
  same frame.
- **Spurious closure-lag branch ROOT-CAUSED + FIXED (2026-07-16): the
  Newton branch-preserving trial guard (§6.3).** Diagnosis (skill loop,
  22-s deterministic repro): the fresh-seed TN D-6967 BP solve converged
  **station 7 (rotor-2 LE) on the SUPERSONIC continuity root** — M_m
  1.997 / Vm 452 m/s, with the A.7 root pair proven by a continuity scan
  on the suspect frozen fields (subsonic root 112 m/s, capacity peak
  ~+1.3 kg/s at ~250, supersonic root 452.1) — the positivity-only Newton
  guard accepts it (positive, finite, convergent) and the closure lag
  locks in a self-consistent spurious fixed point (same PR, work 8% low,
  garbage entropy downstream). Fix: `newton._safe_residual` now also
  rejects trials whose q=0-node meridional Mach JUMPS across 1 relative
  to the seed's per-station branch (`_branch_masks`, hysteresis
  `_BRANCH_DELTA = 0.05` so near-choke convergence toward M→1 stays
  free); masks classified once from the warm-start seed — branch
  SELECTION stays with the caller, exactly the arclength/Newton division
  of labor, and the supersonic-branch handoff is now *protected* from
  falling back (its `_land` passes its own seed masks). Measured after
  fix: the 2.0 and 2.004 seeds land the bit-identical physical fixed
  point; the far 1.95 seed fails TYPED instead of converging wrong; all
  Newton/backpressure/supersonic families green. Regression:
  `test_v6_tnd6967.py::test_backpressure_newton_stays_on_seed_branch`
  (station-7 repro, subsonic-everywhere + measured-work assertions).
- **Rotor 37 matched-PR traversal (2026-07-16, the campaign's last
  recorded comparison).** Descending `BackPressureSpec` continuation from
  a near-choke seed converges down the vertical-characteristic side at
  BOTH tiers (branch guard holding; Tier 2 rides to 22.0 kg/s, past the
  classical patience-declared 21.65 — BP mode reaches closer to true
  capacity than the §6.6 patience declaration). Model mdot at the
  measured PRs (2.056/1.917/1.785; rig 20.74/20.83/20.93): Tier 1
  ~21.3/22.0/22.5 (+2.7→+7.5%), Tier 2 ~20.9/21.5/21.9 (+0.9→+4.9%) —
  **in the correct frame the entire choke-side disagreement is a
  capacity/knee error** (the rig's unique-incidence knee spans 0.2 kg/s;
  the annulus model rounds over ~1), not a PR/loss error. Pinned:
  `test_v5_rotor37.py::test_matched_pr_traversal_down_the_vertical_characteristic`
  (Tier-1 graded descent to PR 1.80, on-branch, mdot 22.4±0.4 recorded
  gap). With this, every V5 speedline hypothesis and comparison frame is
  dispositioned by measurement; the remaining physical levers are the
  inlet-swallowing capacity level and knee sharpness.

- **Centrifugal parasitic losses (2026-07-16, gate #3 discharged).** The
  deferred disk-friction + leakage + recirculation components, grounded
  verbatim from Aungier 2000 ch. 4 (theory notebook; CENT-LOSS.md
  "parasitic" section): `closures/centrifugal/parasitic.py` — Daily-Nece
  four-regime disk torque (largest-C_M rule, ×0.75 experience, Eqs
  4-21..25/4-31), the Δp_CL→U_CL→ṁ_CL leakage chain (4-17..19/4-40),
  and recirculation `I_R U2²` with Aungier's impeller D_eq (4-41..43,
  ≥0 floor). Parasitic accounting = shaft-side, post-solve scalar debits
  (`EckardtO.parasitic_breakdown`/`stage_efficiency`) — machine-level ṁ
  is exactly why they were never per-streamtube LossModel components (the
  M7 deferral, now discharged); the flow solution is untouched, so all
  existing pins stand. **Measured (Eckardt O):** laser point DF 370 +
  leak 765 + recirc 2327 J/kg ≈ 4.4% of work → η 0.969 → **0.9265** vs
  measured stage 0.88 (remaining ~4.6 pts = the unmodelled R/R₂=2
  vaneless diffuser + λ tip-distortion, the recorded refinements);
  design point recirculation grows with loading → η 0.877. Reference
  tests pin each formula against hand-computed chains
  (`test_parasitic_reference.py`) + the Eckardt integration levels.
  Recorded assumptions: disk backface gap s/r2 = 0.02, μ = 1.81e-5,
  blade length = the case chord. **Vaneless-diffuser loss added next day
  (2026-07-17):** the Coppage/Stanitz closed form (Whitfield & Baines Eq
  [30], verbatim via the theory notebook; Aungier's 5-45/5-46 marching =
  recorded refinement) as `parasitic.vaneless_diffuser_loss`, applied as
  a post-solve entropy/p0 debit in `EckardtO.stage_performance` to the
  rig's R/R₂=2 plane (cf = 0.005, the set's Braembussche-typical value).
  **The Eckardt stage comparison is now assembled end-to-end:** laser
  point η 0.969 (internal) → 0.9265 (+parasitics) → **0.9074 (+diffuser)
  vs measured stage 0.88 (+2.7 pt)**; PR_stage 2.167 vs 2.1 (+3.2%, from
  +4.7% impeller-exit); design PR_stage 3.308 vs 3.0, η 0.859.
  Remaining-gap candidates (recorded): λ tip-distortion internal loss,
  closed-form-vs-marching diffuser, cf level, η-definition subtleties.
  Pinned in `test_parasitic_reference.py`. **λ tip-distortion added same
  day — the Eckardt laser-point STAGE validation closes:**
  `parasitic.tip_distortion_loss` (Aungier Eq 4-12 blockage B₂ with its
  three verbatim terms → λ = 1/(1−B₂) Eq 120 → ω̄_λ = [(λ−1)Cm2/W2]²
  Eq 5-36; d_H/A_R per Aungier Eqs 111/113/4-13; B₂ guarded below the λ
  pole; λ's work-input role = recorded refinement). Measured:
  Δh_λ ≈ 2.0 kJ/kg at the laser point → **PR_stage 2.121 vs measured 2.1
  (+1.0%), η_stage 0.8796 vs 0.88 (−0.04 pt)** — the complete chain
  (internal 0.969 → +parasitics 0.9265 → +diffuser+λ → 0.8796) with
  every component grounded verbatim and zero locally fitted constants;
  the agreement level is partly fortuitous given the recorded geometric
  estimates (β_th ≈ β1, L_B = chord, disk gap 0.02) — pinned so drift is
  visible. Design point PR_stage 3.172 vs 3.0 (+5.7%), η 0.824.

- **Krain second impeller (2026-07-17, `KrainImpeller` in
  `v7_eckardt.py`).** The second centrifugal validation point (Krain
  1988 / Krain-Hoffmann 1989 via the Test Cases notebook; geometry from
  the 1989 Cartesian blade-coordinate table: r1h 45 / r1t 112.7 mm,
  D2 ≈ 400 mm, b2 14.7 mm, 24 blades no splitters, 30° backsweep,
  22 363 rpm / 4.0 kg/s; clearance not published → 0.5 mm recorded).
  `EckardtO` subclass, zero new mechanics. **Measured:** Tier 1 + Tier 2
  converge (validity 1.0, 0.2% agreement, impeller PR 5.00/η 0.972);
  stage chain **PR_stage 4.714 vs measured stage max 4.5 (+4.8%)** — the
  PR side generalizes to twice Eckardt's loading; **η_stage 0.905 vs
  0.84 (+6.5 pt)** — the two-point trend finding: the loss set that
  CLOSES at PR 2.1 reads LIGHT at PR 4.7 (≈3.4 pt internal at high
  loading; recirculation floors to exactly 0 at design backsweep;
  clearance assumed) — the quantified target for the next
  centrifugal-loss calibration pass, recorded not tuned. A cot(β₂ᵦ)
  wiring bug in `parasitic_breakdown` (hard-coded radial → spurious
  36.6 kJ/kg Krain recirculation) was found and fixed; Eckardt bit-
  unchanged. Pinned: `test_krain_second_impeller_measured_agreement`.

- **High-loading centrifugal calibration pass (2026-07-17) — three
  mechanisms dispositioned by measurement, Krain gap stands recorded.**
  Diagnosis first (loss-budget probe): BOTH rigs read light on
  impeller-internal loss vs measured impeller η; Eckardt's stage closes
  because the stage-side stack compensates, Krain's (+6.5 pt) doesn't.
  Implemented + measured: (1) the **Oh-native accounting**
  (`jansen_clearance_loss` verbatim Jansen-1967 + `johnston_dean_mixing_
  loss` J&D-1966 with the MERIDIONAL velocity only — Aungier quoted
  verbatim on why; `stage_performance(accounting="oh_native")`, mutually
  exclusive with the λ chain) — swings only ~2.5 kJ/kg, cannot close
  Krain's ~11 kJ/kg gap, overshoots Eckardt to −3.8 pt → **λ stays the
  default**; (2) **Aungier supercritical Mach loss**
  (`supercritical_loss`, Eqs 5-41/42 verbatim, onset = W_max > W*) —
  INERT at both rigs' 1-D mean inlet at design → not the mechanism at
  this fidelity (tip-resolved variant = recorded follow-up, Krain
  M1t′≈0.85); (3) the Krain +6.5 pt stands RECORDED (~5–6% of work at
  PR 4.7; suspects narrowed to: Krain measurement plane/η definition,
  the assumed 0.5 mm clearance, tip-resolved supercritical,
  loading-grown wake fraction). All pinned:
  `test_high_loading_calibration_dispositions` (2×2 stage levels +
  verbatim-form guards + onset semantics). CENT-LOSS.md "high-loading".

- **High-loading gap RESOLVED — the diffuser width law (2026-07-17, the
  calibration pass's conclusion).** The Krain measurement-plane query
  closed the loop: **both rigs' papers specify CONSTANT-AREA vaneless
  diffusers** (stage η measured at the diffuser exit, total-total), which
  the constant-WIDTH Coppage/Stanitz closed form badly understates.
  Implemented the recorded refinement `parasitic.vaneless_diffuser_march`
  (Aungier Eq 5-45 swirl-decay + wall-dissipation entropy ODE pair, RK2
  radial marching, selectable width law, τ-convention cross-checked
  against 5-45 exactly) and measured the cf grid on both rigs: **a
  single cf_diffuser = 0.003 lands BOTH rigs within ~2 pt η and ±2% PR**
  (Eckardt 0.8611/2.091 vs 0.88/2.1; Krain 0.8574/4.412 vs 0.84/4.5;
  Eckardt design 3.093 vs 3.0) — the "+6.5 pt Krain gap" was largely the
  width-law mismatch, and Eckardt's earlier closed-form −0.04 pt closing
  is re-classified as fortuitous cancellation. Adopted as the
  `stage_performance` default (`diffuser="march_area"`,
  `cf_diffuser=0.003` — the one locally-calibrated constant, a two-rig
  joint fit inside the plausible skin-friction range; closed form
  retained as an option); Krain's diffuser radius ratio unprinted →
  R/R₂=2 recorded assumption. All stage pins re-set deliberately
  (`test_parasitic_reference.py`, `test_v7_eckardt.py`). **The two-rig
  centrifugal stage validation now reads: PR ±2%, η ±2 pt at both
  loadings with one calibrated constant.**

- **TN D-6967 multi-speed map (2026-07-17, Fig. 17(a) digitized →
  `MEASURED_MAP`).** The two-stage overall map digitized from a 300-dpi
  render (tick-grid calibrated; the plotted design dot reproduces its
  published (3.22e3, 84.9) exactly — the control). Four speed-line ∩
  PR-contour points (90%/3.4, 90%/3.0, 70%/2.6, 50%/2.2) compared in the
  matched-PR frame by interpolating the warm-chained classical
  characteristic (the BP-secant from a far ṁ=1.9 seed measurably fails —
  near-choke matched-PR needs an adjacent seed, consistent with the §6.3
  branch-guard findings): **work within ±2.2% and flow within ~+1% across
  50–100% speed and PR 2.2–3.765 — the design-point ~1% agreement holds
  across the map.** The rig's choked-flow structure reproduces including
  the trend (map flow rises ~1.5% from 100→50% speed, 2.00→2.03; model
  2.02→2.055). Pinned: `test_v6_tnd6967.py::test_multispeed_map_matched_pr`
  (90% + 50% extremes); TND6967.md "Multi-speed map".
- **TN D-6967 first-stage-only configuration (2026-07-18,
  `TND6967FirstStage`).** The rig's second test build (stage 2 removed,
  exit faired) modelled as rows s1+r1 via a `ROW_IDS` class hook
  (behavior-preserving for the two-stage machine). Anchors: Table IV
  first-stage columns + text PR_tt,eq 2.018/PR_ts 2.298 (isentropic
  back-check from the design column reproduces 2.020 — transcription
  control). Matched-PR (characteristic interpolation): **flow +0.6%,
  work −1.7%, η_tt 0.921 vs 0.93 (−0.9 pt)** — the two-stage ~1%
  agreement class on half the row chain, and the model reproduces the
  report's headline (the rig beat its conservative design η by 6 pt;
  the model sits with the rig, not the design assumption). Same ~+1%
  capacity read (shared stator-1 governs choke in both builds). Pinned:
  `test_v6_tnd6967.py::test_first_stage_configuration_matched_pr`.
- **Gate #5 operability disposition on Rotor 37 (2026-07-18).** One §6.7
  `solve_speedline` traversal across and past the measured 100% line:
  mechanically robust (all points converge classical, PR monotone), but
  **no stall criterion fires near the measured stall 19.60 kg/s** —
  `pr_turnover` cannot fire (the rig's line is stall-truncated while PR
  still rises, 1.785→2.196), `validity_saturated` fires 21% late at
  Tier 1 (~15.5) and trivially early at Tier 2 (first point; the endwall
  D_eq window artifact), `solver_failure` never. Honest disposition: the
  operability machinery works on a real transonic rotor but the stall
  LINE needs a grounded loading criterion (Lieblein/NACA D-factor ≈ 0.6
  tip limit = the library candidate, recorded follow-up). Pinned:
  `test_v5_rotor37.py::test_speedline_operability_criteria_measured_disposition`
  + `..._tier2_validity_flag_is_the_endwall_window_artifact`; ROTOR37.md
  "Operability disposition".
- **Rotor 38 — the second transonic axial rotor (2026-07-17,
  `v5_rotor38.py`).** The axial counterpart of the Krain generalization
  check: TP-2001's high-AR sibling of Rotor 37 (same annulus/speed/flow
  family, 48 short-chord blades, AR 1.63; design rotor PR 2.105/η 0.878;
  the rig's own summary records the shortfall — **stalled before design
  flow**, peak η 0.849 at PR 1.969). Implemented via a `TABLES` class
  hook on Rotor 37 (behavior-preserving refactor, all Rotor 37 pins
  unchanged). **Measured:** both tiers converge; the η level generalizes
  (T1 +1.1 / T2 +0.5 pt); **PR does NOT track the measured high-AR
  shortfall** — T2 matched-flow +6.6% vs Rotor 37's +0.2%: the measured
  Stage 37→38 degradation (early stall / endwall sensitivity at high AR;
  no part-span damper in TP-2001) is not carried by the correlation set
  (Howell's s/h term even moves slightly the wrong way with the 48-blade
  pitch). **The axial two-point trend finding** (the AR-sensitivity gap
  = the quantified axial calibration target); the sibling differential
  is the frame-robust statement. Pinned: `test_v5_rotor38.py`;
  ROTOR37.md "Rotor 38" section.

## Commands

```bash
pip install -e ".[test]"
pytest -q                          # full suite
python tools/check_imports.py      # ARCH-2 dependency-direction
python tools/check_ad6.py          # AD-6 / smoothness lint
```