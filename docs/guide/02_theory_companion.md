# Guide 2 — Theory Companion

> **What this document is.** The pedagogical twin of the theory manual: the
> same physics, derived and explained rather than specified. The goal is
> that after reading it you could re-derive the solver's formulation on a
> whiteboard — and explain *why* each term, sign, and conversion is the way
> it is. It is descriptive, not normative: **the theory manual is the
> source of truth, and its Appendix A is normative for signs.** Where this
> guide walks a derivation, it is walking Appendix A's derivation, not an
> independent one; any discrepancy is a bug here, not there.
>
> Section references (§, A.x, B.x, C.x) are theory-manual sections; `G-n`
> is the geometry/grid module spec; `AD-n` the binding decisions. Worked
> numbers were produced by running the code at the header commit — mostly
> via `closures/conversions.py` on the run-A state from Guide 1.
>
> Written at commit `d7b7b27` (2026-07-07); §8's closing check refreshed to
> `2916c57` (2026-07-08) after the Lieblein loss calibration (the fixed
> illustrative B.2/B.3/B.4 conversions are unchanged — only run A's actual
> Δs moved); suite 378 tests green.

---

## 1. The problem being solved, in one page

A turbomachine's flow is 3-D, unsteady, and turbulent. Throughflow analysis
buys a ~10⁴× cheaper problem with two moves:

1. **Circumferential averaging.** Average the steady flow in θ. The
   blade-to-blade structure disappears; what survives of the blades is (a)
   a mean **blade force** field `f_b` inside blade rows (the averaged
   pressure difference across the passage), (b) **blockage** `B` (metal +
   boundary layer occupying flow area), and (c) **entropy sources** (the
   losses). The flow becomes axisymmetric: fields of `Vm`, `Vθ`, `h0`, `s`
   over the meridional plane `(z, r)`.
2. **Streamlines and quasi-orthogonals as the grid.** Instead of a fixed
   mesh, use the flow's own structure: `N_sl` meridional streamlines
   (surfaces of constant mass fraction ψ) crossed by `N_qo` fixed
   station curves ("quasi-orthogonals", q-o's). Momentum is enforced
   *along* each q-o, transport *along* each streamline. The unknowns are
   where the streamlines sit and how fast the flow moves — a few hundred
   numbers for a whole machine.

Three modeling commitments (§1) shape everything downstream:

- **Entropy is the loss currency.** Correlations may speak in
  pressure-loss or enthalpy-loss coefficients; internally everything is
  Δs (converted at the interface, §8 below). Rationale (Denton 1993):
  entropy is frame-independent (a rotor's loss doesn't change when you
  change reference frame; its p0-loss does), additive across mechanisms,
  and survives the move to real gases.
- **Everything is a residual.** Each balance law is a residual function of
  the state, so a classical Picard driver and a global Newton driver can
  share one assembly (Guide 1 §D; §6).
- **Closures are C¹ everywhere** — smooth saturation outside calibration,
  never clamps or exceptions (§10 below).

The assumptions to keep in mind: steady, adiabatic, axisymmetric;
viscosity appears only as entropy sources, blockage, and optional spanwise
mixing — never as resolved stresses.

## 2. The geometry language (§2, A.1)

Everything is written in the meridional plane with two local frames.

**The streamline frame.** Along a streamline, `m` is arc length
(positive downstream) and φ the slope from the axis: `sinφ = dr/dm`,
`cosφ = dz/dm`. Axial flow: φ = 0; a centrifugal exit: φ = 90°. Unit
vectors:

```
e_m = cosφ e_z + sinφ e_r         (along the flow)
e_n = −sinφ e_z + cosφ e_r        (e_m rotated +90°)
```

The **meridional curvature** is `κ_m = dφ/dm`, positive when the
streamline curves toward `+e_n`. From the Frenet relations
(`de_m/dm = κ_m e_n`, A.1), a particle's meridional acceleration is

```
a_mer = Vm (∂Vm/∂m) e_m + Vm² κ_m e_n
```

— the familiar tangential/centripetal split. That `Vm²κ_m` centripetal
term is the "SLC" in streamline curvature: it is what Tier 2 ignores and
Tier 3 keeps.

**The q-o frame.** A q-o is a *fixed* curve across the span (streamlines
move during iteration; q-o's never do — AD-8). Arc length along it is `q`,
and its lean relative to the local streamline *normal* is ε:

```
e_q = cosε e_n + sinε e_m
```

so ε = 0 means the q-o is genuinely orthogonal to the flow, and ε > 0
means it leans downstream. The mass-flux velocity component through the
q-o surface is `Vm cosε`.

**The orientation rule (A.1.1) — why the code says `wall_0`, not
`hub`.** `e_n` is *always* the +90° rotation of `e_m` — it is not defined
as "hub-to-shroud". The convention orients `q` so that `e_q·e_n ≥ 0`,
which forces ε ∈ (−90°, +90°) and `cosε ≥ 0` everywhere. Which *physical*
wall ends up at q = 0 then depends on the machine: the hub for a
conventional axial compressor, but potentially the shroud for a
radial-inflow turbine, because the sense of `e_n` relative to the walls
flips with the flow direction. Two things lean on `cosε ≥ 0`: the
continuity flux is guaranteed non-negative (A.7), and the subsonic branch
selection is well-posed (§6.5). This is AD-9's physical origin — assuming
`q=0 ⇔ hub` is not a style violation, it is a wrong answer on half the
machine types this solver exists to cover.

**Flow angles.** `tanα = Vθ/Vm` (absolute), `tanβ = Wθ/Vm` (relative,
`Wθ = Vθ − Ωr`), both **positive toward rotor rotation**, for compressors
and turbines alike, metal angles included (§2.4). Any correlation whose
source paper uses another convention re-maps *inside* the closure. This
single-convention rule is where turbomachinery sign bugs go to die — see
the crib card in §11.

**Reading the literature (A.6).** Novak and Aungier write the momentum
equation with the q-o angle measured from the *radial* direction, γ. The
exact mapping is `γ = ε − φ`. When a library source seems to disagree with
the manual, apply this substitution before concluding anything.

## 3. The master equation, derived (A.2–A.5)

The whole solver hangs on one ODE. Here is its derivation, compressed to
the four moves of Appendix A — reproduce these on a whiteboard and you
have re-derived the kernel.

**Move 1 — acceleration (A.2).** For steady axisymmetric flow the
convective operator along the meridian is `Vm ∂/∂m`. Differentiating
`V_mer = Vm e_m` with the Frenet relations, and adding the centripetal
effect of swirl (`−Vθ²/r` along `e_r`):

```
a = Vm(∂Vm/∂m) e_m  +  Vm²κ_m e_n  −  (Vθ²/r) e_r
```

Three accelerations: speeding up along the path, turning in the meridional
plane, and being flung outward by swirl.

**Move 2 — project momentum onto the q-o (A.3).** Take the inviscid
momentum equation `(V·∇)V = −(1/ρ)∇p + f_b` and dot it with `e_q`. Using
`e_r·e_q = cos(ε−φ)`:

```
(1/ρ) ∂p/∂q = f_b,q − Vm(∂Vm/∂m) sinε − Vm²κ_m cosε + (Vθ²/r) cos(ε−φ)   (A.1)
```

Read it as force balance across the span: the spanwise pressure gradient
is whatever is left after blade force and the three accelerations take
their cuts.

**Move 3 — eliminate the pressure gradient (A.4).** We don't want `p` as
a variable. The Gibbs relation `dh = T ds + dp/ρ` along the q-o, with
`h = h0 − ½(Vm² + Vθ²)`, gives

```
(1/ρ) ∂p/∂q = ∂h0/∂q − T ∂s/∂q − Vm ∂Vm/∂q − Vθ ∂Vθ/∂q                  (A.2)
```

This is the move that installs **entropy** as a first-class variable: the
pressure gradient is traded for gradients of the three *transported*
fields (`h0`, `s`, and — after the next step — `rVθ`), which is exactly
what makes the elimination form (§5) possible.

**Move 4 — equate and tidy (A.5).** Set (A.1) = (A.2), solve for
`Vm ∂Vm/∂q`, and merge the two swirl terms using the geometric identity
`∂r/∂q = cos(ε−φ)`:

```
Vθ ∂Vθ/∂q + (Vθ²/r) ∂r/∂q  =  (Vθ/r) ∂(rVθ)/∂q
```

The result is the boxed master equation of §3.1:

```
Vm ∂Vm/∂q = ∂h0/∂q − T ∂s/∂q − (Vθ/r) ∂(rVθ)/∂q
          + Vm² κ_m cosε + Vm (∂Vm/∂m) sinε − f_b,q
```

Note the **minus** on `f_b,q`: it falls straight out of `f_b` sitting with
`+` on the momentum RHS. Physically, a blade force pushing toward `+e_q`
*supports* a spanwise pressure rise, so the `Vm` field doesn't have to.
The manual's changelog records that v0.1 had this sign wrong and v0.2
fixed it — which is precisely why Appendix A is normative and code never
"fixes" a sign locally (see §11).

## 4. Reading the equation

Divide through by `Vm` and it is a first-order ODE for `Vm(q)`:

| Term | Physics | Vanishes when |
|---|---|---|
| `∂h0/∂q` | Spanwise work gradient — a forced-vortex rotor or non-uniform inlet makes the energy differ across the span, and velocity must follow | uniform work + inlet |
| `−T ∂s/∂q` | Spanwise loss gradient — endwall/secondary loss piles entropy near walls; hotter, lighter fluid there must move slower for the same pressure field | uniform loss |
| `−(Vθ/r) ∂(rVθ)/∂q` | Simple radial equilibrium: swirl's centripetal demand | free vortex (`rVθ` const) |
| `+Vm² κ_m cosε` | Streamline curvature (Tier 3): meridional turning needs a spanwise pressure gradient, which redistributes `Vm` | straight streamlines |
| `+Vm (∂Vm/∂m) sinε` | Lean correction (Tier 3): if the q-o is not normal to the flow, part of the *streamwise* acceleration aliases into the spanwise balance | orthogonal q-o's |
| `−f_b,q` | In-blade blade force from lean/sweep (A.8; deferred — zero for radially stacked blades) | radial stacking |

Three special cases are worth carrying in your head, because each is a
verification case:

1. **Classic radial equilibrium** (φ = 0, ε = 0, q = r, duct): the ODE
   collapses to `Vm dVm/dr = dh0/dr − T ds/dr − (Vθ/r) d(rVθ)/dr`. With
   uniform `h0`, `s` and free-vortex swirl, the right side is zero — `Vm`
   uniform across the span. That is **V1a**, and the code reproduces it to
   ~6·10⁻⁸ relative (C.1).
2. **Swirl-free curved duct** (Vθ = 0, ε = 0): `∂(ln Vm)/∂q = κ_m` —
   velocity grows toward the center of curvature, the meridional analogue
   of a free vortex around the bend. That is **V2** (C.2).
3. **Flags off**: every Tier-3-exclusive term carries a factor `κ_m`,
   `sinε`, or `cosε`·flag — zeroing `curvature_term` and `lean_term`
   recovers Tier 2 *exactly*, with no branching. That is **V3**, asserted
   bit-for-bit at 10⁻¹⁰ (C.4). The fidelity flags are physics switches,
   not code switches (AD-1); see §9.

One practical footnote from Guide 1: the lean term's `∂Vm/∂m` is evaluated
from the *lagged* `Vm` field (§5.2 — a streamwise derivative can't be
known while integrating a single q-o), and the curvature entering the ODE
is the §5.5 under-relaxed blend. Both are frozen data inside a residual
evaluation (AD-3/AD-4).

## 5. Continuity, the elimination, and choke (§3.2, §5.3–5.4, §6.1, A.7)

The master equation gives the *shape* of `Vm(q)` from one boundary value
`Vm(q=0)` — a one-parameter family (§5.3). What pins the parameter is
continuity (§3.2):

```
ṁ = 2π ∫ ρ Vm cosε (1−B) r dq
```

one scalar equation per station. And the fixed mass fractions ψᵢ between
adjacent streamlines (§3.2's streamtube form) pin the streamline
*positions*. Hence the elegantly small residual vector of §6.1: momentum
never appears — it is satisfied **by construction** during assembly
("the elimination"); only continuity (per station) and mass-fraction
errors (per interior streamline) remain. Guide 1 §D.3–D.5 shows this
machinery running.

**Why the capacity peak exists (A.7).** During a continuity solve the
per-node invariants are `h0`, `s`, `rVθ`, `r` — so as the trial `Vm`
rises, `dh = −Vm dVm` (energy conservation eats the static enthalpy) and,
at constant entropy, `dρ = ρ dh/a²`. Therefore

```
∂(ρVm)/∂Vm = ρ (1 − Mm²),     Mm ≡ Vm/a
```

The streamtube mass flux `ρVm` peaks at **meridional** Mach one — not
total Mach one, because swirl is frozen through `rVθ` during the scan.
Consequences, all visible in the code (Guide 1 §D.4): the continuity
curve `F_j(Vm_q0)` rises to a peak and falls; the subsonic branch is the
`Mm < 1` side; if the peak is below the demanded ṁ the station simply
cannot pass the flow (`CHOKE_LIMITED`); and a heavily swirling station
chokes at *low* `Vm` (swirl kinetic energy has already eaten the static
enthalpy and with it the sound speed) — the correct physics near a radial
diffuser or turbine exit.

## 6. Streamwise transport: rothalpy, Euler work, entropy (§3.3–3.5)

Between stations, three fields ride each streamtube.

**Rothalpy.** Define `I ≡ h0 − Ω rVθ`. Expanding `h0 = h + ½(Vm² + Vθ²)`
and `W² = Vm² + (Vθ − U)²` gives the identity

```
I = h + ½W² − ½U²
```

— the relative-frame total enthalpy minus the centrifugal potential. In
the rotor frame the blade force is attached to the wall: it does **no
work** on the flow. So for steady adiabatic flow, `I` is conserved through
a rotor **including losses** (dissipation converts kinetic energy to heat;
it does not add energy). The only caveat the manual flags: casing
windage/shear work is neglected (§3.3, a `[VERIFY]` for centrifugal
backface friction).

Rearranged, rothalpy conservation *is* the Euler work equation:

```
Δh0 = Ω Δ(rVθ)
```

Work enters **exclusively** through the swirl change (§4.2) — there is no
independent "work coefficient" anywhere in the solver. This is why Guide
1's Euler check (400 × 41.16 = 16 464 J/kg, exactly the observed Δh0) held
to round-off: the transport update is this identity, literally
(`transport/streamwise.py:82`).

**The transport rules**, then, are three one-liners per station interval
(§3.3–3.5): ducts conserve `rVθ` and `h0`; blade rows set exit `rVθ` from
the swirl closure, `h0` follows by Euler work, and `s` accumulates the
loss closure's Δs (plus, when enabled, the §3.6 mixing redistribution —
the operator itself is Guide 3 material). With ω = 0 the rotor rule *is*
the stator rule *is* the duct rule — the §8 degeneracy at equation level.

**Why entropy and not p0-loss as the running currency:** `s` is additive
across loss mechanisms, indifferent to reference frame (crucial when a
rotor row's loss is assessed in the relative frame but transported in the
absolute), and its transport rule is a bare sum. All the frame- and
radius-dependence gets quarantined in one place: the conversions of §8.

## 7. Blade rows as boundary conditions on transport (§4)

A blade row, to this solver, is *two numbers per streamtube* delivered by
closures at the trailing edge: exit `rVθ` (swirl closure: deviation-based
for axial rows, slip-based for radial — one interface either way, §7.1)
and Δs (loss closure, §4.4). Everything else — incidence (§4.3), metal
angles, solidity, throat — is *input* to those closures through the §4.1
row data contract. The kernel transports what the closures decide; it
never contains a correlation (AD-5).

Two subtleties the manual pins down:

- **Incidence** `i = β1 − β1,blade` is the primary off-design input, and
  everything downstream of it must obey the C¹ saturation rule — Guide 1's
  run C showed what deep off-design looks like when the coefficients are
  honest about being uncalibrated.
- **In-blade distribution** (§3.4–3.5): with INBLADE stations, `rVθ(m)`
  ramps LE→TE by a prescribed C¹ schedule, and Δs follows the same
  schedule class (B.5.1) unless a component is explicitly local. The
  schedule is a *modeling input*, not physics — the physics constraint is
  only that endpoints match and the ramp is smooth.

## 8. Loss → entropy, worked (Appendix B)

This is the part of the formulation where the most codes go wrong
(§4.4 calls it "the single most common implementation bug class in
throughflow codes"), so it earns worked numbers. All conversions live in
`closures/conversions.py`; all are *charged at row-exit conditions*.

**The master relation.** For a perfect gas,
`s₂ − s₁ = cp ln(T₂/T₁) − R ln(p₂/p₁)`. Compare the actual and ideal exit
states at a **common stagnation temperature** (they share it — energy is
set by work, not loss) and the cp term cancels:

```
Δs = −R ln( p0_actual / p0_ideal )        (at common T0)
```

Every pressure-based coefficient reduces to this via one question: *what
is the ideal exit stagnation pressure, and what did loss make of it?*

**B.1 — the ideal reference state.** For a stator/duct: the inlet state
(`T0`, `p0` unchanged). For a rotor: work in the relative frame, where
rothalpy gives `T0r,2 = T0r,1 + (U₂² − U₁²)/(2cp)` and the loss-free
pressure follows the isentrope: `p0r,2,id = p0r,1 (T0r,2/T0r,1)^(γ/(γ−1))`.
This **radius re-referencing** is what makes the conversion correct for
radial machines. Numbers (run-A LE state, `T0r,1 = 316.5 K`,
`p0r,1 = 140.7 kPa`): an axial row (`U₂ = U₁`) leaves the reference
untouched, but a centrifugal-style change `U: 100 → 362 m/s` moves it to
`T0r,2 = 376.8 K`, `p0r,2,id = 259.0 kPa` — the ideal exit pressure nearly
**doubles** before loss even enters. An axial-habit code that skips this
would charge the entire centrifugal pressure rise as if it were loss
context, and get Δs badly wrong.

**B.2 — compressor ω̄** (`ω̄ = (p0r,2,id − p0r,2)/(p0r,1 − p₁)`, *inlet*
relative dynamic head as reference). Worked on the actual run-A meanline
LE state (Guide 1: `Vm = 91.16` at `r = 0.4743`, so `W₁ = 210.5 m/s`,
`T₁ = 294.5 K`, `p₁ = 109.3 kPa`, inlet head `p0r,1 − p₁ = 31.4 kPa`):

```
ω̄ = 0.06  →  p0r deficit = 0.06 × 31 431 = 1 886 Pa
           →  Δs = −R ln(138 862/140 748) = 3.872 J/(kg·K)
first-order check: R·ω̄·(p0r,1−p₁)/p0r,2,id = 3.846  ✓
```

(The `p₁ = 109 kPa` figure is above ambient because the case's `s = 0`
datum sets the pressure level — self-consistent, not sea level.)

A trap worth internalizing: with the *same* `ω̄ = 0.06` but the
centrifugal `U: 100 → 362` re-referencing, Δs = **2.098** J/(kg·K) —
smaller, because the same inlet-head deficit is a smaller *fraction* of
the (re-referenced, much larger) ideal exit pressure. Identical
coefficient, different entropy: the coefficient is meaningless without
its reference state, which is why B.5.2 forbids summing coefficients with
different references — convert each to Δs first, then sum.

**B.3 — turbine Y** (`Y = (p0,1 − p0,2)/(p0,2 − p₂)`, *exit* dynamic head
reference — note it references the *actual* exit p0, hence the algebraic
rearrangement `p0,2 = (p0,2,id + Y p₂)/(1 + Y)`). Worked on a
nozzle-like stator state (`T0 = 1200 K`, `p0 = 8.0 bar`, `V₂ = 550 m/s`
⇒ `T₂ = 1049.5 K`, `p₂ = 5.00 bar`):

```
Y = 0.10  →  p0,2 = (p0,2,id + Y·p₂)/(1+Y) = (800 000 + 0.1×500 411)/1.1
          = 772 765 Pa
          →  Δs = −R ln(772 765/800 000) = 9.943 J/(kg·K)
```

**B.4 — kinetic-energy ζ** (`ζ = (h₂ − h₂s)/(½V₂²)`): compares actual and
isentropic exit *static* states at the same `p₂`, so the conversion is
`Δs = cp ln(T₂/T₂s)` with `T₂s = T₂ − ζV₂²/(2cp)`. Same exit state,
`ζ = 0.05` → Δs = 7.232 J/(kg·K). The guard `ζV₂²/(2cp T₂) < 1` (here
0.007, nowhere near binding) is an **assert, not a clamp**: §7.3 requires
the upstream *correlation* to saturate its coefficient smoothly, so if the
guard ever fires the correlation is broken and hiding it would be worse.
This is also why M6 mapped the turbine trailing-edge loss through B.3
rather than B.4 — B.3's algebra cannot assert, keeping the evaluation
exception-free per AD-10.

**The enthalpy-loss form** (centrifugal internal losses,
`delta_s_enthalpy_loss`): dissipation `Δh_loss` charged at local `T` gives
the reheat form `Δs = cp ln(1 + Δh_loss/(cp T))` — always finite and
non-negative, first-order limit the textbook `Δh_loss/T` (worked:
3 000 J/kg at 320 K → 9.331 vs. 9.375 first-order).

**Closing the loop to Guide 1.** Run A's converged rotor charged
Δs = 1.626 J/(kg·K). The master relation run backwards:
`exp(−Δs/R) = 0.9944` — the loss knocked 0.56% off the achievable
stagnation-pressure ratio. Check: the ideal PR from the work is
`(T0,ex/T0,in)^(γ/(γ−1)) = (316 463/300 000)^3.5 = 1.2056`, and
`1.2056 × 0.9944 = 1.1988` — run A's reported pressure ratio, exactly.
Work sets the ideal PR through Euler; entropy discounts it through the
master relation; the two never mix accounts. (Δs and the PR here are at
the `2916c57` calibration state — see Guide 1's run-table note; the *B.2/B.3
worked conversions above use fixed illustrative coefficients and do not
move*. This closing identity is what is invariant: whatever the loss, PR =
ideal × exp(−Δs/R).)

## 9. Fidelity flags as physics switches (§8)

The degeneration matrix (§8) is now readable as physics:

- **`curvature_term = 0`** removes `Vm²κ_m cosε`: streamlines may be
  curved *geometrically*, but the momentum balance stops feeling the
  centripetal demand of meridional turning. What remains is radial
  equilibrium evaluated along each q-o — Tier 2. Defensible when
  streamlines are nearly straight (parallel-annulus axial machines);
  meaningless for a 90° centrifugal bend, which is why the radial cases
  are the Tier-3 stress tests.
- **`lean_term = 0`** removes the `sinε` aliasing of streamwise
  acceleration. Harmless when q-o's are near-normal to the flow.
- **`n_sl = 1`** is not a flag at all: with one streamline there are no
  spanwise gradients to integrate and the master ODE is trivially
  satisfied — the meanline is the coarsest *quadrature* of the same
  equations (one-point area rule, §5.4), not a different model.
- **`mixing_term`** is not a master-equation term: it scales the §3.6
  operator applied to the transported fields *between* iterates. It
  therefore cannot break the Tier 2 ≡ Tier 3 identity, which is asserted
  with mixing off.

The standing V3 regression (§8's consistency requirement) is what makes
this section *enforced* rather than aspirational: on a straight annulus
Tier 3's extra terms vanish identically, and the code proves it
bit-for-bit.

## 10. Why the C¹ discipline is load-bearing (§7.3)

The §7.3 rules — C¹ continuity everywhere, smooth saturation outside
calibration, validity `v ∈ [0,1]` returned with every evaluation, never
exceptions — read as style until you connect them to the numerics:

- The Newton driver differentiates the residual by finite differences.
  A `C⁰` kink in any quantity that reaches the residual makes the
  Jacobian column garbage precisely where the solver is working hardest
  (near saturation = near operating limits).
- Continuation walks a speedline by small parameter steps; a hard clamp
  turns a smooth solution path into one with corners, where step control
  and warm starts degrade.
- Deep off-design evaluation *must* return something finite and smooth,
  because the classical driver's early iterates (and run C's honest
  failure) routinely visit states far outside any correlation's comfort
  zone. The correlation's opinion of that state is the *validity* channel,
  not an exception.

Division of labor, in one sentence: **correlations saturate smoothly and
report validity; conversions assume admissible inputs and may assert;
the driver converts anything that still goes non-finite into a typed
status** (AD-10, Guide 1 §8). The enforcement stack is
`closures/smoothmath.py` (the C¹ primitive library), the refinement-
scaling C¹ test pattern (`tests/test_smoothmath.py`), and the
`tools/check_ad6.py` lint.

## 11. The sign-convention crib card

The frozen set, distilled. If a derivation, a paper, or generated code
disagrees with one of these, the manual wins (Appendix A preamble) — map
the source's convention, never flip a sign locally.

| Convention | Statement | Trap it prevents |
|---|---|---|
| Rotation | Ω positive right-handed about +z; `U = Ωr` | — |
| Flow & metal angles | α, β positive **toward rotation**, from the meridional; same rule for compressors and turbines | correlations whose papers measure turbine angles positive the other way (re-map inside the closure) |
| Streamline slope | φ from +z; parametric fits so φ = 90° is regular | radial machines break `r(z)` non-parametric fits |
| Curvature | `κ_m = dφ/dm`, positive curving toward `+e_n` | sign of the Tier-3 term on concave vs. convex walls |
| Normal | `e_n` = e_m rotated +90° — **never** "hub-to-shroud" | AD-9; radial-inflow machines reverse the wall↔normal sense |
| Q-o orientation | `e_q·e_n ≥ 0` ⇒ `cosε ≥ 0` always | keeps the continuity flux non-negative and branch selection well-posed (A.7) |
| Blade force | enters the master equation as **−f_b,q** | the v0.1→v0.2 manual correction (A.5) |
| Literature mapping | `γ = ε − φ` (Aungier/Novak angle-from-radial) | "the book disagrees" false alarms (A.6) |

This discipline is not theoretical: the manual's own changelog records the
`f_b,q` sign fix (v0.2), and the 2026-07 independent audit caught and
fixed a real sign defect in the turbine exit-angle closure (the throat
angle must be signed by the TE turning direction — commit `4055e9b`). Sign
errors are the bug class this whole apparatus exists to contain.

## 12. Check your understanding

1. **Why does entropy appear in a momentum equation at all?** Because the
   Gibbs relation was used to eliminate the pressure gradient (A.4):
   `(1/ρ)∂p/∂q` becomes gradients of `h0`, `s`, and the velocities. Loss
   enters the spanwise momentum balance as `−T ∂s/∂q` — entropy
   stratification tilts the velocity profile exactly as a pressure field
   would.
2. **A free-vortex duct flow has uniform `h0` and `s`. What does the
   master equation predict, and which test pins it?** All three REE terms
   vanish (`rVθ` const kills the swirl term), so `∂Vm/∂q = 0` — uniform
   meridional velocity. V1a, reproduced to ~6·10⁻⁸ (C.1).
3. **Why does a q-o choke at meridional Mach 1 rather than total Mach 1?**
   During the continuity scan, `rVθ` is per-streamtube invariant, so swirl
   kinetic energy is frozen: only `Vm` trades against static enthalpy,
   giving `∂(ρVm)/∂Vm = ρ(1 − Mm²)` (A.7). Swirl still matters — it lowers
   `a` and thus the capacity — but the peak is in `Mm`.
4. **A rotor row is lossy. Is rothalpy still conserved through it? Why?**
   Yes. In the rotating frame the blade surfaces are stationary, so the
   blade force does no work; dissipation redistributes energy into heat
   without adding any. `I = h0 − ΩrVθ` changes only via heat transfer or
   casing windage, both neglected (§3.3).
5. **Two loss components arrive as `ω̄ = 0.03` (inlet-head reference) and
   `Y = 0.04` (exit-head reference). What is the one legal way to
   combine them?** Convert each to Δs individually at the B.1-referenced
   exit state, then add the entropies (B.5.2). Adding the coefficients is
   meaningless — they are normalized by different dynamic heads (§8's
   worked B.2 pair shows the same coefficient producing different Δs under
   different references).
6. **Why is the B.4 guard an `assert` when AD-10 forbids exceptions?**
   The division of labor: §7.3 obliges the *correlation* to saturate its
   coefficient smoothly before conversion, so an in-range coefficient can
   never trip the guard. If it trips, the correlation is defective —
   a programming-contract violation (assert loudly), not out-of-domain
   physics (saturate smoothly). Where even that is unacceptable, use a
   conversion that cannot assert — the reason M6 routed trailing-edge
   loss through B.3.
7. **What single substitution reconciles this manual's master equation
   with Aungier's?** `ε = φ + γ` (A.6) — the manual measures q-o lean from
   the streamline normal; Aungier and Novak measure the q-o angle from the
   radial direction.
