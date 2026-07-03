# Module Specification: Geometry & Grid (`slcflow.geometry`, `slcflow.grid`)

**Status:** Draft v0.1 — implementation spec for milestone M1.
**Parents:** Theory Manual v0.2 (§2, §5.1–5.2, A.1) and Architecture Spec v0.1 (ARCH-3.1–3.2, AD-2/3/8/9).
**Purpose:** Everything needed to implement, test, and freeze the meridional geometry and computational grid layer before any flow solving exists. This document is the grounding context for AI-assisted development of these two packages; nothing in it requires knowledge of closures, transport, or drivers.

Items marked **[DECIDE]** are open choices to settle during implementation review; items marked **[VERIFY]** require a check against the reference library.

---

## G-1. Scope and Responsibilities

**In scope:** wall-curve representation; station and q-o definition; grid topology; streamline initialization; streamline geometric representation; evaluation of the metric fields $\phi$, $\kappa_m$, $\varepsilon$ and arc-length quantities from nodal positions; grid-quality diagnostics; the A.1.1 orientation rule.

**Out of scope (hard):** any flow quantity ($V_m$, $h_0$, $s$, $rV_\theta$, $\rho$), any closure, any repositioning *logic* (the solver moves streamlines; this module only re-evaluates geometry given new positions), any I/O parsing beyond typed inputs.

The load-bearing contract with `assembly` (AD-3): **metric evaluation is a pure function of nodal positions.** Given the packed position block of the state vector, `grid` returns the metric arrays with no retained state and no side effects. Grid *topology* (station list, $N_{sl}$, $\psi_i$, q-o curves) is immutable per solve (AD-8).

---

## G-2. Inputs

1. **Wall curves** (two): ordered point sets $(z_k, r_k)$ per wall, or analytic curve callables. Walls are labeled `wall_0`, `wall_1` with a machine-level mapping to hub/shroud (AD-9). Point sets are the primary path (real flow paths come as coordinates); analytic callables exist for verification cases.
2. **Station definitions** (ordered): type (`DUCT | EDGE_LE | EDGE_TE | INBLADE`), owning row for blade stations, and q-o anchor specification (G-4).
3. **Grid resolution:** $N_{sl}$ and the fixed mass fractions $\psi_i$ (default: uniform; cosine clustering toward walls available for endwall resolution **[DECIDE default]**).
4. **Fidelity flags** relevant here: only $N_{sl}$ and whether repositioning will occur (Tier 1 constructs the mean line once, G-6.4).

Validation at construction (raise `ConfigError` — this is the config boundary, exceptions allowed per AD-10): walls must be non-intersecting, consistently ordered in the meridional through-flow direction, and each station's q-o must span wall-to-wall.

---

## G-3. Wall-Curve Representation

Parametric C² smoothing splines in arc length: $z(\sigma), r(\sigma)$ with $\sigma \in [0, L_{wall}]$, per §5.1. Requirements:

1. **Parametric, always.** Non-parametric $r(z)$ fails at $\phi = \pm 90^\circ$ and is banned even for axial machines (one representation, all configurations — the axial case must exercise the same code path the centrifugal case needs).
2. **Arc-length parameterization** obtained iteratively: fit on chord-length parameter, measure arc length by quadrature, re-parameterize, repeat to tolerance (2–3 passes suffice; make it a utility with a convergence assert).
3. **Smoothing:** `scipy.interpolate.make_smoothing_spline` (or `splprep` with smoothing factor) — cubic smoothing splines, C² by construction. Interpolating splines (s = 0) only for analytic verification inputs. Default smoothing weight scaled to input-point noise estimate **[DECIDE: default heuristic; expose as per-wall setting]**.
4. **End conditions:** natural by default; clamped (user-supplied end slope) available for cases where inlet/exit duct angles are known. Record in the grid metadata which was used — §5.1 notes visible sensitivity of end-station curvature.
5. Each wall exposes: `point(sigma)`, `tangent(sigma)` (unit), `slope_phi(sigma)`, `curvature(sigma)`, `arclength`, all vectorized.

---

## G-4. Quasi-Orthogonal Construction

A q-o is defined by two **anchors** — one per wall, specified as fractional wall arc length $\hat\sigma \in [0,1]$ (primary form) or as explicit points projected onto the walls (secondary; projection distance beyond tolerance is a `ConfigError`). Default q-o curve: the straight segment between anchors. General curved q-o's are supported by the same interface (`point(t)`, `tangent(t)`, `arclength`) but deferred to need **[DECIDE if ever needed; keep interface general, implement straight only]**.

Construction rules and validation:

1. Anchors for `EDGE_LE`/`EDGE_TE` come from row meridional extent on each wall (rows may have different hub/shroud LE positions — sweep — so per-wall anchors are independent).
2. `INBLADE` stations subdivide the row's wall-arc intervals by fraction (default uniform in fractional meridional distance; per-row count and spacing configurable).
3. **Non-crossing:** adjacent q-o segments must not intersect within the annulus — check at construction (segment intersection test) and fail loudly; this catches badly swept station definitions before they become solver mysteries.
4. **Orientation (A.1.1):** each q-o's parameter direction is set so that $\mathbf{e}_q \cdot \mathbf{e}_n \ge 0$ against the *expected* flow direction at construction (from wall tangent orientation), then re-validated against actual streamline tangents at every metric evaluation — a violation there is a diagnostic flag, not an exception (it can transiently occur mid-iteration).
5. Per-q-o stored data: anchors, curve, arc length $L_j$, station type, row reference.

**Grid-angle guidance (record as a construction diagnostic, not an error):** the angle between each q-o and the local expected streamline direction should stay within roughly $90^\circ \pm 30^\circ$; beyond that, the $\sin\varepsilon\, V_m \partial V_m/\partial m$ term grows and conditioning of the master-ODE integration degrades **[VERIFY guidance range against Aungier's recommendations]**.

---

## G-5. Streamline Initialization

Initial nodal positions before any flow solution exists (feeds §6.2 step 1):

1. On each q-o $j$, compute the cumulative annulus-area coordinate $A(q) = \int_0^q r\,(1-B_0)\,dq'$ with $B_0$ any prescribed initial blockage (usually 0), by quadrature on the q-o.
2. Place node $(i,j)$ at the $q$ where $A(q)/A(L_j) = \psi_i$.

Rationale: for uniform $\rho V_m \cos\varepsilon$, equal-area fractions equal mass fractions, so this is the exact solution's own definition applied with a uniform-flow assumption — the natural first iterate, and *exactly* correct for the V1 incompressible free-vortex case. The inversion of $A(q)$ must reuse the same quadrature rule the solver's repositioning residual will use (§5.4 consistency rule) — implement it once in `grid.quadrature` and import it from both places.

---

## G-6. Streamline Representation and Metric Evaluation

This is the module's numerical core and the source of most SLC convergence folklore (§5.5). Precision here is non-negotiable.

### G-6.1 Streamline fitting

Given current nodal positions $\{(z_{ij}, r_{ij})\}_j$ for streamline $i$ (derived from the position state $q_{ij}$ via the q-o curves):

1. Fit a **parametric C² smoothing spline** $z_i(\tilde m), r_i(\tilde m)$ with $\tilde m$ the chord-length parameter (one pass; full arc-length re-parameterization each solver iteration is unnecessary — derivatives below are exact in any regular parameter).
2. Slope and curvature from parametric derivatives:
$$\phi = \operatorname{atan2}(r', z'), \qquad \kappa_m = \frac{z' r'' - r' z''}{\left(z'^2 + r'^2\right)^{3/2}},$$
with $'\equiv d/d\tilde m$. The atan2 form and the parametric curvature formula are exact through $\phi = \pm 90^\circ$ and are **the only permitted forms** (no $dr/dz$ anywhere in this module).
3. **Sign check:** with $\tilde m$ increasing downstream, the formula above yields $\kappa_m = d\phi/dm$ per the A.1 convention — add a unit test on a circular-arc streamline of known center to pin the sign permanently.
4. Meridional arc length $m_{ij}$ accumulated by quadrature along the fit (needed for $\partial/\partial m$ stencils and transport).

### G-6.2 Smoothing policy

Nodal positions carry iteration noise; curvature amplifies it (§5.5). Layered defense, each independently switchable for diagnosis:

1. Smoothing weight in the streamline fit — default small but nonzero **[DECIDE default via V2/noise test, G-8.5]**; must → 0 under grid refinement so it never pollutes the discretization order (tie weight to local node spacing).
2. Optional curvature-field under-relaxation lives in the *solver* (§5.5), not here — this module always reports the curvature of the positions it was given. (Keeping this module honest makes solver-side damping auditable.)
3. End treatment: first/last stations inherit end-condition sensitivity from G-3.4; expose per-machine choice and record it.

### G-6.3 Lean angle and metric assembly

At each node: streamline unit tangent $\mathbf{e}_m$ from the fit, q-o unit tangent $\mathbf{e}_t$ from the q-o curve, then
$$\sin\varepsilon = \mathbf{e}_t \cdot \mathbf{e}_m, \qquad \cos\varepsilon = \mathbf{e}_t \cdot \mathbf{e}_n,$$
$\varepsilon = \operatorname{atan2}(\sin\varepsilon, \cos\varepsilon)$, with the A.1.1 check $\cos\varepsilon \ge 0$ flagged if violated. Output bundle (`GridMetrics`, all `(N_sl, N_qo)`):
`phi, kappa_m, eps, m` (meridional arc length), `r, z`, plus per-q-o `L_j` and the local aspect ratio $\Delta m / \Delta q$ field that §6.4's relaxation logic consumes.

`evaluate_metrics(topology, q_positions) -> GridMetrics` is the pure function of ARCH-3.2; property test: idempotent, no mutation of inputs, bit-identical on repeated calls.

### G-6.4 Degenerate configurations

**Tier 1 (N_sl = 1):** the mean line is constructed once at $\psi = 0.5$ by G-5 and never repositioned; metrics evaluate identically (a one-row array). No special-case code beyond `N_sl = 1` — the fit in G-6.1 works on a single streamline unchanged. **Tier 2:** full grid, metrics evaluated, but the solver's fidelity flags zero the $\kappa_m$/$\varepsilon$ terms — this module still reports true metrics (they appear in diagnostics and in the Tier-2→3 consistency test).

---

## G-7. Data Structures and API Sketch

```python
# geometry/
class WallCurve:            # G-3; from_points(), from_callable()
class QOCurve:              # G-4; straight segment impl
class StationDef:           # type, row_ref, anchors
class FlowPath:             # walls + ordered stations; validate()

# grid/
class GridTopology:         # frozen: stations, qo_curves, N_sl, psi, wall labels
class GridMetrics:          # frozen SoA bundle per G-6.3
def initialize_positions(topology) -> Array            # G-5, shape (N_sl, N_qo)
def evaluate_metrics(topology, q_positions) -> GridMetrics   # pure, G-6.3
def grid_quality_report(topology, metrics) -> QualityReport  # G-8.6
# grid/quadrature.py — the single quadrature + cumulative-inversion utility (G-5)
```

Dependencies: `geometry` ← `grid`; neither imports `fluid`, `closures`, `transport`, `assembly` (enforced, AD-5 direction).

---

## G-8. Module Test Plan

Standalone, no solver required; these become permanent regressions.

1. **Analytic annuli, exact metrics:** cylindrical duct ($\phi=0$, $\kappa_m=0$ to machine precision with interpolating fit); conical duct ($\phi=$ const, $\kappa_m=0$); circular-arc walls (known $\kappa_m$, signed check per G-6.1.3); 90°-bend axial→radial path (exercises $\phi$ through $90^\circ$, parametric-form guard).
2. **Radial-inflow orientation:** IFR-turbine-like path (radial-in → axial-out) — assert A.1.1 auto-orientation puts $\cos\varepsilon \ge 0$ and the wall-label mapping is reported correctly.
3. **Convergence order:** metric errors vs. node count on analytic cases — expect 2nd order or better with smoothing → 0 (G-6.2.1); this pins the "smoothing doesn't pollute accuracy" rule.
4. **Initialization exactness:** G-5 positions reproduce the analytic equal-area partition on cylindrical/conical annuli to quadrature tolerance; round-trip $A(q)$ inversion property test.
5. **Noise robustness:** inject controlled nodal noise (amplitude sweep) on a smooth case; measure curvature error vs. smoothing weight → calibrates the G-6.2.1 default and produces the plot that later justifies it.
6. **Quality report:** aspect-ratio field, $\varepsilon$ statistics, q-o angle guidance (G-4), non-crossing checks — snapshot-tested on one axial and one centrifugal path.
7. **Purity/immutability:** `evaluate_metrics` property tests (G-6.3); topology frozen-ness.

Acceptance for M1: all of the above green, plus the V2 curvature-machinery case runs on *frozen* streamlines (metrics feeding a hand-integrated master ODE in a notebook — no solver code) and matches the A.5 special-case 2 relation $\partial(\ln V_m)/\partial q = \kappa_m$.

---

## G-9. Known Pitfalls Checklist (review against every PR touching this module)

Non-parametric representations sneaking in for "simple axial" cases. Curvature from second differences of nodes instead of the fitted spline. Smoothing weight fixed in absolute units (breaks refinement order). $\varepsilon$ from angle subtraction ($\phi - \gamma$ arithmetic) instead of tangent dot products — the identity A.6 is for *transcription*, not computation; dot products are branch-safe at all angles. Assuming $q=0$ is the hub (AD-9). Quadrature rule drift between initialization and (future) repositioning residual (G-5). Metrics evaluation caching across position updates (violates purity; caching belongs to the caller).