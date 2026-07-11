# Streamline Curvature Throughflow Solver — Theory Manual

**Status:** Draft v0.3 — formulation baseline; master-equation derivation and loss-conversion appendices completed. Changelog: v0.2 corrects the sign of the blade-force term in §3.1 ($+f_{b,q} \to -f_{b,q}$, see A.5) and adds the q-o orientation convention (A.1.1). v0.3 resolves the §6.4 relaxation-criterion [VERIFY]: the implementation's measured stability envelope follows $(\Delta m / L_{qo})^{3/2}$, not Wilkinson's $(\Delta m/\Delta q)^2$ aspect form; calibrated constants and the envelope data in Appendix C.3.
**Scope:** Reduced-order aero-thermodynamic modeling of axial, radial, and mixed-flow compressors and turbines via a single streamline-curvature (SLC) kernel with degenerate meanline and radial-equilibrium modes.

Items marked **[VERIFY]** are constants, signs, or model details that must be checked against the primary sources in the reference library before implementation is considered frozen.

---

## 1. Purpose, Scope, and Design Principles

This manual defines the mathematical formulation, discretization, and solution strategy for the throughflow solver. It is the single source of truth for nomenclature, sign conventions, and equation forms; all code and all AI-assisted code generation shall conform to it. Where the literature offers multiple equivalent formulations, this manual selects exactly one and records the rationale.

The governing design principles are:

1. **One kernel, three fidelities.** The meanline (Tier 1) and streamline radial-equilibrium (Tier 2) modes are degenerate configurations of the full SLC solver (Tier 3), obtained by grid collapse and term deactivation — never by separate code paths. See §8.
2. **Configuration agnosticism.** The momentum equation is written in general quasi-orthogonal (q-o) form valid for arbitrary meridional flow-path orientation. Machine-type specificity lives exclusively in closure models (losses, deviation/slip, blockage) behind stable interfaces. See §7.
3. **Entropy-based loss accounting.** Internally, all irreversibility is carried as specific entropy $s$. Correlations may return pressure- or enthalpy-loss coefficients; these are converted at the interface (§4.4). Rationale: entropy is frame-independent, additive, and generalizes cleanly to real-gas property models (Denton, 1993).
4. **Residual-oriented numerics.** Every balance law is implemented as a residual function of the state vector. The classical nested iteration and a global Newton scheme are both drivers over the same residual assembly (§6). This is what makes robust on/off-design operation and gradient availability possible.
5. **C¹-smooth closures.** Every closure model must be at least C¹-continuous over its full input domain, including outside its calibrated validity range, where it must saturate smoothly rather than extrapolate or discontinuously clamp (§7.3).

Working fluid: thermally and calorically perfect gas at first implementation, accessed exclusively through a fluid-property interface (§4.6) so that real-gas backends can be substituted without touching the kernel.

Assumptions inherited by the entire formulation: steady, axisymmetric (circumferentially averaged), adiabatic flow; viscous effects represented through entropy source terms, empirical blockage, and an optional spanwise mixing model rather than resolved stresses.

---

## 2. Coordinate Systems, Geometry, and Sign Conventions

### 2.1 Meridional frame

Cylindrical coordinates $(z, r, \theta)$ with $z$ the machine axis, positive in the nominal through-flow direction. Rotation rate $\Omega$ is positive per the right-hand rule about $+z$; blade speed $U = \Omega r$.

The circumferentially averaged flow is described in the meridional plane $(z, r)$. The **meridional velocity** is
$$V_m = \sqrt{V_z^2 + V_r^2},$$
and the absolute velocity is $V^2 = V_m^2 + V_\theta^2$. Relative (rotor-frame) tangential velocity: $W_\theta = V_\theta - U$; $W_m = V_m$.

### 2.2 Streamline-intrinsic directions

Along each meridional streamline, $m$ denotes arc length, positive downstream. The **streamline slope** $\phi$ is the angle from the $+z$ axis:
$$\sin\phi = \frac{dr}{dm}, \qquad \cos\phi = \frac{dz}{dm}.$$
For a purely axial path $\phi = 0$; for a purely radial (centrifugal exit) path $\phi = \pm 90^\circ$. The streamline unit tangent and normal in the meridional plane are
$$\mathbf{e}_m = \cos\phi\,\mathbf{e}_z + \sin\phi\,\mathbf{e}_r, \qquad \mathbf{e}_n = -\sin\phi\,\mathbf{e}_z + \cos\phi\,\mathbf{e}_r,$$
i.e. $\mathbf{e}_n$ is $\mathbf{e}_m$ rotated $+90^\circ$ (from hub side toward shroud side for a conventional annulus).

The **meridional streamline curvature** is
$$\kappa_m = \frac{d\phi}{dm} = \frac{1}{r_c},$$
positive when $\phi$ increases along the flow (streamline curving toward $+\mathbf{e}_n$). With these conventions the intrinsic acceleration is $\mathbf{a}_{mer} = V_m \frac{\partial V_m}{\partial m}\mathbf{e}_m + V_m^2 \kappa_m\,\mathbf{e}_n$.

### 2.3 Quasi-orthogonals

A **quasi-orthogonal (q-o)** is a curve in the meridional plane, fixed in space, spanning hub to shroud, along which the momentum and continuity equations are enforced. Arc length along a q-o is $q$, measured from the hub, with unit vector $\mathbf{e}_q$ pointing hub → shroud. Q-o's need not be normal to the streamlines nor straight; they are defined by the flow-path/station geometry (§2.5) and do not move during iteration (streamlines move; q-o's do not).

Define the **q-o lean angle relative to the streamline normal**, $\varepsilon$, by
$$\mathbf{e}_q = \cos\varepsilon\,\mathbf{e}_n + \sin\varepsilon\,\mathbf{e}_m,$$
so $\varepsilon = 0$ when the q-o is locally orthogonal to the streamline, and $\varepsilon > 0$ when the q-o leans downstream at that point. $\varepsilon$ varies along a q-o and changes as streamlines move. Useful geometric identities:
$$\frac{\partial r}{\partial q} = \cos(\varepsilon - \phi), \qquad \frac{\partial z}{\partial q} = \sin(\varepsilon - \phi).$$
The velocity component normal to the q-o (the mass-flux component) is $V_m \cos\varepsilon$.

### 2.4 Flow angles

Absolute swirl angle $\alpha$ and relative flow angle $\beta$ are defined from the meridional direction:
$$\tan\alpha = \frac{V_\theta}{V_m}, \qquad \tan\beta = \frac{W_\theta}{V_m}.$$
Sign convention: positive in the direction of rotor rotation ($+\theta$). Blade metal angles use the same convention. This single convention is used for compressors and turbines alike; correlation implementations are responsible for any internal re-mapping their source papers assume **[VERIFY per correlation]**.

### 2.5 Stations and computational grid

The meridional grid consists of $N_{qo}$ q-o stations $j = 1..N_{qo}$ and $N_{sl}$ streamlines $i = 1..N_{sl}$ (including hub $i=1$ and shroud $i=N_{sl}$ as bounding streamlines when endwall blockage is modeled via displacement, or as the annulus walls themselves otherwise). Node $(i,j)$ is the intersection of streamline $i$ with q-o $j$.

Stations are typed:

- **DUCT** — bladeless annulus station.
- **EDGE_LE / EDGE_TE** — blade row leading/trailing edge stations.
- **INBLADE** — station between LE and TE of a row (optional per row; required for long-passage radial/mixed rows and for in-passage choking, §4.5, §6.6).

Every blade row must have EDGE_LE and EDGE_TE stations; INBLADE stations are a per-row modeling choice. Streamlines are identified with normalized stream-function (mass-fraction) values $\psi_i \in [0,1]$, fixed for the run; the solver's geometric unknowns are the nodal radii/positions at which those mass fractions occur.

---

## 3. Governing Equations

### 3.1 Master momentum equation along a quasi-orthogonal

Projecting the circumferentially averaged, steady, inviscid momentum equation onto $\mathbf{e}_q$ and eliminating the pressure gradient via the Gibbs relation $dh = T\,ds + dp/\rho$ with $h = h_0 - \tfrac{1}{2}(V_m^2 + V_\theta^2)$ yields the **master equation** solved on every q-o:

$$\boxed{\;V_m \frac{\partial V_m}{\partial q} \;=\; \frac{\partial h_0}{\partial q} \;-\; T\,\frac{\partial s}{\partial q} \;-\; \frac{V_\theta}{r}\frac{\partial (r V_\theta)}{\partial q} \;+\; \underbrace{V_m^2\,\kappa_m \cos\varepsilon}_{\text{curvature}} \;+\; \underbrace{V_m \frac{\partial V_m}{\partial m}\,\sin\varepsilon}_{\text{q-o lean}} \;-\; f_{b,q}\;}$$

Term-by-term:

| Term | Physics | Active in |
|---|---|---|
| $\partial h_0/\partial q$ | Spanwise work gradient (forced vortex effects, non-uniform inlet) | All tiers ≥ 2 |
| $-T\,\partial s/\partial q$ | Spanwise loss gradient (endwall/secondary loss stratification) | All tiers ≥ 2 |
| $-(V_\theta/r)\,\partial(rV_\theta)/\partial q$ | Swirl / simple radial equilibrium | All tiers ≥ 2 |
| $V_m^2 \kappa_m \cos\varepsilon$ | Streamline curvature | Tier 3 only |
| $V_m (\partial V_m/\partial m) \sin\varepsilon$ | Meridional acceleration on leaned q-o | Tier 3 only |
| $-f_{b,q}$ | Mean blade force per unit mass projected on $\mathbf{e}_q$ (lean/sweep effects), INBLADE stations only; sign derived in Appendix A.5 | Tier 3 w/ in-blade stations |

Full derivation, sign conventions, special-case checks, and the q-o orientation rule are given in Appendix A (normative). The mapping to the Aungier/Novak angle parameterization is the identity $\varepsilon = \phi + \gamma$ (A.6). **[VERIFY: term-by-term transcription of Novak (1967) and Aungier's working equations into $(\phi,\varepsilon)$ form per A.6 — the one remaining external check on this equation.]**

Given distributions of $h_0(q)$, $s(q)$, $rV_\theta(q)$ (from the transport relations, §3.3–3.4) and geometry-derived $\kappa_m(q)$, $\phi(q)$, $\varepsilon(q)$, the master equation is a first-order nonlinear ODE for $V_m(q)$, integrated hub → shroud from the boundary value $V_{m,1} = V_{m,\mathrm{hub}}$ (§5.3). The hub value is closed by continuity (§3.2).

### 3.2 Continuity on a quasi-orthogonal

$$\dot m = 2\pi \int_{q_{\mathrm{hub}}}^{q_{\mathrm{shroud}}} \rho\, V_m \cos\varepsilon\, \big(1 - B\big)\, r \, dq,$$

where $B(q) \in [0,1)$ is the total aerodynamic blockage factor: the sum of endwall boundary-layer blockage $B_{ew}$ (from an annulus BL model or prescribed schedule, §7.2) and, at INBLADE stations, tangential blade-metal blockage
$$B_{blade} = \frac{Z\, t_\theta}{2\pi r},$$
with $Z$ blade count and $t_\theta$ the local tangential blade thickness. Between adjacent streamlines, the streamtube mass fractions are fixed:
$$\int_{q_i}^{q_{i+1}} \rho\, V_m \cos\varepsilon\,(1-B)\, r\, dq = (\psi_{i+1} - \psi_i)\,\frac{\dot m}{2\pi}.$$
This relation defines the streamline positions and is the basis of the streamline-repositioning residual (§6.2).

### 3.3 Energy and work: streamwise transport

Along each streamline (i.e., from station $j$ to $j{+}1$ at fixed $\psi_i$), for adiabatic flow:

- **Stationary rows and ducts:** $h_0 = \text{const}$.
- **Rotor rows:** rothalpy is conserved,
$$I \equiv h_0 - \Omega\, r V_\theta = \text{const},$$
equivalently the Euler work equation $\Delta h_0 = \Omega\, \Delta(r V_\theta)$.

Rothalpy conservation holds per-streamtube including losses, provided the flow is adiabatic and windage/shear work at the casing is neglected (note this assumption explicitly; a windage source term can be added to $h_0$ later for centrifugal backface friction **[VERIFY need per configuration]**).

### 3.4 Swirl transport

- **Ducts:** angular momentum conserved per streamtube, $rV_\theta = \text{const}$ (optionally modified by the spanwise mixing operator, §3.6).
- **Blade rows:** exit $rV_\theta$ is set by the swirl closure: exit relative flow angle $\beta_2 = \beta_{2,blade} + \delta$ with deviation $\delta$ from correlation (axial-style rows), or slip factor $\sigma$ (radial impellers), unified behind one interface returning exit $rV_\theta$ given local geometry and flow (§7.1).
- **INBLADE stations:** $rV_\theta(m)$ interpolated between LE and TE values by a prescribed work-distribution schedule (default: smooth monotone ramp in $m$; expose as per-row model input). The schedule must be C¹ in $m$.

### 3.5 Entropy transport

$$s_{j+1,i} = s_{j,i} + \Delta s_{row}(i) + \Delta s_{mix}(i),$$
where $\Delta s_{row}$ comes from the loss closure (converted per §4.4 and distributed along INBLADE stations by the same type of schedule as work) and $\Delta s_{mix}$ from the spanwise mixing model. Duct wall friction may add a small $\Delta s_{duct}$ (optional model).

### 3.6 Spanwise mixing

Multistage axial machines develop unrealistic spanwise stratification of $h_0$, $s$, and $rV_\theta$ without a mixing model. Adopt the diffusive form (Gallimore–Cumpsty; alternatively Adkins–Smith spanwise-velocity form): after each row's transport step, apply
$$\frac{\partial \chi}{\partial m} = \frac{1}{r(1-B)\rho V_m}\frac{\partial}{\partial q}\!\left( \mu_{mix}\, r \frac{\partial \chi}{\partial q} \right), \qquad \chi \in \{h_0,\, s,\, rV_\theta\},$$
with empirical mixing coefficient $\mu_{mix}$ (Gallimore's correlation as default **[VERIFY form and calibration constants]**). Discretize implicitly along $q$ per marching step for unconditional stability. Off by default in Tiers 1–2.

### 3.7 State relations

All thermodynamic evaluations go through the `WorkingFluid` interface: $\rho(h, s)$, $T(h, s)$, $a(h,s)$, $p(h,s)$, plus inverse forms as needed. Static enthalpy from $h = h_0 - \tfrac{1}{2}V^2$ (absolute frame) or $h = I - \tfrac{1}{2}W^2 + \tfrac{1}{2}U^2$ (relative frame). Perfect-gas backend first; interface signatures must not assume perfect-gas shortcuts (no bare $\gamma$, $c_p$ leakage into kernel code).

---

## 4. Blade Row Modeling

### 4.1 Row data contract

Each `BladeRow` provides, as functions of span fraction (and meridional fraction for INBLADE evaluation): metal angles at LE/TE, chord, stagger, solidity/pitch, thickness distribution (max thickness, LE/TE thickness, tangential thickness for blockage), throat width $o$ (or a model to estimate it), lean/sweep of the mean stream surface, tip clearance, and blade count $Z$. Correlations consume this contract only — never raw CAD.

### 4.2 Work input

Rotor work enters exclusively through the swirl closure and rothalpy conservation (§3.3–3.4). There is no independent "work coefficient" input in SLC mode; in Tier-1 meanline mode the same closure evaluated on the mean streamline plays that role, keeping fidelity levels consistent.

### 4.3 Incidence and off-design

At EDGE_LE, incidence $\imath = \beta_1 - \beta_{1,blade}$ (relative frame for rotors, absolute for stators) is the primary off-design input to loss and deviation closures. Reference (minimum-loss) incidence comes from the closure set (e.g., Lieblein for axial compressors; Ainley–Mathieson-style for turbines). All incidence-dependent behavior must satisfy the C¹ saturation rule (§7.3) so that deep off-design evaluation degrades gracefully.

### 4.4 Loss accounting and conversion to entropy

Internal representation: $\Delta s$ per streamtube per row. Conversion from common correlation outputs, evaluated with the loss charged at row exit conditions:

- Relative total-pressure loss coefficient (compressor convention) $\omega = \dfrac{p_{0r,1,ideal} - p_{0r,2}}{p_{0r,1} - p_1}$: compute $p_{0r,2}$, then
$$\Delta s = -R \ln\!\frac{p_{0r,2}}{p_{0r,2,ideal}}\Big|_{T_{0r,2}},$$
where $p_{0r,2,ideal}$ is the loss-free relative total pressure at the same rothalpy and exit radius (accounts for radius change in rotors — important for radial machines).
- Turbine kinetic-energy or enthalpy loss coefficients ($\zeta$, $\xi$): convert via exit static state to the equivalent $p_{0}$ deficit, then to $\Delta s$ as above.
- Perfect-gas fast path may be used inside the conversion, but the converted $\Delta s$ is what crosses the interface. **[VERIFY each correlation's exact coefficient definition against its source paper — this is the single most common implementation bug class in throughflow codes.]**

Loss decomposition (profile, secondary/endwall, tip clearance, shock, trailing-edge, windage/disk friction for centrifugal) is preserved in the output for design diagnostics, but summed for transport.

### 4.5 Throat, unique incidence, and in-passage choking

For each row with INBLADE stations (or optionally via a 1-D throat check for edge-only rows), compute the throat mass-flow capacity per streamtube from $o$, blockage, and the relative stagnation state. Supersonic-inlet axial rows follow a unique-incidence model **[VERIFY model choice — e.g., Wennerstrom/Cumpsty discussion]**. The row reports a smooth choke-margin scalar
$$c_{row} = 1 - \frac{\dot m_{tube}}{\dot m_{tube,choke}},$$
used by the operability logic (§6.6) — never a hard error.

### 4.6 Fluid interface

`WorkingFluid` exposes: `h(T,p)`, `s(T,p)`, `rho(h,s)`, `T(h,s)`, `p(h,s)`, `a(h,s)`, `h0_from_T0p0`, inverse `T0p0_from_h0s`, and reference-state metadata. All functions vectorized over arrays. Perfect-gas backend is analytic; future real-gas backend (e.g., CoolProp-backed with tabulation) must be drop-in.

---

## 5. Discretization

### 5.1 Streamline geometry representation

Each streamline $i$ is represented by its nodal meridional coordinates $\{(z_{ij}, r_{ij})\}_j$. Slope $\phi$ and curvature $\kappa_m$ are obtained from a smoothing spline fit of $r$ vs. $z$ (or parametric $(z(m), r(m))$ for paths passing through $\phi = 90^\circ$ — mandatory for radial machines; the parametric form is therefore the default). Requirements:

- C² continuity of the fit (curvature needs second derivatives).
- Light smoothing to suppress node-level noise; smoothing weight is a solver setting with a documented default.
- End conditions: natural or geometry-informed (inlet/exit duct slope) — record choice; it visibly affects first/last-station curvature.
- **[VERIFY]** Aungier recommends specific spline end-point and smoothing treatments; compare against standard smoothing splines during validation.

### 5.2 Q-o geometry

Q-o's are straight segments or user-defined curves fixed in space (§2.3). At each node, $\varepsilon$ is computed from the current streamline tangent and the q-o tangent. $\partial V_m/\partial m$ at node $(i,j)$ is evaluated by C¹ finite differences along the streamline through neighboring stations (non-uniform spacing formula), using the current iterate.

### 5.3 Integration of the master equation along a q-o

Between adjacent streamlines, integrate the master ODE with a second-order scheme (trapezoidal in $q$ with terms evaluated at mid-height, or RK2). All right-hand-side distributions ($h_0$, $s$, $rV_\theta$, $\kappa_m$, $\phi$, $\varepsilon$) are interpolated in $q$ from nodal values (monotone cubic — PCHIP-style — to avoid overshoot near endwalls). The result is $V_m(q; V_{m,hub})$: a one-parameter family.

### 5.4 Continuity integral

Evaluate §3.2 with the same quadrature nodes (composite trapezoid or Simpson on the streamline partition; must be the *same* rule used in the repositioning residual for consistency). Define
$$\mathcal{F}_j(V_{m,hub}) = 2\pi\!\int \rho V_m \cos\varepsilon (1-B) r\, dq - \dot m.$$
$\mathcal{F}_j$ is the q-o continuity residual. Its derivative $\partial \mathcal{F}_j / \partial V_{m,hub}$ changes sign at q-o choke (§6.6).

### 5.5 Curvature noise control

Curvature is the noise amplifier of SLC. Mitigations, all on by default in Tier 3: smoothing-spline geometry fit (§5.1); under-relaxed streamline movement with the aspect-ratio-scaled factor (§6.4); optional curvature under-relaxation (blend new/old $\kappa_m$ fields). Diagnostics must expose per-iteration curvature change norms.

---

## 6. Solution Algorithm

### 6.1 State, parameters, residuals

**State vector** $\mathbf{x}$ (Tier 3): hub meridional velocities $\{V_{m,hub,j}\}$, interior streamline positions $\{q_{ij}\}$ ($i = 2..N_{sl}-1$; hub/shroud fixed to walls or to endwall-displacement surfaces). Transported fields ($h_0, s, rV_\theta$) and closure outputs are *functions of the state* updated by lagged (Picard) sweeps or included in the Newton system (implementation choice per driver; start lagged).

**Residual vector** $\mathbf{R}(\mathbf{x})$:
- $R^{cont}_j$: q-o continuity (§5.4) — or its BC-switched replacement (§6.6).
- $R^{pos}_{ij}$: streamtube mass-fraction error, $\int_{hub}^{q_{ij}} \rho V_m \cos\varepsilon (1-B) r\, dq - \psi_i \dot m / 2\pi$.

The master-equation integration (§5.3) is treated as an *elimination*: given $V_{m,hub,j}$ and streamline geometry, it produces $V_m(q)$ on the q-o, so momentum is satisfied by construction and does not appear as a separate residual. (Alternative all-nodal formulation with explicit momentum residuals is reserved as a fallback; record decision.)

### 6.2 Classical nested driver (default far from operability limits)

1. Initialize: streamlines by area/mass-fraction interpolation of the annulus; $V_m$ from 1-D continuity; sweep transport relations once.
2. **Outer loop** until converged:
   1. Geometry pass: spline streamlines; update $\phi, \kappa_m, \varepsilon$ (with damping per §6.4).
   2. Station march $j = 1..N_{qo}$: update transported fields into station $j$ (§3.3–3.6, closures lagged from previous outer iterate); solve $\mathcal{F}_j(V_{m,hub,j}) = 0$ by safeguarded Newton/Brent, each evaluation integrating the master ODE (§5.3).
   3. Reposition streamlines from cumulative mass-flow inversion; apply relaxation factor $\omega_{sl}$ (§6.4).
   4. Update closures (losses, deviation, blockage, mixing) from the new flow field, under-relaxed.
   5. Convergence: $\|\Delta q_{ij}\|_\infty / \Delta q_{ref} < tol_{pos}$, $\max_j |\mathcal{F}_j| / \dot m < tol_{cont}$, and closure-update norms below tolerance. Report all three.

### 6.3 Global Newton driver (default near choke/stall, and for Jacobian export)

Assemble $\mathbf{R}(\mathbf{x})$ exactly as above but solve simultaneously: Newton–Krylov (or dense Newton for typical problem sizes, $N_{sl} \times N_{qo} \lesssim 10^3$ unknowns) with finite-difference or AD Jacobian, line search/trust region globalization, initialized from a classical-driver iterate or a neighboring converged operating point. Closures may remain lagged (quasi-Newton outer) initially; full inclusion is an upgrade path. The residual assembly must therefore be a pure function of $(\mathbf{x}, \text{parameters})$ with no hidden state — this is a hard architectural requirement.

### 6.4 Stability: streamline relaxation (Wilkinson criterion)

Streamline repositioning is unstable at large relaxation factors because curvature feeds back through second derivatives. Wilkinson's (1970) analysis gives the scaling
$$\omega_{sl} \;\lesssim\; C\,\big(1 - M_m^2\big)\left(\frac{\Delta m}{\Delta q}\right)^{2} \quad (\text{binding when } \Delta m < \Delta q),$$
for his discretization. **Calibrated result for this implementation (M3-3, Appendix C.3):** the measured envelope of the dominant instability — a streamwise odd-even mode of the per-q-o continuity solutions, amplified into curvature noise by the streamline fit — depends on *station density alone* (thresholds identical for $N_{sl} = 5, 9, 17$ at fixed stations) and follows
$$\omega_{sl} \;\le\; K\,\big(1 - M_m^2\big)\left(\frac{\Delta m_{min}}{L_{qo}}\right)^{3/2}, \qquad K_{threshold} \approx 7.3,\; K_{default} = 4.4\;(0.6\times\text{margin}),$$
with the §5.5 curvature lag at 0.3 (without which the mode diverges at *any* $\omega_{sl}$ on station-dense curved paths — the lag is mandatory, not optional, whenever the curvature term is active). The Wilkinson aspect form is deliberately not used: it over-throttles coarse spanwise grids and, uncapped, licenses divergent factors on fine ones (both measured). Implement $\omega_{sl}$ per-iteration from this envelope, capped by a user maximum; recalibrate with `tools/calibrate_wilkinson.py` after any change to repositioning or curvature-lag machinery. The exponent rests on one geometry family — revisit if V5/V7-class cases misbehave.

### 6.5 Transonic branch selection

The continuity solve on a q-o admits subsonic- and supersonic-$M_m$ branches. The solver always selects the subsonic meridional branch (standard throughflow practice; supersonic *relative blade-to-blade* Mach numbers are fine and appear only in closures). Branch selection is enforced by bracketing $V_{m,hub}$ below the value maximizing $\mathcal{F}_j + \dot m$.

### 6.6 Choking and boundary-condition switching

Define per-q-o capacity $\dot m_{max,j} = \max_{V_{m,hub}} (\mathcal{F}_j + \dot m)$ and margin $c_j = 1 - \dot m / \dot m_{max,j}$; combine with row throat margins $c_{row}$ (§4.5) into a global choke margin $c = \min(\cdot)$.

- **Normal mode** ($c$ above threshold $c_{sw}$): mass flow $\dot m$ is specified; exit pressure is an output.
- **Choke-proximal mode** ($c < c_{sw}$, or continuity solve loses a root): switch the operating-point specification to exit static pressure (or equivalently a hub-velocity level at the throttling station); $\dot m$ becomes part of the state with an added residual matching the specified back-pressure. The switch must be automatic, logged, and hysteretic (switch back only when $c$ recovers past $c_{sw} + \delta_{hys}$) to prevent limit cycling.

$\dot m_{max}$ evaluation reuses the same continuity machinery ($\partial \mathcal{F}/\partial V_{m,hub} = 0$ point), so choke detection costs one extra scalar solve per flagged q-o. Note the modeling limit: edge-only rows can only exhibit *annulus* choking; in-passage choking requires INBLADE stations + throat model.

### 6.7 Off-design maps and continuation

Speedlines are traversed by natural-parameter continuation: order points from choke toward stall; initialize each from the previous converged solution; adaptive step in $\dot m$ (or back-pressure in choke-proximal mode) with cut-back on convergence failure and driver escalation (classical → Newton) before step rejection. Numerical stall/surge is *reported, not solved through*: flag when (a) the Newton driver fails after step cut-backs, (b) a monotonicity criterion such as $\partial(PR)/\partial \dot m \ge 0$ is met, or (c) closure validity flags (§7.3) saturate over a large span fraction. Record the criterion that fired — surge margin definitions must be traceable.

---

## 7. Closure Model Interfaces

### 7.1 Interfaces (strategy pattern)

- `LossModel.evaluate(row, span_station_flow, geometry) -> LossBreakdown` (components + total, plus validity metadata).
- `SwirlClosure.exit_rVt(row, inlet_flow, geometry) -> rVθ` (deviation- or slip-based internally).
- `BlockageModel` (endwall BL / prescribed schedule) and `MixingModel` (§3.6).
- `CorrelationSet` bundles a consistent family per machine type — e.g., axial compressor: Lieblein incidence/deviation + Koch–Smith or Aungier losses; axial turbine: Kacker–Okapuu or Craig–Cox; centrifugal: Aungier or Galvas set + Wiesner/ von Backström slip. Sets are user-selectable per row; mixing sets across rows of one machine is allowed but warned.

### 7.2 Inputs contract

Closures receive only: the row data contract (§4.1), local circumferentially averaged flow at LE (and TE where iterative), fluid interface, and rotation. They must not reach into solver internals. This is what keeps the kernel configuration-agnostic.

### 7.3 Smoothness and validity rules (mandatory)

1. C¹ continuity everywhere, including across correlation-regime boundaries (blend with smooth switch functions over documented widths, not `if` ladders).
2. Outside calibrated range: smooth saturation toward conservative asymptotes (losses level off or grow smoothly; deviation saturates), never raw extrapolation, never hard clamps (C⁰ kinks), never exceptions.
3. Every evaluation returns a validity measure $v \in [0,1]$ (1 = inside calibration). The solver aggregates $v$ for operability logic (§6.7) and reporting.
4. Unit tests per correlation must sweep the full input domain and assert finiteness, continuity (numerical derivative bounded), and published-figure reproduction at calibration points.

---

## 8. Multi-Fidelity Degeneration Matrix

One kernel; tiers are configurations. "REE" = radial-equilibrium terms (first three RHS terms of the master equation).

| Feature | Tier 1: Meanline | Tier 2: Streamline-REE | Tier 3: Full SLC |
|---|---|---|---|
| $N_{sl}$ | 1 (mean $\psi=0.5$) + wall geometry | 5–11 typical | 11–21+ typical |
| Stations | EDGE only | EDGE only | EDGE + optional INBLADE |
| Master eq. terms | none (trivial) | REE terms only ($\kappa_m, \varepsilon$ terms off) | all |
| Streamline repositioning | off (mean line from area rule) | on | on |
| Spanwise mixing | n/a | optional | on (multistage axial) |
| Closure evaluation | at mean line | per streamline | per streamline |
| Choke detection | 1-D capacity + row throats | q-o capacity + row throats | full (§6.6) |
| Typical use | cycle coupling, MDO outer loops | early spanwise design | detailed preliminary design, map generation |

Consistency requirement: for a free-vortex, uniform-inlet case, Tier 2 and Tier 3 must agree to discretization tolerance, and Tier 1 must equal their mass-averaged result to closure-evaluation error. This is a standing regression test.

---

## 9. Verification & Validation Ladder

1. **V1 — Analytic REE:** incompressible and compressible free-vortex ($rV_\theta = $ const) and forced-vortex swirling flow in an annulus; exact $V_m(r)$ solutions. Verifies master-equation integration, continuity, repositioning. Grid-convergence order check (expect 2nd).
2. **V2 — Curvature:** swirl-free flow through a curved annulus (e.g., circular-arc hub/shroud); compare against a potential-flow/CFD reference; verifies $\kappa_m$, $\phi$, $\varepsilon$ machinery and damping stability envelope (calibrate §6.4 constant here).
3. **V3 — Tier consistency:** §8 consistency requirement, automated.
4. **V4 — Loss/deviation units:** reproduce published correlation figures point-by-point (per §7.3.4).
5. **V5 — Axial compressor:** NASA multistage/fan cases with published throughflow or test data (e.g., NASA two-stage fan; rotor 67 meanline-level checks); speedline generation incl. choke-side behavior.
6. **V6 — Axial turbine:** Kacker–Okapuu validation set / published stage maps.
7. **V7 — Centrifugal:** Eckardt impellers (O/A/B) for impeller exit profiles and stage maps; verifies $\phi \to 90^\circ$ parametric geometry path, in-blade stations, slip.
8. **V8 — Mixed-flow case** as available in the library.
9. **V9 — Operability:** demonstrate stable BC-switching across choke on V5/V7 cases; document surge-flag behavior vs. reported surge lines.

Each V-case ships as an executable regression test with tolerances recorded in this manual's Appendix C.

---

## 10. Nomenclature (normative)

| Symbol | Meaning | Units |
|---|---|---|
| $m$ | meridional arc length along streamline | m |
| $q$ | arc length along quasi-orthogonal, from hub | m |
| $\phi$ | streamline slope from axial | rad |
| $\varepsilon$ | q-o lean from streamline normal | rad |
| $\kappa_m$ | meridional streamline curvature $d\phi/dm$ | 1/m |
| $V_m, V_\theta$ | meridional, tangential absolute velocity | m/s |
| $W_\theta$ | relative tangential velocity $V_\theta - U$ | m/s |
| $\alpha, \beta$ | absolute, relative flow angle from meridional | rad |
| $h, h_0, I$ | static enthalpy, stagnation enthalpy, rothalpy | J/kg |
| $s$ | specific entropy | J/(kg·K) |
| $\psi$ | normalized mass fraction (stream function) | – |
| $B$ | blockage factor | – |
| $\omega$ | total-pressure loss coefficient | – |
| $\omega_{sl}$ | streamline relaxation factor | – |
| $\sigma$ | slip factor | – |
| $\delta$ | deviation angle | rad |
| $\imath$ | incidence angle | rad |
| $c, c_{row}$ | choke margins | – |
| $\Omega, U$ | shaft speed, blade speed $\Omega r$ | rad/s, m/s |

Angle convention: positive toward rotor rotation. SI units throughout; angles stored in radians, displayed in degrees.

---

## 11. Primary References (map to library)

Novak (1967) — general q-o SLC formulation. Smith (1966) — radial equilibrium foundations. Wilkinson (1970) — SLC stability/relaxation. Denton (1978; 1993) — throughflow numerics; loss/entropy accounting. Cumpsty, *Compressor Aerodynamics*. Aungier, *Axial-Flow Compressors*, *Turbine Aerodynamics*, *Centrifugal Compressors* — unified SLC + closure sets. Lieblein; Koch & Smith — axial compressor closures. Kacker & Okapuu; Craig & Cox — axial turbine closures. Wiesner — slip. Gallimore & Cumpsty; Adkins & Smith — spanwise mixing. Eckardt — centrifugal validation. **Acquisition status, source URLs, and the `[VERIFY]`-tag → source map live in [`docs/references/README.md`](references/README.md).** As of 2026-07-08: SP-36 and Ainley-Mathieson R&M 2974 are acquired (public-domain, URLs verified); Kacker-Okapuu, Dunham-Came, Wiesner, Eckardt, Aungier, and Gallimore-Cumpsty are paywalled and pending a readable copy. No coefficient-level tag is discharged yet — the index records what each source can validate (note the Aungier-vs-SP-36 split for the compressor fits: SP-36 validates fit *outputs*, Aungier validates the *coefficients*).

---

## Appendix A — Full Derivation of the Master Equation

This appendix is normative for signs. Any discrepancy found against a library source must be resolved *here* (with the source's convention mapped to ours), never by ad-hoc sign flips in code.

### A.1 Kinematic preliminaries

With $\mathbf{e}_m = \cos\phi\,\mathbf{e}_z + \sin\phi\,\mathbf{e}_r$ and $\mathbf{e}_n = -\sin\phi\,\mathbf{e}_z + \cos\phi\,\mathbf{e}_r$ (a fixed $+90^\circ$ rotation of $\mathbf{e}_m$ in the $(z,r)$ plane), differentiation along the streamline gives the Frenet-type relations
$$\frac{d\mathbf{e}_m}{dm} = \kappa_m\,\mathbf{e}_n, \qquad \frac{d\mathbf{e}_n}{dm} = -\kappa_m\,\mathbf{e}_m, \qquad \kappa_m \equiv \frac{d\phi}{dm}.$$
For axisymmetric flow, the meridional convective operator is $V_z\,\partial_z + V_r\,\partial_r = V_m\,\partial_m$.

**A.1.1 Q-o orientation convention (normative).** $\mathbf{e}_n$ is *always* the $+90^\circ$ rotation of $\mathbf{e}_m$; it is not defined as "hub-to-shroud." The q-o coordinate $q$ is oriented such that $\mathbf{e}_q\cdot\mathbf{e}_n \ge 0$, i.e. $\varepsilon \in (-90^\circ, +90^\circ)$ and $\cos\varepsilon \ge 0$ everywhere. Which physical wall sits at $q=0$ therefore depends on the machine: hub for a conventional axial compressor, but potentially the shroud side for e.g. a radial-inflow turbine, where the sense of $\mathbf{e}_n$ relative to the walls reverses. Implementations must label walls physically (`wall_0`, `wall_1` mapped to hub/shroud per machine) and must not assume $q=0 \Leftrightarrow$ hub. This convention keeps $\cos\varepsilon \ge 0$, which the continuity flux (A.7) and branch-selection logic (§6.5) rely on. All statements in the main text reading "hub" for the $q=0$ boundary are to be read as "wall at $q=0$."

### A.2 Circumferentially averaged momentum

Steady, axisymmetric, circumferentially averaged inviscid momentum with mean blade force per unit mass $\mathbf{f}_b$ (nonzero only inside blade rows, where it represents the averaged pressure-difference force across the passage):
$$(\mathbf{V}\cdot\nabla)\mathbf{V} = -\frac{1}{\rho}\nabla \bar p + \mathbf{f}_b.$$
The acceleration components in cylindrical coordinates for axisymmetric flow:
$$a_z = V_m\frac{\partial V_z}{\partial m},\qquad a_r = V_m\frac{\partial V_r}{\partial m} - \frac{V_\theta^2}{r},\qquad a_\theta = \frac{V_m}{r}\frac{\partial (rV_\theta)}{\partial m}.$$
Writing the meridional velocity vector as $V_m \mathbf{e}_m$ and using A.1:
$$\mathbf{a}_{mer} = V_m\frac{\partial}{\partial m}\big(V_m\mathbf{e}_m\big) - \frac{V_\theta^2}{r}\,\mathbf{e}_r = V_m\frac{\partial V_m}{\partial m}\,\mathbf{e}_m + V_m^2\,\kappa_m\,\mathbf{e}_n - \frac{V_\theta^2}{r}\,\mathbf{e}_r.$$

### A.3 Projection onto the quasi-orthogonal

With $\mathbf{e}_q = \cos\varepsilon\,\mathbf{e}_n + \sin\varepsilon\,\mathbf{e}_m$ and $\mathbf{e}_r\cdot\mathbf{e}_q = \cos\varepsilon\cos\phi + \sin\varepsilon\sin\phi = \cos(\varepsilon-\phi)$:
$$a_q = V_m\frac{\partial V_m}{\partial m}\sin\varepsilon + V_m^2\kappa_m\cos\varepsilon - \frac{V_\theta^2}{r}\cos(\varepsilon-\phi),$$
and the q-projected momentum equation reads
$$\frac{1}{\rho}\frac{\partial \bar p}{\partial q} = f_{b,q} - a_q = f_{b,q} - V_m\frac{\partial V_m}{\partial m}\sin\varepsilon - V_m^2\kappa_m\cos\varepsilon + \frac{V_\theta^2}{r}\cos(\varepsilon-\phi). \tag{A.1}$$

### A.4 Elimination of the pressure gradient

Gibbs relation along $q$ with $h = h_0 - \tfrac12(V_m^2 + V_\theta^2)$:
$$\frac{1}{\rho}\frac{\partial \bar p}{\partial q} = \frac{\partial h}{\partial q} - T\frac{\partial s}{\partial q} = \frac{\partial h_0}{\partial q} - T\frac{\partial s}{\partial q} - V_m\frac{\partial V_m}{\partial q} - V_\theta\frac{\partial V_\theta}{\partial q}. \tag{A.2}$$

### A.5 Master equation

Equating (A.1) and (A.2), solving for $V_m\,\partial V_m/\partial q$, and combining the swirl terms using $\partial r/\partial q = \cos(\varepsilon-\phi)$:
$$V_\theta\frac{\partial V_\theta}{\partial q} + \frac{V_\theta^2}{r}\frac{\partial r}{\partial q} = \frac{V_\theta}{r}\frac{\partial (rV_\theta)}{\partial q},$$
gives
$$V_m \frac{\partial V_m}{\partial q} = \frac{\partial h_0}{\partial q} - T\frac{\partial s}{\partial q} - \frac{V_\theta}{r}\frac{\partial (r V_\theta)}{\partial q} + V_m^2\kappa_m\cos\varepsilon + V_m\frac{\partial V_m}{\partial m}\sin\varepsilon - f_{b,q},$$
which is the boxed equation of §3.1. Note the **minus** sign on $f_{b,q}$: it follows directly from $\mathbf{f}_b$ appearing with $+$ on the momentum RHS. Physically: a blade force pushing toward $+\mathbf{e}_q$ supports a pressure rise along $q$ without requiring the $V_m$ field to supply it. (v0.1 of this manual carried the term with the wrong sign; corrected in v0.2.)

Special-case checks:
1. **Simple radial equilibrium** ($\phi=0$, $\varepsilon=0$, $q=r$, ducts): $V_m\,dV_m/dr = dh_0/dr - T\,ds/dr - (V_\theta/r)\,d(rV_\theta)/dr$ — the classic non-isentropic REE. With uniform $h_0$, $s$ and free vortex $rV_\theta=$ const: $dV_m/dr = 0$, uniform axial velocity, as required.
2. **Static annulus, no swirl, curved walls** ($V_\theta=0$, $\varepsilon=0$): $V_m\,\partial V_m/\partial q = V_m^2\kappa_m$, i.e. $\partial(\ln V_m)/\partial q = \kappa_m$ — velocity increases toward the center of curvature side consistent with a free-vortex-like meridional turning. This is test case V2.
3. $\varepsilon \to 0$, $\kappa_m \to 0$, $N_{sl}$ arbitrary reproduces Tier 2 exactly; all Tier-3 exclusive terms carry a factor $\cos\varepsilon$, $\sin\varepsilon$, or $\kappa_m$ and vanish under the §8 switch flags without code branching.

### A.6 Mapping to library conventions

Let $\gamma$ be the angle of the q-o measured from the radial direction toward $+z$, i.e. $\mathbf{e}_q = \sin\gamma\,\mathbf{e}_z + \cos\gamma\,\mathbf{e}_r$. Expanding $\mathbf{e}_q$ in A.3's basis and comparing components gives the exact identity
$$\boxed{\;\gamma = \varepsilon - \phi \quad\Longleftrightarrow\quad \varepsilon = \phi + \gamma\;}$$
consistent with $\partial r/\partial q = \cos\gamma$ and $\partial z/\partial q = \sin\gamma$. Aungier and Novak parameterize the momentum equation using the q-o (quasi-normal) inclination and the streamline slope separately — i.e., effectively $(\phi, \gamma)$ — whereas this manual uses $(\phi, \varepsilon)$. When transcribing any equation from those sources, substitute $\varepsilon = \phi + \gamma$ (mind each source's own sign convention for $\gamma$ and for angle-from-axial vs. angle-from-radial). **[VERIFY: perform this transcription for Novak (1967) eq. set and Aungier's working equation and confirm term-by-term agreement; file the worked comparison in the library.]**

### A.7 Continuity flux and the q-o choke condition

The unit normal to the q-o (revolved into a surface) lying in the meridional plane and oriented downstream is $\hat{\mathbf{n}} = \cos\varepsilon\,\mathbf{e}_m - \sin\varepsilon\,\mathbf{e}_n$, whence the mass-flux velocity component is $\mathbf{V}\cdot\hat{\mathbf{n}} = V_m\cos\varepsilon \ge 0$ under convention A.1.1, giving §3.2.

During the inner continuity solve on a q-o, the per-streamline invariants are $h_0$, $s$, $rV_\theta$, and (streamlines frozen) $r$ — hence $V_\theta$ is frozen and $dh = -V_m\,dV_m$. With $ds=0$ during the scan, $d\rho = \rho\,dh/a^2$, so
$$\frac{\partial(\rho V_m)}{\partial V_m}\bigg|_{h_0,s,rV_\theta,r} = \rho\left(1 - M_m^2\right), \qquad M_m \equiv V_m/a.$$
The streamtube mass flux is maximized at **meridional** Mach unity (not total Mach unity — swirl is frozen through $rV_\theta$). Consequences: (i) the subsonic branch of §6.5 is $M_m < 1$ pointwise; (ii) the q-o capacity $\dot m_{max,j}$ of §6.6 is reached when the weighted spanwise distribution of $\rho(1-M_m^2)\,\partial V_m/\partial V_{m,0}$ integrates to zero, detected robustly via the sign of $\partial\mathcal{F}_j/\partial V_{m,0}$; and (iii) high swirl chokes a q-o at low $V_m$ (small $a$ from swirl kinetic energy), which is the correct physics for near-stall radial diffuser and turbine-exit stations.

### A.8 In-blade force model

The tangential component follows from the averaged tangential momentum equation (A.2, $\theta$-component):
$$f_{b,\theta} = \frac{V_m}{r}\frac{\partial (r V_\theta)}{\partial m},$$
i.e., it is fully determined by the prescribed in-blade $rV_\theta(m)$ schedule (§3.4) — no additional empiricism. The meridional-plane component is modeled by requiring the *inviscid* part of $\mathbf{f}_b$ to be normal to the mean blade stream surface $\theta = \theta_b(m, q)$. Defining the local lean angle $\lambda$ of that surface in the $(q,\theta)$ plane, $\tan\lambda = r\,\partial\theta_b/\partial q$, the standard first-order model is
$$f_{b,q} = f_{b,\theta}\,\tan\lambda .$$
Radially stacked blades ($\partial\theta_b/\partial q = 0$ with $q$ along the stacking line) give $f_{b,q}=0$; strongly leaned or swept rows (and backswept impeller exits evaluated off the stacking axis) do not. The dissipative part of the blade force acts along $-\mathbf{W}$ and is *not* added here — its effect is already carried by the distributed $\Delta s$ schedule (§3.5); adding both would double-count. **[VERIFY the $\tan\lambda$ model and the surface-normal construction against Aungier's and Denton's in-blade treatments; confirm no double-counting convention mismatch.]**

## Appendix B — Loss-Coefficient → Entropy Conversions

Perfect-gas working forms; the real-gas path replaces the closed forms with `WorkingFluid` inversions but keeps the *definitions* below. All conversions are charged at row-exit conditions (§4.4). Master relation for any coefficient reducible to a stagnation-pressure deficit at common stagnation temperature:
$$\Delta s = -R\,\ln\!\frac{p_{0,actual}}{p_{0,ideal}}\bigg|_{T_0\,\text{common}}.$$

### B.1 Ideal (loss-free) reference states

- **Stator / duct:** $T_{0,2,id} = T_{0,1}$, $p_{0,2,id} = p_{0,1}$.
- **Rotor (relative frame, radius change included):** from rothalpy conservation, $I = c_p T_{0r} - \tfrac12 U^2$, so
$$T_{0r,2} = T_{0r,1} + \frac{U_2^2 - U_1^2}{2 c_p}, \qquad p_{0r,2,id} = p_{0r,1}\left(\frac{T_{0r,2}}{T_{0r,1}}\right)^{\gamma/(\gamma-1)}.$$
The isentropic re-referencing across the radius change is what makes this correct for radial and mixed rotors; omitting it (valid only when $U_2 \approx U_1$) is a known axial-code habit that must not survive here.

### B.2 Compressor-style relative total-pressure loss coefficient

Given $\bar\omega = \dfrac{p_{0r,2,id} - p_{0r,2}}{p_{0r,1} - p_1}$ (confirm each source's exact reference dynamic head — inlet relative dynamic pressure shown here — **[VERIFY per correlation]**):
$$p_{0r,2} = p_{0r,2,id} - \bar\omega\,(p_{0r,1} - p_1), \qquad \Delta s = -R\ln\frac{p_{0r,2}}{p_{0r,2,id}}\bigg|_{T_{0r,2}}.$$
Stator case: same with absolute-frame quantities and B.1's stator reference.

### B.3 Turbine total-pressure loss coefficient (Ainley/Kacker–Okapuu style)

Given $Y = \dfrac{p_{0(r),1} - p_{0(r),2}}{p_{0(r),2} - p_2}$ (frame per row type):
$$p_{0r,2} = \frac{p_{0r,2,id} + Y\,p_2}{1 + Y}\quad\text{(rotor, with B.1 re-referencing)}, \qquad \Delta s = -R\ln\frac{p_{0r,2}}{p_{0r,2,id}}.$$
Note $Y$ references the *exit* dynamic head; the algebraic rearrangement above follows from that. Stator: replace relative by absolute quantities.

### B.4 Kinetic-energy / enthalpy loss coefficients (Craig–Cox style)

Given $\zeta = \dfrac{h_2 - h_{2s}}{\tfrac12 V_2^2}$ (or relative-frame $W_2$; per source), with $h_{2s}$ the isentropic exit static enthalpy at $p_2$:
$$T_{2s} = T_2 - \frac{\zeta V_2^2}{2 c_p}, \qquad \Delta s = c_p \ln\frac{T_2}{T_{2s}}$$
(entropy difference between the actual and isentropic states at the same $p_2$). Guard: $\zeta V_2^2/(2 c_p T_2) < 1$ must hold; the C¹ saturation rule (§7.3) applies to the coefficient before conversion, so the guard should never bind in a converged solve — assert, don't clamp.

### B.5 Distribution and bookkeeping rules

1. In-blade distribution of $\Delta s$ follows the same C¹ schedule class as the work distribution (§3.4–3.5), defaulting to the work schedule itself unless the loss source is explicitly local (e.g., tip-clearance loss weighted toward the tip streamtubes, shock loss toward supersonic-inlet span fractions).
2. Loss components are converted to $\Delta s$ *individually*, then summed — never sum pressure-loss coefficients with different reference dynamic heads.
3. Each conversion records $(\text{coefficient in},\ \Delta s\ \text{out},\ \text{reference state used})$ in the row diagnostics for auditability.

## Appendix C — Regression-test tolerances and reference data provenance

### C.1 V1 — Analytic REE (bound at M2; `tests/test_v1_analytic_ree.py`)

Reference solutions: closed-form Vm(r) families (A.5 case 1) + dense 20001-point 1-D continuity inversion, implemented in `slcflow/verification/v1_analytic_ree.py` independently of the kernel's nodal quadrature. Default annulus $r_0/r_1 = 0.3/0.6$ m, $h_0 = 3\times10^5$ J/kg, $s = 0$, perfect gas (air).

| Case | Grid | Check | Tolerance | Measured (2026-07, M2) |
|---|---|---|---|---|
| V1a free vortex, incompressible limit ($M_m \approx 0.02$) | $N_{sl}=9$ | spanwise Vm uniformity | rtol 1e-8 | ~6e-8 vs. reference |
| | | positions vs. closed-form area rule | 1e-3 span | (compressibility residue) |
| | | positions vs. dense reference | 1e-5 span | 3.2e-8 |
| V1b free vortex, compressible ($M_m \approx 0.43$) | $N_{sl}=9$ | hub Vm vs. reference | rtol 5e-5 | 1.2e-5 |
| | | positions | 2e-4 span | 5.1e-6 |
| | | full residual vector at answer | 1e-7 of $\dot m/2\pi$ | — |
| V1c forced vortex ($\Omega_f = 60$ s⁻¹) | $N_{sl}=17$ | Vm profile / hub level | rtol 2e-3 | 9.6e-5 |
| | | positions | 2e-3 span | 7.0e-6 |
| V1d grid convergence, V1c over $N_{sl} = 5, 9, 17$ | | observed order | > 1.7 | **1.94** (errors 1.40e-3 / 3.69e-4 / 9.55e-5) |

Note: V1b's hub-Vm tolerance is the $N_{sl}=9$ nodal-trapezoid discretization error, not reference precision — the reference quadrature is deliberately independent.

### C.2 V2 — Curved annulus, full Tier 3 (bound at M3-2; `tests/test_v2_curved_annulus.py`)

Reference: **planar-limit concentric-bend solution** (`slcflow/verification/v2_curved_annulus.py`) — meridional free vortex $V_m = A/R_{bend}$ with dense 1-D continuity positions, exact for bend-center machine radius $r_c \gg R_{bend}$; default $r_c = 400$ m with bend radii 0.2/0.5 m puts the $O(R/r_c)$ reference deviation below the comparison floor. The case is defined by $A$; $\dot m$ is derived from the reference. The frozen-streamline V2 gate (metrics + master-ODE only, coupled 2nd-order refinement) remains in `tests/test_grid.py` (M1).

| Check | Grid | Tolerance | Measured (2026-07, M3-2) |
|---|---|---|---|
| central-third Vm vs. reference | $N_{sl}=9$, 7 st. | rtol 2e-2 | 1.0e-2 |
| central-third positions | same | 6e-3 span | 2.6e-3 |
| full residual vector at answer | same | 1e-7 of $\dot m/2\pi$ | — |
| planar-limit family (central Vm err) | $r_c = 4/40/400$ | monotone, ≥4× total | 8.9e-2 / 1.4e-2 / 1.0e-2 |
| reference floor, grid-independence | (5,5) and (17,13) | ≤ 2e-2 | 8.2e-3 / 8.6e-3 |

Caveats (measured, M3-2): (i) the comparison window is the **central third** of the bend — spline end-condition error at the inflow/outflow stations contaminates a fixed *physical* length, so fixed station-count exclusions anti-converge under refinement; (ii) the residual ~1e-2 central disagreement is **grid- and $r_c$-independent** (tested to $r_c = 4000$): it is the boundary-development difference between the solved problem (flow boundaries at the bend ends) and the fully-developed reference vortex, not solver error. **[VERIFY: cross-check against an external potential-flow/CFD reference with straight inlet/exit duct extensions to close caveat (ii).]** Discretization-order evidence for the coupled machinery: M1 frozen-streamline gate (coupled refinement, order ≈ 2) + V1d (repositioning, order 1.94).

### C.3 §6.4 relaxation-envelope calibration (M3-3; `tools/calibrate_wilkinson.py`)

V2 curved-annulus case, Tier 3, κ-lag 0.3, fixed-ω sweep; ω\* = largest converged factor (`x` = divergence, `m` = >150 iterations without divergence, i.e. slow, stability-safe):

| Grid $(N_{sl}, N_{st})$ | $\Delta m_{min}/L_{qo}$ | ω\* measured | model $7.3\,x^{1.5}$ |
|---|---|---|---|
| (9, 5) | 0.262 | ≥ 0.70 (cap) | 0.98 |
| (9, 7) / (17, 7) | 0.175 | ≥ 0.70 (cap) | 0.53 |
| (9, 10) | 0.116 | 0.28 | 0.29 |
| (9, 13) / (5, 13) | 0.087 | 0.20 | 0.19 |
| (17, 13) | 0.087 | 0.14–0.20 (0.20 marginal) | 0.19 |
| (9, 19) | 0.058 | < 0.10 (0.10 diverges) | 0.10 |

Key facts: thresholds independent of $N_{sl}$ (5/9/17 identical at fixed stations); the fitted model passes within 2% of the measured-unstable (9,19) point, hence the mandatory 0.6× margin in the shipped default $K = 4.4$; without the §5.5 κ-lag the mode diverges at any ω (measured at 13 stations down to ω = 0.02). All measurements at peak $M_m \approx 0.3$; the $(1-M_m^2)$ factor is retained from theory, not independently calibrated. Rerun the tool after any repositioning/curvature-lag change.

**Multi-family recalibration (2026-07, post-stabilization; rerunnable via `tools/calibrate_wilkinson.py [duct|bladerow|all]`).**

*Duct rerun (the tool's own instruction after the repositioning changes):* the fit **reproduces exactly** — $p = 1.50$, $K = 7.3$, with every ω\* in the table above unchanged. Two near-threshold *classifications* shifted from "diverges" to "slow-stable within 150 iterations" ((9,19) at ω = 0.10; (17,13) at the previously-marginal 0.20): the positive-branch root validation now prevents the garbage-branch deaths that used to mark instability onset, while the mode itself (and hence the threshold and fit) is unchanged.

*Blade-row family* (V8 parametric bend, centrifugal set, Tier 3, $n_{sl} = 7$; sweeping `wilkinson_c` itself because fixed-ω sweeps die in the closure switch-on transient that the adaptive $(1-M_m^2)$ factor rides out in production; `n_inblade` barely moves $x = \Delta m_{min}/L_{qo}$ — the duct-adjacent station gaps pin it — so these points probe the threshold *constant*, not the exponent):

| point | $x$ | $c^*$ (largest converged) | iters at $c^*$ |
|---|---|---|---|
| $\phi=55°$, `n_inblade=2` | 0.077 | — (see open item) | — |
| $\phi=55°$, `n_inblade=6` | 0.077 | 13.2 (17.6 fails) | 152 |
| $\phi=55°$, `n_inblade=12` | 0.046 | — (see open item) | — |
| $\phi=90°$, `n_inblade=2` | 0.126 | ≥ 30 (scan cap) | 75 |
| $\phi=90°$, `n_inblade=6` | 0.126 | 13.2 (17.6 fails) | 98 |
| $\phi=90°$, `n_inblade=12` | 0.075 | 8.8 (13.2 fails) | 159 |

**Conclusions:** every converging blade-row point's threshold ($c^* \in [8.8, \ge 30]$) sits above the duct threshold (≈7.3), so **the duct family remains binding and the shipped default $K = 4.4$ stands** (0.6× of the binding threshold). Per-case `wilkinson_c` overrides remain safe to ~13 on `n_inblade=6`-class layouts (2–2.6× fewer iterations, identical answers). The two 55° non-converging points are **not envelope failures** (their adaptive ω is the *smallest* in the family). Initially diagnosed as a "freeze-fallback wedge", and the candidate fix — returning the feasible **capacity-peak** $V_{m,q0}$ for a deficient station instead of freezing it — was **implemented, measured, and reverted (2026-07)**: it is a no-op for every passing case (V7/V8/edge-only trajectories bit-identical) and non-curative here. The deeper measured picture: these layouts settle into **self-consistent lag states whose exit-duct station has no positive-branch continuity root at the target $\dot m$ at all**, in two mirror forms. At `n_inblade=2`, a *stationary deficit* — with the peak fallback the solve becomes perfectly stationary (position/closure norms $\sim10^{-15}$) at a constant 24.8% exit mass shortfall, and the deficit persists even with the ib=6 fixed point's closure outputs *prescribed* (exonerating closure coupling; the remaining suspect is the coarse 6-node streamline fit's end-condition curvature at the exit, which the Tier-3 capacity follows). At `n_inblade=12`, a *stationary surplus* — the lag-settled spanwise stratification ($h_0$ span 22 kJ/kg, $s$ span 3→15 J/(kg·K), both consistent with the spanwise closure outputs) forces an REE $V_m$ shear whose minimum feasible mass is ~28.8 kg/s > 12 at *any* boundary value, ending in a $V_m$-singularity rupture. Credible next attacks (recorded, not attempted): the global Newton with closure-in-Newton on such states (the lag structure is what sustains them), an end-condition-aware/compact-support streamline fit (also the route to Jacobian sparsity), or documenting `n_inblade≈6` layouts as the supported radial/mixed configuration. Until one lands, 55°-class bends with `n_inblade` far from 6 are a **known-unsupported region**, reported honestly by CHOKE_LIMITED/NUMERICAL_FAILURE statuses rather than wrong answers.

### C.4 V3 — Tier consistency (bound at M3-4 / M4-5; `tests/test_v3_tier_consistency.py`)

Section 8 requirement on the free-vortex (and forced-vortex) uniform-inlet straight annulus, $N_{sl} = 9$: Tier 2 vs. Tier 3 measured **bit-for-bit identical** (2026-07, M3-4) — the Tier-3-exclusive terms multiply exactly-zero $\kappa_m$/$\varepsilon$ and the relaxation paths coincide at the cap. Asserted at 1e-10 (far below discretization) so hidden tier branching fails loudly; a non-vacuousness guard checks the tiers *do* diverge on a curved path.

**Tier-1 mass-average clause (bound at M4-5).** With the $N_{sl}=1$ meanline now assembling — the one-point area rule $\dot m_j = 2\pi\,[\rho V_m \cos\varepsilon\,(1-B)]_{\psi=0.5}\int r\,dq$, the coarsest instance of the §5.4 quadrature, evaluated at the fixed area-rule mean line (repositioning off) — the meanline $V_m$ is required to equal the mass-flux-weighted ($\rho V_m \cos\varepsilon\,r$) span average of the Tier-2 field to closure-evaluation error. On the prescribed V1 cases the closures are trivial, so the residue is purely the meanline quadrature error.

| Case ($N_{sl}=1$ vs. $N_{sl}=9$) | Check | Tolerance | Measured (2026-07, M4-5) |
|---|---|---|---|
| V1b free vortex (uniform $V_m$) | meanline $V_m$ vs. mass-avg Tier 2 | rtol 1e-3 | 8.3e-4 |
| V1c forced vortex ($V_m(r)$ varies) | meanline $V_m$ vs. mass-avg Tier 2 | rtol 2e-3 | 1.34e-3 |

The forced-vortex residue is larger because its $V_m$ genuinely varies across span (∝ the family $V_m^2 = V_{m0}^2 - 2\Omega_f^2(r^2-r_0^2)$), so the one-point rule has a real profile to miss; the free vortex is spanwise-uniform and residue is the density/$r$-weighting curvature alone. Both are the meanline's own discretization, not solver error — the same $N_{sl}=1$ path is one assembler with no tier branch (AD-1), verified by `test_tier1_is_pure_data_switch_not_a_code_path` (V5).

### C.5m V5 multistage — mixing revisit (bound at M8-3; `tests/test_multistage_mixing.py`)

The M8 revisit of V5 in a **multistage** configuration (`V5MultistageCompressor`, two repeating rotor+stator pairs on a cylindrical annulus, matched Lieblein stage: rotor $\beta=(-48°,-30°)$, stator $(25°,-5°)$ de-swirling to near-axial; annulus retuned to hub/tip 0.73 so the loss runs in the Lieblein validity window). This is the configuration §3.6 exists for; the honest, in-window result is that at Gallimore–Cumpsty calibration mixing is a **modest** damping, not a homogenizer:

| Configuration ($N_{sl}=9$, Tier 3, validity 0.99) | Converged | PR (t-t) | exit $\Delta s$ span spread |
|---|---|---|---|
| **mixing off** | yes | 1.088 | 0.495 J/(kg·K) |
| **mixing on** (default Gallimore, $c_{mix}=5\times10^{-4}$) | yes | 1.088 | 0.377 J/(kg·K) |

(Re-measured 2026-07 with the Howell endwall + tip-clearance loss now in the axial-compressor set, Appendix HOWELL/§4.4: the added spanwise-varying endwall loss lowers PR ~1.11→1.09 and shifts the exit-entropy spreads, but the qualitative finding is unchanged and slightly firmer — mixing damps the spread by $\sim$24% here, still modest, still not a homogenizer.)

**Revised three times in 2026-07 — a cautionary entry.** M8-3 originally recorded the un-mixed two-stage as a NUMERICAL_FAILURE with a runaway ($\sim$40 J/(kg·K)) entropy split and concluded mixing was a *convergence prerequisite*; the Tier-3 stabilization (see C.8) showed that was the driver's stale-split / spurious-branch artifact, and the un-mixed case converges cleanly. The revision then recorded a *dramatic* difference ($\sim$25×, 17.6 vs. 0.69 J/(kg·K)) — the compound artifact of the Lieblein $\bar\omega$-inversion (~4× too much loss) and a $c_{mix}=0.01$ that was ($\sim$20×) stronger than G–C's calibration (GC86.md). With **both** fixed, and the case geometry retuned so the loss runs **in** the Lieblein validity window (the earlier configs ran at closure validity 0 — over-loaded untwisted blades driving $D_{eq}$ out of window, so the loss was saturated), the measured effect is the modest $\sim$24% above ($\sim$0.50 → 0.38 J/(kg·K); it was $\sim$18% before the Howell endwall loss was added, which contributes its own spanwise-varying entropy), and it does **not** catch up as the absolute spread grows with stage count. (The absolute spreads are loss-model-dependent — they moved when the Lieblein $\bar\omega$ and off-design bucket were corrected, and again when the endwall loss was added — but the modest-fraction ratio is stable; only the 2-stage row is regression-pinned, by ratio not absolute value.) The direction (mixing reduces the spread) is a guaranteed operator property; the *smallness* is the finding the regression pins. The "mixing flattens multistage stratification" narrative does not survive an honestly-calibrated coefficient on in-window loss. Bands are structural plausibility gates as for single-stage V5; $c_{mix}$ is G–C-calibrated (option B, r-based retune; GC86.md).

### C.6 V6 — Axial turbine, structural (bound at M6-5; `tests/test_v6_axial_turbine.py`)

**Structural gate only** (as V5): a pre-swirled axial-turbine rotor composed through the `Machine` facade with the Kacker–Okapuu set (`slcflow/verification/v6_axial_turbine.py`) — throat exit angle + profile/secondary/trailing-edge/shock loss — converges, **extracts** real work ($\Delta h_0<0$) with real loss ($\Delta s>0$), de-swirls toward an axial exit, and lands total-to-total expansion ratio and efficiency in physically sane bands. The bands are plausibility gates, **not** validation tolerances: point-by-point reproduction of a specific K–O validation case / published stage map, and speedline/choke traversal, are **[VERIFY]**, blocked on the reference-library correlation calibration (every K–O fit coefficient is `[VERIFY]`) — the same boundary as V5. Reference geometry is a representative reaction rotor ($r=0.35/0.50$ m, $\Omega=250$ s⁻¹, $\dot m=40$ kg/s, inlet $rV_\theta=30$, throat $o=0.030$ m), *not* a digitised K–O case.

| Quantity | Tier 1 ($N_{sl}=1$) | Tier 2 ($N_{sl}=9$) | Tier 3 ($N_{sl}=9$) | Band / check |
|---|---|---|---|---|
| PR (total-to-total) | 0.9323 | 0.9337 | 0.9337 | $(0.85, 0.99)$, $<1$ (expansion) |
| efficiency (facade, inverted; $\eta_t\approx1/\text{this}$) | 1.0467 | 1.0475 | 1.0475 | $(1.01, 1.15)$ |
| $\Delta h_0$ across rotor [J/kg] | −5679 | −5542 | −5543 | $<0$ (work out) |
| exit $rV_\theta$ (from inlet 30) | 7.28 | 7.83 | 7.83 | $< 0.5\times$ inlet (de-swirl) |
| $\Delta s$ across rotor [J/(kg·K)] | 0.905 | 0.905 | — | $>0$ (loss) |
| validity | 0.999 | 0.999 | 0.999 | $>0.5$ |

Tier 2 vs. Tier 3 agree to rtol 2e-3 (PR) / 1.5e-2 ($V_m$) — **not** the bit-identical V3 gate (C.4), because the throat-based exit swirl is spanwise-varying, so repositioning gives the streamlines slight meridional curvature and Tier 3's curvature term is small-but-nonzero. Tier 1 tracks the mass-averaged Tier 2 to rtol 5e-3 (the meanline clause, C.4).

### C.7 V7 — Centrifugal impeller, structural (bound at M7-4; `tests/test_v7_centrifugal.py`)

**Structural gate only** (as V5/V6), and the **first radial end-to-end**: a backswept centrifugal compressor impeller composed through the `Machine` facade with the centrifugal set (`slcflow/verification/v7_centrifugal.py`) — Wiesner slip + representative internal loss — on the parametric $\phi:0\to90°$ axial→radial path (M1), converging at all three tiers, **doing** real centrifugal work ($\Delta h_0>0$, PR $>1$) with real loss ($\Delta s>0$), spinning axial inflow ($rV_\theta=0$) up to a slipped exit swirl ($V_{\theta,2}<U_2$), and exiting radially ($r_{\text{exit}}=r_2$, i.e. $\phi=90°$). Reference geometry is a representative impeller (concentric 90°-bend walls, $r_2=0.25$ m, bend radii $0.08/0.18$ m, $\Omega=1450$ s⁻¹ → $U_2=362$ m/s, $\dot m=12$ kg/s, 18 blades, 30° backsweep), **not** a digitised Eckardt rotor.

| Quantity | Tier 1 ($N_{sl}=1$) | Tier 2 ($N_{sl}=7$) | Tier 3 ($N_{sl}=7$) | Band / check |
|---|---|---|---|---|
| PR (total-to-total) | 2.458 | 2.433 | 2.454 | $(1.5, 4.0)$, $>1$ (compression) |
| efficiency | 0.985 | 0.974 | 0.980 | $(0.6, 0.999)$ |
| exit $rV_\theta$ (from inlet 0) | 61.6 | — | — | $>0$ (spun up) |
| exit $V_\theta / U_2$ (slip) | 0.68 | — | — | $\in(0,1)$ (Wiesner $\sigma<1$) |
| $r_{\text{exit}}$ [m] | 0.25 | 0.25 | 0.25 | $=r_2$ ($\phi=90°$) |
| validity | 1.000 | 1.000 | 1.000 | $>0.5$ |

The three tiers agree on PR to $<5\%$ (meanline vs. spanwise-resolved + repositioning, second-order here). **Efficiency reads high** (~0.98) because only incidence + skin-friction loss are modelled; blade-loading/clearance/disk-friction are deferred — so, as for V5/V6, point-by-point Eckardt reproduction and stage-map traversal are **[VERIFY]**, blocked on the reference library and the deferred loss components.

**Measured (M7-4): "Tier-3 radial repositioning needs in-blade subdivision" — REVISED 2026-07: refuted.** Unlike the axial V5/V6 (straight annulus → Tier 3 ≡ Tier 2 bit-for-bit, C.4), V7 is the first curved path carrying a blade row *and* streamline repositioning. M7-4 originally recorded that edge-only stations diverge "the §6.4 odd-even streamwise mode at any relaxation" and that six INBLADE stations were measured-necessary (a narrow pocket). The 2026-07 stabilization diagnosis refuted the attribution: the edge-only failure was the classical driver accepting a continuity root on a **spurious negative-Vm branch** (mass balancing by sign cancellation → decreasing mass cumulative → abort), one member of the driver-artifact family described in the C.8 revision — not a repositioning envelope. Post-stabilization the edge-only case converges (173 iterations) to the same answer as the subdivided one (PR 2.4433 vs. 2.4540, <0.5%); the flipped tripwire `test_v7_centrifugal.py::test_tier3_edge_only_converges_after_stabilization` pins it. INBLADE stations remain the *resolution* choice for in-blade work/loss distributions (§2.5, §3.4) — they are no longer a convergence requirement.

### C.8 V8 — Mixed-flow compressor, structural (bound at M8-4; `tests/test_v8_mixed_flow.py`)

**Structural gate at all three tiers (Tier 3 since the 2026-07 stabilization)** — a mixed-flow impeller (`V8MixedFlow`) composed through the `Machine` facade with the centrifugal set on a **partial** axial→radial bend (exit $\phi_{max}=55°$, bend centre at $r_c=0.25$ m, $\Omega=1450$ s⁻¹, $\dot m=12$ kg/s). Converges at every tier, compresses (PR $=1.56$, $\eta\approx0.96$) with real loss, spins axial inflow up to slipped exit swirl ($V_\theta/U_2\approx0.45$), and exits mixed-flow: $r_{LE}=0.131 < r_{exit}=0.178 < r_c=0.25$, i.e. the flow turned partway ($0<\phi<90$) with a radius rise — neither the axial V5 nor the fully-radial V7.

**Measured (M8-4): "Tier-3 mixed-flow repositioning is beyond the current stabilization" — REVISED 2026-07: diagnosed and fixed; the attribution was wrong.** The failures across the $(n_{sl}, n_{inblade}, \Omega)$ grid were **not** the §6.4 odd-even repositioning mode (measured: streamlines barely move before death, pure repositioning-with-curvature on the same bend is stable, and station/spanwise density is irrelevant). The actual chain: (i) the master ODE's RHS $\sim$ core$/V_m$ is singular at $V_m=0$; integrating a q-o from boundary values stale relative to the transported fields — the unrelaxed closure switch-on being the main producer, amplified by the REE swirl term $\sim rV_\theta/r^2$ at low-radius mid-bend stations — crosses that singularity into non-finite or negative-$V_m$ garbage; (ii) the driver *fatally* boundary-checked the stale-guess split that the per-q-o continuity solves were about to repair (the killed states were proven solvable: cold re-solve max$|F|/\dot m = 4\times10^{-15}$); (iii) `_solve_qo` could accept roots on spurious negative-$V_m$ branches or at the out-of-domain cliff. The V7 "pocket" and the V8 "angle-specificity" were chaos in whether the transient garbage stayed finite — not a stability envelope. **Stabilization (2026-07):** the AD-10 flow-field check moved to the solved state (a broken *metrics* fit stays fatal); continuity roots are validated onto the strictly-positive branch; a q-o transiently without a positive-branch root has its boundary value frozen, with CHOKE_LIMITED declared only after `choke_patience` consecutive deficient iterations; and the *first* closure application is relaxed from the duct baseline through the same §6.2.4 rule as every later one. With these, V8 Tier 3 **converges** (396 iterations at the §6.4 throttle $\omega_{sl}\approx0.066$ — slow; acceleration, e.g. Newton finishing, is the recorded follow-up) to PR 1.587, within a few percent of Tier 2; `test_tier3_converges_after_stabilization` pins it. Bands are structural plausibility gates; every centrifugal coefficient stays `[VERIFY]`, and $\eta$ reads high (only incidence + skin friction modelled) as for V7.

### C.9 Operability — BC-switching + surge flag (bound at M5-4; `tests/test_v9_operability.py`, `tests/test_backpressure.py`)

The M5 driver stack: global Newton over the pure residual (§6.3), continuation in $\dot m$ (§6.7), and the hysteretic exit-pressure BC-switch (§6.6). Bound as two behaviours on the two cases each is well-posed for (`slcflow/verification/v9_operability.py`).

**Back-pressure residual form (§6.6), round trip vs. normal mode.** In choke-proximal mode $\dot m$ joins the state and the assembler appends one residual: static pressure at the $q=0$ node of the throttling station $= p_{exit}$. Solve normal at $\dot m_0$, read the produced $p_{exit}$, feed it back as a `BackPressureSpec`, and the back-pressure solve recovers the state from a *different* seed:

| Check | Tolerance | Measured (2026-07, M5-3) |
|---|---|---|
| recovered $\dot m$ vs. $\dot m_0$ (V1c forced vortex, $N_{sl}=9$) | rtol 1e-4 | exact to solver tol |
| recovered state $x$ vs. normal-mode $x$ | atol 1e-5 | — |
| appended back-pressure row at the consistent state | $<10^{-3}\,p_{exit}$ | — |
| monotonicity: higher $p_{exit}\Rightarrow$ lower $\dot m$ | sign | holds |

**Stable BC-switching across choke** (swirling-duct testbed, clean annulus capacity): starting near choke ($c\approx0.07$), the traversal switches to the back-pressure branch below $c_{sw}=0.10$, throttles until $c$ clears $c_{sw}+\delta_{hys}=0.15$, switches back, and continues to stall — **exactly one out-and-back switch, no limit-cycling**, achieved $\dot m$ monotone throughout. Newton reaches the fixed point in ~3 iterations vs. ~15 for classical relaxation on V1c (§6.3 quadratic locally).

**Surge-flag behaviour** (V5 meanline rotor): the operating line rises in PR and the traversal reports `pr_turnover` at the peak with the criterion recorded (§6.7 "report, don't solve through"). **[VERIFY]** point-by-point against a *reported* surge line (blocked on reference data, as for V5); and the *V5* choke-knee traversal itself is **[VERIFY]** — the single-node continuity Jacobian is singular at the capacity peak (measured M6-4: V5 meanline chokes at $\dot m\approx175$ kg/s), a *continuity* feature no loss closure moves. **M6-4 correction:** M6 delivered the *turbine* shock term, which by AD-5 does not apply to the Lieblein *compressor* set V5 uses; the V5 supersonic-branch traversal needs (i) $\dot m$ as a state unknown — already built (M5-3 back-pressure mode) — and (ii) a *compressor* shock-loss closure (Koch–Smith/Aungier, a recorded compressor-set deferral). **(ii) delivered 2026-07:** the Aungier §6.7 compressor shock loss is now in the axial-compressor set (`shock_loss`/`normal_shock_pt_ratio`, `docs/references/AUN-C.md`) — a normal shock at the geometric-mean Mach $\sqrt{M_1 M_{ss}}$ (Eq 6-71) with $M_{ss}$ from the equivalent-diffusion bracket, inlet-referenced and added to $\bar\omega$, C¹ at onset, inert subsonic (all current V5 cases unaffected). What remains for the traversal is a *transonic* V5 rotor geometry (an in-window supersonic-inlet design) run with the back-pressure/continuation mode onto the supersonic-$\dot m$ branch — a V5-calibration case-design + continuation matter, not a missing closure. The BC-switch machinery is case-independent and bound on the testbed. **Transonic V5 case — in-window meanline gate (2026-07, corrects the earlier two-branch framing).** The `V5TransonicRotor` (`verification/v5_axial_compressor.py`) is a high-$\Omega$ rotor with supersonic *relative* inlet Mach ($M1_{rel}\approx1.14$) on which the Aungier §6.7 shock loss is active. The 2026-07 characterization **overturned the "two-branch / supersonic-branch traversal" premise recorded above**: the in-window condition is set by blade *loading* — the equivalent-diffusion factor $D_{eq}$ — **not** by which meridional-continuity branch the solve lands on. Direct probing (scratchpad, meanline): the ordinary subsonic-meridional branch the mass-flow driver already reaches is out-of-window only because the original geometry was over-diffused ($D_{eq}\approx2.30 > 2.0$ ceiling → $v_d=0$; every other validity factor, including the shock $v_{sh}=1$ at $M_{shock}=1.32<1.7$, is fine). The supersonic-meridional branch is *worse* ($D_{eq}$ higher still). Reducing the relative turning ($\beta_2$ $-52°\to-58°$) drops $D_{eq}$ into the window while keeping $M1_{rel}>1$ and the shock active, so the **plain mass-flow driver converges a genuine in-window transonic point** (Tier-1 meanline, $\dot m=55$: validity $\approx0.99$, PR $\approx1.51$, $\eta\approx0.86$) — pinned by `test_v5_axial_compressor.py::test_transonic_meanline_is_in_window_and_shock_active`. So the transonic structural gate is **met on the ordinary branch; no supersonic-branch driver was required**. Two case-design bounds remain (not driver matters): the in-window pocket is narrow in $\dot m$ ($\sim55$), and the **spanwise** tiers still read validity 0 (constant metal angle over the radius ratio swings $\beta_1$ across span → endwall $D_{eq}$ out of window; narrowing the annulus to fix it fights the high blade speed the transonic condition needs — an all-tier in-window transonic case is a deferred case-design refinement). The genuine **meridional-supersonic-branch traversal driver** is a real but *separate* capability, needed only for design points deliberately on the supersonic-meridional branch — not this case, and not a V5 blocker. Measured branch structure: at fixed $\dot m$ each station's continuity is folded at $M_m=1$; a back-pressure continuation decreasing $p_{exit}$ from the subsonic point drives $\dot m$ up to the LE capacity peak and pins at $M_{m,LE}=1$ (numerical failure), and a mass-flow Newton seeded on the per-station supersonic roots stalls with the downstream stations pinned at their own $M_m=1$ peaks — so the traversal needs arclength/branch-aware handling, not the existing pieces wired together.

**Meridional-supersonic-branch driver — delivered (2026-07).** Built as a general, reusable capability (`drivers/supersonic.py`, `solve_supersonic_branch`): **pseudo-arclength (Keller) continuation** parametrising the solution curve in $(\text{state},\dot m)$ by arclength, so the augmented Jacobian stays non-singular *at* the fold and the traversal walks from the subsonic branch, through the sonic turning point, onto the supersonic branch. The $\dot m$ column of the augmented Jacobian is analytic ($\dot m$ enters continuity linearly); the state columns are forward-difference with the Newton driver's positive-branch feasibility guard; variable scaling (sonic-speed / mass-flow) balances the arclength (measured necessary — an unscaled arclength creeps in $\dot m$ because $V_m$ dominates the norm near the fold). Once past the fold and descending onto $\dot m=\text{target}$, a fixed-$\dot m$ Newton lands the exact on-target supersonic root (the branch is selected, so that root is regular). **Verified** against an independent isentropic area–Mach reference on a purpose-designed meanline converging–diverging nozzle (`tests/test_supersonic.py`): the classical mass-flow driver chokes above the throat capacity (the fold), while the arclength driver crosses it — the turning point matches the analytic throat capacity to $<0.2\%$, and the landed supersonic throat Mach matches the isentropic area–Mach supersonic root to $<0.3\%$ (e.g. $M_m=1.397$ at the sample point), with the large-area inlet/exit staying subsonic (a rank-1 fold).

**Closure-lagged blade-row extension (2026-07).** For closure-fed rows the swirl/loss closures depend on the flow field and are lagged (AD-4); the supersonic-branch field differs from the subsonic seed's (different $\dot m$, and — where the row's own flow goes supersonic — different inflow), so the closures must be re-evaluated at the landed supersonic state. The driver's ``rows`` path bootstraps onto the supersonic branch by arclength ONCE, then hands the supersonic seed to `solve_newton` at $\dot m=\text{target}$, which runs the SAME outer quasi-Newton closure-lag loop it uses everywhere (§6.3): the fold is behind the seed, so the Newton inner's positive-$V_m$ guard keeps it on the supersonic branch while the outer loop re-lags the closures until the §6.2.5 closure norm converges. Verified on a Lieblein compressor row placed UPSTREAM of the throat (its inflow stays subsonic → closures in-window) with $\dot m_\text{target}\neq\dot m_\text{seed}$ so the flow-dependent closure genuinely re-lags (`tests/test_supersonic.py`): the throat crosses to $M_m\approx1.5$, the row inflow stays subsonic and Lieblein-valid, and the lagged closure is self-consistent at the landed field (a fresh evaluation agrees to $<0.1\%$) — whereas freezing the seed's closures leaves a several-percent inconsistency. The closure lag is a Picard iteration at rate $(1-\text{closure\_relax})$, so the conservative default relax (kept for stiff swirl-continuity loops, M4-4) needs $\sim$60 outer passes; weakly-coupled cases tolerate a larger relax. A **fully supersonic ROW inflow that folds several stations at once** (measured on a transonic rotor, above) is the harder multi-fold regime this single-fold arclength does not claim.