# Independent Audit Charter — slcflow

> **How to use this file.** You are being asked to perform an independent,
> adversarial audit of this codebase. Read this charter, then execute it. You
> have no prior context on this project and that is intentional — your value is
> that you are cold. Do not assume anything below the "Prime directive" is true
> until you have verified it yourself.

## Context

`slcflow` is a reduced-order streamline-curvature (SLC) throughflow solver for
turbomachinery preliminary design. **The entire codebase — code, tests, status
docs, and commit history — was generated end to end by an AI coding agent**
across milestones M0–M8. The human owner wants an independent audit *before*
trusting it, precisely because the same agent that wrote the code also wrote
its own self-assessment. Your job is to audit **both the code and that
self-assessment**.

## Prime directive: what is authoritative, what is a claim

- **Authoritative (the oracle for *intended* behaviour):** the three frozen
  specs in `docs/` — `theory_manual.md` (physics; **Appendix A is normative
  for signs**, Appendix B for loss→entropy, Appendix C for tolerances),
  `architecture_specification.md` (the AD-1…AD-10 binding decisions, package
  layout), `module_specification_geometry_and_grid.md`. Judge the code against
  these.
- **Claims under test — NOT ground truth:** `CLAUDE.md` (the agent's status
  log and "measured findings"), `docs/overview.md` (the agent's guided tour and
  its "what is/isn't established" section), commit messages, and any code
  comment asserting a result. Read them to learn what the author *claims*, then
  independently confirm or refute. Where a doc says "X is proven" or "measured:
  Y," treat that as a hypothesis to test, not a fact.

A passing test is also a claim. A green suite proves the code agrees with the
tests; it does not prove the tests are meaningful.

## Audit axes (ranked by expected value)

### 1. Test quality — is the green suite load-bearing? (highest priority)
The suite is large and green, but much of the verification ladder is described
by its own author as "structural" with wide "plausibility bands" and pervasive
`[VERIFY]`. The central question:

> **Which tests would still pass if the physics were wrong?**

Hunt for: assertions that check outputs against the code's *own* output rather
than an independent reference (tautologies); bands so wide they exclude nothing;
"structural" checks that pin no number; "adjudication" tests that actually read
the implementation they claim to independently check; fixtures that bypass the
path they purport to test. Separately, identify the genuinely load-bearing
tests and confirm they are as strong as claimed (e.g. any grid-convergence-order
or tier-consistency/bit-for-bit claims — reproduce them).

### 2. Sign conventions vs. Theory Manual Appendix A
Appendix A is normative for every sign. The author's own notes admit several
"sign tangles resolved by probing" (turbine throat exit swirl; centrifugal slip
direction). Those are prime suspects for a sign flipped to make a test pass
rather than derived. Re-derive the disputed signs from Appendix A and check the
code and the tests against the derivation, not against each other.

### 3. Binding decisions AD-1…AD-10 (architecture_specification.md)
Look for violations, especially:
- **AD-3** residual purity — any hidden state, mutation, or I/O reachable from
  `assembly.ResidualAssembler.residual`? (The Newton driver's correctness
  depends on this.)
- **AD-1** — any real tier branching (`if tier …`) masquerading as data?
- **AD-5** — any machine-type knowledge (loss/deviation constants) leaking out
  of `closures/`? (`tools/check_imports.py` claims to enforce this — verify the
  checker is not trivially defeatable.)
- **AD-6** — data-dependent Python branching on flow arrays on the residual
  path? (`tools/check_ad6.py` is a token-level grep lint — assess how much it
  actually catches vs. what it misses.)
- **AD-10** — any path that can raise an exception on the residual/flow path
  instead of saturating and returning a typed status?

### 4. C¹ smoothness discipline (Theory Manual §7.3)
Anything touching flow arrays must be C¹ (no raw `if`/`clip`/`abs`/`min`/`max`).
Spot-check that `closures/smoothmath.py` is actually used where required, and
that the C¹ tests are non-vacuous — the author claims a refinement-scaling test
plus a negative control; confirm the negative control genuinely rejects a
known-discontinuous function.

### 5. Numerical correctness of the kernel
- The master q-o momentum ODE integration (§5.3) and the continuity /
  capacity / choke handling (A.7): do an independent hand or small-script check
  of at least one integration step and one choke case.
- The Newton driver: does the finite-difference Jacobian actually match the
  residual it differentiates? Is the line-search feasibility guard sound?
- The author flags **Tier-3 radial/mixed streamline repositioning as fragile**
  (converges only in a narrow, allegedly "angle-specific" pocket) and offers
  "INBLADE subdivision" as the remedy. Verify this characterization: is the
  remedy principled, or a band-aid masking an instability that will resurface?

### 6. Honesty of the `[VERIFY]` disclosure
Is every unvalidated result disclosed where a user would actually see it, or is
anything structural-only presented as "verified/passing"? Are the author's
"measured findings" (e.g. that spanwise mixing is a multistage *convergence
prerequisite*; the repositioning pockets) reproducible, or just-so narratives?

## Method

- Read the relevant frozen-spec section **before** judging the code for that
  section. Cite section numbers in findings.
- Prefer **independent re-derivation / a small independent check** over
  trusting a docstring or a test's own assertion. A hand-computed ODE step, an
  analytic conservation identity, or a from-spec re-derivation of a sign is
  worth more than reading the code's comments.
- Establish the baseline first: run `pytest -q`, `python tools/check_imports.py`,
  `python tools/check_ad6.py`. Confirm the claimed "347 passing, gates green"
  state actually holds before auditing on top of it.

## Deliverable

A findings report, **ranked by severity**, in three clearly separated buckets:
1. **Correctness bugs** — code that is wrong vs. a spec section or an
   independent check. Each: file/line, the spec section or check it fails, and a
   concrete reproduction or failure scenario.
2. **Test-quality gaps** — tests that pass but don't constrain what they claim
   to. Each: which test, why it's vacuous/weak, and what a real test would
   assert.
3. **Disclosure gaps** — anything unvalidated that is presented as trustworthy.

Then, explicitly, **what you independently confirmed as correct** — so the
owner knows what is genuinely trustworthy, not only what is broken. An audit
that only lists problems is half an audit.

## Out of scope (do not spend budget here)

These are *deliberately* deferred (see ARCH-9) — do not report them as defects:
real-gas fluid backend, JAX/AD array backend, closure-in-Newton, endwall
boundary-layer *model* (prescribed blockage stands in), cooling/bleed flows,
GUI/plotting. Likewise, **every correlation coefficient is knowingly `[VERIFY]`
pending a reference library** — flag a coefficient only if it is *wrong versus a
source the code itself cites*, not merely uncalibrated. The point of the audit
is the machinery, the honesty, and the tests — not the correlation calibration,
which is openly unfinished.
