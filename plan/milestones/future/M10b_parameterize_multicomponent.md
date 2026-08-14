# M10b — Multi-dimensional surrogates + multi-component properties

**Effort:** 4 days (may split into two milestones at the M10b/M10c seam below)
· **Depends on:** M10 · **Parallelizable:** with M11, M12

## Goal
M10 generalized the surrogate-swap mechanism (`register_relation`/
`swap_relation`) from a power-only naming convention to an opt-in registry that
can attach any Pyomo-expressible relationship — polynomial or not — to any
relationship a unit declares swappable, so long as its **target is
one-dimensional** (indexed over time alone). That single restriction is this
milestone's whole scope: everything gated behind it.

This milestone removes it in two independent steps, plus two smaller,
independently-useful pieces the M10 feasibility review surfaced:

- **Tier 2 — multi-dimensional surrogate targets.** Let `register_relation`
  accept a target indexed over more than time (e.g. `flux[t, component]`), and
  teach `swap_relation` to build the fitted relationship over that target's
  full index set, with each factor broadcasting from its own (possibly
  lower-dimensional) index.
- **Tier 3 — multi-component properties.** There is currently nothing to write
  a Tier-2 surrogate *for* on a real process stream: `SimpleAqueousFlow` has one
  phase and one component (`H2O`), so there is no per-component concentration
  variable a trace-contaminant rejection model could determine. This tier adds
  a multi-component property package and component-wise conservation across
  arcs — the prerequisite, independent of the surrogate machinery itself.
- **Tier 3b — property-block surrogates.** Swapping a *property correlation*
  (density, a partition coefficient) rather than a unit relationship needs the
  same registration idea on `StateBlockData`, which builds its properties
  on-demand — a distinct mechanism from `OpsBlockData.swap_relation`.
- **Tier 3c — structured surrogate parameters.** `SurrogateSpec.coefficients` is
  `dict[str, float]`; a neural surrogate's weight matrix or an ARIMA lag
  polynomial does not fit that shape today, inline or via `source`.

Tiers 2–3c are additive to a live model; **Tier 3d (the deep swap audit)** is a
debugging tool, not required by anything else here, and is the natural place to
cut this milestone in half if 4 days proves optimistic — see "Splitting this
milestone" below.

## Read first
- `plan/01_architecture.md` §5 (the flexparameterize section this milestone
  extends) and §7 (R10, R11 — the swap contract and why it exists)
- `plan/00_conventions.md` §9 (never delete a Pyomo component — every tier here
  still swaps in place)
- The M10 PR's `swap_relation`/`register_relation` in
  `src/flexops/core/ops_block.py` and `RelationRecord`/`iter_swapped_relations`
  in `src/flexops/core/registration.py` — read the docstrings on both before
  writing code here; this milestone extends them, not replaces them.
- `src/flexops/properties/simple_aqueous.py` (the property package Tier 3
  generalizes) and `src/flexops/unit_models/base/sido.py` (`split_definition`,
  the unit relationship Tier 2's guard currently blocks from going
  per-component)

## Files to create or modify
- `src/flexops/core/ops_block.py` — widen `register_relation`'s dimensionality
  guard; generalize `swap_relation`'s constraint-building loop and the
  polynomial builder to multiple index dimensions.
- `src/flexops/properties/` — a new multi-component property package (name and
  module layout: implementer's choice, but mirror `simple_aqueous.py`'s
  structure so unit models built on it need minimal changes).
- `src/flexops/unit_models/base/` — the topology bases' mass balances gain
  component-wise conservation where the multi-component package is in use.
- `src/flexcore/config/schema.py` — a typed field for structured coefficients
  (Tier 3c); document the shape (conventions §4: no opaque nested blobs).
- `src/flexops/core/registration.py` — the deep-audit helper (Tier 3d), if this
  milestone is not split before reaching it.
- Tests colocated per package, mirroring M10's layout.
- Docs: extend `docs/explanation/config_schema.md`'s "Which relationships are
  swappable" section; update `docs/how_to/parameterize_from_data.md` with a
  multi-component example once Tier 3 exists.

## Specification

### Tier 2 — multi-dimensional surrogate targets
`register_relation` currently rejects any `target` whose `index_set().dimen !=
1`, pointing here. Remove that rejection for a target indexed as `(time, *)`
for any additional dimension(s) `*` (component, phase — whatever the extra
index is), and generalize `swap_relation`'s rule-building:

- The fitted Constraint must be built over `target.index_set()` in full (today
  it is built over `target.index_set()` already — that part is dimension
  agnostic — but the **rule** assumes a single index `t`; it must unpack
  `(t, *rest)` and pass the full index tuple to `body`).
- Each **factor** in a coefficient term may itself be lower-dimensional than
  the target (a per-component flux driven by a scalar-in-`t` pressure and a
  per-component concentration). The real design question is how a
  lower-dimensional factor's value is resolved at a higher-dimensional index —
  broadcast it across the missing dimension(s), or require every factor to
  share the target's full index. Pick the smallest rule that lets
  `flux[t, comp] == f(pressure[t], conc[t, comp])` be expressible, and document
  the choice in `_polynomial_terms`'s docstring; this is the one place this
  tier is not mechanical.
- `Constraint.Skip` handling (M10) must keep working per-index over the full
  multi-dimensional index set, not just per `t`.
- The non-polynomial builder contract (M10: `(unit, surrogate, target) ->
  body`) is unchanged in shape; `body` now takes the full index tuple.

### Tier 3 — multi-component properties
A new property package alongside `SimpleAqueousFlow`/`SimpleGasFlow`
(`plan/01_architecture.md` §3.7), carrying either `conc[t, comp]` (a
concentration state) or `flow_mass_phase_comp[t, phase, comp]` (a per-component
mass flow) — implementer's choice, but document the tradeoff (concentration is
the natural regression target for a rejection model; a per-component flow is
the natural conserved quantity across an arc). Whichever is primary, the other
must be derivable from it and the bulk `flow_vol_phase`.

- **Component-wise conservation.** Every topology base's mass balance
  (`SISOBlock`'s pass-through, `SIDOBlock`'s split, `DIDOBlock`'s transfer) gains
  a per-component version alongside its existing bulk-flow version — a
  component that is not tracked by a unit's specific chemistry (most units)
  simply passes through unchanged, mirroring the existing pass-through pattern
  for pressure/temperature.
- **A worked example.** Extend `ReverseOsmosis` (or add a small new unit) with a
  rejection relationship: `permeate_conc[t, comp] == f(feed_conc[t, comp],
  pressure[t])`, registered via `register_relation` with the now-permitted
  `permeate_conc` (component-indexed) target — this is the concrete consumer
  that proves Tier 2 actually unblocks the trace-contaminant case, not just a
  synthetic test fixture.
- **Arcs.** Confirm `Port.Extensive`/`Port.Equality` (M09, `arc-conservation-port-
  rules`) extend correctly to a per-component flow without new port machinery;
  if they do not, that gap is this tier's, not a new one.

### Tier 3b — property-block surrogates
`SimpleAqueousStateBlockData` builds properties on demand
(`{"dens_mass_phase": {"method": "_dens_mass_phase"}}`), a different
construction pattern from `OpsBlockData`'s eager build. A property correlation
(a density model, a temperature-dependent partition coefficient) needs its own
registration/swap pair on `StateBlockData` — do not attempt to route it through
`OpsBlockData.swap_relation`, which assumes a unit block, not a state block.
Scope the design (registration point, on-demand-construction interaction)
before writing code; this is the least mechanical piece here after Tier 2's
broadcast question.

### Tier 3c — structured surrogate parameters
Add a typed field to `SurrogateSpec` for a parameter shape `coefficients:
dict[str, float]` cannot hold — a weight matrix, a lag-polynomial coefficient
vector. Per conventions §4 ("no opaque nested JSON blobs"), this must be a
documented, typed pydantic field (e.g. a `weights: dict[str, list[list[float]]]`
or a small nested model), not a free-form `dict[str, Any]` escape hatch. Extend
`load_surrogate_source` (M10) to fill it in from a sidecar the same way it
fills `coefficients` today. A builder consumes it exactly like `coefficients` —
no change to the builder-registry contract.

### Tier 3d — the deep swap audit
`iter_swapped_relations` (M10) is O(registered relations) because
`swap_relation` records its own `fitted` field as it goes; it cannot detect a
relationship changed outside that call (a hand-edited rule, a rebuilt
component). A deep audit — construct a shadow default unit from the same
config and diff constraint bodies against the live model — answers "has
anything drifted from its defaults, however it happened?" Gate it behind an
explicit flag (e.g. `iter_swapped_relations(model, deep=True)`); it is
strictly more expensive and only useful once models start arriving from
outside the sanctioned `swap_relation` path.

## Splitting this milestone
If 4 days is optimistic, split after Tier 3c: **M10b** (Tiers 2–3c, the
capabilities other milestones or users would actually reach for) and **M10c**
(Tier 3d alone, a debugging tool with no other dependents). Nothing in 2–3c
depends on 3d.

## Pitfalls
1. **The broadcast rule silently doing the wrong thing.** A lower-dimensional
   factor resolved against the wrong index (e.g. summed instead of broadcast)
   produces a model that builds and solves but is physically wrong — the
   dangerous failure mode. Test the broadcast rule directly against a hand
   computation before testing anything built on top of it.
2. **Conservation drift.** A per-component pass-through that silently drops an
   untracked component (instead of passing it through unchanged) breaks mass
   balance invisibly. Mirror the existing pressure/temperature pass-through
   pattern exactly.
3. **Routing a property-block surrogate through `OpsBlockData.swap_relation`.**
   It assumes a unit block's registry; a state block is a different object with
   a different lifecycle (on-demand construction). Design Tier 3b's own
   registration point rather than forcing the existing one to fit.
4. **An opaque `dict[str, Any]` for Tier 3c.** Conventions §4 forbids it even
   under schedule pressure; a typed, documented field is not optional scope.
5. **Solver-class surprise.** Any nonlinear surrogate (Tier 2 or 3) can push a
   model from LP/MILP to NLP/MINLP; the facade errors loudly rather than
   transforming (R5) — this is expected, not a regression, but example/test
   models should request IPOPT/SCIP explicitly rather than relying on solver
   auto-selection to happen to work.

## Tests
Written test-first, one tier marker each, mirroring M10's structure.

`src/flexops/tests/core/test_ops_block.py` (`unit`)
- `test_register_relation_accepts_a_component_indexed_target` — the guard no
  longer rejects `(time, component)`.
- `test_swap_relation_broadcasts_a_lower_dimensional_factor` — a
  `flux[t, comp] == f(pressure[t], conc[t, comp])` fit at several probe points,
  checked against a hand computation (Pitfall 1).
- `test_swap_relation_skips_multidimensional_indices_a_body_declines` — the M10
  `Constraint.Skip` behavior still works per-index over the full index set.

`src/flexops/tests/properties/` (`unit`)
- Component-wise state variable construction; conservation across a bare arc
  for a tracked and an untracked component (Pitfall 2).

`src/flexops/tests/unit_models/` (`component`)
- The worked rejection-relationship example, end to end: fit-shaped
  probe-point comparison, no solver required, matching M10's own round-trip
  test's style.

`src/flexcore/tests/config/test_schema.py` (`unit`)
- Tier 3c's typed field round-trips through `dump_model_config`/
  `load_model_config` and through a `source` sidecar.

`src/flexops/tests/core/test_registration.py` (`unit`, if Tier 3d is in scope)
- `test_deep_audit_detects_an_externally_altered_relation` — a relation changed
  by hand (not through `swap_relation`) is invisible to the cheap
  `iter_swapped_relations` and visible only with `deep=True`.

## Documentation tasks
- `docs/explanation/config_schema.md` — extend "Which relationships are
  swappable" with the multi-dimensional case and the broadcast rule.
- `docs/how_to/parameterize_from_data.md` — a multi-component worked example.
- `docs/reference/flexops/core.rst` / a new properties reference page for the
  multi-component package.
- CHANGELOG entry under "Unreleased", flagged **BREAKING** if the property
  package addition changes any existing unit's default behavior.

## Definition of Done
- [ ] `register_relation` accepts a target indexed over time plus any number of
      additional dimensions; the one-dimensional-only guard is gone.
- [ ] `swap_relation` builds the fitted relationship over the target's full
      index set, with documented, tested broadcast semantics for
      lower-dimensional factors.
- [ ] `Constraint.Skip` (M10) still works per-index over a multi-dimensional
      index set.
- [ ] A multi-component property package exists with component-wise
      conservation across arcs, verified for at least one tracked and one
      untracked component.
- [ ] A worked rejection/flux relationship on a real unit demonstrates the
      Tier 2 mechanism against a genuine multi-component consumer, not only a
      synthetic test fixture.
- [ ] `SurrogateSpec` carries a typed, documented field for structured
      parameters (no opaque `dict[str, Any]`); `load_surrogate_source` fills it
      from a sidecar.
- [ ] Property-block surrogate swapping is scoped and, if built this
      milestone, implemented behind its own registration point (not routed
      through `OpsBlockData.swap_relation`).
- [ ] If Tier 3d is in scope: a deep audit mode exists, is opt-in, and is
      documented as strictly more expensive than the default.
- [ ] import-linter clean; docs build with `sphinx-build -W`; CHANGELOG updated.
- [ ] plus the generic DoD in CLAUDE.md
