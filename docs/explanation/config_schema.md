# The config schema

flex-pse is **config-driven**: the entire model and run — the TimeBlock, the
property package, costing, and the network/plant/unit tree — are built from a
single version-controlled config artifact. Hand-written Python (like the
API-freeze script) stays a supported thin path, but nothing essential lives only
in imperative code: if it can be configured, it is in the file (decision R3,
`plan/01_architecture.md` §2.3).

## Pydantic is the authority; JSON is canonical on disk

The schema **authority** is the set of pydantic v2 models in
`flexcore.config.schema`. They define every field, its type, and its
one-line description (which renders into these docs), and they reject
undocumented keys (`extra="forbid"` — an undocumented key does not get to exist,
conventions §4).

The **canonical on-disk format is JSON**: it round-trips losslessly, matches
the exported JSON Schema that external writers validate against, and parses
without surprises. An already-parsed `dict` can be passed to the loader
directly. {func}`flexcore.config.io.load_model_config` and
{func}`~flexcore.config.io.dump_model_config` are the entry points.

## Versioning and migrations

Every persisted config carries a mandatory `schema_version`: a semantic-version
string like `"0.0.2"` (`CURRENT_SCHEMA_VERSION`). On load, a missing,
malformed, or too-new version is a `FlexConfigError`; older versions step
through the `MIGRATIONS` table (each hook stamps the version it upgrades to)
before validation. A JSON Schema exported from
`ModelConfig.model_json_schema()` is checked in under
`src/flexcore/config/schemas/` (with `sort_keys=True, indent=2` for stable
diffs, and descriptions collapsed to plain single-line text — wrapping is the
doc builder's job) so schema drift shows up in review and external writers can
validate against it. `export_json_schemas` takes an optional `filename` so
schemas for several versions can sit side by side.

## The model set

The schema shrinks from the top-level {class}`~flexcore.config.schema.ModelConfig`
down to individual variables:

- {class}`~flexcore.config.schema.ModelConfig` — the top-level artifact:
  `schema_version`, a {class}`~flexcore.config.schema.TimeConfig`, a `properties`
  spec, a {class}`~flexcore.config.schema.CostingConfig`, and **exactly one** of
  a {class}`~flexcore.config.schema.NetworkConfig` **or** a
  {class}`~flexcore.config.schema.PlantConfig` (a validator enforces the
  exactly-one-of rule with a clear field-path message).
- {class}`~flexcore.config.schema.NetworkConfig` — a named collection of plants
  plus inter-plant arcs.
- {class}`~flexcore.config.schema.PlantConfig` — a named collection of units plus
  the arcs between them.
- {class}`~flexcore.config.schema.UnitConfig` — one unit model: its class name,
  construction options, declared IO variables, an optional
  {class}`~flexcore.config.schema.SurrogateSpec`, a
  {class}`~flexcore.config.schema.UnitCommitmentConfig`, and an optional
  {class}`~flexcore.config.schema.ExternalDispatchSpec`.
- {class}`~flexcore.config.schema.IOVariableSpec`,
  {class}`~flexcore.config.schema.SurrogateSpec`,
  {class}`~flexcore.config.schema.CostingConfig` (carrying a
  {class}`~flexcore.config.schema.DRConfig` container slot), and
  {class}`~flexcore.config.schema.TimeConfig` fill in the leaves.

`flexops.build_model(config)` constructs the whole Pyomo model from a validated
`ModelConfig` — that function lands in M09.

## How a surrogate describes its function

{class}`~flexcore.config.schema.SurrogateSpec` has to carry relationships this
build has never heard of, so its `functional_form` is an **open string**, not a
fixed list of allowed values. It names a *builder*, and builders are registered
in code (`flexops.core.ops_block._RELATION_BUILDERS`). A form with no
registered builder is rejected when the model is built — with the known forms
listed — rather than when the config is validated. That is deliberate: a new
relationship shape should never force a schema revision, and a config written
by an external tool should survive being read by a build that cannot construct
every form in it.

Coefficients describe the function itself. Each key names the **term** its
coefficient multiplies: a `*`-separated product of input variable names, each
optionally raised to an integer power with `^`, plus the reserved key
`intercept` for the constant term. One grammar therefore spans every polynomial
relationship:

```json
{
  "functional_form": "bilinear",
  "input_variables": ["flow_out", "outlet_state.pressure"],
  "coefficients": {
    "intercept": 5.0,
    "flow_out": 0.42,
    "outlet_state.pressure": 1.1e-5,
    "flow_out*outlet_state.pressure": 2.3e-6
  }
}
```

Every factor is resolved on the unit — dotted paths into a state block work —
and normalized by its own units, so each coefficient is read in **the
relationship's target's own units** over the product of its factors' units (a
power draw's coefficients read in kW; a recovery relation's read in whatever
its target — say a flow — is in). The registered polynomial forms differ only
in the degree they admit (`linear` 1, `quadratic` and `bilinear` 2,
`polynomial` any), which is what makes a mislabelled relationship an error
instead of a silent surprise. `bilinear` is the *expanded* form: a constant,
each input on its own, and their cross term.

The builder registry is not limited to polynomials, either: a builder is
`(unit, surrogate, target) -> body`, where `body(t)` is any
Pyomo-expressible, dimensionless function of the unit's own variables — a
ratio, a softplus/ICNN forward pass, a lag polynomial that returns
`pyomo.environ.Constraint.Skip` for the horizon points its lag does not reach.
Registering another form is an entry in
`flexops.core.ops_block._RELATION_BUILDERS`, never a config-schema change.

When a relationship is too large to inline — or is not a set of coefficients at
all — `source` names a JSON sidecar supplying `coefficients`,
`input_variables` and `output_variables`. A relative path resolves against the
config file's own directory, and
{func}`~flexcore.config.io.load_model_config` fills it in at the boundary, so
nothing downstream ever sees a half-loaded relationship.

## Which relationships are swappable

Not every constraint a unit builds can be swapped — only the ones it
explicitly registered as swappable, via
{meth}`~flexops.core.ops_block.OpsBlockData.register_relation`. A unit's
mass/energy balance is never registered, so it can never be swapped: there is
no naming convention to accidentally satisfy, and
{meth}`~flexops.core.ops_block.OpsBlockData.swap_relation` refuses an
unregistered name outright, listing what *is* registered. Every
constant-intensity unit registers its own energy relation this way; an RO
skid additionally registers its `split_definition` (recovery/flux), and a
tank its `level_definition` (fill geometry) — conservation
(`split_mass_balance`, the holdup difference equation) is not registered on
either. {func}`~flexops.core.registration.iter_swapped_relations` walks a
whole model and reports which registered relations have actually been
swapped — a debugging aid, not a required step on any build or apply path.

## The FlexParameterize ↔ FlexOps seam

The config is also the **durable contract** between FlexParameterize and FlexOps.
FlexParameterize can either mutate a live FlexOps model in place *or* emit a
config that rebuilds the parameterized model; the round-trip invariant is that
both produce equivalent behavior. Because the schema is versioned and
serializable, this seam is where the monorepo splits into separate repositories
later.

<!-- TODO(M14): .. flexops-config-table:: flexcore.config.schema.ModelConfig -->
