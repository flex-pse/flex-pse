# The config schema

flex-pse is **config-driven**. The entire model and run, the TimeBlock, the
property package, costing, and the network/plant/unit tree, get built from
a single version controlled config artifact. Hand written Python (like the
API-freeze script) stays a supported thin path, but nothing that matters
lives only in imperative code. If it can be configured, it's in the file.

## Pydantic is the authority, JSON is canonical on disk

The schema **authority** is the set of pydantic v2 models in
`flexcore.config.schema`. They define every field, its type, and its one
line description (which renders into these docs), and they reject
undocumented keys (`extra="forbid"`, so an undocumented key simply doesn't
get to exist).

The **canonical on disk format is JSON**. It round trips losslessly,
matches the exported JSON Schema that external writers validate against,
and parses without surprises. You can pass an already parsed `dict` to the
loader directly. {func}`flexcore.config.io.load_model_config` and
{func}`~flexcore.config.io.dump_model_config` are the entry points.

## Versioning and migrations

Every persisted config carries a mandatory `schema_version`, a semantic
version string like `"0.0.2"` (`CURRENT_SCHEMA_VERSION`). On load, a
missing, malformed, or too new version raises a `FlexConfigError`. Older
versions step through the `MIGRATIONS` table (each hook stamps the version
it upgrades to) before validation. A JSON Schema exported from
`ModelConfig.model_json_schema()` gets checked in under
`src/flexcore/config/schemas/` (with `sort_keys=True, indent=2` for stable
diffs, and descriptions collapsed to plain single line text, since
wrapping is the doc builder's job), so schema drift shows up in review and
external writers can validate against it. `export_json_schemas` takes an
optional `filename` so schemas for several versions can sit side by side.

## The model set

The schema shrinks from the top level
{class}`~flexcore.config.schema.ModelConfig` down to individual variables.

- {class}`~flexcore.config.schema.ModelConfig` is the top level artifact.
  It holds `schema_version`, a
  {class}`~flexcore.config.schema.TimeConfig`, a `properties` spec, a
  {class}`~flexcore.config.schema.CostingConfig`, and **exactly one** of a
  {class}`~flexcore.config.schema.NetworkConfig` **or** a
  {class}`~flexcore.config.schema.PlantConfig` (a validator enforces the
  exactly one of rule with a clear field path message).
- {class}`~flexcore.config.schema.NetworkConfig` is a named collection of
  plants plus the arcs between them.
- {class}`~flexcore.config.schema.PlantConfig` is a named collection of
  units plus the arcs between them.
- {class}`~flexcore.config.schema.UnitConfig` describes one unit model.
  Its class name, construction options, declared IO variables, an optional
  {class}`~flexcore.config.schema.SurrogateSpec`, a
  {class}`~flexcore.config.schema.UnitCommitmentConfig`, and an optional
  {class}`~flexcore.config.schema.ExternalDispatchSpec`.
- {class}`~flexcore.config.schema.IOVariableSpec`,
  {class}`~flexcore.config.schema.SurrogateSpec`,
  {class}`~flexcore.config.schema.CostingConfig` (carrying a
  {class}`~flexcore.config.schema.DRConfig` container slot), and
  {class}`~flexcore.config.schema.TimeConfig` fill in the leaves.

`flexops.build_model(config)` constructs the whole Pyomo model from a
validated `ModelConfig`.

## How a surrogate describes its function

{class}`~flexcore.config.schema.SurrogateSpec` names a **predefined
surrogate class**, one of {class}`~flexcore.config.schema.SurrogateType`'s
members, and carries an opaque `data` mapping in the shape that class
defines. The class (`flexops.surrogates`) validates `data` and builds the
Pyomo relationship. A type not yet implemented raises `NotImplementedError`
when the model gets built, not when the config gets validated
(`quadratic`, `exponential`, `arima`, and `neural_network` are reserved
names today with no implementation yet, see
{doc}`../reference/flexops/surrogates`).

The only implemented class, `multilinear`
({class}`~flexops.surrogates.multilinear.MultilinearSurrogate`), is a
constant plus a sum of `coefficient * (product of distinct declared
inputs)`. That expanded form covers a linear relationship (no cross terms)
and a bilinear one (one cross term) alike. A coefficient key is a `*`
separated product of names from `input_variables`, each appearing at most
once (no `^` exponent, no repeated factor). The reserved key `intercept`
is the constant term.

```json
{
  "surrogate_type": "multilinear",
  "data": {
    "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
    "output_variables": {"power_electrical": "kW"},
    "coefficients": {
      "intercept": 5.0,
      "flow_out": 0.42,
      "outlet_state.pressure": 1.1e-5,
      "flow_out*outlet_state.pressure": 2.3e-6
    }
  }
}
```

`input_variables` and `output_variables` are `{name: units}` mappings,
each name resolved on the unit (dotted paths into a state block work). The
declared units are **the basis the relationship was fitted or written
in**, not whatever units the model happens to carry. So each factor
converts from its actual unit into its declared one before the coefficient
multiplies it, and the whole body converts from its declared output units
into the registered target's own units. Both conversions double as
validation. A declared unit that's dimensionally incompatible with the
model's variable raises a `FlexConfigError` naming the mismatch, instead
of silently rescaling.

Registering another surrogate class means adding a new class in
`flexops.surrogates`, never a config schema change. See
{doc}`../reference/flexops/surrogates` for the base class every one
implements. The other half of that extension point is a *regressor* that
can produce the class's `SurrogateSpec` from data.
{class}`~flexparameterize.regression.base.Regressor` is the matching
Protocol, and `flexparameterize.regression`'s registry names the same
reserved `SurrogateType` members (`quadratic`, `exponential`, `arima`,
`neural_network`) as `flexops.surrogates`. So building a future regressor
for one of them means implementing the protocol and registering the name,
on both sides at once.

When a relationship is too large to inline, `source` names a JSON sidecar
that supplies `data`. A relative path resolves against the config file's
own directory, and {func}`~flexcore.config.io.load_model_config` fills it
in at the boundary, so nothing downstream ever sees a half loaded
relationship.

## Which relationships are swappable

Not every constraint a unit builds can be swapped. Only the ones it
explicitly registered as swappable, through
{meth}`~flexops.core.ops_block.OpsBlockData.register_relation`, qualify. A
unit's mass or energy balance is never registered, so it can never be
swapped. There's no naming convention to accidentally satisfy, and
{meth}`~flexops.core.ops_block.OpsBlockData.swap_relation` refuses an
unregistered name outright, listing what *is* registered. Every
constant intensity unit registers its own energy relation this way. An RO
skid also registers its `split_definition` (recovery/flux), and a tank its
`level_definition` (fill geometry). Conservation (`split_mass_balance`,
the holdup difference equation) is never registered on either.
{func}`~flexops.core.registration.iter_swapped_relations` walks a whole
model and reports which registered relations have actually been swapped.
It's a debugging aid, not a required step on any build or apply path.

## The FlexParameterize and FlexOps seam

The config is also the **durable contract** between FlexParameterize and
FlexOps. FlexParameterize can either mutate a live FlexOps model in place
*or* emit a config that rebuilds the parameterized model. The round trip
invariant is that both produce equivalent behavior. The schema is
versioned and serializable, and that's what makes this seam the place the
monorepo splits into separate repositories later.

## Field reference

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.ModelConfig
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.TimeConfig
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.CostingConfig
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.NetworkConfig
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.PlantConfig
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.UnitConfig
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.IOVariableSpec
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.SurrogateSpec
```
