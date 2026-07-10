# The config schema

flex-pse is **config-driven**: the entire model and run — the TimeBlock, the
property package, costing, and the network/plant/unit tree — are built from a
single version-controlled config artifact. Hand-written Python (like the
API-freeze script) stays a supported thin path, but nothing essential lives only
in imperative code: if it can be configured, it is in the file (decision R3,
`plan/01_architecture.md` §2.3).

## Pydantic is the authority; YAML is canonical on disk

The schema **authority** is the set of pydantic v2 models in
`flexcore.config.schema`. They define every field, its type, and its
one-line description (which renders into these docs), and they reject
undocumented keys (`extra="forbid"` — an undocumented key does not get to exist,
conventions §4).

The **canonical on-disk format is YAML**: config files are both human-tracked in
version control (comments, readable diffs, anchors for repeated tariff/unit
blocks) and written programmatically by external tools. JSON is also accepted on
load. The loader dispatches on the file suffix and the dumper quotes ambiguous
bare scalars, so the YAML "Norway problem" (`no`/`on`/`yes` parsing as booleans)
never bites. {func}`flexcore.config.io.load_model_config` and
{func}`~flexcore.config.io.dump_model_config` are the entry points.

## Versioning and migrations

Every persisted config carries a mandatory integer `schema_version`. On load, a
missing or too-new version is a `FlexConfigError`; older versions step through
the `MIGRATIONS` table (empty at v1) before validation. A JSON Schema exported
from `ModelConfig.model_json_schema()` is checked in under
`src/flexcore/config/schemas/` (with `sort_keys=True, indent=2` for stable
diffs) so schema drift shows up in review and external writers can validate
against it.

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

## The FlexParameterize ↔ FlexOps seam

The config is also the **durable contract** between FlexParameterize and FlexOps.
FlexParameterize can either mutate a live FlexOps model in place *or* emit a
config that rebuilds the parameterized model; the round-trip invariant is that
both produce equivalent behavior. Because the schema is versioned and
serializable, this seam is where the monorepo splits into separate repositories
later.

<!-- TODO(M14): .. flexops-config-table:: flexcore.config.schema.ModelConfig -->
