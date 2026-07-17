# M03 — SimpleAqueousFlow + OpsBlock base

**Effort:** 2–3 days · **Depends on:** M02 · **Parallelizable:** no

## Goal

Build the base class of every flex-pse unit model (`OpsBlockData` with the
IO/parameter/energy registration API, the external-dispatch hook, the
unit-commitment config slot, and the block-replacement hook), the minimal
property package that carries flow between units, the registration record
dataclasses that FlexParameterize and the docs generator consume, and the
**full** versioned pydantic config schema (`flexcore.config` —
`IOVariableSpec`…`ModelConfig`, YAML-canonical), which registration and
config-driven construction need. After this milestone a dummy unit builds on a
`ConcreteModel` + `TimeBlock`, registers its IO, exposes the dispatch/UC/replace
hooks, and is discoverable model-wide.

## Read first

- `plan/01_architecture.md` §3.2 (OpsBlock — R1: inherit `UnitModelBlockData`,
  **no ControlVolumes**; the `external_dispatch` hook, the `unit_commitment`
  sub-config, `from_config`/`build_model`), §3.5 (the UC logic layer the config
  slot feeds — built in M08), §3.7 (SimpleAqueousFlow), §2.3 (the full config
  schema — R3, YAML canonical), §5 (the two-way FlexParameterize coupling and the
  in-place parameter-update hook — R10), §7 (R1, R3, R8, R9, R10)
- `plan/00_conventions.md` §2 (energy naming), §4 (two config layers, never mixed)
- `plan/02_testing_and_ci.md` §1, §5
- `plan/03_documentation.md` §2 (why every Var/Constraint needs `doc=`)
- IDAES docs: "Custom Property Packages" — `PhysicalParameterBlock`
  requirements (`define_metadata`, state-variable dict, `build_state_block`).
  Study WaterTAP's zero-order package (`prop_ZO`) as the structural reference.

## Files to create or modify

- `src/flexcore/nomenclature.py` — energy-name constants + kind enum
- `src/flexcore/config/schema.py` — the full schema: `IOVariableSpec`,
  `SurrogateSpec`, `UnitConfig`, `PlantConfig`, `NetworkConfig`, `TimeConfig`,
  `CostingConfig` (with a `dr` container slot + `external_dispatch`), and
  top-level `ModelConfig`
- `src/flexcore/config/io.py` — `load_model_config`/`dump_model_config`
  (YAML canonical + JSON accepted) + JSON Schema export + migration table
- `src/flexcore/config/schemas/` — exported JSON Schema, checked in
- `src/flexops/core/registration.py` — record dataclasses, `IORegistry`, `iter_io_registry`
- `src/flexops/core/ops_block.py` — `OpsBlock` / `OpsBlockData`
- `src/flexops/properties/simple_aqueous.py` — `SimpleAqueousFlow` param/state blocks
- `src/flexops/__init__.py` — export `SimpleAqueousFlow` (API freeze uses it)
- Tests: `src/flexops/tests/core/test_ops_block.py`;
  `src/flexops/tests/properties/test_simple_aqueous.py`;
  `src/flexcore/tests/config/test_schema.py` (add `__init__.py`s)
- Docs: `docs/explanation/config_schema.md`, `docs/reference/flexcore/index.rst`,
  updated `docs/reference/flexops/core.rst`, new `properties.rst`

## Specification

### flexcore/nomenclature.py

```python
POWER_ELECTRICAL = "power_electrical"   # unit-level electrical draw, kW
POWER_THERMAL = "power_thermal"         # unit-level thermal/gas duty, kW

class PowerKind(str, enum.Enum):
    ELECTRICAL = "electrical"
    THERMAL = "thermal"
```

Docstring restates the §4 rules (never bare `power`/`energy`/`work`; costing
receives kW only). All code refers to these constants so typos are import errors.

### flexcore/config/schema.py (pydantic v2, 01_architecture §2.3, R3)

The config is the single source of truth: the whole model+run is built from one
version-controlled artifact (`flexops.build_model(config)`, M09). This milestone
creates the **full** schema — from `ModelConfig` shrinking to a single unit is no
longer the shape; `ModelConfig` is now the top-level artifact and the old
single-unit body moves to `UnitConfig`.

Every field carries `Field(description=...)` (they render into docs); every model
sets `model_config = ConfigDict(extra="forbid")` (conventions §4: undocumented
keys don't exist).

- `IOVariableSpec` — `name: str`, `role: Literal["input", "output"]`,
  `units: str`, `tag_hint: str | None = None`, `time_indexed: bool = True`.
- `SurrogateSpec` — `functional_form: Literal["constant_intensity", "linear",
  "nn", "arima", "multiconvex"]` (last three reserved: schema accepts, use
  rejects until post-v0), `coefficients: dict[str, float]`,
  `input_variables: list[str]`, `output_variables: list[str]`,
  `provenance: dict[str, Any] = {}` (free-form at v1 — implementer's choice).
- `ExternalDispatchSpec` — declares an external (DERMS) command source (§3.2/§3.6):
  `variable: str` (the controllable var to fix), `source: str` (file/tag pointing
  at the time-indexed command series), `fix: bool = True`. Field/name choices are
  the implementer's; keep it minimal — the logic that consumes it is first-classed
  on battery in M08.
- `UnitCommitmentConfig` — the per-unit UC sub-config (§3.5), every piece optional
  except status: `status: bool = True`, `startup_shutdown: bool = False`,
  `dwell: bool = False` (with optional min-up/min-down fields), `delays: ... = None`
  (upstream-linked startup delay — shape implementer's choice), `conditional: ...
  = None`. **Config entry only in M03** — the constraint-building logic layer is
  built in M08; here it is just the validated container.
- `UnitConfig` — the former `ModelConfig` body: `unit_model_class: str`,
  `construction_options: dict[str, Any] = {}`,
  `io_variables: list[IOVariableSpec] = []`, `surrogate: SurrogateSpec | None =
  None`, `unit_commitment: UnitCommitmentConfig = UnitCommitmentConfig()`,
  `external_dispatch: ExternalDispatchSpec | None = None`.
- `PlantConfig` — a named collection of units + arcs: `name: str`,
  `units: dict[str, UnitConfig]`, `arcs: list[...] = []` (arc shape — source/dest
  `unit.port` strings — implementer's choice).
- `NetworkConfig` — a named collection of plants + inter-plant arcs:
  `name: str`, `plants: dict[str, PlantConfig]`, `arcs: list[...] = []`.
- `TimeConfig` — `start_date: str`, `end_date: str`, `time_step: str` (a
  units-carrying expression rendered as a string, e.g. `"15 min"`; parsed at
  build time, §3.1).
- `CostingConfig` — `tariff_source: str`, `dr: DRConfig | None = None` (the
  DR **container** slot — §2.4/§3.6, no DR constraints built in v0), plus
  solve/objective options (implementer's choice); `DRConfig` is a placeholder
  container model.
- `ModelConfig` — the **top-level artifact**: `schema_version: int` (**mandatory,
  no default**), `time: TimeConfig`, `properties: dict[str, Any] = {}` (the
  property-package spec — kept loose at v1), `costing: CostingConfig`, and exactly
  one of `network: NetworkConfig | None` **or** `plant: PlantConfig | None` (a
  validator enforces exactly-one-of). `flexops.build_model(config)` (M09)
  constructs the whole Pyomo model from it.

Also `CURRENT_SCHEMA_VERSION = 1` and `export_json_schemas(directory)` writing
`model_config.schema.json` from `ModelConfig.model_json_schema()` with
`indent=2, sort_keys=True` (stable diffs). Run once; commit the output to
`src/flexcore/config/schemas/`.

### flexcore/config/io.py (YAML canonical, JSON accepted — R3)

`load_model_config(path) -> ModelConfig`; `dump_model_config(cfg, path) -> None`;
`MIGRATIONS: dict[int, Callable[[dict], dict]] = {}` (version→one-step upgrade
hook table, empty at v1).

- **Format.** JSON is the canonical on-disk format (pydantic remains the schema
  authority); 
- **Dump.** `dump_model_config` writes JSON for `.json` targets (round-trip
  through `cfg.model_dump(mode="json")`).
- **Version handling.** Load: missing or greater-than-current `schema_version` →
  `FlexConfigError`; older versions step through `MIGRATIONS`; then
  `ModelConfig.model_validate`, wrapping `ValidationError` in `FlexConfigError`
  while preserving pydantic's field-path text (e.g.
  `plant.units.tank.io_variables.0.role`).

### flexops/core/registration.py

Plain dataclasses; `var`/`param` fields hold live Pyomo references typed `Any`:

- `IOVariableRecord(var, name: str, role: str, tag_hint: str | None,
  units: str, time_indexed: bool)`
- `ParameterRecord(param, name: str, regressable: bool)`
- `PowerRecord(var, name: str, kind: str)` — `name` is the nomenclature
  constant value; `kind` a `PowerKind` value
- `IORegistry` — container with `io_variables: list[IOVariableRecord]`,
  `parameters: list[ParameterRecord]`, `power: list[PowerRecord]`
  (all `field(default_factory=list)`)
- `iter_io_registry(model) -> Iterator[tuple[Any, IORegistry]]` — walk
  `model.block_data_objects(descend_into=True)` (plus the model itself), yield
  `(block, registry)` for every block exposing a non-empty `_io_registry`.
  The yield shape is implementer's choice recorded here; the architecture only
  requires model-wide discoverability.

### flexops/core/ops_block.py

```python
from flexcore.compat.idaes import declare_process_block_class, UnitModelBlockData

@declare_process_block_class("OpsBlock")
class OpsBlockData(UnitModelBlockData):
    ...
```

R1: no ControlVolumes; subclasses hand-write their 1–3 balance constraints.
CONFIG (extends `UnitModelBlockData.CONFIG`):

- `property_package` — parameter block, default `None` (not every unit needs one).
- `flexops_config` — optional **already-validated `UnitConfig` instance**, default
  `None`; type-check in the ConfigValue `domain` — never a raw dict.
- `unit_commitment` — a `UnitCommitmentConfig` sub-config (§3.5), default an
  all-defaults instance (status on, everything else off). **Config slot only in
  M03**: it is validated and stored, but no UC constraints are built here — the
  status/startup/shutdown/dwell/delays/conditional constraint logic is the M08
  logic layer. `StorageTank` (M04) will set `status=False` (a tank has no on/off).
- `external_dispatch` — optional `ExternalDispatchSpec`, default `None`; the
  declared source/variable that `set_external_dispatch` (below) consumes.
- `relaxation` — the discrete-structure relaxation policy (default the exact/MIP
  policy), and `allow_bypass` (bool, default `False`). **Config slots only in
  M03**, same as `unit_commitment`: validated and stored, but the relaxation
  switching and bypass *constraints* are built in the M08 logic layer.

Set inherited `dynamic`/`has_holdup` defaults explicitly to `False` (R2). No
logic-layer *constraints* are built in M03 — the config slots above
(`unit_commitment`, `relaxation`, `allow_bypass`, `external_dispatch` wiring) are
declared so the schema is stable from M03; M08 builds the constraints that
consume them. (There is no `include_onoff` flag — on/off is the `status` piece of
`unit_commitment`, §3.5.)

`build()` creates `self._io_registry = IORegistry()`, then the registration
API — copy these signatures exactly (later milestones and docs reference them):

```python
def register_io_variable(self, var, role="input", tag_hint=None) -> None: ...
def register_process_parameter(self, param_or_var, regressable=True) -> None: ...
def register_power(self, var, kind="electrical") -> None: ...
```

- `register_io_variable`: `role` ∉ {"input","output"} → `FlexConfigError`.
  Derive `name=var.local_name`, `units=str(pyunits.get_units(...))` (first
  data object if indexed), `time_indexed=var.is_indexed()`. IO variables get
  **fixed during regression**.
- `register_process_parameter`: Param or Var; **found during regression** when
  `regressable=True`.
- `register_power`: `kind` must be a `PowerKind` value else `FlexConfigError`.

#### External dispatch hook (DERMS, §3.2)

Every unit exposes a base method to fix a controllable actuator variable to an
externally supplied, time-indexed command series — removing the dispatch degree
of freedom while leaving sizing free (the DERMS/aggregator case, Project 1).
Available on **all** units here; first-classed on `BatteryModel` in M08. Copy
this signature:

```python
def set_external_dispatch(self, var, series, *, fix=True) -> None: ...
```

- `var` is a time-indexed `Var` on this unit; `series` is a pandas Series/mapping
  aligned to `time_block.datetime_index` (accept a mapping keyed by timestamp or
  by integer index — coerce via the TimeBlock's `index_of`). For each `t`, set
  `var[t]` to `series[t]`; when `fix=True`, also `var[t].fix()` so the DOF is
  removed. Length/alignment mismatch or an unindexed `var` → `FlexConfigError`.
- Reads the declared source from the `external_dispatch` config when called with
  no explicit `series` is **not** required in M03 (the loader is M09); here the
  method takes an explicit `series`.

#### In-place parameter-update hook (FlexParameterize 2-way, §5 / R10)

FlexParameterize mutates a live model in place: it writes fitted parameter
values back onto the existing unit blocks. **flex-pse never deletes model
components** (blocks, Vars, Params, constraints): anything else on the model
that referenced a deleted component — an aggregated power constraint, an
expanded arc — would silently keep the stale reference, and every referencing
constraint/expression would have to be found and rebuilt. A built model is
updated only by mutating parameter values on existing components or by
adding/deactivating constraints. (This supersedes the earlier
delete-and-rewire `replace_unit` design.)

The mechanism is `OpsBlockData.update_parameters(values)`:

- Takes a mapping of registered process-parameter name → new value and calls
  `set_value` on the live (mutable) `Param`/`Var`, so every constraint that
  references it sees the new value with no rebuild. An unregistered name →
  `FlexConfigError` naming the field and listing what is registered.
- Surrogate parameterization (M10) works the same way: units are built with
  their surrogate structure in place and FlexParameterize updates the fitted
  parameter values.

Power Vars are base-provided, created when the unit declares it consumes that
kind (01_architecture §3.2):

```python
def declare_power(self, kind="electrical"):   # method name: implementer's choice
    """Create power_electrical[t] / power_thermal[t] (kW), register it, return it."""
```

Creates `Var(tb.time_points, initialize=0.0, units=pyunits.kW, doc="Electrical
draw of the unit")`, attached via `setattr` under
`flexcore.nomenclature.POWER_ELECTRICAL` (resp. `POWER_THERMAL`), then calls
`register_power`. No bounds at the base (implementer's choice).

Time access: the `flowsheet()` chain arrives with PlantBlock in M09. Interim
(implementer's choice, clearly marked): `_find_time_block()` searches
`self.model().component_objects(TimeBlock)` for exactly one instance; zero or
several → `FlexConfigError` ("build a TimeBlock on the model first").

```python
@classmethod
def from_config(cls, cfg: UnitConfig, **kwargs):
    raise NotImplementedError("Config-driven construction lands in M09.")
```

A unit is built from a `UnitConfig` (§3.2); whole-model construction is
`flexops.build_model(model_config)` (§2.3), also built in M09. `from_config`
stays a stub here so the signature is pinned for the milestones that reference it.

### flexops/properties/simple_aqueous.py (01_architecture §3.7)

Minimal `PhysicalParameterBlock`/`StateBlock` pair, structurally modeled on
WaterTAP `prop_ZO`; all IDAES bases via `flexcore.compat.idaes`.

- `@declare_process_block_class("SimpleAqueousFlow")` →
  `SimpleAqueousFlowData(PhysicalParameterBlock)`. CONFIG: `fixed_density`
  (bool, default `True`) and `density` (default `1000 * pyunits.kg/pyunits.m**3`
  — implementer's choice). `build()` sets the state-block class and, when
  `fixed_density`, a `dens_mass` Param. `define_metadata` classmethod declares
  `flow_vol` as supported and sets default units for all five base quantities
  (time=hr, length=m, mass=kg, temperature=K, amount=mol — exact bases are
  implementer's choice, but IDAES requires all five).
- State side: `_SimpleAqueousStateBlock(StateBlock)` +
  `SimpleAqueousStateBlockData(StateBlockData)` with one state Var `flow_vol`
  (`units=pyunits.m**3/pyunits.hr`, `initialize=1.0`, `doc="Volumetric
  flowrate"`); `define_state_vars()` returns `{"flow_vol": self.flow_vol}`.
  Optional pressure/temperature are omitted in v0 (smallest choice). Copy the
  prop_ZO skeleton for the required fix/unfix-state hooks and delete what a
  flow-only package doesn't need.

## Pitfalls

1. **`UnitModelBlockData` on a bare `ConcreteModel`.** IDAES resolves
   `dynamic=useDefault` by asking for a parent flowsheet. Explicit
   `dynamic=False, has_holdup=False` defaults avoid the lookup on current
   IDAES; if your version still demands a flowsheet in `build()`, override the
   offending resolution with the smallest working variant and flag the
   deviation in the PR (conventions §9).
2. **Registries as Pyomo components.** `_io_registry` is a plain
   underscore-prefixed attribute; non-underscore assignment risks Pyomo's
   `Block.__setattr__` machinery interfering.
3. **Missing `doc=`.** The M04 harness and M14 docs generator require non-empty
   `doc` on registered variables. Set `doc=` on every Var/Constraint, starting now.
4. **Hand-typing `"power_electrical"`.** Always import from
   `flexcore.nomenclature`; the literal string anywhere else is review-blocking.
5. **pydantic v1 idioms.** Use `model_validate`/`model_dump_json`/
   `model_json_schema`; no `parse_obj`, no `class Config`.
6. **Persisting a ConfigDict / passing raw dicts through layers** — conventions
   §4. `flexops_config` accepts only a validated `ModelConfig`.
7. **JSON Schema churn.** Dump with `sort_keys=True, indent=2` or the checked-in
   schema diffs on every regeneration.
8. **`define_metadata` omissions** cause obscure IDAES errors — set all five
   default units even though only volume/time matter here.
9. **YAML "Norway problem".** Bare `no`/`on`/`yes` scalars parse as booleans.
   Load with `yaml.safe_load` and let the dumper quote ambiguous scalars; never
   `yaml.load` untrusted input. Pydantic coercion + a typed schema is the guard.
10. **`ModelConfig` exactly-one-of `network`/`plant`.** A config with both, or
    neither, must fail validation with a clear field-path message — add the
    model-level validator; don't leave it to build time.
11. **UC/external-dispatch as *logic*, not config.** M03 ships only the validated
    `unit_commitment` config slot and the base `set_external_dispatch` hook.
    Building UC constraints, or wiring the dispatch source from config, is M08 —
    do not build ahead (conventions §9). Model updates go through
    `OpsBlockData.update_parameters` (in-place `set_value` on registered
    mutable params) — never delete-and-rebuild components; FlexParameterize
    drives it in M10.

## Tests

All `@pytest.mark.unit` (no solver; DoF checks count, they don't solve).

`src/flexops/tests/core/test_ops_block.py` — defines `DummyOps` in the test
module: registers `flow_in[t]`/`flow_out[t]` (m³/hr; input/output), a mutable
`energy_intensity` Param (kWh/m³) as a regressable process parameter, calls
`declare_power("electrical")`, and adds constraints
`flow_out[t] == 0.9 * flow_in[t]` and
`power_electrical[t] == pyunits.convert(energy_intensity * flow_in[t], pyunits.kW)`.

- `test_dummy_ops_builds` — on `ConcreteModel` + 4-point TimeBlock;
  `power_electrical` exists, indexed by `time_points`, in kW.
- `test_registration_records` — registry holds 2 `IOVariableRecord` (roles
  input/output, `time_indexed=True`, non-empty units strings), 1
  `ParameterRecord` (`regressable=True`), 1 `PowerRecord`
  (`kind == "electrical"`, `name == POWER_ELECTRICAL`).
- `test_iter_io_registry_finds_dummy` — yields exactly one `(block, registry)`
  pair; the block is the DummyOps instance.
- `test_units_consistent` — `assert_units_consistent(unit)` (via compat).
- `test_dof_zero_when_inputs_fixed` — fix `flow_in[t]` ∀t;
  `degrees_of_freedom(m) == 0` (via compat).
- `test_bad_role_raises` / `test_bad_kind_raises` — `FlexConfigError`.
- `test_no_time_block_raises` — DummyOps on a TimeBlock-less model → `FlexConfigError`.
- `test_from_config_not_implemented` — message mentions M09.
- `test_set_external_dispatch_removes_dof` — build DummyOps, fix `flow_in[t]` ∀t
  so `degrees_of_freedom(m) == 0` with the balance/energy constraints; then
  `unfix` an added controllable var (or use a variant where a controllable var is
  free) and confirm `set_external_dispatch(var, series)` fixes `var[t]` to
  `series[t]` for every `t` and drops the model's degrees of freedom by
  `n_points` (`var[t].fixed` is `True`, values match via `pytest.approx`). A
  misaligned/short `series` → `FlexConfigError`.
- `test_update_parameters_in_place` — build DummyOps, capture a constraint that
  references the registered `energy_intensity` Param, call
  `update_parameters({"energy_intensity": <new>})`, and confirm the *same*
  constraint object evaluates with the new value (no component deleted or
  rebuilt); an unregistered name → `FlexConfigError`.

`src/flexops/tests/properties/test_simple_aqueous.py`:
- `test_build_parameter_and_state_block` — `build_state_block([0])` works;
  `flow_vol` exists with m³/hr units.
- `test_define_state_vars` — dict has exactly the key `"flow_vol"`.
- `test_fixed_density_param` — `dens_mass` ≈ 1000 kg/m³
  (`pytest.approx(1000, rel=1e-6)`).
- `test_units_consistent` — on the state block.

`src/flexcore/tests/config/test_schema.py` — build a representative full
`ModelConfig` fixture (a `TimeConfig`, a `CostingConfig` with a `dr` slot, and a
`PlantConfig` holding a couple of `UnitConfig`s — one with a linear
`SurrogateSpec`, one with an `external_dispatch` and a non-default
`unit_commitment`):

- `test_model_config_yaml_roundtrip` — full fixture →
  `dump_model_config`(tmp_path, `.yaml`) → `load_model_config` → equal via
  `model_dump()`. (YAML is the canonical format.)
- `test_model_config_json_roundtrip` — same fixture through a `.json` target →
  `load_model_config` → equal via `model_dump()`. (JSON is also accepted.)
- `test_invalid_role_names_field_path` — a `UnitConfig` IO spec with
  `role="both"` → error text contains the full field path (e.g.
  `plant.units.tank.io_variables.0.role`).
- `test_missing_schema_version_raises` — a config without `schema_version` →
  `FlexConfigError` from load.
- `test_schema_version_too_new_raises` — `schema_version` greater than
  `CURRENT_SCHEMA_VERSION` → `FlexConfigError`.
- `test_network_or_plant_exactly_one` — a `ModelConfig` with both `network` and
  `plant`, and one with neither, each fail validation; exactly one passes.
- `test_unknown_key_rejected` — `extra="forbid"` enforced on a nested model.
- `test_exported_schema_up_to_date` — in-memory `ModelConfig.model_json_schema()`
  equals the checked-in `schemas/model_config.schema.json`; failure message says
  "re-run export_json_schemas and commit".

## Documentation tasks

- `docs/explanation/config_schema.md` — narrative of R3 (pydantic is the schema
  authority; **YAML** canonical on disk, JSON accepted; versioned by
  `schema_version`; exported JSON Schema in `config/schemas/`; the whole
  model+run builds from one config via `flexops.build_model` in M09; the
  FlexParameterize↔FlexOps seam), covering the full model set
  (`IOVariableSpec`…`ModelConfig`) with a
  `<!-- TODO(M14): .. flexops-config-table:: flexcore.config.schema.ModelConfig -->`
  marker where the auto-rendered field table goes (directive arrives in M14).
- `docs/reference/flexcore/index.rst` — autosummary for `flexcore.config.schema`,
  `flexcore.config.io`, `flexcore.nomenclature`, `flexcore.exceptions`.
- `docs/reference/flexops/core.rst` — add `ops_block` and `registration`.
- New `docs/reference/flexops/properties.rst` — `SimpleAqueousFlow` (plain
  autosummary; `flexops-unit-tables` is for unit models, M04+).
- Start `docs/explanation/energy_nomenclature.md` skeleton restating §4 rules.
- CHANGELOG: OpsBlock base + registration API + external-dispatch/replace hooks +
  unit-commitment config slot, SimpleAqueousFlow, full config schema v1 (YAML
  canonical).

## Definition of Done

- [ ] `fo.SimpleAqueousFlow(fixed_density=True)` constructs (API-freeze line)
- [ ] DummyOps builds on `ConcreteModel` + TimeBlock; registers 2 IO vars, 1
      parameter, electrical energy; `iter_io_registry` finds them
- [ ] `assert_units_consistent` passes; DoF == 0 with inputs fixed
- [ ] `set_external_dispatch(var, series)` fixes the var to the series and removes
      its DOF; misaligned series errors
- [ ] `update_parameters(values)` mutates registered params in place (existing
      constraints see the new value; nothing deleted); unknown name errors
- [ ] Full config schema (`IOVariableSpec`…`ModelConfig`, with `UnitConfig`
      carrying `unit_commitment` + `external_dispatch`, `CostingConfig` with a
      `dr` slot) implemented; `ModelConfig` is the top-level artifact with
      exactly-one-of `network`/`plant`
- [ ] `ModelConfig` **YAML** round-trip *and* JSON round-trip green; invalid configs error with field path; `schema_version` missing/too-new errors
- [ ] JSON Schema exported to `src/flexcore/config/schemas/`, checked in, up-to-date test green
- [ ] `from_config(UnitConfig)` raises `NotImplementedError` referencing M09;
      whole-model `flexops.build_model` noted as M09
- [ ] No literal `"power_electrical"`/`"power_thermal"` outside `flexcore/nomenclature.py`
- [ ] `pytest -m unit` green; `lint-imports` passes
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W --keep-going -b html docs docs/_build` clean
- [ ] plus the generic DoD in CLAUDE.md
