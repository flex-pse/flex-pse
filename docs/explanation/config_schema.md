# The config schema

flex-pse is **config-driven**. A whole model, its time horizon, property
packages, costing, plants, units, operating logic and fixed variables, can be
written as one version-controlled JSON file, the **spec**, and built from it.
Building in Python stays fully supported, and the two meet in a loop:

```text
code ──emit_model──► spec (JSON) ──build_model──► model
  ▲                                                 │
  └──────────── same model, compared ───────────────┘
```

{func}`~flexops.core.emit.emit_model` reads the spec a live model was, or
could have been, built from. {func}`~flexops.core.build.build_model` turns a
spec back into a model. A model that survives the loop unchanged is fully
described by its file: it can be reviewed, diffed, rebuilt somewhere else and
changed without touching code. {doc}`../how_to/round_trip_a_model` walks
through the loop end to end.

## The spec

A spec ({class}`~flexcore.config.spec.FlowsheetSpec`) is a header (the
`schema_version`, the model `name`, and the fix policy described below) and one
**flat list of elements**. Each element has a `kind`. The rules:

- **Names are full component paths.** A unit named `"waterfacility.tank"` is
  built as `model.waterfacility.tank`, so its parent, the plant
  `"waterfacility"`, must be an element too.
- **Elements refer to each other by name.** A connection goes to
  `"waterfacility.plant.inlet"`; a `status` element names the unit
  `"plant.ro"` (every member of an indexed unit) or `"plant.ro[0]"` (one).
- **File order is irrelevant.** The build sorts elements by kind, depth and
  name. Shuffling the list builds the same model, component for component.

So this scrambled spec is valid. It is the frozen API example, with the
objective listed before the units it optimizes and the plant after the units
inside it:

```json
{
  "schema_version": "0.1.1",
  "name": "waterfacility",
  "elements": [
    {"kind": "unit", "name": "waterfacility.plant",
     "unit_model_class": "ConstantEnergyIntensityModel",
     "construction_options": {"energy_intensity": {"value": 0.5, "units": "kWh/m^3"}}},
    {"kind": "objective", "name": "objective", "expression": "cost"},
    {"kind": "unit", "name": "waterfacility.tank", "unit_model_class": "Tank",
     "connections": [{"port": "outlet", "to": "waterfacility.plant.inlet"}]},
    {"kind": "costing", "name": "costing", "tariff_source": "tariff.json",
     "dr": {"events_source": "dr_events.json"}},
    {"kind": "plant", "name": "waterfacility"},
    {"kind": "time", "start_date": "2025-01-01", "end_date": "2025-01-30",
     "time_step": "15 min"},
    {"kind": "property_package", "name": "properties",
     "property_class": "SimpleAqueousFlow"}
  ]
}
```

A quantity with units is data, not code: `{"value": 0.5, "units": "kWh/m^3"}`,
or the string `"15 min"` for the time step. Units are written as Pyomo prints
them, with `**` changed to `^`.

## Assembly order

`build_model` runs eight stages in a fixed order
({data}`~flexops.core.stages.STAGES`): declare, topology, surrogates,
ramping, logic, state, extensions, costing. Each kind belongs to one
stage, and is either **mutable** (it can be changed on a built model) or not:

| Kind | Stage | Mutable |
| --- | --- | --- |
| `time` | declare | no |
| `property_package` | declare | no |
| `costing` | declare | no |
| `network` | topology | no |
| `plant` | topology | no |
| `set` | topology | no |
| `unit` | topology | no |
| `surrogate` | surrogates | yes |
| `ramp` | ramping | no |
| `status` | logic | no |
| `startup_shutdown` | logic | no |
| `dwell` | logic | no |
| `bypass` | logic | no |
| `startup_delay` | logic | no |
| `conditional` | logic | no |
| `parallel_group` | logic | no |
| `resource` | resources | no |
| `dispatch` | state | yes |
| `fix` | state | yes |
| `unfix` | state | yes |
| `extension` | extensions | no |
| `objective` | costing | no |

Within the logic stage, kinds build in {data}`~flexops.core.stages.LOGIC_ORDER`:
status first, because startup, conditional and parallel logic hang off a
status binary, and logic between units last. The `resources` stage runs after
`logic`, since a balance sums what units and their logic registered, and
before `state`, so a `fix` can name a balance's variables.
**Costing is processed last because `cost_process` aggregates whatever is
registered when it runs**: every power term a unit, a surrogate or an extension
registers has to exist by then.

## Mutable and immutable kinds

A kind is mutable when changing it on a solved, costed model leaves the
costing and the objective valid. Swapping a surrogate replaces a relation in
place, and fixing, unfixing or dispatching a variable changes only its value or
fixed flag. {func}`~flexops.core.stages.apply_spec` applies mutable elements to a
built model, and refuses an immutable one before changing anything.

Everything else changes the model's structure, so it is changed in the spec and
rebuilt. Swapping a unit, for example, means editing its `unit_model_class` and
calling `build_model` again. flex-pse never deletes a component from a built
model, since anything that captured it would keep a stale reference.

A new element kind is classified the same way: it gets a stage and a mutability
flag in {data}`~flexcore.config.spec.KINDS`.

## Data

Short lists sit inline. A long one, such as a year of prices, a dispatch series
or a fixed demand profile, can live beside the spec:

```json
{"kind": "fix", "variable": "plant.product.outlet_state.flow_vol_phase[*,Liq]",
 "value": {"$source": "data/demand.json.gz"}}
```

A `$source` path is relative to the spec file and names a `.json` or gzipped
`.json.gz` file. `emit_model(data_dir=..., inline_limit=2000)` writes every list
longer than `inline_limit` this way, and `relative_to=` writes paths relative to
where the spec will be saved.

A unit that takes several property packages receives them as
`{"$package": "biogas"}` construction options, which name a `property_package`
element.

## Fixing variables

A `fix` element names **any** variable by its path from the model root and
gives its values. An `unfix` element frees one, such as a parameter a unit
fixes by default:

```json
{"kind": "fix", "variable": "plant.battery.soc[0]", "value": 0.5},
{"kind": "fix", "variable": "plant.pretreatment[*].status", "value": 1},
{"kind": "fix", "variable": "plant.intake.flow_in", "value": [120, 120, 95, 80], "units": "m^3/hr"},
{"kind": "unfix", "variable": "plant.ro[*].recovery"}
```

**Paths** are Pyomo `ComponentUID` strings. `plant.x` is a whole variable,
`plant.x[3]` one point, `plant.ro[*].x` the variable on every member of `ro`,
and `plant.u.flow_vol_phase[*,Liq]` every time point of the `Liq` phase. A
path that matches nothing is an error that suggests the closest variable names;
a path that matches a `Param` is an error too, since parameters are set, not
fixed.

**Values** come in four forms: one number for every matched point; a list with
one number per time point (or per entry of `at`); a `{time index: value}`
mapping; or a `$source` file holding a list or mapping. `at` restricts the fix
to some time indices. `units` converts the values into each variable's own
units. The time index is the variable's first index, when that is the
TimeBlock's.

**Conflicts are errors.** Since file order is irrelevant, there is no "the later
fix wins". Two `fix` or `dispatch` elements that set one point to different
values, or a `fix` and an `unfix` of the same point, stop the build and name the
elements involved. Two elements writing the same value are fine.

**Bounds and domain.** Pyomo itself only logs a warning when a variable is fixed
outside its bounds, or a binary to 0.5. The state stage checks every value
first, and what happens next is the **policy**: the element's `on_violation`,
or the spec's `fix_on_violation` when the element sets none.

| Policy | Outside bounds | Outside domain (e.g. a binary to 0.5) | Valid points |
| --- | --- | --- | --- |
| `"error"` (default) | Error | Error | Not fixed: nothing in the stage is applied |
| `"warn"` | Fixed anyway, with a `FlexFixWarning` | Error | Fixed |
| `"skip"` | Left free, with a `FlexFixWarning` | Left free, with a `FlexFixWarning` | Fixed |

A bound check passes within an absolute tolerance: the element's `tolerance`,
or the spec's `fix_tolerance` (`1e-9`). A skipped point is left free for the
solver, never clamped to its bound, since a clamped value is one nobody wrote.
The state stage is all or nothing: every fix, unfix and dispatch is planned and
checked before any of them changes the model. Every warned or skipped point is
also logged at `WARNING`, and listed in the model's `_flex_fix_report`
({class}`~flexops.core.fixing.FixReport`).

`"warn"` and `"skip"` let a run go on with commands that differ from what the
spec says. A spec that drives equipment with nobody reading the log should keep
the default `"error"`, or escalate the warning so it stops the run:
`warnings.simplefilter("error", FlexFixWarning)`.

`emit_model` writes `.fix()` and `.unfix()` calls made in code as these
elements. A variable fixed to one value over the whole horizon becomes one
number, the same fix on every member of an indexed unit becomes one `[*]`
element, and a fix on some time points uses `at`. Bounds changed in code
(`setlb`, `setub`) can't be written yet and raise a `FlexEmitWarning`.

## Logic, ramps and extensions

Each `flexops.logic` helper (except `relax` and `unrelax`, which are solve
strategy) has one element kind: `status`, `startup_shutdown`, `dwell`, `bypass`,
`ramp`, `startup_delay`, `conditional` and `parallel_group`. An element names
its unit and never sits inside it. The spec checks that every unit named exists
and that nothing is set twice on the same unit. Two status rules are checked
when the model is built, since only the unit knows them: logic needing an on/off
status must find one on the unit (a battery with `unit_commitment.status` builds
its own), and a `status` element may not target a unit whose
`allows_status_element` is `False` (`Tank`, `BatteryModel`).

An `extension` element names a user function, as `"package.module:function"` or
an entry point in the `flexpse.extensions` group, with JSON options. Decorating a
function with `@flexops.extension` makes calling it record the element. Logic
helpers called inside an extension record nothing, because rebuilding re-runs
the extension. An objective with `"expression": "extension"` means an extension
adds the objective. A spec that uses an extension builds only where its package
is installed.

## What emit_model checks

`emit_model` rebuilds the spec it wrote and compares the two models. Anything the
spec can't express yet, such as a constraint added by hand, an objective no
extension made, or a surrogate object built without a spec, raises a
`FlexEmitWarning` that names it. Nothing is dropped silently. Things that would
change the model if left out (an option with no spec form, indexed members
configured differently, an arc no `{i}` template fits) raise a
`FlexConfigError`.

## Surrogates

A {class}`~flexcore.config.schema.SurrogateSpec`, carried by a `surrogate`
element, names a **predefined surrogate class**, one of
{class}`~flexcore.config.schema.SurrogateType`'s members, and an opaque `data`
mapping in the shape that class defines. The class (`flexops.surrogates`)
validates `data` and builds the Pyomo relationship. A type not yet implemented
raises `NotImplementedError` when the model is built (`quadratic`,
`exponential`, `arima` and `neural_network` are reserved names, see
{doc}`../reference/flexops/surrogates`).

The only implemented class, `multilinear`
({class}`~flexops.surrogates.multilinear.MultilinearSurrogate`), is a constant
plus a sum of `coefficient * (product of distinct declared inputs)`. A
coefficient key is a `*`-separated product of names from `input_variables`,
each at most once. The reserved key `intercept` is the constant term.

```json
{
  "kind": "surrogate",
  "unit": "plant.pump",
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

The declared units are the basis the relationship was fitted in. Each input
converts from the model's units into its declared ones before the coefficient
multiplies it, and the result converts into the target's units. A dimensionally
incompatible declaration raises a `FlexConfigError` instead of silently
rescaling.

Only relationships a unit registered with
{meth}`~flexops.core.ops_block.OpsBlockData.register_relation` can be swapped:
every constant-intensity energy relation, an RO skid's `split_definition`, a
tank's `level_definition`. Conservation constraints are never registered, so
they can never be swapped. {func}`~flexops.core.registration.iter_swapped_relations`
reports which registered relations have been swapped.

## Versioning

Every spec carries a mandatory `schema_version`, a semantic version string
({data}`~flexcore.config.spec.SCHEMA_VERSION`). On load, a missing, malformed
or too-new version raises a `FlexConfigError`, and an older one steps through
{data}`~flexcore.config.io.MIGRATIONS` before validation. Pydantic models in
`flexcore.config.spec` are the schema authority and reject undocumented keys.
Their exported JSON Schema is checked in under `src/flexcore/config/schemas/`,
so schema changes show up in review and other tools can validate against it.
Changing the default fix policy is a breaking change and needs a new version.

## The FlexParameterize and FlexOps seam

The spec is also the contract between FlexParameterize and FlexOps.
FlexParameterize changes a live FlexOps model in place, and
{func}`~flexops.core.emit.emit_model` writes that model out as a spec that
rebuilds it.

## Field reference

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.FlowsheetSpec
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.TimeElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.PropertyPackageElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.CostingElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.NetworkElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.PlantElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.SetElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.UnitElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.Connection
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.SurrogateElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.RampElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.StatusElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.StartupShutdownElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.DwellElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.BypassElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.StartupDelayElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.ConditionalElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.ParallelGroupElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.ResourceElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.DispatchElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.FixElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.UnfixElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.ExtensionElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.spec.ObjectiveElement
```

```{eval-rst}
.. flexops-config-table:: flexcore.config.schema.SurrogateSpec
```
