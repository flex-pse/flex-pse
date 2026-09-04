# Parameterize a model from data

FlexParameterize turns plant data into a parameterized FlexOps model. The
pipeline has one shape and two endings:

**tabular data → tag aliasing → sufficiency validation → regression →
{apply to a live model | emit a config}**

The data source does not matter. A historian export, a spreadsheet saved as
CSV, a database query — anything that becomes a `pandas.DataFrame` works,
provided its columns are mapped to the right model aliases. No stage below
requires a historian connection.

The starting point is a **built** FlexOps model, because FlexOps is what
declares the containers: each unit registers its process IO variables and its
regressable parameters, and FlexParameterize reads that registry to know what
can be fitted (architecture §1, §5).

## 1. Map the data's columns onto model aliases

A {py:class}`~flexparameterize.tags.TagMap` renames source columns to
{term}`model alias`es — the dotted `plant.unit.variable` names of the model's
registered variables.

```python
from flexparameterize import TagMap, model_alias

tagmap = TagMap({
    "FT_0231.PV": model_alias(m.facility.pump.outlet_state.flow_vol_phase),
    "MTR_KW_04": model_alias(m.facility.pump.power_electrical),
})
print(tagmap.report_unmapped(raw))   # near misses, via difflib
aliased = tagmap.apply(raw)          # a renamed COPY; raw is untouched
```

Extra columns are legal — the sufficiency check decides what matters. Tag maps
are loadable from JSON, the same canonical format as the config
(`TagMap.from_file("tags.json")`).

## 2. Check that the data determines the fit

{py:func}`~flexparameterize.validate.check_sufficiency` walks the model's
registered IO pairs and reports, per pair, which columns are present, how many
non-null rows they carry, and whether the frame's index is a usable time axis
(v0 requires a `DatetimeIndex`). Together that is the
{term}`zero-degree-of-freedom regression` condition.

```python
from flexparameterize import check_sufficiency

report = check_sufficiency(m, aliased, m.time_block)
if not report.sufficient:
    print(report)          # names every missing column and what to do
```

It reports, it never raises — the caller decides. (`apply_to_model` is the one
caller that does raise, since it mutates a real model.)

## 3. Fit

```python
from flexparameterize import ConstantIntensityRegressor

regressor = ConstantIntensityRegressor().fit(
    aliased[[model_alias(m.facility.pump.outlet_state.flow_vol_phase)]],
    aliased[[model_alias(m.facility.pump.power_electrical)]],
)
regressor.coefficient      # kWh/m^3
regressor.metrics          # {"r2": ..., "rmse": ...}
```

The flow on the left is the unit's **outlet**, because that is what its own
energy relation is metered on: an intensity is energy per unit of *product*.
For a unit that passes flow straight through the two coincide, but for one with
a recovery or a loss they do not, and regressing against the wrong stream would
recover a coefficient that means something the model does not.
`apply_to_model` resolves the same basis for itself from the unit's registry.

**Choosing a regressor** — every regressor conforms to the
{py:class}`~flexparameterize.regression.base.Regressor` protocol: `fit(X, y)`
returns the fitted regressor itself, `to_fit_result()` returns the shared
{py:class}`~flexparameterize.regression.base.FitResult` (coefficients,
metrics, sample count, data window), and `to_surrogate_spec(**kwargs)` returns
the `SurrogateSpec`. `ConstantIntensityRegressor` is the one-input,
one-coefficient fit shown above; `LinearRegressor` fits an ordinary-least-
squares line against one or more input columns and emits a `multilinear`
`SurrogateSpec`:

```python
from flexparameterize import LinearRegressor

regressor = LinearRegressor().fit(
    aliased[["flow_out", "outlet_state.pressure"]],
    aliased[["power_electrical"]],
)
regressor.coefficients   # {"flow_out": ..., "outlet_state.pressure": ..., "intercept": ...}
spec = regressor.to_surrogate_spec(
    input_units={"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
    output_units="kW",
)
```

`LinearRegressor` needs the `[parameterize]` extra's `scikit-learn`
dependency: `pip install 'flex-pse[parameterize]'`. A fitted
`LinearRegressor`'s `SurrogateSpec` works identically to a hand-built one in
§4a/§4b below — no changes needed there.

`get_regressor(surrogate_type)` resolves a
{py:class}`~flexcore.config.schema.SurrogateType` (member or its string
value) to the regressor class that fits it — the config-driven lookup for a
spec's own provenance, not something `apply_to_model`'s fit-from-data path
calls itself.

## 4a. Apply the fit to the live model

```python
from flexparameterize import apply_to_model

report = apply_to_model(m, raw, tagmap)
print(report)              # what was fixed, what was swapped, DOF before/after
```

Every unit registering a regressable parameter is fitted and updated **in
place** — flex-pse never deletes a built component. A constant-intensity
relationship is written by fixing the unit's `energy_intensity` parameter (so
the model's degrees of freedom drop); a richer one deactivates the unit's
`power_electrical_relation` Constraint and attaches an equality built from the
fitted `SurrogateSpec`, on the same unit object, reusing the same registered IO
variables. Ports and arcs are untouched — there is nothing to reconnect.

## 4b. Or emit a config

```python
from flexcore.config.io import dump_model_config
from flexops.core.build import build_model
from flexparameterize import emit_model_config

cfg = emit_model_config(m.facility.pump, regressor, {"data_source": "2025 export"})
dump_model_config(cfg, "pump_fitted.json")
rebuilt = build_model(cfg)
```

The emitted config carries the unit's class name, its construction options
(the fitted coefficient among them), its IO variable specs, the
`SurrogateSpec`, and provenance: the fit's `n_samples`/`r2`/`rmse`, its
`data_window`, and the versions of `flex-pse`, `pyomo` and `pandas` read at
emit time. Pass `costing=` a real
{py:class}`~flexcore.config.schema.CostingConfig` if the emitted config is
meant to be solved — the default is a 0 USD/kWh placeholder, since a unit
carries no tariff of its own.

## The two directions agree

Both endings consume the same `SurrogateSpec` from the same fit, so the model
`apply_to_model` mutated and the model rebuilt from the emitted config describe
the same behaviour (architecture §5, decision R10). That invariant is asserted
by `test_apply_and_emit_agree`.

## Supplying a relationship you already know

A fit is one way to get a `SurrogateSpec`, not the only one. When a unit's
energy relationship is already known in closed form — a vendor curve, a
datasheet coefficient, a physics-derived expression — hand the spec over
directly. No data, no sufficiency check, and no regressor are involved for that
unit:

```python
from flexcore.config.schema import SurrogateSpec, SurrogateType

vendor = SurrogateSpec(
    surrogate_type=SurrogateType.MULTILINEAR,
    data={
        "input_variables": {"flow_out": "m^3/hr"},
        "output_variables": {"power_electrical": "kW"},
        "coefficients": {"flow_out": 0.38, "intercept": 12.0},
    },
    provenance={"source": "vendor_datasheet"},
)

apply_to_model(m, raw, tagmap, surrogates={"facility.skid": vendor})
emit_model_config(m.facility.skid, vendor, {"source": "vendor_datasheet"})
```

`surrogate_type` names a predefined surrogate class
({class}`~flexops.surrogates.multilinear.MultilinearSurrogate` here); `data` is
that class's own contract, validated when the surrogate is realized. A
coefficient key names the **term** it multiplies — a `*`-separated product of
distinct names from `input_variables`, plus the reserved `intercept` — and a
draw that depends on outlet flow *and* outlet pressure together adds their
cross term:

```python
multilinear = SurrogateSpec(
    surrogate_type=SurrogateType.MULTILINEAR,
    data={
        "input_variables": {"flow_out": "m^3/hr", "outlet_state.pressure": "Pa"},
        "output_variables": {"power_electrical": "kW"},
        "coefficients": {
            "intercept": 12.0,
            "flow_out": 0.38,
            "outlet_state.pressure": 1.1e-5,
            "flow_out*outlet_state.pressure": 2.3e-6,
        },
    },
)
apply_to_model(m, raw, tagmap, surrogates={"facility.skid": multilinear})
```

Naming pressure needs a property package built with `has_pressure=True`. Each
name in `input_variables`/`output_variables` is resolved on the unit — dotted
paths into a state block work — and declares the units the relationship was
fitted or written in, which need not match the model's own: every factor is
converted from its actual unit into its declared one before use, and the whole
body is converted into the registered target's own units. See [the config
schema](../explanation/config_schema.md) for the full contract, and for
`source`, which points a relationship at a JSON sidecar instead of inlining it.

The two paths mix freely across units in one call: units named in
`surrogates=` are skipped for fitting, and every other unit is still fit from
`data`/`tagmap` as usual. Provenance for a supplied spec documents its origin
instead of fit metrics — there are no `n_samples`, `r2` or `rmse` for an
algebraic form that was never fitted.

## Swapping a relationship other than the energy draw

A unit can register more than one relationship as swappable — not only its
energy draw. A reverse-osmosis skid registers `split_definition` (its
recovery/flux relation, determining `permeate`); a tank registers
`level_definition` (its fill geometry). Name the relation explicitly by
passing a `{relation_name: spec}` mapping instead of a bare spec:

```python
recovery = SurrogateSpec(
    surrogate_type=SurrogateType.MULTILINEAR,
    data={
        "input_variables": {"feed": "m^3/hr", "inlet_state.pressure": "Pa"},
        "output_variables": {"permeate": "m^3/hr"},
        "coefficients": {
            "intercept": 0.3,
            "feed": 0.01,
            "inlet_state.pressure": 1e-6,
            "feed*inlet_state.pressure": 1e-7,
        },
    },
)
apply_to_model(m, raw, tagmap, surrogates={"facility.ro": {"split_definition": recovery}})
```

The two forms of `surrogates=` mix per unit in one call: a plain spec attaches
the unit's own energy relation, a mapping attaches one or more of its other
registered relations. Only a relationship the unit registered via
`register_relation` can be named this way — its mass/energy balance was never
registered and so can never be swapped; see
[the config schema](../explanation/config_schema.md) for which relations each
unit registers.
