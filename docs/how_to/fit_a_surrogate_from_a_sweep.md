# Fit a surrogate from a parameter sweep

When a detailed process model (an IDAES or WaterTAP flowsheet, or any Pyomo
model) is too large to embed in a flex-pse operations model, sweep it over the
inputs that matter, fit the output to those inputs, and embed the fit as a
surrogate. `flexparameterize.sweep` does this with watertap-org's
`parameter-sweep` package for the sampling and solving, and with flex-pse's own
regressors for the fit.

Install the extras:

```bash
pip install 'flex-pse[parameterize,sweep]'
```

## 1. Describe the sweep

A {py:class}`~flexparameterize.config.ParameterizeConfig` holds everything the
sweep and the fit need. Nothing is chosen at run time, so the same config and
package versions give the same spec.

```python
from flexparameterize import ParameterizeConfig

config = ParameterizeConfig(
    inputs={
        "feed": {"path": "fs.feed.flow_vol", "units": "m^3/s", "range": (5e-4, 1.2e-3)},
        "recovery": {"path": "fs.ro.recovery", "units": "dimensionless", "range": (0.35, 0.55)},
    },
    context={"fs.feed.conc_mass_nacl": 35.0},
    sampler={"type": "lhs", "n": 200, "seed": 20261118},
    solver={"name": "ipopt", "options": {"tol": 1e-8}},
    outputs={
        "power": {
            "path": "fs.total_pump_work",
            "units": "W",
            "unit": "facility.ro",
            "relation": "power_electrical_relation",
            "output_name": "power_electrical",
            "output_units": "kW",
            "input_names": {"feed": "flow_in", "recovery": "recovery"},
            "terms": ["feed", "recovery", "feed*recovery"],
        }
    },
    max_failed_fraction=0.05,
    holdout={"k": 5, "r": 0},
    acceptance={"max_rel_error": 0.05, "max_rel_rmse": 0.02},
)
```

- `inputs` are model components the sweep sets at each point. Each must be a
  fixed Var or a mutable Param of the built model, so the model stays square,
  and its `units` must match the model.
- `context` fixes other components after every build, in their model units.
- `outputs` names the model output to record, the flex-pse unit and relation
  its fit replaces, and the terms to fit. `output_name` is the unit's variable
  that the relation determines, and `input_names` maps each config input to the
  unit's variable that the surrogate reads. `relation` and `output_name`
  default to the electrical power relation and its variable, as written here.
- `sampler` is a grid (`n` points per input) or a Latin hypercube (`n` points
  in total, drawn from `seed`).

## 2. Run the sweep

Supply three functions: one that builds the model, one that initializes it,
and one that solves it with the solver built from `config.solver` and returns
the solver results.

```python
from flexparameterize.sweep import run_sweep

data = run_sweep(config, build_model, initialize, optimize)
data.to_csv("ro_sweep.csv", index=False)
```

The returned frame has one row per point: `index`, the inputs, the outputs and
`solve_successful`. The model is rebuilt and re-initialized before every
point, so no result depends on the point solved before it. Latin hypercube
points are sorted by the first input, so row order is fixed too.

## 3. Fit and embed

```python
from flexparameterize import apply_to_model
from flexparameterize.sweep import fit_sweep

specs = fit_sweep(config, data)
apply_to_model(m, plant_data, tagmap, surrogates=specs)
```

`fit_sweep`:

1. Holds out the rows at positions `p` with `p % k == r`. These are chosen
   before failed rows are dropped, so a failure never moves another row
   between the training and holdout sets.
2. Raises if more than `max_failed_fraction` of the points failed. Otherwise
   it drops the failed points.
3. Converts each output to `output_units`, builds the term columns (a
   product for `"feed*recovery"`), and fits them with the regressor for the
   output's surrogate type.
4. Checks the holdout maximum relative error and relative RMSE (RMSE over the
   mean output) against `acceptance`, and the holdout R² too when `r2_min` is
   set. Both limits default to 10%. A miss raises. Nothing is adjusted to make
   it pass.

Each spec's `provenance` records the config hash, a hash of the data, the
installed package versions, the input range the fit was trained on, and the
training and holdout metrics. The fit is only valid inside that input range.

## Checking and refining a fit

Two more building blocks support checking a fit, and refining it in a trust
region:

- `run_sweep(..., bounds={"feed": (6e-4, 8e-4)})` resamples a box that lies
  inside the config ranges.
- `solve_point(config, build_model, initialize, optimize, {"feed": 7e-4,
  "recovery": 0.5})` solves one point through the same path as a sweep. It
  returns the row that point would get inside a sweep, so you can compare an
  optimum of the flex-pse model with the detailed model.

Rows from `run_sweep` and `solve_point` can be concatenated and passed to
`fit_sweep` together.
