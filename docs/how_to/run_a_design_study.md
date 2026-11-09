# Run a design study

A design study sizes equipment once, for a plant that then operates across several representative months and years. You write one base flowsheet and describe how it varies.

## Write the study file

```json
{
  "schema_version": "0.1.0",
  "name": "pump_battery_solar",
  "base": "base_flowsheet.json",
  "years": [
    {"year": 2026, "represents": 5},
    {"year": 2031, "represents": 5}
  ],
  "months": [
    {"month": 1, "weight": 6},
    {"month": 7, "weight": 6}
  ],
  "changes": [
    {"element": "pv", "months": [1], "values": {"construction_options": {"capacity_factor": {"$source": "pv_jan.json"}}}},
    {"element": "pv", "months": [7], "values": {"construction_options": {"capacity_factor": {"$source": "pv_jul.json"}}}},
    {"element": "costing", "years": [2031], "values": {"energy_prices": {"electrical": {"value": 0.12}}}}
  ],
  "capacities": [
    {"name": "battery_energy", "variable": "battery.energy_capacity", "capital_cost": 300.0, "lifetime_years": 15, "upper": 1500.0},
    {"name": "battery_power", "variable": "battery.power_capacity", "capital_cost": 200.0, "lifetime_years": 15, "upper": 1500.0},
    {"name": "solar", "variable": "pv.capacity", "capital_cost": 1200.0, "lifetime_years": 25, "upper": 1500.0}
  ]
}
```

Then build, solve and report:

```python
from flexcore.config.design import load_design_spec
from flexcore.solvers import get_solver
from flexdesign import build_design_model, design_results

m = build_design_model(load_design_spec("study.json"))
get_solver(m).solve(m)
print(design_results(m).summary())
```

## Choose years and months

- **`years`** sets how many years are modeled. `represents` says how many calendar years each one stands in for, and the study length is the sum. Modeled years must be consecutive under it: 2026 representing 5 must be followed by 2031.
- **`months`** sets how many representative months each year runs. `weight` says how many months of the year each one stands in for, and weights must sum to 12.

The model holds one horizon per year and month, so its size is the product of the two. Start small: a 5-year, 4-month hourly study is 20 months of constraints.

To try a study quickly, set `"days": 1` on a month. Only the first days are modeled, and that bill is scaled up to the full month. Every series for that month must then have one value per modeled time step. A prorated month doesn't capture a whole month's peak, so use whole months for final sizes.

## Vary parameters by month and year

Everything that varies goes through `changes`. An entry names a base element and the values to deep-merge into it, and selects where it applies.

| To vary | Selector | Example |
|---|---|---|
| By month, every year | `months` | Solar forecast, seasonal prices |
| By year, every month | `years` | Price escalation, battery `soh` |
| One month of one year | `years` and `months` | A forecast that changes both ways |
| Everywhere | Neither | A study-wide replacement of a base value |

- Lists are replaced whole, and later entries win when two match the same field.
- To change a `fix` element, give it a `name` in the base spec.
- Dates come from the year and month, so changes can't edit `time`.
- Values are absolute, so a series that varies by both month and year needs a change for each combination.

## Prepare solar forecasts

A forecast is a capacity factor series: output per kW of nameplate, from 0 to 1, with one value per time step. It has to be normalized because output is `capacity_factor[t] * capacity`, which is what lets it scale as the size changes. A forecast for a 1 kW system is already in this form. Otherwise, divide by the system's rated kW.

```{warning}
Forecast files are matched by position, starting at the first time step of the month, and only their length is checked. January and July both have 744 hours, so swapping those files builds and solves with wrong results.
```

By default solar output cannot be curtailed, so all generation has to go somewhere. Limiting exports to zero with `GridConnection` can then make a model infeasible. Set `"curtailable": true` on the `GenericRenewables` unit to let the optimizer spill the excess.

## Prepare the base flowsheet

- **Free the starting state of storage.** Set `"fix_initial_state": false` on each battery and tank, since a study rejects any fixed initial state. Also set `"cyclic": true`, which requires each month to end at least as full as it started, so no month can drain free stored energy.
- **Set constructor limits at or above sizing bounds.** A decision's `upper` can't exceed its Var's own bound, and `battery.power_capacity` is bounded by the larger of `power_charge_max` and `power_discharge_max`.
- **Give every decision an `upper`.** With net metering, an unbounded capacity can make the problem unbounded. If a report flags a size as at its upper bound, the bound is probably driving the answer.
- **Set an export price if exports aren't credited at the import price.** Otherwise surplus solar earns the full import price.

## Work with the merged model

The merged model holds several time blocks, so helpers that look for a single one raise an error once horizons are merged:

- `set_external_dispatch`, `set_dispatch`, `add_stream_ports` and `declare_power`
- logic helpers
- `apply_spec` with `dispatch` elements
- `emit_model` and `model_differences` at the merged root

Apply these per horizon through `changes`, or to the standalone models from `build_horizons`. Fit parameters with `flexparameterize` before merging, since unit names gain a prefix afterward. The saved form of a merged model is its study file.
