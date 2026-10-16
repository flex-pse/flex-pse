# Round-trip a model

A model you build in Python can be written out as a spec and built back from
it. `emit_model` reads the spec from the live model, `dump_spec` saves it as
JSON, and `build_model` turns the file back into the same model. This page walks
through that loop for a two-unit plant with one piece of operating logic, one
extension and one fixed variable. A test runs every snippet on each pull
request.

## 1. Put your extensions in a module

An extension is a function that adds constraints or an objective. The spec only
names it, as `"module:function"`, so it has to live in a module Python can
import. Here it is `plant_extensions.py`, next to your script.

```python
# file: plant_extensions.py
import pyomo.environ as pyo

import flexops


@flexops.extension
def end_full(model, *, tank: str, fraction: float):
    """End the horizon with the tank at least ``fraction`` full."""
    block = model.find_component(tank)
    last = model.time_block.time_index.last()
    block.end_full = pyo.Constraint(
        expr=block.volume[last] >= fraction * block.capacity
    )
```

## 2. Build the model in code

A tank feeds a pump. The pump gets an on/off status, the extension holds the
tank's final level, and the pump is fixed off at the first step.

```python
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc

import flexops as fo
from flexops import logic
from flexops.costing import currency_units

from plant_extensions import end_full

m = pyo.ConcreteModel(name="site")
m.time_block = fo.TimeBlock(
    start_date="2025-01-01", end_date="2025-01-02", time_step=1 * pyunits.hr
)
m.properties = fo.SimpleAqueousFlow()
m.costing = fo.FlexCosting(
    time_block=m.time_block,
    energy_prices={"electrical": 0.1 * currency_units("USD") / pyunits.kWh},
)
m.site = fo.PlantBlock(time_block=m.time_block)
m.site.tank = fo.Tank(property_package=m.properties)
m.site.pump = fo.ConstantEnergyIntensityModel(
    property_package=m.properties,
    energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    costing_package=m.costing,
)
m.site.tank_to_pump = Arc(source=m.site.tank.outlet, destination=m.site.pump.inlet)
pyo.TransformationFactory("network.expand_arcs").apply_to(m)

flow = pyunits.m**3 / pyunits.hr
logic.add_status(m.site.pump, m.site.pump.flow_in, 0 * flow, 10 * flow)
end_full(m, tank="site.tank", fraction=0.5)
m.site.pump.status[0].fix(0)

m.costing.cost_process()
m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
```

## 3. Write the spec

`emit_model` returns the spec and checks it: it rebuilds the spec and warns with
`FlexEmitWarning` about anything that came out different. Turning that warning
into an error makes the round trip a hard requirement.

```python
import warnings

from flexcore.config.io import dump_spec
from flexcore.exceptions import FlexEmitWarning

with warnings.catch_warnings():
    warnings.simplefilter("error", FlexEmitWarning)
    spec = fo.emit_model(m)

dump_spec(spec, "site.json")
```

`site.json` lists the time horizon, the package, the costing block, the plant,
both units with their arc, a `status` element, an `extension` element naming
`plant_extensions:end_full`, the objective, and this element for the fix:

```json
{"kind": "fix", "variable": "site.pump.status", "value": [0.0], "at": [0]}
```

## 4. Build it back

```python
from flexcore.config.io import load_spec
from flexops.core.compare import assert_models_equivalent

rebuilt = fo.build_model(load_spec("site.json"))

assert_models_equivalent(m, rebuilt)
```

`assert_models_equivalent` compares the two models by component path: the same
variables (with their bounds, domains and fixed values), parameters,
constraints, arcs and objective. The rebuilt model also carries
`_flex_fix_report`, the record of what its `fix` and `unfix` elements did.

## Changing a built model

Fix, unfix, dispatch and surrogate elements can be applied to a model that is
already built, without rebuilding it. The batch is checked as a whole before
anything changes, and `emit_model` writes the new values.

```python
report = fo.apply_spec(
    rebuilt, [{"kind": "fix", "variable": "site.pump.status[1]", "value": 0}]
)

assert report.fixed == ["site.pump.status[1]"]
```

See {doc}`../explanation/config_schema` for every element kind, and for the
policy that decides what happens when a fixed value is outside its variable's
bounds.
