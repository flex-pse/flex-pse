# Build a plant

A flex-pse model is built in one order: **TimeBlock → property package →
costing → plant → units → arcs → `cost_process()` → objective**. Costing comes
*before* the units on purpose — `FlexCosting` aggregates power at
`cost_process()` time, so it may be constructed while the plant is still empty.
Plant totals are deferred for the same reason.

The frozen [`examples/api_freeze.py`](https://github.com/arao53/flexPSE/blob/main/examples/api_freeze.py)
is exactly this sequence and is guarded by a component test: from M09 on, a
change that breaks it is a breaking change.

## The imperative path

```python
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
import flexops as fo

m = pyo.ConcreteModel()
m.time_block = fo.TimeBlock(
    start_date="2025-01-01", end_date="2025-01-30", time_step=15 * pyunits.min
)
m.properties = fo.SimpleAqueousFlow()
m.costing = fo.FlexCosting(
    time_block=m.time_block,
    tariff_file="tariff.json",
    dr_event_file="dr_events.json",
)

# A PlantBlock is a collection of UNITS. It takes the TimeBlock explicitly;
# omit it only when the model carries exactly one.
m.waterfacility = fo.PlantBlock(time_block=m.time_block)
m.waterfacility.tank = fo.Tank(property_package=m.properties)
m.waterfacility.plant = fo.ConstantEnergyIntensityModel(
    property_package=m.properties,
    energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    costing_package=m.costing,
)
m.waterfacility.tank_to_plant = Arc(
    source=m.waterfacility.tank.outlet, destination=m.waterfacility.plant.inlet
)
m.waterfacility.battery = fo.BatteryModel(
    capacity=1 * pyunits.kWh, costing_package=m.costing
)

m.costing.cost_process()
m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
```

Arc expansion and the solve stay explicit — no library code applies the
transformation implicitly:

```python
from flexcore.solvers import get_solver

pyo.TransformationFactory("network.expand_arcs").apply_to(m)
results = get_solver(model=m).solve(m)
```

After solving, read the **reported** cost — never the raw objective — from
`m.costing.report_cost(m)`, which returns a categorized breakdown (operating vs.
capital, each itemized).

## The config-driven twin

Nothing essential lives only in imperative code: the same model is built from
one version-controlled JSON file (architecture §2.3, R3). The config twin of the
script above is
[`examples/api_freeze_config.json`](https://github.com/arao53/flexPSE/blob/main/examples/api_freeze_config.json),
and a component test holds the two to the same solved objective.

```python
from flexcore.config.io import load_model_config

m = fo.build_model(load_model_config("api_freeze_config.json"))
```

`build_model` constructs the TimeBlock, the property package, the `FlexCosting`
block, the plant/unit tree, its arcs, any declared external (DERMS) dispatch,
and the objective. Two conventions matter when writing one of these files:

- **Units are data.** A config cannot carry a Pyomo expression, so a
  units-carrying quantity is written `{"value": 0.5, "units": "kWh/m^3"}` —
  or, for `time.time_step`, the string `"15 min"`. Anything else in
  `construction_options` is passed to the unit's constructor unchanged.
- **Runtime-only options are supplied by the builder.** `property_package` and
  `costing_package` are live Pyomo objects, not serializable, so `build_model`
  wires them into every unit; do not list them in `construction_options`.

A malformed config raises `FlexConfigError` naming the exact field path (e.g.
`plant.units.surrogate.unit_model_class`), with the underlying pydantic
`ValidationError` as its cause.

## More than one plant: NetworkBlock

A plant containing plants is a `NetworkBlock`, not a nested `PlantBlock` (R7):
`PlantBlock` composes units, `NetworkBlock` composes plants. Its totals are the
sums of its child plants' totals, so every unit is counted exactly once.

```python
m.campus = fo.NetworkBlock(time_block=m.time_block)
m.campus.north = fo.PlantBlock(time_block=m.time_block)
m.campus.south = fo.PlantBlock(time_block=m.time_block)
# ... units on each plant ...

# Plants are related at the quantity level, not by copying stream state:
m.campus.add_link(
    "north_product_to_south_feed",
    m.campus.north.ro.flow_out_a,
    m.campus.south.ro.flow_in,
)
```

A plant can also register a **product** — a flow, optionally with a quality
indicator — and the network sums it across plants:

```python
m.campus.north.register_product(
    m.campus.north.ro.flow_out_a, name="permeate", quality=m.campus.north.tds
)
```

Registering a quality declares that the resource is only interchangeable at
equal quality, so the network permits mixing only between like-quality streams
(`eq_product_quality`). `total_product["permeate", t]` is the network's summed
flow.
