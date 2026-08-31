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

## Bracketing the plant: Feed and Product

Nothing above owns the streams that *cross* the facility boundary — the raw
water coming in and the potable water going out. `Feed` and `Product` are the
two unit models that do. A `Feed` is a source (zero inlets, N named outlets); a
`Product` is a sink (N named inlets, zero outlets). Each meters the total
resource crossing the boundary, can bound it over time, and can price it:

```python
m.waterfacility.raw_water = fo.Feed(
    property_package=m.properties,
    resource_name="raw_water",
    max_withdrawal=500 * pyunits.m**3 / pyunits.hr,
    price=0.35,                       # $/m3 withdrawn: positive is a cost
    costing_package=m.costing,
)
m.waterfacility.potable = fo.Product(
    property_package=m.properties,
    resource_name="potable_water",
    price=-1.20,                      # negative price = revenue
    costing_package=m.costing,
)
m.waterfacility.feed_to_tank = Arc(
    source=m.waterfacility.raw_water.outlet_a,
    destination=m.waterfacility.tank.inlet,
)
m.waterfacility.plant_to_product = Arc(
    source=m.waterfacility.plant.outlet,
    destination=m.waterfacility.potable.inlet_a,
)
```

The plant **discovers** these the same way it discovers power and fuel — no
registration call — and sums them into `total_feed["raw_water", t]` and
`total_product["potable_water", t]`. `resource_name` is the aggregation key and
is independent of the Pyomo block name: two `Feed` blocks with different names
give two rows, and two sharing one `resource_name` sum into a single row, which
is how you model the same resource entering at two points. Several inbound
resources means several `Feed` blocks — one block, one resource.

A plant normally has several of each (raw water, citric acid and antiscalant
in; potable water, brine and waste out), so this is the expected case rather
than the edge case. Costing stays per block: the opex line item is named from
the block's own dotted Pyomo name, so two blocks sharing a resource still get
separate, differently-priced line items even though their flows aggregate
together.

### Rate limits and horizon allowances are different constraints

`max_withdrawal=500 * pyunits.m**3 / pyunits.hr` above is a **rate**, and it
binds in every period. A permit or a delivery contract is usually not that — it
is a **quantity** over the whole horizon, and how you shape the profile beneath
it is exactly the flexibility you are trying to optimize. Say which you mean
with `withdrawal_basis` (`demand_basis` on a `Product`):

```python
m.waterfacility.raw_water = fo.Feed(
    property_package=m.properties,
    resource_name="raw_water",
    max_withdrawal=240_000 * pyunits.m**3,   # a quantity, not a rate
    withdrawal_basis="horizon",
)
```

That builds a scalar `withdrawal_total` and the equality `eq_withdrawal_total`
defining it as `sum_t withdrawal[t] * dt`, then bounds *that*. Nothing else
moves: `withdrawal[t]` stays time-indexed, so costing, the plant's
`total_feed[resource, t]` aggregation and `set_external_dispatch` all work
exactly as before. The scalar total is readable after a solve, and its limit
carries a dual — the shadow price of the permit.

Because the two bases take different dimensions, a limit whose units contradict
the declared basis is rejected rather than silently rescaled:

```python
fo.Feed(..., max_withdrawal=500 * pyunits.m**3 / pyunits.hr,
        withdrawal_basis="horizon")      # FlexConfigError(field="max_withdrawal")
```

Limits are **mutable Params**, not variable bounds, so a limit that varies over
time is written rather than configured:

```python
for t in m.time_block.time_index:
    m.waterfacility.raw_water.withdrawal_max[t].set_value(profile[t])
```

(On the horizon basis there is one period, so the same rewrite is
`withdrawal_max.set_value(v)` with no index.)

An *exact* profile is not a limit at all — fix the metered flow with the
inherited external-dispatch hook:

```python
m.waterfacility.raw_water.set_external_dispatch(
    m.waterfacility.raw_water.withdrawal, series, fix=True
)
```

A `Product` aggregates flow and does **not** blend: each inlet's composition,
temperature and pressure arrive from its own arc and stay independent. Put a
`Mixer` upstream when a single blended stream is wanted.

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

Linking one plant's product to another's feed needs no new API — `add_link`
takes any two time-indexed quantities, so a plant total is passed as a
`Reference` and a boundary block's own meter directly:

```python
m.campus.add_link(
    "north_product_to_south_feed",
    pyo.Reference(m.campus.north.total_product["potable_water", :]),
    m.campus.south.raw_water.withdrawal,
)
```

The network's `total_feed`/`total_product` are the sums of its child plants'
totals — never a second walk over their units — so nothing is double-counted.
