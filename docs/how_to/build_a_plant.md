# Build a plant

A flex-pse model builds in one order. **TimeBlock, property package, costing,
plant, units, arcs, `cost_process()`, objective.** Costing comes *before* the
units on purpose. `FlexCosting` aggregates power at `cost_process()` time, so
you can construct it while the plant is still empty. Plant totals are
deferred for the same reason.

The script below follows exactly this order. A component test runs it on
every pull request. If a change breaks it, that change breaks the public
API.

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

# A PlantBlock is a collection of UNITS. It takes the TimeBlock explicitly.
# Omit it only when the model carries exactly one.
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

Arc expansion and the solve stay explicit. No library code applies the
transformation for you.

```python
from flexcore.solvers import get_solver

pyo.TransformationFactory("network.expand_arcs").apply_to(m)
results = get_solver(model=m).solve(m)
```

After solving, read the **reported** cost, never the raw objective, from
`m.costing.report_cost(m)`. It returns a categorized breakdown, operating
against capital, each itemized.

## Bracketing the plant with Feed and Product

Nothing above owns the streams that cross the facility boundary, the raw
water coming in and the potable water going out. `Feed` and `Product` are
the two unit models that do. A `Feed` is a source, with zero inlets and N
named outlets. A `Product` is a sink, with N named inlets and zero outlets.
Each one meters the total resource crossing the boundary. Each one can bound
that resource over time, and each one can price it.

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

The plant **discovers** these the same way it discovers power and fuel. You
don't register anything. It sums them into `total_feed["raw_water", t]` and
`total_product["potable_water", t]`. `resource_name` is the aggregation key,
and it's independent of the Pyomo block name. Two `Feed` blocks with
different names give you two rows. Two blocks sharing one `resource_name`
sum into a single row, which is how you model the same resource entering at
two points. If you have several inbound resources, you add several `Feed`
blocks. One block, one resource.

A plant normally has several of each (raw water, citric acid, and
antiscalant coming in, potable water, brine, and waste going out). That's
the expected case, not the edge case. Costing stays per block. The opex line
item takes its name from the block's own dotted Pyomo name, so two blocks
sharing a resource still get separate, differently priced line items. Their
flows aggregate together, but the billing doesn't.

### Rate limits and horizon allowances are different constraints

`max_withdrawal=500 * pyunits.m**3 / pyunits.hr` above is a **rate**. It
binds in every period. A permit or a delivery contract usually isn't that.
It's a **quantity** over the whole horizon, and shaping the profile beneath
it is exactly the flexibility you're trying to optimize. Say which you mean
with `withdrawal_basis` (`demand_basis` on a `Product`).

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
moves. `withdrawal[t]` stays indexed by time, so costing, the plant's
`total_feed[resource, t]` aggregation, and `set_external_dispatch` all work
exactly as before. You can read the scalar total after a solve, and its
limit carries a dual, the shadow price of the permit.

The two bases take different dimensions. A limit whose units contradict the
declared basis gets rejected instead of silently rescaled.

```python
fo.Feed(..., max_withdrawal=500 * pyunits.m**3 / pyunits.hr,
        withdrawal_basis="horizon")      # FlexConfigError(field="max_withdrawal")
```

Limits are **mutable Params**, not variable bounds. If a limit varies over
time, you write that directly instead of configuring it.

```python
for t in m.time_block.time_index:
    m.waterfacility.raw_water.withdrawal_max[t].set_value(profile[t])
```

On the horizon basis there's just one period, so the same rewrite becomes
`withdrawal_max.set_value(v)` with no index.

An *exact* profile isn't a limit at all. Fix the metered flow with the
inherited external dispatch hook instead.

```python
m.waterfacility.raw_water.set_external_dispatch(
    m.waterfacility.raw_water.withdrawal, series, fix=True
)
```

A `Product` aggregates flow and does **not** blend. Each inlet's
composition, temperature, and pressure arrive from its own arc and stay
independent. Put a `Mixer` upstream if you want a single blended stream.

## The config driven twin

Nothing that matters has to live only in imperative code. You can build the
same model from one version-controlled JSON file. The config twin of the
script above is `api_freeze_config.json`, and a component test holds the two
to the same solved objective.

```python
from flexcore.config.io import load_model_config

m = fo.build_model(load_model_config("api_freeze_config.json"))
```

`build_model` constructs the TimeBlock, the property package, the
`FlexCosting` block, the plant and unit tree, its arcs, any declared
external (DERMS) dispatch, and the objective. Two conventions matter when
you write one of these files.

- **Units are data.** A config can't carry a Pyomo expression, so you write
  a unit carrying quantity as `{"value": 0.5, "units": "kWh/m^3"}`, or, for
  `time.time_step`, as the string `"15 min"`. Anything else in
  `construction_options` passes to the unit's constructor unchanged.
- **Runtime only options come from the builder.** `property_package` and
  `costing_package` are live Pyomo objects, not serializable, so
  `build_model` wires them into every unit. Don't list them in
  `construction_options`.

A malformed config raises `FlexConfigError` naming the exact field path
(for example, `plant.units.surrogate.unit_model_class`), with the
underlying pydantic `ValidationError` as its cause.

## More than one plant with NetworkBlock

A plant containing plants is a `NetworkBlock`, not a nested `PlantBlock`.
`PlantBlock` composes units. `NetworkBlock` composes plants. Its totals are
the sums of its child plants' totals, so every unit gets counted exactly
once.

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

A plant can also register a **product**, a flow with an optional quality
indicator, and the network sums it across plants.

```python
m.campus.north.register_product(
    m.campus.north.ro.flow_out_a, name="permeate", quality=m.campus.north.tds
)
```

Registering a quality declares that the resource is only interchangeable at
equal quality, so the network permits mixing only between streams of like
quality (`eq_product_quality`). `total_product["permeate", t]` is the
network's summed flow.

Linking one plant's product to another's feed needs no new API. `add_link`
takes any two quantities indexed by time, so you pass a plant total as a
`Reference` alongside a boundary block's own meter.

```python
m.campus.add_link(
    "north_product_to_south_feed",
    pyo.Reference(m.campus.north.total_product["potable_water", :]),
    m.campus.south.raw_water.withdrawal,
)
```

The network's `total_feed`/`total_product` are the sums of its child
plants' totals. It never walks their units a second time, so nothing gets
counted twice.

## See it running

Want a worked, solved example built on this API? Check the interactive
examples at
[flex-pse.github.io/flex-pse-examples](https://flex-pse.github.io/flex-pse-examples/).
One shows a pump scheduling facility shifting load away from peak hours,
with the resulting schedule plotted against the tariff. Each example page
opens directly in a browser. You don't need to install anything.
