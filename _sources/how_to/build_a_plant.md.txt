# Build a plant

```{note}
Skeleton (M07). This walks the minimal pump → tank → costing model — the same
system as the headline load-shifting test. The full, executed guide becomes a
notebook in M14.
```

A minimal flex-pse model has a `TimeBlock` (the discrete horizon), a property
package, a `FlexCosting` block (the EECO-backed cost), and one or more unit
models wired with `Arc`s. `FlexCosting` may be constructed before any units
exist — all aggregation is deferred to `cost_process()`.

```python
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
from pyomo.network import Arc
import flexops as fo
from flexops.unit_models import Pump, Tank

m = pyo.ConcreteModel()
m.time_block = fo.TimeBlock(
    start_date="2025-07-08", end_date="2025-07-09", time_step=1 * pyunits.hr
)
m.properties = fo.SimpleAqueousFlow(fixed_density=True)

# Costing first — it aggregates the units' power at cost_process() time.
m.costing = fo.FlexCosting(time_block=m.time_block, tariff_file="tariff.json")

# A pump feeds a storage tank; the pump registers its power with the costing.
m.pump = Pump(
    property_package=m.properties,
    energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    costing_package=m.costing,
)
m.tank = Tank(
    property_package=m.properties,
    max_volume=1000 * pyunits.m**3,
    initial_volume=200 * pyunits.m**3,
)
m.arc = Arc(source=m.pump.outlet, destination=m.tank.inlet)
pyo.TransformationFactory("network.expand_arcs").apply_to(m)

# Fix the demand the tank must serve; leave the pump flow free to be scheduled.
for t in m.time_block.time_index:
    m.tank.outlet_state.flow_vol_phase[t, "Liq"].fix(100.0)
    m.pump.inlet_state.flow_vol_phase[t, "Liq"].setub(300.0)

# Build the opex/capex blocks and minimize the operating cost.
m.costing.cost_process()
m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
```

Solved against a time-of-use tariff, the LP shifts pumping into off-peak hours.
After solving, read the **reported** cost — never the raw objective — from
`m.costing.report_cost(m)`, which returns a categorized breakdown (operating vs.
capital, each itemized).
