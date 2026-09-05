# flex-pse in ten minutes

Here is the frozen public API, start to finish. You'll build a facility with a
storage tank, a treatment plant modeled as one energy intensity surrogate, and
a battery sitting behind the meter. You'll schedule it against a tariff that
changes through the day, for one month at 15 minute steps. A component test
runs this exact script on every pull request. It cannot drift from the
library.

## 1. Install

```bash
conda env create -f environment.yml
conda activate flex-pse
```

## 2. Build the model

Run this from a directory holding the two data files it loads by name.
`tariff.json` is a time of use price sheet. `dr_events.json` is a
demand response program, and it can be empty.

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

Five things are worth noticing in those thirty lines.

- **The `TimeBlock` is the substrate.** 29 days at 15 minutes gives you 2 784
  discrete time points, an ordered integer set. It's not a DAE. See
  [why](../explanation/time_and_dynamics.md).
- **Costing is built before the plant.** `FlexCosting` defers all aggregation
  to `cost_process()`, so you can hand it units that don't exist yet.
- **`PlantBlock` composes units.** If you have several facilities, compose
  *plants* in a `NetworkBlock` instead. See [how](../how_to/build_a_plant.md).
- **Units carry units.** Every physical quantity is a `pyunits` carrying
  expression. Get the dimensions wrong and it fails loudly instead of
  silently rescaling.
- **The objective is the relaxed proxy, not the answer.** Read the real cost
  back from `report_cost` in step 4.

## 3. Solve

Arc expansion is explicit. No library code does it for you.

```python
from flexcore.solvers import get_solver

pyo.TransformationFactory("network.expand_arcs").apply_to(m)
results = get_solver(model=m).solve(m)
```

`get_solver` classifies the model and picks a capable installed solver, or
raises with install instructions. It never relaxes or transforms the model
behind your back.

## 4. Read the result

```python
report = m.costing.report_cost(m)
print(report.operating.electricity)
```

The raw solver objective is a convex relaxed, possibly scalarized internal
quantity. It's never the number you want. `report_cost` re-evaluates the
realized power trajectory after the solve, through EECO, to get the true
bill.

## 5. Or build the same model from a config

None of this has to live in Python. You can describe the same model with a
JSON config file, `api_freeze_config.json`, and it solves to the same
objective.

```python
from flexcore.config.io import load_model_config

m = fo.build_model(load_model_config("api_freeze_config.json"))
```

## Where next

- [Build a plant](../how_to/build_a_plant.md). The build order in detail, the
  config conventions, and multi plant networks.
- [Time and dynamics](../explanation/time_and_dynamics.md). Why discrete
  time.
- [Energy nomenclature](../explanation/energy_nomenclature.md).
  `power_electrical` / `power_thermal`, and why fuel is a volume.
- [Examples](../examples/index.md). Solved, interactive walkthroughs built on
  this API.
