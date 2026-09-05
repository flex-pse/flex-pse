# flex-pse

[![CI](https://github.com/flex-pse/flex-pse/actions/workflows/ci.yml/badge.svg)](https://github.com/flex-pse/flex-pse/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/flex-pse/flex-pse/graph/badge.svg)](https://codecov.io/gh/flex-pse/flex-pse)

An open-source Pyomo/IDAES platform for industrial energy-flexibility
optimization — model an industrial facility (water/wastewater/desalination or
similar) as a time-discretized optimization problem, parameterize it from
plant data, and solve scheduling problems against real electricity tariffs and
demand-response signals. Three tools share one config-driven substrate:
**FlexOps** (unit models, plant/network composition, EECO-backed tariff
costing), **FlexParameterize** (fit a FlexOps model from tabular plant data),
and **FlexSchedule** (rolling-horizon scheduling — reserved for a future
release; see the [changelog](CHANGELOG.md) for what 0.1.0 does and does not
cover).

## Install

```bash
pip install "flex-pse[solvers]"
```

`[solvers]` pulls in HiGHS (`highspy`) for LP/MILP. IPOPT (NLP) ships its
binaries separately: run `idaes get-extensions` once after installing (it
fetches the HSL-linked build `flexcore.solvers.get_solver` prefers).

## Example

Build a tank feeding a treatment process, cost it against a tariff, and solve:

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
m.costing = fo.FlexCosting(time_block=m.time_block, tariff_file="tariff.json")
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
m.costing.cost_process()
m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
```

A runnable copy, with tariff fixtures and its config-driven twin, lives at
[`src/flexops/tests/fixtures/api_freeze/`](src/flexops/tests/fixtures/api_freeze/).

## Documentation
Hosted on [GitHub Pages](https://flex-pse.github.io/flex-pse/). Worked
examples live in the companion [`flex-pse-examples`](https://github.com/flex-pse/flex-pse-examples)
repository, with a [web-based interface](https://flex-pse.github.io/flex-pse-examples/)
to interact with select results.

## License
Apache 2.0 — see [`LICENSE`](LICENSE).

## Development

This project is built milestone-by-milestone. See [`PLAN.md`](PLAN.md) for the
roadmap and [`plan/00_conventions.md`](plan/00_conventions.md) for the rules
that govern every change.

Contributors use **conda**, not the published wheel: [`environment.yml`](environment.yml)
pins the optimization stack (Python, `pyomo`, `idaes-pse`, `highspy`, `scip`) and
installs the editable package through a `pip:` subsection, so
`conda env create` is the only setup command.

```bash
conda env create -f environment.yml
conda activate flex-pse
idaes get-extensions          # HSL-linked IPOPT
pre-commit install
pre-commit install --hook-type pre-push
```

## Disclaimer
This repository contains a significant amount of code that was generated via a large-language model. 