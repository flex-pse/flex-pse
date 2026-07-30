# Pump + tank + battery load shifting

An interactive marimo notebook duplicating the headline economic result from
`src/flexops/tests/costing/test_load_shifting_component.py`: a pump fills a
storage tank against a fixed draw, and minimizing the `FlexCosting` operating
cost under a time-of-use tariff shifts pumping (and battery discharge) out of
the peak-price window. The horizon is stretched from one day to a full month
(July 2025, hourly), a behind-the-meter battery and pump unit-commitment logic
are added (M08), and the pump/tank/battery sizing and tariff demand charges
are exposed as sliders.

**The model is built entirely from `config.json`**, not from the sliders
directly: clicking **Solve** writes every slider value into an
`ExampleConfig` (`helpers/config.py`) and saves it to `config.json` in this
directory; the notebook then reads that file straight back off disk and
`helpers/build.py` constructs and solves the Pyomo model from it. Edit
`config.json` by hand (or drive it from another script) and it builds and
solves exactly the same way, with no notebook involved.

This example is standalone (its own copy of the demo tariff, no dependency on
the test suite) and intentionally lives outside the milestone system. It is
expected to move into a separate examples repo after the M09 API freeze.

## Run it

Requires the `dev` extra (installed automatically by `environment.yml`, or via
`pip install -e ".[dev]"`), which includes `marimo` and `matplotlib`.

```bash
marimo edit examples/pump_tank_load_shifting/load_shifting.py   # developer mode
marimo run examples/pump_tank_load_shifting/load_shifting.py    # interactive app
python examples/pump_tank_load_shifting/load_shifting.py        # plain script, default slider values
```

Or skip the notebook entirely and build/solve straight from the config file:

```python
from pathlib import Path

from helpers.build import build_model, load_tariff_for_config, solve_model
from helpers.config import load_config
from helpers.results import extract_results

example_dir = Path("examples/pump_tank_load_shifting")
config = load_config(example_dir / "config.json")
tariff = load_tariff_for_config(config, example_dir)
model = build_model(config, tariff)
solve_model(model)
results = extract_results(model, config, tariff)
```

## Files

- `load_shifting.py` — the marimo notebook: sliders write an `ExampleConfig`
  to `config.json`, then read it back to build, solve, and plot.
- `config.json` — the full model configuration (time horizon, tariff,
  facility draw, pump/tank/battery parameters, and unit connections/arcs);
  the single source of truth the model is built from.
- `tariff_tou_demo.json` — a copy of
  `src/flexops/tests/fixtures/tariff_tou_demo.json` (the demo TOU tariff), kept
  local so the example has no path dependency on the test suite.
- `helpers/` — config-driven build/solve/plot helpers, factored out of the
  notebook so they're reusable and legible on their own:
  - `units.py` — parses `"<value> <units>"` config strings (e.g.
    `"300 m**3/hr"`) into unit-carrying Pyomo quantities.
  - `config.py` — the `ExampleConfig` pydantic schema (reusing
    `flexcore.config.schema`'s `TimeConfig`, `UnitCommitmentConfig`, and
    `ArcSpec` directly) plus `load_config`/`save_config`.
  - `build.py` — `build_model`/`solve_model`, and `load_tariff_for_config`.
  - `results.py` — `extract_results`, pulling time series and summary metrics
    off a solved model.
  - `plotting.py` — `plot_results`, the price/load/pump/tank/battery figure.
