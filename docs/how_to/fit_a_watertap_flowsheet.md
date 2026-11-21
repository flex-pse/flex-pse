# Fit a WaterTAP reverse-osmosis flowsheet

This example takes WaterTAP's public seawater reverse-osmosis flowsheet with a
pressure exchanger (`watertap.flowsheets.RO_with_energy_recovery`), sweeps it
over feed flow and water recovery, fits its total electricity draw, and embeds
the fit in a flex-pse {py:class}`~flexops.unit_models.reverseosmosis.ReverseOsmosis`
unit. It follows the steps in
{doc}`fit_a_surrogate_from_a_sweep`.

WaterTAP is not a flex-pse dependency. Install it next to the extras:

```bash
pip install 'flex-pse[parameterize,sweep,solvers]' 'watertap>=1.8'
```

## The flowsheet functions

`run_sweep` rebuilds the flowsheet before every point, sets the swept inputs,
initializes, and solves. The flowsheet specifies its feed by component mass
flows, so `build_model` re-specifies it by volumetric flow and salt mass
fraction. That makes the feed flow a fixed Var the sweep can set.
`initialize` re-derives the mass flows and the pump pressure for the point
before WaterTAP's own initialization.

```python
from watertap.flowsheets.RO_with_energy_recovery import RO_with_energy_recovery as ro


def hold_volumetric_feed(m):
    props = m.fs.feed.properties[0]
    props.flow_mass_phase_comp["Liq", "H2O"].unfix()
    props.flow_mass_phase_comp["Liq", "NaCl"].unfix()
    props.flow_vol_phase["Liq"].fix()
    props.mass_frac_phase_comp["Liq", "NaCl"].fix()


def build_model():
    m = ro.build()
    ro.set_operating_conditions(m)
    hold_volumetric_feed(m)
    return m


def initialize(m):
    props = m.fs.feed.properties[0]
    flow = props.flow_vol_phase["Liq"].value
    recovery = m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"].value
    props.flow_vol_phase["Liq"].unfix()
    props.mass_frac_phase_comp["Liq", "NaCl"].unfix()
    ro.set_operating_conditions(m, water_recovery=recovery, flow_vol=flow)
    ro.initialize_system(m)
    hold_volumetric_feed(m)


def optimize(m, solver):
    return solver.solve(m)
```

## The config

```python
from flexparameterize import ParameterizeConfig

config = ParameterizeConfig(
    inputs={
        "feed": {
            "path": "fs.feed.properties[0].flow_vol_phase[Liq]",
            "units": "m^3/s",
            "range": (5e-4, 1.2e-3),
        },
        "recovery": {
            "path": "fs.RO.recovery_mass_phase_comp[0,Liq,H2O]",
            "units": "dimensionless",
            "range": (0.40, 0.50),
        },
    },
    sampler={"type": "lhs", "n": 30, "seed": 20261118},
    solver={"name": "ipopt-watertap"},
    outputs={
        "power": {
            "path": "fs.costing.aggregate_flow_electricity",
            "units": "kW",
            "unit": "facility.ro",
            "output_units": "kW",
            "input_names": {"feed": "feed", "recovery": "recovery"},
            "terms": ["feed", "recovery", "feed*recovery"],
        }
    },
    max_failed_fraction=0.05,
    holdout={"k": 5, "r": 0},
)
```

- `ipopt-watertap` is WaterTAP's scaled IPOPT. It is registered when WaterTAP
  is imported.
- The swept recovery is WaterTAP's mass-based water recovery. The flex-pse
  unit's `recovery[t]` is volumetric. In this flowsheet the two differ by
  about 1%: a mass recovery of 0.40 is a volumetric recovery of 0.396.
- `fs.costing.aggregate_flow_electricity` is the flowsheet's total electricity
  draw: the high-pressure pump plus the booster pump behind the pressure
  exchanger.
- `acceptance` is left at its defaults: 10% maximum relative error and 10%
  relative RMSE on the holdout points.

## Sweep, fit, and embed

```python
from flexparameterize import apply_to_model
from flexparameterize.sweep import fit_sweep, run_sweep

data = run_sweep(config, build_model, initialize, optimize)
specs = fit_sweep(config, data)
apply_to_model(m, plant_data, tagmap, surrogates=specs)
```

The 30 points take about a minute. All of them solve, and a second run with
the same config returns an identical frame.

Over 40% to 50% recovery the fitted bilinear relationship meets the holdout
with R² 0.998, a 3.3% maximum relative error, and 1.0% relative RMSE. The
feed pump's pressure rises steeply as recovery grows, so a bilinear fit over
a wider recovery window is less accurate. Over 35% to 55%, the holdout
maximum relative error reaches about 12%. Keep the window to the range the
plant operates in, or sweep several narrower windows.

On the flex-pse model, `facility.ro.power_electrical_relation` now reads the
unit's `feed[t]` and `recovery[t]`. The surrogate converts `feed` from the
unit's m³/hr to the m³/s it was fitted in. Each spec's `provenance` records
the trained input ranges; outside them the fit is an extrapolation.
