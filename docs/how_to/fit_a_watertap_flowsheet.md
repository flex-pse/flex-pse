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
flows and its recovery by mass of water. `build_model` re-specifies both on a
volumetric basis: feed volumetric flow, salt mass fraction, and the volumetric
recovery `fs.RO.recovery_vol_phase`. That makes them fixed Vars the sweep can
set, in the same basis as the flex-pse unit's `feed[t]` and `recovery[t]`.
`initialize` re-derives the mass flows and the pump pressure for the point.
WaterTAP's initialization needs a mass-based recovery, so it uses the
volumetric value as a guess and then fixes the volumetric recovery again. The
final solve finds the matching mass-based recovery.

```python
from watertap.flowsheets.RO_with_energy_recovery import RO_with_energy_recovery as ro


def hold_volumetric_spec(m):
    props = m.fs.feed.properties[0]
    props.flow_mass_phase_comp["Liq", "H2O"].unfix()
    props.flow_mass_phase_comp["Liq", "NaCl"].unfix()
    props.flow_vol_phase["Liq"].fix()
    props.mass_frac_phase_comp["Liq", "NaCl"].fix()
    m.fs.RO.recovery_mass_phase_comp[0, "Liq", "H2O"].unfix()
    m.fs.RO.recovery_vol_phase[0, "Liq"].fix()


def build_model():
    m = ro.build()
    ro.set_operating_conditions(m)
    hold_volumetric_spec(m)
    return m


def initialize(m):
    props = m.fs.feed.properties[0]
    flow = props.flow_vol_phase["Liq"].value
    recovery = m.fs.RO.recovery_vol_phase[0, "Liq"].value
    props.flow_vol_phase["Liq"].unfix()
    props.mass_frac_phase_comp["Liq", "NaCl"].unfix()
    m.fs.RO.recovery_vol_phase[0, "Liq"].unfix()
    # WaterTAP initializes from a mass-based recovery; the volumetric one is a guess.
    ro.set_operating_conditions(m, water_recovery=recovery, flow_vol=flow)
    ro.initialize_system(m)
    hold_volumetric_spec(m)
    m.fs.RO.recovery_vol_phase[0, "Liq"].fix(recovery)


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
            "path": "fs.RO.recovery_vol_phase[0,Liq]",
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
- The swept recovery is volumetric, like the flex-pse unit's `recovery[t]`.
  WaterTAP's mass-based water recovery is about 1% higher: a volumetric
  recovery of 0.40 is a mass recovery of 0.404.
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
