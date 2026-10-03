# Steam boilers

{py:class}`~flexops.unit_models.powergeneration.boiler.Boiler` is a linear
steam boiler for scheduling. It comes in two types, chosen with
`boiler_type`:

- **`BoilerType.FIRED`** burns one or more fuels.
- **`BoilerType.HEAT_RECOVERY`** (a heat recovery steam generator, HRSG)
  raises steam from a hot gas stream, such as a `Combustor`'s flue gas. Duct
  firing is optional.

Both types share the same thermal lag, feedwater inlet, steam outlet, on/off
status, and auxiliary power draw. Every coefficient in the model is a fixed,
regressable parameter or a swappable relation. Each one maps onto an output
of a detailed first-principles boiler model, so a high-fidelity simulation
can be reduced to boiler parameters directly (see the mapping table below).

## Ports

| Port | Stream | Property package |
|---|---|---|
| `inlet_<fuel>` | one per name in `fuel_inlet_names` | `fuel_property_package` (gas) |
| `inlet_hot_gas` | hot gas, heat recovery only | `hot_gas_property_package` (gas) |
| `outlet_gas` | cooled gas, heat recovery only | `hot_gas_property_package` (gas) |
| `inlet_feedwater` | feedwater | `feedwater_property_package` (liquid) |
| `outlet_steam` | steam | `steam_property_package` (gas) |

Fuels can enter through inlet ports, for example biogas from a `Digestor`.
They can also come from utilities (`utility_fuel_source`), for example
natural gas billed through the tariff. A boiler can use both kinds at once.
Every fuel needs an entry in `heating_values`. A fired boiler needs at least
one fuel. A fired boiler has no flue gas outlet, because combustion air and
flue gas are not modeled.

## Equations

Write {math}`Q_t` for `heat_output[t]` (kW of steam heat), {math}`f_{i,t}`
for each fuel's volumetric flow, {math}`g_t` for `flow_hot_gas[t]`, and
{math}`s_t` for `status[t]`. {math}`s_t = 1` when unit commitment status is
off.

**Heat input.** Recovered heat is its own relation, so a detailed recovery
curve can replace it without touching the firing term:

```{math}
\dot{Q}^{rec}_t = c_{gas}\, g_t
```

```{math}
\dot{Q}^{abs}_t = \eta \sum_i \mathrm{HV}_i\, f_{i,t} + \dot{Q}^{rec}_t - \dot{Q}^{loss}\, s_t,
\qquad \sum_i \mathrm{HV}_i\, f_{i,t} \le \dot{Q}^{fire}_{max}
```

Here {math}`c_{gas}` is `gas_heat_content`, {math}`\eta` is `efficiency`,
{math}`\dot{Q}^{loss}` is `no_load_loss`, and {math}`\dot{Q}^{fire}_{max}` is
`max_firing_rate`, the burner capacity. Without fuel the firing term drops
out, and without hot gas the recovered term drops out.

**Thermal lag.** The water and metal in the pressurized parts store heat in
proportion to load. The boiler therefore follows its firing with a
first-order lag of time constant {math}`\tau` (`time_constant`). Following
the codebase's backward-difference convention:

```{math}
\tau\,(Q_t - Q_{t-1}) = \Delta t\,\big(\dot{Q}^{abs}_t - Q_t - \dot{Q}^{dis}_t\big),
\qquad Q_{-1} = Q_{init}
```

Raising load by {math}`\Delta Q` in one step costs an extra
{math}`\tau\,\Delta Q / \Delta t` of absorbed heat, which means overfiring.
When {math}`\Delta t \gg \tau` the lag fades away and the boiler sits at
steady state. With {math}`\tau = 0` the boiler is static. The term
{math}`\dot{Q}^{dis}_t \ge 0` (`heat_dissipated`) is heat vented or lost.
Without it a fast ramp-down or shutdown would be infeasible, because the
stored heat has to go somewhere and firing can't go negative. An unfired HRSG
also uses it to dump recovered heat while it's off. The term has no price,
since it only wastes heat that was already paid for. `initial_heat_output` is
a mutable `Param` registered as the rolling-horizon initial state.

**Steam and feedwater.**

```{math}
e_{steam}\, \dot{V}^{steam}_t = Q_t,
\qquad \dot{V}^{fw}_t = r_{\rho}\, \dot{V}^{steam}_t
```

Here {math}`e_{steam}` is `steam_heat_content`, the heat carried per m³ of
outlet steam relative to the feedwater (Δh·ρ). {math}`r_{\rho}` is
`steam_to_water_density_ratio`, which conserves mass across the two
volumetric streams. Outlet pressure equals `steam_pressure`
(`steam_pressure_relation`) and outlet temperature equals
`steam_temperature`. In heat recovery mode, cooled gas flow equals hot gas
flow, pressure passes through, and the outlet temperature equals
`stack_temperature`.

**Power.** Steam heat is a thermal export,
`power_thermal = -heat_output`, registered at the steam temperature. The
auxiliary electrical draw for fans and controls is
`power_electrical = energy_intensity * flow_out`, with
`energy_intensity = aux_energy_intensity`. With unit commitment status on,
`min_heat_output * status <= heat_output <= max_heat_output * status`.
Startup costs and delays come from the usual unit commitment helpers
(`add_startup_shutdown`, `add_startup_delay`), applied to the built unit.

The swappable relations are `heat_absorbed_relation`,
`heat_recovered_relation` (heat recovery only), `steam_flow_relation`,
`steam_pressure_relation`, and `power_electrical_relation`. A config file's
`surrogate` swaps `power_electrical_relation`, as for every unit. The other
relations are swapped at runtime with `swap_relation`. The energy balance,
feedwater balance, and sign constraints are never swappable.

## From a detailed model to boiler parameters

The mapping below follows the experiments of Taler et al. (2019), who built a
distributed-parameter model of a 910 MW supercritical once-through boiler.
That model solves mass, momentum, and energy balances along every heating
surface and is validated against the manufacturer's steady-state data.

| Boiler item | Detailed-model experiment or output |
|---|---|
| `efficiency`, `no_load_loss`, or a surrogate on `heat_absorbed_relation` | Steady states along the sliding-pressure curve (e.g. 40, 80, 87.5, and 100 % load): heat absorbed by the working fluid vs. fuel heat input |
| `gas_heat_content`, or a surrogate on `heat_recovered_relation` (inputs: gas flow and inlet temperature) | HRSG steady states: heat absorbed vs. hot gas flow and inlet temperature |
| `stack_temperature` | Gas temperature leaving the last heat exchanger |
| `max_firing_rate` | Burner or mill capacity |
| `time_constant` | A fuel or gas step test, such as the paper's +7.5 % load step from 80 %. Use the time to reach about 63 % of the steam heat output change, less the fuel transport delay |
| `steam_heat_content`, `steam_to_water_density_ratio` | Outlet Δh·ρ and ρ_steam/ρ_feedwater, or a surrogate on `steam_flow_relation` fitted to outlet flow vs. heat output |
| surrogate on `steam_pressure_relation` (input: `heat_output`) | The sliding-pressure curve, live steam pressure vs. load |
| `min_heat_output`, `max_heat_output` | Minimum stable load and rated output |
| `aux_energy_intensity` | Auxiliary electrical consumption per unit of steam |
| startup helpers | Cold, warm, and hot start-up simulations: fuel used and time to stable load |

Set fitted scalars with `update_parameters`. Attach fitted curves with
`swap_relation` and a `MultilinearSurrogate`.

## What was left out, and why

The detailed model resolves many transients that are much faster than a
scheduling step. They are dropped to keep the model small:

- **Fuel transport delay** (about 5 s from mill to burner) and the **mill
  load ramp** (about 40–50 s per load step).
- **Pressure-sliding reserve.** Dropping live steam pressure by 10 bar
  releases stored steam, about 22 kg/s for about 40 s in the paper's boiler.
  Once the pressure stops falling, that steam has to be paid back by
  overfiring. The released energy is about 2 GJ, roughly one second of the
  boiler's 1838 MW thermal output. At 15-minute or hourly steps it can't shift
  energy, so it matters only for sub-minute reserve products, which call for a
  different formulation.
- **Air heater temperature oscillations** and the **evaporator switch** from
  circulation to once-through flow.

The thermal lag is the one transient that is kept. It is worth keeping
whenever {math}`\tau` is not small compared with {math}`\Delta t`, for example
minutes against a 15-minute step. Otherwise set {math}`\tau = 0`.

## Operating pressure

The model has no pressure ceiling. Every quantity that depends on pressure
is a parameter, and the property packages only require pressure and
temperature to be positive. Two example parameter sets:

| | 10 bar saturated (default) | 1500 psig (≈103 bar), 540 °C, 230 °C feed |
|---|---|---|
| `steam_pressure` | 10 bar | 103 bar |
| `steam_temperature` | 453 K | 813 K |
| `steam_heat_content` | 3.4 kWh/m³ | ≈20 kWh/m³ |
| `steam_to_water_density_ratio` | 0.0054 | ≈0.036 |

The paper's 285 bar supercritical unit fits the same way, since "vapor" is
only the label of the outlet's single phase. Under sliding-pressure
operation, steam density changes strongly with load, so a single
`steam_heat_content` misstates part-load volumetric flow. In that case,
swap a detailed-model flow curve onto `steam_flow_relation`.

## Example: heat recovery behind a combustor

```python
import pyomo.environ as pyo
import flexops as fo

m.chp = fo.Combustor(property_package=m.gas, inlet_names=("fuel",))
m.hrsg = fo.Boiler(
    boiler_type=fo.BoilerType.HEAT_RECOVERY,
    hot_gas_property_package=m.gas,
    feedwater_property_package=m.water,
    steam_property_package=m.steam,
    utility_fuel_source="natural_gas",  # duct firing
    heating_values={"natural_gas": 10.5 * pyo.units.kWh / pyo.units.m**3},
)
m.flue = pyo.network.Arc(source=m.chp.outlet, destination=m.hrsg.inlet_hot_gas)
```

## Reference

J. Taler, W. Zima, P. Ocłoń, S. Grądziel, D. Taler, A. Cebula,
M. Jaremkiewicz, A. Korzeń, P. Cisek, K. Kaczmarski, and K. Majewski,
"Mathematical model of a supercritical power boiler for simulating rapid
changes in boiler thermal loading," *Energy* 175 (2019) 580–592.
[doi:10.1016/j.energy.2019.03.085](https://doi.org/10.1016/j.energy.2019.03.085)
