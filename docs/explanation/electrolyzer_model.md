# Electrolyzer model

This page sets out the physics the electrolyzer models solve, every symbol and
the model quantity it corresponds to, a worked operating point you can check
by hand, and how to read a solved schedule back against the equations.

## What the unit represents

The unit boundary encloses the electrolyzer stack, the gas-liquid separators on
the anode and cathode sides, and the internal electrolyte recirculation loop.
Each stream crossing the boundary is therefore single-phase:

| Port | Direction | Carries |
|---|---|---|
| `inlet_water` | in | liquid make-up water (or electrolyte) |
| `outlet_cathode_gas` | out | cathode product gases, plus undissolved unreacted CO2 on a CO2 electrolyzer |
| `outlet_anode_gas` | out | anode product gas (oxygen by default), plus crossover CO2 |
| `outlet_liquid` | out, optional | electrolyte bleed and any liquid products |
| `inlet_co2` | in, CO2 electrolyzer only | CO2 feed |

The operator's lever is the **current density** {math}`j[t]`. Using current
per unit electrode area keeps the operating range and the cell resistance
comparable across stacks of different size. The stack current is
{math}`I[t] = j[t] \, A_{cell}`, and every other quantity (products, power,
heat, water use, gas volumes) follows from it through the relations below. With
the coefficients fixed and no ohmic loss, every relation is linear in current
density, so a schedule optimization over many time points stays a linear
program.

## Nomenclature

| Symbol | Meaning | Units | Type | Model name |
|---|---|---|---|---|
| {math}`j[t]` | current density | A/cm² | decision variable, {math}`0 \le j \le j_{rated}` | `current_density[t]` |
| {math}`j_{rated}` | rated (maximum) current density | A/cm² | config | `rated_current_density` |
| {math}`A_{cell}` | active electrode area of one cell | cm² | config | `electrode_area` |
| {math}`I[t]` | stack current, {math}`j[t] \, A_{cell}` | A | derived | `current[t]` |
| {math}`N_{cells}` | cells in series | – | config | `n_cells` |
| {math}`F` | Faraday constant, 96 485.33 | C/mol | constant | – |
| {math}`\dot{n}_e[t]` | electron flow through the stack | mol/s | derived | `electron_flow[t]` |
| {math}`FE_k` | Faradaic efficiency of product {math}`k` (share of charge producing it) | – | fitted parameter | `faradaic_efficiency_{k}` |
| {math}`z_k` | electrons transferred per mole of product {math}`k` | – | product table | `electrons` |
| {math}`\dot{n}_k[t]` | molar production rate of product {math}`k` | mol/s | derived | `production_{k}[t]` |
| {math}`V_{cell}` | cell voltage before ohmic loss | V | fitted parameter | `cell_voltage` |
| {math}`ASR` | area-specific cell resistance (`ohmic_loss=True` only) | Ω·cm² | fitted parameter | `area_specific_resistance` |
| {math}`V_{op}[t]` | cell voltage including ohmic loss | V | derived | `operating_voltage[t]` |
| {math}`V_{tn}` | thermoneutral cell voltage | V | config | `thermoneutral_voltage` |
| {math}`f_{BoP}` | balance-of-plant draw as a fraction of stack power | – | fitted parameter | `bop_fraction` |
| {math}`P_{elec}[t]` | electrical draw of the whole unit | kW | derived | `power_electrical[t]` |
| {math}`\dot{Q}_{waste}[t]` | stack heat released above thermoneutral | kW | derived | `waste_heat[t]` |
| {math}`T_{op}, P_{op}` | separator temperature and pressure | K, Pa | config | `operating_temperature`, `operating_pressure` |
| {math}`R` | gas constant, 8.314 | J/(mol·K) | constant | – |
| {math}`\dot{V}_{cathode}[t], \dot{V}_{anode}[t]` | gas outlet volumetric flows at {math}`T_{op}, P_{op}` | m³/h | derived | `outlet_cathode_gas_state.flow_vol_phase[t, ...]`, `outlet_anode_gas_state.flow_vol_phase[t, ...]` |
| {math}`\nu_{r,k}` | moles of reactant {math}`r` consumed per mole of product {math}`k` (negative if produced) | – | product table | `reactants[r]` |
| {math}`\dot{n}_{r,consumed}[t]` | molar consumption of reactant {math}`r` | mol/s | derived | `consumption_{r}[t]` |
| {math}`\dot{V}_{consumed}[t]` | water consumed by the cell reactions | m³/h | derived | `water_consumption[t]` |
| {math}`\dot{V}_{make\text{-}up}[t]` | make-up water fed | m³/h | decision variable | `inlet_water_state.flow_vol_phase[t, ...]` |
| {math}`b` | electrolyte bleed per unit volume of water consumed | – | fitted parameter | `bleed_fraction` |
| {math}`V_{liq}[t]` | separator liquid inventory | m³ | derived, bounded | `liquid_volume[t]` |
| {math}`\Delta t` | time step | h | time grid | `time_block.dt` |
| {math}`k_m` | CO2 mass-transfer coefficient, {math}`D_{CO_2}/\delta_{BL}` (`CO2Electrolyzer` only) | m/s | fitted parameter | `mass_transfer_coefficient` |
| {math}`C_{CO_2,bulk}` | CO2 concentration in the bulk electrolyte (`CO2Electrolyzer` only) | mol/m³ | fitted parameter | `co2_bulk_concentration` |
| {math}`C_{CO_2,local}[t]` | CO2 concentration at the cathode (`CO2Electrolyzer` only) | mol/m³ | derived | `co2_local_concentration[t]` |

"Fitted parameter" means a fixed value in a schedule optimization that
parameter estimation can regress from plant data.

## Equations

### Electrochemistry

Charge passes through every cell in series, so the electron flow is the stack
current times the cell count. Each product takes its Faradaic share of that
charge (Faraday's law):

```{math}
\dot{n}_e[t] = \frac{N_{cells} \, I[t]}{F}
\qquad
\dot{n}_k[t] = \frac{FE_k \, \dot{n}_e[t]}{z_k}
```

The Faradaic efficiencies of the cathode products sum to at most 1, and so do
those of the anode products. Any shortfall is charge lost to reactions that are
not modeled.

A cathode product named by `balance_product` takes no Faradaic efficiency of
its own. Its rate is whatever charge the other cathode products leave:

```{math}
z_b \, \dot{n}_b[t] = \dot{n}_e[t] - \sum_{k \ne b} z_k \, \dot{n}_k[t]
```

reported as `charge_balance_{b}`. This keeps the charge balance closed when the
other products' Faradaic relations are replaced by fitted curves, and
{math}`\dot{n}_b \ge 0` stops those curves from using more charge than the stack
passes. `CO2Electrolyzer` uses hydrogen evolution as the balance product by
default.

### Cell voltage and ohmic loss

The `ohmic_loss` flag sets whether cell resistance enters the operating voltage:

```{math}
V_{op}[t] =
\begin{cases}
V_{cell} & \texttt{ohmic\_loss=False} \\
V_{cell} + ASR \; j[t] & \texttt{ohmic\_loss=True}
\end{cases}
```

The ohmic term is resistance loss through the membrane, electrolyte and
contacts, and it grows with current density. Because power is voltage times
current, `ohmic_loss=True` makes power quadratic in {math}`j`. The schedule then needs
a nonlinear (or quadratic-capable) solver and solves more slowly, and the model
logs a warning when it is enabled. A higher {math}`ASR` is also a natural proxy
for membrane degradation. It is a fixed parameter for now, which you can update
in place as the stack ages.

### Power and heat

```{math}
P_{elec}[t] = N_{cells} \, V_{op}[t] \, I[t] \, (1 + f_{BoP})
\qquad
\dot{Q}_{waste}[t] = N_{cells} \, I[t] \, (V_{op}[t] - V_{tn})
```

Only the part of the cell voltage above thermoneutral turns into heat, so the
ohmic loss ends up as stack heat. At {math}`V_{op} = V_{tn}` the stack runs
thermally neutral, and below it the stack would absorb heat (a negative
{math}`\dot{Q}_{waste}`). The balance-of-plant draw (pumps, power electronics,
controls) is counted in {math}`P_{elec}` but not in the stack heat. Stack
voltage efficiency on a higher-heating-value basis is {math}`V_{tn} / V_{op}`.

### Gas outlets

Each gas outlet leaves at the separator temperature and pressure, and its
volume follows from the ideal-gas law:

```{math}
\dot{V}_{cathode}[t] = \frac{R \, T_{op}}{P_{op}} \Big(\sum_{k \in \text{cathode gas}} \dot{n}_k[t] + (1 - d) \, \dot{n}_{CO_2,unreacted}[t]\Big)
\qquad
\dot{V}_{anode}[t] = \frac{R \, T_{op}}{P_{op}} \Big(\sum_{k \in \text{anode}} \dot{n}_k[t] + \dot{n}_{CO_2,crossover}[t]\Big)
```

The CO2 terms are zero on a water electrolyzer.

### Water and the separator inventory

Each reactant is consumed in proportion to the production of every product,
cathode and anode, that lists it:

```{math}
\dot{n}_{r,consumed}[t] = \sum_k \nu_{r,k} \, \dot{n}_k[t]
\qquad
\dot{V}_{consumed}[t] = \frac{M_{H_2O}}{\rho_{liq}} \, \dot{n}_{H_2O,consumed}[t]
```

The separators hold a liquid inventory, so make-up water can arrive at a
different time from when the stack consumes it:

```{math}
V_{liq}[t] = V_{liq}[t-1]
    + \Delta t \, \big(\dot{V}_{make\text{-}up}[t] - (1 + b) \, \dot{V}_{consumed}[t]\big)
```

```{math}
V_{liq}[0] = V_{liq,initial}
\qquad
f_{min} \, V_{sep} \le V_{liq}[t] \le f_{max} \, V_{sep}
```

{math}`b` is zero unless the unit has a liquid outlet. With a liquid outlet,
that outlet carries the bleed {math}`b \, \dot{V}_{consumed}[t]` plus the
volume of any liquid products. Liquid products are made and drawn off in the
same loop, so they leave without changing the inventory.

## Assumptions and limits

- **Linear polarization at most.** With `ohmic_loss=False`, specific energy
  (kWh per kg of product) does not change with load. `ohmic_loss=True` adds the
  ohmic slope but leaves out activation and mass-transport losses. For a full
  polarization curve, replace `power_electrical_relation` with a fitted curve
  (see {meth}`~flexops.core.ops_block.OpsBlockData.swap_relation`).
- **Constant Faradaic efficiency.** A fitted Faradaic-efficiency curve
  replaces `faradaic_relation_{k}` in the same way. The balance product, if
  one is set, picks up the rest of the charge.
- **Fixed CO2 mass transport.** In `CO2Electrolyzer`, {math}`k_m` is a fixed
  parameter, so the optimizer cannot raise electrolyte flow to raise the
  transport limit. The electrolyte recirculation pump is counted in
  {math}`f_{BoP}`.
- **Fixed temperature and pressure.** The separators sit at
  {math}`T_{op}, P_{op}`. The model has no thermal dynamics or warm-up, and
  {math}`\dot{Q}_{waste}` is reported but not balanced against a cooling loop.
- **Ideal gas, dry products.** Gas volumes ignore water vapour and gas
  crossover through the membrane, and the model makes no purity calculation.
- **Current can change freely from step to step.** The unit itself has no
  ramp-rate or minimum-load constraint. Degradation is not modeled over the
  horizon: {math}`ASR` is fixed within a solve.

## Worked example: PEM water electrolyzer at rated current

`WaterElectrolyzer` with its PEM defaults: 100 cells of 1000 cm², rated
current density 2 A/cm², {math}`V_{cell} = 1.9` V, {math}`V_{tn} = 1.48` V, no
ohmic loss, no balance-of-plant draw, 60 °C and 30 bar, a 15-minute time step.
Run at {math}`j = 2` A/cm², so {math}`I = 2000` A:

| Quantity | Hand calculation | Model value |
|---|---|---|
| electron flow | {math}`100 \times 2000 / 96\,485.33` | `electron_flow` = 2.0729 mol/s |
| hydrogen | {math}`1.0 \times 2.0729 / 2` | `production_H2` = 1.0364 mol/s = **7.52 kg/h** |
| oxygen | {math}`1.0 \times 2.0729 / 4` | `production_O2` = 0.5182 mol/s = 59.7 kg/h |
| electrical draw | {math}`100 \times 1.9 \times 2000 \times 1.0` | `power_electrical` = **380 kW** |
| specific energy | {math}`380 / 7.52` | **50.5 kWh/kg H2** |
| stack heat | {math}`100 \times 2000 \times (1.9 - 1.48)` | `waste_heat` = **84 kW** |
| voltage efficiency (HHV) | {math}`1.48 / 1.9` | 78 % |
| water consumed | {math}`1.0364 \times 18.015 \text{ g/mol}` | `water_consumption` = 0.0672 m³/h (8.94 kg per kg H2) |
| cathode gas volume | {math}`1.0364 \times 8.314 \times 333.15 / 3\times10^6` | 3.445 m³/h at 60 °C, 30 bar |
| anode gas volume | half the cathode volume | 1.723 m³/h |
| inventory, one step with no make-up | {math}`0.5 - 0.25 \times 0.0672` | `liquid_volume` 0.500 → 0.4832 m³ |

As a check, the heat value of the hydrogen (about 39.4 kWh/kg × 7.52 kg/h ≈
296 kW) plus the stack heat (84 kW) adds back up to the 380 kW drawn.

## Reading a solved schedule

In a flexibility study the optimizer picks {math}`j[t]` (and the make-up water
timing) at every step, usually to shift load into cheaper hours while meeting a
production target. Every quantity in the nomenclature table is available at
each time point of the solved model. These are the views that answer the usual
questions:

| Question | Plot against time | Model names |
|---|---|---|
| When does the stack run, and how hard? | current density next to the electricity price | `current_density[t]`, tariff |
| What does it cost to run? | electrical draw | `power_electrical[t]` (the series sent to costing) |
| Is the production target met? | hydrogen rate and its running total | `production_H2[t]`, cumulative sum × {math}`\Delta t` |
| How much cooling is needed? | stack heat | `waste_heat[t]` |
| Is the water supply shifting in time? | make-up versus consumption, and the inventory against its fill window | `inlet_water_state.flow_vol_phase[t, ...]`, `water_consumption[t]`, `liquid_volume[t]` |
| How efficient is each hour? | specific energy | `power_electrical[t] / production_H2[t]` |

With no ohmic loss and constant {math}`FE`, hydrogen output, power and heat
are all proportional to current, so their curves have the same shape and differ
only in scale. With ohmic loss or a fitted polarization curve, the
specific energy rises at high current density. The optimizer then has a reason
to run longer at part load instead of at full current in fewer hours.

A solved model's values can be collected into a table for plotting:

```python
import pandas as pd
import pyomo.environ as pyo

u = m.unit  # the solved electrolyzer
times = list(m.time_block.time_index)
results = pd.DataFrame(
    {
        "j_A_cm2": [pyo.value(u.current_density[t]) for t in times],
        "power_kW": [pyo.value(u.power_electrical[t]) for t in times],
        "h2_mol_s": [pyo.value(u.production_H2[t]) for t in times],
        "waste_heat_kW": [pyo.value(u.waste_heat[t]) for t in times],
        "liquid_m3": [pyo.value(u.liquid_volume[t]) for t in times],
    },
    index=times,
)
results["h2_kg_h"] = results["h2_mol_s"] * 2.016e-3 * 3600
```

## Products and anode reactions

Each cathode product is an `ElectrochemicalProduct`. Its `reactants` map
gives the moles of each reactant consumed per mole of product, and a negative
entry means the reaction produces that reactant. For example, CO2 → CO + ½ O2
has `reactants = {"CO2": 1}`, and H2O → H2 + ½ O2 has `reactants = {"H2O": 1}`.
The model builds one `consumption_{r}` Expression per reactant named in either
table; water is always built, since it drives the separator inventory. Gas
products leave through the gas outlet on their side and liquid products
through the liquid outlet.

The anode reaction is a second product table, `anode_products`, which follows
the same Faraday's law. The default is oxygen evolution at a Faradaic
efficiency of 1. A chlorine anode would instead list Cl2 ({math}`z = 2`,
`reactants = {"NaCl": 2}`) with an oxygen side reaction, and a hypochlorite
anode lists NaClO as a liquid product. Product names must be unique across
both tables.

Stoichiometry can be counted per overall reaction or per half-reaction, as
long as the sum over both tables is right. The default counts the
overall-reaction water on hydrogen, so oxygen consumes none. On a half-reaction
basis, an alkaline cell lists 2 H2O per H2 at the cathode and −2 H2O per O2 at
the anode, which nets to the same water use while letting a chlorine side
reaction at the anode carry its own, different water balance.

## Water electrolyzer

{class}`~flexops.unit_models.electrolyzer.WaterElectrolyzer` fixes the products
to hydrogen at the cathode and oxygen at the anode. Its `technology` sets the
defaults below, and any option given explicitly overrides them:

| Technology | {math}`V_{cell}` | {math}`T_{op}` | {math}`P_{op}` |
|---|---|---|---|
| PEM | 1.9 V | 60 °C | 30 bar |
| AEM | 2.0 V | 50 °C | 35 bar |

An AEM stack's KOH bleed is modeled with `has_liquid_outlet` and
`bleed_fraction`.

## CO2 electrolyzer

{class}`~flexops.unit_models.electrolyzer.CO2Electrolyzer` takes a configurable
product table (default CO at a Faradaic efficiency of 0.9 plus hydrogen at 0.1,
{math}`V_{cell} = 3.0` V, 25 °C) and adds a CO2 feed inlet. The cell reactions
consume

```{math}
\dot{n}_{CO_2,consumed}[t] = \sum_k \nu_{CO_2,k} \, \dot{n}_k[t]
```

where {math}`\nu_{CO_2,k}` is the product's `reactants["CO2"]`, reported as
`consumption_CO2[t]`. Carbonate formation
carries {math}`c` (`co2_crossover_per_electron`) moles of CO2 per mole of
electrons to the anode. That is about 0.5 in an alkaline or neutral membrane
electrode assembly and about 0 in acid:

```{math}
\dot{n}_{CO_2,crossover}[t] = c \, \dot{n}_e[t]
```

The feed follows from the single-pass conversion {math}`X`
(`single_pass_conversion`, consumed plus crossover over fed), and the rest of
the feed leaves unreacted:

```{math}
\dot{n}_{CO_2,feed}[t] = \frac{\dot{n}_{CO_2,consumed}[t] + \dot{n}_{CO_2,crossover}[t]}{X}
\qquad
\dot{n}_{CO_2,unreacted}[t] = \dot{n}_{CO_2,feed}[t] - \dot{n}_{CO_2,consumed}[t] - \dot{n}_{CO_2,crossover}[t]
```

A buffered electrolyte, such as bicarbonate, holds part of the unreacted CO2 in
solution. A fraction {math}`d` (`co2_dissolved_fraction`) leaves dissolved in
the electrolyte, and the rest leaves with the cathode gas:

```{math}
\dot{n}_{CO_2,dissolved}[t] = d \, \dot{n}_{CO_2,unreacted}[t]
```

`co2_dissolved[t]` is reported as a molar flow only. The dilute dissolved CO2
does not change the liquid outlet volume, and the outlet carries no speciation.
A linearized kinetic model of the aqueous and gas split is planned to replace
the fixed fraction.

### Mass-transport limit

CO2 reaches the cathode by diffusing across a boundary layer of thickness
{math}`\delta_{BL}`, which electrolyte flow controls. Fick's law gives the CO2
concentration at the cathode from the CO2 the cells use, both by reduction and
by buffering the hydroxide the cathode generates (the carbonate term
{math}`c \, \dot{n}_e`). This follows eq. 2 of Matthews et al., *ACS Catal.*
2025, 15, 381:

```{math}
C_{CO_2,local}[t] = C_{CO_2,bulk}
  - \frac{\dot{n}_{CO_2,consumed}[t] + \dot{n}_{CO_2,crossover}[t]}{k_m \, N_{cells} \, A_{cell}}
\qquad
k_m = \frac{D_{CO_2}}{\delta_{BL}}
```

Setting `mass_transfer_coefficient` builds `co2_local_concentration[t]` and
the constraint `co2_transport_limit[t]`, which keeps it non-negative. The
constraint is written multiplied through by {math}`k_m`, so it stays linear if
{math}`k_m` is regressed:

```{math}
\dot{n}_{CO_2,consumed}[t] + \dot{n}_{CO_2,crossover}[t] \le k_m \, C_{CO_2,bulk} \, N_{cells} \, A_{cell}
```

With {math}`D_{CO_2} = 1.97 \times 10^{-9}` m²/s, a flow cell with
{math}`\delta_{BL}` between 11 and 235 µm has {math}`k_m` between roughly
{math}`8 \times 10^{-6}` and {math}`2 \times 10^{-4}` m/s. The default
{math}`C_{CO_2,bulk}` is 34 mol/m³, CO2-saturated electrolyte at 1 atm and
25 °C. Without `mass_transfer_coefficient`, the unit has no transport limit.

Selectivity also depends on transport. Faster flow raises CO2 reduction rates
at a given potential, while hydrogen evolution barely changes. A fitted
`faradaic_relation_{k}` captures this at the fitted flow, and the balance
product absorbs the rest of the charge.

## Property packages

Every port is single-phase. The liquid make-up inlet and the liquid outlet
share `liquid_property_package`. Each gas port builds from its own package when
one is given (`cathode_gas_property_package`, `anode_gas_property_package`, or
`co2_property_package`) and from `gas_property_package` otherwise, so each gas
stream can carry a different gas.

## Swappable relations

Each product's `faradaic_relation_{k}` and the `power_electrical_relation` are
registered relations, so a fitted Faradaic-efficiency curve or polarization
curve replaces them in place (see
{meth}`~flexops.core.ops_block.OpsBlockData.swap_relation`). The balance
product's `charge_balance_{b}`, the separator holdup and the CO2 transport
limit are conservation laws and are never swapped.
