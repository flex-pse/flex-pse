# Electrolyzer model

The unit boundary encloses the electrolyzer stack, the anode- and cathode-side
gas-liquid separators, and the internal electrolyte recirculation loop, so every
port carries a single phase: liquid make-up in, product gas out of each
electrode side, and (optionally) a liquid bleed out. The stack is driven by its
current, and every relationship is linear in it while the scalar coefficients
are fixed:

```{math}
\dot{n}_{e}[t] &= N_{cells} \, I[t] / F \\
\dot{n}_{k}[t] &= FE_k \, \dot{n}_{e}[t] / z_k \\
P_{elec}[t] &= N_{cells} \, V_{cell} \, I[t] \, (1 + f_{BoP}) \\
\dot{V}_{gas}[t] &= \tfrac{R T_{op}}{P_{op}} \sum \dot{n}_{gas}[t]
```

Each product's `faradaic_relation_{name}` and the `power_electrical_relation`
are registered relations, so a fitted Faradaic-efficiency curve or polarization
curve replaces them in place (see
{meth}`~flexops.core.ops_block.OpsBlockData.swap_relation`). The anode evolves
oxygen with all the charge, {math}`\dot{n}_{O_2} = \dot{n}_e / 4`.

## Separators

The cathode and anode gas outlets each carry one lumped stream at the fixed
operating temperature and pressure; per-species molar flows are the internal
`production_{name}` Vars. The liquid side is one lumped inventory whose holdup
lets make-up water be fed at a different time from when the stack consumes it:

```{math}
V_{liq}[t] = V_{liq}[t-1]
    + \Delta t \, (\dot{V}_{make\text{-}up}[t] - (1 + b) \, \dot{V}_{consumed}[t])
```

where {math}`b` is the bleed fraction (zero without a liquid outlet). Liquid
products are created and drawn off in the same loop, so they leave through the
liquid outlet without changing the inventory. The holdup equation is
conservation and is never registered.

## Products

Each cathode product is an
`ElectrochemicalProduct`. Its
stoichiometry is per mole of product for the overall cell reaction, anode
oxygen evolution included. For example, CO2 -> CO + 1/2 O2 has
`co2_per_mol = 1` and `water_per_mol = 0`, and H2O -> H2 + 1/2 O2 has
`water_per_mol = 1`. Gas products leave through the cathode gas outlet and
liquid products through the liquid outlet.

## Property packages

Every port is single-phase. The liquid make-up inlet and the liquid outlet
share `liquid_property_package`. Each gas port builds from its own package
when one is given (`cathode_gas_property_package`,
`anode_gas_property_package`, or `co2_property_package`) and from
`gas_property_package` otherwise, so each gas stream can carry a different gas.

## Water electrolyzer

{class}`~flexops.unit_models.electrolyzer.WaterElectrolyzer` fixes the product
slate to hydrogen (H2O -> H2 + 1/2 O2), so it has no `products` option. Its
`technology` (PEM or AEM) picks the defaults for `cell_voltage`,
`operating_temperature`, and `operating_pressure`; an option given explicitly
overrides its technology default. An AEM stack's KOH bleed is modeled with
`has_liquid_outlet` and `bleed_fraction`.

## CO2 electrolyzer

{class}`~flexops.unit_models.electrolyzer.CO2Electrolyzer` takes a
configurable product table (default CO at a Faradaic efficiency of 0.9 plus
hydrogen at 0.1) and adds a CO2 feed inlet. Of the CO2 fed, the cell reactions
consume

```{math}
\dot{n}_{CO_2,consumed}[t] = \sum_k \nu_{CO_2,k} \, \dot{n}_k[t]
```

carbonate formation carries `co2_crossover_per_electron` ({math}`c`) per
electron to the anode, about 0.5 in an alkaline or neutral membrane electrode
assembly and about 0 in acid,

```{math}
\dot{n}_{CO_2,crossover}[t] = c \, \dot{n}_{e}[t]
```

and the rest leaves unreacted with the cathode gas. The feed follows from the
single-pass conversion {math}`X` (consumed plus crossover, over fed):

```{math}
\dot{n}_{CO_2,feed}[t] = (\dot{n}_{CO_2,consumed}[t] + \dot{n}_{CO_2,crossover}[t]) / X
```

The unreacted CO2 adds to the cathode gas volume, and the crossover CO2 adds to
the anode gas volume.
