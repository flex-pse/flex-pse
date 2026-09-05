# Energy nomenclature

flex-pse uses **one** project wide naming standard for a unit's energy
draw, so plant aggregation and costing can find every unit's contribution
by name. The canonical names live as constants in `flexcore.nomenclature`.
A typo becomes an import error instead of a silently unaggregated
variable.

| Name | Meaning | Units | Consumer |
|---|---|---|---|
| {data}`~flexcore.nomenclature.POWER_ELECTRICAL` (`power_electrical[t]`) | electrical draw of one unit (motor/drive) | kW | FlexCosting → EECO (energy, demand charges, DR) and plant aggregation |
| {data}`~flexcore.nomenclature.POWER_THERMAL` (`power_thermal[t]`) | heat or gas driven duty of one unit | kW | separate thermal aggregation/costing |
| {data}`~flexcore.nomenclature.FUEL_USAGE` (`fuel_usage_<fuel>[t]`) | combustible fuel consumption of one unit | m³/hr | FlexCosting → EECO's gas leg |

Rules:

- Every unit model registers at least one power draw, through
  {meth}`~flexops.core.ops_block.OpsBlockData.register_power` or by
  creating one with
  {meth}`~flexops.core.ops_block.OpsBlockData.declare_power`.
- Both **power** quantities are **powers in kW**, despite the domain word
  "energy." FlexCosting aggregates them into a kW time series and hands
  the electrical one to EECO, both in the model (objective) and as a
  numpy array after the solve (reporting). EECO derives kWh internally
  from the time step.
- Never introduce a variable named bare `power`, `energy`, or `work`.
  Always use the `flexcore.nomenclature` constants. A literal name string
  anywhere else blocks review.
- Storage units (for example
  {class}`~flexops.unit_models.storage.battery.BatteryModel`) sign
  `POWER_ELECTRICAL` from the storage device's own frame. **Charging is
  positive, discharging is negative** (an export). This matches the
  general draw convention above. Charging is a draw, discharging offsets
  it, and plant aggregation stays a plain sum with no per unit sign
  flipping.
- A unit that *generates* rather than stores follows the same export
  convention. {class}`~flexops.unit_models.powergeneration.combustor.Combustor`
  signs `POWER_ELECTRICAL` **negative** (upper bounded at 0), so a
  combustor's output nets against plant load exactly the way a
  discharging battery does, again with no per unit sign flipping at the
  plant level.
- **A generating unit whose relation is swappable splits the magnitude
  from the sign.** `Combustor` carries `power_generated[t]`, the
  generation magnitude, bounded below at 0, and ties it to the draw
  through the constraint `power_electrical_sign[t]`. Only the magnitude
  relation is registered, so a surrogate swap can never replace the sign
  constraint, and the sign survives a fit that no expression check could
  have vouched for. Nothing can prove an arbitrary fitted body stays non
  positive over the feasible region, but a body feeding a target bounded
  below at 0 either respects the convention or violates a bound that names
  it directly. It also means a generation surrogate gets fitted in
  **magnitude space**, positive, the way generation data is normally
  logged. That's why `power_generated`, not `power_electrical`, is the
  unit's registered IO output.

## Fuel is a volume, not a power

A combustible fuel gets metered and billed on **volume**, so `FUEL_USAGE`
is a volumetric flow in m³/hr, and deliberately not a
{class}`~flexcore.nomenclature.PowerKind`. A unit that burns fuel registers
that flow with
{meth}`~flexops.core.ops_block.OpsBlockData.register_fuel_usage`. For a
stream built from {class}`~flexops.properties.simple_gas.SimpleGasFlow`,
whose `flow_vol_phase` is already m³/hr, that's a `Reference` to the
stream's flow. FlexCosting sums every registered flow into
`aggregate_fuel_usage[t, fuel]` and hands that volume series to EECO's gas
leg.

flex-pse applies **no heating value** anywhere. If a tariff prices gas on
an energy basis, converting volume to energy is EECO's job. It applies its
own fuel heating value assumption. Don't model a fuel as a kW
`power_thermal` duty and convert it back. A heat duty is a different
quantity from metered fuel volume, and it isn't the fuel cost input.

The team agreed on these names to clear up the "work vs. power"
inconsistencies in IDAES/WaterTAP. The exact words may still change, which
is why they all live in one constants module.
