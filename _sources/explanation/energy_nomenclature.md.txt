# Energy nomenclature

flex-pse uses **one** project-wide naming standard for a unit's energy draw, so
that plant aggregation and costing can find every unit's contribution by name.
The canonical names live as constants in `flexcore.nomenclature` — a typo
becomes an import error rather than a silently-unaggregated variable.

| Name | Meaning | Units | Consumer |
|---|---|---|---|
| {data}`~flexcore.nomenclature.POWER_ELECTRICAL` (`power_electrical[t]`) | unit-level electrical draw (motor/drive) | kW | FlexCosting → EECO (energy + demand charges + DR); plant aggregation |
| {data}`~flexcore.nomenclature.POWER_THERMAL` (`power_thermal[t]`) | unit-level heat/gas-driven duty | kW | separate thermal aggregation/costing |
| {data}`~flexcore.nomenclature.FUEL_USAGE` (`fuel_usage_<fuel>[t]`) | unit-level combustible-fuel consumption | m³/hr | FlexCosting → EECO's gas leg |

Rules (`plan/00_conventions.md` §2, `plan/01_architecture.md` §4):

- Every unit model registers at least one power draw via
  {meth}`~flexops.core.ops_block.OpsBlockData.register_power` (or creates one
  with {meth}`~flexops.core.ops_block.OpsBlockData.declare_power`).
- Both **power** quantities are **powers in kW**, despite the domain word
  "energy". FlexCosting aggregates them into a kW time series and hands the
  electrical one to EECO both in-model (objective) and as a post-solve numpy
  array (reporting); EECO derives kWh internally from the time step.
- Never introduce a variable named bare `power`, `energy`, or `work`. Always
  refer to the `flexcore.nomenclature` constants; a literal name string
  anywhere else is review-blocking.
- Storage units (e.g. {class}`~flexops.unit_models.storage.battery.BatteryModel`) sign
  `POWER_ELECTRICAL` from the storage device's own frame: **charging is
  positive, discharging is negative** (an export). This matches the general
  draw convention above -- charging is a draw, discharging offsets it -- and
  keeps plant aggregation a plain sum with no per-unit sign-flipping.
- A unit that *generates* rather than stores follows the same export
  convention: {class}`~flexops.unit_models.powergeneration.combustor.Combustor` signs
  `POWER_ELECTRICAL` **negative** (upper-bounded at 0), so a combustor's
  output nets against plant load exactly as a discharging battery does, with
  no per-unit sign-flipping at the plant level either.
- **A generating unit whose relation is swappable splits the magnitude from
  the sign.** `Combustor` carries `power_generated[t]` — the generation
  magnitude, bounded below at 0 — and ties it to the draw through the
  constraint `power_electrical_sign[t]`. Only the magnitude relation is
  registered, so a surrogate swap can never replace the sign constraint, and
  the sign survives a fit no expression check could have vouched for: nothing
  can prove an arbitrary fitted body is non-positive over the feasible region,
  but a body feeding a target bounded below at 0 either respects the
  convention or violates a bound that names it. It also means a generation
  surrogate is fitted in **magnitude space** — positive, as generation data is
  normally logged — which is why `power_generated` rather than
  `power_electrical` is the unit's registered IO output.

## Fuel is a volume, not a power

A combustible fuel is metered and billed on **volume**, so `FUEL_USAGE` is a
volumetric flow in m³/hr and deliberately **not** a
{class}`~flexcore.nomenclature.PowerKind`. A unit that burns fuel registers that
flow with {meth}`~flexops.core.ops_block.OpsBlockData.register_fuel_usage` — for a
stream built from {class}`~flexops.properties.simple_gas.SimpleGasFlow`, whose
`flow_vol_phase` is already m³/hr, that is a `Reference` to the stream's flow.
FlexCosting sums every registered flow into `aggregate_fuel_usage[t, fuel]` and
hands that volume series to EECO's gas leg.

flex-pse applies **no heating value** anywhere. If a tariff prices gas on an
energy basis, converting volume to energy is EECO's job — it applies its own fuel
heating-value assumption. Do not model a fuel as a kW `power_thermal` duty and
back-convert it: a heat duty is a different quantity from metered fuel volume, and
is not the fuel-cost input.

The names were agreed to disambiguate the "work vs. power" inconsistencies in
IDAES/WaterTAP. The group may revisit the exact words, which is why they live in
one constants module.
