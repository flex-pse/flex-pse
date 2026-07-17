# Energy nomenclature

flex-pse uses **one** project-wide naming standard for a unit's energy draw, so
that plant aggregation and costing can find every unit's contribution by name.
The canonical names live as constants in `flexcore.nomenclature` — a typo
becomes an import error rather than a silently-unaggregated variable.

| Name | Meaning | Units | Consumer |
|---|---|---|---|
| {data}`~flexcore.nomenclature.POWER_ELECTRICAL` (`power_electrical[t]`) | unit-level electrical draw (motor/drive) | kW | FlexCosting → EECO (energy + demand charges + DR); plant aggregation |
| {data}`~flexcore.nomenclature.POWER_THERMAL` (`power_thermal[t]`) | unit-level heat/gas-driven duty | kW | separate thermal aggregation/costing |

Rules (`plan/00_conventions.md` §2, `plan/01_architecture.md` §4):

- Every unit model registers at least one of these via
  {meth}`~flexops.core.ops_block.OpsBlockData.register_power` (or creates one
  with {meth}`~flexops.core.ops_block.OpsBlockData.declare_power`).
- Both quantities are **powers in kW**, despite the domain word "energy".
  FlexCosting aggregates them into a kW time series and hands it to EECO both
  in-model (objective) and as a post-solve numpy array (reporting); EECO derives
  kWh internally from the time step. **Only kW ever crosses the costing
  boundary.**
- Never introduce a variable named bare `power`, `energy`, or `work`. Always
  refer to the `flexcore.nomenclature` constants; a literal name string
  anywhere else is review-blocking.

The names were agreed to disambiguate the "work vs. power" inconsistencies in
IDAES/WaterTAP. The group may revisit the exact words, which is why they live in
one constants module.
