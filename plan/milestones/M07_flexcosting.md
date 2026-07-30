# M07 — FlexCosting

**Effort:** 3 days · **Depends on:** M04, M05, M06 · **Parallelizable:** no

## Goal

Build `FlexCosting`, the costing block that **wraps EECO** (via the M06
interface) and produces the project's **first end-to-end economic result**: a
pump+tank system that, solved as an LP against the M06 demo tariff, shifts
pumping to off-peak hours. FlexCosting subclasses IDAES costing for
registration/CapEx machinery and organizes every cost into **two sub-blocks it
owns**:

- **`opex`** — the operating-cost block, holding **all** operating cost:
  **electricity cost** and **fuel (gas) cost** (both delegated to EECO, M06) plus
  a **fixed operating cost** (user-defined; maintenance/labor/chemicals — *not*
  from EECO). Its `total_operating_cost` is the sum of the three.
- **`capex`** — the capital-cost block. In v0 it is an **empty placeholder**:
  `total_capital_cost == 0`. It exists so later milestones can aggregate capital
  costs from all unit models into it additively (M08 battery, M16 design). **No
  per-unit capex correlations are built in M07.**

FlexCosting hands the aggregated unit electrical work (a kW series) plus the
tariff to EECO (which builds the convex-relaxed operating-cost Expressions on the
`opex` block for the objective), re-exposes the result under clear flex-pse names
(decision R4), and — post-solve — extracts the realized aggregate power and calls
EECO's post-hoc evaluator to produce the **reported**, categorized cost breakdown
via `report_cost(model)` — capital vs. operating, each itemized (R4/R9, §6
reporting rule).

**Capital cost enters the objective only in design mode.** The operations-mode
objective is `aggregate_operating_cost` alone (the `opex` total; API-freeze
script). The design-mode objective is `aggregate_operating_cost +
aggregate_capital_cost`. Because the `capex` block is empty in M07
(`aggregate_capital_cost == 0`), the two coincide numerically here — but the
boundary is defined so that when a sizing-capable unit later registers a capital
cost, its capex equations appear in the objective **only** when
`set_design_mode()` is active.

**Restating R4** (architecture §3.6 — read the original): `FlexCosting`
subclasses `FlowsheetCostingBlockData` but **delegates all tariff operating cost
to the external EECO package**, in two ways. It does not re-implement price
series, demand-charge epigraphs, or cost math — it calls the `flexops.costing`
(`opex.py`) bridge: `add_electricity_cost` / `add_gas_cost` (in-objective, M06,
built **onto the `opex` block**) and `evaluate_cost` / `evaluate_gas_cost`
(post-solve, M06). FlexCosting's own jobs are: (1) aggregate registered units'
`power_electrical[t]` into the kW series EECO consumes (in-model *and* as the
post-solve numpy array); (2) build the **`opex` block** — electricity + fuel +
fixed operating cost — and map its total into the IDAES aggregate name
`aggregate_operating_cost` under a stable flex-pse name; (3) build the **empty `capex` block** and map its (zero)
total into `aggregate_capital_cost`, keeping capex out of the objective except in
design mode; (4) provide `report_cost(model)` — a categorized cost breakdown
(capital vs. operating, each itemized), never the raw objective (§6/M13); (5)
design/operations modes. EECO receives kW and converts to kWh internally.

**DR is containers-only in v0** (architecture §2.4/§3.6): `FlexCosting` holds a
`dr` placeholder attribute fed by `CostingConfig.dr` and a no-op `_build_dr()`
hook; it builds **no** DR constraints. Turning DR on later is additive.

**Not here: multi-period design.** Merging several representative months into one
model with sizing vars tied across periods is the **M16 design wrapper**
(`flexops.design`, architecture §3.6). `set_design_mode()` in this milestone
remains the **single-model** CapEx-active mode (one TimeBlock, capex constraints
active, sizing vars unfixed) — do not build any multi-period merging here.

**Not here: per-unit capex correlations.** The `costing/unit_models/unit_costing.py`
correlations, the Tank capital-cost/sizing-var registration, and the
cross-unit capex aggregation are **deferred** — the `capex` block is an empty
placeholder in M07 (see the Capex-scope decision in the PR). The first unit to
populate it is M08's battery.

## Read first

- `plan/01_architecture.md` §3.6 (costing wraps EECO — the whole section, especially R4, the in-objective + post-solve `report_cost` split, DR-containers-only, the construction-order invariant, and the note that multi-period design is the M16 wrapper not this mode), §2.4 (the EECO decision: two ways, DR containers), §6 (reporting rule R9: report `report_cost`, never the objective), §4 (energy nomenclature: EECO receives kW only), §3.2 (`register_power`), R4/R9 in §7
- `plan/milestones/M06_eeco_integration.md` — the wrapper API you consume: `load_tariff`, `load_dr_program`, `add_operating_cost` (the facility umbrella over `add_electricity_cost` + `add_gas_cost`), `OperatingCostHandles`, `evaluate_cost` / `evaluate_gas_cost` (post-hoc), `DRConfig`, and the demo tariff fixture + golden-bill semantics. Note the M06 "Total operating cost" scope note: the user-facing operating cost is **electricity + gas + user-defined fixed operating cost**; M06 delivered the two EECO pieces and deferred fixed operating cost — **M07 adds it**
- `plan/00_conventions.md` §4 (config rules: fixed operating cost is user-configured, so it goes through a documented pydantic field in `flexcore.config` **and** a `description=`'d ConfigDict entry — no bare kwargs), §9 (never delete Pyomo components)
- `plan/02_testing_and_ci.md` §1–§2, §5, §1a (constraint-body checks; test-first; the unit-model harness does **not** apply — FlexCosting is not a unit model)
- `plan/03_documentation.md` §1 (where `costing.rst` and `how_to/build_a_plant.md` live)

## Files to create or modify

- `src/flexcore/config/schema.py` — **modify**: add `fixed_operating_cost` to `CostingConfig`. This is the only flexcore change in M07.
- `src/flexops/costing/__init__.py` — **modify**: also export `FlexCosting` and the `report_cost` breakdown dataclasses (`CostReport`, `OperatingCostBreakdown`, `CapitalCostBreakdown`), so M13/downstream can consume them
- `src/flexops/costing/flex_costing.py` — the costing block (owns the `opex` and `capex` sub-blocks) and the `CostReport` breakdown dataclasses (§6)
- `src/flexops/core/ops_block.py` — **modify**: `register_power` forwards to the costing package when `costing_package=` was given (feeds the `opex` electricity aggregation)
- `src/flexops/__init__.py` — **modify**: export `FlexCosting` (`fo.FlexCosting`, per the API-freeze script)
- `src/flexops/tests/costing/test_flex_costing.py`, `test_load_shifting_component.py` — tests (reuse the M06 tariff/DR fixtures under `src/flexops/tests/fixtures/`)
- `docs/reference/flexops/costing.rst`, `docs/how_to/build_a_plant.md`, `CHANGELOG.md` — docs

*Not modified in M07 (deferred with the capex block):*
`src/flexops/costing/unit_models/unit_costing.py` (per-unit capex correlations)
and `src/flexops/unit_models/storage_tank.py` (capacity as a sizing var). Leave
the `costing/unit_models/` directory as-is.

## Specification

### 1. Class and configuration

```python
@declare_process_block_class("FlexCosting")
class FlexCostingData(FlowsheetCostingBlockData):   # FlowsheetCostingBlockData imported directly from idaes.core (R12)
    CONFIG entries (all with description=):
      time_block            — required; the fo.TimeBlock instance
      tariff_file           — path to an EECO tariff file  } exactly one of these two;
      tariff                — an EECO tariff object          }  both/neither → FlexConfigError
      dr_event_file         — optional path to an EECO DR program file (v0: loaded into
                              a container only — no DR constraints built)
      fixed_operating_cost  — optional float, default 0.0; the fixed operating cost
                              ($ over the horizon: maintenance, labor, chemicals).
                              A Layer-2 runtime entry populated from
                              CostingConfig.fixed_operating_cost (Layer-1) when
                              config-driven. Distinct from the tariff's EECO
                              fixed_charge (which is part of electricity cost).
```

The DR config comes from `CostingConfig.dr` (architecture §2.4/§3.6) when
config-driven; `dr_event_file` is the thin-path equivalent. Either way it feeds
the `dr` placeholder + no-op `_build_dr()` hook below. `fixed_operating_cost`
likewise comes from `CostingConfig.fixed_operating_cost` when config-driven and
from the constructor keyword on the thin path (data flows one way, conventions
§4: pydantic field → ConfigDict entry → built model).

### 2. On `build()` (construction time)

- Load the tariff via `flexops.costing.load_tariff(tariff_file or tariff)`; load
  the DR program via `load_dr_program(dr_event_file)` if given into a
  `flexops.costing.DRConfig` container. Store the tariff and the DR container
  (`self.dr` placeholder attribute) as attributes. Do **not** call EECO's cost
  builders yet, do **not** build the `opex`/`capex` sub-blocks yet, and do **not**
  build any DR constraints (containers-only, §2.4).
- Initialize the empty registries (`self._registered_power = []`,
  `self._registered_sizing = []`). Both stay empty in M07's headline (a tank does
  not draw power and its capex is deferred; only the pump registers power); the
  sizing registry exists so M08's battery reuses it (§3).
- **Build no aggregation, no cost, and no sub-blocks here.** The
  construction-order invariant (architecture §3.6) is that FlexCosting may be
  constructed before any units exist *because all aggregation, both sub-blocks,
  and the EECO call are deferred to `cost_process()`*. This is what lets the
  API-freeze script construct `m.costing` before `m.svcw.tank` (PLAN.md §2).

### 3. Energy + sizing registration mechanism (mirrors the WaterTAP pattern)

FlexCosting keeps its own registries, populated as units are constructed:

- `FlexCostingData.register_unit_power(unit, var, kind)` appends
  `(unit, var, kind)` to `self._registered_power`.
- `FlexCostingData.register_sizing_variable(var, capex_constraint=None)` appends
  to `self._registered_sizing` (names are implementer's choice; keep them methods
  on FlexCosting so M08's battery reuses them). **Nothing calls this in M07** —
  it is wiring for the (deferred) capex work, and the empty registry is what makes
  the `capex` block empty and the modes no-ops (§5).
- **Modify `OpsBlockData.register_power`** (M03): after its existing
  bookkeeping, if `self.config.costing_package` is not None, call
  `costing_package.register_unit_power(self, var, kind)`. Units built without
  `costing_package=` still work standalone (M04 tests must stay green — the
  forwarding is strictly conditional).
- **Tank is not modified in M07.** Registering `capacity` as a sizing var
  with a capex correlation is deferred with the capex block (see "Not here" above).

### 4. `cost_process()` builds (in order)

1. `aggregate_electrical_power[t]` — `Expression`, kW: sum of `var[t]` over
   registered `kind="electrical"` entries (include an explicit `0 * pyunits.kW`
   term so the Expression always exists even with an empty registry). Likewise
   `aggregate_thermal_power[t]` for `kind="thermal"`. These live on `self` (the
   physical aggregation), not on a sub-block. `aggregate_electrical_power` is what
   feeds EECO's electricity leg and `report_cost`. **`aggregate_thermal_power` is
   a kW heat duty — it is NOT the fuel-cost input.** EECO's fuel/gas leg consumes
   a *gas-usage series in EECO gas units* (m³/hr), a different quantity (M06 memory).
   No M07 unit registers gas usage, so the fuel leg is not built and
   `opex.fuel_cost` is `0` (step 2); wiring `aggregate_thermal_power` into the gas
   cost would be a units bug.
2. **Build the `opex` block via `flexops.costing` (`opex.py`).** Create
   `self.opex` (a Pyomo `Block`) and build the electricity and fuel costs **onto
   it** with the M06 `opex.py` bridge — the sole EECO import point. FlexCosting
   writes no tariff cost math of its own. Use the single-utility builders so each
   leg maps one-to-one to a named line item:
   ```python
   dt_hours = pyo.value(pyunits.convert(time_block.dt, to_units=pyunits.hr))
   elec = flexops.costing.add_electricity_cost(
       block=self.opex,
       electrical_power=self.aggregate_electrical_power,
       time_index=time_block.datetime_index,
       dt_hours=dt_hours,
       tariff=self._tariff,
       dr_config=self.dr,   # containers-only in v0; None-safe
   )
   # fuel/gas leg only when a gas-usage series is registered (absent for pump+tank):
   gas = flexops.costing.add_gas_cost(block=self.opex, gas_power=..., ...)  # else None
   ```
   Each builder asks EECO to build the **convex-relaxed** in-objective cost
   `Expression`s on `self.opex` (EECO namespaces its components by utility —
   `electric_*` / `gas_*` — so both legs coexist on one block) and returns an
   `OperatingCostHandles`. (The `add_operating_cost` umbrella is equivalent — it
   builds both legs on one block and returns *combined* handles whose
   `total_operating_cost` is pre-summed across utilities; the single-utility
   builders are preferred here precisely because they keep the two line items
   separate. Either way the Expressions are built by `opex.py` on `self.opex`,
   never re-derived in FlexCosting.) Then expose the line items under clear
   flex-pse names on the `opex` block:
   - `opex.electricity_cost` — `elec.total_operating_cost` (the electric leg total).
   - `opex.fuel_cost` — `gas.total_operating_cost` when the gas leg was built,
     else a `0` expression (mirrors the `0*kW` power placeholder).
   - `opex.fixed_operating_cost` — a scalar `Param` in dollars set from the
     `fixed_operating_cost` config entry, made dimensionally consistent with
     EECO's cost expressions (§8).
   - `opex.total_operating_cost` — `Expression` = `electricity_cost + fuel_cost +
     fixed_operating_cost`.

   The only arithmetic FlexCosting adds is the `total_operating_cost` sum and the
   constant fixed term. Then call the no-op `self._build_dr()` hook
   (containers-only — builds nothing in v0).
3. `aggregate_operating_cost` — scalar `Expression` on `self` mapped into the
   IDAES aggregate naming: `= self.opex.total_operating_cost`. This is the name
   the operations-mode objective and downstream code use (API-freeze script:
   `expr=m.costing.aggregate_operating_cost`).
4. **Build the empty `capex` block.** Create `self.capex` (a Pyomo `Block`) with
   `capex.total_capital_cost` = an `Expression` summing registered units'
   `capital_cost` — which, with an empty `_registered_sizing`, is just an explicit
   `0 * pyunits.USD` placeholder term (so the Expression always exists). Then
   `aggregate_capital_cost` — scalar `Expression` on `self` = `capex.total_capital_cost`
   (`0` in M07). Finally call `set_operations_mode()` as the default final state
   (scheduling first).
5. **Objective composition (the design-mode rule).** Expose the design-mode total
   as a convenience `Expression`, `total_cost = aggregate_operating_cost +
   aggregate_capital_cost`. The operations-mode objective must remain
   `aggregate_operating_cost` alone (API-freeze); the design-mode objective is
   `total_cost`. Capital cost therefore enters the objective **only in design
   mode** — in M07 both reduce to the same value because `aggregate_capital_cost`
   is `0`, but the boundary is defined for when the capex block is later populated.
   Whether to also invoke the parent class's aggregation machinery is
   implementer's choice — the required contract is that
   `aggregate_operating_cost` exists and equals `opex.total_operating_cost`,
   `aggregate_capital_cost` exists and equals `0`, and the built model classifies
   **LP** (`flexcore.solvers.classify`).

### 5. Design/operations modes

- `set_operations_mode()` — fix every registered sizing Var at its current value
  and `deactivate()` every capex constraint registered with the sizing vars. The
  operations-mode objective is `aggregate_operating_cost` only.
- `set_design_mode()` — unfix those sizing Vars and `activate()` those capex
  constraints. The design-mode objective is `total_cost` (opex **+** capex).
- Both are idempotent and callable any time after `cost_process()`.
- **In M07 both registries are empty**, so both setters iterate over empty
  collections and are effectively no-ops — the model classifies **LP in both
  modes** (there is no sizing var to unfix and no capex constraint to activate).
  The setters exist as the documented single-model toggle that later milestones
  populate; the design-mode-objective rule is defined now so it is not retrofitted.
- Neither touches EECO's `opex` cost components — those are load-driven and
  mode-independent.
- **Single-model only.** `set_design_mode()` here activates capex on the one
  model; merging multiple representative months and equality-linking sizing vars
  across them is the **M16 design wrapper** (`flexops.design`, architecture §3.6),
  not this mode. Do not build multi-period merging in M07.
- **Design-mode nonlinearity returns with the first sizing unit.** No unit
  registers a sizing var in M07, so design mode adds no bilinear terms and stays
  LP. When a sizing-capable unit later registers capex (e.g. the tank, whose
  `level_definition` constraint `volume[t] == level[t] * capacity` becomes a
  Var×Var product once `capacity` is unfixed — M04), a model containing it in
  design mode will classify **NLP**, to be solved with IPOPT or a
  `flexschedule.SolveSequence` (R5). That is future work, not M07.

### 6. `report_cost(model) -> CostReport` — the REPORTED, categorized cost (R4/R9)

The user-facing cost, produced **post-solve** and **never** the raw solver
objective (architecture §6 reporting rule; M13 surfaces it to callers).
`report_cost` returns a **full categorized breakdown** — capital vs. operating,
each itemized — as documented dataclasses (explicit named fields, not a bare
dict; conventions style):

```python
@dataclasses.dataclass
class OperatingCostBreakdown:
    electricity: float     # EECO post-hoc electricity bill on the realized aggregate power
    fuel: float            # EECO post-hoc gas bill (0 in v0: no gas-consuming unit)
    fixed: float           # the config fixed operating cost (a constant)
    dr_revenue: float      # DR incentive credit (>= 0), subtracted; 0 in v0 (DR containers-only)
    total: float           # electricity + fuel + fixed - dr_revenue

@dataclasses.dataclass
class CapitalCostBreakdown:
    by_component: dict[str, float]   # unit block name -> capital cost; {} in v0 (empty capex)
    total: float                     # sum of by_component; 0 in v0

@dataclasses.dataclass
class CostReport:
    operating: OperatingCostBreakdown
    capital: CapitalCostBreakdown
    total: float           # operating.total + capital.total
```

How each field is produced:

- **Operating — electricity & fuel** are EECO **post-hoc** evaluations on the
  realized dispatch. Extract the realized aggregate electrical power as a
  time-indexed numpy array — `np.array([pyo.value(self.aggregate_electrical_power[t])
  for t in time_block.time_index])`, kW, ordered by `time_block.time_index` — and
  the realized gas usage likewise when a fuel leg exists. Compute `dt_hours =
  pyo.value(pyunits.convert(time_block.dt, to_units=pyunits.hr))`. Then
  `electricity = flexops.costing.evaluate_cost(realized_power, self._tariff,
  dt_hours, dr_config=self.dr, time_index=time_block.datetime_index)` and
  `fuel = flexops.costing.evaluate_gas_cost(...)` (or `0` when no gas leg — the
  pump+tank case). **`time_index=time_block.datetime_index` is required** — EECO's
  charge windows are keyed on month/weekday/hour, so a bare array cannot reproduce
  a real (e.g. July) bill (M06 deviation; see the M06 EECO memory).
- **Operating — fixed** = `pyo.value(self.opex.fixed_operating_cost)` (the config
  constant).
- **Operating — dr_revenue** = `0.0` in v0 (DR is containers-only, §7). The field
  exists so the breakdown is stable when DR turns on later; **do not fabricate a
  credit** — a DR file present must not change any reported number.
- **Capital** — read each registered unit's `capital_cost` value off the `capex`
  block, keyed by unit block name into `by_component`; `{}` and `total = 0` in v0
  (the `capex` block is empty). This is a plain model read (the sizing is fixed in
  operations mode and solved in design mode), not an EECO call.
- `total` fields are the obvious sums.

`report_cost` writes no cost math and builds no Pyomo components; it only reads
the solved model and calls M06's evaluators. The reported **operating total** —
never `value(model.objective)` — is the user-facing operating cost (§6/M13); it
is an **independent** post-hoc recomputation on the realized dispatch (a
convex-relaxed and possibly scalarized objective is not the bill — R4/R9). The
two **can diverge** and the divergence is real: once EECO's convex relaxation
drops the tiered surcharge (consumption above the monthly tier limit), the
relaxed objective is strictly below the post-hoc bill (M06
`test_relaxed_leq_or_approx_true`). On a short horizon that never reaches the
tier the relaxation is tight and they numerically **coincide** — so tests assert
the reporting rule by recomputing `report_cost` independently, not by asserting a
strict inequality that only holds under the tier. Point the docstring at the §6 /
M13 reporting rule.

**v0 shape.** In M07 the report always carries every category, with the deferred
ones zero: `electricity` from EECO, `fuel = 0`, `fixed` from config,
`dr_revenue = 0`, and an empty `capital` breakdown. The structure is complete and
stable, so fuel/DR/capital light up additively as those features land.

### 7. `_build_dr()` — no-op DR hook (containers-only, v0)

- `FlexCosting` has a `dr` placeholder attribute (the loaded `DRConfig`
  container, §2/`build()`) and a `_build_dr()` hook called at the end of
  `cost_process()` step 2. In v0 it **does nothing** but validate/store the
  container — no DR event, curtailment, incentive, or capacity constraints
  (architecture §2.4/§3.6; PLAN §4). It exists so later DR work is additive.

### 8. Fixed operating cost (config-driven; distinct from the tariff)

- **Layer-1 (persisted).** Add `fixed_operating_cost: float = 0.0` to
  `flexcore.config.schema.CostingConfig`, with a `description` (conventions §4):
  the fixed operating cost in dollars over the horizon — non-tariff operating
  costs such as maintenance, labor, and chemicals. Default `0.0` keeps every
  existing config valid.
- **Layer-2 (runtime).** The `fixed_operating_cost` ConfigDict entry on
  `FlexCosting` (§1) receives it; `cost_process()` builds `opex.fixed_operating_cost`
  as a scalar dollar `Param` from that value and includes it in
  `opex.total_operating_cost`.
- **Not the tariff's fixed charge.** EECO's `fixed_charge` (e.g. $150/month in the
  demo tariff) is a *tariff* line item and is already inside `opex.electricity_cost`
  via EECO. `opex.fixed_operating_cost` is the *facility's* non-tariff fixed cost.
  Keep them separate and never double-count (Pitfall 11).
- Per-unit **capital** cost correlations (`unit_costing.py`) remain deferred — the
  `capex` block is empty (Capex-scope decision). Do not add them in M07.

### Worked example (the headline component test)

24 hourly steps covering 2025-07-08 (a summer Tuesday: peak 16:00–21:00 in the
demo tariff). Pump (`energy_intensity=0.5` kWh/m³, inlet `flow_vol_phase[t, "Liq"]`
bounded [0, 300] m³/hr) → Arc → Tank (`max_volume=1000`, `initial_volume=200`, outlet flow
fixed at 100 m³/hr). The **pump** is built with `costing_package=m.costing` (so it
registers `power_electrical`); the tank has no costing package (its capex is
deferred). `m.costing.cost_process()`; objective =
`pyo.Objective(expr=m.costing.aggregate_operating_cost)` (operations mode —
opex only); add a test-local terminal constraint `volume[23] >= 200` (else the LP
drains the tank). Optimal behavior: zero pumping during the five peak hours (tank
capacity comfortably covers the 500 m³ peak demand), with the anytime demand
charge flattening the off-peak profile. With no gas and `fixed_operating_cost=0`,
`opex.total_operating_cost` is exactly EECO's relaxed electricity cost. This
worked example stays in **operations mode**, and because the `capex` block is
empty, the model classifies **LP** in both modes.

## Pitfalls

1. **Aggregating or costing at build time.** Anything summed over units — or any
   EECO call, or building the `opex`/`capex` sub-blocks — in `build()` breaks the
   construction-order invariant; the permutation test exists to catch this. Defer
   everything to `cost_process()`.
2. **Re-implementing EECO in FlexCosting.** If you write a price loop or a demand
   epigraph here — in `cost_process` **or** in `report_cost` — you have
   duplicated M06/EECO. Aggregate the kW series and call `add_operating_cost`
   (in-objective) / `evaluate_cost` (post-solve); the tariff cost math is not
   yours. The only arithmetic FlexCosting adds is the `total_operating_cost` sum
   and the constant fixed term.
3. **Passing kWh instead of kW to EECO.** EECO does the kW→kWh conversion with
   `dt_hours`; hand it the raw kW `aggregate_electrical_power` and the timestep,
   never a pre-integrated energy series (double-counts on non-hourly grids).
4. **IDAES `cost_process` collisions.** `FlowsheetCostingBlockData` has its own
   `cost_process`/aggregate machinery; if the parent call fights the flex names
   (`aggregate_*`, the `opex`/`capex` sub-blocks), build flex-native components
   first and skip/override the conflicting parent step — record what you did under
   "Deviations from spec".
5. **Breaking costing-less units.** M04 constructs Pump/Tank with no
   `costing_package`; the `register_power` forwarding must be strictly
   conditional.
6. **Objective referencing EECO internals.** The operations objective must use
   `aggregate_operating_cost` (flex-pse name), never a raw EECO handle — that
   naming boundary is the whole point of R4.
7. **Reporting the objective as the bill.** The user-facing cost is
   `report_cost(model).operating.total` (EECO post-hoc on the realized power, plus
   fuel + fixed − DR revenue), never `value(model.objective)` — the objective is a
   relaxed/scalarized proxy. It equals the bill only when the relaxation is tight;
   it is strictly below the bill once the tiered surcharge is reached (M06). Tests
   encode the rule by recomputing `report_cost` **independently** of the
   objective — not by assuming the two are always equal, nor always unequal.
14. **Fabricating DR revenue or capital in the report.** In v0 `report_cost`'s
    `dr_revenue` is `0` (DR containers-only) and its `capital` breakdown is empty
    (`{}` / `0`). Do not synthesize a DR credit or per-unit capex — a DR file
    present must not change any reported number, and no unit registers capex yet.
8. **Building DR constraints.** DR is containers-only (§2.4). `_build_dr()` is a
   no-op; supplying a DR file must not add DR constraints or change the objective.
9. **Multi-period design in M07.** `set_design_mode()` is single-model
   CapEx-active only. Merging representative months / linking sizing vars is M16.
10. **Regression constant laundering.** The stored objective baseline must be a
    literal constant in the test with a comment naming the run that produced it —
    never recomputed from the model.
11. **Double-counting the fixed charge.** EECO's tariff `fixed_charge` lives
    inside `opex.electricity_cost`; `opex.fixed_operating_cost` is the separate
    non-tariff facility cost. Adding the tariff fixed charge again as a fixed
    operating cost double-counts it.
12. **Capex in the operations objective.** `aggregate_capital_cost` must stay out
    of the operations-mode objective (`aggregate_operating_cost` only). Capex is
    in the objective **only** via `total_cost` in design mode.
13. **Populating the capex block.** In M07 the `capex` block is an empty
    placeholder (`total_capital_cost == 0`). Do not add per-unit capex
    correlations or sizing-var registration here — that is deferred (M08/M16).

## Tests

Test-first (02 §1a). Reuse the M06 fixtures.

- `src/flexops/tests/costing/test_flex_costing.py`:
  - `test_config_exclusivity` — `@pytest.mark.unit`. Both or neither of `tariff_file`/`tariff` → `FlexConfigError` naming the options.
  - `test_construct_before_units` — `@pytest.mark.unit`. `FlexCosting` builds on a bare model with a TimeBlock and no units; `cost_process()` runs; `aggregate_electrical_power` exists (its body is the `0*kW` placeholder) and both `opex` and `capex` sub-blocks exist.
  - `test_aggregate_electrical_power` — `@pytest.mark.unit`. Pump+tank with `costing_package`; after `cost_process()`, fix a known flow profile and assert `value(aggregate_electrical_power[t])` equals the sum of registered units' `power_electrical[t]` at several t (pure `pyo.value`, no solve).
  - `test_opex_block_line_items` — `@pytest.mark.unit`. After `cost_process()`, the `opex` block exposes `electricity_cost`, `fuel_cost`, `fixed_operating_cost`, and `total_operating_cost`; with no gas registered, `fuel_cost` evaluates to `0`; `value(opex.total_operating_cost) == value(electricity_cost) + value(fuel_cost) + value(fixed_operating_cost)` (pure `pyo.value`, no solve).
  - `test_fixed_operating_cost_flows_through` — `@pytest.mark.unit`. Build two costing blocks, `fixed_operating_cost=0.0` and `fixed_operating_cost=1234.0`; assert the difference in `value(aggregate_operating_cost)` (fixed flow held identical) is exactly `1234.0`, and that it is **not** part of `opex.electricity_cost` (distinct from the tariff `fixed_charge`).
  - `test_operating_cost_is_eeco_total` — `@pytest.mark.unit`. With `fixed_operating_cost=0` and no gas, assert `aggregate_operating_cost` evaluates equal to `handles.total_operating_cost` — i.e. FlexCosting exposes EECO's total, not a re-derived one, and maps it through `opex.total_operating_cost`.
  - `test_capex_block_empty` — `@pytest.mark.unit`. After `cost_process()`, `capex` exists, `value(capex.total_capital_cost) == 0`, and `value(aggregate_capital_cost) == 0`.
  - `test_report_cost_breakdown_shape` — `@pytest.mark.unit`. Build pump+tank+costing with `fixed_operating_cost=500.0` and `dr_event_file=dr_events_demo.json`; after `cost_process()`, fix a known flow profile (no solve) and call `report_cost(m)`. Assert it returns a `CostReport` whose fields are the documented categories: `operating.electricity` a float, `operating.fuel == 0`, `operating.fixed == 500.0`, `operating.dr_revenue == 0` (DR containers-only — the loaded DR file must not produce a credit), `capital.by_component == {}`, `capital.total == 0`, and every `total` equals the sum of its parts. This pins the breakdown structure and the v0 zero placeholders without needing a solver.
  - `test_capex_excluded_from_operations_objective` — `@pytest.mark.unit`. `aggregate_operating_cost` equals `opex.total_operating_cost` and does **not** include `aggregate_capital_cost`; the convenience `total_cost` equals `aggregate_operating_cost + aggregate_capital_cost` (so capex reaches the objective only via the design-mode `total_cost`).
  - `test_mode_toggles` — `@pytest.mark.unit`. After `cost_process()`: `set_design_mode()` then `set_operations_mode()` run without error and are idempotent (toggle twice). With M07's empty registries there is no sizing var to unfix and no capex constraint to (de)activate; assert `flexcore.solvers.classify` returns `LP` in **both** modes (empty capex adds no nonlinearity). (Single-model mode only — no multi-period merging; that is M16.)
  - `test_construction_order_permutation` — `@pytest.mark.unit`. Build the pump+tank+costing system in ≥ 2 component-creation orders (costing first vs. costing just before `cost_process`; pump-then-tank vs. tank-then-pump), fix the same flow profile, assert `value(aggregate_operating_cost)` identical (`pytest.approx(rel=1e-12)`).
  - `test_dr_container_loads_noop` — `@pytest.mark.unit`. Build `FlexCosting` with `dr_event_file=dr_events_demo.json`; assert `self.dr` is a populated `DRConfig` container and that after `cost_process()` **no** DR constraints were built (component count / classification unchanged vs. the no-DR build). The DR hook is a no-op (containers-only, §2.4).
  - `test_model_classifies_lp` — `@pytest.mark.unit`. Built pump+tank+costing model → `flexcore.solvers.classify` returns `LP` (no `max()`/nonlinearity from the EECO bridge, empty capex).
- `src/flexops/tests/costing/test_load_shifting_component.py` (each
  `@pytest.mark.component` + `@pytest.mark.needs_highs`, < 10 s):
  - `test_load_shifting_headline` — the worked example; assert optimal termination; `sum(value(pump.inlet_state.flow_vol_phase[t, "Liq"]) for t in peak hours 16–20) == pytest.approx(0.0, abs=1e-6)`; `value(m.objective) == pytest.approx(EXPECTED_OBJECTIVE, rel=1e-6)` where `EXPECTED_OBJECTIVE` is a stored module constant recorded from the first verified run (regression baseline — changing it is a deliberate diff).
  - `test_report_cost_post_hoc` — after the headline solve, `m.costing.report_cost(m)` returns a `CostReport`. Assert its categorized shape and that it is an **independent** post-hoc recomputation: `report.operating.electricity == pytest.approx(evaluate_cost(realized_power, tariff, dt_hours, time_index=...))` (recomputed in the test straight from the realized aggregate power — a stored regression constant), `report.operating.fuel == 0`, `report.operating.fixed == 0`, `report.operating.dr_revenue == 0`, `report.operating.total == pytest.approx(report.operating.electricity)`, `report.capital.by_component == {}` and `report.capital.total == 0`, `report.total == pytest.approx(report.operating.total)`. This encodes the reporting rule (§6): the bill is recomputed post-hoc, never read off the objective. (On this short horizon the relaxation is tight so the value happens to coincide with `value(m.objective)`; do **not** assert a strict inequality — that only holds once the tiered surcharge is reached, which M06's `test_relaxed_leq_or_approx_true` covers.)
  - `test_demand_charge_reduces_peak` — solve the same system twice: demo tariff vs. a copy with demand charges removed. Assert `max_t value(aggregate_electrical_power[t])` is strictly lower with demand charges.

(Note: the tariff *math* — price alignment, epigraph correctness, kWh
conversion, the golden bill, and the post-hoc `evaluate_cost` accuracy — is
EECO's and is tested in M06. M07 tests the FlexCosting wrapper: aggregation, the
opex/capex block structure, fixed operating cost, delegation, naming, modes and
the capex-in-objective-only-in-design-mode rule, `report_cost` plumbing, the DR
container no-op, and the end-to-end optimization behavior.)

## Documentation tasks

- `docs/reference/flexops/costing.rst` — extend the M06 costing page: autodoc
  `FlexCosting`; restate R4 in one paragraph (delegates OpEx to EECO two ways —
  in-objective relaxed cost + post-solve `report_cost`; owns aggregation, the
  `opex`/`capex` block structure, naming, modes; DR containers-only); document the
  **`opex` block** (electricity + fuel + fixed operating cost, and that the fixed
  operating cost is a non-tariff facility cost distinct from EECO's `fixed_charge`),
  the **empty `capex` block** placeholder and that capex enters the objective only
  in design mode, the `report_cost` reporting rule (autodoc the `CostReport` /
  `OperatingCostBreakdown` / `CapitalCostBreakdown` dataclasses — the categorized
  capital-vs-operating breakdown it returns, never the objective — §6/M13, and
  note the v0 zero placeholders: fuel/DR revenue/capital), the mode-toggle API
  (single-model only; multi-period is M16), the construction-order invariant, and
  that tariff/cost math + limitations live with EECO/M06.
- `docs/how_to/build_a_plant.md` — **skeleton**: title, a code block walking
  pump → tank → `FlexCosting` → `cost_process()` → objective (essentially the
  headline test's model), and a note that the full guide becomes an executed
  notebook in M14.
- `CHANGELOG.md` "Unreleased": **"First end-to-end economic result: EECO-backed
  FlexCosting; operating costs grouped in an `opex` block (electricity + fuel +
  fixed), with an empty `capex` placeholder; pump+tank LP shifts load off-peak."**
- Class docstring per conventions §3.

## Definition of Done

- [ ] `fo.FlexCosting(time_block=..., tariff_file=..., dr_event_file=...)` constructs exactly as in the API-freeze script (PLAN.md §2), before any units exist
- [ ] `cost_process()` aggregates `power_electrical` into a kW series and builds the **`opex` block** with `electricity_cost` (EECO), `fuel_cost` (EECO gas leg; `0` when no gas), and `fixed_operating_cost` (config), exposing `aggregate_operating_cost == opex.total_operating_cost`; no tariff cost math written in FlexCosting
- [ ] `fixed_operating_cost` is a `description`'d `CostingConfig` field (default `0.0`) + a FlexCosting ConfigDict entry; it flows into `opex.total_operating_cost`, is kept distinct from the tariff `fixed_charge`, and defaults leave existing configs valid
- [ ] `cost_process()` builds the **empty `capex` block** (`total_capital_cost == 0`), exposes `aggregate_capital_cost == 0`, and capital cost enters the objective **only in design mode** (`total_cost`; operations objective stays `aggregate_operating_cost`)
- [ ] `report_cost(model)` returns a categorized `CostReport` — `operating` (electricity via EECO post-hoc with `time_index=time_block.datetime_index`, fuel, fixed, dr_revenue) and `capital` (by-component + total), each summing to its total. In v0 fuel/dr_revenue/capital are zero placeholders; the headline test asserts `operating.total` equals an **independent** EECO post-hoc recomputation on the realized power (the reporting rule — never read off the objective; the two coincide only because the relaxation is tight on this horizon, R4/R9/§6)
- [ ] **DR is containers-only**: `dr` placeholder + no-op `_build_dr()` fed by the DR file/`CostingConfig.dr`; no DR constraints built; the container-loads test passes
- [ ] `set_design_mode()` / `set_operations_mode()` are single-model, idempotent toggles over the (empty in M07) sizing/capex registries; multi-period merging is M16, not built here
- [ ] Built model classifies **LP** via `flexcore.solvers.classify` in both modes (empty capex adds no nonlinearity)
- [ ] Construction-order permutation test passes; M04's costing-less unit tests still green
- [ ] Headline load-shifting component test passes under HiGHS in < 10 s with a stored objective regression constant; demand-charge and `report_cost` post-hoc tests pass
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W` passes with `costing.rst` and the `build_a_plant.md` skeleton
- [ ] CHANGELOG "first end-to-end" entry present
- [ ] plus the generic DoD in CLAUDE.md
