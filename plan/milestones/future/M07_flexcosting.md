# M07 — FlexCosting

**Effort:** 3 days · **Depends on:** M04, M05, M06 · **Parallelizable:** no

## Goal

Build `FlexCosting`, the costing block that **wraps EECO** (via the M06
interface) and produces the project's **first end-to-end economic result**: a
pump+tank system that, solved as an LP against the M06 demo tariff, shifts
pumping to off-peak hours. FlexCosting subclasses IDAES costing for
registration/CapEx machinery, aggregates unit electrical work into a kW series,
hands that plus the tariff to EECO (which builds the convex-relaxed
operating-cost Expressions for the objective), re-exposes the result under clear
flex-pse names (decision R4), and — post-solve — extracts the realized aggregate
power and calls EECO's post-hoc evaluator to produce the **reported** cost via
`report_cost(model)` (R4/R9, §6 reporting rule).

**Restating R4** (architecture §3.6 — read the original): `FlexCosting`
subclasses `FlowsheetCostingBlockData` but **delegates all tariff operating cost
to the external EECO package**, in two ways. It does not re-implement price
series, demand-charge epigraphs, or cost math — it calls
`flexops.costing.add_operating_cost` (in-objective, M06) and
`flexops.costing.evaluate_cost` (post-solve, M06). FlexCosting's own jobs are:
(1) aggregate registered units' `electrical_work[t]` into the kW series EECO
consumes (in-model *and* as the post-solve numpy array); (2) map EECO's outputs
into IDAES aggregate names (`aggregate_operating_cost`, `aggregate_capital_cost`)
under stable flex-pse names; (3) provide `report_cost(model)` — the user-facing
cost, never the raw objective (§6/M13); (4) CapEx + design/operations modes.
EECO receives kW and converts to kWh internally.

**DR is containers-only in v0** (architecture §2.4/§3.6): `FlexCosting` holds a
`dr` placeholder attribute fed by `CostingConfig.dr` and a no-op `_build_dr()`
hook; it builds **no** DR constraints. Turning DR on later is additive.

**Not here: multi-period design.** Merging several representative months into one
model with sizing vars tied across periods is the **M16 design wrapper**
(`flexops.design`, architecture §3.6). `set_design_mode()` in this milestone
remains the **single-model** CapEx-active mode (one TimeBlock, sizing vars
unfixed, CapEx terms active) — do not build any multi-period merging here.

## Read first

- `plan/01_architecture.md` §3.6 (costing wraps EECO — the whole section, especially R4, the in-objective + post-solve `report_cost` split, DR-containers-only, the construction-order invariant, and the note that multi-period design is the M16 wrapper not this mode), §2.4 (the EECO decision: two ways, DR containers), §6 (reporting rule R9: report `report_cost`, never the objective), §4 (energy nomenclature: EECO receives kW only), §3.2 (`register_energy`), R4/R9 in §7
- `plan/milestones/M06_eeco_integration.md` — the wrapper API you consume: `load_tariff`, `load_dr_program`, `add_operating_cost`, `OperatingCostHandles`, `evaluate_cost` (post-hoc), `DRConfig`, and the demo tariff fixture + golden-bill semantics
- `plan/02_testing_and_ci.md` §1–§2, §5, §1a (constraint-body checks; test-first; the unit-model harness does **not** apply — FlexCosting is not a unit model)
- `plan/03_documentation.md` §1 (where `costing.rst` and `how_to/build_a_plant.md` live)

## Files to create or modify

- `src/flexops/costing/__init__.py` — **modify**: also export `FlexCosting`
- `src/flexops/costing/flex_costing.py` — the costing block
- `src/flexops/costing/unit_costing.py` — per-unit capex correlations (v0 placeholders)
- `src/flexops/core/ops_block.py` — **modify**: `register_energy` forwards to the costing package when `costing_package=` was given
- `src/flexops/unit_models/storage_tank.py` — **modify**: register `capacity` as a sizing var with the costing package when present
- `src/flexops/__init__.py` — **modify**: export `FlexCosting` (`fo.FlexCosting`, per the API-freeze script)
- `src/flexops/tests/costing/test_flex_costing.py`, `test_load_shifting_component.py` — tests (reuse the M06 tariff/DR fixtures under `src/flexops/tests/fixtures/`)
- `docs/reference/flexops/costing.rst`, `docs/how_to/build_a_plant.md`, `CHANGELOG.md` — docs

## Specification

### 1. Class and configuration

```python
@declare_process_block_class("FlexCosting")
class FlexCostingData(FlowsheetCostingBlockData):   # via flexcore.compat.idaes
    CONFIG entries (all with description=):
      time_block      — required; the fo.TimeBlock instance
      tariff_file     — path to an EECO tariff file  } exactly one of these two;
      tariff          — an EECO tariff object         }  both/neither → FlexConfigError
      dr_event_file   — optional path to an EECO DR program file (v0: loaded into
                        a container only — no DR constraints built)
```

The DR config comes from `CostingConfig.dr` (architecture §2.4/§3.6) when
config-driven; `dr_event_file` is the thin-path equivalent. Either way it feeds
the `dr` placeholder + no-op `_build_dr()` hook below.

### 2. On `build()` (construction time)

- Load the tariff via `flexops.costing.load_tariff(tariff_file or tariff)`; load
  the DR program via `load_dr_program(dr_event_file)` if given into a
  `flexops.costing.DRConfig` container. Store the tariff and the DR container
  (`self.dr` placeholder attribute) as attributes. Do **not** call EECO's cost
  builders yet, and do **not** build any DR constraints (containers-only, §2.4).
- Initialize the empty registries (`self._registered_energy = []`,
  `self._registered_sizing = []`).
- **Build no aggregation and no cost here.** The construction-order invariant
  (architecture §3.6) is that FlexCosting may be constructed before any units
  exist *because all aggregation and the EECO call are deferred to
  `cost_process()`*. This is what lets the API-freeze script construct
  `m.costing` before `m.svcw.tank` (PLAN.md §2).

### 3. Energy + sizing registration mechanism (mirrors the WaterTAP pattern)

FlexCosting keeps its own registries, populated as units are constructed:

- `FlexCostingData.register_unit_energy(unit, var, kind)` appends
  `(unit, var, kind)` to `self._registered_energy`.
- `FlexCostingData.register_sizing_variable(var, capex_constraint=None)` appends
  to `self._registered_sizing` (names are implementer's choice; keep them
  methods on FlexCosting so M08's battery reuses them).
- **Modify `OpsBlockData.register_energy`** (M03): after its existing
  bookkeeping, if `self.config.costing_package` is not None, call
  `costing_package.register_unit_energy(self, var, kind)`. Units built without
  `costing_package=` still work standalone (M04 tests must stay green — the
  forwarding is strictly conditional).
- **Modify `StorageTank`**: when `costing_package=` is given, call
  `unit_costing.cost_storage_tank(...)` and register `capacity` as a sizing var.

### 4. `cost_process()` builds (in order)

1. `aggregate_electrical_work[t]` — `Expression`, kW: sum of `var[t]` over
   registered `kind="electrical"` entries (include an explicit `0 * pyunits.kW`
   term so the Expression always exists even with an empty registry). Likewise
   `aggregate_thermal_work[t]` for `kind="thermal"`.
2. **Delegate in-objective OpEx to EECO** — call the M06 bridge:
   ```python
   dt_hours = pyunits.convert(time_block.dt, to_units=pyunits.hr)
   handles = flexops.costing.add_operating_cost(
       block=self,
       electrical_work=self.aggregate_electrical_work,
       time_index=time_block.datetime_index,
       dt_hours=pyo.value(dt_hours),
       tariff=self._tariff,
       dr_config=self.dr,   # containers-only in v0; None-safe
   )
   ```
   EECO builds the **convex-relaxed** energy-cost / demand-charge Expressions on
   `self`; `handles` exposes them under clear flex-pse names. FlexCosting writes
   **no** cost math of its own here. Then call the no-op `self._build_dr()` hook
   (containers-only — builds nothing in v0). The `total_operating_cost` handle is
   the relaxed objective proxy, **not** the reported bill (that comes from
   `report_cost`, §7 below).
3. `aggregate_operating_cost` — scalar `Expression` mapped into the IDAES
   aggregate naming: `= handles.total_operating_cost`. This is the name the
   objective and all downstream code use (API-freeze script:
   `expr=m.costing.aggregate_operating_cost`).
4. Capex aggregation from `unit_costing` results into `aggregate_capital_cost`
   (an `Expression` summing registered units' `capital_cost`), then
   `set_operations_mode()` as the default final state (scheduling first;
   implementer's choice, documented).
   Whether to also invoke the parent class's aggregation machinery is
   implementer's choice — the required contract is that
   `aggregate_operating_cost` exists and equals `handles.total_operating_cost`,
   and the built model classifies **LP** (`flexcore.solvers.classify`).

### 5. Design/operations modes

- `set_operations_mode()` — fix every registered sizing Var at its current
  value and `deactivate()` every capex constraint built by `unit_costing`.
- `set_design_mode()` — unfix them and `activate()` the capex constraints.
- Both are idempotent and callable any time after `cost_process()`.
- Neither touches EECO's cost components — those are load-driven and mode-
  independent.
- **Single-model only.** `set_design_mode()` here activates CapEx on the one
  model; merging multiple representative months and equality-linking sizing vars
  across them is the **M16 design wrapper** (`flexops.design`, architecture §3.6),
  not this mode. Do not build multi-period merging in M07.

### 6. `report_cost(model) -> float` — the REPORTED electricity cost (R4/R9)

The user-facing cost, produced **post-solve** and **never** the raw solver
objective (architecture §6 reporting rule; M13 surfaces it to callers):

- After a solve, extract the realized **aggregate electrical power** as a
  time-indexed numpy array — `np.array([pyo.value(self.aggregate_electrical_work[t])
  for t in time_block.time_points])`, kW, ordered by `time_block.time_points`.
- Compute `dt_hours = pyo.value(pyunits.convert(time_block.dt,
  to_units=pyunits.hr))`.
- Return `flexops.costing.evaluate_cost(realized_power, self._tariff, dt_hours,
  dr_config=self.dr)` — EECO's post-hoc, de-relaxed evaluation on the fixed
  dispatch. This is the true bill; it legitimately **differs** from
  `value(model.objective)` (the objective is a convex-relaxed and possibly
  scalarized proxy — R4/R9).
- `report_cost` writes no cost math and builds no Pyomo components; it only reads
  the solved model and calls M06's `evaluate_cost`. Point the docstring at the §6
  / M13 reporting rule: downstream reporting uses `report_cost`, never the
  objective value.

### 7. `_build_dr()` — no-op DR hook (containers-only, v0)

- `FlexCosting` has a `dr` placeholder attribute (the loaded `DRConfig`
  container, §2/`build()`) and a `_build_dr()` hook called at the end of
  `cost_process()` step 2. In v0 it **does nothing** but validate/store the
  container — no DR event, curtailment, incentive, or capacity constraints
  (architecture §2.4/§3.6; PLAN §4). It exists so later DR work is additive.

### 8. `unit_costing.py` (v0 placeholders — CapEx is flex-pse's, not EECO's)

- Module constants `TANK_CAPEX_USD_PER_M3 = 50.0`,
  `BATTERY_CAPEX_USD_PER_KWH = 300.0` (placeholders — implementer's choice; M08
  consumes the battery one).
- `cost_storage_tank(...)` / `cost_battery(...)`: add a `capital_cost` Var and
  constraint `capital_cost == rate * capacity` on the unit's costing block, and
  register the sizing var + capex constraint with FlexCosting. Exact signatures
  are implementer's choice; keep them one-liner-simple.

### Worked example (the headline component test)

24 hourly steps covering 2025-07-08 (a summer Tuesday: peak 16:00–21:00 in the
demo tariff). Pump (`energy_intensity=0.5` kWh/m³, `flow_vol` bounded [0, 300]
m³/hr) → Arc → StorageTank (`max_volume=1000`, `initial_volume=200`, outlet flow
fixed at 100 m³/hr), built with `costing_package=m.costing`;
`m.costing.cost_process()`; objective =
`pyo.Objective(expr=m.costing.aggregate_operating_cost)`; add a test-local
terminal constraint `V[23] >= 200` (else the LP drains the tank). Optimal
behavior: zero pumping during the five peak hours (tank capacity comfortably
covers the 500 m³ peak demand), with the anytime demand charge flattening the
off-peak profile.

## Pitfalls

1. **Aggregating or costing at build time.** Anything summed over units — or any
   EECO call — in `build()` breaks the construction-order invariant; the
   permutation test exists to catch this. Defer everything to `cost_process()`.
2. **Re-implementing EECO in FlexCosting.** If you write a price loop or a demand
   epigraph here — in `cost_process` **or** in `report_cost` — you have
   duplicated M06/EECO. Aggregate the kW series and call `add_operating_cost`
   (in-objective) / `evaluate_cost` (post-solve); the cost math is not yours.
3. **Passing kWh instead of kW to EECO.** EECO does the kW→kWh conversion with
   `dt_hours`; hand it the raw kW `aggregate_electrical_work` and the timestep,
   never a pre-integrated energy series (double-counts on non-hourly grids).
4. **IDAES `cost_process` collisions.** `FlowsheetCostingBlockData` has its own
   `cost_process`/aggregate machinery; if the parent call fights the flex names
   (`aggregate_*`), build flex-native components first and skip/override the
   conflicting parent step — record what you did under "Deviations from spec".
5. **Breaking costing-less units.** M04 constructs Pump/StorageTank with no
   `costing_package`; the `register_energy` forwarding must be strictly
   conditional.
6. **Objective referencing EECO internals.** The objective must use
   `aggregate_operating_cost` (flex-pse name), never a raw EECO handle — that
   naming boundary is the whole point of R4.
7. **Reporting the objective as the bill.** The user-facing cost is
   `report_cost(model)` (EECO post-hoc on the realized power), never
   `value(model.objective)` — the objective is a relaxed/scalarized proxy and
   they legitimately differ (R9, §6). Do not assert them equal in tests.
8. **Building DR constraints.** DR is containers-only (§2.4). `_build_dr()` is a
   no-op; supplying a DR file must not add DR constraints or change the objective.
9. **Multi-period design in M07.** `set_design_mode()` is single-model
   CapEx-active only. Merging representative months / linking sizing vars is M16.
10. **Regression constant laundering.** The stored objective baseline must be a
    literal constant in the test with a comment naming the run that produced it —
    never recomputed from the model.

## Tests

Test-first (02 §1a). Reuse the M06 fixtures.

- `src/flexops/tests/costing/test_flex_costing.py`:
  - `test_config_exclusivity` — `@pytest.mark.unit`. Both or neither of `tariff_file`/`tariff` → `FlexConfigError` naming the options.
  - `test_construct_before_units` — `@pytest.mark.unit`. `FlexCosting` builds on a bare model with a TimeBlock and no units; `cost_process()` runs and `aggregate_electrical_work` exists (its body is the `0*kW` placeholder).
  - `test_aggregate_electrical_work` — `@pytest.mark.unit`. Pump+tank with `costing_package`; after `cost_process()`, fix a known flow profile and assert `value(aggregate_electrical_work[t])` equals the sum of registered units' `electrical_work[t]` at several t (pure `pyo.value`, no solve).
  - `test_operating_cost_is_eeco_total` — `@pytest.mark.unit`. Assert `aggregate_operating_cost` is (or evaluates equal to) `handles.total_operating_cost` — i.e. FlexCosting exposes EECO's total, not a re-derived one.
  - `test_mode_toggles` — `@pytest.mark.unit`. After `cost_process()`: `set_design_mode()` → tank `capacity.fixed is False`, capex constraint `.active is True`; `set_operations_mode()` → fixed/inactive. Toggle twice (idempotence). (Single-model mode only — no multi-period merging; that is M16.)
  - `test_construction_order_permutation` — `@pytest.mark.unit`. Build the pump+tank+costing system in ≥ 2 component-creation orders (costing first vs. costing just before `cost_process`; pump-then-tank vs. tank-then-pump), fix the same flow profile, assert `value(aggregate_operating_cost)` identical (`pytest.approx(rel=1e-12)`).
  - `test_dr_container_loads_noop` — `@pytest.mark.unit`. Build `FlexCosting` with `dr_event_file=dr_events_demo.json`; assert `self.dr` is a populated `DRConfig` container and that after `cost_process()` **no** DR constraints were built (component count / classification unchanged vs. the no-DR build). The DR hook is a no-op (containers-only, §2.4).
  - `test_model_classifies_lp` — `@pytest.mark.unit`. Built pump+tank+costing model → `flexcore.solvers.classify` returns `LP` (no `max()`/nonlinearity from the EECO bridge).
- `src/flexops/tests/costing/test_load_shifting_component.py` (each
  `@pytest.mark.component` + `@pytest.mark.needs_highs`, < 10 s):
  - `test_load_shifting_headline` — the worked example; assert optimal termination; `sum(value(pump.flow_vol[t]) for t in peak hours 16–20) == pytest.approx(0.0, abs=1e-6)`; `value(m.objective) == pytest.approx(EXPECTED_OBJECTIVE, rel=1e-6)` where `EXPECTED_OBJECTIVE` is a stored module constant recorded from the first verified run (regression baseline — changing it is a deliberate diff).
  - `test_report_cost_post_hoc` — after the headline solve, `m.costing.report_cost(m)` returns a float (the EECO post-hoc cost on the realized aggregate power) and is a stored regression constant. **Assert it is NOT equal to `value(m.objective)`** (they legitimately differ — the objective is a relaxed/scalarized proxy, R4/R9): `assert report_cost != pytest.approx(value(m.objective), rel=1e-6)` (or an explicit inequality with a comment). This encodes the reporting rule (§6).
  - `test_demand_charge_reduces_peak` — solve the same system twice: demo tariff vs. a copy with demand charges removed. Assert `max_t value(aggregate_electrical_work[t])` is strictly lower with demand charges.

(Note: the tariff *math* — price alignment, epigraph correctness, kWh
conversion, the golden bill, and the post-hoc `evaluate_cost` accuracy — is
EECO's and is tested in M06. M07 tests the FlexCosting wrapper: aggregation,
delegation, naming, modes, `report_cost` plumbing, the DR container no-op, and
the end-to-end optimization behavior.)

## Documentation tasks

- `docs/reference/flexops/costing.rst` — extend the M06 costing page: autodoc
  `FlexCosting` (+ `unit_costing`); restate R4 in one paragraph (delegates OpEx to
  EECO two ways — in-objective relaxed cost + post-solve `report_cost`; owns
  aggregation, naming, CapEx, modes; DR containers-only); document the
  `report_cost` reporting rule (never the objective — §6/M13), the mode-toggle API
  (single-model only; multi-period is M16), the construction-order invariant, and
  that tariff/cost math + limitations live with EECO/M06.
- `docs/how_to/build_a_plant.md` — **skeleton**: title, a code block walking
  pump → tank → `FlexCosting` → `cost_process()` → objective (essentially the
  headline test's model), and a note that the full guide becomes an executed
  notebook in M14.
- `CHANGELOG.md` "Unreleased": **"First end-to-end economic result: EECO-backed
  FlexCosting; pump+tank LP shifts load off-peak."**
- Class docstring per conventions §3.

## Definition of Done

- [ ] `fo.FlexCosting(time_block=..., tariff_file=..., dr_event_file=...)` constructs exactly as in the API-freeze script (PLAN.md §2), before any units exist
- [ ] `cost_process()` aggregates `electrical_work` into a kW series, calls `add_operating_cost` (EECO, in-objective relaxed cost), and exposes `aggregate_operating_cost == handles.total_operating_cost`; no cost math written in FlexCosting
- [ ] `report_cost(model)` extracts realized aggregate power as a numpy array and calls `flexops.costing.evaluate_cost` (M06) to return the reported cost; the headline test asserts it is **not** equal to `value(model.objective)` (§6/R9)
- [ ] **DR is containers-only**: `dr` placeholder + no-op `_build_dr()` fed by the DR file/`CostingConfig.dr`; no DR constraints built; the container-loads test passes
- [ ] `set_design_mode()` is single-model CapEx-active only (multi-period merging is M16, not built here)
- [ ] Built model classifies **LP** via `flexcore.solvers.classify`
- [ ] Construction-order permutation test passes; M04's costing-less unit tests still green
- [ ] Headline load-shifting component test passes under HiGHS in < 10 s with a stored objective regression constant; demand-charge and `report_cost` post-hoc tests pass
- [ ] `set_operations_mode()` / `set_design_mode()` fix/unfix and deactivate/activate as specified, idempotently
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W` passes with `costing.rst` and the `build_a_plant.md` skeleton
- [ ] CHANGELOG "first end-to-end" entry present
- [ ] plus the generic DoD in CLAUDE.md
