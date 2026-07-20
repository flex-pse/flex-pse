# M08 — Battery (DERMS) + customizable unit commitment

**Effort:** 2–3 days · **Depends on:** M07 · **Parallelizable:** no

## Goal
Add the first storage-with-state unit model, `BatteryModel`, first-class its
**external (DERMS) dispatch** hook, and build the **customizable unit-commitment
(UC) layer** — a composable set of optional constraint pieces (`status`,
`startup_shutdown`, `dwell`, `delays`, `conditional`, `bypass`) applied per unit
via its `unit_commitment` config, plus a **model-level** parallel-train
degeneracy detector — that turns LP scheduling models into MILPs with first-class
relaxation toggling. By the end: a battery arbitrages a TOU tariff as a MIP and
solves relaxed as an LP; an externally supplied dispatch series fixes the
battery's power so only sizing remains free (the Project-1 third-party-controller
case); UC transition and conditional logic are proven correct by exhaustive
constraint-body truth tables; a k-step startup delay ties a unit to an upstream
unit's status; and a model-level pass adds symmetry-breaking constraints to two
identical parallel trains — no solver required for the logic proofs.

## Read first
- `plan/01_architecture.md` §3.2 (OpsBlock registration API; inherited config
  flags `relaxation` / `unit_commitment` / `allow_bypass` / `external_dispatch`;
  the `set_external_dispatch(var, series, *, fix=True)` base method — DERMS)
- `plan/01_architecture.md` §3.4 (unit model library table — `BatteryModel` row;
  first-class `external_dispatch`; note the electrolysis/separator units move to M09)
- `plan/01_architecture.md` §3.5 (customizable unit commitment: status base +
  optional startup/shutdown/dwell/delays/conditional + bypass; **model-level**
  parallel-train degeneracy detection; relax as a first-class domain toggle)
- `plan/01_architecture.md` §3.6 (R4: capacity as sizing Var; operations vs design
  mode; external dispatch first-classed on `BatteryModel`)
- `plan/01_architecture.md` §3.1 (`time_block.register_initial_state`, `dt`) and
  §4 (energy nomenclature)
- `plan/01_architecture.md` §7 — decision log **R8** (customizable UC; degeneracy
  is model-level, not per-unit — a unit can't see its siblings) and **R9**
  (external DERMS dispatch fixes the dispatch DOF; sizing stays free)
- `plan/02_testing_and_ci.md` §1, §2, §5 (tier markers, harness, constraint-body
  truth tables)

## Files to create or modify
- `src/flexops/unit_models/battery.py` — `BatteryModel` OpsBlock subclass (SOC
  dynamics, charge/discharge, fixable capacity, first-class `external_dispatch`).
- `src/flexops/logic/__init__.py` — re-export the UC pieces:
  `add_status`, `relax`, `unrelax`, `add_startup_shutdown`, `add_min_uptime`,
  `add_min_downtime`, `add_startup_delay`, `add_conditional`,
  `break_parallel_symmetry`, `detect_parallel_trains`, `add_bypass`.
- `src/flexops/logic/status.py` — Binary `status[t]` base + semicontinuous
  linking + relaxation toggling.
- `src/flexops/logic/startup_shutdown.py` — **optional**: `startup[t]`/`shutdown[t]`
  + transition logic.
- `src/flexops/logic/dwell.py` — **optional**: minimum up/down-time sequence
  constraints.
- `src/flexops/logic/delays.py` — **optional**: startup/response delay connected
  to an upstream unit's status/startup.
- `src/flexops/logic/conditional.py` — **optional**: if-x-on-then-y-on /
  then-y-off implication constraints between units.
- `src/flexops/logic/degeneracy.py` — **model-level** parallel-train degeneracy
  detection + symmetry-breaking (operates over a PlantBlock/NetworkBlock).
- `src/flexops/logic/bypass.py` — bypass-stream constraints around a unit.
- `src/flexops/tests/unit_models/test_battery.py` — harness subclass + hand
  checks + arbitrage MIP + external-dispatch test.
- `src/flexops/tests/logic/test_status.py`, `test_startup_shutdown.py`,
  `test_dwell.py`, `test_delays.py`, `test_conditional.py`,
  `test_degeneracy.py`, `test_bypass.py` — logic-layer tests.
- `docs/reference/flexops/unit_models/index.rst`,
  `docs/reference/flexops/logic.rst` — reference entries.

## Specification

### BatteryModel (`src/flexops/unit_models/battery.py`)
An OpsBlock subclass declared with `declare_process_block_class` (via
`flexcore.compat.idaes`), exactly like `Pump`/`StorageTank` from M04. Constructor
is keyword-only; the API-freeze call it must support is
`fo.BatteryModel(capacity=1 * pyunits.kWh, costing_package=m.costing)`.

CONFIG entries (Pyomo `ConfigDict`, each with `description=`):
- `capacity` — initial capacity value with units (kWh). Required in v0 (implementer's choice: no default).
- `charge_power_max`, `discharge_power_max` — kW limits (implementer's choice: default `None` = unbounded unless status is enabled, which requires both).
- `eta_charge`, `eta_discharge` — efficiencies in (0, 1], default 1.0 (implementer's choice).
- `soc_min_frac`, `soc_max_frac` — SOC bounds as fraction of capacity, defaults 0.0 / 1.0.
- `initial_soc_frac` — initial SOC as fraction of capacity, default 0.5 (implementer's choice).
- Inherited flags from OpsBlock (§3.2): `relaxation`, the `unit_commitment`
  sub-config (§3.5), `allow_bypass`, and the optional `external_dispatch` source.

Components (all Vars/Constraints carry `doc=` — the docs generator renders them):
- `capacity` — **Var**, kWh, non-negative. Initialized to the config value **and
  fixed** at construction. This is the sizing Var per R4: register it with the
  costing package so `costing.set_design_mode()` unfixes it and
  `set_operations_mode()` re-fixes it. Do NOT make it a Param.
- `soc[t]` — Var, kWh, non-negative, `doc="State of charge"`.
- `charge_power[t]`, `discharge_power[t]` — Vars, kW, non-negative, upper-bounded
  by the config maxima when given.
- `electrical_work[t]` — provided by the OpsBlock base (declared electrical via
  `register_energy(..., kind="electrical")`). Domain must be **Reals**, not
  NonNegativeReals: discharge exports power.
- `soc_init` — **mutable Param**, kWh, initialized to
  `initial_soc_frac * capacity_value`, registered with
  `time_block.register_initial_state(soc_init)` so the M12 rolling-horizon driver
  can mutate it between windows.

Constraints (`N = len(time_block.time_points)`, `dt = time_block.dt`):
- `soc_balance[t]` for `t = 0 .. N-2`:
  `soc[t+1] == soc[t] + dt * (eta_charge * charge_power[t] - discharge_power[t] / eta_discharge)`
  (copy this equation exactly; note kW·time → kWh, so units must work out — see Pitfall 2).
- `soc_initial`: `soc[0] == soc_init`.
- `soc_lower[t]`: `soc[t] >= soc_min_frac * capacity` and
  `soc_upper[t]`: `soc[t] <= soc_max_frac * capacity`. These must be
  **Constraints, not Var bounds**, because `capacity` is a Var (Pitfall 1).
- `net_electrical[t]`: `electrical_work[t] == charge_power[t] - discharge_power[t]`.
  Discharge makes the unit's draw negative (an export). v0 assumes behind-the-meter
  operation; any "facility net draw ≥ 0" constraint belongs at the plant/costing
  level, not here **(implementer's choice — record this note in the class docstring)**.
- Unit commitment enabled (via the `unit_commitment` config, §3.5): attach a
  mutually-exclusive charge/discharge binary through the logic layer — call
  `add_status(self, self.charge_power, 0, charge_power_max)` to get `status[t]`,
  then add `discharge_power[t] <= discharge_power_max * (1 - status[t])`
  (implementer's choice on the exact wiring; requirement: one Binary per t, charging
  and discharging cannot both be positive, and `relax(unit)` covers it). The
  battery does **not** enable startup/shutdown/dwell/delays by default; those are
  opt-in per the config and are exercised by the logic tests, not the battery.

### External dispatch (DERMS) — first-classed on `BatteryModel` (§3.2/§3.6, R9)
The base `set_external_dispatch(var, series, *, fix=True)` (from M03/§3.2) fixes a
controllable actuator variable to an externally supplied, time-indexed command
series, **removing that dispatch degree of freedom while leaving sizing free**.
This is the mechanism for third-party-controlled assets (a battery under a
DERMS/aggregator — Project 1). For `BatteryModel`:
- A convenience wrapper (implementer's choice of name, e.g.
  `set_dispatch(series)`) that calls `set_external_dispatch` on the battery's
  net-power actuator (`charge_power`/`discharge_power`, or an equivalent net-power
  Var) so `series[t]` fixes the dispatch at every `t`. `series` is a mapping or
  sequence aligned to `time_block.time_points` (implementer's choice; document it).
- An `external_dispatch` config block (§3.2) may declare the variable and data
  source so the same fixing happens config-driven; wiring only, the actual series
  comes from the caller in v0.
- Invariant: with dispatch fixed, the battery's power is **not** a decision
  variable and only `capacity` remains free — sizing still optimizes. FlexSchedule
  and FlexParameterize must respect fixed dispatch (they never unfix it).

Registration: `register_process_parameter(capacity)` (regressable=False —
implementer's choice), `register_io_variable` for `charge_power`/`discharge_power`
as outputs is optional in v0 (implementer's choice; document what you pick). No
inlet/outlet material ports — the battery is an energy-only unit (no property
package needed; keep the constructor tolerant of `property_package` being absent).

### logic/status.py — the UC base (always present when a unit can shut off)
```python
def add_status(unit, output_var, min_output, max_output):  # returns the Binary Var status[t]
def relax(unit) -> None
def unrelax(unit) -> None
```
- `add_status` attaches `status[t]` (Binary, indexed by the unit's time points)
  plus the semicontinuous links `min_output * status[t] <= output_var[t]` and
  `output_var[t] <= max_output * status[t]` (two Constraint objects, `doc=` set).
  This is the base UC piece: present on **every unit that can be shut off**; a
  `StorageTank` disables it (a tank has no on/off status — §3.4).
- Track every Binary the logic layer attaches to a unit in a list on the unit,
  e.g. `unit._flexops_logic_binaries` (implementer's choice of attribute name; keep
  it private and documented in the module docstring). All optional pieces below
  append their binaries to this same list so `relax`/`unrelax` cover them.
- `relax(unit)` switches the domain of each tracked Binary to
  `pyo.UnitInterval`; `unrelax(unit)` switches back to `pyo.Binary`, honoring the
  unit's `relaxation` config policy (v0 policy values: `"allow"` (default) and
  `"never"` which makes `relax()` raise `FlexConfigError` — implementer's choice).
  Toggling is a **first-class domain switch on the live model — never a rebuild**
  (architecture §3.5). It must not touch constraints, bounds, or fixed values.

### logic/startup_shutdown.py (**optional**)
```python
def add_startup_shutdown(unit, status_var):  # returns (startup[t], shutdown[t])
```
Attaches `startup[t]`, `shutdown[t]` binaries (indexed by the unit's time points,
tracked in `_flexops_logic_binaries`) with the standard transition logic, for each
`t >= 1`:
`status_var[t] - status_var[t-1] == startup[t] - shutdown[t]`
plus `startup[t] + shutdown[t] <= 1` (cannot start and stop in one step;
implementer's choice, document it). `t = 0` is governed by an initial-state Param
only when one exists; in v0 leave `t = 0` unconstrained (document it in the
function docstring). All Constraints carry `doc=`.

### logic/dwell.py (**optional**)
```python
def add_min_uptime(unit, status_var, k):
def add_min_downtime(unit, status_var, k):
```
Standard unit-commitment sequence constraints over integer time indices. For
min-uptime `k`: for each `t >= 1` and each `tau in range(t, min(t + k, N))`:
`status_var[tau] >= status_var[t] - status_var[t-1]` (a startup at `t` forces on
for `k` steps, truncated at horizon end). Min-downtime is the mirror:
`1 - status_var[tau] >= status_var[t-1] - status_var[t]`. `t = 0` is governed by
the initial-state Param only when one exists; in v0, leave `t = 0` unconstrained
(implementer's choice — document it in the function docstring).

### logic/delays.py (**optional**)
```python
def add_startup_delay(unit, upstream, k):
```
Startup/response **delay connected to an UPSTREAM unit** — this unit may not
start (or be on) until `k` steps after the upstream unit's status/startup (the
chemical-stabilization delays of Rao 2024). Smallest useful form (implementer's
choice, document it in the function docstring): for each `t`,
`unit.status[t] <= upstream.status[t - k]` for `t >= k`, and
`unit.status[t] == 0` for `t < k` (the downstream unit cannot be on until the
upstream has been on `k` steps earlier). The upstream reference is another unit
carrying a `status` Var (added via `add_status`); raise `FlexConfigError` if the
upstream has no status. All Constraints carry `doc=`. Full delay-*chain*
templates from Rao 2024 are post-v0 (PLAN.md §4) — this is the single-hop
primitive they build on.

### logic/conditional.py (**optional**)
```python
def add_conditional(x_unit, y_unit, *, then="on"):
```
Conditional implication constraints between two units' statuses: "if x is on then
y is on" (`then="on"`) or "if x is on then y is off" (`then="off"`). As linear
implications on the Binary statuses (both units must carry a `status` Var):
- `then="on"`: `y.status[t] >= x.status[t]` for all `t`.
- `then="off"`: `y.status[t] <= 1 - x.status[t]` for all `t`.
Raise `FlexConfigError` if either unit lacks `status` or `then` is not in
`{"on","off"}`. All Constraints carry `doc=`.

### logic/degeneracy.py — **model-level** parallel-train symmetry breaking (§3.5, R8)
Symmetry among identical parallel trains creates solver-time degeneracy.
Detection + symmetry-breaking is **implemented outside the unit level** — a unit
cannot see its siblings, so this is a **model-level** pass over a
`PlantBlock`/`NetworkBlock`, NOT a method on `OpsBlockData` (R8).
```python
def detect_parallel_trains(block) -> list[list[unit]]:
def break_parallel_symmetry(block) -> int:  # returns number of ordering constraints added
```
- `detect_parallel_trains(block)` walks the block's child units and groups
  **interchangeable** units — same class, same relevant config (capacity/limits),
  and (v0 smallest form) the same connectivity — into equivalence classes of size
  ≥ 2 (implementer's choice on the exact interchangeability predicate; document
  it). Groups of size 1 are not degenerate and are ignored.
- `break_parallel_symmetry(block)` calls `detect_parallel_trains`, and for each
  group of ≥ 2 interchangeable units adds **ordering (lex) constraints** on their
  status series that force a canonical ordering among the identical trains — e.g.
  for units ordered `u0, u1, ...` in the group, `u_{i}.status[t] >= u_{i+1}.status[t]`
  for all `t` (a train may not be on unless its predecessor is; implementer's
  choice of lex form, document it). It returns the count of constraints added and
  attaches them to the block (not to any single unit). All Constraints carry `doc=`.
- This is a diagnostic/transform pass invoked explicitly by the user or a solve
  sequence, never automatically inside `build()`.

### logic/bypass.py
`add_bypass(unit, flow_var, bypass_max)` — smallest useful version (implementer's
choice on exact form; the choice MUST be documented in the module docstring and
reference page). Suggested minimum: attach `bypass_flow[t]` (Var, same units as
`flow_var`, bounds `[0, bypass_max]`) and constraint
`treated_flow[t] == flow_var[t] - bypass_flow[t]`, where `treated_flow` is the
quantity the unit's energy relation consumes. Rewiring Ports/Arcs is explicitly
out of scope for v0.

### Applying UC per unit
Each piece is applied to a unit according to its `unit_commitment` sub-config
(§2.3/§3.5), **every piece optional except `status`** (which is present whenever a
unit can be shut off). A unit that disables unit commitment (a tank) gets none of
them. The battery enables `status` (for mutually-exclusive charge/discharge) but
none of the optional pieces by default. Wiring the config → which pieces are
attached is the unit's responsibility; the logic functions are the primitives.

### Electrolyzer / separators — NO LONGER in this milestone
The old electrolyzer/separator units have moved to **M09** as part of the
IO-topology zoo (`Separator(SIDOBlock)`, `ElectrolysisSeparator`, etc. —
architecture §3.4, R6). Do **not** build any electrolyzer/separator unit here.

## Pitfalls
1. **SOC bounds as Var bounds.** `soc[t].setub(frac * capacity)` fails or silently
   snapshots the value because `capacity` is a Var. Write inequality Constraints.
2. **Units in the SOC equation.** `dt` carries `pyunits` (e.g. minutes); kW·min is
   not kWh. Let Pyomo units handle it and prove it with `assert_units_consistent`
   (harness stage) — do not insert magic `/60` factors.
3. **`electrical_work` domain.** The base class may default to non-negative; the
   battery must override to Reals or discharging is infeasible.
4. **`relax()` that rebuilds.** Deleting and re-adding constraints breaks warm
   starts and any references held by costing. Only `var.domain` changes.
5. **Dwell truncation at the horizon end.** `range(t, min(t + k, N))` — an
   off-by-one here is exactly what the truth-table test exists to catch.
6. **Last time point in `soc_balance`.** The difference equation has `N-1`
   members, not `N`; building `soc[t+1]` at `t = N-1` raises KeyError.
7. **Registering `soc_init` too late.** Register with the TimeBlock during
   `build()`, not lazily — M12 discovers initial states from that registry.
8. **Degeneracy as a per-unit method.** Do NOT put symmetry-breaking on
   `OpsBlockData` — a unit cannot see its siblings (R8). It is a model-level pass
   over a PlantBlock/NetworkBlock and adds constraints to the *block*.
9. **External dispatch that also fixes sizing.** `set_external_dispatch` fixes the
   *power* actuator only; `capacity` must stay a free (unfixed in design mode)
   sizing Var. Fixing dispatch removes the dispatch DOF, never the sizing DOF.
10. **Startup/shutdown at `t = 0`.** The transition equation references `t-1`;
    guard `t >= 1` and leave `t = 0` unconstrained in v0 (no initial-state Param).

## Tests
All in colocated test packages; every test carries exactly one tier marker.

`src/flexops/tests/unit_models/test_battery.py`
- `TestBatteryModel(UnitModelTestHarness)` — `configure()` builds a 4-step
  TimeBlock + battery; `expected_dof` / `expected_solution` per the harness
  contract (build/units/registration/DoF stages are `unit`; solve is `component`).
- `test_soc_constraint_bodies` (`unit`) — fix `charge_power`/`discharge_power`/`soc`
  to hand-picked values over 4 steps; evaluate each `soc_balance[t]` body with
  `pyo.value()` and compare to a hand computation (`pytest.approx(..., rel=1e-6)`).
- `test_capacity_fix_unfix` (`unit`) — after construction `capacity.fixed` is True
  and equals the constructor value; `costing.set_design_mode()` unfixes it;
  `set_operations_mode()` re-fixes it.
- `test_external_dispatch_fixes_power_not_sizing` (`unit`) — apply an external
  dispatch series to the battery; assert the battery's power actuator is fixed to
  `series[t]` at every `t` (`var.fixed` True, values match) and is **not** a
  decision variable, while `capacity` remains a free sizing Var (unfixed under
  design mode). No solver.
- `test_battery_arbitrage_mip` (`component`, `needs_highs`) — 24-step battery vs. a
  two-level TOU price (M06/M07 fixtures), objective = energy cost via FlexCosting;
  solve as MIP with unit commitment (mutually-exclusive charge/discharge) enabled;
  assert optimal, charging concentrated in off-peak steps and discharging in
  on-peak steps.
- `test_arbitrage_relaxation_bounds_mip` (`component`, `needs_highs`) — call
  `relax(battery)` on the same model, re-solve as LP; assert optimal and
  LP objective ≤ MIP objective (min problem) within tolerance.
- `test_external_dispatch_sizing_only_solve` (`component`, `needs_highs`) —
  with an external dispatch series fixed and design mode on (capacity free),
  solve; assert optimal and that `capacity` took an optimized value (sizing still
  optimizes when dispatch is fixed).

`src/flexops/tests/logic/test_status.py`
- `test_add_status_constraint_bodies` (`unit`) — semicontinuous bodies evaluated at
  fixed points: status=1/x=max feasible, status=0/x>0 infeasible, etc.
- `test_relax_unrelax_round_trip` (`unit`) — record domains, constraint set, and
  fixed flags; `relax()` → all tracked binaries have domain `UnitInterval` and
  **nothing else changed**; `unrelax()` → domains are `Binary` again; model
  component count identical throughout.

`src/flexops/tests/logic/test_startup_shutdown.py`
- `test_startup_shutdown_transition_truth_table` (`unit`) — the constraint-body
  testing style (02_testing §5). Enumerate **all 2^6 = 64 six-step on/off
  schedules** via `itertools.product((0, 1), repeat=6)`. For each schedule: fix
  `status[t]` to it, derive the implied `startup[t]`/`shutdown[t]` from an
  independent pure-Python reference (`startup=1` where 0→1, `shutdown=1` where
  1→0), fix them, and assert every transition-constraint body
  (`status[t]-status[t-1]-(startup[t]-shutdown[t])` and
  `startup[t]+shutdown[t]<=1`) evaluates satisfied with `pyo.value`. No solver.

`src/flexops/tests/logic/test_dwell.py`
- `test_min_uptime_truth_table` / `test_min_downtime_truth_table` (`unit`) — the
  exemplar of the constraint-body testing style (02_testing §5). Enumerate **all
  2^6 = 64 six-step on/off schedules** via `itertools.product((0, 1), repeat=6)`.
  For each schedule: (a) fix `status[t]` to it and check every dwell constraint
  body is satisfied (evaluate lower/body/upper with `pyo.value`); (b) compute
  feasibility with an independent hand-written pure-Python reference (scan the
  schedule for runs shorter than `k` after a switch). Assert (a) == (b) for every
  schedule, for `k in (2, 3)`. No solver anywhere in this file.

`src/flexops/tests/logic/test_delays.py`
- `test_startup_delay_blocks_early_start` (`unit`) — two units, downstream tied to
  upstream by `add_startup_delay(down, up, k)` for `k = 2`; fix the upstream
  `status` schedule; assert the delay-constraint bodies force `down.status[t] == 0`
  for `t < k` and forbid `down.status[t] == 1` while `up.status[t-k] == 0`
  (evaluate bodies with `pyo.value`; hand-check a couple of `t`). No solver.
- `test_startup_delay_requires_upstream_status` (`unit`) — upstream without a
  `status` Var → `FlexConfigError`.

`src/flexops/tests/logic/test_conditional.py`
- `test_conditional_on_implication_bodies` (`unit`) — `add_conditional(x, y, then="on")`;
  fix `x.status`/`y.status` at points; assert `y.status[t] >= x.status[t]` body
  holds where feasible and is violated where `x=1,y=0` (constraint-body eval, no solver).
- `test_conditional_off_implication_bodies` (`unit`) — `then="off"`; assert
  `y.status[t] <= 1 - x.status[t]` body semantics by hand.
- `test_conditional_bad_args_raise` (`unit`) — missing `status` or bad `then` →
  `FlexConfigError`.

`src/flexops/tests/logic/test_degeneracy.py`
- `test_detect_two_identical_trains` (`unit`) — a PlantBlock with two identical
  parallel units (same class/config) plus one different unit; assert
  `detect_parallel_trains(plant)` returns exactly one group containing the two
  identical units and not the third. No solver.
- `test_break_symmetry_adds_ordering_constraints` (`unit`) — call
  `break_parallel_symmetry(plant)`; assert it returns a positive count, the
  ordering (lex) constraints exist **on the block** (not on a unit), and their
  bodies encode `u0.status[t] >= u1.status[t]` for all `t` (evaluate a couple by
  hand). Assert a single-unit group adds nothing. No solver.
- `test_symmetry_breaking_makes_solve_deterministic` (`component`, `needs_highs`)
  — small model with two identical parallel trains and a demand that needs exactly
  one train on; solve with symmetry-breaking applied; assert optimal and that the
  canonical (lower-indexed) train is the one selected, i.e. the ordering
  constraints pick a deterministic optimum among the solver-equivalent optima.

`src/flexops/tests/logic/test_bypass.py`
- `test_bypass_constraint_bodies` (`unit`) — bodies at fixed flows; bypass at its
  bound; treated flow arithmetic checked by hand.

## Documentation tasks
- `BatteryModel` reference page: autosummary entry in
  `docs/reference/flexops/unit_models/index.rst` using the `unit_model` template
  (`.. flexops-unit-tables::` does the tables). Class docstring per conventions §3:
  description, `.. math::` SOC equation, usage snippet, config cross-refs, the
  behind-the-meter assumption, "no on/off binary unless unit commitment is
  enabled", and the external-dispatch (DERMS) usage — fixing dispatch leaves only
  sizing free.
- `docs/reference/flexops/logic.rst` — document the customizable UC layer:
  `add_status`/`relax`/`unrelax` (base), the optional
  `add_startup_shutdown`, `add_min_uptime`/`add_min_downtime`,
  `add_startup_delay`, `add_conditional`, `add_bypass` (including the documented
  v0 bypass form), and the **model-level** `detect_parallel_trains` /
  `break_parallel_symmetry` (state clearly these are block-level, not per-unit, per R8).
- Add a short "unit-level relaxation policy" subsection to
  `docs/explanation/relaxation_policies.md` (create the page as a stub if M07 did
  not); full R5/SolveSequence narrative lands in M12.
- CHANGELOG entry under "Unreleased".

## Definition of Done
- [ ] `BatteryModel` builds, is units-consistent, and passes its harness subclass.
- [ ] Capacity Var is fixed by the constructor and toggles with design/operations mode.
- [ ] `soc_init` registered via `time_block.register_initial_state`.
- [ ] External (DERMS) dispatch fixes the battery's power (not a decision var) while `capacity` stays free; the sizing-only solve optimizes capacity.
- [ ] `add_status` / `relax` / `unrelax` implemented; round-trip test proves domain-only toggling; all optional pieces' binaries are covered by `relax`.
- [ ] Startup/shutdown transition truth-table passes for all 64 schedules.
- [ ] Dwell truth-table tests pass for all 64 schedules, k ∈ {2, 3}, up and down.
- [ ] Startup delay ties a unit to an upstream unit's status (can't start until k steps after); missing upstream status raises `FlexConfigError`.
- [ ] Conditional on/off implications proven by constraint-body tests; bad args raise `FlexConfigError`.
- [ ] Model-level degeneracy: two identical parallel trains detected; symmetry-breaking ordering constraints added to the block (not a unit); the small solve is deterministic (canonical train selected).
- [ ] Bypass implemented and its chosen form documented.
- [ ] Arbitrage MIP (`needs_highs`) charges off-peak / discharges on-peak; relaxed LP bounds it.
- [ ] Every test carries exactly one tier marker; `unit` tests never invoke a solver.
- [ ] Reference pages build with `sphinx-build -W`; CHANGELOG updated.
- [ ] plus the generic DoD in CLAUDE.md
