# M13 — Set-point extraction + smoothing + cost reporting

**Effort:** 1–2 days · **Depends on:** M12 · **Parallelizable:** no

## Goal

Turn solved models into plant-consumable set points **plus the reported cost**.
`extract_setpoints` walks a solved FlexOps model (or an M12 `ScheduleResult`) and
returns **both** a tidy long-format DataFrame of physical trajectories (a fixed
six-column schema — a public contract for downstream control systems) **and the
EECO post-hoc electricity cost** (`FlexCosting.report_cost`). The raw solver
objective is surfaced only behind an explicit flag (architecture §6, decision
R9). `smoothing.py` post-processes those trajectories (moving average,
minimum-hold snapping) while provably preserving energy and volume totals.

## Read first

- `plan/01_architecture.md` §6 (setpoints.py and smoothing.py — the spec this milestone implements; **the reporting rule R9**: set-point extraction returns physical trajectories + the EECO post-hoc cost; the objective is surfaced only behind an explicit flag)
- `plan/01_architecture.md` §3.6 / §2.4 (`FlexCosting.report_cost(model)` is the de-relaxed, user-facing electricity cost; the objective is a relaxed proxy)
- `plan/01_architecture.md` §7 (decision **R9**)
- `plan/01_architecture.md` §3.2 (IO registration: `register_io_variable`, `IORegistry`, `iter_io_registry(model)`)
- `plan/01_architecture.md` §3.3 (nested PlantBlocks — set-point `plant` paths must handle nesting)
- `plan/01_architecture.md` §4 (energy nomenclature — `electrical_work[t]` is kW; kWh needs `dt`)
- `plan/02_testing_and_ci.md` §5 ("Fixtures for solved models: setpoints/smoothing tests operate on stored solved-model value dictionaries, not fresh solves — keeps them unit tier")
- `plan/00_conventions.md` §2 (keyword-only args), §3 (exceptions: `FlexDataError`)

## Files to create or modify

- `src/flexschedule/setpoints.py` — `extract_setpoints`, `report_setpoints`, `SetpointReport`
- `src/flexschedule/smoothing.py` — `Smoother` protocol, `MovingAverageSmoother`, `MinHoldSmoother`, `check_conservation`
- `src/flexschedule/__init__.py` — export the seven names above
- `src/flexschedule/tests/test_setpoints.py`, `test_smoothing.py`
- `src/flexschedule/tests/fixtures/` — checked-in solved-model value dicts (JSON) + `generate_fixtures.py`
- `docs/how_to/schedule_rolling_horizon.md` — complete the M12 stub (see Documentation tasks)
- `docs/reference/flexschedule/index.rst` — add the two modules

## Specification

### 1. `setpoints.py`

```python
def extract_setpoints(model_or_result, *, registry=None) -> pd.DataFrame:
    """Extract registered IO/actuator variables into a tidy long-format frame.

    Schema (public contract — downstream control systems parse this):

    | column    | dtype          | meaning                                        |
    |-----------|----------------|------------------------------------------------|
    | timestamp | datetime64[ns] | wall-clock time of the step (TimeBlock mapping) |
    | plant     | str            | dotted PlantBlock path, e.g. "campus.svcw"      |
    | unit      | str            | unit-model local name, e.g. "tank"              |
    | variable  | str            | variable local name, e.g. "flow_vol"            |
    | value     | float64        | solved value at that timestep                   |
    | units     | str            | pyunits string, e.g. "kW", "m**3/s"             |
    """
```

- Column names and order are EXACTLY `timestamp, plant, unit, variable, value,
  units`. Reproduce the table above in the docstring **and** in the docs (see
  Documentation tasks). Changing this schema is a breaking change.
- Input, model path: a solved Pyomo model containing FlexOps blocks. Default
  discovery walks `flexops.core.registration.iter_io_registry(model)` and emits
  one row per (registered IO variable, timestep). `timestamp` comes from the
  owning TimeBlock's `timestamp_of(i)` / `datetime_index` (§3.1).
- Input, result path: an M12 `ScheduleResult` — melt its `committed` wide frame
  (flattened `plant.unit.variable` column names plus the units mapping stored in
  `.attrs`) into the same schema (implementer's choice on the exact melt
  mechanics; the output schema is not negotiable).
- `registry=None`: walk the model as above. A non-None `registry` is an explicit
  iterable of registry entries (the `IORegistry` dataclasses) restricting
  extraction to just those variables — used to pull actuator subsets.
- `plant` is the dotted path of PlantBlocks from the model root, so nested
  plants (§3.3) yield `"campus.svcw"`; `unit` and `variable` are local component
  names only (no dots). Derive the path from the Pyomo block hierarchy names
  (implementer's choice on traversal mechanics).
- Rows sorted by (`plant`, `unit`, `variable`, `timestamp`); stable output makes
  golden-file diffs readable (implementer's choice, but pick one order and
  document it).
- Unsolved variables (value `None`) raise `FlexDataError` naming the variable
  and telling the user to solve first.

### 1a. Reported cost (architecture §6, decision R9)

Set-point extraction returns physical trajectories **and the EECO post-hoc
cost** — never the raw solver objective by default. The six-column trajectory
schema above does **not** change (it is a hard contract); the reported cost rides
alongside it:

```python
def report_setpoints(model_or_result, *, debug_objective: bool = False) -> SetpointReport:
    """Physical trajectories + the reported EECO cost.

    Returns a SetpointReport bundling:
      - setpoints: the six-column DataFrame from extract_setpoints;
      - reported_cost: float — FlexCosting.report_cost(model) (EECO post-hoc,
        de-relaxed; the user-facing electricity cost);
      - objective_value: float | None — the raw solver objective, populated ONLY
        when debug_objective=True (a relaxed/scalarized internal quantity, not the
        reported cost).
    """
```

- `reported_cost` is always `FlexCosting.report_cost(model)` — EECO evaluated
  post-solve on the realized aggregate-power numpy array (§3.6). For a
  `ScheduleResult` input, reuse its already-computed `reported_cost` (M12) rather
  than re-evaluating.
- The **raw solver objective is surfaced only behind the explicit
  `debug_objective` flag** (default `False`): with the flag off, `objective_value
  is None` and the report excludes it entirely; with the flag on it carries the
  objective for debugging, documented as a relaxed proxy, never the reported cost.
- `SetpointReport` is a small dataclass (implementer's choice of exact field
  names, but `setpoints` and `reported_cost` are the contract). `extract_setpoints`
  itself keeps returning just the DataFrame (the schema contract is untouched);
  `report_setpoints` is the cost-aware wrapper. Both are exported.

### 2. `smoothing.py`

```python
class Smoother(Protocol):
    def smooth(self, values: pd.Series, *, dt: float) -> pd.Series: ...

def check_conservation(before: pd.Series, after: pd.Series,
                       dt: float, tol: float = 1e-6) -> None:
    """Raise FlexDataError if the dt-weighted total changed by more than tol
    (relative). For kW series this is the kWh total; for flow series, volume."""
```

- `dt` is a plain float in consistent time units (hours for kW→kWh; the caller
  converts from `time_block.dt`). Because `dt` is uniform, conservation reduces
  to `abs(after.sum() - before.sum()) <= tol * max(abs(before.sum()), eps)` —
  implement it that way but keep `dt` in the signature (the tolerance statement
  in docs is in kWh / m³ terms, and non-uniform dt is a possible future).
- **Every smoother calls `check_conservation` on its own output before
  returning.** Tests reuse the same helper. This is the module's core invariant:
  a smoothed schedule that gained or lost energy/volume is a plant-control bug.

```python
class MovingAverageSmoother:
    def __init__(self, *, window: int): ...   # steps; centered rolling mean
```

- Centered rolling mean with `min_periods=1` (pandas `rolling`), which distorts
  edge totals — so after averaging, add a uniform correction
  `(before.sum() - after.sum()) / len(after)` to every point, then call
  `check_conservation`. Document the correction in the docstring.

```python
class MinHoldSmoother:
    def __init__(self, *, hold_steps: int): ...
```

- Snap the series piecewise-constant so no value change happens faster than the
  hold interval: partition the index into consecutive segments, each at least
  `hold_steps` long (segment boundaries at multiples of `hold_steps`; the final
  segment absorbs the remainder — implementer's choice, document it), and set
  each segment to the **mean of the original values over that segment**. Segment
  means preserve totals exactly (up to float error), so `check_conservation`
  passes by construction.
- Smoothers operate on a single `pd.Series` (one variable's trajectory). To
  smooth a set-point frame, group by (`plant`, `unit`, `variable`) and apply —
  show this as the worked example in the module docstring; do not build a frame-
  level API in v0 (implementer's choice to add a thin helper, but the Series
  protocol is the contract).

### 3. Test fixtures (`src/flexschedule/tests/fixtures/`)

Per 02 §5, setpoints/smoothing tests run on **stored solved-model value
dictionaries**, never fresh solves — keeping everything unit tier.

- `solved_tank_pump.json`, `solved_nested_plant.json` — flat dicts mapping fully
  qualified component names (`"svcw.tank.volume[3]"`) to float values, plus a
  small metadata block (horizon start/dt/n so tests can rebuild the matching
  model). Include the realized aggregate-power series and the tariff reference so
  the cost-reporting test can evaluate `FlexCosting.report_cost` post-hoc without
  a solver (EECO runs on the stored numpy array — arch §3.6).
- `generate_fixtures.py` in the same directory: builds each model, solves with
  HiGHS, dumps the dicts. Runnable as
  `python src/flexschedule/tests/fixtures/generate_fixtures.py` (requires
  `highspy`; guard with a clear error if missing). It is a **script, not a
  test** — it must not be collected by pytest (no `test_` prefix). Regeneration
  procedure (put this in a module docstring): rerun the script, eyeball the
  JSON diff, commit — a fixture change is a deliberate, reviewable diff exactly
  like a golden file.
- Tests rebuild the (unsolved) model, load fixture values with `set_value`, then
  exercise `extract_setpoints`. Model *construction* without a solve is fine in
  unit tier (02 §1).

## Pitfalls

1. **Schema drift.** Extra columns, reordered columns, `object` value dtype —
   all break downstream parsers. `test_schema_exact` pins names, order, dtypes;
   never weaken it.
2. **Dotted unit names.** Only `plant` may contain dots. A nested plant path
   accidentally landing in `unit` (or a unit path in `variable`) is the classic
   traversal bug — the nested-plant fixture test exists for this.
3. **Smoothing that leaks energy.** A plain centered rolling mean does NOT
   conserve totals at the edges; the uniform correction is mandatory. Never skip
   `check_conservation` "because the math is obviously fine".
4. **MinHold segments shorter than `hold_steps`.** Remainder handling must merge
   the tail into the last full segment, not emit a short one — the run-length
   test enumerates every run.
5. **Fresh solves in tests.** The repo conftest breaks solver calls under
   `-m unit` (02 §1). If a test here needs a solver, you've drifted from the
   fixture design — stop and use the stored dicts.
6. **Exact float equality on conservation.** Use explicit tolerances
   (`pytest.approx(rel=...)`) per conventions §7; segment means are exact only
   up to float summation order.
7. **Units as pyunits objects in the frame.** The `units` column is a plain
   string (serialization target); convert via `str(pyunits.get_units(var))` or
   equivalent, and keep it stable.
8. **Reporting the objective as the cost.** The reported number is the EECO
   post-hoc cost (`FlexCosting.report_cost`), never the solver objective (arch
   §6, R9). `report_setpoints` excludes the objective by default; it appears only
   under `debug_objective=True`, documented as a relaxed proxy.

## Tests

All in `src/flexschedule/tests/`, ALL `@pytest.mark.unit` (no solver anywhere).

`test_setpoints.py`:
- `test_schema_exact` — column list `== ["timestamp", "plant", "unit", "variable", "value", "units"]` (order included); dtypes: `datetime64[ns]`, str/object ×3 around a `float64` value column.
- `test_values_match_fixture` — every row's `value` equals the fixture dict entry for `plant.unit.variable` at that timestep.
- `test_nested_plant_paths` — nested-plant fixture yields `plant == "campus.svcw"`, `unit == "tank"`, no dots in `unit`/`variable`.
- `test_registry_subset` — passing an explicit `registry` of one entry yields rows for only that variable.
- `test_unsolved_raises` — model with unset values → `FlexDataError` naming the variable.
- `test_schedule_result_input` — a hand-built `ScheduleResult` (wide frame + units attrs) melts to the identical schema.
- `test_report_returns_eeco_cost_not_objective` — `report_setpoints` on a fixture (a solved-value model with a small `FlexCosting`/tariff, or a hand-built `ScheduleResult` carrying `reported_cost`) returns a `SetpointReport` whose `reported_cost` equals `FlexCosting.report_cost(model)` (or the `ScheduleResult`'s stored value) to rel=1e-6, and whose `objective_value is None` by default; with `debug_objective=True` the objective field is populated (and the docstring flags it as a relaxed proxy). Pins the reporting rule (arch §6, R9). Stays `unit`: no fresh solve — `report_cost` evaluates EECO on a stored aggregate-power array (add the tariff + realized-power array to the fixture metadata; no solver call).

`test_smoothing.py`:
- `test_moving_average_conserves_totals` — seeded random series (fixed seed), several window sizes; dt-weighted totals within `rel=1e-9`.
- `test_moving_average_adversarial_spike` — flat series with one 1000× spike; totals conserved; smoothed peak strictly below original.
- `test_min_hold_run_lengths` — for `hold_steps` in {2, 3, 5} on a seeded noisy series: every constant run in the output has length ≥ `hold_steps` (enumerate runs; the tail merge means the last run may be longer).
- `test_min_hold_conserves_totals` — totals within `rel=1e-9`, including a length not divisible by `hold_steps`.
- `test_check_conservation_raises` — perturbed series beyond tol → `FlexDataError`; within tol → no raise.
- `test_smoother_protocol_conformance` — both classes satisfy the `Smoother` protocol (`isinstance` with `runtime_checkable`, or a typed call).

## Documentation tasks

- Complete `docs/how_to/schedule_rolling_horizon.md`: rolling-horizon solve
  (M12) → `extract_setpoints` → smoothing → the schema table; link the
  `03_rolling_horizon` notebook slot (notebook itself lands in M14).
- Document the six-column set-point schema **as a table** in the how-to page
  and in `docs/reference/flexschedule/` (it appears in the docstring too —
  three places, kept identical).
- Add `setpoints`/`smoothing` to `docs/reference/flexschedule/index.rst`
  (including `report_setpoints`/`SetpointReport`); note the reported cost is the
  EECO post-hoc number, never the objective (arch §6, R9).
- CHANGELOG entry under "Unreleased" (the schema is user-visible).

## Definition of Done

- [ ] `extract_setpoints` emits exactly the six-column schema from models and from `ScheduleResult`s; nested plant paths correct
- [ ] `report_setpoints`/`SetpointReport` return physical trajectories + the EECO post-hoc cost (`FlexCosting.report_cost`); the raw objective is excluded by default and surfaced only under the explicit `debug_objective` flag (arch §6, R9)
- [ ] `Smoother` protocol + `MovingAverageSmoother` + `MinHoldSmoother` implemented; both call the shared `check_conservation`
- [ ] Conservation holds on adversarial input; MinHold never changes faster than the hold interval
- [ ] Solved-model fixtures checked into `src/flexschedule/tests/fixtures/` with a documented regeneration script
- [ ] All tests above written and marked `unit`; suite passes with no solver installed
- [ ] `docs/how_to/schedule_rolling_horizon.md` completed; schema documented as a table; reference pages build with `sphinx-build -W`
- [ ] CHANGELOG updated; PR records implementer's-choice decisions
- [ ] plus the generic DoD in CLAUDE.md
