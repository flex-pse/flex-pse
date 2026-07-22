# M06 — EECO integration (tariffs & costs; DR containers)

**Effort:** 2 days · **Depends on:** M00 · **Parallelizable:** with M02–M05

## Goal

Wire in the external **EECO** package (`eeco`, PyPI) as flex-pse's tariff /
operating-cost engine, and build the thin flex-pse interface around it in
`flexops/costing/opex.py`. EECO is used **two ways** (architecture §2.4, §3.6):

1. **In-objective (Pyomo-aware).** EECO builds the time-indexed, **convex-relaxed**
   operating-cost `Expression`s on a Pyomo model — the tractable proxy the
   scheduler minimizes.
2. **Post-optimization (the reported number).** After a solve, EECO is called on
   a **realized aggregate-power numpy array** to compute the TRUE (de-relaxed)
   electricity cost — the user-facing bill.

This milestone delivers (a) the dependency + the sole `eeco` import point, (b)
tariff-signal helper functions (peak/window/gradient) for logic constraints, (c)
the Pyomo in-objective bridge `add_operating_cost(...)` that FlexCosting (M07)
calls, (d) the post-optimization evaluator `evaluate_cost(...)` that produces the
reported electricity cost, (e) the gas-utility mirrors `add_gas_cost(...)` /
`evaluate_gas_cost(...)`, and (f) a golden-file test proving EECO reproduces a
hand-computed bill to the cent. flex-pse does **not** re-implement price series,
demand-charge epigraphs, or cost math — that is EECO's job.

**Total operating cost, and what's in v0 (scope note).** The user-facing
operating cost flex-pse ultimately reports is: **electricity cost** (EECO,
`evaluate_cost`) **+ gas cost** (EECO, `evaluate_gas_cost`) **+ user-defined
fixed operating costs** (maintenance, labor, chemicals, etc. — *not* from
EECO). This milestone delivers the two EECO-sourced pieces (electricity and
gas). Fixed operating costs are **out of scope for M06** — see "Deferred:
fixed operating costs" below; do not build that config in this milestone.

**Why two evaluations (architecture §2.4, R4/R9).** The in-objective cost is a
*convex relaxation* of a non-convex pricing structure: relaxing is what makes
tariff-aware scheduling tractable, but the relaxed value is a proxy, not the
bill. Once the solve fixes the dispatch, the aggregate power is a known numpy
array and the non-convexity is harmless — so the reported cost is a
straightforward **post-hoc evaluation** on that fixed array (cleaner and exact,
no epigraph/relaxation artifacts). The relaxed in-objective cost is therefore
≤ or ≈ the post-hoc true cost, and the raw solver objective is **never** the
user-facing number (§6 reporting rule, R9).

**DR is containers-only in v0 (architecture §2.4, §3.6).** flex-pse does **not**
build DR constraints in v0. This milestone provides a `DRConfig` container slot
and a **no-op DR hook** so the wiring exists and turning DR on later is additive;
nothing builds DR event/curtailment constraints yet.

> **You do not have EECO's API memorized. Read EECO's docs first.** This file
> specifies the **flex-pse-facing** wrapper API (stable, ours) precisely, and
> describes what each function must accomplish *via EECO*. Map our wrappers onto
> EECO's real API; wherever a name/signature below meets EECO, verify it against
> the installed package and record what you found in the PR. Everything marked
> "(verify against EECO)" is a discovery task, not a spec you may invent.

## Read first

- `plan/01_architecture.md` §2.4 (the EECO decision: external, Pyomo-aware, used **two ways** — in-objective convex relaxation + post-optimization evaluation; **DR containers-only in v0**; sole import point) and §3.6 (how FlexCosting wraps it — in-objective `add_operating_cost` + post-solve `report_cost`); decisions R4 and R9 and the §6 reporting rule
- `plan/00_conventions.md` §6 (import discipline: `eeco` only inside `flexops.costing`), §3 (`FlexConfigError`/`FlexDataError` message style), §4 (config rules)
- `plan/02_testing_and_ci.md` §5 (golden-file test: hand-typed truth, deliberate diffs), §1a (test-first)

## Files to create or modify

- `pyproject.toml` — add `eeco` to core runtime dependencies (verify the exact PyPI/distribution name; import name assumed `eeco`) - import from the latest pip installable version. 
- `.importlinter` — **no change**: per decision R12 there is **no `eeco` import-linter contract** (the only contract is the package DAG, conventions §6). Localizing `eeco` in `flexops/costing/opex.py` is a convention verified by `grep`, not an enforced boundary
- `src/flexops/costing/opex.py` — **the sole `eeco` import point**: loaders, the CSV→dict tariff helper, signal helpers, the in-objective `add_operating_cost` / `add_gas_cost` bridges, the post-optimization `evaluate_cost` / `evaluate_gas_cost` evaluators, and the no-op `DRConfig` / DR hook
- `src/flexops/costing/__init__.py` — export `load_tariff`, `load_dr_program`, `tariff_csv_to_dict`, `price_series`, `is_peak`, `peak_windows`, `price_gradient`, `add_operating_cost`, `add_gas_cost`, `OperatingCostHandles`, `evaluate_cost`, `evaluate_gas_cost`, `DRConfig`
- `src/flexops/tests/costing/test_tariff_signals.py`, `test_operating_cost.py`
- `src/flexops/tests/fixtures/tariff_tou_demo.json`, `tariff_tou_demo.csv`, `dr_events_demo.json` — the demo tariff/DR in **EECO's** file format (the `.csv` fixture is the same tariff in `rate_data` CSV form, for the `tariff_csv_to_dict` test; the DR file is loaded into the container only; no DR constraints are built)
- `docs/reference/flexops/costing.rst` — start the costing reference (EECO integration section + tz/DST note)

Note: `flexcore.config.schema.CostingConfig.tariff_source` (and `.dr.events_source`) already
exist (built in M03) as the **Layer-1 persisted** home for "which tariff/DR file to use." This
milestone does not add config fields — `load_tariff`/`load_dr_program` are the functions that
resolve a `tariff_source`/`events_source` string into an EECO object; FlexCosting (M07) is what
reads `ModelConfig.costing` and calls them. Keep `load_tariff`'s `source` param compatible with
whatever `tariff_source` can hold (a file path — JSON or CSV — today).

## Specification

### 1. Dependency + import localization

- Add `eeco` to `[project] dependencies` (not an extra — it is as core as pyomo).
  Pin it to an **exact** tested version (`==`, per R12), matching the pin already
  in `pyproject.toml` (`eeco==0.2.1` at time of writing); note it in the PR.
- Every `import eeco` in the codebase lives in `flexops/costing/opex.py`. Per
  decision R12 this is a **convention, not an import-linter contract** (conventions
  §6): a `grep` DoD item verifies it. Rationale: EECO is under active upstream
  rework — one import point means one file to fix when its API moves.

### 2. Flex-pse-facing wrapper API (ours; stable regardless of EECO churn)

```python
def load_tariff(source: str | Path | dict | "eeco tariff object") -> "eeco tariff object":
    """Return an EECO tariff object from a path/dict, or pass one through.
    `source` is what a CostingConfig.tariff_source string resolves to (a JSON
    or CSV file path, or an in-memory dict/records structure); a `.csv` path
    is routed through tariff_csv_to_dict() first. Wrap EECO/pydantic load
    errors in FlexDataError naming the file + field."""

def load_dr_program(source) -> "eeco DR object | None":
    """Same, for a demand-response program; None-safe."""

def tariff_csv_to_dict(
    source: str | Path | pd.DataFrame,
    *,
    write_to: str | Path | None = None,
) -> dict:
    """Read a tariff/rate-data CSV (EECO's `rate_data` column schema: utility,
    type, name, month_start/end, weekday_start/end, hour_start/end, charge,
    ...; verify exact columns against EECO) and return the equivalent
    dict/records structure load_tariff accepts.

    `source` is either a path to a CSV file (read from disk) or an already-
    loaded `pd.DataFrame` with the same rate_data columns (e.g. read from disk
    by the caller and pre-filtered/edited before conversion) — both go through
    the same column validation. If `write_to` is given, also persist the
    converted tariff as a JSON tariff-config file at that path (so a CSV can
    be converted once and reused thereafter as a `tariff_source` JSON file,
    per the resolution note above); the dict is returned either way, whether
    or not `write_to` is set.

    A convenience for authoring or importing tariffs in the common CSV exchange
    format (e.g. utility rate-sheet exports) instead of hand-typed JSON; does
    no charge math, just a schema conversion — wrap malformed-CSV errors in
    FlexDataError naming the file + column."""

# --- tariff signal helpers (pandas out, for logic/heuristic constraints) ---
def price_series(tariff, index: pd.DatetimeIndex) -> pd.Series:      # $/kWh per stamp
def is_peak(tariff, index: pd.DatetimeIndex) -> pd.Series:          # bool, highest-price period
def peak_windows(tariff, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
def price_gradient(tariff, index: pd.DatetimeIndex) -> pd.Series:    # Δprice between stamps

# --- DR container (v0 is containers-only; NO DR constraints built) ---
@dataclasses.dataclass
class DRConfig:
    """A container/config slot for a demand-response program (v0 no-op).
    Holds the loaded EECO DR object (or None) so the wiring exists; building
    actual DR constraints is post-v0 (architecture §2.4, PLAN §4)."""
    program: object | None = None        # loaded EECO DR object, or None

# --- the in-objective Pyomo bridge (FlexCosting calls this in M07) ---
@dataclasses.dataclass
class OperatingCostHandles:
    energy_cost: "pyo.Expression"        # $ per timestep, indexed by time
    demand_charge: "pyo.Expression"      # $ per demand window
    total_operating_cost: "pyo.Expression"   # scalar $, horizon total (CONVEX-RELAXED proxy)
    eeco_block: object                   # the raw EECO block/handles, for debugging only

def add_operating_cost(
    *,
    block: "pyo.Block",
    electrical_power,                     # time-indexed kW Var/Expression (aggregate load)
    time_index: pd.DatetimeIndex,        # aligns to electrical_power's index order
    dt_hours: float,                     # timestep length in hours (kW→kWh)
    tariff,
    dr_config: "DRConfig | None" = None, # v0: container only; no DR constraints built
) -> OperatingCostHandles:
    """Ask EECO to build the CONVEX-RELAXED in-objective operating-cost
    Expressions on `block` from the kW series `electrical_power`, and return
    handles under clear flex-pse names. EECO owns the math (energy cost,
    demand-charge epigraphs, kWh conversion); this function only: (1) hands EECO
    the kW series + tariff + timestep, (2) renames EECO's outputs to the stable
    handles above, (3) keeps demand charges linear (epigraph, not max()).
    `total_operating_cost` here is a RELAXED proxy for the objective, NOT the
    reported bill — use evaluate_cost() post-solve for that. DR is containers-only
    in v0: `dr_config` is accepted and stored but builds NO DR constraints
    (call the no-op DR hook)."""

# --- the post-optimization evaluator (the REPORTED bill; R4/R9, §2.4) ---
def evaluate_cost(
    aggregate_power_kw: "np.ndarray",    # realized aggregate power per timestep, kW
    tariff,
    dt_hours: float,                     # timestep length in hours (kW→kWh)
    *,
    dr_config: "DRConfig | None" = None, # v0: ignored (containers-only)
) -> float:
    """Compute the TRUE (de-relaxed) electricity cost by evaluating EECO on a
    FIXED, realized aggregate-power numpy array. This is the user-facing bill
    (the §6 reporting rule): once the dispatch is fixed the pricing
    non-convexity is harmless, so this is an exact post-hoc evaluation, not a
    relaxation. Route all `eeco.*` calls through here (sole import point).
    Returns a plain float ($, horizon total)."""

# --- the gas-utility mirrors (verify against EECO: gas is typically a second
# `utility` key alongside "electric" in the same charge/consumption dicts, not
# a separate tariff object — confirm and adjust `tariff` typing if so) ---
def add_gas_cost(
    *,
    block: "pyo.Block",
    gas_power,                            # time-indexed gas usage Var/Expression (EECO's gas units, e.g. therms/hr or m3/hr)
    time_index: pd.DatetimeIndex,
    dt_hours: float,
    tariff,                               # same tariff object as add_operating_cost, filtered to the gas utility
    dr_config: "DRConfig | None" = None,  # v0: container only; no DR constraints built
) -> OperatingCostHandles:
    """Mirrors add_operating_cost for the gas utility: ask EECO to build the
    CONVEX-RELAXED in-objective gas-cost Expressions on `block` from `gas_power`,
    returning the same OperatingCostHandles shape (gas-flavored). Same rules as
    add_operating_cost: EECO owns the math, no cost math in the wrapper, DR is
    a no-op container in v0."""

def evaluate_gas_cost(
    aggregate_gas_usage: "np.ndarray",    # realized aggregate gas usage per timestep
    tariff,
    dt_hours: float,
    *,
    dr_config: "DRConfig | None" = None,  # v0: ignored (containers-only)
) -> float:
    """Mirrors evaluate_cost for the gas utility: the post-hoc REPORTED gas
    bill, evaluated on a fixed realized gas-usage array. Returns a plain float
    ($, horizon total)."""
```

- The signal helpers should prefer **EECO's own** peak/window/price utilities;
  only compute in pandas here what EECO does not expose. Document, per helper,
  whether it delegates to EECO or is a flex-pse fallback.
- Use the `costs.calculate_itemized_cost` method from EECO to separate the charges. 
- Neither `add_operating_cost` nor `evaluate_cost` may itself write cost math —
  if you find yourself summing prices or building an epigraph here, that logic
  belongs to EECO; call the EECO API instead (verify against EECO). These
  functions are glue + naming.
- `add_operating_cost` (in-objective) and `evaluate_cost` (post-solve) must both
  route through EECO; the in-objective value is the relaxed proxy, the
  `evaluate_cost` value is the reported bill. Document this relationship in the
  module docstring (relaxed ≤ or ≈ true; R4/R9).
- Keep the LP/relaxable character (architecture §3.6): choose EECO options that
  express demand charges as epigraphs. If EECO can only produce a `max()`,
  raise `FlexConfigError` explaining the incompatibility (do not silently ship a
  nonlinear objective) and file an EECO issue.
- **DR no-op hook.** Provide a small internal hook (e.g. `_build_dr(block,
  dr_config)`) that is called by `add_operating_cost` and, in v0, **does
  nothing** but store/verify the container. It exists so M07's `FlexCosting`
  wiring and later DR work are additive (architecture §2.4/§3.6). Do not build DR
  event, curtailment, incentive, or capacity constraints in v0.

### 3. Timezones / DST

Defer to **EECO's** convention — do not implement date arithmetic here.
Determine what EECO does with tz-aware vs. naive indices, make
`flexops.costing.tariff` consistent with it, and document it in the module
docstring and `docs/reference/flexops/costing.rst`. If EECO is naive-local-time
only, reject tz-aware indices with a `FlexDataError` at the wrapper boundary.

### 4. The demo fixture and golden bill

Express this tariff in **EECO's** file format (translate the structure below;
the numbers are the checkable truth): tariff `"flexdemo-b20"` (made-up,
PG&E-B-20-flavored) — summer-weekday (months 6–9) peak `16:00–21:00` at
**$0.18/kWh**; off-peak catch-all **$0.09/kWh**; demand charges `peak_demand`
**$21.50/kW** (same window) and `anytime_demand` **$19.00/kW** (all hours);
`fixed_charge` **$150.00/month**; a tier surcharge **$0.01/kWh** above
**50,000 kWh/month**. (If EECO cannot represent one of these features, note it
in the PR and drop that line from both the fixture and the reference total.)

Reference load (built in the test, hourly, July 2025 = 744 h, 23 weekdays):
constant **100 kW**, except **200 kW at 2025-07-10 03:00** (Thursday, off-peak).
The hand-computed bill — these exact constants are the golden truth EECO must
reproduce:

| Line item | Computation | $ |
|---|---|---|
| Peak energy | 23 d × 5 h × 100 kW = 11,500 kWh × 0.18 | 2,070.00 |
| Off-peak energy | 63,000 kWh × 0.09 | 5,670.00 |
| Tier surcharge | (74,500 − 50,000) kWh × 0.01 | 245.00 |
| `peak_demand` | 100 kW × 21.50 | 2,150.00 |
| `anytime_demand` | 200 kW × 19.00 | 3,800.00 |
| Fixed | | 150.00 |
| **Total** | | **14,085.00** |

`dr_events_demo.json`: one event 2025-07-15 16:00–19:00, `incentive: 0.50`
($/kWh curtailed), in EECO's DR format. **In v0 this file is loaded into a
`DRConfig` container only — no DR constraints are built from it.**

The golden test computes the bill the **clean way**: it builds the realized
aggregate-power numpy array directly (the reference load below — no solve
needed, since the dispatch is given) and asserts
`evaluate_cost(realized_power, tariff, dt_hours=1.0)` reproduces
`total = 14,085.00` and each line item to the cent. This is the post-hoc
evaluation path (R4/R9) and is exact — there is no relaxation once the power is
fixed. A second, smaller check builds the in-objective cost on a toy
`ConcreteModel` (a datetime-ordered time set + the same kW load as fixed
Params — **no FlexOps stack, no TimeBlock dependency**) via
`add_operating_cost(...)`, solves the trivial LP with HiGHS, and asserts the
**relaxed** in-objective `total_operating_cost` is ≤ or ≈ the post-hoc
`evaluate_cost` value (the documented relaxation relationship), within tolerance.

## Pitfalls

1. **Scattering `eeco` imports.** Anything outside `flexops/costing/opex.py`
   importing `eeco` breaks the contract and defeats the churn-localization. Grep
   is a DoD item.
2. **Reimplementing EECO.** If you write a price-lookup loop or a demand epigraph
   here, you have duplicated EECO. Delete it and call EECO. The wrapper is glue.
3. **kW vs kWh double-counting.** EECO converts kW→kWh with the timestep. Pass
   `dt_hours` once; do not also multiply inside the wrapper. The (future) 15-min
   vs hourly agreement check in M07 catches this.
4. **Half-open windows.** Peak `16:00–21:00` excludes hour 21; the 115 peak
   hours in July 2025 depend on it. Confirm EECO's convention and align the
   fixture; if EECO differs, the reference total changes — recompute and note it.
5. **Assuming EECO's API from this file.** Names like `add_operating_cost` and
   `evaluate_cost` are *ours*. EECO's real entry points must be read from its
   docs; record the mapping in the PR so M07 and future readers trust it.
6. **Nonlinear demand charge slipping in.** If EECO emits `max()`, the model
   stops being LP and M07's classifier will flag it late. Fail loud here.
7. **Building DR constraints in v0.** DR is containers-only (architecture §2.4).
   Do not add DR event/curtailment/incentive constraints, and do not let the
   objective change when a DR file is supplied — the DR hook is a no-op that only
   loads the container. Test that supplying `dr_events_demo.json` leaves the
   in-objective cost unchanged.
8. **Confusing the relaxed proxy with the bill.** `add_operating_cost` returns a
   convex-*relaxed* cost for the objective; the reported bill comes from
   `evaluate_cost` on the realized power. Never present the in-objective total as
   the user-facing cost (R9, §6).

## Tests

Test-first (02 §1a). Fixtures in EECO format; golden constants hand-typed.

- `src/flexops/tests/costing/test_tariff_signals.py` — mostly `@pytest.mark.unit`
  (pandas helpers, no solver):
  - `test_price_series_values` — $0.18 on summer-weekday 16:00–20:59 stamps, $0.09 elsewhere.
  - `test_is_peak_and_peak_windows` — `is_peak` True exactly on those stamps; 115 peak hours in July 2025; `peak_windows` is that sub-index.
  - `test_price_gradient` — nonzero at the off-peak→peak and peak→off-peak transitions, 0 within a flat period.
  - `test_tz_or_load_errors` — whatever the EECO-consistent tz policy is (documented), the wrong input raises `FlexDataError` naming the fix.
  - `test_loaders_wrap_errors` — malformed tariff/DR file → `FlexDataError` with file + field path.
  - `test_tariff_csv_to_dict_accepts_path_or_dataframe` — `tariff_csv_to_dict(tariff_tou_demo.csv)` (a path) and `tariff_csv_to_dict(pd.read_csv(tariff_tou_demo.csv))` (an already-loaded DataFrame) return the same dict; the result round-trips through `load_tariff` to the same EECO tariff object as loading `tariff_tou_demo.json` directly.
  - `test_tariff_csv_to_dict_write_to` — `tariff_csv_to_dict(tariff_tou_demo.csv, write_to=tmp_path/"tariff.json")` writes a JSON file at that path (loadable via `load_tariff`) **and** returns the same dict as the no-`write_to` call.
  - `test_dr_container_loads_noop` — `@pytest.mark.unit`. `load_dr_program(dr_events_demo.json)` populates a `DRConfig` container; the DR hook is a no-op (no exception, builds nothing). Assert the container carries the loaded program and that passing it to `add_operating_cost` (see below) adds **no** DR components.
- `src/flexops/tests/costing/test_operating_cost.py`:
  - `test_golden_monthly_bill` — `@pytest.mark.unit` (no solver). Build the realized aggregate-power numpy array (the reference load) and assert `evaluate_cost(realized_power, tariff, dt_hours=1.0)` reproduces each line item **and** the 14,085.00 total, `pytest.approx(abs=0.005)`. This is the clean post-hoc path — no model, no solve.
  - `test_relaxed_leq_or_approx_true` — `@pytest.mark.component` + `@pytest.mark.needs_highs`. Build the in-objective cost on the toy fixed-load `ConcreteModel` via `add_operating_cost`, solve the trivial LP with HiGHS, and assert the relaxed in-objective `total_operating_cost` is ≤ or ≈ (documented relationship) the `evaluate_cost` value on the same realized load, within tolerance.
  - `test_dr_container_is_noop_on_objective` — `@pytest.mark.component` + `@pytest.mark.needs_highs`. Build/solve the toy model with vs. without a `DRConfig` loaded from `dr_events_demo.json`; assert the in-objective cost is **identical** (DR is containers-only — no constraint changes the objective in v0).
  - `test_demand_charge_is_linear` — `@pytest.mark.component` + `@pytest.mark.needs_highs`. `flexcore.solvers.classify` on the built model returns `LP` (no `max()`/nonlinearity slipped in).

## Documentation tasks

- `docs/reference/flexops/costing.rst` — "EECO integration" section: autodoc the
  wrapper API (`load_tariff`, `load_dr_program`, signal helpers,
  `add_operating_cost`, `OperatingCostHandles`, `evaluate_cost`, `DRConfig`); a
  short JSON example of the demo tariff in EECO format; a **"Timezones / DST"**
  admonition stating the EECO-consistent policy; a short **"In-objective vs.
  reported cost"** note (relaxed proxy vs. post-hoc bill, R4/R9) and a
  **"Demand response (containers-only in v0)"** note.
- Module docstring of `flexops/costing/opex.py`: the $/kWh–$/kW conventions,
  the "EECO owns the math; this file is glue" rule, the two-ways-EECO
  relationship (relaxed in-objective ≤ or ≈ post-hoc true bill), the
  DR-containers-only stance, and the import-localization rationale.
- `docs/explanation/` (optional): one paragraph on why tariff/cost is an
  external dependency (EECO) rather than in-repo, and why the reported cost is a
  post-hoc evaluation rather than the solver objective.

## Definition of Done

- [ ] `eeco` is a core dependency pinned exactly (`==`, R12); `import eeco` appears **only** in `flexops/costing/opex.py` (verified by `grep` — per R12 there is no `eeco` import-linter contract, only the package DAG)
- [ ] Wrapper API exports exactly the names above (including `evaluate_cost`, `DRConfig`); signatures match this file; EECO→wrapper API mapping recorded in the PR
- [ ] Signal helpers pass (`unit` tier, no solver); each documents delegate-to-EECO vs. fallback
- [ ] `add_operating_cost` builds the **convex-relaxed** in-objective cost Expressions via EECO (no cost math in the wrapper); model classifies **LP**
- [ ] `evaluate_cost(realized_power)` reproduces the golden bill to the cent (post-hoc, no solve); the relaxed in-objective cost is ≤ or ≈ the post-hoc true cost as documented
- [ ] **DR is containers-only**: `DRConfig` loads the DR file and the DR hook is a no-op; supplying a DR file leaves the objective unchanged (no DR constraints built). The container-loads test passes
- [ ] tz/DST policy matches EECO and is documented; wrong input raises `FlexDataError`
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W` passes with the costing reference section
- [ ] CHANGELOG "Unreleased" entry
- [ ] plus the generic DoD in CLAUDE.md
