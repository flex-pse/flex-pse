# M16 — Design-mode multi-period wrapper

**Effort:** 3 days · **Depends on:** M09, M07 · **Parallelizable:** no

## Goal

Build `flexops.design` — the multi-period design wrapper (architecture §3.6,
decision R3). Real sizing decisions must hold across *several* representative
months, not one. `merge_for_design(period_configs) -> DesignModel` builds
**multiple** representative ≤1-month operations models (each its own TimeBlock +
NetworkBlock via `build_model`), merges them into **one larger Pyomo model**, and
adds **equality constraints tying the sizing variables** (battery capacity, tank
volume, …) across the sub-models — every period shares one size but has its own
operations. `set_design_mode()` (CapEx active, sizing free) applies across all
sub-models. The objective is the sum of per-period operating costs (via each
period's `FlexCosting`) + shared annualized CapEx, and the reported cost
aggregates each period's EECO post-hoc cost (R9). Deliverable: two representative
months (e.g. a summer + a winter tariff) with one shared battery capacity solve
to the **same** capacity but **different** dispatch, and the sizing-var count is
linked, not duplicated.

## Read first

- `plan/01_architecture.md` §3.6 (Costing — the "Multi-period design (the
  design-mode wrapper — `flexops/design/`)" paragraph: this milestone's spec; also
  the single-model `set_operations_mode()`/`set_design_mode()` CapEx machinery
  M16 composes over)
- `plan/01_architecture.md` §3.1 (TimeBlock: each period is its own ≤1-month
  TimeBlock — one TimeBlock never exceeds a month; multi-month studies are
  composed here)
- `plan/01_architecture.md` §3.3 (`NetworkBlock`/`PlantBlock` — each period is a
  NetworkBlock; R7)
- `plan/01_architecture.md` §2.3 (R3: `ModelConfig`, `build_model(config)` — each
  period is built from a `ModelConfig`; the config-driven-everything requirement)
- `plan/01_architecture.md` §6 / §7 (decision **R9**: reported cost is the EECO
  post-hoc `report_cost`, never the raw objective — the aggregated design cost
  sums each period's post-hoc cost)
- `plan/00_conventions.md` §2 (keyword-only constructors), §3 (exceptions:
  `FlexConfigError`), §6 (import discipline — `flexops.design` is inside
  `flexops`, so it may use `eeco` only through `flexops.costing`)
- `plan/02_testing_and_ci.md` §1, §5 (tier markers; solver-availability markers;
  keep the solve small)

## Files to create or modify

- `src/flexops/design/__init__.py` — export `DesignModel`, `merge_for_design`, `DesignConfig`
- `src/flexops/design/merge.py` — `merge_for_design`, the sizing-var linking mechanism
- `src/flexops/design/design_model.py` — `DesignModel` (the merged Pyomo model wrapper: `set_design_mode`, objective assembly, reported-cost aggregation)
- `src/flexops/design/config.py` — `DesignConfig` (a list of period `ModelConfig`s + which vars are shared) OR add it to `flexcore.config.schema` (implementer's choice; if it is a persisted config it belongs in the pydantic schema — conventions §4)
- `src/flexops/__init__.py` — re-export `design` (so `flexops.design.merge_for_design` resolves; the top-level `fo.design` access matches architecture §3.6 naming)
- `src/flexops/tests/design/test_merge.py`, `test_linking.py`, `test_design_solve.py`
- `docs/reference/flexops/design.rst` — reference page for `DesignModel`/`merge_for_design` (M14 sweep expects it)
- `docs/how_to/design_multiperiod.md` — how-to (size across representative periods)

## Specification

### 1. `DesignConfig` (`config.py` or `flexcore.config.schema`)

The persisted config for a design study:

```python
class DesignConfig(...):
    schema_version: str                 # semver string, e.g. "0.0.1" (matches ModelConfig)
    periods: list[ModelConfig]          # N representative ≤1-month operations models
    shared_sizing_vars: list[str]       # dotted names of the sizing vars linked across periods
    annualized_capex: ...               # how CapEx is annualized (implementer's choice of fields)
```

- `periods` is a list of the top-level `ModelConfig` artifact (§2.3) — one per
  representative period (e.g. a summer-tariff month and a winter-tariff month).
  Each period builds a full, independent operations model.
- `shared_sizing_vars` names the sizing variables (dotted
  `plant.unit.variable`-style paths, resolved per period) that must be equal
  across periods — e.g. `["svcw.battery.capacity"]`. A name absent from a period's
  registered sizing vars → `FlexConfigError` naming the period and the var.
- If `DesignConfig` is persisted (it should be — it drives a whole study), it is a
  pydantic v2 model with `schema_version` and field descriptions, exported to JSON
  Schema like the rest (R3, conventions §4). If the implementer keeps it a plain
  dataclass for v0, document why in the PR — but prefer the schema.

### 2. `merge_for_design` (`merge.py`)

```python
def merge_for_design(period_configs, *, shared_sizing_vars=None) -> "DesignModel":
    """Build one merged Pyomo model from N period ModelConfigs and link sizing vars.

    Accepts either a DesignConfig (reads periods + shared_sizing_vars from it)
    or an explicit list of ModelConfigs plus shared_sizing_vars= (implementer's
    choice; support at least the DesignConfig path).
    """
```

Construction sequence:

1. **Build each period.** For each period `ModelConfig`, call
   `flexops.build_model(config)` to construct a full independent operations model
   — its own TimeBlock (≤1 month, §3.1) + NetworkBlock (§3.3) + FlexCosting. No
   period shares state with another; only the sizing vars are linked (step 3).
2. **Merge into one Pyomo model.** Attach each period's model as a named
   sub-block on one parent `ConcreteModel` (e.g. `design.period[k]` or
   `design.summer` / `design.winter` — implementer's choice; keep names stable and
   derivable from the config). Nothing in a period couples to another period's
   time-indexed variables.
3. **Link the sizing vars (the mechanism).** For each name in
   `shared_sizing_vars`:
   - collect that registered sizing Var from **every** sub-model (via the
     costing/sizing registry each period's `FlexCosting` holds — the sizing Vars
     units register with costing, §3.6);
   - create **one shared parent Var** for the size (e.g. `design.size[var_name]`)
     with the common bounds/units;
   - add an equality constraint **per period** tying that period's sizing Var to
     the shared Var: `size_link[period, var_name]: period_var == design.size[var_name]`.
   Every period therefore sees the **same** size but keeps its **own** operations
   (dispatch, SOC trajectory, tank levels). The count of independent sizing DOF is
   one shared Var per linked name — **linked, not duplicated** (N periods do not
   create N independent capacities).

### 3. `DesignModel` (`design_model.py`)

A thin wrapper around the merged Pyomo model:

```python
class DesignModel:
    model: pyo.ConcreteModel          # the merged model
    periods: dict[str, ...]           # name -> period sub-model
    shared_sizes: dict[str, pyo.Var]  # var name -> shared Var

    def set_design_mode(self) -> None: ...
    def build_objective(self) -> None: ...
    def report_cost(self) -> "DesignCostReport": ...
```

- **`set_design_mode()`** applies the single-model design mode (§3.6) **across all
  sub-models**: for each period call its `FlexCosting.set_design_mode()` (unfix
  sizing vars, activate CapEx). Because the sizing vars are equality-linked, the
  free size is effectively the single shared Var. (The single-period
  `set_design_mode` is the special case of N=1.)
- **`build_objective()`** assembles the design objective:
  `objective = Σ_period (period operating cost) + shared annualized CapEx`.
  Per-period operating cost is each period's in-objective EECO-relaxed operating
  cost `Expression` (via that period's `FlexCosting`, §3.6). CapEx is computed
  **once** off the shared sizing Vars and annualized (so N periods do not pay N×
  CapEx). Document the annualization and weighting (representative periods may
  carry a weight — implementer's choice; default equal weight, note it).
- **`report_cost()`** — the reported number (R9): aggregate each period's **EECO
  post-hoc** cost (`period FlexCosting.report_cost`), **not** the solver objective.
  Return a `DesignCostReport` (dataclass) bundling per-period `report_cost` values
  and their aggregate (plus the shared CapEx). The raw objective is surfaced only
  behind an explicit debug flag, exactly as M12/M13 (R9).

## Pitfalls

1. **Duplicating sizing DOF.** N periods must share **one** capacity Var, linked
   by equality — not N free capacities. If your solved model reports a different
   size per period, the linking constraint is missing or wrong; the
   `test_shared_capacity_equal` test exists to catch this.
2. **A single full-horizon model instead of merged ≤1-month models.** Each period
   is its own ≤1-month TimeBlock + NetworkBlock (§3.1/§3.3); do **not** try to
   build one multi-month TimeBlock (a TimeBlock never exceeds one month — §3.1).
   Merge independent period models; link only the sizes.
3. **Coupling period operations.** Only sizing vars are linked. Never add
   constraints across periods' time-indexed dispatch/SOC/level variables — the
   periods are independent operations problems sharing a size.
4. **Reporting the objective as the cost.** The design cost report aggregates each
   period's EECO post-hoc `report_cost` (R9), never the merged solver objective
   (a relaxed/scalarized proxy). The objective is debug-flag only.
5. **Paying CapEx N times.** CapEx is computed once off the shared size and
   annualized; summing each period's CapEx term would multiply it by N. Assemble
   CapEx from the shared Vars, not per-period.
6. **Sizing var not registered.** A name in `shared_sizing_vars` that a period did
   not register with costing → `FlexConfigError` naming the period and var; do not
   silently create an unlinked Var.
7. **`eeco` import discipline.** `flexops.design` is inside `flexops`; it must
   reach EECO only through `flexops.costing` (conventions §6). Do not import
   `eeco` in the design package.

## Tests

All in `src/flexops/tests/design/`. Exactly one tier marker each.

`test_merge.py`:
- `test_merge_builds_n_period_models` (`unit`) — `merge_for_design` on two
  period `ModelConfig`s produces a `DesignModel` with two named period sub-models,
  each a full independent model (its own TimeBlock, NetworkBlock, FlexCosting).
- `test_periods_operations_independent` (`unit`) — no constraint couples the two
  periods' time-indexed dispatch variables (assert the linking constraints touch
  only sizing Vars + the shared Var).

`test_linking.py` — all `@pytest.mark.unit` (mechanics, no solver):
- `test_shared_var_created_once` — one shared Var per linked name regardless of N
  periods; `size_link[period, var]` equality constraints exist one per period.
- `test_sizing_dof_linked_not_duplicated` — independent sizing DOF equals the
  number of shared names, not `N × names` (count free sizing Vars after
  `set_design_mode`; the per-period sizing Vars are pinned to the shared Var by
  equality).
- `test_unregistered_shared_var_raises` — a `shared_sizing_vars` name absent from
  a period's registered sizing vars → `FlexConfigError` naming period + var.
- `test_set_design_mode_all_periods` — after `set_design_mode()`, every period's
  CapEx terms are active and its sizing vars are unfixed (subject to the equality
  link).

`test_design_solve.py`:
- `test_two_month_shared_battery` (`component` + `needs_highs`) — two
  representative months (a **summer** and a **winter** TOU tariff) with a small
  tank+battery NetworkBlock, one shared `battery.capacity`; build the objective
  (Σ period operating cost + annualized CapEx), `set_design_mode()`, solve with
  HiGHS. Assert: (a) the two periods report the **same** capacity
  (`pytest.approx(rel=1e-6)` — the equality link holds); (b) the two periods'
  **dispatch differs** (the SOC/charge trajectories are not equal — different
  tariffs drive different operations under one size); (c) `report_cost()`
  aggregates the two periods' EECO post-hoc costs and does **not** return the raw
  solver objective by default. Keep it small (≤ 1-day periods at coarse
  resolution) to stay in the component budget.
  **Note (M04 tank ripple):** `set_design_mode()` unfixes *every* registered
  sizing Var, so if this NetworkBlock's tank is also unfixed, its bilinear
  `level_definition` (`volume[t] == level[t] * capacity`, M04) makes the model
  **NLP**, not LP — even though only `battery.capacity` is the equality-linked
  var. If the tank is present here, either exclude its `capacity` from
  design-mode unfixing (keep it fixed at `max_volume` for this test) or adjust
  the solver marker to `needs_ipopt` / a relax→NLP `SolveSequence` (R5) instead
  of `needs_highs`.

## Documentation tasks

- `docs/reference/flexops/design.rst` — reference page for `DesignModel` and
  `merge_for_design` (autodoc/autosummary); added to the flexops reference index.
  M14's reference sweep expects this page to exist (clear its `TODO(M16)` stub if
  M14 ran first).
- `docs/how_to/design_multiperiod.md` — how-to: size a battery across a summer +
  winter representative month with `merge_for_design`, `set_design_mode`, solve,
  and read the aggregated EECO post-hoc cost via `report_cost`. Small horizons,
  fixed seeds, ends on a numeric assertion if it is notebook-backed.
- `docs/explanation/` cross-reference: link the reported-cost note (M14 §4b, R9)
  — the design cost aggregates per-period post-hoc costs, never the objective.
- CHANGELOG entry under "Unreleased" (`flexops.design` is user-visible).

## Definition of Done

- [ ] `merge_for_design` builds N independent ≤1-month period models via
      `build_model` and merges them into one Pyomo model (each its own TimeBlock +
      NetworkBlock + FlexCosting)
- [ ] Sizing vars are equality-linked across periods to one shared Var
      (`size_link[period]`); the sizing DOF is linked, not duplicated
- [ ] `set_design_mode()` applies CapEx-active / sizing-free across all sub-models
- [ ] Objective = Σ per-period operating cost + shared annualized CapEx (CapEx paid
      once off the shared size)
- [ ] `report_cost()` aggregates each period's EECO post-hoc cost (R9); the raw
      objective is surfaced only behind an explicit debug flag
- [ ] Two-month shared-battery solve: both periods report the same capacity, with
      different dispatch (component + `needs_highs`)
- [ ] Linking/merge mechanics covered by unit tests; unregistered shared var raises
      `FlexConfigError`
- [ ] `flexops.design` reaches EECO only through `flexops.costing`; import-linter clean
- [ ] `docs/reference/flexops/design.rst` + how-to build with `sphinx-build -W`;
      CHANGELOG updated
- [ ] plus the generic DoD in CLAUDE.md
