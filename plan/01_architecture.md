# 01 — Architecture

The design reference for all milestones. Milestone work orders cite sections of
this file; when in doubt, this file wins.

## 1. Package dependency DAG

```
            ┌──────────────────┐   ┌──────────────────┐
            │ flexparameterize │   │   flexschedule   │
            └────────┬─────────┘   └────────┬─────────┘
                     │      (never import   │
                     │       each other)    │
                     ▼                      ▼
                  ┌─────────────────────────────┐
                  │           flexops           │
                  └──────────────┬──────────────┘
                                 ▼
                  ┌─────────────────────────────┐
                  │          flexcore           │   ← imports only pyomo/idaes/
                  └─────────────────────────────┘     pandas/pydantic
```

The external **EECO** package (`eeco`, PyPI) is a core runtime dependency used
only by `flexops.costing` (§2.4, §3.6).

Enforced by import-linter (see `plan/00_conventions.md` §6). The *import*
direction is one-way (`flexparameterize` imports `flexops`, never the reverse),
but the **data/control coupling between FlexParameterize and FlexOps is
two-way** (see §5):

- **FlexOps → FlexParameterize:** FlexOps constructs the model *containers* — the
  plant/network/unit blocks with their registered IO variables and regressable
  parameters (the `IORegistry`). FlexParameterize reads that structure to know
  what can be fitted.
- **FlexParameterize → FlexOps:** given plant data, FlexParameterize matches data
  streams to registered parameters and **mutates the live FlexOps model** —
  fixing regressed parameters and, where the fit yields a richer relationship
  than the unit's default constant intensity, swapping the unit's
  energy-relationship constraint *in place* for one built from the fitted
  `SurrogateSpec` (same unit object, same ports/arcs — nothing to reconnect).

Both directions still flow through `flexops`' public API (registration +
constraint-swap hooks that `flexparameterize` calls), so the import layering holds.
The **durable, serializable** contract between them remains the versioned config
(§2.3): FlexParameterize can either mutate a live model *or* emit a config that
rebuilds the parameterized model. That config seam is where the repo splits
later.

## 2. flexcore

### 2.1 upstream dependencies — pin, don't isolate

**Decision R12: pin exact `idaes-pse`/`pyomo` versions; import both directly; no
compat/isolation layer.** There is no re-export gateway, no whitelist, no
tracked-import allowlist, no `check_environment()`, and no upstream canary.
`idaes.*` and `pyomo.*` are imported **directly at point of use** throughout the
codebase.

A re-export whitelist only ever guards against *mechanical import-path drift* —
the cheapest upstream failure to fix, at a bureaucratic cost that exceeds the
saving. The *expensive* failure, semantic/behavioral drift, is invisible to a
re-export layer anyway. So the isolation layer buys little. (WaterTAP, the
closest sibling project, imports `idaes.*` directly throughout.) Instead we
follow a standard dependency-management cycle:

- `pyproject.toml` **pins `idaes-pse` and `pyomo` to exact tested versions**
  (`==`), defaulting to the latest release at implementation time (M00 sets
  them).
- Maintainers **bump the pins manually**, roughly quarterly, only after the full
  test suite passes against the newer versions. If manual bumps ever become
  hectic, revisit automating a compatibility check then — not before.
- `pyunits` is imported directly (`from pyomo.environ import units as pyunits`);
  the canonical energy names live in `flexcore.nomenclature` (§4).

The external **EECO** package (§2.4) is handled the same way: no import-linter
whitelist or isolation contract — `eeco` is imported directly where it is used
(naturally, that is `flexops/costing/`), its version is pinned in
`pyproject.toml`, and a maintainer bumps it manually.

### 2.2 solvers — capability-detecting facade

- `classify.py`: walk a Pyomo model → `ProblemClass` enum
  (`LP`, `QP`, `MILP`, `NLP`, `MINLP`). Discrete vars that are *fixed* do not
  count as discrete (an LP with fixed binaries classifies LP).
- `registry.py`: which solvers are installed; capability matrix —
  `highs: {LP, MILP}`, `cbc: {LP, MILP}`, `ipopt: {NLP}`,
  `gurobi: {LP, QP, MILP}` (optional), extensible.
- `facade.py`: `get_solver(model=None, problem_class=None, prefer=None) -> SolverFacade`.
  Picks the best available solver for the class; raises `FlexSolverError` with
  install instructions when none qualifies.
- **R5 (decision): classify loudly, never transform silently.** The facade never
  relaxes integrality, never decomposes, never sets up trust regions on its own.
  A MINLP with only HiGHS+IPOPT installed errors with: "this model is MINLP;
  compose a `flexschedule.SolveSequence` (relax → MIP → fix → NLP) or install a
  MINLP-capable solver." Silent model surgery is a correctness landmine for
  plant-control users. MINLP strategies (outer approximation, trust region) are
  reserved enum slots, deferred post-v0.

### 2.3 config — the versioned contract, and the single source of truth

**Design principle: the entire model and run are built from one
version-controllable config file — no manual, in-code configuration is
required.** A user (or, eventually, an external module that writes the config
programmatically — GUI, PyPES/WaTr translator, FlexParameterize) hands flex-pse
one config artifact, and flex-pse constructs the TimeBlock, properties, costing,
the network/plant/unit tree, and the solve settings from it. Hand-written Python
like the API-freeze script (PLAN.md §2) stays as a supported *thin* path and as
the readable illustration of what the config drives, but nothing essential may
live only in imperative code — if it can be configured, it is in the file.

- Pydantic v2 models in `config/schema.py` are the **schema authority**:
  - `IOVariableSpec` — name, role (`input`/`output`), units, tag hint,
    time-indexed flag.
  - `SurrogateSpec` — functional form (`constant_intensity`, `linear`, reserved:
    `nn`, `arima`, `multiconvex`), coefficients, input/output variable names,
    `provenance` (fit metrics, data window, tool versions).
  - `UnitConfig` — unit-model class name, construction options, IO specs,
    optional `SurrogateSpec`, optional logic/UC block (§3.5), optional external
    dispatch source (§3.6).
  - `PlantConfig` — a named collection of `UnitConfig`s + arcs.
  - `NetworkConfig` — a named collection of `PlantConfig`s + inter-plant arcs.
  - `TimeConfig` — `start_date`, `end_date`, `time_step` (§3.1).
  - `CostingConfig` — tariff source, DR container (§3.6), solve/objective options.
  - `ModelConfig` — the top-level artifact: `schema_version: int` (mandatory),
    `TimeConfig`, properties, `CostingConfig`, and a `NetworkConfig` *or*
    `PlantConfig`. `flexops.build_model(config)` constructs the whole Pyomo model
    from it.
- `config/io.py`: `load_model_config(path) -> ModelConfig`,
  `dump_model_config(cfg, path)`; a `schema_version` migration hook table.
- **R3 (decision):** the config is **pydantic v2 (schema authority) serialized
  to YAML as the canonical on-disk format**, versioned by `schema_version`, with
  JSON Schema exported into `config/schemas/` and checked in (schema drift shows
  in diffs; docs render it; external writers validate against it). YAML is chosen
  over JSON as canonical because these files are both human-tracked in VCS
  (comments, readable diffs, anchors for repeated tariff/unit blocks) *and*
  written programmatically; JSON is still accepted on load. Dodge the YAML
  "Norway problem" with a strict, typed loader (pydantic coerces; ambiguous bare
  scalars like `no`/`on` are quoted by the dumper). Pyomo `ConfigDict` remains
  runtime-only and is never persisted. *(This reverses the earlier
  JSON-canonical call in response to the config-driven-everything requirement;
  the format is a recommendation — the schema-authority-in-pydantic decision is
  the load-bearing part and is format-independent.)*

### 2.4 tariffs and costs — the external EECO package

flex-pse does **not** build its own tariff/cost engine. Tariffs, demand charges,
tiered/fixed charges, and both the optimization-time and post-optimization cost
computations come from the external **EECO** package (Energy Economics / Cost
Optimization), installed from PyPI as `eeco` and listed as a core runtime
dependency. EECO applies a **convex relaxation** to the (non-convex) pricing
structure, which is what makes tariff-aware scheduling tractable.

EECO is used in **two** ways (§3.6):

1. **In-objective (Pyomo-aware).** Given the model's aggregate electrical-work
   (kW) indexed variable + a tariff, EECO builds the convex-relaxed cost
   `Expression` the scheduler minimizes.
2. **Post-optimization evaluation (the reported number).** After a solve, EECO
   is called on a **time-indexed numpy array of realized aggregate power
   (kW)** to compute the electricity cost that is reported to the user. Because
   the in-objective cost is a *relaxed* proxy (and the objective may also carry
   scalarized emissions / penalty / lost-production terms), the raw solver
   objective value is **never** the user-facing cost — see the reporting rule in
   §6. This post-hoc evaluator is also the comparison metric for multi-stage /
   iterative schemes that do not need duals.

Consequences for this architecture:

- There is no `flexcore.econ` module. `flexcore` holds only `solvers` and
  `config`.
- The EECO integration lives under **`flexops.costing`** (see §3.6): a thin
  interface layer (`flexops/costing/opex.py`) plus the `FlexCosting` block.
  The user's tariff files are in **EECO's own format**, handed to EECO's loaders.
- **`eeco` is imported directly where used** — no import-linter whitelist or
  isolation contract (decision R12). Keeping EECO calls collected in
  `flexops/costing/opex.py` is a sensible design convention (one thin wrapper
  for a package under active rework), not an enforced boundary. EECO's version is
  pinned in `pyproject.toml` and a maintainer bumps it manually.
- **Demand response (DR) is not implemented in v0** — provide **containers only**
  (a `DRConfig` slot in `CostingConfig`, a `dr` placeholder on `FlexCosting`, and
  a no-op DR hook) so the wiring exists and turning DR on later is additive.
  Nothing builds DR constraints yet. (This walks back the earlier "DR constraints
  in EECO" for v0 per the container-for-now decision.)
- Timezone/DST behavior follows **EECO's** conventions — verify and document.
- Clear naming: `FlexCosting` re-exposes EECO's cost quantities under stable
  flex-pse names and maps totals into IDAES aggregates (§3.6).
- Tariff **signal helpers** for writing logic/heuristic constraints (peak
  detection, windowing, price gradient) live in `flexops/costing/opex.py` and
  return plain pandas objects aligned to `time_block.datetime_index`. Prefer
  EECO's own helpers; wrap only what EECO lacks.

## 3. flexops block architecture

Mirrors the architecture slides: TimeBlock, OpsBlock, PlantBlock, CostingBlock,
plus a property package. ProcessBlock comes from IDAES as-is (no custom version).

### 3.1 TimeBlock (`flexops/core/time_block.py`)

- The discrete-time substrate for everything. Holds:
  - `time_index`: ordered Pyomo Set of integer indices `0..N-1`;
  - `time`: Param indexed over `time_index` giving elapsed time `i*dt` in the
    user's units (the "actual" time points);
  - `dt`: Param with `pyunits` (the resolution, e.g. `15 * pyunits.min`);
  - datetime↔index utilities: `index_of(timestamp)`, `timestamp_of(i)`,
    `datetime_index` (a `pd.DatetimeIndex` mirror for tariff/EECO alignment);
  - rolling-horizon hooks: `register_initial_state(param)` — a registry of
    Params (tank level, battery SOC, unit on/off state) that a rolling-horizon
    driver mutates between windows; `window(start, length)` slicing metadata.
- Constructor: `TimeBlock(start_date=..., end_date=..., time_step=...)` with
  ISO-8601 strings or datetimes.
- **Horizon scope:** the block spans `[start_date, end_date]` and represents a
  problem of **at most one calendar month** (validate `end_date - start_date <=`
  one calendar month from `start_date` — *not* a fixed 30 days; a Feb→Mar or
  Jan→Feb window are both "one month"). Longer studies are built as multiple
  ≤1-month models composed by the design-mode wrapper (§3.6) or stepped by the
  rolling-horizon driver (§6) — a single TimeBlock never exceeds one month.
- **Resolution is free.** `time_step` defaults to `15 * pyunits.min` but accepts
  any positive duration (1 min … hours). `N = (end - start) / dt` must be a whole
  number; a non-divisible pair raises `FlexConfigError`. Nothing in the code
  assumes 15 min — always compute from `dt`.

### 3.2 OpsBlock (`flexops/core/ops_block.py`)

The base class of every flex-pse unit model.

- **R1 (decision): inherit IDAES `UnitModelBlockData`** (via
  `declare_process_block_class`, imported directly from `idaes.core`), because it provides
  ConfigBlock machinery, Port construction, the initialization framework, and
  the costing registration hooks FlexCosting needs — and keeps the "WaterTAP
  ecosystem" positioning honest. **Do NOT use IDAES ControlVolumes**: they drag
  in the full property/holdup framework that is wrong for scheduling surrogates.
  Mass/energy balances are written by hand in each unit (1–3 constraints each).
- Registration API (consumed by FlexParameterize and by the docs generator):
  - `register_io_variable(var, role="input"|"output", tag_hint=None)` —
    declares a variable as a process input/output. IO variables get **fixed
    during regression** (they carry the data).
  - `register_process_parameter(param_or_var, regressable=True)` — design or
    regression parameters, **found during regression**.
  - `register_energy(var, kind="electrical"|"thermal")` — wires a unit's kW
    draw into plant/costing aggregation.
  - Registries are dataclasses in `flexops/core/registration.py`
    (`IORegistry`), discoverable model-wide via `iter_io_registry(model)`.
- Base-class-provided `electrical_power[t]` / `thermal_power[t]` Vars (kW),
  created when the unit declares it consumes that energy kind.
- Config flags every unit inherits: `relaxation` policy for its discrete
  structure, a `unit_commitment` sub-config (§3.5), `allow_bypass`, and an
  optional **`external_dispatch`** source (below).
- **External dispatch (DERMS).** Every unit exposes a hook to fix its
  controllable actuator variable(s) to an externally supplied, time-indexed
  command series — removing the dispatch degree of freedom while leaving sizing
  free. This is the mechanism for third-party-controlled assets whose logic the
  facility does not own (batteries under a DERMS/aggregator being the motivating
  case — Project 1). Base method
  `set_external_dispatch(var, series, *, fix=True)` fixes `var[t]` to
  `series[t]`; a `external_dispatch` config block declares which variable and the
  data source. It is available on **all** units and tools (schedule/parameterize
  respect fixed dispatch), first-classed on `BatteryModel` (§3.4/§3.6).
- Constructible from validated config — a unit is built from a `UnitConfig`
  (§2.3) via `OpsBlock.from_config(cfg, ...)`; whole models are built via
  `flexops.build_model(model_config)`.

### 3.3 Composition: PlantBlock and NetworkBlock (`flexops/core/`)

Two levels of composition, mirroring the "collection of" pattern:

- **`PlantBlock`** (`plant_block.py`) = a collection of **unit** blocks (a
  facility). Holds arcs between its units and aggregates
  `total_electrical_power[t]`, `total_thermal_power[t]`, and chemical-use
  Expressions over its registered child units.
- **`NetworkBlock`** (`network_block.py`) = a composition of **plant** blocks
  (a portfolio / campus / multi-facility system), with inter-plant arcs and
  recursive aggregation over its child plants. This is the "more aptly named"
  home for plant-of-plants composition: `PlantBlock` composes units,
  `NetworkBlock` composes plants. (The earlier design overloaded `PlantBlock`
  to nest into itself; a plant containing plants is now a `NetworkBlock`.)
- **R2 (decision): never `dynamic=True`, never Pyomo.DAE.** Both blocks are thin
  `FlowsheetBlockData` subclasses constructed with `dynamic=False` and `time_set`
  injected from the TimeBlock's ordered discrete set. All dynamics (tank holdup,
  battery SOC) are explicit difference equations against `time_block.dt`.
  Rationale: MIP scheduling needs integer index arithmetic (dwell times, startup
  delays, rolling windows); DAE discretization fights binaries and rolling
  horizons.
- Both take an explicit `time_block=` (units/plants inherit time from their
  parent via the IDAES `flowsheet()` chain). Auto-discovery of a unique
  root-model TimeBlock may be a convenience default, but explicit must work
  first. Aggregation composes: `NetworkBlock` totals = sum of child
  `PlantBlock` totals = sum of their unit `electrical_power`/`thermal_power`.

### 3.4 Unit model library — IO-topology base classes + a physical zoo

Unit models are organized by **inlet/outlet topology** first, then specialized
physically. The topology base classes (`flexops/unit_models/base/`) own port
construction, per-stream mass balance, and the energy-registration wiring; the
physical subclasses add the flow↔energy relationship and any bounds.

**Base topology blocks** (all subclass `OpsBlockData`):

| Base | Module | Ports | Meaning |
|---|---|---|---|
| `SISOBlock` | `base/siso.py` | 1 in → 1 out | single input, single output |
| `SIDOBlock` | `base/sido.py` | 1 in → 2 out | single input, double output (a split) |
| `DIDOBlock` | `base/dido.py` | 2 in → 2 out | double input, double output (two coupled streams) |

**Physical units** (v0 build order in the milestones):

| Class | Base | Notes |
|---|---|---|
| `Pump` | `SISOBlock` | `electrical_power[t] = energy_intensity * flow_vol[t]` |
| `Tank` | `SISOBlock` | holdup `V[t+1] = V[t] + dt*(in − out)`; level bounds; initial level is rolling-horizon state. **Logic/unit-commitment constraints disabled** (a tank has no on/off status) |
| `Separator` | `SIDOBlock` | one feed split into two product streams (replaces the old `Electrolyzer` name) |
| `Exchanger` | `DIDOBlock` | two inlet / two outlet streams exchanging mass/energy |
| `ElectrolysisSeparator` | `Separator` | electrolysis modeled as a separation; exercises `thermal_power` |
| `ElectrolysisExchanger` | `Exchanger` | electrolysis with two coupled streams |
| `ReverseOsmosisSkid` | `Separator` | RO skid: feed → permeate + concentrate |
| `Combustor` | `Separator` | combustion as a separation of products |
| `BatteryModel` | `OpsBlockData` (no fluid ports) | SOC dynamics, charge/discharge power + efficiency, capacity as fixable design var; optional mutually-exclusive charge/discharge binary; first-class `external_dispatch` (DERMS, §3.6) |
| `ConstantEnergyIntensityModel` | `SISOBlock` | generic "energy factor × flow" unit — the default building block for anything without a bespoke physical topology (e.g. a whole plant modeled as a single surrogate, as in the api-freeze script's `svcw.plant`) |

The general pattern (the platform's core idea): a unit model defines flows in/out (its topology) and energy draw; **every unit defaults to a constant energy-intensity relationship**. FlexParameterize (§5) is what upgrades that relationship to a fitted linear/multiconvex/NN/ARIMA form — by swapping the
unit's energy-relationship constraint in place, not by introducing a different unit class. These are usually energy relationships but can also modify the input/output relationship for quantities like biogas production, salt and permeate flux, etc., not by introducing a different unit class. Same base topology, controllable functional form. `Tank` inheriting `SISOBlock` but *disabling* logic constraints is the canonical example of a physical subclass turning off a base capability.

### 3.5 logic layer (`flexops/logic/`) — customizable unit commitment

A composable unit-commitment (UC) formulation, applied per unit via its
`unit_commitment` config (§2.3), every piece optional except status:

- `status.py`: the base — Binary `status[t]` on **every unit that can be shut
  off** (a tank cannot, so it disables this — §3.4), with semicontinuous linking
  (`min_output*status[t] <= x[t] <= max_output*status[t]`), plus
  `relax()`/`unrelax()` switching the domain to `UnitInterval` per the unit's
  relaxation policy. Toggling relaxed↔exact is first-class, not a rebuild.
- `startup_shutdown.py` (**optional**): `startup[t]`, `shutdown[t]` binaries with
  the standard `status[t] - status[t-1] = startup[t] - shutdown[t]` transition
  logic.
- `dwell.py` (**optional**): minimum up/down-time (dwell) sequence constraints so
  decisions can't flap faster than a minimum interval.
- `delays.py` (**optional**): startup/response **delays connected to an upstream
  unit** — e.g. this unit may not start until `k` steps after an upstream unit's
  status/startup (the chemical-stabilization delays of Rao 2024).
- `conditional.py` (**optional**): conditional logic between units — "if x is on
  then y is on" / "…then y is off" — as implication constraints on the statuses.
- **Parallel-train degeneracy detection** (`degeneracy.py`): symmetry among
  identical parallel trains creates solver-time degeneracy. Detection +
  symmetry-breaking is **implemented outside the unit level** — a model-level
  pass over a `NetworkBlock`/`PlantBlock` that identifies interchangeable units
  and adds ordering (lex) constraints. It is *not* a per-unit concern (a unit
  cannot see its siblings), so it lives as a model-level utility, not on
  `OpsBlockData`.
- `bypass.py`: bypass-stream constraints around a unit.
- Post-v0 (backlog): full parallel-train *replication* helper and startup-delay
  chain templates from Rao 2024 built on these primitives.

### 3.6 Costing (`flexops/costing/flex_costing.py`) — wraps EECO

- **R4 (decision):** `FlexCosting(FlowsheetCostingBlockData)` — subclass IDAES
  costing for its registration and CapEx machinery, and **delegate tariff
  operating cost to the external EECO package** (§2.4) in two ways:
  - **In-objective:** hand EECO the aggregate electrical-work (kW) indexed Var +
    tariff; EECO builds the convex-relaxed operating-cost `Expression` the
    scheduler minimizes. `FlexCosting` does not re-implement price series or
    demand-charge epigraphs — that is EECO's code.
  - **Post-optimization:** after a solve, `FlexCosting.report_cost(model)`
    extracts the realized aggregate power as a time-indexed numpy array and
    calls EECO's evaluator to compute the **reported** electricity cost (the
    true, de-relaxed cost). This — not the solver objective — is the user-facing
    number (§6 reporting rule).
- **DR is containers-only in v0** (§2.4): `CostingConfig.dr` slot, a `dr`
  placeholder attribute, and a no-op `_build_dr()` hook that later milestones
  fill. No DR constraints are built yet.
- `FlexCosting`'s own responsibilities (the flex-pse boundary around EECO):
  - **Clear naming.** Re-expose EECO's cost quantities under stable flex-pse
    names, mapping totals into IDAES aggregates (`aggregate_operating_cost`,
    `aggregate_capital_cost`) — objective and downstream code never touch raw
    EECO internals (§2.4).
  - **Aggregation.** Sum registered units' `electrical_power[t]` (and
    `thermal_power[t]`) into the kW series EECO consumes, in-model and as the
    post-solve numpy array.
  - **CapEx + modes** (below).
- EECO receives a **kW series**; kWh conversion is EECO's. Keep the LP/relaxable
  character (epigraph demand charges, not `max()`). By convention `eeco` calls
  are collected in `flexops/costing/opex.py` (not enforced — see §2.4).
- **CapEx + operations vs. single-model design.** Sizing Vars (battery capacity,
  tank volume) are created by units and registered with costing; constructor
  values initialize them. `set_operations_mode()` fixes all sizing vars
  (scheduling problem); `set_design_mode()` unfixes them and activates CapEx
  terms (a single Pyomo (de)activation call — everything stays in one model).
- **Multi-period design (the design-mode wrapper — `flexops/design/`).**
  Real sizing decisions must hold across *several* representative months, not
  one. The design wrapper builds **multiple representative ≤1-month operations
  models** (each its own TimeBlock + NetworkBlock, §3.1/§3.3), merges them into
  **one larger Pyomo model**, and adds **equality constraints tying the sizing
  variables across the sub-models** (each period sees the same battery/tank
  size, but its own operations). The per-period CapEx-active design mode above is
  the single-period special case. This wrapper is a distinct tool
  (`flexops.design.DesignModel` / `merge_for_design(...)`) around the operations
  model, not a mode flag — it composes operations models rather than living
  inside one.
- Construction-order invariant: `FlexCosting` may be constructed before any
  units exist **because all aggregation is deferred to `cost_process()`** —
  document and test this (construction-order permutation test).

### 3.7 Properties (`flexops/properties/simple_aqueous.py`)

- `SimpleAqueousFlow(fixed_density=True)`: minimal
  `PhysicalParameterBlock`/StateBlock pair — volumetric flow, optional pressure/
  temperature, fixed density. Modeled on WaterTAP's zero-order property package
  (`prop_ZO`) as the structural reference. Ports carry flow between units via
  standard IDAES/Pyomo `Arc`s.

## 4. Energy nomenclature (project standard)

| Name | Meaning | Units | Consumer |
|---|---|---|---|
| `electrical_power[t]` | unit-level electrical draw (motor/drive) | kW | FlexCosting → EECO (energy + demand charges + DR); plant aggregation |
| `thermal_power[t]` | unit-level heat/gas-driven duty | kW | separate thermal aggregation/costing |

Rules: every unit model registers at least one of these via `register_energy`.
FlexCosting aggregates them into a kW time series and hands it to EECO both
in-model (objective) and as a post-solve numpy array (reporting, §3.6/§6); EECO
computes kWh internally from the timestep.
Never name a variable bare `power`/`energy`/`work`. Defined as constants in
`flexcore.nomenclature` so string typos are import errors.
(Naming was agreed to disambiguate "work vs power" inconsistencies in
IDAES/WaterTAP; the group may revisit the words, which is why they live in one
constants module.)

## 5. flexparameterize

A **two-way** partner to FlexOps (§1): FlexOps constructs the model containers
(registered IO variables + regressable parameters); FlexParameterize matches
data streams to those parameters and then either **mutates the live model** or
**emits a config** that rebuilds the parameterized model.

Pipeline: **tabular data → tag aliasing → sufficiency validation → regression →
{apply to model | emit config}**.

- `tags.py`: `TagMap` — historian/database tag ↔ `plant.unit.variable` alias
  pairs (YAML/JSON loadable). Reports unmapped tags with fuzzy suggestions.
- `validate.py`: given the `IORegistry` of a built (or config-declared) FlexOps
  model plus a DataFrame, produce a `SufficiencyReport`: does the data cover
  every required IO pair, enough non-null rows, aligned time index?
  "Zero-degree-of-freedom regression" (project term, define it in docs): the
  registered IO pairs exactly determine the regression problem — no unmapped
  inputs, no free parameters after fit. Multiple valid IO pairs per unit are
  supported; validation checks each.
- `regression/base.py`: `Regressor` Protocol — `fit(X, y) -> FitResult`,
  `to_surrogate_spec() -> SurrogateSpec`. `constant.py` (mean energy intensity),
  `linear.py` (sklearn, behind the `[parameterize]` extra). NN/ARIMA/multiconvex
  are post-v0 implementations of the same protocol.
- `apply.py` (**the FlexParameterize → FlexOps direction**): `apply_to_model(
  model, data, tagmap)` — for each fitted unit, **fix regressed parameters in
  place** and, where the fit produces a richer relationship than the unit's
  default constant intensity, **swap the energy-relationship constraint in
  place**: deactivate the unit's default equality constraint and construct a new
  one from the fitted `SurrogateSpec`, reusing the same registered IO variables
  (ports and arcs are untouched — there is no block to replace and nothing to
  reconnect). FlexOps provides the constraint-swap hook on `OpsBlockData`;
  FlexParameterize drives it. This is why the coupling is two-way even though
  the import is one-way.
- `emit.py` (**the serializable direction**): fitted result + model identity →
  `ModelConfig`/`UnitConfig` (with `provenance`: fit metrics, data window,
  package versions). **The round-trip invariant**: an emitted config rebuilds a
  FlexOps model whose behavior matches the fit — and matches what `apply.py`
  would have produced in place.
- Data source adapters (Aquarium, OSIsoft, WaterTAP/ASPEN exports) are out of
  scope: anything that becomes a `pd.DataFrame` is accepted.

## 6. flexschedule

- `horizon.py`: `RollingHorizon(time_block, window, overlap)` — yields window
  index ranges; `StateCarryOver` maps end-of-window variable values into the
  next window's initial-state Params (the ones units registered with TimeBlock).
- `sequences.py`: `SolveSequence` — ordered steps such as
  `RelaxIntegers → SolveMIP(warm_start=True) → FixIntegers → SolveNLP`, executed
  via `flexcore.solvers`, with per-step failure policy (abort / fall back /
  accept relaxed). This is where relaxation strategies live explicitly (see R5).
- `setpoints.py`: walk a solved model, extract registered IO/actuator variables
  into a tidy long-format DataFrame
  (`timestamp, plant, unit, variable, value, units`) for downstream control.
- `smoothing.py`: post-processing before set points go to a plant — moving
  average, minimum-hold snapping; must preserve energy/volume totals within
  tolerance.
- **Reporting rule (decision R9).** The raw solver **objective value is never the
  user-facing output** — it is a relaxed, possibly scalarized internal quantity
  (§2.4). The reported electricity cost is always
  `FlexCosting.report_cost(model)` — EECO evaluated post-solve on the realized
  aggregate-power numpy array. Set-point extraction returns physical
  trajectories + this EECO cost; the objective is surfaced only when a caller
  explicitly asks for it (a debug/advanced flag). The same post-hoc EECO
  evaluation is the comparison metric for multi-stage / iterative schemes that
  don't need duals.
- Forecasting is **not** built here: the interface is "forecast externally,
  then fix parameters" (adapter in the post-v0 backlog).

## 7. Decision log (summary)

| ID | Decision | One-line rationale |
|---|---|---|
| R1 | OpsBlock inherits IDAES `UnitModelBlockData`, no ControlVolumes | config/ports/costing machinery for free; ControlVolumes are wrong for scheduling surrogates |
| R2 | Discrete TimeBlock + `dynamic=False` flowsheets; hand-written difference equations | MIP logic needs integer index arithmetic; DAE fights binaries and rolling horizons |
| R3 | Pydantic v2 is the schema authority; the whole model+run builds from one version-controlled config (canonical format YAML, JSON accepted); no essential config lives only in code | config-driven-everything requirement; files are both human-tracked and written programmatically by external modules |
| R4 | FlexCosting subclasses IDAES costing and delegates tariff cost to EECO — convex-relaxed cost in the objective + post-solve EECO evaluation for reporting; owns CapEx, modes, clear naming; DR is containers-only in v0 | reuse the lab's maintained cost engine; the objective is a relaxed proxy so the reported cost must be re-evaluated post-hoc |
| R5 | Solver facade never transforms models silently | relaxed-MIP schedules sent to a real plant are a correctness hazard; explicit SolveSequence instead |
| R6 | Unit models organized by IO topology (SISO/SIDO/DIDO bases) then specialized physically; `Electrolyzer`→`Separator` etc. | one place for ports/mass-balance per topology; physical zoo (RO skid, combustor, exchangers) reuses it; Tank = SISO with logic disabled |
| R7 | `NetworkBlock` composes plants; `PlantBlock` composes units | explicit two-level composition instead of overloading PlantBlock to nest into itself |
| R8 | Customizable unit commitment: `status` base, optional startup/shutdown/dwell/delays/conditional; parallel-train degeneracy detection is model-level, not per-unit | a unit can't see its siblings, so symmetry-breaking lives above the unit; everything else is opt-in per unit |
| R9 | Never report the solver objective as the user-facing result; report EECO's post-solve cost; battery/all units accept external (DERMS) dispatch commands | objective is relaxed/scalarized; third-party-controlled assets need their dispatch fixed from outside |
| R10 | FlexParameterize↔FlexOps coupling is two-way at runtime (FlexOps builds containers; FlexParameterize fixes params and swaps a unit's energy-relationship constraint in place) though the import stays one-way | matches how parameterization actually works; keeps the layering + serialized-config split seam intact |
| R11 | Every unit defaults to a constant energy-intensity relationship; there is no separate `LinearRegressionModel` unit class — FlexParameterize upgrades a unit's relationship via an in-place constraint swap, reusing the same registered IO variables | keeps the unit library small and one generic class (`ConstantEnergyIntensityModel`) covers anything without a bespoke physical topology; regression sophistication is FlexParameterize's concern, not FlexOps' |
| R12 | No compat/isolation layer or import-linter whitelist for `idaes`/`pyomo`/`eeco`; import them directly, pin exact versions in `pyproject.toml`, and have maintainers bump them manually (~quarterly) after tests pass | a re-export whitelist guards only cheap import-path drift, not semantic drift; standard pinning is simpler and the sibling project (WaterTAP) imports `idaes` directly. Collecting `eeco` calls in `costing/opex.py` stays a convention, not an enforced boundary |
