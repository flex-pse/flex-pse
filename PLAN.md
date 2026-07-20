# flex-pse Development Plan

**flex-pse** is an open-source Python platform for industrial energy-flexibility
optimization — think "WaterTAP for flexible operations scheduling." It lets an
engineer model a water/wastewater/desalination (or other industrial) facility as a
time-discretized optimization problem, parameterize that model from plant data, and
solve rolling-horizon scheduling problems against real electricity tariffs and
demand-response signals.

This document is the master roadmap. It is written to be executed **milestone by
milestone by a junior engineer or a coding agent** (Sonnet-class or similar). Every
milestone is a self-contained work order in [`plan/milestones/`](plan/milestones/).

---

## 1. How to use this plan

If you are the implementer (human or agent), for each work session:

1. Read [`plan/00_conventions.md`](plan/00_conventions.md) — non-negotiable rules
   for style, naming, testing, and documentation. Re-read it at the start of every
   milestone until it is second nature.
2. Read the **one** milestone file you are executing (e.g.
   [`plan/milestones/M02_timeblock.md`](plan/milestones/M02_timeblock.md)).
   Milestones are strictly ordered by dependency; never start `M(n)` before
   `M(n-1)` is merged, except where a milestone is explicitly marked parallelizable.
3. Consult [`plan/01_architecture.md`](plan/01_architecture.md) whenever a design
   question arises. If the answer is not there, prefer the smallest decision that
   does not contradict it, and record the decision in the milestone PR description.
4. Work **test-first**: the milestone's Tests section is a behavioral spec —
   write those tests, watch them fail for the right reason, then implement until
   they pass (full loop in `plan/02_testing_and_ci.md` §1a). Run the complete
   local suite (`pytest -q`, all tiers) before every push.
5. Build exactly what the milestone specifies. Do not build ahead. Do not refactor
   earlier milestones except where the current milestone says to.
6. A milestone is done when every item in its **Definition of Done** checklist
   passes, including tests and docs. "Code complete, tests later" is never done —
   and neither is "tests written after the code."
7. .gitkeep files are added to empty directories. Remove .gitkeep files when a file is placed in that directory and add it to the associated commit.

One milestone ≈ one pull request ≈ 1–3 junior-engineer days ≈ one focused agent session.

### Reading order for a brand-new contributor

1. This file, top to bottom.
2. `plan/00_conventions.md`
3. `plan/01_architecture.md`
4. `plan/02_testing_and_ci.md` and `plan/03_documentation.md`
5. The milestone you are assigned.

---

## 2. What we are building

Three tools in one repository (monorepo now, engineered to split later):

| Tool | Import package | What it does |
|---|---|---|
| **FlexOps** | `flexops` | Modeling library: time-discretized unit models organized by IO topology (SISO/SIDO/DIDO → pump, tank, separator, exchanger, RO skid, combustor, battery, surrogates), LP/NLP/MIP with configurable relaxations, customizable unit commitment (status/dwell/delays/conditional), plant + network composition, external (DERMS) dispatch, and EECO-backed costing. Everything buildable from one config file. |
| **FlexParameterize** | `flexparameterize` | Data-driven parameterization: ingests tabular plant data, maps historian tags to model variables, validates data sufficiency, regresses registered parameters, and emits a config file that rebuilds the parameterized FlexOps model. |
| **FlexSchedule** | `flexschedule` | Scheduling engine: rolling-horizon driver, relaxation/warm-start solve sequences, set-point extraction and smoothing for plant control. |

Plus a shared substrate, `flexcore`, holding everything the three tools have in
common: the exception hierarchy, the solver facade, and the versioned config
schema (the contract between FlexParameterize and FlexOps).
Tariffs, demand response (including DR constraints), and time-indexed operating
costs come from the external **EECO** package (`eeco` on PyPI); `flexops.costing`
wraps it — flex-pse does not build its own tariff/cost engine.

### Target user-facing API (the "API freeze" script)

This script — adapted from the architecture slides — must run top-to-bottom and
solve by the end of milestone M09. It is checked in as `examples/api_freeze.py`
and guarded by a component test; any change that breaks it is a breaking change.

```python
import pyomo.environ as pyo
from pyomo.environ import units as pyunits
import flexops as fo

m = pyo.ConcreteModel()
m.time_block = fo.TimeBlock(
    start_date="2025-01-01", end_date="2025-01-30", time_step=15 * pyunits.min
)
m.properties = fo.SimpleAqueousFlow(fixed_density=True)
m.costing = fo.FlexCosting(
    time_block=m.time_block,
    tariff_file="tariff.json",
    dr_event_file="dr_events.json",
)
m.svcw = fo.PlantBlock(time_block=m.time_block)
m.svcw.tank = fo.StorageTank(property_package=m.properties)
m.svcw.plant = fo.ConstantEnergyIntensityModel(
    property_package=m.properties,
    energy_intensity=0.5 * pyunits.kWh / pyunits.m**3,
    costing_package=m.costing,
)
m.svcw.tank_to_plant = pyo.Arc(
    source=m.svcw.tank.outlet, destination=m.svcw.plant.inlet
)
m.svcw.battery = fo.BatteryModel(
    capacity=1 * pyunits.kWh, costing_package=m.costing
)
m.costing.cost_process()
m.objective = pyo.Objective(expr=m.costing.aggregate_operating_cost)
```

(Names differ slightly from the original slide pseudo-code: ISO-8601 dates,
keyword configuration arguments, and an explicit `time_block=` on `PlantBlock`.
The reasons are recorded as decisions R2 in `plan/01_architecture.md`.)

### Design constraints (why the architecture looks the way it does)

- **Manage Pyomo/IDAES upstream churn by pinning, not isolating** (decision R12).
  `idaes.*`/`pyomo.*` are imported directly at point of use; `pyproject.toml`
  pins exact tested versions and maintainers bump them manually (~quarterly)
  after the suite passes. No compat layer, no upstream canary.
- **Testable in blocks, test-first, fully gated.** Development is test-driven:
  each milestone's tests are written before its implementation, and the full
  suite runs locally before every push. Every test carries exactly one tier
  marker (`unit` / `component` / `integration`), enforced at collection time —
  the tiers exist to keep the local TDD loop sub-second, not to defer anything:
  the repo is public, CI minutes are free, and **all tiers run and must pass on
  every PR before merge**. (WaterTAP let heavy tests pile into its fast lanes
  until CI became unusable; the collection hook guards the fast lane
  mechanically.)
- **Splittable monorepo.** A CI-enforced import DAG
  (`flexcore ← flexops ← {flexparameterize, flexschedule}`), colocated tests, and
  a serialized JSON config contract mean any package can move to its own repo
  later by moving a directory.
- **Auto-generated documentation.** Sphinx with a custom extension that builds
  each unit model at docs-build time and generates its Variables / Constraints /
  Degrees-of-Freedom tables from the model itself, so docs cannot drift from code.
- **Solver-agnostic.** Default open-source stack (HiGHS for LP/MILP, IPOPT for
  NLP), optional Gurobi/CPLEX/CBC, all behind a capability-detecting facade that
  errors loudly rather than silently transforming the model.

---

## 3. Milestone index

Legend: each milestone links to its work order. Effort is junior-engineer days.
Dependencies are strict unless marked ∥ (parallelizable with the previous one).

| # | Title | Effort | Depends on | Headline deliverable |
|---|---|---|---|---|
| [M00](plan/milestones/M00_repo_scaffold.md) | Repo scaffold & CI skeleton | 1 | — | conda env from `environment.yml` (installs the stack and the editable package via its `pip:` subsection) works; PR CI green; import-linter contracts active |
| [M01](plan/milestones/M01_compat_layer.md) | Exception hierarchy & dependency pinning | 0.5 | M00 | `FlexError` hierarchy; pinned idaes-pse/pyomo versions |
| [M02](plan/milestones/M02_timeblock.md) | TimeBlock | 2 | M01 | Discrete ≤1-month substrate, any resolution (15-min default); horizon builds < 1 s |
| [M03](plan/milestones/M03_properties_opsblock.md) | SimpleAqueousFlow + OpsBlock base | 2–3 | M02 | IO/parameter registration; `power_electrical`/`power_thermal`; external-dispatch + UC config hooks; config schema (`UnitConfig`…`ModelConfig`) |
| [M04](plan/milestones/M04_harness_pump_tank.md) | Test harness + SISO base + Pump + StorageTank | 2–3 | M03 | Public `UnitModelTestHarness`; `SISOBlock`; Pump; StorageTank (logic disabled) |
| [M05](plan/milestones/M05_solver_facade.md) | Solver abstraction | 2 | M00 ∥ | Model classifier + capability-matrix `get_solver()` |
| [M06](plan/milestones/M06_eeco_integration.md) | EECO integration (tariffs & costs) | 2 | M00 ∥ | `eeco` wired; tariff-signal helpers; in-objective cost + post-solve numpy evaluator; DR containers |
| [M07](plan/milestones/M07_flexcosting.md) | FlexCosting | 3 | M04, M05, M06 | First end-to-end result: tank+pump shifts load off-peak; `report_cost` (EECO post-hoc) |
| [M08](plan/milestones/M08_battery_logic.md) | Battery (DERMS) + customizable unit commitment | 3 | M07 | SOC + external dispatch; status/startup/shutdown/dwell/delays/conditional; model-level degeneracy detection |
| [M09](plan/milestones/M09_plantblock_api_freeze.md) | Network/Plant + topology bases + surrogates + config build + API freeze | 3 | M08 | `NetworkBlock`/`PlantBlock`; SIDO/DIDO + Separator/Exchanger; `build_model(config)`; `api_freeze.py` runs |
| [M10](plan/milestones/M10_parameterize_core.md) | FlexParameterize core (2-way) | 3 | M09 | Tag aliasing; sufficiency; constant-EI round-trip; `apply_to_model` fixes params + replaces blocks in place |
| [M11](plan/milestones/M11_regressor_protocol.md) | Regressor protocol + linear regression | 2 | M10 | Pluggable regressors; fit provenance in emitted configs |
| [M12](plan/milestones/M12_rolling_horizon.md) | FlexSchedule: rolling horizon + solve sequences | 3 | M09 | 7-day windowed solve within 2 % of monolithic |
| [M13](plan/milestones/M13_setpoints_smoothing.md) | Set-point extraction + smoothing + cost reporting | 1–2 | M12 | Tidy set-point schema; totals-preserving smoothing; reports EECO cost (never the objective) |
| [M14](plan/milestones/M14_docs_notebooks.md) | Docs completion + example notebooks | 2–3 | M13 | `sphinx-build -W` clean; auto unit-model tables; 3 executed notebooks |
| [M15](plan/milestones/M15_release.md) | Hardening + 0.1.0 release | 2 | M14 | TestPyPI install runs the API-freeze example |
| [M16](plan/milestones/M16_design_multiperiod.md) | Design-mode multi-period wrapper | 3 | M09, M07 | `flexops.design`: merge N representative ≤1-month models; equality-link sizing vars |

Dependency sketch:

```
M00 ─ M01 ─ M02 ─ M03 ─ M04 ─┐
  ├─ M05 ────────────────────┼─ M07 ─ M08 ─ M09 ─┬─ M10 ─ M11
  └─ M06 ────────────────────┘                   ├─ M12 ─ M13 ─ M14 ─ M15
                                                  └─ M16 (design wrapper)
```

---

## 4. Post-v0 backlog (explicitly NOT in scope for 0.1.0)

Do not build any of these during M00–M16, even partially, unless a milestone says
so. They are recorded here so design choices keep the door open.

- **Parallel-train replication + delay-chain templates** — the customizable unit
  commitment in M08 (status/startup/shutdown/dwell/delays/conditional) and its
  model-level degeneracy detection are the primitives; the higher-level helpers
  that *replicate* identical parallel trains/skids and assemble the
  chemical-stabilization startup-delay chains of Rao et al. 2024 (*Optimizing
  desalination operations for energy flexibility*) are post-v0.
- **Demand response (DR)** — v0 ships **containers only** (config slots + no-op
  hooks in `FlexCosting`, per architecture §2.4/§3.6). The working DR
  formulation (events, curtailment constraints, incentives/capacity payments) is
  post-v0.
- **Scenario / sweep tool** — hundreds of scenarios with per-scenario constraint
  addition/removal and objective swapping (cost-min vs. production-max), discrete
  stochastic events (process upsets). The WaterTAP parameter-sweep tool is too
  rigid to reuse; this will be purpose-built.
- **Case studies** — (1) behind-the-meter battery sizing with third-party
  dispatch injected via the DERMS external-dispatch hook (M08), size as the only
  free design variable, sized across representative periods with the M16 design
  wrapper; (2) flexibility baseline-vs-optimal with logic-encoded heuristics
  ("shut down one parallel train during peak hours") built on the M08 UC layer.
- **DR capacity estimation** — production-maximizing solves as a function of event
  duration and preparation time, to inform reliable DR bids.
- **External forecaster interface** — "forecast, then fix parameters" adapter so
  any forecasting tool (ARIMA, NN, vendor) can feed FlexOps parameters; a specific
  wastewater-inflow forecaster pairs with it. FlexOps stays forecaster-agnostic.
- **Knowledge-graph / WaTr / PyPES connector** — map each FlexOps unit-model class
  to a Water-ontology class; build FlexOps flowsheets from Pipes/WaTr knowledge
  graphs by pattern/string matching. Depends on the (external) Pipes→TTL
  translator maturing.
- **Validate/Evaluate** — run detailed ODE/PDE models in parallel with the
  scheduler and flag when optimization outputs leave the feasible envelope.
  Requires an ODE model library that does not exist yet.
- **Repo split** — when a package reaches roughly 20 modules / 10k lines *and* has
  external users of its own, promote it to its own repository. `flexcore.config`'s
  versioned schema (YAML canonical, pydantic authority, exported JSON Schema) is
  the seam; the import-linter DAG guarantees the move is mechanical. Keep a
  single shared environment/dependency file across repos.

---

## 5. Where decisions live

| Question | Document |
|---|---|
| Naming, style, docstrings, PR checklist, agent rules | [`plan/00_conventions.md`](plan/00_conventions.md) |
| Package layout, block architecture, ADRs R1–R5, energy nomenclature, config schema, solver facade | [`plan/01_architecture.md`](plan/01_architecture.md) |
| Test tiers, harness, CI workflows, coverage | [`plan/02_testing_and_ci.md`](plan/02_testing_and_ci.md) |
| Sphinx setup, auto-generated tables, notebooks | [`plan/03_documentation.md`](plan/03_documentation.md) |
| What to build this session | [`plan/milestones/`](plan/milestones/) |
