# M09 — Network/Plant + topology bases + surrogates + config build + API freeze

**Effort:** 2–3 days · **Depends on:** M08 · **Parallelizable:** no

## Goal
Complete the flexops composition layer and the config-driven build path, then
freeze the public API. Deliver:
- **`PlantBlock`** (collection of units) and the new **`NetworkBlock`**
  (composition of plants with inter-plant arcs) — both thin `dynamic=False`
  flowsheets with recursive aggregation (§3.3, R7).
- The **IO-topology base classes** `SIDOBlock` (1→2) and `DIDOBlock` (2→2)
  (`SISOBlock` already exists from M04), and the **physical zoo** built on them:
  `Separator`, `Exchanger`, and the derived electrolysis / RO / combustor units
  (§3.4, R6). This is where the old `Electrolyzer` becomes `Separator`.
- The generic surrogate unit **`ConstantEnergyIntensityModel`** — the default
  building block for anything without a bespoke physical topology (e.g. a whole
  plant modeled as a single surrogate). There is no separate
  `LinearRegressionModel` class (R11): every unit defaults to constant energy
  intensity; FlexParameterize (M10/M11) is what later upgrades that relationship
  via an in-place constraint swap.
- **Config-driven build**: `flexops.build_model(config)` constructing the whole
  Pyomo model (TimeBlock + properties + costing + network/plant/unit tree + arcs)
  from one validated config, and a real `OpsBlock.from_config` (§2.3, R3).

Then freeze the public API: check in `examples/api_freeze.py` — the verbatim
script from PLAN.md §2 — with a component test that runs and solves it, plus a
parallel `examples/api_freeze_config.yaml` that `build_model` turns into an
equivalent, equally-solving model. From this milestone on, breaking that script is
a breaking change.

## Read first
- `PLAN.md` §2 — the API-freeze script (you will copy it verbatim; it uses
  `PlantBlock`/`StorageTank`/`ConstantEnergyIntensityModel`/`BatteryModel` — all
  valid, do NOT change it).
- `plan/01_architecture.md` §3.3 (R7: `PlantBlock` composes units, `NetworkBlock`
  composes plants; both `dynamic=False`; explicit `time_block=`; recursive
  aggregation; auto-discovery as convenience)
- `plan/01_architecture.md` §3.4 (R6: IO-topology bases `SISOBlock`/`SIDOBlock`/
  `DIDOBlock`; physical zoo `Separator`/`Exchanger`/`ElectrolysisSeparator`/
  `ElectrolysisExchanger`/`ReverseOsmosisSkid`/`Combustor`;
  `ConstantEnergyIntensityModel`; the "helper functions attach the
  flow↔energy relationship" paragraph; `Electrolyzer`→`Separator`)
- `plan/01_architecture.md` §2.3 (R3: config-driven-everything;
  `ModelConfig`/`PlantConfig`/`NetworkConfig`/`UnitConfig`/`SurrogateSpec`/
  `IOVariableSpec`, `load_model_config`, YAML canonical) and §3.2
  (`OpsBlock.from_config`, `flexops.build_model(model_config)`)
- `plan/00_conventions.md` §1 (repo layout — `core/network_block.py`,
  `core/build.py`, `unit_models/base/{sido,dido}.py`, the physical-zoo modules),
  §4 (pydantic at the boundary, ConfigDict at runtime)
- `plan/01_architecture.md` §7 — decision log **R6**, **R7**, and **R11** (every
  unit defaults to constant energy intensity; no separate surrogate-regression
  unit class).
- `plan/02_testing_and_ci.md` §1–2 (tier markers, harness)

## Files to create or modify
- `src/flexops/core/plant_block.py` — `PlantBlock`, recursive energy aggregation over child **units**.
- `src/flexops/core/network_block.py` — `NetworkBlock`, composition of **plants** with inter-plant arcs + recursive aggregation over child plants.
- `src/flexops/core/build.py` — `build_model(config: ModelConfig) -> ConcreteModel`.
- `src/flexops/unit_models/base/sido.py` — `SIDOBlock` (1 in → 2 out) topology base.
- `src/flexops/unit_models/base/dido.py` — `DIDOBlock` (2 in → 2 out) topology base.
- `src/flexops/unit_models/separator.py` — `Separator(SIDOBlock)` (replaces the old `Electrolyzer`).
- `src/flexops/unit_models/exchanger.py` — `Exchanger(DIDOBlock)`.
- `src/flexops/unit_models/electrolysis.py` — `ElectrolysisSeparator(Separator)` (v0) and `ElectrolysisExchanger(Exchanger)` (stretch).
- `src/flexops/unit_models/ro_skid.py` — `ReverseOsmosisSkid(Separator)` (v0).
- `src/flexops/unit_models/combustor.py` — `Combustor(Separator)` (stretch).
- `src/flexops/unit_models/constant_intensity.py` — `ConstantEnergyIntensityModel`.
- `src/flexops/core/ops_block.py` — implement `from_config` for real (stub since M03); optional TimeBlock auto-discovery.
- `src/flexops/__init__.py` — export `PlantBlock`, `NetworkBlock`, `SIDOBlock`, `DIDOBlock`, `Separator`, `Exchanger`, the v0 derived units, `ConstantEnergyIntensityModel`, and `build_model` (the script and config path use `fo.<Name>` / `fo.build_model`).
- `examples/api_freeze.py` — VERBATIM copy of the PLAN.md §2 script.
- `examples/api_freeze_config.yaml` — the config-driven twin of the frozen script.
- `examples/data/tariff.json`, `examples/data/dr_events.json` — small fixtures the script loads.
- `src/flexops/tests/core/test_plant_block.py`, `src/flexops/tests/core/test_network_block.py`, `src/flexops/tests/core/test_from_config.py`, `src/flexops/tests/core/test_build_model.py`, `src/flexops/tests/unit_models/test_base_topology.py`, `src/flexops/tests/unit_models/test_separator.py`, `src/flexops/tests/unit_models/test_exchanger.py`, `src/flexops/tests/unit_models/test_constant_intensity.py`, `src/flexops/tests/test_api_freeze.py`.
- Docs: `docs/how_to/build_a_plant.md`, `docs/explanation/time_and_dynamics.md`, reference pages.

## Specification

### PlantBlock (`src/flexops/core/plant_block.py`)
Thin subclass of `FlowsheetBlockData` (imported via `flexcore.compat.idaes`),
declared with `declare_process_block_class("PlantBlock", ...)` per R2/R7 — a
**collection of unit blocks** (a facility):
- CONFIG: `time_block` (the `TimeBlock` instance; `description=` set). Explicit
  `PlantBlock(time_block=m.time_block)` is the primary API.
- `build()` forces `dynamic=False` and injects
  `time_set = time_block.time_points` into the flowsheet config before calling
  `super().build()` (exact IDAES mechanism is implementer's choice — smallest
  variant that makes `unit.flowsheet()` resolve time correctly; never Pyomo.DAE).
- Holds arcs between its units. Aggregation **Expressions** (not Vars):
  `total_electrical_work[t]` and `total_thermal_work[t]`, each summing the
  corresponding registered energy Vars (via `register_energy`, architecture
  §3.2/§4) over child units. Because units are added after the plant exists, build
  these lazily — smallest choice: construct them in a method
  `plant._build_aggregates()` called idempotently by FlexCosting and by users, OR
  as Expressions whose rule iterates children when first constructed after units
  are added (implementer's choice — document which; the api_freeze script adds
  units after both plant and costing exist, so whatever you pick must survive that
  ordering — FlexCosting's `cost_process()` deferral from M07 is the precedent).
- **`PlantBlock` composes units, NOT plants** (R7): a plant containing plants is
  now a `NetworkBlock`. Do not overload `PlantBlock` to nest into itself.
- **`replace_unit` wrapper** (R10, §5): expose `PlantBlock.replace_unit(name,
  new_block)` as a thin method that calls the hierarchy-agnostic core helper
  `flexops.core.replace_unit(self, name, new_block)` introduced in M03 (delete
  old child, attach new under `name`, re-point arcs; `FlexConfigError` on missing
  name / port mismatch). This is the hook FlexParameterize's in-place path (M10)
  drives. `NetworkBlock` exposes the same wrapper for replacing a child plant.
  Add a `unit`-tier test that `plant.replace_unit(...)` swaps a child unit and
  rewires its arc.

### NetworkBlock (`src/flexops/core/network_block.py`)
The **composition of plants** — a portfolio / campus / multi-facility system
(§3.3, R7). Same thin `FlowsheetBlockData` / `dynamic=False` / injected
`time_set` construction as `PlantBlock`:
- CONFIG: `time_block` (explicit; `description=` set), same primary API.
- Holds **child `PlantBlock`s** and **inter-plant arcs** between them.
- Recursive aggregation: `total_electrical_work[t]` / `total_thermal_work[t]` are
  Expressions summing each child **plant's** own totals (which in turn sum their
  units), so `NetworkBlock` total = Σ child `PlantBlock` totals = Σ their units'
  `electrical_work`/`thermal_work` (the composition invariant, §3.3). Use the same
  lazy/idempotent aggregation pattern as `PlantBlock`.
- Unit/plant discovery must **not double-count**: recurse into child plants OR
  their units, never both (Pitfall 7). A `NetworkBlock` sums plant totals; it does
  not re-walk each plant's units directly.

### IO-topology base classes (`src/flexops/unit_models/base/`)
Extend the M04 topology-base pattern (`base/siso.py` — `SISOBlock` — already
exists). Each subclasses `OpsBlockData`, owns port construction, per-stream mass
balance, and energy-registration wiring (§3.4):
- `base/sido.py` — **`SIDOBlock`** (1 in → 2 out): one inlet port, two outlet
  ports on `SimpleAqueousFlow`; split mass balance `flow_in[t] == flow_out_a[t] +
  flow_out_b[t]` (or the smallest correct per-stream form; document it). Registers
  the streams as IO; energy-registration wiring provided for subclasses.
- `base/dido.py` — **`DIDOBlock`** (2 in → 2 out): two inlet, two outlet ports;
  two coupled per-stream mass balances (implementer's choice on the exact coupling
  form; document it). Registers streams as IO; energy wiring for subclasses.
All Vars/Constraints carry `doc=`. Follow the `SISOBlock` conventions from M04
exactly (naming, references to state `flow_vol`, no ControlVolumes — R1).

### Physical zoo (built on the topology bases, §3.4, R6)
The general pattern: a unit defines its topology (ports + mass balance from the
base) and its energy draw; **helper functions attach the flow↔energy
relationship** (constant intensity in v0). Same base topology, controllable
functional form.

**v0 (must build this session):** the bases + `Separator` + `Exchanger` + at
least `ReverseOsmosisSkid` + one electrolysis variant:
- `Separator(SIDOBlock)` — one feed split into two product streams. **This
  replaces the old `Electrolyzer` name** (R6). Constant-intensity energy relation
  wiring `electrical_work[t]`.
- `Exchanger(DIDOBlock)` — two inlet / two outlet streams exchanging mass/energy.
- `ReverseOsmosisSkid(Separator)` — RO skid: feed → permeate + concentrate; thin
  subclass fixing the split semantics + energy relation.
- `ElectrolysisSeparator(Separator)` — electrolysis modeled as a separation;
  exercises **`thermal_work`** in addition to `electrical_work` (register both
  energy kinds). Thin subclass.

**Stretch (this session if time; otherwise MUST land before M14 docs — they are
in the v0 library table §3.4; note any deferral in the PR):**
- `ElectrolysisExchanger(Exchanger)` — electrolysis with two coupled streams.
- `Combustor(Separator)` — combustion as a separation of products.

Derived units may be **thin subclasses** of `Separator`/`Exchanger`
(implementer's choice on how much each specializes; keep them small and
reasonable for one session). Each shipped unit gets a `UnitModelTestHarness`
subclass (below).

### ConstantEnergyIntensityModel (`src/flexops/unit_models/constant_intensity.py`)
Generic "energy factor × flow" unit (library table §3.4), generalizing the M04
Pump pattern. This is FlexOps' one generic building block for anything without a
bespoke physical topology (R11) — the api_freeze script uses it directly as
`m.svcw.plant`, standing in for a whole treatment plant:
- CONFIG: `property_package`, `energy_intensity` (kWh/m³, with units), optional
  `costing_package`; inherited OpsBlock flags.
- Inlet/outlet ports via the property package; pass-through mass balance
  `flow_vol_out[t] == flow_vol_in[t]` (or shared state — copy Pump's pattern).
- `energy_intensity` — Var, fixed at config value, registered via
  `register_process_parameter(..., regressable=True)` (this is what M10 regresses).
- `electrical_work[t] == energy_intensity * flow_vol[t]`, energy registered
  electrical; `flow_vol` registered as IO input, `electrical_work` as IO output.
  Name this Constraint discoverably (e.g. `electrical_work_relation`) — R11's
  in-place constraint swap (§5, M10) deactivates exactly this Constraint and
  attaches a new one built from a fitted `SurrogateSpec` when FlexParameterize
  upgrades the relationship; document the name as the swap contract.
- Pump relationship: keep `Pump` an independent class in v0 rather than
  retrofitting it to subclass this **(implementer's choice — justify in the PR;
  rationale: conventions §9 forbids refactoring earlier milestones unless told,
  and the two may diverge when Pump grows pressure terms post-v0)**.
- There is no separate surrogate/regression unit class (R11): `json_config`/
  `SurrogateSpec`-driven construction is FlexParameterize's concern (M10/M11),
  not M09's — `ConstantEnergyIntensityModel` only ever builds the constant-intensity
  relationship above.

### OpsBlockData.from_config (in `src/flexops/core/ops_block.py`)
Real implementation replacing the M03 stub:
```python
@classmethod
def from_config(cls, cfg: UnitConfig, **kwargs):
```
Validates `cfg` (accept a path/dict by round-tripping through the pydantic
schema — never pass raw dicts onward, conventions §4), resolves the unit-model
class from the config's unit-model class name when called on the base class,
merges `cfg` construction options with `kwargs`, and returns the constructible
block (the object you assign onto a model — implementer's choice on the exact
IDAES construction mechanics; requirement: `m.u = SomeModel.from_config(cfg)`
followed by normal use works, and invalid configs raise pydantic
`ValidationError` whose message names the offending field path).
If `cfg` carries a `SurrogateSpec` whose `functional_form` is not
`constant_intensity` (M10/M11's regressed forms), `from_config` builds the unit
normally and then applies the **same in-place constraint-swap helper**
FlexParameterize's `apply_to_model` uses (R11) — one swap implementation, two
callers (construction-time for config-driven rebuild, runtime for
`apply_to_model`). This is what lets `ConstantEnergyIntensityModel.from_config`
serve as the rebuild target for any fitted `SurrogateSpec`, not just
`constant_intensity`.

### build_model (`src/flexops/core/build.py`) — config-driven everything (§2.3, R3)
```python
def build_model(config: ModelConfig) -> ConcreteModel:
```
Constructs the **entire** model from one validated config — no in-code
configuration required (R3). Given a `ModelConfig` (a path/dict is round-tripped
through `load_model_config`/pydantic first — never a raw dict, conventions §4):
1. build the `TimeBlock` from `TimeConfig` (§3.1);
2. build the property package(s) and the `FlexCosting` block from `CostingConfig`
   (§3.6, M06/M07);
3. build the composition tree from the config's `NetworkConfig` **or**
   `PlantConfig` (§2.3): a `NetworkBlock` of `PlantBlock`s, or a single
   `PlantBlock`, each populated with units via `OpsBlock.from_config` on their
   `UnitConfig`s;
4. construct the arcs (intra-plant and inter-plant) declared in the config;
5. apply any per-unit `unit_commitment` (§3.5) and `external_dispatch` (§3.2)
   declared in the config;
6. finalize costing (`cost_process()` deferral, M07) and return the
   `ConcreteModel`.
Requirement: a config that mirrors the imperative api_freeze script yields an
**equivalent** model (same components, same solved objective). Bad config →
`pydantic.ValidationError` (or `FlexConfigError` wrapping it) whose message names
the offending field path. `build_model` is the single config-driven entry point;
`from_config` is the per-unit primitive it uses.

### TimeBlock auto-discovery (documented convenience)
When `time_block=` is omitted on `PlantBlock`/`NetworkBlock` (and on units that
take one), search the root model for `TimeBlock` instances: exactly one → use it;
zero or several → `FlexConfigError` telling the user to pass `time_block=`
explicitly. Explicit argument is the primary, tested-first path (architecture §3.3).

### API freeze
- `examples/api_freeze.py`: the PLAN.md §2 script **verbatim** — every line,
  including the `Arc` — do not "improve" it. It uses
  `PlantBlock`/`StorageTank`/`ConstantEnergyIntensityModel`/`BatteryModel` (all
  still valid). If it cannot run verbatim, the library is wrong, not the script.
- `examples/api_freeze_config.yaml`: the **config-driven twin** — a valid
  `ModelConfig` (YAML canonical, §2.3/R3) describing the same model as the
  imperative script (one plant with the tank, the `ConstantEnergyIntensityModel`
  plant surrogate, the battery; the tank→plant arc; the tariff/DR/costing
  settings). `build_model(load_model_config("api_freeze_config.yaml"))` must
  yield a model that solves to the **same objective** as the imperative path.
- Fixtures under `examples/data/` (the imperative script uses bare filenames like
  `"tariff.json"`, so its test runs with cwd set to the fixture directory):
  `tariff.json` (small TOU tariff in EECO's format, per M06), `dr_events.json`
  (one DR event or an empty program — EECO's format). Keep them tiny and
  hand-readable.
- `src/flexops/tests/test_api_freeze.py` — see Tests. **This test is henceforth
  the breaking-change tripwire**: any PR that has to edit `api_freeze.py` to stay
  green is a breaking change and must say so (state this in a comment at the top
  of both files). The frozen imperative script is unchanged by this milestone.

## Pitfalls
1. **Editing the frozen script.** It is verbatim from PLAN.md §2. ISO dates,
   keyword args, `time_block=` on PlantBlock are already the corrected form (R2).
   It uses electrolyzer/separator **nowhere** — do not add units to it.
2. **`dynamic=True` or Pyomo.DAE sneaking in.** R2 forbids both, on `PlantBlock`
   AND `NetworkBlock`. `time_set` is the TimeBlock's ordered integer set; nothing else.
3. **Construction order.** The script builds costing before the plant and adds
   units after both. Aggregation Expressions built too early see zero units.
   Reuse the M07 `cost_process()` deferral pattern; add a permutation test.
4. **The Arc doesn't expand itself.** The test must apply
   `pyo.TransformationFactory("network.expand_arcs")` before solving. Do not bury
   the transformation inside library code in v0 (implementer's choice, document).
5. **30 days at 15 min = 2880 points.** Fine for build (M02 guarantees < 1 s) but
   keep the freeze-test solve within the component 30 s budget; if solve time is
   the problem, the fixture tariff/surrogate should stay trivially easy — do NOT
   shrink the script's dates.
6. **ValidationError laundering.** Don't catch pydantic errors and re-raise
   stringly — the field path is the tested contract for `from_config` and `build_model`.
7. **Double-counting composition.** `PlantBlock` recurses into its units;
   `NetworkBlock` recurses into its child plants' totals — never both levels at
   once. A `NetworkBlock` that also re-walks each plant's units double-counts.
8. **Overloading `PlantBlock` to nest.** A plant-of-plants is a `NetworkBlock`
   (R7). `PlantBlock` composes units only.
9. **`Electrolyzer` name.** There is no `Electrolyzer` class — it is `Separator`
   (and `ElectrolysisSeparator` for the electrolysis specialization), per R6.
10. **Config vs. imperative drift.** `api_freeze_config.yaml` must describe the
    *same* model; the equal-objective test is the contract. Keep the config
    minimal and hand-readable; validate it against the checked-in JSON Schema.

## Tests
`src/flexops/tests/core/test_plant_block.py`
- `test_aggregation_two_units` (`unit`) — plant with 2 units
  (`ConstantEnergyIntensityModel` + `BatteryModel`); fix all energy Vars to
  hand-picked values; evaluate `total_electrical_work[t]` /
  `total_thermal_work[t]` bodies with `pyo.value` and match the hand sum
  (`pytest.approx`, rel=1e-6). No solver.
- `test_time_block_autodiscovery` (`unit`) — omit `time_block=` with exactly one
  TimeBlock → works; with two → `FlexConfigError`.

`src/flexops/tests/core/test_network_block.py`
- `test_network_aggregates_over_plants` (`unit`) — a `NetworkBlock` with two child
  `PlantBlock`s (each with ≥1 energy-consuming unit) and an inter-plant arc; fix
  all energy Vars; assert `network.total_electrical_work[t]` equals Σ plant totals
  equals Σ unit `electrical_work[t]` (constraint-body/expression eval, no solver).
- `test_no_double_count_units` (`unit`) — assert the network sums plant totals and
  does not additionally re-walk each plant's units (a hand count of contributing
  terms matches the number of units exactly once).

`src/flexops/tests/unit_models/test_base_topology.py`
- `test_sido_mass_balance_bodies` (`unit`) — build a `SIDOBlock`; fix inlet/outlet
  flows; assert the split mass-balance body evaluates satisfied by hand. Ports
  exist (1 in, 2 out). No solver.
- `test_dido_mass_balance_bodies` (`unit`) — build a `DIDOBlock`; 2 in / 2 out
  ports exist; coupled mass-balance bodies checked by hand. No solver.

`src/flexops/tests/unit_models/test_separator.py`
- `TestSeparator(UnitModelTestHarness)` — harness subclass (build/units/
  registration/DoF `unit`; solve `component`).
- `TestReverseOsmosisSkid(UnitModelTestHarness)` and
  `TestElectrolysisSeparator(UnitModelTestHarness)` — harness subclasses; the
  electrolysis one asserts **both** `electrical_work` and `thermal_work` are
  registered (the `thermal_work` exerciser, §3.4).

`src/flexops/tests/unit_models/test_exchanger.py`
- `TestExchanger(UnitModelTestHarness)` — harness subclass on the `DIDOBlock`-based
  `Exchanger`.

`src/flexops/tests/core/test_from_config.py`
- `test_from_config_builds_unit` (`unit`) — valid unit config fixture →
  constructed unit with registered IO vars matching the spec.
- `test_from_config_bad_config_raises` (`unit`) — config with a wrong-typed /
  missing field → `pydantic.ValidationError`; assert the field path appears in
  `str(exc)`.

`src/flexops/tests/core/test_build_model.py`
- `test_build_model_matches_hand_built` (`component`, `needs_highs`) — a YAML
  `ModelConfig` fixture describing a small plant (tank + `ConstantEnergyIntensityModel`
  + battery, one arc); `build_model` constructs it; a hand-built equivalent is
  constructed imperatively; assert both build, both solve optimal, and their
  objectives match within tolerance (`pytest.approx`, rel=1e-6) — the
  config-driven path equals the imperative path.
- `test_build_model_bad_config_raises` (`unit`) — malformed config (wrong-typed /
  missing field, e.g. a bad `UnitConfig`) → `pydantic.ValidationError`; assert the
  offending field path appears in `str(exc)`.

`src/flexops/tests/unit_models/test_constant_intensity.py`
- `TestConstantEnergyIntensityModel(UnitModelTestHarness)` — standard ~30-line
  harness subclass (build/units/registration/DoF as `unit`; solve as `component`).
- `test_electrical_work_relation_constraint_is_named` (`unit`) — the
  constant-intensity equality Constraint has the documented discoverable name
  (§ ConstantEnergyIntensityModel spec) — this is the swap contract M10 relies on.

`src/flexops/tests/test_api_freeze.py`
- `test_api_freeze_runs_and_solves` (`component`, `needs_highs`) — copy
  `examples/data/*` into `tmp_path`, `monkeypatch.chdir(tmp_path)`, execute
  `examples/api_freeze.py` via `runpy.run_path`, apply arc expansion, solve via
  `flexcore.solvers.get_solver`, assert termination optimal and
  `aggregate_operating_cost` is finite. A top-of-file comment declares this the
  breaking-change tripwire.
- `test_api_freeze_config_matches_imperative` (`component`, `needs_highs`) — load
  `examples/api_freeze_config.yaml`, `build_model` it, apply arc expansion, solve;
  assert optimal and that its objective equals the imperative script's objective
  within tolerance (`pytest.approx`, rel=1e-6) — the config-driven path equals the
  imperative path.

## Documentation tasks
- Flesh out `docs/how_to/build_a_plant.md`: TimeBlock → properties → costing →
  PlantBlock → units → Arc → solve, mirroring api_freeze.py (link to it), plus a
  short section on the config-driven twin (`build_model(load_model_config(...))`)
  and where `NetworkBlock` fits for multi-plant systems.
- Reference pages (autosummary + `.. flexops-unit-tables::`) for the new units:
  `Separator`, `Exchanger`, `ReverseOsmosisSkid`, `ElectrolysisSeparator` (and the
  stretch units if built), `ConstantEnergyIntensityModel`; document the
  `SIDOBlock`/`DIDOBlock` topology bases; add `PlantBlock`, `NetworkBlock`, and
  `build_model` to `docs/reference/flexops/core.rst`.
- Complete `docs/explanation/time_and_dynamics.md` — the R2 rationale (discrete
  TimeBlock, `dynamic=False`, difference equations, why not DAE), including the
  slide-API correction note about explicit `time_block=` (architecture §3.3), and
  the R7 note that `NetworkBlock` composes plants while `PlantBlock` composes units.
- `docs/getting_started/ten_minutes.md` — update the walk-through to match the
  now-working script.
- CHANGELOG entry under "Unreleased".

## Definition of Done
- [ ] `examples/api_freeze.py` is byte-for-byte the PLAN.md §2 script and runs top-to-bottom.
- [ ] `test_api_freeze_runs_and_solves` passes with HiGHS (`component`, `needs_highs`).
- [ ] `examples/api_freeze_config.yaml` builds via `build_model` and solves to the same objective as the imperative script (config-driven == imperative).
- [ ] `PlantBlock` aggregates its units correctly (constraint-body test).
- [ ] `NetworkBlock` composes plants with inter-plant arcs; network total == Σ plant totals == Σ unit works; no double-counting.
- [ ] `SIDOBlock` and `DIDOBlock` topology bases build with correct ports and mass balances; harness-tested physical units on top.
- [ ] v0 zoo present: `Separator`, `Exchanger`, `ReverseOsmosisSkid`, and one electrolysis variant (`ElectrolysisSeparator`, exercising `thermal_work`); stretch units built or deferral noted in the PR.
- [ ] No `Electrolyzer` class exists — it is `Separator`/`ElectrolysisSeparator` (R6).
- [ ] `from_config` builds a real unit; bad config → `ValidationError` naming the field path.
- [ ] `build_model(config)` constructs TimeBlock + properties + costing + tree + arcs from one config; equals the hand-built model; bad config → `ValidationError` naming the field path.
- [ ] `ConstantEnergyIntensityModel`'s energy-relationship Constraint has the
      documented discoverable name (the swap contract M10 relies on); no
      separate `LinearRegressionModel`/surrogate-regression unit class exists (R11).
- [ ] Auto-discovery convenience works and errors actionably on 0/2+ TimeBlocks.
- [ ] Fixtures in `examples/data/` and `api_freeze_config.yaml` are valid against the checked-in JSON Schemas.
- [ ] All new tests carry exactly one tier marker; docs build with `sphinx-build -W`.
- [ ] Breaking-change tripwire comment present in both api_freeze files; CHANGELOG updated.
- [ ] plus the generic DoD in CLAUDE.md
