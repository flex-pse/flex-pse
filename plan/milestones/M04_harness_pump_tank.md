# M04 — Test harness + SISO base + Pump + StorageTank

**Effort:** 2–3 days · **Depends on:** M03 · **Parallelizable:** with M05, M06

## Goal

Ship the public `UnitModelTestHarness` that every subsequent unit-model milestone
(and every user writing a custom unit model) will subclass, build the first
IO-topology base class `SISOBlock` (single-inlet/single-outlet ports on
`SimpleAqueousFlow`, per-stream mass balance, energy-registration wiring —
architecture §3.4/R6), and then the first two real unit models on top of it:
`Pump(SISOBlock)` (constant energy intensity, LP) and `StorageTank(SISOBlock)`
(holdup difference equation, LP). The tank **inherits SISO but disables its
logic/unit-commitment constraints** — a tank has no on/off status; this is the
canonical "physical subclass turns off a base capability" example (architecture
§3.4, R6). By the end, `pytest -m "unit or component"` exercises the SISO base
plus two fully harness-tested unit models, and the docs reference tree for unit
models exists.

## Read first

- `plan/02_testing_and_ci.md` §1 (tiers), §2 (the harness — copy the class skeleton exactly), §5 (constraint-body point checks)
- `plan/01_architecture.md` §3.4 (IO-topology base classes — the `SISOBlock` row and the "physical subclass turns off a base capability" pattern; the Pump and StorageTank rows), §3.2 (OpsBlock registration API), §3.5 (logic/UC layer — `status` is the base capability a tank disables), §3.1 (TimeBlock: `time_points`, `dt`, `register_initial_state`), §4 (energy nomenclature), §3.7 (SimpleAqueousFlow), and R6 in the §7 decision log
- `plan/03_documentation.md` §2 (flexdoc — lands in M14, but it consumes `doc=` strings and the `flexops.testing` dummy helper you build now), §3 (unit-model autosummary template)
- `plan/00_conventions.md` §2 (naming), §7 (testing summary)

## Files to create or modify

- `src/flexops/testing/__init__.py` — exports `UnitModelTestHarness`, `dummy_time_block`
- `src/flexops/testing/harness.py` — the public, shipped harness
- `src/flexops/unit_models/base/__init__.py` — exports `SISOBlock`
- `src/flexops/unit_models/base/siso.py` — the `SISOBlock` IO-topology base
- `src/flexops/unit_models/__init__.py` — exports `Pump`, `StorageTank`
- `src/flexops/unit_models/pump.py` — `Pump(SISOBlock)` unit model
- `src/flexops/unit_models/storage_tank.py` — `StorageTank(SISOBlock)` unit model
- `src/flexops/tests/testing/test_harness.py` — harness self-checks
- `src/flexops/tests/unit_models/base/test_siso.py` — SISO base port + mass-balance unit test
- `src/flexops/tests/unit_models/test_pump.py` — `TestPump` harness subclass
- `src/flexops/tests/unit_models/test_storage_tank.py` — `TestStorageTank` + hand mass-balance test + logic-disabled test
- `src/flexops/tests/unit_models/test_pump_tank_component.py` — 24-step LP system solve
- `docs/reference/flexops/unit_models/index.rst`, `docs/reference/flexops/testing.rst`, `docs/_templates/autosummary/unit_model.rst` — reference docs (see Documentation tasks)

## Specification

### 1. `flexops/testing/harness.py`

Copy the public surface from `plan/02_testing_and_ci.md` §2 exactly:

```python
class UnitModelTestHarness:
    """Subclass per unit model; pytest collects the subclass."""

    expected_dof: int = 0
    expected_solution: dict[str, float] = {}   # component name -> value
    solver_tolerance: float = 1e-6

    def configure(self):
        """Build and return (model, unit). Override this."""
        raise NotImplementedError
```

Provided stages, each a real `test_*` method on the class so pytest collects
them on every subclass:

- `test_build` — `@pytest.mark.unit`. `configure()` returns `(model, unit)` without exception; declared ports exist.
- `test_units_consistent` — `@pytest.mark.unit`. `assert_units_consistent(unit)` (import via `flexcore.compat.idaes`; add to the compat whitelist if M01 did not).
- `test_io_registration` — `@pytest.mark.unit`. Every entry in the unit's `IORegistry` (`flexops/core/registration.py`, M03): the referenced component exists on the unit, carries `pyunits`, is indexed by `t`, and has a **non-empty `doc=` string** (the M14 flexdoc generator renders these).
- `test_energy_naming` — `@pytest.mark.unit`. `electrical_work` / `thermal_work` exist **iff** the unit called `register_energy` for that kind; no component named bare `power`/`energy`/`work` exists.
- `test_dof` — `@pytest.mark.unit`. Fix every registered `role="input"` IO variable at its current value, then assert `degrees_of_freedom(model) == self.expected_dof` (helper from `flexcore.compat.idaes`; whitelist it with a comment).
- `test_solve` — `@pytest.mark.component`. Obtain a solver via `from flexcore.solvers import get_solver` guarded: if the import fails or `get_solver` raises `FlexSolverError`/is a stub, `pytest.skip("flexcore.solvers.get_solver not available (M05 may land in parallel)")`. Solve with inputs fixed; assert optimal termination.
- `test_solution` — `@pytest.mark.component`. For each `name -> value` in `expected_solution`, resolve `name` dotted against the unit (e.g. `"electrical_work[0]"`) and assert `pyo.value(...) == pytest.approx(value, rel=self.solver_tolerance)`. Skip when `expected_solution` is empty. Resolution helper `_component_by_name(unit, name)` (implementer's choice on internals).

`test_solve`/`test_solution` must also carry `needs_highs` (these v0 models are
LP). Cache the `configure()` result per test method call — each stage builds its
own fresh model (stages must not share mutated state); do **not** cache across
tests (implementer's choice, but document it in the class docstring).

### 2. `flexops.testing.dummy_time_block(n=3)`

Helper reused by the M14 docs generator. Returns a `pyo.ConcreteModel` with
`m.time_block = TimeBlock(...)` spanning exactly `n` points at
`time_step=15 * pyunits.min` starting `"2025-01-01"`, and
`m.properties = SimpleAqueousFlow(fixed_density=True)` attached (start date and
return shape are implementer's choice, but keep the signature
`dummy_time_block(n: int = 3) -> pyo.ConcreteModel` exactly — M14 imports it).

### 3. `unit_models/base/siso.py` — `SISOBlock` (the first IO-topology base)

The first of the IO-topology base classes (architecture §3.4/R6). It owns port
construction, per-stream mass balance, and the energy-registration wiring so the
physical subclasses (`Pump`, `StorageTank`) only add the flow↔energy relationship
and any bounds. Subclass the OpsBlock base from M03 via the
`declare_process_block_class` pattern established in `flexops/core/ops_block.py`
(class name **`SISOBlock`**, data class `SISOBlockData(OpsBlockData)`).

The **SISO base contract** (what every SISO subclass inherits, so subclasses do
not re-declare it):

- Config (Pyomo ConfigDict, `description=` on each): `property_package` (IDAES
  standard), plus the base OpsBlock config flags from M03 (`unit_commitment`,
  `relaxation`, `allow_bypass`, `external_dispatch`, `costing_package`).
- One **`inlet`** `Port` and one **`outlet`** `Port`, each built from a
  `SimpleAqueousFlow` `StateBlock` indexed by `time_block.time_points` (no
  ControlVolumes — decision R1).
- **Per-stream mass balance**, written by hand, one constraint indexed by `t`:
  `outlet.flow_vol[t] == inlet.flow_vol[t]` (a `doc=` string on it). Subclasses
  that need a different balance (e.g. the tank's holdup) may override or replace
  it — document when they do.
- Convenience `flow_vol` handle: a `pyo.Reference` to the inlet state's
  `flow_vol` (implementer's choice of exact wiring; keep the name `flow_vol`).
- Energy-registration wiring inherited from `OpsBlockData` (§3.2) —
  `SISOBlock` itself registers **no** energy (it does not know its subclass's
  draw); subclasses call `register_energy` as needed.
- Logic/unit-commitment is available (the base `status` capability, §3.5) but
  **not built by `SISOBlock`** — it is opt-in via the `unit_commitment` config,
  and a subclass may disable it entirely (see `StorageTank`, §5).

### 4. `unit_models/pump.py` — `Pump(SISOBlock)`

Subclass **`SISOBlock`** via the `declare_process_block_class` pattern (class
name **`Pump`**, data class `PumpData(SISOBlockData)`). It inherits the inlet/
outlet ports, mass balance, and `flow_vol` handle from the SISO base, and adds
only the electrical-work relationship. Behavior (architecture §3.4):

- Config: inherits the SISO config; adds `energy_intensity` — default `0.5` in
  kWh/m³ (implementer's choice of default; document it).
- Inlet/outlet `Port`s and the mass balance `outlet flow_vol[t] == inlet
  flow_vol[t]` come **from `SISOBlock`** — do not re-declare them in `Pump`.
- `energy_intensity`: **mutable Param**, units `pyunits.kWh / pyunits.m**3`,
  registered `register_process_parameter(regressable=True)`.
- `electrical_work[t]` Var in kW (declared by the base class when the unit
  registers electrical energy), with constraint
  `electrical_work[t] == energy_intensity * flow_vol[t]`.
  **Unit algebra:** `flow_vol` is m³/hr, so kWh/m³ × m³/hr = kW — dimensionally
  exact with no conversion factor. Put this sentence in the constraint's
  `doc=` and the class docstring; `assert_units_consistent` is the referee.
- Registration: `register_io_variable(flow_vol, role="input")`,
  `register_io_variable(electrical_work, role="output")`,
  `register_energy(electrical_work, kind="electrical")`.

### 5. `unit_models/storage_tank.py` — `StorageTank(SISOBlock)`

Subclass **`SISOBlock`** (class name **`StorageTank`**, data class
`StorageTankData(SISOBlockData)`). It inherits the inlet/outlet ports and the
`flow_vol` handles from the SISO base; the tank's dynamics **replace** the SISO
pass-through mass balance with a holdup difference equation (inlet and outlet
flows differ — the tank stores the difference). **A tank has no on/off status,
so it disables the logic/unit-commitment layer** (architecture §3.4, §3.5, R6):
this is the canonical example of a physical subclass turning off a base
capability. Concretely, `StorageTankData` must ensure no `status[t]` Binary /
semicontinuous UC constraints are ever built for it — force the
`unit_commitment` config off (and/or skip the logic build) regardless of what a
caller passes, and document why in the class docstring.

- Config: inherits the SISO config; adds `min_volume` (default 0 m³),
  `max_volume` (required, m³), `initial_volume` (required, m³).
- `V[t]` Var, m³, `doc=` set, bounds `(min_volume, max_volume)` from config.
- Holdup difference equation, **indexed over `t = 0 .. N-2` only** (N =
  `len(time_block.time_points)`):
  `V[t+1] == V[t] + dt * (flow_in[t] - flow_out[t])`.
  Off-by-one: `V[N-1]` is defined by the constraint at `t = N-2`; there is no
  constraint indexed `N-1` (it would reference `V[N]`, which does not exist).
  Convert `dt` to hours inside the expression so m³/hr × hr = m³.
- `flow_in[t]` / `flow_out[t]`: References to the inlet / outlet state
  `flow_vol` on the ports **inherited from `SISOBlock`** (do not re-create the
  ports). The tank's holdup equation replaces SISO's pass-through mass balance
  (in a pump inlet == outlet; in a tank they differ by the stored volume).
- `initial_volume`: **mutable Param** (m³) + constraint `V[0] == initial_volume`.
  Register it **both** via `time_block.register_initial_state(param)` (rolling-
  horizon hook, architecture §3.1) **and**
  `register_process_parameter(regressable=False)`.
- `capacity`: Var (m³), a fixable design variable, **fixed by default** to
  `max_volume`, plus constraint `V[t] <= capacity` for all `t` (M07's design
  mode unfixes it). Do not register it as IO.
- Registration: `register_io_variable(flow_in, role="input")`,
  `register_io_variable(flow_out, role="input")`,
  `register_io_variable(V, role="output")`. **No** `register_energy` call — the
  tank draws nothing; this is the "iff" case for `test_energy_naming`.

## Pitfalls

1. **Holdup off-by-one.** Writing the constraint over all `t` raises KeyError at
   `V[N]` or silently skips — index the Constraint over
   `list(time_points)[:-1]` explicitly and unit-test the constraint count (N−1).
2. **Unit algebra by luck.** Do not multiply by 3600 or 1/60 anywhere in Pump;
   with kWh/m³ and m³/hr the equation is already in kW. In the tank, use
   `pyunits.convert(dt, to_units=pyunits.hr)` — `dt` is 15 min by default.
3. **Missing `doc=` strings.** `test_io_registration` fails on registered vars
   without docs; write them at declaration time, not as a fix-up pass.
4. **Hard dependency on M05.** The harness must import `flexcore.solvers`
   lazily inside `test_solve` and skip on failure — M05 may not be merged yet.
5. **Harness collected as a test.** Pytest must not collect the base class
   itself (its `configure` raises). Name test-file subclasses `Test*`; the base
   name `UnitModelTestHarness` does not match pytest's collection pattern —
   keep it that way and add a self-check test.
6. **Mutable class attribute `expected_solution`.** It is a class-level
   declaration copied from the spec — fine — but never mutate it in a stage.
7. **Forgetting tier markers on harness stages.** The root conftest (M00) fails
   collection on unmarked tests; mark the methods on the harness itself so
   subclasses inherit them.
8. **Re-declaring ports/mass balance in subclasses.** `Pump` and `StorageTank`
   inherit inlet/outlet ports and the `flow_vol` handles from `SISOBlock`;
   redeclaring them duplicates components. Add only the flow↔energy relationship
   (Pump) or the holdup dynamics (Tank). The tank *replaces* the pass-through
   balance rather than adding a second one.
9. **Tank silently keeping a `status` var.** The disable must be real: if a
   caller passes a `unit_commitment` config, `StorageTank` must still build **no**
   `status[t]`/UC constraints. Test it (`test_tank_logic_disabled`) — a tank with
   an on/off binary is a modeling bug, not just clutter (R6).

## Tests

- `src/flexops/tests/testing/test_harness.py` (all `@pytest.mark.unit`):
  - `test_base_class_not_collected` — no items collected from the bare harness (use `pytest.main` on a tiny in-line module or inspect collection; implementer's choice).
  - `test_dummy_time_block_shape` — `dummy_time_block(3)` has 3 time points, a `SimpleAqueousFlow`, 15-min `dt`.
- `src/flexops/tests/unit_models/base/test_siso.py` (all `@pytest.mark.unit`):
  - `test_siso_ports_and_mass_balance` — build a bare `SISOBlock` on `dummy_time_block(3)`: assert an `inlet` and an `outlet` `Port` exist and expose `flow_vol`, and that the per-stream mass-balance constraint exists indexed by `t` (count == N). Fix `inlet.flow_vol` to a known profile and assert the mass-balance constraint **body** evaluates to 0 at each `t` when `outlet.flow_vol` is set equal (constraint-body check, no solver — testing doc §5).
  - `test_siso_registers_no_energy` — a bare `SISOBlock` has no `electrical_work`/`thermal_work` (energy is a subclass concern), matching the base contract.
- `src/flexops/tests/unit_models/test_pump.py` — `class TestPump(UnitModelTestHarness)` (~30 lines): `configure()` builds `dummy_time_block(3)` + one `Pump`, fixes nothing; `expected_dof = 0`; `expected_solution` = hand-computed `electrical_work` for a fixed `flow_vol` (e.g. 100 m³/hr × 0.5 kWh/m³ → `{"electrical_work[0]": 50.0, ...}`).
- `src/flexops/tests/unit_models/test_storage_tank.py`:
  - `class TestStorageTank(UnitModelTestHarness)` — `configure()` on a 4-point `dummy_time_block(4)`, `max_volume=1000`, `initial_volume=200`; `expected_dof = 0` with flows fixed.
  - `test_mass_balance_by_hand` — `@pytest.mark.unit`. Build a 4-step tank, fix `flow_in = [100, 100, 0, 0]`, `flow_out = [50, 50, 50, 50]` m³/hr, set `V` values to the hand-computed trajectory from `V[0]=200` with `dt=0.25 h` (200, 212.5, 225, 212.5), and assert each holdup-constraint **body** evaluates to 0 within `pytest.approx(abs=1e-9)` — no solver (testing doc §5). Also assert exactly 3 holdup constraints exist.
  - `test_tank_logic_disabled` — `@pytest.mark.unit`. The canonical R6 check: build a `StorageTank` (once with no `unit_commitment` config, once **explicitly passing one on**) and assert it has **no** `status` Var and no unit-commitment/semicontinuous logic constraints in either case — a tank has no on/off status (architecture §3.4/§3.5, R6). Contrast with the `Pump`, which does not disable logic.
- `src/flexops/tests/unit_models/test_pump_tank_component.py`:
  - `test_pump_fills_tank_lp` — `@pytest.mark.component` + `@pytest.mark.needs_highs`. 24-point hourly TimeBlock; Pump → Arc → StorageTank; tank `flow_out` fixed to 50 m³/hr; objective: minimize total `electrical_work`; solve via `get_solver` (same skip guard). Assert optimal, and total pumped volume equals total demand ± initial/final holdup by mass balance, `pytest.approx(rel=1e-6)`.

## Documentation tasks

- `docs/_templates/autosummary/unit_model.rst` — the template from docs plan §3
  **without** the `.. flexops-unit-tables::` directive (it does not exist until
  M14); leave a reST comment `.. TODO(M14): insert flexops-unit-tables here`.
- `docs/reference/flexops/unit_models/index.rst` — autosummary listing the
  `SISOBlock` base plus `Pump` and `StorageTank` with `:template:
  unit_model.rst`; a one-line note that Pump/Tank subclass the SISO topology base.
- `docs/reference/flexops/testing.rst` — autodoc `UnitModelTestHarness` and
  `dummy_time_block`, plus a ~15-line "testing your own unit model" snippet
  showing a `configure()` subclass.
- Class docstrings on `Pump`/`StorageTank` per conventions §3: description,
  governing equations in `.. math::`, usage snippet, config cross-references —
  the Pump docstring must state the kWh/m³ × m³/hr = kW algebra.

## Definition of Done

- [ ] `UnitModelTestHarness` API matches `plan/02_testing_and_ci.md` §2 verbatim (attribute names, stage names, tier markers).
- [ ] `dummy_time_block(n=3)` exported from `flexops.testing`.
- [ ] `SISOBlock` exists in `flexops/unit_models/base/siso.py`: inlet/outlet ports on `SimpleAqueousFlow`, per-stream mass balance, energy-registration wiring; registers no energy itself.
- [ ] `Pump(SISOBlock)` and `StorageTank(SISOBlock)` importable from `flexops.unit_models`; both inherit SISO ports/mass balance; all IO/parameter/energy registrations present.
- [ ] `StorageTank` disables logic/unit-commitment (no `status` var / no UC constraints) even when a `unit_commitment` config is passed (R6); `test_tank_logic_disabled` passes.
- [ ] SISO base port/mass-balance unit test passes.
- [ ] `pytest -m unit` passes with no solver installed; `pytest -m component` passes with HiGHS (and skips cleanly when M05 is absent).
- [ ] Hand mass-balance test evaluates constraint bodies (no solve).
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W` passes with the three new/updated docs pages.
- [ ] CHANGELOG "Unreleased" entry: public test harness + first two unit models.
- [ ] plus the generic DoD in CLAUDE.md.
