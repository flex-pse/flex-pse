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
- `src/flexops/core/ops_block.py` — **modify**: add `add_bypass_constraints(inlet,
  outlet, *, exclude_vars=())`, the generic inlet-to-outlet state-variable
  pass-through gated by `allow_bypass` (§3.2's slot, wired for real here)
- `src/flexops/properties/simple_aqueous.py`, `simple_gas.py` — **modify**: add
  `get_flow_basis_var_name() -> str` (returns `"flow_vol_phase"`) so callers can
  exclude "the flow" from a generic pass-through without hardcoding a name that
  varies by property package
- `src/flexops/unit_models/base/__init__.py` — exports `SISOBlock`
- `src/flexops/unit_models/base/siso.py` — the `SISOBlock` IO-topology base
- `src/flexops/unit_models/__init__.py` — exports `Pump`, `StorageTank`
- `src/flexops/unit_models/pump.py` — `Pump(SISOBlock)` unit model
- `src/flexops/unit_models/storage_tank.py` — `StorageTank(SISOBlock)` unit model
- `src/flexops/tests/testing/test_harness.py` — harness self-checks
- `src/flexops/tests/unit_models/base/test_siso.py` — SISO base port + bypass unit
  tests (including with pressure/temperature enabled, and `allow_bypass=False`)
- `src/flexops/tests/unit_models/test_pump.py` — `TestPump` harness subclass
- `src/flexops/tests/unit_models/test_storage_tank.py` — `TestStorageTank` + hand
  mass-balance test + capacity/level-bounds tests + logic-disabled test
- `src/flexops/tests/unit_models/test_pump_tank_component.py` — 24-step LP system solve
- `src/flexops/tests/properties/test_simple_aqueous.py`, `test_simple_gas.py` —
  **modify**: add `test_get_flow_basis_var_name`
- `docs/reference/flexops/unit_models/index.rst`, `docs/reference/flexops/testing.rst`, `docs/_templates/autosummary/unit_model.rst` — reference docs (see Documentation tasks)

## Specification

### 1. `flexops/testing/harness.py`

Copy the public surface from `plan/02_testing_and_ci.md` §2 exactly:

```python
class UnitModelTestHarness:
    """Subclass per unit model; pytest collects the subclass."""

    expected_dof: int = 0
    expected_solution: dict[str, float] = {}   # component name -> value
    solution_rtol: float = 1e-6

    def configure(self):
        """Build and return (model, unit). Override this."""
        raise NotImplementedError
```

Provided stages, each a real `test_*` method on the class so pytest collects
them on every subclass:

- `test_build` — `@pytest.mark.unit`. `configure()` returns `(model, unit)` without exception; declared ports exist.
- `test_units_consistent` — `@pytest.mark.unit`. `assert_units_consistent(unit)` (import directly: `from pyomo.util.check_units import assert_units_consistent` — R12, no compat layer).
- `test_io_registration` — `@pytest.mark.unit`. Every entry in the unit's `IORegistry` (`flexops/core/registration.py`, M03): the referenced component exists on the unit, carries `pyunits`, is indexed by `t`, and has a **non-empty `doc=` string** (the M14 flexdoc generator renders these).
- `test_energy_naming` — `@pytest.mark.unit`. `power_electrical` / `power_thermal` exist **iff** the unit called `declare_power`/`register_power` for that `PowerKind`; no component named bare `power`/`energy`/`work` exists.
- `test_dof` — `@pytest.mark.unit`. Fix every registered `role="input"` IO variable at its current value, then assert `degrees_of_freedom(model) == self.expected_dof` (import directly: `from idaes.core.util.model_statistics import degrees_of_freedom` — R12, no compat layer).
- `test_solve` — `@pytest.mark.component`. Obtain a solver via `from flexcore.solvers import get_solver` guarded: if the import fails or `get_solver` raises `FlexSolverError`/is a stub, `pytest.skip("flexcore.solvers.get_solver not available (M05 may land in parallel)")`. Solve with inputs fixed; assert optimal termination.
- `test_solution` — `@pytest.mark.component`. For each `name -> value` in `expected_solution`, resolve `name` dotted against the unit (e.g. `"power_electrical[0]"`) and assert `pyo.value(...) == pytest.approx(value, rel=self.solution_rtol)`. Skip when `expected_solution` is empty. Resolution helper `_component_by_name(unit, name)` (implementer's choice on internals).

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
  **`SISOBlock` overrides the inherited `allow_bypass` default to `True`**
  (`CONFIG.get("allow_bypass").set_default_value(True)`) so the SISO topology
  (and `Pump`/`StorageTank` on top of it) is well-posed (DoF == 0) out of the
  box; pass `allow_bypass=False` to leave state variables unlinked and wire a
  custom relationship instead. The base `OpsBlock` default stays `False`.
- One **`inlet`** `Port` and one **`outlet`** `Port`, built via the inherited
  `add_stream_ports()` helper (M03, `flexops/core/ops_block.py`) from the
  configured `SimpleAqueousFlow` property package (no ControlVolumes — decision
  R1). The helper creates the `inlet_state`/`outlet_state` `StateBlock`s indexed
  by `time_block.time_index`, builds the two ports, and registers each state's
  `flow_vol_phase` as a process IO variable (inlet → `role="input"`, outlet →
  `role="output"`).
- **Per-stream mass balance via the generic bypass helper.** `OpsBlockData`
  gains `add_bypass_constraints(inlet, outlet, *, exclude_vars=())` (§3.2): for
  every state variable the inlet exposes and not named in `exclude_vars`, add
  an equality `outlet_var[idx] == inlet_var[idx]` over its full index set —
  unless the variable is already fully `fixed` (e.g. `dens_mass` under
  `fixed_density=True`), in which case no redundant constraint is built. Gated
  by `self.config.allow_bypass`: `False` builds nothing (a developer wires the
  relationship by hand); `True` builds it. `SISOBlock._build_mass_balance()`
  calls `self.add_bypass_constraints(self.inlet, self.outlet, exclude_vars=())`
  — flow's pass-through *is* a bypass equality here, named
  `bypass_flow_vol_phase_eq` (a `doc=` string on it). `SimpleAqueousFlow`
  carries the single `"Liq"` phase, so the constraint is indexed by `(t,
  "Liq")`. Subclasses that need a different balance (e.g. the tank's holdup)
  override `_build_mass_balance` and call `add_bypass_constraints` with the
  flow variable excluded instead — document when they do.

  **Implementation note (Port References vs. state blocks):** ports built via
  `add_inlet_port`/`add_outlet_port` expose their members as auto-generated
  `Reference` objects (`port.vars[name]`), which carry an extra leading
  `UnindexedComponent_set` dimension and are awkward to index directly.
  `add_bypass_constraints` instead resolves each port's sibling
  `"{port_name}_state"` block (the convention `add_stream_ports` establishes)
  and builds constraints directly against its state variables, so indices stay
  exactly `inlet_var.index_set()` (e.g. `(t, phase)`) with no leaked reference
  dimension.
- Energy-registration wiring inherited from `OpsBlockData` (§3.2) —
  `SISOBlock` itself registers **no** power (it does not know its subclass's
  draw); subclasses call `declare_power` as needed.
- Logic/unit-commitment is available (the base `status` capability, §3.5) but
  **not built by `SISOBlock`** — it is opt-in via the `unit_commitment` config,
  and a subclass may disable it entirely (see `StorageTank`, §5).

### 4. `unit_models/pump.py` — `Pump(SISOBlock)`

Subclass **`SISOBlock`** via the `declare_process_block_class` pattern (class
name **`Pump`**, data class `PumpData(SISOBlockData)`). It inherits the inlet/
outlet ports (and their `flow_vol_phase` IO registration) and the mass balance
from the SISO base, and adds only the electrical-work relationship. Behavior
(architecture §3.4):

- Config: inherits the SISO config; adds `energy_intensity` — default `0.5` in
  kWh/m³ (implementer's choice of default; document it).
- Inlet/outlet `Port`s and the mass balance
  `outlet_state.flow_vol_phase[t, "Liq"] == inlet_state.flow_vol_phase[t, "Liq"]`
  come **from `SISOBlock`** — do not re-declare them in `Pump`.
- `energy_intensity`: **mutable Param**, units `pyunits.kWh / pyunits.m**3`,
  registered `register_process_parameter(regressable=True)`.
- `power_electrical[t]` Var in kW (created and registered by the base-class
  `declare_power(PowerKind.ELECTRICAL)` helper from M03), with constraint
  `power_electrical[t] == pyunits.convert(energy_intensity *
  inlet_state.flow_vol_phase[t, "Liq"], pyunits.kW)`.
  **Unit algebra:** `flow_vol_phase` is m³/hr, so kWh/m³ × m³/hr = kWh/hr = kW —
  dimensionally exact with no fudge factor (the `pyunits.convert` to kW applies a
  factor of 1). Put this sentence in the constraint's `doc=` and the class
  docstring; `assert_units_consistent` is the referee.
- Registration: `add_stream_ports()` (inherited, called by the SISO base) has
  already registered the inlet `flow_vol_phase` as `role="input"`; create the
  power var via `declare_power(PowerKind.ELECTRICAL)` (which both creates
  `power_electrical[t]` and registers it as a power draw), then
  `register_io_variable(power_electrical, role="output")`.
- **Post-v0 TODO (do not implement now):** a detailed pump power law (e.g.
  `power ~ density * flowrate * head / efficiency`) as an alternative to the
  constant energy-intensity relationship above. Leave a `.. todo::` note in the
  module docstring; the constant-intensity model is the only relationship this
  milestone builds.

### 5. `unit_models/storage_tank.py` — `StorageTank(SISOBlock)`

Subclass **`SISOBlock`** (class name **`StorageTank`**, data class
`StorageTankData(SISOBlockData)`). It inherits the inlet/outlet ports (and their
`flow_vol_phase` state variables) from the SISO base; the tank's dynamics **replace** the SISO
pass-through mass balance with a holdup difference equation (inlet and outlet
flows differ — the tank stores the difference). **A tank has no on/off status,
so it disables the logic/unit-commitment layer** (architecture §3.4, §3.5, R6):
this is the canonical example of a physical subclass turning off a base
capability. Concretely, `StorageTankData` must ensure no `status[t]` Binary /
semicontinuous UC constraints are ever built for it — force the
`unit_commitment` config off (and/or skip the logic build) regardless of what a
caller passes, and document why in the class docstring.

**`max_volume` vs. `capacity` (be explicit about the distinction — both in code
and in the class docstring):**

- `max_volume` — the maximum **possible** tank volume: fixed by prior investment
  in an existing tank, or by space constraints on a potential build. A static
  config constant and the **upper bound on `capacity`**.
- `capacity` — the **chosen** tank volume (a design `Var`), which may be `<=
  max_volume`. Fixed at `max_volume` by default (operations mode); unfixed,
  subject to that same upper bound, in the M07 design mode.

- Config: inherits the SISO config; adds `min_volume` (default 0 m³),
  `max_volume` (required, m³), `initial_volume` (required, m³), `level_min`
  (default `0.0`), `level_max` (default `1.0`).
- `volume[t]` Var, m³, `doc=` set, bounds `(min_volume, max_volume)` from config
  (renamed from the earlier terse `V[t]` — spell it out).
- Holdup difference equation, **indexed over `t = 0 .. N-2` only** (N =
  `len(time_block.time_points)`):
  `volume[t+1] == volume[t] + dt * (flow_in[t] - flow_out[t])`.
  Off-by-one: `volume[N-1]` is defined by the constraint at `t = N-2`; there is
  no constraint indexed `N-1` (it would reference `volume[N]`, which does not
  exist). **Do not assume the flow basis's units** — wrap the whole
  right-hand-side difference in `pyunits.convert(dt * (flow_in[t] -
  flow_out[t]), to_units=pyunits.m**3)` rather than converting `dt` alone and
  trusting the flow to already be in m³/hr; this stays correct if a future
  property package's flow basis differs (e.g. m³/s).
- `flow_in[t]` / `flow_out[t]`: `pyo.Reference`s to the single-phase inlet /
  outlet flow — `inlet_state.flow_vol_phase[:, "Liq"]` and
  `outlet_state.flow_vol_phase[:, "Liq"]` on the ports **inherited from
  `SISOBlock`** (do not re-create the ports). The tank's holdup equation
  replaces SISO's pass-through mass balance (in a pump inlet == outlet; in a
  tank they differ by the stored volume).
- `initial_volume`: **mutable Param** (m³) + constraint `volume[0] ==
  initial_volume`. Register it **both** via
  `time_block.register_initial_state(param)` (rolling-horizon hook, architecture
  §3.1) **and** `register_process_parameter(regressable=False)`.
- `capacity`: Var (m³), a fixable design variable, `bounds=(min_volume,
  max_volume)` (encodes `capacity <= max_volume`), **fixed by default** to
  `max_volume`, plus constraint `volume[t] <= capacity` for all `t` (M07's design
  mode unfixes it, subject to that same upper bound). Do not register it as IO.
- **`level[t]`: bounded fractional fill relative to the *chosen* capacity.** A
  `Var`, dimensionless, `bounds=(level_min, level_max)`, defined by
  `level_definition[t]: volume[t] == level[t] * capacity`. Bounding `level`
  keeps the tank from draining all the way down (`level_min`) or overfilling
  (`level_max`); defaults `(0.0, 1.0)` reproduce today's feasible region exactly.
  **Deliberate tradeoff, document it:** `level[t] * capacity` is a product of
  two variables — linear (LP) when `capacity` is fixed (operations mode), but
  bilinear (**NLP**) when `capacity` is free (design mode / M16 multi-period
  sizing). Design-mode solves of a model containing a tank therefore need IPOPT
  or an explicit `flexschedule.SolveSequence` (R5), not HiGHS.
- **Bypass for every state variable other than flow.** A tank governs flow
  itself (via holdup), but any other state variable a richer property package
  exposes (pressure, temperature, …) should still pass straight through. Call
  `self.add_bypass_constraints(self.inlet, self.outlet, exclude_vars=[pkg.
  get_flow_basis_var_name()])` in `_build_mass_balance()`, where `pkg =
  self.config.property_package` — this excludes exactly the flow-basis
  variable (`"flow_vol_phase"` for both shipped packages, via
  `get_flow_basis_var_name()`) rather than hardcoding a name that varies by
  package. **Deferred (note only, do not build):** if a future mass/TDS
  property package exposes both `flow_vol_phase` and
  `flow_mass_phase_comp`, composition mixing at the tank is not modeled here —
  assume constant composition, or add real mixing constraints later; detect the
  case by checking for `flow_mass_phase_comp` on the port.
- Registration: the inherited `add_stream_ports()` registers the inlet
  `flow_vol_phase` as `role="input"`, but for a tank **both** flows are dispatch
  inputs, so re-register the outlet `flow_vol_phase` as `role="input"` (it is not
  an output here) and `register_io_variable(volume, role="output")`. **No**
  `declare_power`/`register_power` call — the tank draws nothing; this is the
  "iff" case for `test_energy_naming`.

## Pitfalls

1. **Holdup off-by-one.** Writing the constraint over all `t` raises KeyError at
   `volume[N]` or silently skips — index the Constraint over
   `list(time_points)[:-1]` explicitly and unit-test the constraint count (N−1).
2. **Unit algebra by luck.** Do not multiply by 3600 or 1/60 anywhere in Pump;
   kWh/m³ × m³/hr = kWh/hr is already kW dimensionally — wrap the product in
   `pyunits.convert(..., pyunits.kW)` (a factor of 1), never a hand fudge factor.
   In the tank, do not convert `dt` alone and assume the flow is already m³/hr —
   wrap the whole `dt * (flow_in[t] - flow_out[t])` expression in
   `pyunits.convert(..., to_units=pyunits.m**3)` so the holdup equation stays
   correct regardless of the property package's flow basis units.
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
   inherit inlet/outlet ports and their `flow_vol_phase` state variables from `SISOBlock`;
   redeclaring them duplicates components. Add only the flow↔energy relationship
   (Pump) or the holdup dynamics (Tank). The tank *replaces* the pass-through
   balance rather than adding a second one.
9. **Tank silently keeping a `status` var.** The disable must be real: if a
   caller passes a `unit_commitment` config, `StorageTank` must still build **no**
   `status[t]`/UC constraints. Test it (`test_tank_logic_disabled`) — a tank with
   an on/off binary is a modeling bug, not just clutter (R6).
10. **Port `Reference`s vs. state blocks in `add_bypass_constraints`.**
    `port.vars[name]` is an auto-generated `Reference` carrying an extra leading
    `UnindexedComponent_set` dimension (its index looks like `(None, t, phase)`,
    not `(t, phase)`) — building constraints directly against it leaks that
    dimension into every constraint name a caller might try to access naturally
    (e.g. `unit.bypass_flow_vol_phase_eq[t, "Liq"]` would `KeyError`). Resolve
    the port's sibling `"{port_name}_state"` block instead and use its
    `define_state_vars()` — the real state variables, cleanly indexed.
11. **`capacity` with no bounds.** Giving `capacity` no `bounds=` lets a
    design-mode solve pick a size above `max_volume` — the "chosen size never
    exceeds the maximum possible" invariant only holds if `capacity`'s Var
    bounds are literally `(min_volume, max_volume)`.
12. **`level` without an accepted NLP tradeoff.** `level[t] * capacity` (Var ×
    Var) is intentionally bilinear so that `level` is capacity-relative, not
    `max_volume`-relative. Don't "fix" this by silently swapping the denominator
    to `max_volume` — that changes the physical meaning; instead document that
    design-mode tank solves need IPOPT/a `SolveSequence`, not HiGHS.

## Tests

- `src/flexops/tests/testing/test_harness.py` (all `@pytest.mark.unit`):
  - `test_base_class_not_collected` — no items collected from the bare harness (use `pytest.main` on a tiny in-line module or inspect collection; implementer's choice).
  - `test_dummy_time_block_shape` — `dummy_time_block(3)` has 3 time points, a `SimpleAqueousFlow`, 15-min `dt`.
- `src/flexops/tests/unit_models/base/test_siso.py` (all `@pytest.mark.unit`):
  - `test_siso_ports_and_mass_balance` — build a bare `SISOBlock` on `dummy_time_block(3)`: assert an `inlet` and an `outlet` `Port` exist and expose `flow_vol_phase`, and that the flow bypass constraint `bypass_flow_vol_phase_eq` exists indexed by `(t, phase)` (count == N). Fix `inlet_state.flow_vol_phase[t, "Liq"]` to a known profile and assert the bypass constraint **body** evaluates to 0 at each `t` when `outlet_state.flow_vol_phase[t, "Liq"]` is set equal (constraint-body check, no solver — testing doc §5).
  - `test_siso_registers_no_energy` — a bare `SISOBlock` has no `power_electrical`/`power_thermal` (power draw is a subclass concern), matching the base contract.
  - `test_siso_bypasses_all_state_vars` — build with `SimpleAqueousFlow(has_pressure=True, has_temperature=True)` (the default `allow_bypass=True`); assert bypass equalities exist for `flow_vol_phase`, `pressure`, `temperature`, but **not** `dens_mass` (fixed under `fixed_density`, so no redundant constraint). Verify each non-flow bypass body evaluates to 0 when inlet==outlet (no solver).
  - `test_siso_units_consistent_with_options` — the pressure/temperature-enabled build stays `assert_units_consistent`.
  - `test_siso_allow_bypass_false_leaves_dof` — `allow_bypass=False` on the pressure/temperature-enabled build → no bypass constraints exist; after fixing only the inlet states, the outlet states remain unfixed/free (a developer must wire the relationship by hand). `degrees_of_freedom` counts only vars in active constraints, so with none built it is trivially 0 — assert that, and separately assert the outlet vars' `.fixed is False`.
- `src/flexops/tests/unit_models/test_pump.py` — `class TestPump(UnitModelTestHarness)` (~30 lines): `configure()` builds `dummy_time_block(3)` + one `Pump`, fixes nothing; `expected_dof = 0`; `expected_solution` = hand-computed `power_electrical` for a fixed inlet `flow_vol_phase` (e.g. 100 m³/hr × 0.5 kWh/m³ → `{"power_electrical[0]": 50.0, ...}`).
- `src/flexops/tests/unit_models/test_storage_tank.py`:
  - `class TestStorageTank(UnitModelTestHarness)` — `configure()` on a 4-point `dummy_time_block(4)`, `max_volume=1000`, `initial_volume=200`; `expected_dof = 0` with flows fixed.
  - `test_mass_balance_by_hand` — `@pytest.mark.unit`. Build a 4-step tank, fix `flow_in = [100, 100, 0, 0]`, `flow_out = [50, 50, 50, 50]` m³/hr, set `volume` values to the hand-computed trajectory from `volume[0]=200` with `dt=0.25 h` (200, 212.5, 225, 212.5), and assert each holdup-constraint **body** evaluates to 0 within `pytest.approx(abs=1e-9)` — no solver (testing doc §5). Also assert exactly 3 holdup constraints exist.
  - `test_capacity_bounded_by_max_volume` — `@pytest.mark.unit`. `capacity.lb == min_volume`, `capacity.ub == max_volume`, fixed at `max_volume` by default.
  - `test_level_bounds` — `@pytest.mark.unit`. Build with `level_min=0.1, level_max=0.9`; assert every `level[t]`'s bounds; with `capacity` at its default fixed value, set `volume`/`level` to a matching pair (e.g. 500/0.5 at capacity=1000) and assert `level_definition[t].body` evaluates to 0 (no solver).
  - `test_tank_logic_disabled` — `@pytest.mark.unit`. The canonical R6 check: build a `StorageTank` (once with no `unit_commitment` config, once **explicitly passing one on**) and assert it has **no** `status` Var and no unit-commitment/semicontinuous logic constraints in either case — a tank has no on/off status (architecture §3.4/§3.5, R6). Contrast with the `Pump`, which does not disable logic.
- `src/flexops/tests/unit_models/test_pump_tank_component.py`:
  - `test_pump_fills_tank_lp` — `@pytest.mark.component` + `@pytest.mark.needs_highs`. 24-point hourly TimeBlock; Pump → Arc → StorageTank; tank `flow_out` fixed to 50 m³/hr; objective: minimize total `power_electrical`; solve via `get_solver` (same skip guard). Assert optimal, and total pumped volume equals total demand ± initial/final holdup by mass balance, `pytest.approx(rel=1e-6)`. Operations mode (capacity fixed) keeps this model LP.
- `src/flexops/tests/properties/test_simple_aqueous.py`, `test_simple_gas.py`:
  - `test_get_flow_basis_var_name` — `@pytest.mark.unit`. `props.get_flow_basis_var_name() == "flow_vol_phase"` on each package.

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
- [ ] `SISOBlock` exists in `flexops/unit_models/base/siso.py`: inlet/outlet ports on `SimpleAqueousFlow` (via `add_stream_ports`), pass-through built via the generic `add_bypass_constraints` (no exclusions), power-registration wiring; registers no power itself; `allow_bypass` defaults `True` for the topology.
- [ ] `OpsBlockData.add_bypass_constraints(inlet, outlet, *, exclude_vars=())` exists: bypasses every non-excluded, non-fixed state variable; gated by `self.config.allow_bypass`; resolves each port's sibling state block rather than the port's leaky `Reference` vars.
- [ ] `SimpleAqueousFlow`/`SimpleGasFlow` expose `get_flow_basis_var_name() -> "flow_vol_phase"`.
- [ ] `Pump(SISOBlock)` and `StorageTank(SISOBlock)` importable from `flexops.unit_models`; both inherit SISO ports/bypass; all IO/parameter/energy registrations present. `Pump` carries a `.. todo::` note for a future detailed power law (not implemented).
- [ ] `StorageTank` uses `volume`/`level` (not the terse `V`); `capacity` is bounded `(min_volume, max_volume)`; `level[t] = volume[t]/capacity` is bounded `(level_min, level_max)` via `level_definition`; the holdup equation converts the whole flow-difference expression, not just `dt`.
- [ ] `StorageTank` disables logic/unit-commitment (no `status` var / no UC constraints) even when a `unit_commitment` config is passed (R6); `test_tank_logic_disabled` passes.
- [ ] SISO base bypass unit tests pass, including with pressure/temperature enabled and with `allow_bypass=False`.
- [ ] `pytest -m unit` passes with no solver installed; `pytest -m component` passes with HiGHS (and skips cleanly when M05 is absent).
- [ ] Hand mass-balance test evaluates constraint bodies (no solve).
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W` passes with the three new/updated docs pages.
- [ ] CHANGELOG "Unreleased" entry: public test harness + first two unit models.
- [ ] plus the generic DoD in CLAUDE.md.
