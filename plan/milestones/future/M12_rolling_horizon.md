# M12 — FlexSchedule: rolling horizon + solve sequences

**Effort:** 3 days · **Depends on:** M09 · **Parallelizable:** with M10/M11

## Goal

Build the FlexSchedule scheduling engine core: a `RollingHorizon` iterator that
slices a long horizon into overlapping solve windows, a `StateCarryOver` that
threads tank levels / battery SOC between windows, an explicit `SolveSequence`
(relax → MIP → fix → NLP) that embodies decision R5, and a driver
`solve_rolling_horizon` that stitches committed window solutions into one result.
Deliverable: a 7-day tank+battery+TOU problem solved as 24 h windows lands within
2 % of the monolithic solve.

## Read first

- `plan/01_architecture.md` §6 (flexschedule: horizon.py, sequences.py — the spec this milestone implements; **the reporting rule R9**: report the EECO post-hoc cost, never the raw solver objective)
- `plan/01_architecture.md` §2.4 / §3.6 (EECO in-objective cost is a relaxed proxy; `FlexCosting.report_cost(model)` is the de-relaxed, user-facing number)
- `plan/01_architecture.md` §7 (decision **R9**)
- `plan/01_architecture.md` §2.2 (solver facade, `get_solver`, **decision R5**: classify loudly, never transform silently)
- `plan/01_architecture.md` §3.1 (TimeBlock: `register_initial_state`, `window(start, length)`, `dt`, `time_index`)
- `plan/01_architecture.md` §3.4 (Tank initial level and BatteryModel SOC are the carried states)
- `plan/02_testing_and_ci.md` §1 (tier markers, solver-availability markers) and §5 (test-writing guidance)
- `plan/00_conventions.md` §2 (keyword-only constructors), §3 (exceptions), §6 (flexschedule imports flexops/flexcore only, never flexparameterize)

## Files to create or modify

- `src/flexschedule/horizon.py` — `HorizonWindow`, `RollingHorizon`, `StateCarryOver`
- `src/flexschedule/sequences.py` — `SolveSequence`, step classes, `FailurePolicy`, `StepResult`/`SequenceResult`
- `src/flexschedule/driver.py` — `solve_rolling_horizon`, `ScheduleResult` (module name is implementer's choice; conventions §1 lists only horizon/sequences/setpoints/smoothing — add `driver.py` alongside them)
- `src/flexschedule/__init__.py` — export `RollingHorizon`, `StateCarryOver`, `SolveSequence`, `RelaxIntegers`, `SolveStep`, `FixIntegers`, `FailurePolicy`, `solve_rolling_horizon`, `ScheduleResult`
- `src/flexschedule/tests/test_horizon.py`, `test_sequences.py`, `test_driver.py`
- `docs/reference/flexschedule/index.rst` — autosummary entries for the new modules
- `docs/explanation/relaxation_policies.md` — write the R5 narrative (see Documentation tasks)
- `docs/how_to/schedule_rolling_horizon.md` — create as a stub (M13 completes it)

## Specification

### 1. `horizon.py`

```python
@dataclasses.dataclass(frozen=True)
class HorizonWindow:
    index: int                 # 0-based window number
    start: int                 # first global time index in this window (inclusive)
    end: int                   # one past the last global time index (exclusive)
    implementation_slice: slice  # committed prefix, in the window's LOCAL indices
    lookahead_slice: slice       # overlapped tail, LOCAL indices; solution discarded

class RollingHorizon:
    def __init__(self, *, time_block, window, overlap): ...
    def __iter__(self) -> Iterator[HorizonWindow]: ...
    def __len__(self) -> int: ...          # number of windows
    @property
    def window_steps(self) -> int: ...
    @property
    def overlap_steps(self) -> int: ...
```

- `window` and `overlap` each accept **either** an `int` (a step count) **or** a
  `pyunits` duration (e.g. `24 * pyunits.hr`). Precedence rule (document in the
  docstring): an `int` is always interpreted as a step count; a Pyomo quantity is
  converted via `time_block.dt` and must divide into a whole number of steps,
  otherwise raise `FlexConfigError` naming the offending value and the dt.
- Window arithmetic: windows advance by `stride = window_steps - overlap_steps`
  (require `0 <= overlap_steps < window_steps`, else `FlexConfigError`). Window
  `k` spans global indices `[k*stride, min(k*stride + window_steps, N))`. Only
  the non-overlapped prefix is committed: `implementation_slice = slice(0, stride)`
  for all windows except the last, whose implementation slice extends to its end
  (the tail window commits everything it covers and its `lookahead_slice` is
  empty). Tail windows may be shorter than `window_steps` (truncation). Every
  global index `0..N-1` must be committed exactly once — this is the invariant
  the unit tests pin down.

```python
class StateCarryOver:
    def __init__(self, *, pairs: list[tuple[pyo.Var, pyo.Param]]): ...
    def apply(self, *, window: HorizonWindow) -> dict[str, float]:
        """Read each Var at the end of the committed prefix; write into its Param.

        Returns {param_name: value} for logging/assertions.
        """
```

- `pairs` maps a time-indexed Var on the **window model** (e.g. `tank.volume`) to
  an initial-state Param on the same model — the Params units registered via
  `time_block.register_initial_state(param)` (§3.1). Constructing the pair list
  from that registry is the caller's job; `StateCarryOver` itself is dumb and
  explicit.
- Carry index (implementer's choice, document it): read the Var at local index
  `window.implementation_slice.stop` — the first lookahead point, which *is* the
  state at the start of the next window under the difference-equation convention
  `volume[t+1] = volume[t] + dt*(...)`. If that index is not in the Var's index set (tail
  window), read `implementation_slice.stop - 1`. No carry-over is applied after
  the final window.

**Design note — one window model, mutated between windows (follow this exactly).**
Building ONE Pyomo model on the full 7-day horizon and "solving parts of it" is
**wrong** for rolling horizon: exogenous data (prices, DR events, forecasts) and
initial states must change between windows, and full-horizon constraints would
couple windows. Instead: build a **window-length** TimeBlock model **once**
(window_steps points), then between windows mutate only `mutable=True` Params —
initial-state Params via `StateCarryOver`, exogenous price/DR Params via the
driver's `data_updates` hook. Do not rebuild the model per window in v0 unless
the tail window is shorter, in which case building one extra truncated model for
the tail is acceptable (implementer's choice; note it in the PR).

### 2. `sequences.py`

```python
class FailurePolicy(enum.Enum):
    ABORT = "abort"                    # raise FlexSolverError immediately
    FALLBACK = "fallback"              # undo/skip this step, continue with the next
    ACCEPT_RELAXED = "accept_relaxed"  # keep the current (possibly relaxed/fractional)
                                       # solution, flag the result, continue

class RelaxIntegers:
    def __init__(self, *, on_failure: FailurePolicy = FailurePolicy.ABORT): ...

class SolveStep:
    def __init__(self, *, problem_class=None, warm_start: bool = True,
                 prefer=None, on_failure: FailurePolicy = FailurePolicy.ABORT): ...

class FixIntegers:
    def __init__(self, *, round: float = 1e-5,
                 on_failure: FailurePolicy = FailurePolicy.ABORT): ...

class SolveSequence:
    def __init__(self, *, steps: Sequence[object]): ...
    def execute(self, model) -> "SequenceResult": ...
    @classmethod
    def canonical(cls) -> "SolveSequence": ...
```

- `RelaxIntegers`: switch every unfixed Binary/Integer Var domain to its
  continuous relaxation, preferring `flexops.logic.status`'s first-class `relax()`
  where the unit exposes it (§3.5), plain domain swap otherwise. Record the
  original domains on the model (e.g. a `dict` stashed in a private attribute —
  implementer's choice) so integrality can be restored.
- `SolveStep`: `problem_class` is a **hint** (`flexcore.solvers.classify.ProblemClass`);
  when set, classify the model and raise `FlexSolverError` on mismatch — with one
  documented exception: if the hint requires integrality (`MILP`/`MINLP`) and a
  prior `RelaxIntegers` recorded relaxed domains, restore those domains first,
  then solve (implementer's choice, but this is what makes the canonical sequence
  expressible with three step classes — document it in the class docstring).
  Obtain solvers **only** via `flexcore.solvers.facade.get_solver(model=...,
  problem_class=..., prefer=...)`. `warm_start=True` passes the current variable
  values as a warm start when the selected solver supports it; silently ignore
  (debug log) when it does not (implementer's choice).
- `FixIntegers`: round each integer/binary Var to the nearest integer and fix it.
  If any value deviates from its rounded value by more than `round`, the step's
  failure policy applies (ABORT raises listing the offending vars; ACCEPT_RELAXED
  fixes at the rounded value anyway and flags the result; FALLBACK leaves the
  vars unfixed and continues).
- `SolveSequence.execute` runs steps in order, timing each, and returns:

```python
@dataclasses.dataclass
class StepResult:
    step: str                       # e.g. "SolveStep(MILP)"
    status: str                     # "ok" | "failed" | "skipped" | "accepted_relaxed"
    termination: object | None      # pyomo TerminationCondition, None for non-solve steps
    wall_time_s: float

@dataclasses.dataclass
class SequenceResult:
    steps: list[StepResult]
    success: bool
    accepted_relaxed: bool          # True iff any step resolved via ACCEPT_RELAXED
```

- `SolveSequence.canonical()` is the sequence architecture §6 names —
  relax → MIP (warm start) → fix ints → NLP polish:

```python
SolveSequence(steps=[
    RelaxIntegers(),
    SolveStep(problem_class=ProblemClass.LP),                      # solve relaxation
    SolveStep(problem_class=ProblemClass.MILP, warm_start=True),   # restores integrality
    FixIntegers(round=1e-5),
    SolveStep(problem_class=ProblemClass.NLP,
              on_failure=FailurePolicy.ACCEPT_RELAXED),            # polish; optional
])
```

**R5, restated — this module is where it lives.** The solver facade (§2.2)
classifies loudly and never relaxes integrality, never decomposes, never
transforms a model silently: a relaxed-MIP schedule sent to a real plant is a
correctness hazard. Every relaxation in flex-pse is therefore an **explicit,
named, ordered step in a `SolveSequence`** that the user composed and can read
back in the `SequenceResult`. Repeat this paragraph (in your own words) in the
module docstring and in `docs/explanation/relaxation_policies.md`.

### 3. `driver.py`

```python
def solve_rolling_horizon(*, model, horizon: RollingHorizon,
                          sequence: SolveSequence,
                          carry_over: StateCarryOver | None = None,
                          data_updates=None) -> "ScheduleResult": ...
```

- `model`: a built window-length model **or** a zero-argument callable returning
  one (`model_factory`); detect with `callable()` (implementer's choice).
- `data_updates`: optional callable `(window: HorizonWindow, model) -> None`
  invoked before each window's solve to write window-local exogenous data
  (prices, DR flags) into mutable Params — sliced from full-horizon series using
  `window.start`/`window.end`.
- Per window: apply `data_updates` → `sequence.execute(model)` → collect the
  values of registered IO variables (via `flexops.core.registration.iter_io_registry`)
  over `implementation_slice` → **evaluate the window's committed EECO post-hoc
  cost via `FlexCosting.report_cost(model)`** over the committed prefix →
  `carry_over.apply(window=...)`.
- `ScheduleResult` (dataclass): `committed: pd.DataFrame` — one row per committed
  global timestep, indexed by timestamp, one column per collected variable using
  flattened `plant.unit.variable` names plus a units mapping in `.attrs`
  (implementer's choice; M13's `extract_setpoints` defines the public tidy
  schema, so keep this internal shape simple); `windows: list[SequenceResult]`;
  and the **reported cost** fields below.

**Reporting rule (architecture §6, decision R9) — the reported number is the
EECO post-hoc cost, never the raw solver objective.** The in-objective cost EECO
builds is a *convex-relaxed* proxy and the objective may also carry scalarized
emissions / penalty / lost-production terms (§2.4), so the solver objective value
is never the user-facing cost. Concretely:
- `ScheduleResult.reported_cost: float` — the sum over committed windows of
  `FlexCosting.report_cost(model)` (EECO evaluated post-solve on the realized
  committed-step aggregate-power numpy array). This is the default, user-facing
  number and the comparison metric for the monolithic-vs-rolling check below
  (multi-stage / iterative comparison uses this post-hoc cost — no duals needed).
- The **raw solver objective is surfaced only behind an explicit debug flag** —
  e.g. `solve_rolling_horizon(..., debug_objective=False)` (implementer's choice
  of flag name/location). When off (the default), `ScheduleResult` does **not**
  expose the objective value; when on, an `objective_committed: float | None`
  field carries the committed-step objective contribution for debugging only.
  Document that it is a relaxed internal quantity, not the reported cost.

**Billing-period-aware demand + fixed charges (do not naively sum
`report_cost` across windows).** A demand charge is a *peak assessed over a
billing period* — monthly or daily, per the tariff's `assessed` column — and the
customer/fixed charge is a flat per-period charge. Summing each window's
`report_cost` over windows shorter than that period recounts the charge in every
window, overstating the bill. `FlexCosting.report_cost(model, *,
prev_demand_dict=..., scale_fixed_charges=...)` already threads both controls
into the EECO evaluation (the plumbing landed on the `demand-charges` branch):
the prior-demand carry makes EECO bill only the demand *incremental* above the
running peak, and `scale_fixed_charges` spreads the customer charge per timestep
so it stays additive. This milestone must:
- carry a `prev_demand_dict` across the windows of each billing period, seeded
  `None` at the first window and updated after each committed window from the
  realized demand — using a demand-carry builder from a future EECO release
  (so this milestone is gated on bumping EECO), and
- set `scale_fixed_charges=True` so each window's committed prefix pays only its
  share of the fixed charge and the windows sum to one charge.

Proration *within* a single sub-billing-period window is already EECO's, not
flex-pse's: the `demand-charges` branch delegated it to `get_charge_dict` —
monthly-assessed demand scaled by its assessed-aware `demand_scale_factor`
(daily-assessed demand, billed per day, is left unscaled) and the fixed charge
spread by `scale_fixed_charges` — and dropped flex-pse's own rate-level proration
helpers.

## Pitfalls

1. **Building one full-horizon model and slicing it.** Wrong, per the design note
   above — build one window-length model and mutate Params. If you find yourself
   deactivating constraints per window, stop.
2. **Off-by-one in the carry index.** The state at the start of window k+1 is the
   value at the *first lookahead point* of window k, not the last committed
   point. The toy-model unit test exists to catch exactly this.
3. **Double-committing overlap steps.** Overlapped indices belong to the *earlier*
   window's lookahead and the *later* window's implementation prefix — commit
   each global index exactly once. Assert this invariant in `RollingHorizon`
   tests before touching the driver.
4. **Mutating Params that are not `mutable=True`.** Pyomo raises confusingly late;
   check and raise `FlexConfigError` up front in `StateCarryOver.__init__`.
5. **Silently relaxing on solver mismatch.** If the model is MINLP and only
   HiGHS+IPOPT are installed, `get_solver` must raise (R5) — do not catch and
   "helpfully" relax inside `SolveStep`. The user composes the sequence.
6. **Warm starts that crash solvers lacking support.** Gate on solver capability;
   never let `warm_start=True` turn into a hard failure.
7. **Unit-tier tests that solve.** The repo conftest monkeypatches the facade to
   raise under `-m unit` (02 §1) — sequence unit tests must inject their own mock
   solver (monkeypatch `flexcore.solvers.facade.get_solver`), never a real one.
8. **Importing flexparameterize.** Forbidden by the import DAG; the two are
   mutually independent (conventions §6).
9. **Reporting the solver objective as the result.** The objective is a
   convex-relaxed, possibly scalarized proxy (arch §2.4). `ScheduleResult`'s
   user-facing number is `reported_cost` from `FlexCosting.report_cost` (R9); the
   objective appears only behind the explicit debug flag. Comparing rolling vs.
   monolithic on objective values (instead of post-hoc costs) is the classic bug.

## Tests

All in `src/flexschedule/tests/`. Exactly one tier marker each.

`test_horizon.py` — all `@pytest.mark.unit`:
- `test_window_count_exact_fit` — N=48, window=24, overlap=0 → 2 windows, slices tile `0..47`.
- `test_window_overlap_arithmetic` — N=48, window=24, overlap=4 → stride 20; starts 0/20/40; per-window `implementation_slice`/`lookahead_slice` checked literally; every global index committed exactly once.
- `test_tail_window_truncation` — tail shorter than `window_steps`; its implementation slice covers its full length; `lookahead_slice` empty.
- `test_duration_inputs_pyunits` — `window=6 * pyunits.hr` with `dt=15 min` → `window_steps == 24`; non-divisible duration raises `FlexConfigError`.
- `test_int_means_steps` — `window=24` (int) is 24 steps regardless of dt (precedence rule).
- `test_invalid_overlap_raises` — `overlap >= window` → `FlexConfigError`.
- `test_carry_over_toy_model` — toy `ConcreteModel` with two indexed Vars and two mutable Params; hand-set var values; `apply()` writes the expected two values into the Params and returns them.

`test_sequences.py` — all `@pytest.mark.unit`, solver mocked via `monkeypatch` of `flexcore.solvers.facade.get_solver`:
- `test_step_ordering` — recording mock asserts steps execute in declared order; `SequenceResult.steps` names match.
- `test_relax_and_restore_domains` — `RelaxIntegers` turns Binary → continuous [0,1]; a subsequent `SolveStep(problem_class=MILP)` restores Binary before solving.
- `test_fix_integers_rounds_and_fixes` — values 0.99999/1e-6 fixed to 1/0; value 0.4 with `round=1e-5` triggers the failure policy.
- `test_failure_abort` — mock returns infeasible → `FlexSolverError` naming the step.
- `test_failure_fallback` — failed step recorded `"skipped"`, later steps still run, `success` reflects final state.
- `test_failure_accept_relaxed` — fractional binaries accepted; `accepted_relaxed is True`.
- `test_canonical_sequence_shape` — `SolveSequence.canonical()` has the five documented steps in order.

`test_driver.py`:
- `test_two_window_smoke` — `@pytest.mark.component` + `@pytest.mark.needs_highs`: 2 windows × 12 steps (overlap 0 or 2), Tank + Pump + TOU FlexCosting; end-to-end `solve_rolling_horizon`; asserts 24 committed rows, tank level continuous across the window boundary (`pytest.approx(rel=1e-6)`), all `SequenceResult.success`.
- `test_seven_day_vs_monolithic` — `@pytest.mark.integration` + `@pytest.mark.needs_highs` (runs in the PR CI `integration` job and in the local pre-push suite): 7-day, 15-min tank+battery+TOU problem; rolling solve with 24 h windows / 4 h overlap vs. one monolithic 7-day solve, both via HiGHS; the comparison uses the **EECO post-hoc cost** (R9): `result.reported_cost == pytest.approx(monolithic_report_cost, rel=0.02)`, where `monolithic_report_cost` is the monolithic model's own `FlexCosting.report_cost(model)` — not either solve's objective value.
- `test_schedule_result_reports_eeco_cost_not_objective` — `@pytest.mark.component` + `@pytest.mark.needs_highs`: run `solve_rolling_horizon` with defaults on the two-window tank+pump+TOU case and assert (a) `ScheduleResult.reported_cost` is populated and equals the summed committed `FlexCosting.report_cost` (rel=1e-6), and (b) the objective value is **not** surfaced by default (no `objective_committed`, or it is `None`); only when the explicit debug flag is set does the objective field appear. This pins the reporting rule (arch §6, R9).

## Documentation tasks

- Google-style docstrings throughout, including the design note (one window model,
  mutated Params) in the `horizon.py` module docstring and the R5 restatement in
  `sequences.py`.
- Write `docs/explanation/relaxation_policies.md`: R5, why the facade never
  transforms, how `SolveSequence` makes relaxation explicit, the canonical
  sequence, failure policies.
- Add the three modules to `docs/reference/flexschedule/index.rst` (autosummary).
- Create `docs/how_to/schedule_rolling_horizon.md` as a stub linking the
  reference pages (M13 completes it).
- CHANGELOG entry under "Unreleased".

## Definition of Done

- [ ] `RollingHorizon` accepts step counts and pyunits durations; every global index committed exactly once, tail truncation correct
- [ ] `StateCarryOver` threads registered initial-state Params between windows; toy-model test passes
- [ ] `SolveSequence` with `RelaxIntegers`/`SolveStep`/`FixIntegers`, per-step `FailurePolicy`, per-step status/timings in `SequenceResult`; `canonical()` matches architecture §6
- [ ] All solves go through `flexcore.solvers.facade.get_solver`; no silent model transformation anywhere (R5)
- [ ] `solve_rolling_horizon` returns a `ScheduleResult` with committed-trajectory DataFrame
- [ ] `ScheduleResult` reports the EECO post-hoc cost (`reported_cost` via `FlexCosting.report_cost`), never the raw solver objective by default; the objective is surfaced only behind an explicit debug flag (arch §6, R9)
- [ ] Monolithic-vs-rolling comparison uses the post-hoc EECO cost on both sides (no duals)
- [ ] All tests above written, correctly tier-marked; unit tests pass with no solver installed
- [ ] Integration test (PR CI `integration` job): 7-day windowed vs. monolithic within 2 %
- [ ] `docs/explanation/relaxation_policies.md` written; flexschedule reference pages build with `sphinx-build -W`
- [ ] CHANGELOG updated; PR description records any implementer's-choice decisions
- [ ] plus the generic DoD in CLAUDE.md
