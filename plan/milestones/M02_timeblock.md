# M02 — TimeBlock

**Effort:** 2 days · **Depends on:** M01 · **Parallelizable:** no

## Goal

Build `TimeBlock`, the discrete-time substrate every other block indexes
against: an ordered integer time set, a unit-carrying `dt`, datetime↔index
utilities, and the rolling-horizon hooks (initial-state registry, window
metadata) that FlexSchedule drives in M12. A single TimeBlock spans **at most one
calendar month** at a **free resolution** (15-minute default, any positive
duration). It must construct a representative worst-case horizon (a full month at
15-minute resolution, ≈ 2976 points) in under one second, because users rebuild
models constantly.

## Read first

- `plan/01_architecture.md` §3.1 (TimeBlock spec — this milestone implements it
  verbatim), §3.3 (R2: why discrete time, never `dynamic=True`/Pyomo.DAE — you
  document this), §7 decision R2
- `plan/00_conventions.md` §2 (naming: `t`, ISO-8601, keyword-only constructors), §3–§4
- `plan/02_testing_and_ci.md` §1 (`unit` tier: < 1 s, no solver), §5
- `plan/03_documentation.md` §1 (docs layout — you create the first reference page)

## Files to create or modify

- `src/flexops/core/time_block.py` — `TimeBlock` / `TimeBlockData`
- `src/flexops/__init__.py` — export `TimeBlock` (API-freeze script uses `fo.TimeBlock`)
- `src/flexops/tests/core/test_time_block.py` (+ `__init__.py` for `tests/core/`)
- `docs/` — minimal Sphinx skeleton + first reference/explanation pages (see
  Documentation tasks)

## Specification

### Declaration

PlantBlock does not exist yet; TimeBlock is a `declare_process_block_class`
block usable directly on a `ConcreteModel`:

```python
from idaes.core import declare_process_block_class, ProcessBlockData

@declare_process_block_class("TimeBlock")
class TimeBlockData(ProcessBlockData):
    ...
```

Constructor usage (the API-freeze form — must work exactly like this):

```python
m.time_block = fo.TimeBlock(
    start_date="2025-01-01", end_date="2025-01-31", time_step=15 * pyunits.min
)
```

### CONFIG entries (Pyomo `ConfigValue`s, each with `description=`)

- `start_date` — ISO-8601 string or `datetime.datetime`/`datetime.date`.
- `end_date` — same; exclusive horizon end (see grid semantics below).
- `time_step` — a `pyunits`-carrying expression (e.g. `15 * pyunits.min`). A
  bare number raises `FlexConfigError` telling the user to multiply by a
  `pyomo.environ.units` unit.

Normalize dates with a single helper `_parse_date(value) -> datetime` using
`datetime.fromisoformat` for strings; anything unparseable raises
`FlexConfigError` naming the offending value and showing an ISO-8601 example.

### Horizon scope and grid semantics

`time_index` members are **interval starts**: point `i` corresponds to timestamp
`start_date + i * dt`. `end_date` is the exclusive end of the horizon and is
*not* a time point. `N = (end_date - start_date) / dt`.

- **At most one calendar month** (architecture §3.1): validate that
  `end_date - start_date` does not exceed one calendar month measured *from
  `start_date`* — **not** a fixed 30 days. `2025-01-01 → 2025-02-01` and
  `2025-02-01 → 2025-03-01` are both exactly "one month" and both allowed; a span
  longer than one such month raises `FlexConfigError` pointing the user at the
  rolling-horizon driver (M12) / design wrapper (§3.6) for longer studies.
  Compute the one-month bound with calendar arithmetic
  (`dateutil.relativedelta(months=1)`, or stdlib month/year rollover — add
  `python-dateutil` to core deps if used); do **not** hard-code 30 days.
- **Resolution is free.** `time_step` defaults to `15 * pyunits.min` but accepts
  any positive duration (1 min … hours). Nothing here assumes 15 min — always
  compute `N` from `dt`.
- `N` **must be a whole number**: if `(end_date - start_date)` is not an integer
  multiple of `dt`, raise `FlexConfigError` reporting the span, `dt`, and the
  nearest valid `end_date`s (no truncation, no warning — an off-grid horizon is a
  configuration error, not something to silently fix). `N <= 0` also raises
  `FlexConfigError`.

### Components and attributes (built in `build()`)

- `time_index` — `pyo.Set(initialize=range(n), ordered=True, doc=...)`.
  Members are plain integers `0..N-1`, never timestamps (see Pitfalls).
- `time` — `pyo.Param(time_index, initialize={i: i*dt_value}, units=<units of
  time_step>, doc=...)`: the "actual" time points, i.e. elapsed time `i*dt` in
  the user's units. This is the one sanctioned per-point indexed Param (a single
  numeric value per index); see the amended Pitfall 3.
- `dt` — `pyo.Param(initialize=<numeric value>, units=<units of time_step>,
  mutable=False, doc="Time-step length")`. Keep the user's units: extract them
  with `pyunits.get_units(time_step)` and the magnitude with `pyo.value(...)`.
  For internal date arithmetic convert once to seconds via
  `pyo.value(pyunits.convert(time_step, pyunits.s))`.
- `_datetime_index` — a plain-Python attribute (underscore-prefixed; NOT a
  Pyomo component): `pd.date_range(start, periods=n, freq=pd.Timedelta(seconds=step_s))`.
- `_initial_state_params` — plain-Python `list`, starts empty.

### Properties and methods (copy these signatures)

```python
@property
def n_points(self) -> int: ...                     # len(time_index)
@property
def horizon(self):                                  # n_points * dt, a unit-carrying
    ...                                             # Pyomo expression (implementer's choice)
@property
def datetime_index(self) -> pd.DatetimeIndex: ...  # returns _datetime_index
@property
def initial_state_params(self) -> tuple: ...        # tuple(_initial_state_params)

def index_of(self, timestamp) -> int: ...
def timestamp_of(self, i: int) -> pd.Timestamp: ...
def register_initial_state(self, param) -> None: ...
def window(self, start: int, length: int) -> TimeWindow: ...
```

- `index_of(timestamp)`: accept a `pd.Timestamp`, `datetime`, or ISO-8601
  string; coerce with `pd.Timestamp(...)`; look up via
  `self._datetime_index.get_loc(ts)`. An off-grid or out-of-range timestamp
  raises `FlexConfigError` reporting the timestamp, the grid step, and the
  horizon bounds. `timestamp_of(i)` is the inverse; out-of-range `i` raises
  `FlexConfigError`.
- `register_initial_state(param)`: append a **mutable** Pyomo `Param` to the
  registry — this is the set of values (tank level, battery SOC, on/off state)
  that the M12 rolling-horizon driver mutates between windows. Raise
  `FlexConfigError` if `param` is not a Param or not `mutable=True` (message:
  "declare it with mutable=True").
- `window(start, length)`: pure metadata helper (no Pyomo components) returning
  a frozen dataclass defined in the same module (shape is implementer's choice):

  ```python
  @dataclass(frozen=True)
  class TimeWindow:
      start_index: int
      indices: range            # range(start_index, start_index + length), clipped to n_points
      start_time: pd.Timestamp
      end_time: pd.Timestamp    # exclusive
  ```

  `start` is an integer index; `start` out of range raises `FlexConfigError`;
  a window running past the horizon is clipped (its `indices` are shorter than
  `length`).

Docstrings: module docstring states R2 in one paragraph (discrete integer time,
hand-written difference equations, never `dynamic=True`/DAE) and cross-links
`docs/explanation/time_and_dynamics.md`. Every component gets `doc=`.

## Pitfalls

1. **Timestamps as Set members.** A Pyomo Set of thousands of `Timestamp`
   objects (a full month at fine resolution is tens of thousands of points) is
   slow to construct and breaks integer index arithmetic (dwell times, `t-1`
   difference equations). Set members are `range(n)` integers; the
   `DatetimeIndex` lives beside the Set as a plain attribute.
2. **Storing pandas objects as Pyomo components** (e.g.
   `self.datetime_index = pd.date_range(...)` without the underscore
   attribute + property split). Pyomo's `Block.__setattr__` will complain or,
   worse, wrap it. Use `self._datetime_index` and expose via `@property`.
3. **Per-point Params.** Beyond `time_index` and the single numeric `time`
   Param (elapsed `i*dt`), do not build further O(n) per-point components —
   especially an indexed Param of timestamps — as that blows the 1-second
   budget. The `time` Param is cheap (one float per index) and must stay within
   the worst-case build budget; the `datetime_index` timestamps remain a plain
   pandas attribute, never a Pyomo component.
4. **Unit handling on `dt`.** `pyo.value(15 * pyunits.min)` is `15`, in
   minutes. Comparing against a horizon in seconds without
   `pyunits.convert` gives a 60× error. Convert exactly once, in `build()`.
5. **`fromisoformat` quirks.** Python 3.10's `datetime.fromisoformat` does not
   accept `"Z"` suffixes. v0 policy is naive local time (01_architecture §2.4);
   reject timezone-aware inputs with a `FlexConfigError` pointing at that policy.
6. **Keyword-only:** block construction goes through CONFIG so positional abuse
   is unlikely, but do not add any positional parameters to helper methods'
   public signatures that take dates.
7. **Import `idaes.core` directly** in `time_block.py` (decision R12 — no compat
   layer to route through).

## Tests

`src/flexops/tests/core/test_time_block.py` — every test `@pytest.mark.unit`
(< 1 s, no solver anywhere here). Shared fixture: `tb()` building a TimeBlock
on a fresh `ConcreteModel`, 1 day at 15 min (96 points).

- `test_n_points_and_horizon` — 96 points; `pyo.value(pyunits.convert(tb.horizon, pyunits.hr)) == pytest.approx(24.0, rel=1e-9)`.
- `test_index_roundtrip` — for `i in [0, 1, 47, 95]`: `index_of(timestamp_of(i)) == i`.
- `test_datetime_index_matches` — `datetime_index.equals(pd.date_range("2025-01-01", periods=96, freq="15min"))`.
- `test_dt_units_min_vs_hr_consistent` — two blocks, `time_step=15*pyunits.min`
  and `time_step=0.25*pyunits.hr`, same dates: identical `n_points`, identical
  `datetime_index`, and `pyunits.convert(dt, pyunits.s)` equal within `rel=1e-9`.
- `test_bare_number_time_step_raises` — `time_step=15` → `FlexConfigError`
  mentioning `pyunits`.
- `test_off_grid_timestamp_raises` — `index_of("2025-01-01 00:07:00")` →
  `FlexConfigError`; message names the timestamp.
- `test_out_of_range_raises` — `index_of("2026-01-01")` and `timestamp_of(9999)`
  → `FlexConfigError`.
- `test_non_divisible_dt_raises` — end 25 min past a 15-min grid start (span not
  an integer multiple of `dt`) ⇒ `FlexConfigError`; message names `dt` and the
  span. (No truncation, no `UserWarning` — non-divisible is now an error.)
- `test_over_one_month_rejected` — `start_date="2025-01-01"`,
  `end_date="2025-02-02"` (just over one calendar month from the start), 15-min
  step ⇒ `FlexConfigError` mentioning the one-month limit; and
  `end_date="2025-02-01"` (exactly one calendar month) **builds** without error.
  Include a February case: `start_date="2025-02-01"`, `end_date="2025-03-01"`
  (one calendar month, 28 days) builds; `end_date="2025-03-02"` raises — proving
  the bound is calendar-based, not a fixed 30 days.
- `test_coarse_resolution_horizon_builds` — `start_date="2025-01-01"`,
  `end_date="2025-01-15"`, `time_step=1*pyunits.hr` ⇒ 336 points; `dt` in hours;
  `datetime_index` matches `pd.date_range(..., freq="1h")`.
- `test_fine_resolution_horizon_builds` — `start_date="2025-01-01"`,
  `end_date="2025-01-02"`, `time_step=1*pyunits.min` ⇒ 1440 points; `dt` in
  minutes.
- `test_build_speed_worst_case` — a representative worst case: a full month at
  15-min resolution, `start_date="2025-01-01"`, `end_date="2025-02-01"` ⇒ 2976
  points (31-day January); wrap construction in `time.perf_counter()` and assert
  `< 1.0` s (plain time assertion; pytest-timeout is an acceptable alternative —
  implementer's choice). Nothing hard-codes 15 min or 30 days: the count is
  derived from the dates and `dt`.
- `test_register_initial_state_roundtrip` — register a
  `Param(initialize=0.5, mutable=True)`; it appears in `initial_state_params`;
  mutating its value is visible through the registry; a non-mutable Param
  raises `FlexConfigError`.
- `test_window_metadata` — `window(4, 8)` on the 96-point block: indices
  `range(4, 12)`, `start_time == timestamp_of(4)`, `end_time == timestamp_of(12)`
  (exclusive); `window(90, 20)` clips to `range(90, 96)`; `window(96, 1)` raises.

## Documentation tasks

- Create the minimal Sphinx skeleton (first milestone that ships docs —
  implementer's choice sanctioned here): `docs/conf.py` per
  `plan/03_documentation.md` §1 (autodoc, autosummary generate-on, napoleon
  Google style, intersphinx→pyomo/idaes/pandas/pydantic, myst_nb honoring
  `NB_EXECUTION_MODE=off`, furo, `nitpicky = True` with a curated ignore list)
  and `docs/index.rst` with a reference toctree. Skip `docs.yml` CI and the
  flexdoc extension (M14); the DoD gate is a local `-W` build.
- `docs/reference/flexops/core.rst` **starts here**: autosummary entries for
  `flexops.core.time_block` (`TimeBlock`, `TimeBlockData`, `TimeWindow`).
- `docs/explanation/time_and_dynamics.md` skeleton: state decision R2 (discrete
  TimeBlock, `dynamic=False`, hand-written difference equations; DAE fights
  binaries and rolling horizons) in 2–3 paragraphs; link from the module docstring.
- CHANGELOG: "Added TimeBlock."

## Definition of Done

- [ ] API-freeze construction line works verbatim on a `ConcreteModel`
- [ ] All tests above green under `pytest -m unit`; worst-case (≈2976-point,
      full-month/15-min) build < 1 s
- [ ] Over-one-calendar-month span rejected (calendar-based, not 30 days);
      exactly-one-month (incl. February) builds; non-divisible `dt` rejected;
      coarse (1-hr) and fine (1-min) resolutions build
- [ ] `ProcessBlockData` imported directly from `idaes.core`; `lint-imports` passes
- [ ] `TimeBlock` importable as `flexops.TimeBlock`
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W --keep-going -b html docs docs/_build` passes locally with the new pages
- [ ] Explanation page states R2; module/class docstrings Google-style with `doc=` on every component
- [ ] plus the generic DoD in CLAUDE.md
