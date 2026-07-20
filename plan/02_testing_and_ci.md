# 02 — Testing & CI

The testing architecture exists to keep two promises: (1) every block of the
platform is testable in isolation, and (2) **every test in the repository runs
and passes before any merge** — the packages are open source, CI minutes on
public repos are free, so nothing is deferred to after-merge schedules. Upstream
Pyomo/IDAES versions are pinned (decision R12) and bumped manually; there is no
upstream canary.

Development is **test-driven**: tests are written before the implementation,
and the full suite runs locally before every push (see §1a).

## 1. Test tiers (pytest markers)

Registered in `pyproject.toml` under `[tool.pytest.ini_options]` with
`filterwarnings`/`markers` set so **unknown markers are errors**.

All three tiers gate every merge. The tiers exist for **feedback speed**, not
for deferral: `unit` is the sub-second TDD inner loop, `component` the
pre-commit loop, `integration` the pre-push/PR loop — and in CI each tier is a
separate job so the fast signal reports first.

| Marker | Time budget | Solver allowed | Runs in |
|---|---|---|---|
| `unit` | < 1 s each | **No** | TDD inner loop; PR CI |
| `component` | < 30 s each | HiGHS / IPOPT only | pre-commit loop; PR CI |
| `integration` | minutes | any | pre-push loop; PR CI (dedicated job) |
| `needs_highs`, `needs_ipopt`, `needs_cbc`, `needs_gurobi`, `needs_scip` | — | availability skip | any |

- `unit` tests verify construction, units-consistency, registration, constraint
  *bodies* (evaluate expressions at fixed variable values and compare by hand —
  no solve needed).
- `component` tests solve small models (≤ ~100 timesteps) with the default
  open-source solvers.
- `integration` tests are end-to-end (multi-day horizons, parameterize
  round-trips on large frames, rolling-horizon vs. monolithic comparisons).

### Collection-time enforcement (the tier-discipline guard)

`conftest.py` at the repo root implements `pytest_collection_modifyitems`:

- **Fail collection** if any test carries zero or more than one tier marker.
- Under `-m unit`, automatically attach a guard fixture to every collected test that (a) monkeypatches the solver facade (`flexcore.solvers.facade`) so any attempt to invoke a solver raises immediately, and (b) blocks socket access so no test can reach the network. A `unit` test that secretly builds and solves a 30-day MIP then fails instantly and loudly — at the offending call — instead of silently turning the sub-second loop into a multi-minute one. The guard is installed *only* under `-m unit`; `component` and `integration` runs are meant to solve, so they leave it off. 
- Solver-availability markers consult `flexcore.solvers.registry` and `skip` (not fail) when the solver is absent.

This is the mechanical guard against the failure mode where heavy tests
accumulate in the fast lane until the sub-second TDD loop stops being
sub-second (WaterTAP/EnergyFlows accumulated heavy tests in fast lanes until
CI became unusable; vigilance does not scale, collection hooks do). Everything
still runs before merge — the hook protects the *fast local loop*, not a CI
budget.

## 1a. Test-driven development (the required workflow)

Every milestone is implemented test-first. The milestone work orders are
written to support this: their **Tests** section is a complete specification of
observable behavior, so it is the first thing you implement, not the last.

For each unit of work inside a milestone:

1. **Write the tests first**, copied/expanded from the milestone's Tests
   section, with their tier markers. Run them and **watch them fail** for the
   right reason (a missing class must fail with `ImportError`/`AttributeError`,
   not a typo in the test).
2. **Implement** the smallest code that makes them pass, running
   `pytest -m unit -x -q` (sub-second per test) as the inner loop.
3. **Refactor** with the tests green, then run the enclosing tiers:
   `pytest -m "unit or component" -q` after each work unit.
4. **Before every push, run the full suite locally**:

   ```bash
   ruff check . && black --check . && lint-imports
   pytest -q          # ALL tiers: unit + component + integration
   ```

   Always push with a fully green suite — every test passing, none skipped for
   convenience. A pre-push git hook running exactly these commands is installed
   by `pre-commit install --hook-type pre-push` (configured in M00); always let
   it run, reserving `--no-verify` for WIP branches that no one will merge.

Milestone Definition-of-Done checklists assume this ordering: a milestone whose
tests were written after the implementation is not done, it is untested code
with decorative assertions. Reviewers should be able to see test commits
precede or accompany implementation commits.

## 2. The unit-model test harness

`flexops/testing/harness.py` — a **public, shipped** module (users writing
custom unit models get the same harness). Built in M04, used by every unit-model
milestone after.

```python
class UnitModelTestHarness:
    """Subclass per unit model; pytest collects the subclass."""

    expected_dof: int = 0
    expected_solution: dict[str, float] = {}   # component name -> value
    solution_rtol: float = 1e-6

    def configure(self):
        """Build and return (model, unit). Override this."""
        raise NotImplementedError

    # Provided test stages (each is a real test method):
    # test_build             — constructs without exception; ports exist
    # test_units_consistent  — assert_units_consistent(unit)
    # test_io_registration   — registered IO vars exist, have units, are time-indexed
    # test_energy_naming     — power_electrical/power_thermal present iff declared
    # test_dof               — degrees of freedom == expected_dof after fixing declared inputs
    # test_solve             — (component tier) get_solver() solve is optimal
    # test_solution          — solved values match expected_solution within solution_rtol
```

A concrete unit's test file is ~30 lines: `configure()` plus the two expected
dicts. `expected_solution` doubles as a solution-regression baseline — changing
it requires a deliberate, reviewable diff. The build/units/registration/DoF
stages are marked `unit`; solve/solution are marked `component`.

## 3. CI workflows (`.github/workflows/`)

### ci.yml — the PR gate (every test, every merge)

The repository is public, so GitHub Actions minutes are free — **all tiers run
on every PR and must pass before merge**. Jobs are staged so the fast signal
reports first, but nothing is deferred:

- Trigger: pull_request + push to main.
- Job 1 `lint` (fast fail, ~2 min): ruff, `black --check`, `lint-imports`.
- Job 2 `fast-tests` (needs lint): matrix `py310 / py312` ×
  `pytest -m "unit or component" --cov --cov-report=xml` — reports in a few
  minutes.
- Job 3 `integration` (needs lint):
  `pytest -m integration --cov --cov-report=xml`. If this job's wall clock
  grows past ~30 min, shard it with a pytest-split matrix — do not demote tests
  out of the gate.
  - HiGHS from the `highspy` wheel; IPOPT via `idaes get-extensions`, with the
    binary directory **cached** keyed on IDAES version (both test jobs).
- Coverage gate: **diff coverage** ≥ 85 % on changed lines (`diff-cover`) over
  the combined unit+component+integration data, not a global threshold — global
  gates on a young codebase invite threshold-gaming. Per-subpackage totals
  (`--cov=src/flexcore --cov=src/flexops …` separately) are reported on every
  run so each future standalone repo already knows its number.
- Branch protection: `lint`, `fast-tests`, `integration`, and `docs` are all
  required status checks. Merging with any of them red is not possible.

### nightly.yml — safety net, never a gate

Everything that gates a merge already ran on the PR; nightly exists only to
catch flakes and environment drift.

- Trigger: cron (daily) + manual dispatch.
- Full suite (`pytest -q`, all tiers) in a **fresh, uncached** environment;
  forced (cache-free) re-execution of the example notebooks; coverage-trend
  upload.
- On failure: opens/updates a pinned "nightly drift" issue. A nightly failure
  with a green PR history means flakiness or environment drift — fix the flake,
  don't relax the gate.

### docs.yml

- PR: `sphinx-build -W --keep-going` **with notebook execution on** (myst-nb
  `cache` mode, jupyter-cache directory cached in CI) — a broken notebook blocks
  the merge, same as any other test. `NB_EXECUTION_MODE=off` remains available
  for fast local docs iteration only.
- main: build (cached execution), deploy (GitHub Pages or RTD).

## 4. Coverage policy

- PR gate: diff coverage (85 % changed lines) over all tiers combined.
- Per-subpackage totals reported on every PR run; M15 turns on gentle
  per-subpackage floors as required checks.
- Excluded from denominators: `flexops/testing/` (harness code).

## 5. Test-writing guidance for implementers

- Prefer constraint-body point checks (`unit`) over solves (`component`)
  wherever the math allows — they localize failures and cost milliseconds.
  Example: dwell-time constraints are verified by enumerating 6-step on/off
  schedules and asserting feasibility/infeasibility of each constraint body
  (truth tables), no solver involved.
- Golden-file test for the EECO integration (`flexops.costing`): feed EECO a
  known tariff/DR fixture + a known load, and match its computed cost to a
  hand-computed reference bill (energy + demand charge, PG&E-style TOU) to the
  cent. This validates *our use of EECO*, not EECO itself; update requires
  editing the golden file — a deliberate diff. (EECO is Pyomo-aware, so this
  test is typically `component` tier.)
- Fixtures for solved models: `setpoints`/`smoothing` tests operate on stored
  solved-model value dictionaries, not fresh solves — keeps them `unit` tier.
- Every numeric tolerance is explicit. Seeds are fixed. No test depends on
  wall-clock time or network access.
