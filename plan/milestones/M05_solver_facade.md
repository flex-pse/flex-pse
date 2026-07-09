# M05 — Solver abstraction

**Effort:** 2 days · **Depends on:** M00 · **Parallelizable:** with M02–M04, M06

## Goal

Build `flexcore.solvers`: a model classifier, a solver capability registry, and
the `get_solver()` facade that picks the best available solver for a problem
class — and errors loudly (decision R5) instead of ever transforming a model.
This also replaces the M00 conftest solver stub so `-m unit` runs mechanically
cannot solve, and gives the `needs_*` markers a real availability source.

## Read first

- `plan/01_architecture.md` §2.2 (solvers — the spec for all three modules), §7 row R5
- `plan/02_testing_and_ci.md` §1 (tiers, `needs_*` markers), §1 "Collection-time enforcement" (the conftest hook you are wiring up)
- `plan/00_conventions.md` §3 (exceptions: `FlexSolverError` messages state what was wrong **and what to do**)

## Files to create or modify

- `src/flexcore/solvers/__init__.py` — exports `get_solver`, `SolverFacade`, `ProblemClass`, `classify`, `available_solvers`
- `src/flexcore/solvers/classify.py` — `ProblemClass` enum + `classify(model)`
- `src/flexcore/solvers/registry.py` — capability matrix + availability probing
- `src/flexcore/solvers/facade.py` — `get_solver()` + `SolverFacade`
- `conftest.py` (repo root) — replace the M00 stub: real monkeypatch under `-m unit`; `needs_*` skips consult `registry`
- `src/flexcore/tests/solvers/test_classify.py`, `test_registry.py`, `test_facade.py`, `test_solve_component.py` — tests
- `docs/reference/flexcore/index.rst`, `docs/explanation/relaxation_policies.md` — docs

## Specification

### 1. `classify.py`

```python
class ProblemClass(enum.Enum):
    LP = "LP"
    QP = "QP"
    MILP = "MILP"
    NLP = "NLP"
    MINLP = "MINLP"
    # Reserved strategy slots — documented, deliberately unimplemented (R5):
    MINLP_OA = "MINLP_OA"   # outer approximation, post-v0
    MINLP_TR = "MINLP_TR"   # trust region, post-v0

def classify(model: pyo.ConcreteModel) -> ProblemClass: ...
```

Algorithm (use only Pyomo public API):

1. Walk **active** constraints and the active objective. For each expression
   take `expr.polynomial_degree()`: degree 0/1 → linear; 2 → quadratic;
   `None` or > 2 → general nonlinear.
2. Collect variables appearing in those expressions
   (`pyomo.core.expr.identify_variables`). A variable counts as discrete iff
   `var.is_binary() or var.is_integer()` **and `not var.fixed`** — an LP with
   fixed binaries classifies **LP** (architecture §2.2).
3. Map: no discrete + all linear → `LP`; no discrete + quadratic objective +
   all-linear constraints → `QP`; no discrete otherwise → `NLP` (quadratic
   *constraints* classify NLP, not QP — smallest choice, document it in the
   docstring; implementer's choice); discrete + all linear → `MILP`; discrete +
   any nonlinearity → `MINLP`.
4. `MINLP_OA` / `MINLP_TR` are never returned by `classify`; docstring says so.

### 2. `registry.py`

```python
CAPABILITIES: dict[str, set[ProblemClass]] = {
    "highs":  {ProblemClass.LP, ProblemClass.MILP},
    "cbc":    {ProblemClass.LP, ProblemClass.MILP},
    "ipopt":  {ProblemClass.NLP},
    "gurobi": {ProblemClass.LP, ProblemClass.QP, ProblemClass.MILP},
}

def available_solvers() -> dict[str, set[ProblemClass]]: ...
```

- `available_solvers()` returns the subset of `CAPABILITIES` whose solver is
  installed, probing with
  `pyomo.environ.SolverFactory(name).available(exception_flag=False)` wrapped
  in `try/except` (some plugins raise instead of returning False).
- Probing is **cached** module-level (probes can take ~1 s each); provide
  `_reset_availability_cache()` for tests (implementer's choice: `functools.lru_cache`
  on a private probe function + `cache_clear`).
- The matrix is extensible: keep `CAPABILITIES` a plain module constant so a
  user can register an entry before calling `get_solver`.
- This is the source of truth for the pytest `needs_highs` / `needs_ipopt` /
  `needs_cbc` / `needs_gurobi` markers (see conftest below).

### 3. `facade.py`

```python
def get_solver(
    model=None, problem_class=None, prefer=None
) -> SolverFacade: ...
```

- Exactly one of `model` / `problem_class` may be given; if `model`, call
  `classify(model)`; if neither, default to `ProblemClass.LP` (smallest choice;
  implementer's choice — document it).
- Candidate order: `prefer` first if named, then a fixed priority list
  `["gurobi", "highs", "cbc", "ipopt"]` (implementer's choice); pick the first
  candidate that is available **and** lists the problem class in its
  capability set. If `prefer` is given but unavailable/incapable, fall through
  to the priority list and log a warning (`logging`, never `print`).
- `SolverFacade`: a thin wrapper — attributes `name: str`,
  `problem_class: ProblemClass`; method `solve(model, **kwargs)` delegating to
  the underlying Pyomo solver object with `tee=False` default. No option
  translation in v0 (implementer's choice on internals; keep the public
  surface to `name`, `problem_class`, `solve`).
- Failure: raise `FlexSolverError` (from `flexcore.exceptions`) with install
  instructions, e.g. `"No available solver supports NLP. Install IPOPT with
  'idaes get-extensions' or pass prefer=<installed solver>."`
- **MINLP (R5, quote it):** the facade never relaxes integrality, never
  decomposes, never sets up trust regions on its own. `get_solver` for
  `ProblemClass.MINLP` (with no MINLP-capable solver registered — v0 registers
  none) raises `FlexSolverError`: *"this model is MINLP; compose a
  `flexschedule.SolveSequence` (relax → MIP → fix → NLP) or install a
  MINLP-capable solver."* Silent model surgery is a correctness landmine for
  plant-control users.

### 4. Root `conftest.py` — replace the M00 stub

- Under `-m unit` (detect via `config.getoption("-m")` containing `unit`;
  reuse M00's detection), install an autouse fixture that monkeypatches
  `flexcore.solvers.facade.SolverFacade.solve` **and**
  `flexcore.solvers.facade.get_solver` to raise
  `RuntimeError("solver invocation is forbidden in unit-tier tests")`.
  Keep M00's socket blocking.
- `needs_*` markers: in `pytest_collection_modifyitems`, for each item carrying
  `needs_<name>`, call `flexcore.solvers.registry.available_solvers()` (probe
  once per session, not per item) and add
  `pytest.mark.skip(reason=f"solver {name} not installed")` when absent.
  Skip, never fail.

## Pitfalls

1. **Counting fixed binaries as discrete.** The whole point of the relax-fix
   workflow (M12) is that a fixed-binary model resolves as LP/NLP; test case 6
   below guards this.
2. **`polynomial_degree()` on constraints vs bodies.** Call it on
   `constraint.body` (the relational expression itself is not an arithmetic
   expression). Inactive constraints/blocks must be excluded — iterate with
   `model.component_data_objects(pyo.Constraint, active=True, descend_into=True)`.
3. **Probing cost.** Un-cached `SolverFactory(...).available()` in the marker
   hook makes collection take seconds; probe once and cache.
4. **`available()` raising.** Some Pyomo solver plugins raise
   `ApplicationError` from `available()`; treat any exception as "not
   available", never crash the registry.
5. **Import cycle with conftest.** conftest imports `flexcore.solvers`; keep
   `flexcore.solvers` free of imports from `flexops` (import-linter enforces
   the layer, but a test-time cycle can still bite — import inside the hook).
6. **Softening the MINLP error.** Do not add a `relax=True` escape hatch, even
   "temporarily" — R5 says relaxation strategies live only in
   `flexschedule.SolveSequence` (M12).

## Tests

- `src/flexcore/tests/solvers/test_classify.py` — all `@pytest.mark.unit`, six
  synthetic 2–3 variable models, one test each, asserting the enum value:
  - `test_classify_lp` — linear objective + linear constraint → `LP`.
  - `test_classify_milp` — LP + one unfixed Binary in a constraint → `MILP`.
  - `test_classify_nlp` — `pyo.exp(x)` in a constraint → `NLP`.
  - `test_classify_minlp` — nonlinear constraint + unfixed Binary → `MINLP`.
  - `test_classify_qp` — quadratic objective, linear constraints → `QP`.
  - `test_classify_lp_with_fixed_binaries` — the MILP model with the binary `.fix(1)` → `LP`.
- `src/flexcore/tests/solvers/test_registry.py` — `@pytest.mark.unit`:
  `available_solvers()` returns a dict whose keys ⊆ `CAPABILITIES`; probing is
  cached (monkeypatch the probe, call twice, assert one probe call).
- `src/flexcore/tests/solvers/test_facade.py` — `@pytest.mark.unit` (monkeypatch
  `available_solvers` so no real solver is touched):
  - `test_no_capable_solver_message` — empty availability → `FlexSolverError`; assert the message names the problem class **and** contains an install instruction (`"idaes get-extensions"` or equivalent).
  - `test_minlp_error_mentions_solve_sequence` — MINLP request → message contains `"flexschedule.SolveSequence"` and `"relax"`.
  - `test_prefer_respected_and_fallback` — `prefer="cbc"` picked when capable; incapable `prefer` falls back with a logged warning (`caplog`).
  - `test_model_vs_problem_class_exclusive` — passing both raises `ValueError` (or `FlexSolverError`; pick one and test it — implementer's choice).
- `src/flexcore/tests/solvers/test_solve_component.py` — `@pytest.mark.component`
  smoke solves of 2-variable problems, asserting optimal termination and the
  known optimum with `pytest.approx(rel=1e-6)`:
  - `test_highs_lp_smoke` — `@pytest.mark.needs_highs`, tiny LP via `get_solver(problem_class=ProblemClass.LP, prefer="highs")`.
  - `test_highs_milp_smoke` — `@pytest.mark.needs_highs`, tiny MILP.
  - `test_ipopt_nlp_smoke` — `@pytest.mark.needs_ipopt`, tiny convex NLP.

## Documentation tasks

- `docs/reference/flexcore/index.rst` — add a "Solvers" section: autodoc
  `get_solver`, `SolverFacade`, `ProblemClass`, `classify`,
  `available_solvers`; render the capability matrix as a table.
- `docs/explanation/relaxation_policies.md` — start the page with a short
  statement of R5 ("classify loudly, never transform silently"), the MINLP
  error text verbatim, and a forward pointer to `flexschedule.SolveSequence`
  (M12). Keep it ~half a page; M12 extends it.
- Docstrings: `get_solver` gets an `Example:` block (conventions §3 / docs plan §5).

## Definition of Done

- [ ] `ProblemClass`, `classify`, `available_solvers`, `get_solver`, `SolverFacade` importable from `flexcore.solvers` with the exact signatures above.
- [ ] Six classification unit tests pass, including LP-with-fixed-binaries.
- [ ] `FlexSolverError` messages tested for content (install instructions; MINLP → SolveSequence wording).
- [ ] Root conftest: `-m unit` forbids solver invocation (verify: a deliberate solve under `-m unit` fails fast); `needs_*` markers skip via `registry`.
- [ ] Component smoke solves green with HiGHS/IPOPT installed, skipped cleanly without.
- [ ] `NB_EXECUTION_MODE=off sphinx-build -W` passes; relaxation_policies.md states R5.
- [ ] CHANGELOG "Unreleased" entry; import-linter still green (no `flexops` imports).
- [ ] plus the generic DoD in CLAUDE.md.
