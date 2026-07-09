# M00 — Repo scaffold & CI skeleton

**Effort:** 1 day · **Depends on:** — · **Parallelizable:** no

## Goal

Stand up the repository skeleton so that every later milestone drops code into a
structure that already enforces the rules: the four-package src layout, the
import-linter DAG, pytest tier markers with collection-time enforcement, and a
green PR CI pipeline. Nothing here contains domain logic — the deliverable is
that the environment install (`conda env create -f environment.yml`, which
installs the Python version + `pyomo`/`idaes-pse`/solvers and then the editable
package via its `pip:` subsection — see README), `pytest -m unit`,
`lint-imports`, and CI all pass on an essentially empty codebase.

## Read first

- `plan/00_conventions.md` §1 (repository layout — build exactly this tree), §3
  (style & tooling), §6 (import discipline — the layered contract), §7 (testing summary)
- `plan/02_testing_and_ci.md` §1 (test tiers + collection-time enforcement — the
  conftest hook you implement here), §3 (ci.yml spec)
- `plan/01_architecture.md` §1 (package dependency DAG)
- `plan/03_documentation.md` §1 (only the note that the Sphinx theme is picked
  once, in M00 — we pick **furo**; no docs build is set up yet)

## Files to create or modify

- `pyproject.toml` — packaging, deps, extras, pytest/ruff/black config
- `environment.yml` — conda environment for the conda-installed stack (Python
  version, `pyomo`, `idaes-pse`)
- `.importlinter` — the layered import contract from conventions §6
- `.pre-commit-config.yaml` — black, ruff, import-linter hooks
- `conftest.py` (repo root) — tier-marker enforcement + unit-tier solver-block stub
- `CHANGELOG.md` — Keep a Changelog format, empty "Unreleased" section
- `LICENSE` — Apache License 2.0
- `README.md` — stub: one-paragraph description, install command, plan link
- `CLAUDE.md` — agent rules copied from conventions §9 + the generic
  Definition-of-Done checklist that all milestone DoDs reference
- `.github/workflows/ci.yml` — lint job (fast fail) + test matrix
- `src/flexcore/`, `src/flexops/`, `src/flexparameterize/`, `src/flexschedule/` —
  full package tree from conventions §1 as **directories with empty
  `__init__.py` files only** (subpackages: `flexcore/{solvers,config,tests}`,
  `flexops/{core,unit_models,logic,costing,costing/unit_models,properties,testing,tests}`,
  `flexparameterize/{regression,tests}`, `flexschedule/tests`). Do NOT create the
  future module files (`pump.py`, `opex.py`, …) — later milestones own those.
  Also create `src/flexcore/config/schemas/` with a `.gitkeep`.
- One placeholder test per package: `src/<pkg>/tests/test_import.py`
- `src/flexcore/tests/test_marker_enforcement.py` — meta-test via pytester

## Specification

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "flex-pse"
version = "0.0.1.dev0"
description = "Industrial energy-flexibility optimization platform"
requires-python = ">=3.10"
license = { text = "Apache 2.0" }
dependencies = ["pyomo", "idaes-pse", "pandas", "pydantic>=2", "eeco"]
# pyomo / idaes-pse / eeco are left unpinned here; M01 pins them to exact tested
# versions (decision R12). eeco = external tariff/DR/cost engine (flexops.costing,
# M06); verify the exact PyPI distribution name when pinning in M01.

[project.optional-dependencies]
parameterize = ["scikit-learn", "pandas>=2.0"]
solvers = ["highspy"]
dev = ["pytest", "pytest-cov", "import-linter", "ruff", "black", "pre-commit", "diff-cover"]
docs = ["sphinx", "myst-nb", "furo"]

[tool.setuptools.packages.find]
where = ["src"]
```

Register pytest markers under `[tool.pytest.ini_options]` with
`addopts = "--strict-markers"` (unknown markers become errors) and
`testpaths = ["src"]`. Markers, each with a one-line description:
`unit`, `component`, `integration`, `needs_highs`, `needs_ipopt`,
`needs_cbc`, `needs_gurobi` (exact names — `plan/02_testing_and_ci.md` §1).
Pin the ruff rule set in `[tool.ruff.lint]` — `select = ["E", "F", "I", "B", "UP"]`
with google pydocstyle convention configured but `D` rules deferred
(implementer's choice); black with default settings.

### .importlinter (the layered contract from conventions §6)

```ini
[importlinter]
root_packages =
    flexcore
    flexops
    flexparameterize
    flexschedule

[importlinter:contract:layers]
name = flexcore <- flexops <- {flexparameterize, flexschedule}
type = layers
layers =
    flexparameterize | flexschedule
    flexops
    flexcore
```

Note: this is the **only** import-linter contract. There are no
dependency-isolation (`forbidden`) contracts — `idaes`/`pyomo`/`eeco` are
imported directly at point of use and pinned in `pyproject.toml` (decision R12).
Verify `lint-imports` passes both now and with a deliberate violation
(temporarily add `import flexops` to `flexcore/__init__.py`, confirm the layered
contract fails, revert).

### Root conftest.py (the CI-credit guard, 02_testing_and_ci.md §1)

```python
TIER_MARKERS = {"unit", "component", "integration"}

def pytest_collection_modifyitems(config, items):
    errors = []
    for item in items:
        tiers = TIER_MARKERS & {m.name for m in item.iter_markers()}
        if len(tiers) != 1:
            errors.append(
                f"{item.nodeid}: needs exactly one tier marker "
                f"(unit/component/integration), got {sorted(tiers) or 'none'}"
            )
    if errors:
        raise pytest.UsageError("\n".join(errors))
```

Also add an autouse fixture `_no_solver_in_unit_tier(request, monkeypatch)` that
returns immediately unless the test is `unit`-marked, and otherwise will
monkeypatch `flexcore.solvers.facade` to raise on any solver invocation — **in
M00 this body is a stub with a `# TODO(M05): make real when the facade exists`
comment**, because `flexcore.solvers` does not exist yet. Same for the
`needs_*` skip logic (consults `flexcore.solvers.registry` from M05 on): leave a
documented no-op stub. Add `pytest_plugins = ["pytester"]` for the meta-test.

### CI workflow (.github/workflows/ci.yml)

- Trigger: `pull_request` + `push` to `main`. Add a `concurrency` group that
  cancels superseded runs.
- Job `lint` (fast fail): checkout, python 3.12, `pip install -e ".[dev]"`,
  then `ruff check .`, `black --check .`, `lint-imports`.
- Job `fast-tests` (`needs: lint`): matrix `python-version: ["3.10", "3.12"]`;
  `pip install -e ".[dev]"`; `pytest -m "unit or component" --cov --cov-report=xml`.
- Job `integration` (`needs: lint`, parallel with `fast-tests`):
  `pytest -m integration --cov --cov-report=xml`. The integration set is empty
  until M12, but the job exists from day one so "all tests gate every merge"
  (02 §3) is structural, not aspirational — handle pytest's exit code 5
  (no tests collected) as success.
  Solver installation/caching (highspy wheel, `idaes get-extensions`) and the
  diff-cover gate are added when the first solving tests appear (M04/M05) — leave
  a `# TODO(M05)` comment rather than dead YAML.
- Pre-push hook in `.pre-commit-config.yaml` (a local hook with
  `stages: [pre-push]`): `ruff check .`, `black --check .`, `lint-imports`,
  `pytest -q` (all tiers) — the "full suite locally before every push" TDD rule
  from 02 §1a. Document `pre-commit install --hook-type pre-push` as a dev-setup
  step in the README.

### Placeholder tests

Each package gets `src/<pkg>/tests/test_import.py`:

```python
import pytest

@pytest.mark.unit
def test_import():
    import flexcore  # noqa: F401
```

(and correspondingly for `flexops`, `flexparameterize`, `flexschedule`).

## Pitfalls

1. **Forgetting `__init__.py` in a subpackage** — setuptools find will silently
   skip it and a later milestone's import fails confusingly. After scaffolding,
   run `python -c "import flexops.core, flexcore.solvers, flexops.costing"` etc.
2. **The marker-enforcement hook killing pytester's inner runs** — the meta-test
   spawns an inner pytest session; make sure the inner session gets its own
   conftest (see Tests) and the outer hook only sees outer items.
3. **`--strict-markers` without registering `needs_*`** — the availability
   markers are used from M04 on but must be registered *now* or their first use
   errors collection.
4. **Putting tests in a top-level `tests/`** — tests are colocated under
   `src/<pkg>/tests/` (conventions §1) so they move with a future repo split.
5. **Pinning happens in M01, not here** — M00 lists `pyomo`/`idaes-pse`/`eeco` as
   unpinned dependencies; M01 pins them to exact tested versions (decision R12).

## Tests

All in this milestone are `@pytest.mark.unit`:

- `src/flexcore/tests/test_import.py::test_import` — `import flexcore` works.
- `src/flexops/tests/test_import.py::test_import` — same for flexops.
- `src/flexparameterize/tests/test_import.py::test_import` — same.
- `src/flexschedule/tests/test_import.py::test_import` — same.
- `src/flexcore/tests/test_marker_enforcement.py::test_unmarked_test_fails_collection`
  — meta-test: uses the `pytester` fixture; installs the real enforcement hook
  into the inner session with
  `pytester.makeconftest((request.config.rootpath / "conftest.py").read_text())`
  (strip the `pytest_plugins` line if pytester objects), then
  `pytester.makepyfile("def test_nothing(): pass")` (deliberately unmarked, defined
  in-memory), runs `pytester.runpytest()`, and asserts the run errored with the
  "exactly one tier marker" message in output.
- `...::test_double_marked_test_fails_collection` — same setup, inner test
  carrying both `@pytest.mark.unit` and `@pytest.mark.component`; collection errors.

## Documentation tasks

- `README.md` stub: what flex-pse is (one paragraph, lift from PLAN.md §2),
  install instructions (`conda env create -f environment.yml`, which installs
  both the optimization stack and the editable package via its `pip:`
  subsection), pointer to `PLAN.md` and `plan/00_conventions.md`.
- `CHANGELOG.md` initialized ("Unreleased" heading; entry: project scaffold).
- `CLAUDE.md`: conventions §9 agent rules verbatim + a "Generic Definition of
  Done" checklist (all tests green, lint/black/import-linter clean, CHANGELOG
  updated, PR description per conventions §5).
- No Sphinx build yet. Record the theme decision (furo) by putting it in the
  `docs` extra — `docs/` itself starts in M02.

## Definition of Done

- [ ] Fresh conda env from `environment.yml` succeeds (its `pip:` subsection installs the editable package and dev deps)
- [ ] `pytest -m unit` green (4 placeholder import tests + 2 meta-tests pass)
- [ ] `pytest -m "unit or component"` green (component set is empty; collection succeeds)
- [ ] A deliberately unmarked test fails collection with an actionable message
- [ ] `lint-imports` passes; a deliberate `import flexops` in `flexcore` makes the layered contract fail (verified, then reverted)
- [ ] `pre-commit run --all-files` clean; pre-push hook installed and runs the full suite
- [ ] CI green on the PR: lint, fast-tests matrix (py310, py312), and integration jobs all required
- [ ] CHANGELOG, LICENSE, README, CLAUDE.md present
- [ ] plus the generic DoD in CLAUDE.md
