# M15 — Hardening + 0.1.0 release

**Effort:** 2 days · **Depends on:** M14 · **Parallelizable:** no

## Goal

Ship 0.1.0. Single-source the version, run the upstream canary and clear (or pin
and file) anything red, finalize CHANGELOG and README, turn on gentle coverage
floors, and prove the packaged artifact works: build → TestPyPI → fresh-venv
install → both the imperative `examples/api_freeze.py` **and** the config-driven
`examples/api_freeze_config.yaml` path run. That dual step is THE release gate
(the config-driven path exercises the versioned schema — YAML canonical, pydantic
authority, exported JSON Schema — which is the config-driven-everything promise,
architecture §2.3 / R3). Also audit repo-split readiness (import DAG + per-package
test isolation) so the "splittable monorepo" promise is verified, not assumed.

## Read first

- `PLAN.md` §2 (what we are building — README source material; the api_freeze script), §4 (post-v0 backlog — release notes link here)
- `plan/01_architecture.md` §2.3 (R3: versioned schema — YAML canonical, pydantic authority, exported JSON Schema; config-driven `build_model` — the second release-gate path)
- `plan/01_architecture.md` §3.6 (M16 `flexops.design` — decide "what works" vs. "deferred" in the release notes based on whether M16 merged)
- `plan/00_conventions.md` §1 (single distribution `flex-pse`; repo layout), §5 (CHANGELOG: Keep a Changelog, "Unreleased" on top), §6 (import discipline — the split-later insurance this milestone audits)
- `plan/02_testing_and_ci.md` §3 (PR-CI per-subpackage coverage reporting; ci/nightly/docs workflows — there is no upstream-canary workflow, R12), §4 (coverage policy: "M15 turns on gentle floors")
- `plan/03_documentation.md` §1 (installation page)

## Files to create or modify

- `pyproject.toml` — version `0.1.0`; verify `[solvers]` extra exists (add if missing: at minimum `highspy`); packaging metadata complete (classifiers, urls, license); ensure package-data ships `config/schemas/` JSON Schema exports and `examples/api_freeze_config.yaml`
- `examples/api_freeze_config.yaml` — the versioned YAML config that `build_model` turns into the same model as `api_freeze.py` (add if not already present from M09); part of the release gate
- `src/flexcore/__init__.py`, `src/flexops/__init__.py`, `src/flexparameterize/__init__.py`, `src/flexschedule/__init__.py` — `__version__` from shared distribution metadata
- `src/<pkg>/tests/test_version.py` × 4 — one per package (see Tests)
- `CHANGELOG.md` — "Unreleased" → `0.1.0` with release date; fresh empty "Unreleased" section on top
- `README.md` — filled from PLAN.md §2
- `.github/workflows/ci.yml` — per-subpackage coverage floors (PR gate)
- `docs/getting_started/installation.md` — verified against the TestPyPI artifact
- `docs/` release-notes page (e.g. `docs/changelog.md` including CHANGELOG — implementer's choice of mechanism)

## Specification

### 1. Version single-sourcing

The distribution name is **flex-pse**; the import packages are `flexcore`,
`flexops`, `flexparameterize`, `flexschedule` (conventions §1). Version lives
ONCE, in `pyproject.toml` (`[project] version = "0.1.0"`). Each of the four
top-level packages exposes it via metadata lookup:

```python
from importlib.metadata import version as _dist_version

__version__ = _dist_version("flex-pse")   # distribution name, NOT the package name
```

Verify all four expose `__version__ == "0.1.0"`. Do not hardcode the string
anywhere else; `docs/conf.py` should read it the same way if it needs one.

### 2. Dependency pins + hardening

- Per decision **R12** there is **no upstream canary and no `flexcore/compat/`
  layer**: `idaes-pse`, `pyomo`, and `eeco` are imported directly at point of use
  and pinned to exact tested versions in `pyproject.toml`, bumped manually
  (architecture §2.1). Before release, confirm the pins are exactly the versions
  the suite actually passed against.
- Run the full local gauntlet once against the pinned stack: `pytest -q` (all
  tiers, needs solvers) and both docs build modes (03 §6). Any failure is fixed,
  or the offending pin is moved to a known-good version and the change noted in
  the PR — never release on a combination the suite has not passed.
- If a maintainer wants to test newer upstream versions, do it by bumping the
  pins locally and re-running the full suite (R12's manual-bump cycle); do **not**
  add an automated canary or a compat shim.

### 3. CHANGELOG + release notes

- Convert "Unreleased" to `## [0.1.0] - <release date>`; add a new empty
  "Unreleased" above it (Keep a Changelog format, conventions §5).
- Write **honest** release notes at the top of the 0.1.0 section: what works
  (build a plant — imperative and config-driven via `build_model`, tariff/DR
  costing with EECO post-hoc reported cost, on/off + dwell + bypass logic,
  two-way parameterize round-trip + in-place apply, rolling-horizon scheduling,
  set-point extraction), and what is explicitly deferred — link `PLAN.md` §4
  (parallel-train replication, scenario sweeps, forecaster interface, etc.).
- **M16 (design-mode multi-period wrapper, `flexops.design`):** if M16 is merged,
  list it under "what works" (size across N representative months with
  equality-linked sizing vars). If M16 is **not** merged at release time, list
  multi-period design under **deferred** (single-period `set_design_mode` still
  works; the multi-period wrapper follows) and note it in the notes — do not
  claim it. Users trust honest notes; do not oversell v0.
- Surface the changelog in docs as a release-notes page (myst include of
  `CHANGELOG.md` is the smallest mechanism — implementer's choice).

### 4. Coverage floors (PR CI)

Per 02 §4, `ci.yml` already reports coverage **per subpackage** on every PR.
Turn on gentle floors as part of the PR gate: from the latest main-branch CI
run, take each subpackage's actual percentage and set its floor to
**actual − 2 %, rounded down** (implementer picks the exact numbers; record
them in the PR). Mechanism (implementer's choice): either one
`coverage report --fail-under=<floor>` invocation per subpackage over the
combined unit+component+integration data, or per-package runs — whatever fits
the existing ci.yml jobs with the least machinery. Keep the exclusion of
02 §4 (`flexops/testing/`; there is no `flexcore/compat/` — R12).

### 5. README.md

Fill from PLAN.md §2, in order: one-paragraph "what it is" (the three tools +
flexcore, the FlexOps/FlexParameterize/FlexSchedule table condensed to prose or
a small table); install (`pip install flex-pse[solvers]`, plus the IPOPT note
`idaes get-extensions`); a ~10-line example distilled from
`examples/api_freeze.py` (TimeBlock → tank → costing → objective); a link to
the hosted docs; license (Apache 2.0). Keep it under a screen and a half.

### 6. Packaging + THE release gate (procedural — run exactly this)

```bash
# 0. clean tree on the release branch, all CI green
git status                                    # must be clean

# 1. build
python -m pip install --upgrade build twine
rm -rf dist/
python -m build                               # sdist + wheel
twine check dist/*                            # must PASS both

# 2. TestPyPI upload (needs a TestPyPI token configured)
twine upload --repository testpypi dist/*

# 3. THE GATE — fresh venv, install from TestPyPI, run the API-freeze example
python -m venv /tmp/flexpse-gate && source /tmp/flexpse-gate/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            "flex-pse[solvers]"
# --extra-index-url: dependencies (pyomo, idaes-pse, pandas, ...) live on real
# PyPI; only flex-pse itself comes from TestPyPI.
python -c "import flexcore, flexops, flexparameterize, flexschedule as fs; \
           print(flexcore.__version__, flexops.__version__)"
python examples/api_freeze.py                 # imperative path: run top-to-bottom and solve
# config-driven path: build the SAME model from the versioned YAML config and solve
python -c "import flexops as fo; \
           m = fo.build_model(fo.load_model_config('examples/api_freeze_config.yaml')); \
           fo.solve_and_report(m)"   # (invocation is implementer's choice; it must build from YAML and solve)
deactivate
```

Both `examples/api_freeze.py` (imperative) **and**
`examples/api_freeze_config.yaml` (the config-driven `build_model` path) must run
from the freshly installed wheel — they build the same model two ways and are the
readable illustration of the config-driven-everything promise (architecture
§2.3 / R3). The YAML config validates against the exported JSON Schema in
`config/schemas/`, so this path also proves those schema files shipped in the
wheel.

If either path fails for a packaging reason (missing data files — tariff/DR JSON
fixtures, `config/schemas/` JSON Schema exports, the `api_freeze_config.yaml`
itself, `py.typed` markers), fix `pyproject.toml` package-data, rebuild, bump
nothing (TestPyPI allows post suffixes like `0.1.0.post1` for retries —
implementer's choice), and rerun the whole gate from step 1. The gate passes only
when the freshly installed artifact executes **both** `api_freeze.py` and the
`api_freeze_config.yaml` path unmodified.

### 7. Tag + GitHub release checklist

```bash
git tag -a v0.1.0 -m "flex-pse 0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "flex-pse 0.1.0" --notes-file <notes> dist/*
```

Release body = the CHANGELOG 0.1.0 section. Attach the sdist + wheel.
(Real-PyPI publication timing is a maintainer decision — the milestone's gate
is TestPyPI; note in the PR whether real PyPI was pushed.)

### 8. Repo-split readiness audit

The split-later insurance (conventions §6, architecture §1) gets verified:

```bash
lint-imports                                          # all contracts pass
pytest src/flexcore  -q                               # each package's FULL suite
pytest src/flexops   -q                               #   passes in ISOLATION
pytest src/flexparameterize -q
pytest src/flexschedule     -q
```

Record all five results (pass/fail + counts) in the PR description. A failure
means hidden coupling (e.g. a test importing a sibling package's fixtures) —
fix it now; it is exactly the debt the audit exists to catch.

## Pitfalls

1. **`version("flex_pse")` / `version("flexops")`.** The metadata lookup takes
   the *distribution* name `flex-pse`. The four-package `test_version.py` tests
   exist to catch this.
2. **Missing package data in the wheel.** JSON Schema exports
   (`flexcore/config/schemas/`), example tariff files and the
   `examples/api_freeze_config.yaml` the notebooks/examples need — sdist-only
   inclusion (MANIFEST) is not enough; the wheel is what pip installs. The
   fresh-venv gate (both paths) catches it; don't skip either path.
3. **Running the gate from the repo checkout directory.** `python` resolves
   `import flexops` from `src/` or a stale editable install instead of the
   wheel. Use a fresh venv AND verify with
   `python -c "import flexops; print(flexops.__file__)"` that the path is
   site-packages.
4. **TestPyPI resolving dependencies.** Without `--extra-index-url` to real
   PyPI the install fails or, worse, picks squatted TestPyPI packages. Use the
   exact command above.
5. **Coverage floors set aspirationally.** Floors are actuals − 2 %, not round
   numbers you wish were true — an aspirational floor makes every PR
   permanently red and gets ignored (02 §4's rationale).
6. **Editing CHANGELOG history.** Only convert "Unreleased" and add the date;
   never rewrite released entries.
7. **Releasing on an untested upstream combination.** The pins in
   `pyproject.toml` must be exactly the `idaes-pse`/`pyomo`/`eeco` versions the
   full suite passed against (R12 — there is no canary to catch drift for you);
   bumping a pin without re-running the suite ships breakage to every new install.
8. **Forgetting the fresh "Unreleased" section.** The next PR after release
   needs somewhere to write; conventions §5 requires it on top.

## Tests

No new test code beyond version tests — the release gate is procedural (§6
above is the checklist; paste it, with outputs, into the PR description).

- `src/flexcore/tests/test_version.py` — `@pytest.mark.unit`:
  `test_version_matches_distribution` asserts
  `flexcore.__version__ == importlib.metadata.version("flex-pse")` and it
  matches `^\d+\.\d+\.\d+`. Mirror the same ~8-line test in
  `src/flexops/tests/`, `src/flexparameterize/tests/`,
  `src/flexschedule/tests/` (one per package rather than one central test,
  because no single package may import all four — flexparameterize and
  flexschedule are mutually independent, conventions §6; the duplication is
  the point: each file moves with its package at split time).
- Existing suites: the full suite (all tiers) green on PR CI; upstream canary
  green or pinned+issued.
- The repo-split audit commands (§8) — results recorded in the PR.

## Documentation tasks

- Release-notes page in docs (CHANGELOG include) linked from the docs index.
- `docs/getting_started/installation.md` — walk it verbatim against the
  TestPyPI artifact in the fresh venv (install command, solver setup via
  `idaes get-extensions` + `highspy`, verify-install snippet); fix any drift.
- README.md per §5.
- CHANGELOG 0.1.0 section per §3.

## Definition of Done

- [ ] `pyproject.toml` version `0.1.0`; all four packages expose `__version__` via `importlib.metadata` lookup of `flex-pse`; four `test_version.py` unit tests pass
- [ ] Dependency pins (`idaes-pse`/`pyomo`/`eeco`) confirmed as the exact versions the full suite passed against (R12 — no canary, no `flexcore/compat/`); full suite green on the pinned stack
- [ ] CHANGELOG converted to 0.1.0 with date, honest release notes linking PLAN.md §4; fresh "Unreleased" section on top
- [ ] PR-CI per-subpackage coverage floors on (actuals − 2 %), values recorded in the PR
- [ ] `python -m build` + `twine check` pass; artifact uploaded to TestPyPI
- [ ] THE GATE: fresh venv, `pip install` from TestPyPI with `[solvers]`, and **both** `examples/api_freeze.py` (imperative) **and** the config-driven `examples/api_freeze_config.yaml` `build_model` path run top-to-bottom and solve
- [ ] README.md filled from PLAN.md §2 (what/install/10-line example/docs link/license)
- [ ] `git tag v0.1.0` pushed; GitHub release created with notes + artifacts
- [ ] Repo-split audit: `lint-imports` clean and each package's test suite passes in isolation; all five results recorded in the PR
- [ ] Installation docs verified against the TestPyPI artifact; release-notes page builds with `sphinx-build -W`
- [ ] plus the generic DoD in CLAUDE.md
