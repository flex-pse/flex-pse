# M14 — Docs completion + example notebooks

**Effort:** 2–3 days · **Depends on:** M13 · **Parallelizable:** no

## Goal

Finish the documentation system: final `conf.py`, the `flexdoc` Sphinx extension
that generates unit-model Variables/Constraints/DoF tables from built models (so
docs cannot drift from code), the unit-model autosummary template, a sweep of all
reference pages onto the generated directives, three executed example notebooks,
and the finalized `docs.yml` workflow. This milestone also lands
**reference-page + flexdoc coverage for every public surface built in
M08/M09/M16** — `NetworkBlock`, the SISO/SIDO/DIDO topology bases, the full
physical unit zoo, the customizable unit-commitment logic modules,
`flexops.design` (M16), the EECO post-hoc `evaluate_cost`/`report_cost`, and the
config-driven `build_model` with its YAML config schema. Exit state:
`sphinx-build -W` is clean in both execution modes and every public unit model
page renders generated tables.

## Read first

- `plan/03_documentation.md` — ALL of it; this milestone implements §1–§6 to completion
- `plan/01_architecture.md` §3.2 (IORegistry / registration API — flexdoc's data source), §3.4 (the unit-model table: **every** class needs a reference page — SISO/SIDO/DIDO bases and the physical zoo: Pump, StorageTank, Separator, Exchanger, ElectrolysisSeparator, ElectrolysisExchanger, ReverseOsmosisSkid, Combustor, BatteryModel, ConstantEnergyIntensityModel)
- `plan/01_architecture.md` §3.3 (`NetworkBlock`/`PlantBlock` composition — reference pages), §3.5 (the customizable unit-commitment logic modules), §3.6 (`report_cost`/post-hoc EECO evaluation; the reported cost is never the objective — R9), §2.3 (config-driven `build_model` + the YAML config schema)
- `plan/01_architecture.md` §3.6 (`flexops.design` — the multi-period design wrapper documented in M16)
- `plan/02_testing_and_ci.md` §3 (docs.yml and nightly.yml specs), §1 (tier markers for the flexdoc unit test)
- `PLAN.md` §2 (the api_freeze script — notebook 01's source material)
- `plan/00_conventions.md` §8 (docs summary rules)

## Files to create or modify

- `docs/conf.py` — final configuration (see spec)
- `docs/_ext/flexdoc.py` — `flexops-unit-tables` and `flexops-config-table` directives
- `docs/_templates/autosummary/unit_model.rst` — per 03 §3
- `docs/reference/**` — sweep every existing reference page onto the directives; remove TODO markers left by M02–M13; add pages for the SISO/SIDO/DIDO bases, the full physical zoo, `NetworkBlock`, the logic modules, `flexops.design` (§4a), and `FlexCosting.evaluate_cost`/`report_cost`
- `docs/reference/flexops/design.rst` — new page for `flexops.design` (`DesignModel`, `merge_for_design`), added to the flexops reference index
- `docs/explanation/config_schema.md` — render pydantic models via `flexops-config-table`; document the YAML config schema driving `build_model`
- `docs/explanation/reported_cost.md` (or a section in `energy_nomenclature.md`) — the reported cost is EECO post-hoc, never the objective (§4b, R9)
- `examples/01_build_a_plant.ipynb`, `examples/02_parameterize_from_data.ipynb`, `examples/03_rolling_horizon.ipynb`
- `.github/workflows/docs.yml` — finalize PR + deploy jobs
- `.github/workflows/nightly.yml` — ensure a notebook-execution step exists (02 §3 requires it; add if missing)
- `src/flexops/tests/docs/test_flexdoc_tables.py` — unit test for table generation

## Specification

### 1. `docs/conf.py` (final)

Per 03 §1:

- Extensions: `sphinx.ext.autodoc`, `sphinx.ext.autosummary`
  (`autosummary_generate = True`), `sphinx.ext.napoleon` (Google style),
  `sphinx.ext.intersphinx`, `myst_nb`, plus the local `flexdoc` (add
  `docs/_ext` to `sys.path` in conf.py).
- Intersphinx mappings to pyomo, idaes, pandas, pydantic (stable/latest docs
  URLs — implementer's choice of exact URLs; verify each resolves during build).
- Theme: `furo` (chosen in M00 — do not revisit).
- `nitpicky = True` with a curated `nitpick_ignore` list; every entry gets a
  one-line comment saying why (typically upstream objects missing from
  intersphinx inventories). Do not blanket-ignore whole modules.
- Notebook execution switch:
  `nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "cache")` — PR docs
  CI sets `NB_EXECUTION_MODE=off`; local/main builds cache (03 §4, §6).
- Notebooks are globbed from `/examples` into the docs tree per the 03 §1
  layout (`docs/examples/`); use whichever include mechanism M00 set up
  (symlink/copy in conf.py is implementer's choice if not already settled).

### 2. `docs/_ext/flexdoc.py`

Two directives, per 03 §2.

```rst
.. flexops-unit-tables:: flexops.unit_models.pump.Pump
```

At docs-build time the directive:

1. Imports the class from its dotted path; constructs it on
   `flexops.testing.dummy_time_block(n=3)` with `SimpleAqueousFlow` defaults
   (the helper is provided by `flexops.testing`; if a unit needs a costing
   package to build, the helper provides a stub — implementer's choice,
   mirroring what `UnitModelTestHarness.configure` defaults do).
2. Reads the unit's `IORegistry`, the `doc=` strings on its Vars/Constraints,
   and their units.
3. Emits three `list-table` nodes:
   - **Variables** — name, index sets, units, IO role, description;
   - **Constraints** — name, description;
   - **Degrees of Freedom** — the registered inputs that must be fixed.

Implementation notes:

- Structure the extension as a thin directive over a pure function
  `collect_unit_tables(cls) -> dict[str, list[list[str]]]` returning the three
  tables as rows of strings — the unit test calls this function directly, no
  Sphinx app needed (implementer's choice, but keep the split: it is what makes
  the extension testable).
- A build-time failure (class won't import, model won't construct, registered
  variable missing a `doc=`) must fail the build with a clear error naming the
  class — under `sphinx-build -W` a warning suffices, but prefer raising. An
  empty table is a silent-drift bug, never acceptable output.
- Descriptions come from component `doc=` strings; 03 §2 already obligates every
  public Var/Constraint to carry one (the harness asserts non-empty for
  registered variables).

```rst
.. flexops-config-table:: flexcore.config.schema.ModelConfig
```

- Renders a pydantic v2 model's fields as one list-table: name, type, default,
  description (from `model_fields` / `FieldInfo.description`). Used by
  `docs/explanation/config_schema.md` and unit-model config sections.

### 3. `docs/_templates/autosummary/unit_model.rst`

Per 03 §3, composes in order: (1) the napoleon-rendered class docstring,
(2) `.. flexops-unit-tables:: {{ fullname }}`, (3) `automethod` entries for
public methods beyond the standard block interface.
`docs/reference/flexops/unit_models/index.rst` autosummary uses this template
for the full v0 unit zoo (architecture §3.4). Coverage is **every public unit
model**, not just the original six:

- **Topology bases** (`flexops/unit_models/base/`): `SISOBlock`, `SIDOBlock`,
  `DIDOBlock` — one reference page each, documenting the ports / per-stream mass
  balance / energy-registration wiring they own.
- **Physical zoo**: `Pump`, `StorageTank`, `Separator`, `Exchanger`,
  `ElectrolysisSeparator`, `ElectrolysisExchanger`, `ReverseOsmosisSkid`,
  `Combustor`, `BatteryModel`, `ConstantEnergyIntensityModel`.

(The old `Electrolyzer` name is gone — R6 renamed it to `Separator`; the
electrolysis units are `ElectrolysisSeparator`/`ElectrolysisExchanger`. Do not
reference `Electrolyzer` anywhere in the docs.) `StorageTank`'s page notes that
its unit-commitment logic is disabled (a tank has no on/off status, §3.4).

### 4. Reference-page sweep

Earlier milestones left hand-written stubs and `TODO(M14)` markers. Sweep every
page under `docs/reference/` so that:

- every unit model renders via the autosummary template (delete hand-written
  variable tables — the generated ones are canonical);
- config-schema pages use `flexops-config-table`;
- `grep -rn "TODO" docs/` returns nothing.

### 4a. Reference coverage for the M08/M09/M16 public surfaces

Beyond the unit zoo, add or complete reference pages (autodoc/autosummary; use
generated directives where a directive fits) for:

- **Composition** (`docs/reference/flexops/core.rst`): `NetworkBlock` (composes
  plants) alongside `PlantBlock` (composes units) — architecture §3.3, R7.
  Document the recursive aggregation (`NetworkBlock` totals = Σ `PlantBlock`
  totals = Σ unit `electrical_work`/`thermal_work`).
- **Logic layer** (`docs/reference/flexops/logic.rst`): the customizable
  unit-commitment modules — `status`, `startup_shutdown`, `dwell`, `delays`,
  `conditional`, `bypass`, and the model-level `degeneracy` pass (architecture
  §3.5). State which pieces are optional (everything except `status`).
- **`flexops.design`** (M16): a reference page for `DesignModel` /
  `merge_for_design` (architecture §3.6) — new page
  `docs/reference/flexops/design.rst`, added to the flexops reference index. If
  M16 is not yet merged when M14 runs, stub the page and mark it `TODO(M16)` —
  but the sweep's "no TODO" rule means M16 must clear it (note the ordering in
  the PR); prefer landing M16 first (it depends only on M09/M07).
- **Costing post-hoc evaluation** (`docs/reference/flexops/costing.rst`):
  document `FlexCosting.evaluate_cost`/`report_cost` — the post-solve EECO
  evaluation on the realized aggregate-power numpy array (architecture §3.6).
- **Config-driven `build_model`** (architecture §2.3): document
  `flexops.build_model(config)` and render the **YAML config schema** — the
  `ModelConfig` tree (`TimeConfig`, `CostingConfig`, `NetworkConfig`/`PlantConfig`,
  `UnitConfig`, `IOVariableSpec`, `SurrogateSpec`) via the
  `.. flexops-config-table::` directive on `docs/explanation/config_schema.md`
  (and cross-referenced from the how-to / getting-started build pages).

### 4b. Explanation note — the reported cost is EECO post-hoc, never the objective

Add a short note (smallest home: a section in `docs/explanation/energy_nomenclature.md`
or a dedicated `docs/explanation/reported_cost.md` — implementer's choice, but it
must be linkable) stating the reporting rule (architecture §6, decision R9): the
user-facing electricity cost is always `FlexCosting.report_cost` — EECO evaluated
post-solve on the realized aggregate-power array — because the in-objective cost
is a convex-relaxed, possibly scalarized proxy. The raw solver objective is never
the reported number; it appears only behind an explicit debug flag. Cross-ref the
M12/M13 `ScheduleResult.reported_cost` / `report_setpoints` surfaces.

### 5. Notebooks (`/examples`)

Rules from 03 §4, non-negotiable: horizons ≤ 2 days at 15 min; fixed seeds; no
network access; **every notebook ends with an assert cell** on a numeric result
(so "execution passed" means something). Keep nightly execution cheap.

- `01_build_a_plant.ipynb` — the `examples/api_freeze.py` walkthrough as
  narrative: TimeBlock → properties → costing → PlantBlock → tank/surrogate/
  battery → `cost_process()` → objective → solve, with matplotlib plots of the
  load shift (electrical_work vs. TOU price). Shrink the horizon to 2 days
  (api_freeze.py itself uses 30 — say so in a note). Tariff/DR/surrogate JSON
  inputs are small files written inline by the notebook or shipped in
  `examples/` (implementer's choice), never fetched. The notebook **may** show
  the config-driven path (`fo.build_model(load_model_config("plant.yaml"))`,
  architecture §2.3) alongside or instead of the imperative build, to illustrate
  that the whole model builds from one config file.
- `02_parameterize_from_data.ipynb` — the M10 constant-EI round-trip as a
  story: synthesize plant data with a fixed seed → `TagMap` aliasing →
  sufficiency validation → regression → emitted `ModelConfig` → rebuild the
  FlexOps model from the config → assert the rebuilt model's behavior matches
  the fit.
- `03_rolling_horizon.ipynb` — M12's small case: tank + TOU over 1–2 days,
  2–4 windows, `SolveSequence.canonical()` or a plain LP sequence,
  `solve_rolling_horizon`, then `extract_setpoints` + `MinHoldSmoother` (M13);
  plot committed trajectory across window boundaries; assert committed cost.

### 6. Workflows

`docs.yml` (final, per 02 §3):

- PR job: `sphinx-build -W --keep-going -b html docs docs/_build` **with
  notebook execution on** (myst-nb `cache` mode; cache the jupyter-cache
  directory in CI keyed on the notebook file hashes so only changed notebooks
  re-run). A broken notebook blocks the merge like any other test. This job is
  a required status check. `NB_EXECUTION_MODE=off` remains for fast local
  iteration only.
- main job: full build with cached notebook execution, then deploy to
  **GitHub Pages** via `actions/upload-pages-artifact` + `actions/deploy-pages`
  (implementer's choice — 02 §3 allows Pages or RTD; record the choice in the
  PR description).

`nightly.yml` (safety net, never a gate — 02 §3): force-execute the notebooks
cache-free (e.g. `NB_EXECUTION_MODE=force sphinx-build ...` or
`jupyter nbconvert --execute` over `examples/*.ipynb` — implementer's choice;
the requirement is that a stale-cache or environment-drift breakage fails
nightly within a day even when PR builds hit warm caches).

## Pitfalls

1. **The docs build passing while tables are empty.** If `flexdoc` swallows a
   construction error and emits nothing, docs drift silently forever — the
   whole point of the extension dies. Fail loudly; keep the unit test that
   asserts real rows.
2. **Notebook horizons creeping up.** A 30-day notebook makes the PR docs gate
   slow and flaky. ≤ 2 days at 15 min = ≤ 192 steps; check before committing.
3. **PR docs CI re-executing unchanged notebooks.** The PR build executes
   notebooks (they gate the merge), but with a warm jupyter-cache only changed
   notebooks should re-run; if every PR re-executes all three, the CI cache key
   is wrong.
4. **`nitpick_ignore` as a dumping ground.** Every ignore is curated + commented.
   If you're ignoring your own project's references, fix the docstring instead.
5. **Building the docs model at import time.** `flexdoc` must construct models
   inside directive `run()`, not at module import — Sphinx imports extensions
   before the environment is ready, and the unit test imports it too.
6. **Missing `doc=` strings discovered late.** Run the full `-W` build early;
   each missing description on a registered variable is a build failure to fix
   in the owning module (tiny diffs, but they touch earlier milestones' files —
   allowed here, this milestone says to sweep).
7. **Notebook outputs committed stale.** With `cache` mode, stale caches hide
   breakage locally; nightly's forced execution is the safety net — make sure
   the step actually runs all three notebooks.
8. **Intersphinx flakiness.** Network fetch of inventories can fail in CI; pin
   the URLs and, if flakiness appears, commit local inventory fallbacks
   (implementer's choice — note it in the PR).

## Tests

The docs build IS the test suite for this milestone:

- `NB_EXECUTION_MODE=off sphinx-build -W --keep-going -b html docs docs/_build` — clean (PR mode).
- `sphinx-build -W -b html docs docs/_build` — clean with executed/cached notebooks (main mode; run locally before merging).

Plus one real test file, `src/flexops/tests/docs/test_flexdoc_tables.py`
(location is implementer's choice — it lives under `flexops` because it
exercises flexops models; load `docs/_ext/flexdoc.py` via
`importlib.util.spec_from_file_location` with a path resolved from the repo
root, and `pytest.skip` with a clear reason if `docs/` is absent, e.g. in an
installed-package run):

- `test_unit_tables_pump` — `@pytest.mark.unit`: call `collect_unit_tables(Pump)`;
  assert the Variables table rows contain the registered variable names
  (`flow_vol`, `electrical_work` at minimum), every row has a non-empty
  description and units string, and the DoF table is non-empty. This is the
  guard against the extension silently emitting empty tables.
- `test_config_table_model_config` — `@pytest.mark.unit`: field rows for
  `flexcore.config.schema.ModelConfig` include `schema_version` with its
  description.

The PR docs build executes the three notebooks (merge gate); nightly
re-executes them cache-free as the drift safety net. Their final assert cells
are the pass/fail criterion in both.

## Documentation tasks

This whole milestone is documentation; specifically also:

- `docs/getting_started/ten_minutes.md` — verify it matches the final
  api_freeze walkthrough and links notebook 01.
- `docs/how_to/build_a_plant.md`, `parameterize_from_data.md`,
  `schedule_rolling_horizon.md` — each becomes a thin wrapper pointing at its
  executed notebook (03 §1).
- `docs/explanation/config_schema.md` — rendered via `flexops-config-table`
  for the whole YAML config tree that drives `build_model` (§2.3): `ModelConfig`,
  `TimeConfig`, `CostingConfig`, `NetworkConfig`/`PlantConfig`, `UnitConfig`,
  `IOVariableSpec`, `SurrogateSpec`.
- `docs/explanation/reported_cost.md` (or the `energy_nomenclature.md` section):
  the reported cost is EECO post-hoc (`report_cost`), never the objective (R9).
- CHANGELOG entry under "Unreleased" (docs system + notebooks are user-visible).

## Definition of Done

- [ ] `docs/conf.py` final: napoleon, autosummary generate, myst_nb, intersphinx (pyomo/idaes/pandas/pydantic), furo, nitpicky + curated ignore list, `NB_EXECUTION_MODE` switch
- [ ] `flexdoc.py` provides `flexops-unit-tables` and `flexops-config-table`; failures are loud, never empty tables
- [ ] `_templates/autosummary/unit_model.rst` in place; **every** public unit model renders generated Variables/Constraints/DoF tables — the SISO/SIDO/DIDO bases and the full physical zoo (Pump, StorageTank, Separator, Exchanger, ElectrolysisSeparator, ElectrolysisExchanger, ReverseOsmosisSkid, Combustor, BatteryModel, ConstantEnergyIntensityModel); no `Electrolyzer` reference anywhere
- [ ] Reference pages exist for `NetworkBlock` (§3.3), the unit-commitment logic modules (§3.5), `flexops.design` (M16), and `FlexCosting.evaluate_cost`/`report_cost` (§3.6)
- [ ] `build_model` documented and the YAML config schema (`ModelConfig` tree) rendered via `flexops-config-table` (§2.3)
- [ ] Explanation note: reported cost is EECO post-hoc, never the objective (R9), linkable and cross-referenced from M12/M13 surfaces
- [ ] Reference sweep complete: no hand-written variable tables, `grep -rn TODO docs/` is empty
- [ ] Three notebooks committed, each ≤ 2-day horizon, fixed seeds, no network, ending in an assert cell; all execute clean
- [ ] `docs.yml` finalized (PR: cached notebook execution ON, required check; main: cached execution + GitHub Pages deploy); nightly cache-free notebook-execution step present
- [ ] `sphinx-build -W --keep-going` clean in BOTH execution modes — zero warnings
- [ ] `test_flexdoc_tables.py` unit tests pass
- [ ] CHANGELOG updated; PR records implementer's-choice decisions (Pages vs RTD, notebook data shipping)
- [ ] plus the generic DoD in CLAUDE.md
