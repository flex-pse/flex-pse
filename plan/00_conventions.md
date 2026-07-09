# 00 — Conventions

Non-negotiable rules for everything written into this repository. Milestone work
orders assume these; they are not repeated there.

## 1. Repository layout

```
flex-pse/
├── pyproject.toml                  # single distribution: "flex-pse"
├── .importlinter                   # import DAG contracts (see §6)
├── .pre-commit-config.yaml         # black, ruff, import-linter
├── CHANGELOG.md                    # Keep a Changelog format, "Unreleased" section on top
├── LICENSE                         # Apache 2.0
├── .github/workflows/              # ci.yml, nightly.yml, docs.yml
├── src/
│   ├── flexcore/                   # shared substrate — never imports the other three
│   │   ├── exceptions.py           # FlexError hierarchy (no compat layer — idaes/pyomo imported directly, R12)
│   │   ├── solvers/                # classify.py, registry.py, facade.py
│   │   ├── config/                 # schema.py (pydantic: Unit/Plant/Network/Time/Costing/ModelConfig),
│   │   │                           # io.py (YAML canonical), schemas/ (exported JSON Schema)
│   │   └── tests/                  # (no econ module — tariff/cost is the external eeco package)
│   ├── flexops/
│   │   ├── core/                   # time_block.py, ops_block.py, plant_block.py,
│   │   │                           # network_block.py, registration.py, build.py (build_model)
│   │   ├── unit_models/
│   │   │   ├── base/               # siso.py, sido.py, dido.py (IO-topology base blocks)
│   │   │   └──                     # pump.py, storage_tank.py, battery.py, separator.py,
│   │   │                           # exchanger.py, electrolysis.py, ro_skid.py, combustor.py,
│   │   │                           # constant_intensity.py
│   │   ├── logic/                  # status.py, startup_shutdown.py, dwell.py, delays.py,
│   │   │                           # conditional.py, degeneracy.py (model-level), bypass.py
│   │   ├── costing/                # flex_costing.py, opex.py (eeco calls collected here by convention)
│   │   │   └── unit_models/        # unit_costing.py (per-unit costing methods, one costing package per unit)
│   │   ├── design/                 # multi-period design wrapper (M16): DesignModel, merge_for_design
│   │   ├── properties/             # simple_aqueous.py
│   │   ├── testing/                # harness.py (public, shipped)
│   │   └── tests/
│   ├── flexparameterize/
│   │   ├── tags.py, validate.py, apply.py, emit.py  # apply.py = the 2-way mutate-in-place path
│   │   ├── regression/             # base.py, constant.py, linear.py
│   │   └── tests/
│   └── flexschedule/
│       ├── horizon.py, sequences.py, setpoints.py, smoothing.py
│       └── tests/
├── examples/                       # myst-nb notebooks + api_freeze.py
└── docs/                           # sphinx (see plan/03_documentation.md)
```

Tests are **colocated** with their package (`src/<pkg>/tests/`), mirroring the
module layout (`tests/core/test_time_block.py` tests `core/time_block.py`).
When a package is later split into its own repo, its tests move with it.

## 2. Naming

- Packages/modules: `snake_case`. Classes: `CapWords`. Functions/variables:
  `snake_case`. Constants: `UPPER_SNAKE`.
- Pyomo model components follow IDAES conventions where one exists
  (`flow_vol`, `pressure`, `temperature`), and this project's energy
  nomenclature otherwise (see `plan/01_architecture.md` §4):
  - `electrical_power[t]` — electrical draw of a unit, **kW** (a power, despite
    the name — the name is the project-wide standard).
  - `thermal_power[t]` — thermal/gas-driven duty of a unit, **kW**.
  - Never introduce variables named bare `power`, `energy`, or `work`.
- Time index is always named `t`, iterating `time_block.time_index`.
- User-facing constructors take **keyword arguments only** (enforce with `*` in
  signatures). ISO-8601 date strings (`"2025-01-01"`) or `datetime` objects; never
  ambiguous `"1-1-2025"`.
- Config file keys: `snake_case`; every persisted config has a top-level
  `schema_version: int`.

## 3. Style & tooling

- Python ≥ 3.11. Target the oldest supported version in code.
- Format with **black** (default settings); lint with **ruff** (rule set pinned in
  `pyproject.toml`; do not inline-silence rules without a comment explaining why).
- **Type hints on all public functions, methods, and class attributes.** Internal
  helpers should have them too unless Pyomo typing makes it hopeless (then annotate
  what you can).
- **Google-style docstrings** on every public module, class, and function.
  A unit-model class docstring must include: one-paragraph model description,
  the governing equations in LaTeX (``.. math::``), a short usage snippet, and
  cross-references to its config options.
- No mutable default arguments. No `print` (use `logging`, logger per module:
  `_log = logging.getLogger(__name__)`).
- Exceptions: raise project exceptions (`FlexConfigError`, `FlexSolverError`,
  `FlexDataError` — defined in `flexcore.exceptions`) with messages that state
  what was wrong **and what the user should do** ("Solver 'ipopt' not found.
  Install it with `idaes get-extensions` or pass solver='highs'.").

## 4. Configuration rules

There are **exactly two** kinds of config, and they never mix. Pick by asking
one question: *does this get saved to a file, or is it only used while building
a model in memory?*

| | **Layer 1 — Persisted config** | **Layer 2 — Runtime options** |
|---|---|---|
| **What it is** | Settings saved to disk and reloaded later | Options a Pyomo/IDAES block takes when it's built |
| **Who writes it** | A user, or FlexParameterize | Our code, at the moment of construction |
| **How it's defined** | pydantic v2 model in `flexcore.config.schema` | Pyomo `ConfigDict` entries |
| **How it's stored** | JSON file, tagged with a `schema_version` | Never stored — lives only in memory |
| **Every field must have** | a `description` (these render into the docs) | `description=` set on the entry |

**How they connect:** persisted config is the source of truth. When a model is
built from config, a validated Layer-1 pydantic object is read *once* at the
boundary and used to populate the Layer-2 `ConfigDict`. Data flows one way:
`JSON file → pydantic (validate) → ConfigDict → built model`.

**Four hard rules:**

1. **Never save a `ConfigDict` to disk.** Runtime options are throwaway; only
   Layer-1 pydantic models get persisted.
2. **Never pass a raw dict through more than one function call without
   validating it.** Turn it into a pydantic model (or `ConfigDict`) first.
3. **Every config key must be documented.** If you can't write a description for
   a key, it doesn't get to exist. No opaque nested JSON blobs (the old
   "FlowsNPC config files" are the anti-pattern to avoid).
4. **Validation errors must name the exact field** that was wrong (e.g. the
   field path), so a user knows what to fix.

## 5. Commits & PRs

- One milestone per PR. PR title: `M07: FlexCosting — tariff-driven operating cost`.
- PR description includes: milestone link, Definition-of-Done checklist copied
  and ticked, "Deviations from spec" section (write "none" if none).
- CHANGELOG entry under "Unreleased" for anything user-visible.
- Keep commits reviewable; a reviewer should be able to read the PR in one sitting.

## 6. Import discipline (the split-later insurance)

Enforced by import-linter in CI (`.importlinter`):

- Layered contract: `flexcore` ← `flexops` ← {`flexparameterize`, `flexschedule`}.
  Lower layers never import higher ones; `flexparameterize` and `flexschedule`
  are mutually independent. **This is the only import-linter contract.**

**No dependency-isolation contracts.** Per decision R12 (architecture §2.1)
`idaes.*`, `pyomo.*`, and `eeco` are imported **directly at point of use** — no
compat layer, no whitelist, no forbidden contract. The project pins exact tested
versions of these in `pyproject.toml` and maintainers bump them manually. By
convention `eeco` calls are collected in `flexops/costing/opex.py` (a thin
wrapper), but this is not enforced.

## 7. Testing (summary — full spec in plan/02_testing_and_ci.md)

- **Test-driven development is the required workflow**: write the milestone's
  tests first (they are the behavioral spec), watch them fail for the right
  reason, then implement. Tests written after the code do not satisfy any
  Definition of Done.
- **Run the full suite locally before every push**: `ruff check . &&
  black --check . && lint-imports && pytest -q` (all tiers). The pre-push hook
  installed by `pre-commit install --hook-type pre-push` runs exactly this;
  never bypass it on a branch intended for merge.
- Exactly one tier marker per test: `@pytest.mark.unit` (< 1 s, no solver),
  `@pytest.mark.component` (< 30 s, HiGHS/IPOPT only),
  `@pytest.mark.integration` (minutes, end-to-end). Collection fails otherwise.
  All three tiers run and must pass on every PR before merge (public repo, free
  CI); the tiers exist to keep the local TDD loop fast, not to defer tests.
- Solver-availability markers (`needs_highs`, `needs_ipopt`, ...) on anything
  that calls a specific solver.
- Every unit model gets a test class subclassing
  `flexops.testing.UnitModelTestHarness` — about 30 lines: a `configure()`
  method plus expected-DoF and expected-solution data.
- Numerical assertions use explicit tolerances (`pytest.approx(x, rel=1e-6)`),
  never exact float equality.
- Deterministic tests only: fixed seeds, no wall-clock dependence, no network.

## 8. Documentation (summary — full spec in plan/03_documentation.md)

- Docs build (`sphinx-build -W`) is a CI gate; a warning is a failure.
- Every public unit model has a reference page using the
  `.. flexops-unit-tables::` directive (auto-generates Variables / Constraints /
  Degrees-of-Freedom tables from the built model).
- How-to content goes in executable myst-nb notebooks under `examples/`;
  narrative design rationale goes in `docs/explanation/`.

## 9. Agent-specific rules

(Also in `CLAUDE.md`, which agents read automatically.)

- Build only the current milestone. No speculative abstractions "for later" —
  the later milestones are already written; trust them.
- If upstream (Pyomo/IDAES) behavior contradicts the milestone spec, prefer the
  spec's *intent*, implement the smallest working variant, and flag the deviation
  in the PR description.
- When a milestone says "copy this signature," copy it exactly — other
  milestones and docs reference these names verbatim.
