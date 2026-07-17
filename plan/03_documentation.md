# 03 — Documentation

Documentation is a CI-gated deliverable of every milestone, not an afterthought.
The docs stack is Sphinx; the differentiating piece is the `flexdoc` extension,
which generates unit-model reference tables **from the built model**, so docs
cannot drift from code.

## 1. Layout (diátaxis)

```
docs/
├── conf.py                     # napoleon (Google style), autosummary, myst_nb,
│                               # intersphinx → pyomo / idaes / pandas / pydantic
├── index.rst
├── getting_started/
│   ├── installation.md         # includes SOLVER installation (idaes get-extensions, highspy)
│   └── ten_minutes.md          # the api_freeze.py walk-through
├── how_to/                     # task guides, mostly thin wrappers pointing at executed notebooks
│   ├── build_a_plant.md
│   ├── parameterize_from_data.md
│   └── schedule_rolling_horizon.md
├── explanation/                # narrative design rationale
│   ├── time_and_dynamics.md    # R2: why discrete TimeBlock, never dynamic=True
│   ├── relaxation_policies.md  # R5: explicit SolveSequence philosophy
│   ├── energy_nomenclature.md  # power_electrical / power_thermal standard
│   └── config_schema.md        # rendered JSON Schema of ModelConfig et al.
├── reference/
│   ├── flexops/
│   │   ├── unit_models/index.rst    # autosummary using the unit_model template
│   │   ├── core.rst / logic.rst / costing.rst / properties.rst / testing.rst
│   ├── flexparameterize/index.rst
│   ├── flexschedule/index.rst
│   └── flexcore/index.rst
├── _templates/autosummary/
│   └── unit_model.rst          # custom template (see §3)
├── _ext/
│   └── flexdoc.py              # custom directives (see §2)
└── examples/                   # myst-nb notebooks, globbed from /examples
```

`conf.py` essentials: `sphinx.ext.autodoc`, `autosummary` (generate on),
`napoleon` (Google style), `sphinx.ext.intersphinx`, `myst_nb`, `furo` theme
(or another maintained theme — pick once in M00 and stop thinking about it),
`nitpicky = True` with a curated `nitpick_ignore` list.

## 2. The flexdoc extension (`docs/_ext/flexdoc.py`)

Provides:

```rst
.. flexops-unit-tables:: flexops.unit_models.pump.Pump
```

At docs-build time the directive:

1. Imports the class, constructs it on a **3-step dummy TimeBlock** with
   `SimpleAqueousFlow` defaults (helper provided by `flexops.testing`).
2. Reads the unit's `IORegistry`, component docstrings (`doc=` on Vars/
   Constraints), and units.
3. Emits three list-tables:
   - **Variables** — name, index sets, units, IO role, description;
   - **Constraints** — name, description;
   - **Degrees of Freedom** — the registered inputs that must be fixed.

WaterTAP hand-writes these tables and they drift; because OpsBlock *registers*
IO variables and parameters, we generate them. Consequence for implementers:
**every public Var/Constraint gets a `doc=` string** — it is user-facing
documentation, and the harness's `test_io_registration` asserts it is non-empty
for registered variables.

A second small directive, `.. flexops-config-table::`, renders a pydantic
model's fields (name, type, default, description) for the config-schema pages.

## 3. The unit-model autosummary template

`_templates/autosummary/unit_model.rst` composes, in order:

1. Napoleon-rendered class docstring — one-paragraph model description,
   governing equations as `.. math::`, a short usage snippet.
2. `.. flexops-unit-tables:: {{ fullname }}`
3. `automethod` entries for public methods beyond the standard block interface.

Every unit-model milestone's Definition of Done includes "reference page exists
and builds with `-W`" — a mechanical check, because the tables are generated.

## 4. Notebooks

- Live in `/examples`, authored as myst-nb-compatible notebooks; docs glob them.
- `nb_execution_mode = "cache"` everywhere; **the docs PR build executes the
  notebooks** (jupyter-cache directory cached in CI, so only changed notebooks
  re-run) — a broken notebook blocks the merge like any other test. The
  `NB_EXECUTION_MODE=off` env switch exists for fast local docs iteration only.
  Nightly re-executes everything cache-free as a flake/drift safety net.
- Notebook rules: small horizons (≤ 2 days at 15 min) so PR execution stays
  cheap; fixed seeds; no network access; every notebook ends by asserting a
  numeric result (so "execution passed" means something).
- v0 notebook set (M14): `01_build_a_plant`, `02_parameterize_from_data`,
  `03_rolling_horizon`.

## 5. Docstring standard (enforced by review, sampled by ruff pydocstyle rules)

- Google style everywhere.
- Unit-model class docstrings additionally include: assumptions and valid range;
  what is/isn't in the default formulation (e.g., "no on/off binary unless
  `include_onoff=True`"); cross-reference to its config options and, where
  relevant, to `explanation/` pages.
- Public function docstrings: Args/Returns/Raises, plus an `Example:` block for
  anything a user calls directly.

## 6. Build commands

```bash
# fast local iteration (skips notebook execution — NOT what CI runs):
NB_EXECUTION_MODE=off sphinx-build -W --keep-going -b html docs docs/_build
# full build with executed notebooks — matches the PR gate; run before pushing docs changes:
sphinx-build -W -b html docs docs/_build
```
