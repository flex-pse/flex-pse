# CLAUDE.md — agent instructions for the flex-pse repository

You are implementing **flex-pse**, an open-source Pyomo/IDAES platform for
industrial energy-flexibility optimization. The build is organized as strictly
ordered milestones. Your job in any one session is exactly **one milestone**.

## Session start checklist

1. Read `PLAN.md` (§1 "How to use this plan" and the milestone index).
2. Read `plan/00_conventions.md` in full. These rules are non-negotiable.
3. Identify the current milestone: the lowest-numbered file in
   `plan/milestones/` whose Definition of Done is not yet satisfied by the code
   on the current branch. If unsure, ask; do not guess ahead.
4. Read that one milestone file in full before writing any code.
5. Skim `plan/01_architecture.md` sections referenced by the milestone.

## Operating rules

- **One milestone per branch/PR.** Never mix milestones. Never build ahead of the current milestone "while you're in there."
- **Do not refactor previous milestones** unless the current milestone file
  explicitly instructs it.
- **Work test-first.** The milestone's Tests section is the behavioral spec:
  write those tests before the implementation, run them, and confirm they fail
  for the right reason (missing class → `ImportError`, not a test typo). Then
  implement until green. See `plan/02_testing_and_ci.md` §1a.
- **Every test you write carries exactly one tier marker** (`unit`,
  `component`, or `integration`) — collection fails otherwise. `unit` tests must
  not invoke a solver. All tiers run on every PR and gate the merge; the tiers
  exist to keep your local loop fast. See `plan/02_testing_and_ci.md`.
- **Local loop, then full suite before any push** (from the repo root):
  ```bash
  pytest -m "unit" -x -q                # sub-second inner TDD loop
  pytest -m "unit or component" -q      # after each work unit
  # before EVERY push (the pre-push hook runs the same):
  ruff check . && black --check .
  lint-imports
  pytest -q                             # ALL tiers, must be green
  ```
  Never push with a red suite and never bypass the pre-push hook
  (`--no-verify`) on a branch intended for merge.
- **Docs are part of Done.** If you added or changed public API, update the
  corresponding `docs/reference/` page and confirm `sphinx-build -W` passes
  (notebook execution off): `sphinx-build -W --keep-going -b html docs docs/_build`.
- **Import `idaes.*`/`pyomo.*`/`eeco` directly at point of use.** There is no
  compat layer or dependency-isolation contract (decision R12); exact versions
  are pinned in `pyproject.toml` and bumped manually. By convention `eeco` calls
  are collected in `flexops/costing/opex.py`, but that is not enforced. The
  only import-linter contract is the package DAG below.
- **Respect the package DAG**: `flexcore` imports no sibling packages;
  `flexops` imports only `flexcore`; `flexparameterize` and `flexschedule`
  import `flexcore`/`flexops` but never each other.
- **Config over cleverness.** Anything a user configures goes through a
  documented pydantic model in `flexcore.config` (persisted) or a declared
  Pyomo ConfigDict entry (runtime). No bare `**kwargs`, no undocumented dict keys.
- **Never delete Pyomo components** (blocks, Vars, Params, constraints) from a
  built model — anything else that captured the old component keeps a stale
  reference. Update models in place: `set_value` on mutable Params
  (`OpsBlockData.update_parameters`), add new Vars/constraints, or
  `deactivate()` constraints (conventions §9).
- If the milestone spec conflicts with reality (upstream API changed, spec
  ambiguous), make the smallest choice consistent with
  `plan/01_architecture.md`, and record the deviation prominently in your PR
  description under a "Deviations from spec" heading.
- **Code simplicity.**: Write minimal code to accomplish the task such that a human can easily parse and verify all functionality. Keep all APIs simple and explicit. Do not use redundant functions or classes within this codebase. 

## Definition of Done (applies to every milestone, in addition to the
milestone-specific checklist)

- [ ] Tests were written before the implementation (commits show it).
- [ ] All new code has type hints on public functions and Google-style docstrings.
- [ ] `pytest -q` (all tiers) passes locally.
- [ ] `ruff`, `black --check`, and `lint-imports` pass.
- [ ] Docs build clean with `-W`; new public API is documented.
- [ ] CHANGELOG.md has an entry under "Unreleased".
- [ ] The milestone file's own Definition of Done checklist is fully satisfied.
