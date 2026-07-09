# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Install pipeline is now conda-only: `uv` is removed and `environment.yml` installs the editable package and its Python dependencies (including the pip-only `eeco`) through a `pip:` subsection, so `conda env create -f environment.yml` is the single install command. README, `PLAN.md`, and the M00 milestone spec updated accordingly.

### Added

- Project scaffold: four-package `src/` layout (`flexcore`, `flexops`, `flexparameterize`, `flexschedule`), import-linter DAG contract, pytest tier markers with collection-time enforcement, and CI skeleton (M00).
- Exception hierarchy; pinned idaes-pse/pyomo versions (M01).
- Added `TimeBlock`: the discrete-time substrate (ordered integer `time_index` set, `time` Param of elapsed `i*dt` in the user's units, unit-carrying `dt`, datetime↔index utilities, rolling-horizon initial-state registry and window metadata); a configurable `max_length` (dateutil `relativedelta`, default one calendar month) bounding the horizon; minimal Sphinx docs skeleton (M02).
- CI `standard-install` job (committed but commented out, pending the repo going public): remote-only install from the git ref (`pip install "git+<repo>@<ref>"`, no checkout) matrixed over Python 3.11–3.14 that imports every subpackage from a scratch dir, catching subpackages missing from the built distribution — or files left uncommitted — that the editable dev install would mask (M00).
- Added `flexcore.solvers`: a model classifier (`classify`/`ProblemClass`), a capability-matrix registry with cached availability probing (`available_solvers`), and the `get_solver`/`SolverFacade` facade that selects a solver by problem class and errors loudly (decision R5) when no capable solver is installed. Default open-source stack: HiGHS for LP, **SCIP for MILP and MINLP** (preferred over HiGHS for MILP; added to `environment.yml`), and IPOPT for NLP — built from idaes (HSL `ma27` linear solver) when idaes is importable, falling back to stock `SolverFactory` IPOPT (MUMPS) otherwise. When no MINLP-capable solver is installed, MINLP still raises pointing at `flexschedule.SolveSequence`. SCIP's dynamic-library path is isolated at invocation so it does not clash with idaes's bundled `libipopt`. The root `conftest.py` now forbids solver invocation and network access under `-m unit` and skips `needs_*`-marked tests via the registry; adds a `flexcore` reference page and a relaxation-policies explanation page (M05).
