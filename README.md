# flex-pse

An open-source Pyomo/IDAES platform for industrial energy-flexibility
optimization — model a facility as a time-discretized optimization problem,
parameterize it from plant data, and solve rolling-horizon scheduling problems
against real electricity tariffs and demand-response signals.

## Install

The environment is managed entirely with **conda**.
[`environment.yml`](environment.yml) pins the optimization stack — the Python
version, `pyomo`, `idaes-pse`, `highspy`, and the `scip` solver, which ship
their binaries through conda-forge — and installs the editable package and its
remaining Python dependencies through a `pip:` subsection, so
`conda env create` is the only install command.

The default open-source solver stack behind `flexcore.solvers.get_solver` is
**HiGHS** for LP, **SCIP** for MILP and MINLP, and **IPOPT** for NLP (built from
the HSL-linked binaries installed by `idaes get-extensions` in step 2).

```bash
# 1. Create and activate the environment (installs the stack and the package).
conda env create -f environment.yml
conda activate flex-pse

# 2. Install idaes solver binaries (the HSL-linked IPOPT).
idaes get-extensions

# 3. Enable the git hooks.
pre-commit install
pre-commit install --hook-type pre-push
```

## Development

This project is built milestone-by-milestone. See [`PLAN.md`](PLAN.md) for the
roadmap and [`plan/00_conventions.md`](plan/00_conventions.md) for the rules that govern every change.
