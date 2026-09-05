# Installation

```bash
pip install "flex-pse[solvers]"
```

This installs `flexcore`, `flexops`, `flexparameterize`, and `flexschedule`
(the last is an unpopulated scaffold in 0.1.0 — see the
[release notes](../changelog.md)), plus their required dependencies
(`idaes-pse`, `pyomo`, `eeco`, `pandas`, `pydantic`). The `[solvers]` extra
adds [HiGHS](https://highs.dev/) (`highspy`), a self-contained wheel covering
LP and MILP problems.

## Solver setup: IPOPT

NLP problems (a design-mode solve with a `Tank`, for instance) need IPOPT,
which does not ship as a plain wheel. After installing flex-pse, fetch the
HSL-linked binaries `flexcore.solvers.get_solver` prefers:

```bash
idaes get-extensions
```

This is a one-time step per environment; `idaes-pse` is already a core
dependency, so no separate install is needed to run it.

## Verify the install

```bash
python -c "
import flexcore, flexops, flexparameterize, flexschedule
print(flexcore.__version__, flexops.__file__)
"
```

The second line should print a path under your environment's `site-packages`
(not a local repo checkout) — if you are developing flex-pse itself from a
cloned repository, see the [`README`](https://github.com/flex-pse/flex-pse#development)
for the conda-based contributor setup instead.
