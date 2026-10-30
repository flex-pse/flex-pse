# Contributing to flex-pse

Thanks for your interest in contributing! This project welcomes issues, pull requests, and discussion from anyone. To keep things maintainable and trustworthy for everyone, please read through these guidelines before submitting.

## Table of Contents

- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Human Readability Standards](#human-readability-standards)
- [Contributor Responsibility](#contributor-responsibility)
- [Use of AI Tools](#use-of-ai-tools)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Building a Unit Model](#building-a-unit-model)
- [Config schema versioning](#config-schema-versioning)
- [Review & Verification by Maintainers](#review--verification-by-maintainers)
- [Code of Conduct](#code-of-conduct)
- [Reporting Bugs & Security Issues](#reporting-bugs--security-issues)
- [Licensing](#licensing)

---

## Ways to Contribute

- **Bug reports** — open an issue with clear reproduction steps.
- **Feature requests** — open an issue describing the problem, not just the solution.
- **Code contributions** — via pull request (see below).
- **Documentation** — fixes, clarifications, and examples are always welcome.
- **Reviewing open PRs/issues** — thoughtful feedback from the community speeds everything up.

## Getting Started

1. Fork the repository and clone it locally.
2. Install dependencies.
3. Run the test suite to confirm a clean baseline.
4. Create a branch for your work: `git checkout -b your-short-branch-name`

## Human Readability Standards

Code and documentation in this repository are read far more often than they are written. All contributions must be **understandable by a human reviewer without special tooling**. Specifically:

- **Clear naming.** Variables, functions, and files should be named for what they do, not abbreviated cryptically.
- **Comments explain "why," not "what."** Don't restate the code in prose; explain intent, trade-offs, and non-obvious decisions.
- **Reasonable diffs.** Keep pull requests focused on a single logical change. Large, unrelated changes bundled together will be asked to be split up.
- **Consistent style.** Follow the existing formatting/linting configuration in this repo. Run it before submitting.
- **Docs and comments must read as if a person wrote them for another person** — plain, direct language, not padded or filler text.

## Contributor Responsibility
By submitting a contribution, **you are personally vouching for it**. Specifically, you confirm that:

- You wrote or meaningfully reviewed the content — you are not simply relaying output you don't understand and asking maintainers to check it for you.
- You have tested the change yourself, locally, before opening the PR.
- You have the right to submit the contribution (it's your own work, or appropriately licensed/attributed).
- You will respond to review feedback on your own submission. Contributions that go unanswered for 10 days after requested changes may be closed.

Maintainers are volunteers reviewing in good faith. Submitting content you can't explain or defend shifts your verification burden onto them, which isn't a fair trade and will typically result in the PR being closed.

## Use of AI Tools

AI-assisted contributions (code, docs, or issue text) are welcome, but the same responsibility bar applies as above — the tool doesn't lower it.

- **Disclose it.** If a substantial part of a PR or issue was generated or drafted with an AI tool, say so in the PR description (a short note is enough — no need to detail every prompt).
- **You must still understand and verify it.** Review, test, and be prepared to explain any AI-assisted content as if you wrote it yourself. "The AI generated it" is not an acceptable answer to a review question.
- **No unreviewed bulk submissions.** Large batches of AI-generated issues or PRs that haven't been individually vetted by a human will be closed without full review.
- **Fabricated content is not acceptable.** Do not submit AI-generated citations, benchmarks, test results, or changelog entries that you haven't personally verified against reality.

## Submitting a Pull Request

1. Ensure your branch is up to date with `main`.
2. Include a clear PR description: what changed, why, and how you tested it.
3. Link any related issue(s).
4. Make sure CI passes before requesting review.
5. Keep the scope of the PR narrow — one concern per PR.
6. If the PR adds or changes a unit option, logic helper, element kind or spec field, add a case to the round-trip corpus (`src/flexops/tests/fixtures/roundtrip_corpus.py`) or the unit contract test, or explain in the PR why not. A new element kind is classified in `KINDS` (stage and mutability).

## Building a Unit Model

Every unit model is an IDAES process block that subclasses `OpsBlockData` (`src/flexops/core/ops_block.py`), directly or through a topology base. A unit hand-writes its own balance constraints (no ControlVolumes) and registers its variables so that costing, FlexParameterize, the spec emitter and the docs generator can find them.

### Modeling principles

Read this before you write any equations. A unit model in flex-pse is a scheduling model: a solver evaluates it at every time step of a horizon, alongside every other unit in the plant, often inside a mixed-integer program. That changes what a "good" model looks like compared with a design or simulation model.

#### Timescale: minutes to hours, any time step

flex-pse decides how a plant operates over minutes to hours: what runs, at what rate, and when to store or shift load. The user picks the time step (e.g. 5 min, 15 min, 1 h) on the `TimeBlock`, and the same unit model must give physically consistent answers at all of them.

- **Model at quasi-steady state.** Anything that settles much faster than one time step (pump spin-up, electrochemical double-layer charging, control-loop transients) is treated as instantaneous. Represent its *effect* instead: an efficiency, a startup energy, a minimum up time. Don't add fast dynamic states that only make sense at a particular step size.
- **Write accumulation as a backward difference scaled by `tb.dt`.** Write rates in per-time units and let `dt` convert them. A tank's holdup is `V[t] = V[t-1] + dt * (Q_in[t] - Q_out[t])` (`storage/tank.py`), and ramp limits are `rate * window * dt` (`logic/ramp.py`). Never write "per step" coefficients: a `0.1 m^3 per step` loss means something different at 5 min than at 1 h. See `docs/explanation/time_and_dynamics.md`.
- **Check the discretization is stable for every reasonable `dt`.** The linear decay factor in `x[t] = x[t-1] * (1 - k*dt) + ...` is accurate only while `k*dt` is small, and it goes negative once `k*dt > 1`. Bound `k` so this holds for any `dt` the user could choose. `storage/battery.py` does this: `charge_leakage_rate` is capped at 0.001 per day and a `TimeBlock` spans at most one month, so `k*dt` stays below about 0.03. If a fast rate can't be bounded that way, use the exact factor `exp(-k*dt)`. For a fixed `k` it's computed once at build time, so the constraint stays linear.
- **Take durations as time quantities, not step counts.** An option such as a startup delay or a minimum run time should accept `30 * pyunits.min`. Convert it to whole steps at build time, and raise if the result can't be represented on the chosen grid.
- **Test across time steps.** Build the unit over the same horizon at two step sizes (e.g. 15 min and 1 h), with the same input profile, and check that the physical totals (energy used, volume delivered, final state of charge) agree within discretization error.

#### Fidelity levels, linearity and convexity

Every unit should give the user a choice between the **simplest credible model** of the process and a **more detailed, physically resolved model**, selected by a config option. `Pump.power_relation` is the pattern to follow: a `StrEnum` option whose default is the simplest form.

A typical fidelity ladder looks like this:

| Level | Example | Problem class |
|---|---|---|
| Assumed performance | `P[t] = energy_intensity * Q[t]` (`add_constant_intensity_relation`, the default on every unit) | LP |
| Physically motivated, linearized | Pump power from a measured pump curve fit as an affine or piecewise-linear function of flow | LP / MILP |
| Detailed physics | `P[t] = dP[t] * Q[t] / eta` with pressure as a decision variable; conduction and convection; mass-transport limits | NLP / MINLP |

The detailed level is valuable only if a scheduler can solve it reliably, so convexify or linearize it wherever you can. State in the unit's docs which problem class each level produces. The solver facade classifies the model and never relaxes it silently (`docs/explanation/relaxation_policies.md`).

- **Fix what the scheduler doesn't decide.** `Pump`'s hydraulic law is bilinear (`dP * Q`) and nonconvex if both factors are free. The inlet and outlet pressures are registered as inputs that the caller fixes, so the law becomes linear in `Q`. Ask which variables the scheduler actually controls, and make the rest parameters.
- **Use the epigraph form for convex curves.** If power rises convexly with throughput (e.g. a blower or RO high-pressure pump), approximate it with tangent lines: `P[t] >= a_k + b_k * Q[t]` for each segment `k`. When power is costed in a minimizing objective, the solver sits on the curve, and the model stays an LP with no binaries.
- **Use binaries only for genuinely nonconvex shapes.** Non-monotonic efficiency curves and forbidden operating zones need SOS2 or binary piecewise formulations (MILP). Link any binaries to the unit's `status` and honor the unit's `relaxation` option (`RelaxationPolicy.EXACT` / `RELAXED`).
- **Watch the off state.** An affine law `P = a*Q + b` draws `b` even when the unit is off. Write it as `P[t] = a*Q[t] + b*status[t]` so the intercept switches off with the unit, and require a status when the intercept is non-zero.
- **Bound every variable that appears in a nonlinear term.** Envelopes (e.g. McCormick for bilinear terms) and the solver's local search both rely on finite, physical bounds.
- **Avoid non-smooth functions** (`abs`, `max`, `min`, `if`) in NLP formulations. Reformulate them with auxiliary variables, or move them into the logic layer.

#### Surrogates and grey-box models

Many processes are best characterized by a detailed simulator (e.g. an IDAES/WaterTAP flowsheet, CFD or a kinetic model) or by experiments, and that model is too large or too nonconvex to embed at every time step. Bring its behavior in as a **surrogate** fitted to parametric sweeps or measured data, instead of approximating it by hand:

- **Use the swap contract.** The `power_<kind>_relation` Constraint (or any relation you register with `register_relation`) can be replaced in place by a `flexops.surrogates` shape: `multilinear`, `quadratic`, `exponential`, `arima` or `neural_network`. FlexParameterize fits these from data and attaches them with `swap_relation`. A new unit whose main relationship isn't a constant intensity should still call `register_relation(constraint, target=...)` so that it can be swapped.
- **Keep conservation laws first-principles; fit only the closures.** A grey-box unit writes its mass and energy balances and its stoichiometry exactly, and uses a surrogate only for the hard-to-model terms: an efficiency, a kinetic rate, or a voltage–current curve. The electrolyzer is the example. Its charge balance, separator holdup and CO2 transport limit are conservation laws and are never swapped. Each product's `faradaic_relation_{k}` and its `power_electrical_relation` are registered relations, so a fitted Faradaic-efficiency curve or polarization curve can replace them.
- **Choose surrogate inputs the scheduler can act on.** Fit against variables in the schedule (flow, power, temperature setpoint), not against internal states that the model doesn't otherwise represent.
- **Declare and enforce the fitted domain.** A surrogate is valid only over the range of its training data. Put that range on the input variables as bounds, so the optimizer can't extrapolate into regions the fit never saw.
- **Prefer surrogate shapes that keep the problem class.** A multilinear fit with fixed inputs keeps an LP an LP, while a neural network makes it an NLP. Black-box models with their own derivatives can be embedded with Pyomo's `ExternalGreyBoxModel` (`pyomo.contrib.pynumero`), but that limits the model to interior-point NLP solvers. Say so in the docs.

#### Validation and references

flex-pse models inform real operating decisions, so every modeling choice has to be traceable to evidence.

- **Base equations and defaults on peer-reviewed literature or documented experiments.** This covers the governing equations, default parameter values (efficiencies, intensities, recovery windows) and operating bounds. If a default comes from a vendor datasheet or an industry rule of thumb, say that rather than leaving it unattributed.
- **Cite the source where readers will look for it.** For a unit with non-trivial physics, use a dedicated page in `docs/explanation/<unit>_model.md`; for a short unit, use a `References:` block in the module docstring. Give a full citation with a DOI where one exists, and point to the specific equation or figure. For example, `docs/explanation/electrolyzer_model.md` writes the CO2 mass-transport limit "following eq. 2 of Matthews et al., *ACS Catal.* 2025, 15, 381".
- **State the range of validity.** Add an "Assumptions and limits" section (see the electrolyzer page) listing the operating window, the time-step range and the phenomena that are deliberately neglected.
- **Reproduce a published result in an example.** Where you can, add a small example to [`flex-pse-examples`](https://github.com/flex-pse/flex-pse-examples) that builds the unit at the conditions of a published case or dataset and compares a reported quantity against the source, e.g. specific energy consumption at rated load. Cite the source in the example and state how closely the unit matches it.
- **Record where a fitted surrogate came from.** Note the simulator or experiment it was fitted to, the input ranges covered and the fit error.
- **Never fabricate a citation.** Every reference must be one you have read and checked against the equation it supports. This applies equally to AI-assisted work (see [Use of AI Tools](#use-of-ai-tools)).

### 1. Pick a base class

- **Every port shares one `property_package`:** subclass the topology base that matches the port count: `SISOBlockData`, `SIDOBlockData` or `DIDOBlockData` (`src/flexops/unit_models/base/`). The base builds the ports and the pass-through mass balance; your class adds only the flow-to-energy relationship. See `Pump` and `ReverseOsmosis`.
- **Streams need different property packages** (e.g. a liquid feed and a gas product), **or the port count is a config option:** subclass `OpsBlockData` directly and write the ports and balance yourself. See `Combustor`, `Mixer` and `Electrolyzer`. Give each stream its own port; one port carries one arc.
- **No streams at all** (e.g. a grid connection or a battery): subclass `OpsBlockData` and declare no ports. See `GridConnection`.

To rename a topology base's generic components (e.g. `flow_out` → `flow_permeate`), pass a complete `naming_dict` to the base `build()`: spread the base's `_component_names` and override what you rename. Ports are never renamed.

### 2. Write the class

Put the module under `src/flexops/unit_models/`, in a domain sub-package if one fits (`storage/`, `powergeneration/`, `wastewater/`). A minimal single-stream unit:

```python
"""UVDisinfection(SISOBlock): a UV reactor with a constant energy intensity."""

from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexops.unit_models.base.siso import SISOBlockData


@declare_process_block_class("UVDisinfection")
class UVDisinfectionData(SISOBlockData):
    """A UV reactor whose electrical draw scales with the flow it treats.

    Config:
        ``energy_intensity`` (default 0.05 kWh/m^3): electrical energy per
        unit volume treated.
    """

    CONFIG = SISOBlockData.CONFIG()
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.05 * pyunits.kWh / pyunits.m**3,
            description="Electrical energy per unit volume treated, kWh/m^3.",
        ),
    )

    def build(self) -> None:
        """Build the SISO ports and balance, then the power relation."""
        super().build()
        self.add_constant_intensity_relation(
            self.flow_out,
            kind=nm.PowerKind.ELECTRICAL,
            intensity=self.config.energy_intensity,
        )
```

Rules every unit follows:

- **Configuration goes through `CONFIG.declare(...)`.** Every option has a `description`. Validate in a `domain=` function that raises `FlexConfigError(field=..., value=...)`, and use a `StrEnum` for a fixed set of choices (see `PumpPowerRelation`). Option values must be JSON-serializable (numbers, strings, enums, unit-carrying quantities, sequences) so `emit_model` can write them to a spec.
- **Use the `OpsBlockData` helpers. Don't create these components by hand:**
  - `add_constant_intensity_relation(flow, kind=..., intensity=...)` is the default flow-to-energy relationship. It builds `power_<kind>`, a regressable `energy_intensity` and the Constraint `power_<kind>_relation`. That Constraint name is what FlexParameterize looks for when it swaps in a fitted relationship.
  - `declare_power(kind)` creates `power_electrical`/`power_thermal` (kW) with the names from `flexcore.nomenclature`, and registers them for plant aggregation and costing. Use it when a unit's power law is not a constant intensity.
  - `declare_process_parameter(name, value, units, doc, bounds=...)` creates a design or fitted coefficient as a fixed scalar Var, not a `Param`, so it can be unfixed later. Give it bounds whenever unfixing it could allow a nonsensical value.
  - `register_io_variable(var, role="input"|"output")` marks the measured inputs and outputs. Registered variables need a `doc=`.
  - `register_fuel_usage` / `register_boundary_flow` cover fuel draws and plant boundary flows.
- **Units on every Var and Constraint.** Use `pyunits.convert` rather than numeric conversion factors; the test harness checks dimensional consistency.
- **Look up dynamic names with `self.find_component(name)`**, not `getattr`.
- **Never delete components** from a built model. Change it in place: `set_value` on mutable Params, `update_parameters`, adding Vars or Constraints, or `deactivate()`.
- **Unit commitment:** if the unit manages its own on/off status (as `Tank` and `BatteryModel` do), set `allows_status_element = False` on the class so a spec `status` element can't target it.
- Write docstrings as one sentence followed by `Args`/`Returns`/`Raises`/`Config`. Longer derivations and equations belong in `docs/explanation/` (see `electrolyzer_model.md`).

### 3. Write the tests first

Add `src/flexops/tests/unit_models/test_<unit>.py`. Write the tests before the implementation, and confirm they fail for the right reason (an `ImportError`, not a typo) before writing the class. Every test carries exactly one tier marker (`unit`, `component` or `integration`); `unit` tests must not call a solver.

Start from `UnitModelTestHarness` (`flexops.testing`). It provides build, units-consistency, IO-registration, energy-naming, degrees-of-freedom and solution checks:

```python
from flexops.testing import UnitModelTestHarness, dummy_time_block
from flexops.unit_models import UVDisinfection


class TestUVDisinfection(UnitModelTestHarness):
    """A fixed 100 m^3/hr inlet at 0.05 kWh/m^3 draws 5 kW."""

    expected_dof = 0
    expected_solution = {f"power_electrical[{t}]": 5.0 for t in range(3)}

    def configure(self):
        m = dummy_time_block(3)
        m.unit = UVDisinfection(property_package=m.properties)
        for t in m.time_block.time_index:
            m.unit.inlet_state.flow_vol_phase[t, "Liq"].set_value(100.0)
        return m, m.unit
```

Then add focused `unit`-marked tests for each config option and each `FlexConfigError` path. Use descriptive names, e.g. `test_uv_disinfection_rejects_negative_intensity`.

### 4. Register the unit

A new unit has to be wired into several places. The tests in parentheses fail if you skip that step:

1. **Export it.** Add the class to `__all__` in `src/flexops/unit_models/__init__.py` (this list is the registry that `unit_model_class` in a spec resolves against) and to the top-level `src/flexops/__init__.py`.
2. **Contract test.** Add a builder to `CONTRACT_BUILDERS` in `src/flexops/tests/test_unit_contract.py`. It checks that the unit round-trips through `emit_model` → JSON → `build_model` with no `FlexEmitWarning` (`test_every_unit_class_has_a_contract_builder`). If an option changes the model's structure, add a case to `src/flexops/tests/fixtures/roundtrip_corpus.py` as well.
3. **Reference docs.** Add an `autosummary` entry with a short description of the unit to `docs/reference/flexops/unit_models/index.rst`. If it uses the `autosummary/unit_model.rst` template, add a builder to `_BUILDERS` in `docs/_ext/flexdoc.py` so the variable, constraint and DoF tables render. Confirm that `sphinx-build -W --keep-going -b html docs docs/_build` passes.
4. **Changelog.** Add an entry under `## [Unreleased]` in `CHANGELOG.md`.

### 5. Verify before pushing

```bash
pytest -m "unit" -x -q           # inner loop
ruff check . && black --check .
lint-imports                     # flexops may import only flexcore
pytest -q                        # all tiers
```

## Config schema versioning

- Any change to the flat `FlowsheetSpec` bumps `SCHEMA_VERSION` and regenerates `schemas/flowsheet_spec.schema.json` in the same PR.
- Once a spec version has been released, a breaking change also adds a `MIGRATIONS` hook from the old version; a migration that cannot translate raises `FlexConfigError` naming the field and how to fix it.

## Review & Verification by Maintainers

All contributions are subject to maintainer review before merging. This is a verification step, not a rubber stamp:

- **Maintainer approval** is required before merge.
- Maintainers may request changes, ask clarifying questions, or ask the contributor to explain specific parts of the change — please respond rather than resubmitting silently.
- Automated checks (tests, linting, security scanning) must pass, but passing CI does not guarantee merge — human review of correctness, readability, and intent is still required.
- Maintainers reserve the right to decline contributions that don't meet the standards above, even if functionally correct, if they harm long-term readability or maintainability.

## Code of Conduct

This project follows a Code of Conduct. Please be respectful and constructive in issues, PRs, and discussions. 

## Reporting Bugs & Security Issues

- Regular bugs: open a GitHub issue with reproduction steps.
- Security vulnerabilities: do **not** open a public issue. Instead, email `akshayr135 at gmail dot com` or follow the process in `SECURITY.md`.

## Licensing

By contributing, you agree that your contributions will be licensed under this project's Apache 2.0 license (see `LICENSE`).

---

Questions not covered here? Open a discussion or reach out to the maintainers at `akshayr135 at gmail dot com`.
