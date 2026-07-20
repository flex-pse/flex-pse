# M11 — Regressor protocol + linear regression

**Effort:** 2 days · **Depends on:** M10 · **Parallelizable:** with M12

## Goal
Formalize the pluggable-regressor seam: a runtime-checkable `Regressor` Protocol
with a shared `FitResult`, a sklearn-backed `LinearRegressor` behind the
`[parameterize]` extra, and a small name→class registry for config-driven
selection. After this milestone, adding NN/ARIMA/multiconvex regressors post-v0
is "implement the protocol, register the name" — no pipeline changes.

A fitted regressor feeds **both** consumers of the two-way pipeline (architecture
§5, R10/R11): `emit.py` (which turns the `SurrogateSpec` into a config that
rebuilds the model) **and** `apply.py` (which uses the same `SurrogateSpec` to
swap a unit's energy-relationship Constraint in place on a live model). There is
one `SurrogateSpec` per fit and two consumers — no structural change in this
milestone, but the linear regressor's `SurrogateSpec` must be equally consumable
by the apply path.

## Read first
- `plan/01_architecture.md` §5 (regression/base.py `Regressor` Protocol; constant
  and linear regressors; NN/ARIMA/multiconvex as post-v0 implementations of the
  same protocol; provenance in emitted configs; the same `SurrogateSpec` feeds
  both `emit.py` and `apply.py` — R10, one fit, two consumers)
- `plan/01_architecture.md` §2.3 (R3: `SurrogateSpec` — functional forms
  `constant_intensity`, `linear`, reserved `nn`/`arima`/`multiconvex`;
  `provenance` field)
- `plan/01_architecture.md` §3.4 ("helper functions attach the flow↔energy
  relationship … same base units, controllable functional form")
- `plan/00_conventions.md` §3 (exceptions: `FlexConfigError` with actionable
  messages), §7 (markers, determinism, fixed seeds)
- `plan/02_testing_and_ci.md` §1 (solver/availability skip pattern — mirror it
  for the optional sklearn dependency)

## Files to create or modify
- `src/flexparameterize/regression/base.py` — `Regressor` Protocol + `FitResult` dataclass.
- `src/flexparameterize/regression/constant.py` — refactor `ConstantIntensityRegressor` to conform.
- `src/flexparameterize/regression/linear.py` — `LinearRegressor` (sklearn, optional extra).
- `src/flexparameterize/regression/__init__.py` — `get_regressor(name)` registry + re-exports.
- `pyproject.toml` — add/confirm the `[parameterize]` extra (`scikit-learn`).
- `src/flexparameterize/tests/regression/test_base.py`, `test_linear.py`, `test_registry.py`, `test_linear_roundtrip.py`.
- Docs: `docs/how_to/parameterize_from_data.md` (complete), `docs/reference/flexparameterize/index.rst`, explanation note on surrogate extension points.

## Specification

### Protocol and FitResult (`src/flexparameterize/regression/base.py`)
```python
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

@dataclass
class FitResult:
    coefficients: dict[str, float]      # coefficient name -> value (include "intercept" when present)
    metrics: dict[str, float]           # at minimum: "r2", "rmse"
    n_samples: int
    data_window: tuple                  # (start, end) of the fitted index

@runtime_checkable
class Regressor(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> FitResult: ...
    def to_surrogate_spec(self) -> SurrogateSpec: ...
```
- `typing.Protocol` + `@runtime_checkable`, exactly as architecture §5 names it.
  No ABC, no base-class inheritance requirement — conformance is structural.
- `fit` takes DataFrames for both `X` (inputs, one column per input variable) and
  `y` (outputs — a DataFrame, not a Series, because multi-output is in scope).
  It returns **and stores** the `FitResult` (implementer's choice of private
  attribute; `to_surrogate_spec` before `fit` raises `FlexDataError` telling the
  user to fit first).
- `data_window` tuple element type follows the index (timestamps for
  `DatetimeIndex`); it must serialize into provenance (implementer's choice:
  ISO-8601 strings at the SurrogateSpec boundary).

### Refactor ConstantIntensityRegressor (`regression/constant.py`)
Make it conform to the Protocol: `fit(X, y) -> FitResult` and
`to_surrogate_spec()`. If M10 followed its spec (coefficient + metrics +
n_samples + data window already stored on the instance), **this is a mechanical
rename onto the `FitResult` dataclass — it should be, and if it is not, fix the
divergence here rather than widening the protocol.** Update M10's tests only
where the return type changed; behavior (fit rule, zero-flow guard, 1e-6
round-trip) is unchanged.

### LinearRegressor (`src/flexparameterize/regression/linear.py`)
Ordinary least squares via scikit-learn, available only with the
`[parameterize]` extra:
- Import guard at module import time (implementer's choice: guard inside
  `__init__`/`fit` instead so the module imports cleanly for docs — pick one;
  the error is the contract):
  ```python
  raise FlexConfigError(
      "LinearRegressor requires scikit-learn. "
      "Install it with `pip install 'flex-pse[parameterize]'`."
  )
  ```
- `fit(X, y)` wraps `sklearn.linear_model.LinearRegression` with
  `fit_intercept=True` (implementer's choice: expose as keyword). **Multi-output
  support**: `y` may have several columns (e.g. flow → power AND chemical dose);
  sklearn handles this natively — preserve column names so coefficients come out
  as `{output}:{input}` keys (implementer's choice of key scheme; document it —
  both `emit_model_config`'s rebuild path and `apply.py`'s in-place constraint
  swap onto `ConstantEnergyIntensityModel` (M09, R11) must be able to consume
  the resulting `SurrogateSpec`, so match whatever coefficient layout that spec
  defined).
- `FitResult.metrics`: `r2` (sklearn `score`) and `rmse` per output (aggregate
  key scheme implementer's choice, document); `n_samples` after NaN-dropping;
  `data_window` from the surviving index.
- `to_surrogate_spec()` → `SurrogateSpec` with functional form `linear`,
  coefficients (including intercepts), input/output variable names from column
  names, and **fit metrics landed in `SurrogateSpec.provenance`** (merged with
  whatever `emit_model_config` adds — do not duplicate keys; provenance layout is
  R3's, keep it schema-valid).
- NaN policy: drop rows with any NaN across `X`/`y` before fitting; if fewer
  rows remain than columns + 1, raise `FlexDataError` saying how many rows
  survived and what the minimum is.

### Registry (`src/flexparameterize/regression/__init__.py`)
```python
def get_regressor(name: str) -> type: ...
```
- Known names → classes: `"constant_intensity"` → `ConstantIntensityRegressor`,
  `"linear"` → `LinearRegressor`. These strings deliberately equal the
  `SurrogateSpec` functional-form names (R3) — config-driven selection is
  `get_regressor(spec.functional_form)`.
- **Reserved names** `"nn"`, `"arima"`, `"multiconvex"` raise
  `NotImplementedError` with a message pointing at the post-v0 backlog (PLAN.md
  §4) — reserving them now keeps configs forward-compatible.
- Unknown names raise `FlexConfigError` listing the valid and reserved names.
- Keep the registry a plain module-level dict (implementer's choice; no
  entry-points plugin machinery in v0).
- Re-export `Regressor`, `FitResult`, both regressor classes, and
  `get_regressor` from the package init. Importing the package must NOT import
  sklearn (Pitfall 1).

## Pitfalls
1. **Registry import pulls in sklearn.** `get_regressor("linear")` may import
   lazily, but `import flexparameterize.regression` must succeed without sklearn
   installed — otherwise the base install breaks. Lazy-import inside the
   registry lookup or guard inside the class, never at package-init import time.
2. **`runtime_checkable` checks names, not signatures.** `isinstance(obj,
   Regressor)` only verifies the methods exist. The conformance tests must
   therefore also *call* `fit`/`to_surrogate_spec` and check the return types —
   say this in a comment in the test.
3. **Monkeypatching the sklearn-absent path.** Simulate absence with
   `monkeypatch.setitem(sys.modules, "sklearn", None)` (plus the submodule keys
   you import) and reload/re-invoke the guard; do not uninstall anything.
4. **Skips leaking red into the extra-less env.** Every test that genuinely
   imports sklearn carries
   `pytest.importorskip`/`@pytest.mark.skipif(not HAS_SKLEARN, ...)` so a
   checkout without the extra passes cleanly (skip, not fail) — mirror the
   solver-availability pattern from 02_testing §1.
5. **Unseeded noise.** The synthetic-recovery test uses
   `numpy.random.default_rng(42)` (or any fixed seed) — conventions §7 forbids
   nondeterminism.
6. **Provenance key collisions.** Both `to_surrogate_spec` and
   `emit_model_config` write provenance; define who owns which keys (fit metrics
   here; data window / package versions in emit — implementer's choice, but
   assert both survive in the emitted JSON).
7. **Breaking M10's round-trip.** The constant-regressor refactor must leave
   `test_constant_intensity_round_trip` green — it is the pipeline's regression
   baseline.

## Tests
`src/flexparameterize/tests/regression/test_base.py`
- `test_protocol_conformance` (`unit`) — `isinstance(ConstantIntensityRegressor(), Regressor)`
  and (skipif sklearn absent) `isinstance(LinearRegressor(), Regressor)` are
  True; a class missing `to_surrogate_spec` is False. Then call `fit` on a tiny
  frame and assert the return is a `FitResult` (see Pitfall 2).
- `test_fitresult_fields` (`unit`) — dataclass carries coefficients, metrics
  with `r2`/`rmse`, `n_samples`, `data_window`.
- `test_to_surrogate_spec_before_fit_raises` (`unit`) — `FlexDataError`.

`src/flexparameterize/tests/regression/test_linear.py` (module-level
`pytest.importorskip("sklearn")` except the absence test)
- `test_recovers_coefficients_noisy_pump` (`unit`) — synthetic pump data:
  `power = 0.4 * flow + 2.0 + noise`, 200 rows, `default_rng(42)`, noise sigma
  0.01; fitted coefficient within `pytest.approx(0.4, rel=1e-2)` and intercept
  within `pytest.approx(2.0, abs=0.05)`.
- `test_multi_output` (`unit`) — `y` with two columns (power, chemical dose);
  both outputs' coefficients recovered; spec lists both output names.
- `test_provenance_populated` (`unit`) — `to_surrogate_spec().provenance`
  contains `r2`/`rmse`; values finite; JSON-serializable.
- `test_sklearn_absent_raises` (`unit`, NO skipif) — monkeypatch sklearn out of
  `sys.modules`; instantiating/fitting raises `FlexConfigError` whose message
  contains `flex-pse[parameterize]`.

`src/flexparameterize/tests/regression/test_registry.py` (all `unit`)
- `test_get_regressor_known_names` — both names return the right classes.
- `test_reserved_names_raise_notimplemented` — `nn`, `arima`, `multiconvex` →
  `NotImplementedError`.
- `test_unknown_name_raises` — `FlexConfigError` listing valid names.
- `test_package_import_without_sklearn` — with sklearn monkeypatched absent,
  `importlib.reload(flexparameterize.regression)` succeeds.

`src/flexparameterize/tests/regression/test_linear_roundtrip.py`
- `test_linear_fit_emit_rebuild_predictions` (`component`, skipif sklearn
  absent) — end-to-end: synthetic linear data (fixed seed) → `LinearRegressor.fit`
  → `emit_model_config` → `dump_model_config`/`load_model_config` → rebuild via
  `ConstantEnergyIntensityModel.build_from_config` on a small TimeBlock (the `linear`
  `SurrogateSpec` triggers the in-place constraint swap at construction time,
  M09) → fix inputs at **5 probe points**, evaluate the resulting output-constraint
  bodies, and assert the rebuilt predictions match `regressor` predictions at
  those points within `pytest.approx(rel=1e-6)`. No solver needed
  (constraint-body evaluation).
- `test_linear_surrogate_spec_applies_in_place` (`component`, skipif sklearn
  absent) — the same fitted `SurrogateSpec` also drives `apply.py`: build a live
  model with the default constant-intensity relationship, run `apply_to_model`,
  and assert the swapped-in Constraint reproduces the same predictions at the 5
  probe points (rel=1e-6). This is the one-fit/two-consumers check (R10) — the
  linear spec must be consumable by the apply path exactly as by the emit path.

## Documentation tasks
- Complete `docs/how_to/parameterize_from_data.md`: fill the regressor-selection
  section (constant vs. linear, `get_regressor`, the `[parameterize]` extra and
  its install line), ending with the emit→rebuild step.
- `docs/reference/flexparameterize/index.rst`: add `regression.base`,
  `regression.linear`, and `get_regressor`; document `FitResult` fields (they
  render from docstrings — write them).
- Explanation note on surrogate extension points (smallest home: a section in an
  existing `docs/explanation/` page or a short new
  `docs/explanation/surrogate_extension_points.md` — implementer's choice):
  the Regressor Protocol + reserved functional-form names are the post-v0
  NN/ARIMA/multiconvex hooks; cross-reference PLAN.md §4 and architecture §3.4.
- Installation docs: mention the `[parameterize]` extra in
  `docs/getting_started/installation.md`.
- CHANGELOG entry under "Unreleased".

## Definition of Done
- [ ] `Regressor` Protocol (runtime_checkable) + `FitResult` dataclass exist with
      the exact signatures above.
- [ ] `ConstantIntensityRegressor` conforms; M10's round-trip test still green.
- [ ] `LinearRegressor` fits multi-output data, recovers seeded synthetic
      coefficients within tolerance, and lands metrics in
      `SurrogateSpec.provenance`.
- [ ] sklearn-absent path raises `FlexConfigError` with the install instruction;
      base install imports `flexparameterize.regression` cleanly.
- [ ] `get_regressor` resolves both names; reserved names raise
      `NotImplementedError`; unknown names raise `FlexConfigError`.
- [ ] All sklearn-dependent tests skip (not fail) when the extra is absent.
- [ ] Component round-trip reproduces predictions at 5 probe points (rel=1e-6).
- [ ] The fitted `SurrogateSpec` feeds both consumers — `emit.py` (config) and
      `apply.py` (in-place replacement); the apply-path test reproduces the same
      predictions (one fit, two consumers — arch §5, R10).
- [ ] `pyproject.toml` `[parameterize]` extra present; how-to page completed;
      reference + explanation docs build with `sphinx-build -W`; CHANGELOG updated.
- [ ] plus the generic DoD in CLAUDE.md
