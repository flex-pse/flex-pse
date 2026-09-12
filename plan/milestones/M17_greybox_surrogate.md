# M17 — External-model grey-box surrogate

**Effort:** 2–3 days · **Depends on:** M09 (surrogate registry; the
post-#104 `Surrogate.build() -> (block, body)` contract, already merged to
`main`) · **Parallelizable:** with anything not touching `flexops.surrogates`

## Status

`avdudchenko:update-surrogate-structure` (#104) merged before this document
was rewritten. `Surrogate.build()` returns `(block, body(t))`; `swap_relation`
attaches `block` to the unit itself, wraps `body(t)` in
`target[t] == pyunits.convert(body(t), target's units)`, and (when `block`
carries a `.coefficients` `CoefficientRegistry`) auto-registers it. This
milestone's surrogate registers no coefficients — a wrapped model's internal
weights are not something `flexparameterize` can regress (see Goal) — so its
block simply omits `.coefficients`.

This is a from-scratch rewrite of an earlier, much larger draft of this
document (dotted-path config indirection, finite-difference fallback,
batching/JAX open questions, a t0-alpha-specific prototype writeup). None of
that survived: the goal now is the smallest amount of code that lets an
external differentiable model be swapped in as a unit's relation and solved,
so it can actually be exercised from an example (which lives in the companion
`flex-pse-examples` repo, not here — see memory
`examples-moved-to-separate-repo`). Anything cut below is cut deliberately,
not overlooked; extend later against a real use case, not speculatively.

## Goal

Let `flexops.surrogates` wrap an arbitrary **external differentiable model**
as a unit's registered relation, solved via `SolverFactory("cyipopt")`. No
closed-form Pyomo expression is ever derived — the model stays opaque to
Pyomo; only its numeric output and derivatives are used.

The surrogate class is framework-agnostic. Which framework a given spec uses
is a declared `framework` field resolved to a **driver** — a small object that
knows how to evaluate one model and its first two derivatives at a point.
PyTorch (via `torch.autograd`) is the one driver implemented here; the other
members of the framework enum are reserved and raise `NotImplementedError`,
exactly as the unimplemented `SurrogateType` members do today. Only the driver
module imports its framework, so the surrogate class itself — and therefore
`import flexops.surrogates` — stays free of any heavyweight dependency.

**Not this milestone:**
- **Not a replacement for `NeuralNetworkSurrogate`** (reserved for a network
  translated into literal Pyomo expressions — e.g. an OMLT big-M/ICNN
  encoding — fully ASL/ipopt-solvable, no grey box). An external model is
  never translated; it always needs CyIpopt.
- **Not the "forecast, then fix parameters" adapter** in PLAN.md §4.2
  ("External forecaster interface"). That is a separate, cheaper capability
  (forecaster runs once, outside the optimization, output becomes a fixed
  `Param`) with no feedback loop and no grey box. This milestone is for the
  case where the model's output must live *inside* the NLP — the unit's own
  decision variables feed it and it feeds back into the objective/constraints.
- **No finite-difference fallback, no batching, no cross-time coupling.** The
  PyTorch driver differentiates the model exactly, called independently at
  each time index (the relation is static, same shape as every other
  surrogate's `body(t)`). A future milestone can revisit any of these against
  a concrete need; none is built speculatively here.
- **Only one driver is implemented.** `tensorflow`, `onnx` and `jax` are enum
  members with a `NotImplementedError`, not code.
- **No config-serializable model reference.** `SurrogateSpec.data` is
  normally a JSON-persistable Layer-1 config (`plan/00_conventions.md` §4),
  but a live, already-instantiated model object cannot round-trip through
  JSON. This surrogate's `data["model_path"]` is a dotted import path
  (`importlib`-resolved), so the *spec* stays JSON-serializable even though
  what it names is a live, trusted-as-code Python object — the same
  trade-off every other surrogate's `data` makes, just resolved at build time
  instead of inline.

## Specification

### 1. `SurrogateSpec.data` contract

```python
{
    "framework": "pytorch",                  # ExternalFramework member value
    "model_path": "myproject.models.my_fitted_model",  # dotted path
    "input_variables": {"flow_in": "m^3/hr", "ambient_temperature": "degK"},
    "output_variables": {"fouling_rate": "1/hr"},      # exactly one entry
    "probe_point": {"flow_in": 1.0, "ambient_temperature": 298.15},  # optional
}
```

`model_path` is resolved via `importlib.import_module` + `getattr` and must
name an object that is directly callable as `model(x) -> scalar`, where `x`
carries the inputs ordered exactly as `input_variables` (dict order). The
object must already be a ready-to-use, fitted model (an `nn.Module` instance,
a closure, anything callable) — there is no kwargs/factory step; if a caller
needs one, `model_path` names a module-level variable that already does the
construction.

`probe_point` (optional, defaults to `1.0` for every input) is the point in
the surrogate's **declared input units** at which the model is smoke-tested
during validation, and the value the grey-box block's input Vars are
initialized to. It is not a bound and not a fit; it exists because the default
of zero is outside the domain of perfectly ordinary models (`log`, `sqrt`,
`1/x`) and because PyNumero evaluates derivatives there before the first
solver iteration (Pitfall 6). Keys must be a subset of `input_variables`.

### 2. Framework enum and driver registry — `flexops/surrogates/external.py`

This module imports **no** framework. It holds:

```python
class ExternalFramework(StrEnum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    JAX = "jax"


class ExternalModelDriver(ABC):
    """Evaluate one external model and its first two derivatives at a point."""

    framework: ClassVar[ExternalFramework]

    def __init__(self, model, n_inputs: int) -> None: ...

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float: ...

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense gradient, shape ``(n_inputs,)``."""

    @abstractmethod
    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense *full symmetric* Hessian, shape ``(n_inputs, n_inputs)``."""

    @abstractmethod
    def check_differentiable(self, x: np.ndarray) -> None:
        """Raise FlexConfigError if the model is not differentiable at ``x``."""


_DRIVERS: dict[ExternalFramework, str] = {
    ExternalFramework.PYTORCH: "flexops.surrogates.drivers.torch_driver.TorchDriver",
}
_RESERVED_FRAMEWORKS = {
    ExternalFramework.TENSORFLOW: "TensorFlowDriver",
    ExternalFramework.ONNX: "OnnxDriver",
    ExternalFramework.JAX: "JaxDriver",
}


def get_driver(framework: ExternalFramework | str) -> type[ExternalModelDriver]:
    """Resolve a framework to its driver class, importing it lazily."""
```

`get_driver` mirrors `flexparameterize.regression.get_regressor` exactly:
`NotImplementedError` naming the reserved class for a reserved member,
`FlexConfigError` listing known values for an unknown string, and a lazy
`importlib` import of the driver module for an implemented one. The lazy
import is what keeps `torch` out of `import flexops.surrogates` — there is
**no** special case in `surrogate_from_spec` (contrast the earlier draft).

Drivers return **dense numpy arrays**. Every sparse-matrix concern — fixed
sparsity pattern, lower-triangular-only Hessian, multiplier scaling — lives in
one framework-free place (§4), so it is unit-testable with a two-line
`ExternalModelDriver` stub and no framework installed at all.

### 3. `flexops/surrogates/drivers/torch_driver.py`

`TorchDriver(ExternalModelDriver)` — the one module in this milestone that
imports `torch` at its own top level.

- `__init__` records the model and coerces the dtype once. Call
  `model.double()` when the object is an `nn.Module` whose parameters are not
  already float64; every call site then uses `torch.float64` (Pitfall 3). A
  plain function/closure has no parameters and needs no coercion.
- `_as_tensor(x)`: `torch.as_tensor(x, dtype=torch.float64)`.
- `evaluate`: `float(self._model(self._as_tensor(x)).reshape(()))` — the
  `reshape(())` makes 0-d and 1-element outputs behave identically and turns
  any other shape into a clear error.
- `jacobian`: `torch.autograd.functional.jacobian(self._scalar_fn, t)`
  reshaped to `(n_inputs,)`, `.detach().numpy()`.
- `hessian`: `torch.autograd.functional.hessian(self._scalar_fn, t)` reshaped
  to `(n_inputs, n_inputs)`, `.detach().numpy()`. Full symmetric — §4 takes
  the lower triangle.
- `_scalar_fn` wraps the model so it always returns a 0-d tensor;
  `torch.autograd.functional.hessian` requires a single-element result, and
  normalizing once here means the `(n,)`-vs-`(1,n)` shape difference between
  a 0-d and a 1-element output never leaks out of this file.
- `check_differentiable(x)`: build `t = self._as_tensor(x).requires_grad_(True)`,
  then `grad, = torch.autograd.grad(self._scalar_fn(t), t, allow_unused=True)`.
  Raise `FlexConfigError` if `grad is None` (a `.detach()`/`.item()`/
  `no_grad()` break in the forward pass), naming `model_path`. Use
  `torch.autograd.grad`, **not** `.backward()`: `.backward()` accumulates into
  the caller's `model.parameters()[...].grad` as a side effect, mutating a
  model the user may be using elsewhere. Do **not** reject a non-finite
  gradient — `sqrt`/`log`/`1/x` have legitimately infinite derivatives at
  ordinary probe points, and rejecting those would be a false positive
  (Pitfall 7). A non-finite gradient is a `_log.warning`.

Using `torch.autograd.functional.jacobian`/`hessian` (rather than manual
`torch.autograd.grad(..., create_graph=True)` chains) means there is no
double-backward footgun to get wrong here — the classic gotcha from the
original prototype is sidestepped by using the higher-level API, not by
being careful.

### 4. `flexops/surrogates/grey_box.py` — the PyNumero adapter

`_ExternalModelGreyBox(ExternalGreyBoxModel)` (private; from
`pyomo.contrib.pynumero.interfaces.external_grey_box`, core Pyomo and
importable with neither `cyipopt` nor `torch` installed). One instance per
time index. It imports no framework — it holds an `ExternalModelDriver`.

- `input_names`/`output_names`: the surrogate's declared names.
- `set_input_values(values)`: cache `np.asarray(values, dtype=float)`.
- `evaluate_outputs`: `np.array([self._driver.evaluate(self._x)])`.
- `evaluate_jacobian_outputs`: a `scipy.sparse.coo_matrix` of shape
  `(1, n_inputs)` built with an **explicit, fixed** index pattern —
  `row=np.zeros(n, int)`, `col=np.arange(n)`, `data=driver.jacobian(x)` — so
  every one of the `n` entries is present whether or not it is numerically
  zero (Pitfall 5).
- `set_output_constraint_multipliers(multipliers)`: cache `multipliers[0]`.
- `evaluate_hessian_outputs`: `H = multiplier * driver.hessian(x)`, then
  `rows, cols = np.tril_indices(n)` and
  `coo_matrix((H[rows, cols], (rows, cols)), shape=(n, n))` — lower-triangular
  by construction (Pitfall 4) with a fixed pattern (Pitfall 5). Shape is
  `(n_inputs, n_inputs)`, *not* `(n_inputs + n_outputs)²`. The multiplier
  scaling is correct because PyNumero's output residual is
  `evaluate_outputs() - output_vars`, whose Hessian in the inputs is
  `λ · ∇²f`.
- `finalize_block_construction(block)`: PyNumero calls this once per block.
  Set `block.inputs[name].set_value(probe_point[name])` and
  `block.outputs[name].set_value(driver.evaluate(probe_point))`. Never `fix()`
  anything on the block (Pitfall 6).

### 5. `ExternalModelSurrogate` — `flexops/surrogates/grey_box.py`

```python
class ExternalModelSurrogate(Surrogate):
    surrogate_type: ClassVar[SurrogateType] = SurrogateType.EXTERNAL_MODEL

    def _validate(self) -> None:
        # 1. key/shape checks mirroring MultilinearSurrogate (unknown/missing
        #    keys, input/output_variables are {name: units} with parseable
        #    units, output_variables has exactly one entry, probe_point keys
        #    are a subset of input_variables and its values are numbers)
        # 2. resolve `framework` via get_driver (NotImplementedError for a
        #    reserved member, FlexConfigError for an unknown string)
        # 3. resolve model_path eagerly via importlib; FlexConfigError (not
        #    ImportError/AttributeError) on a bad path, naming the path and
        #    the underlying error
        # 4. construct the driver and call driver.check_differentiable(probe)

    @property
    def input_variables(self) -> dict[str, str]: ...
    @property
    def output_variables(self) -> dict[str, str]: ...

    def build(self, unit, target):
        # index_set = target.index_set()  -- same set swap_relation uses
        # resolve each input Var via unit.resolve_variable (as every surrogate does)
        # block.egb = ExternalGreyBoxBlock(
        #     index_set,
        #     external_model=lambda b, t: _ExternalModelGreyBox(driver, ...),
        # )
        # block.input_links: Constraint(index_set, input_names) linking each
        #   egb[t].inputs[name] to the resolved unit Var, unit-converted AND
        #   divided by the declared units (Pitfall 8)
        # body(t) -> block.egb[t].outputs[output_name] * output_units
        return block, body
```

`external_model=` **must be a callable** `(block, index) -> model`, not a
model instance — see Pitfall 1.

`ExternalModelSurrogate` is added to the eagerly-imported `SURROGATES` dict
like every other class. Nothing in its import chain touches a framework; the
framework is imported only when `_validate` calls `get_driver`. A bare install
therefore imports `flexops.surrogates` cleanly, and only *constructing* an
`external_model` surrogate raises `ModuleNotFoundError: No module named
'torch'` — acceptable for v1 (no need to wrap it in a friendlier message yet).

### 6. Build-time solver guard

`flexcore.solvers.facade.get_solver(model=...)` cannot pick a solver for a
model containing an `ExternalGreyBoxBlock` — its whole priority list
(`gurobi, scip, highs, cbc, ipopt`) is ASL/NL-file solvers, none of which can
call back into Python. Worse, this fails *silently* today: `classify()` sees
no Pyomo constraints inside a grey-box block and returns `ProblemClass.LP`, so
`get_solver` hands back HiGHS and the model solves to a confidently wrong
answer. Add a narrow, generic check (no coupling to `flexops` needed —
`ExternalGreyBoxBlockData` is core Pyomo, present regardless of whether
`cyipopt`/`torch` are installed):

```python
def _has_grey_box_block(model) -> bool:
    from pyomo.contrib.pynumero.interfaces.external_grey_box import (
        ExternalGreyBoxBlockData,
    )
    return any(
        isinstance(b, ExternalGreyBoxBlockData)
        for b in model.block_data_objects(descend_into=True)
    )
```

called at the **top** of `get_solver`'s `if model is not None:` branch,
**before** `classify(model)`, raising `FlexSolverError` naming
`SolverFactory("cyipopt")` as the required alternative.

Deliberately *not* done: adding `cyipopt` to
`flexcore.solvers.registry.CAPABILITIES` so `get_solver(prefer="cyipopt")`
could select it. That would put the guard above in direct conflict with the
one selection it should allow, and routing grey-box solves through
`SolverFacade` is a larger design question than this milestone. v1 users call
`SolverFactory("cyipopt")` themselves; the guard's message says so.

### 7. `get_regressor` must not `KeyError`

`flexparameterize.regression.get_regressor` looks a `SurrogateType` up in
`_REGRESSORS`, falling back to `_RESERVED` for a clear `NotImplementedError`.
`EXTERNAL_MODEL` belongs to neither, so it currently raises a bare `KeyError`.
Give it its own branch: an external model's weights are not regressable by
`flexparameterize` at all (that is the whole point of wrapping it), so this is
a permanent `FlexConfigError` explaining that, not a reserved-for-later
`NotImplementedError`.

### 8. Test-marker plumbing (read this before writing any test)

`needs_<x>` markers do **not** work for non-solvers today. The root
`conftest.py` resolves every `needs_` marker against
`flexcore.solvers.registry.available_solvers()`, whose keys come from a
hardcoded `CAPABILITIES` dict. A `needs_torch` marker added naively is never
satisfied, so **every test in this milestone would silently skip forever** and
the milestone would merge with zero executed coverage. This is the
`flex-pse-ipopt-not-on-path` failure mode, made permanent.

Extend `pytest_collection_modifyitems` with an explicit optional-package set
resolved by import, checked *before* the solver registry:

```python
OPTIONAL_PACKAGES = {"torch", "cyipopt"}
...
if name in OPTIONAL_PACKAGES:
    if importlib.util.find_spec(name) is None:
        item.add_marker(pytest.mark.skip(reason=f"package {name} not installed"))
    continue
```

`cyipopt` goes here rather than in `CAPABILITIES` for the reason in §6.

Test modules must **not** `import torch` at module scope: on a bare install
that is a collection *error*, not a skip, which violates the DoD. Use
`torch = pytest.importorskip("torch")` inside the tests (or a module-level
`pytest.importorskip`) even though the marker also skips them — the marker
governs the run, `importorskip` governs collection.

## Files to create or modify

- `src/flexops/surrogates/external.py` — `ExternalFramework`,
  `ExternalModelDriver`, `_DRIVERS`, `get_driver` (new; imports no framework).
- `src/flexops/surrogates/drivers/__init__.py`,
  `src/flexops/surrogates/drivers/torch_driver.py` — `TorchDriver` (new; the
  only module importing `torch`).
- `src/flexops/surrogates/grey_box.py` — `ExternalModelSurrogate` (new),
  `_ExternalModelGreyBox` (private).
- `src/flexops/surrogates/surrogates.py` — add `ExternalModelSurrogate` to
  `SURROGATES`. No special case in `surrogate_from_spec`.
- `src/flexops/surrogates/__init__.py` — export `ExternalFramework`,
  `ExternalModelSurrogate`, `get_driver`. Importing the package must still
  not import `torch`; a test asserts this.
- `src/flexcore/config/schema.py` — add `EXTERNAL_MODEL = "external_model"`
  to `SurrogateType`.
- `src/flexcore/config/schemas/model_config.schema.json` — **regenerate and
  commit**. Adding a `SurrogateType` member without this fails
  `flexcore/tests/config/test_schema.py::test_exported_schema_up_to_date`.
- `src/flexcore/solvers/facade.py` — the guard from Specification §6.
- `src/flexparameterize/regression/__init__.py` — the `EXTERNAL_MODEL` branch
  from Specification §7.
- `src/flexops/core/ops_block.py` — in `swap_relation`, when deactivating a
  previous surrogate's components, also deactivate any nested
  `ExternalGreyBoxBlock` *container* found via
  `block.component_objects(ExternalGreyBoxBlock, descend_into=True)`. This is
  an explicitly instructed, narrow change to a previous milestone's code (see
  Pitfall 2 for why block-level `deactivate()` is not enough); the pynumero
  import is function-scope and core Pyomo.
- `conftest.py` — the optional-package marker resolution from §8.
- `pyproject.toml` — new `[greybox]` extra: `cyipopt`, `torch`. Not core
  deps, not in `[solvers]`. A CPU-only torch build is an install-time flag
  (`--extra-index-url https://download.pytorch.org/whl/cpu`), not something
  `pyproject.toml`'s dependency list can pin by itself — note this in
  installation docs rather than trying to encode it here.
- `pyproject.toml` `[tool.pytest.ini_options]` markers — add
  `needs_cyipopt` and `needs_torch`, alongside the existing `needs_ipopt`
  etc. (markers alone do nothing; §8 is what makes them work).
- `.github/workflows/ci.yml` — add `greybox` to the **coverage-floors** job's
  install (`-e ".[dev,solvers,parameterize,greybox]"`). That job is pinned to
  py3.13 (torch publishes wheels for it) and is the job that enforces
  `--fail-under=94` on `src/flexops/*`: without torch installed there the new
  modules report 0% and drop `flexops` below its floor. Leave the
  standard-install job bare — it is what proves the optional dependency stays
  optional (memory `flex-pse-dependency-packaging`).
- `src/flexops/tests/surrogates/test_external.py` (new) — driver registry.
- `src/flexops/tests/surrogates/test_grey_box.py` (new).
- Docs: one short `docs/reference/flexops/surrogates` section for
  `ExternalModelSurrogate` and `ExternalFramework` (data contract, the
  CyIpopt-only solve requirement, the "no `Regressor` counterpart" note). Not
  a new how-to page — the reference section plus the companion example is
  enough for v1. **`docs/conf.py` needs `autodoc_mock_imports = ["torch"]`**:
  the docs CI job installs `[dev,docs]` only, `autosummary_generate = True`
  imports every documented module, and `nitpicky = True` + `-W` turns an
  unresolvable `torch.*`/pynumero cross-reference into a build failure. Add
  `nitpick_ignore` entries for any framework/pynumero type named in a
  docstring, or do not name them in a cross-referencing role.
- `CHANGELOG.md` — Unreleased entry.

## Pitfalls

Pitfalls 1–8 were each verified against the installed Pyomo 6.10.1 source;
they are not speculative.

1. **`external_model=<instance>` silently shares one model across every time
   index.** `ExternalGreyBoxBlock.__init__` wraps the argument in
   `Initializer(...)`. A *callable* becomes a per-index rule; a
   non-callable `ExternalGreyBoxModel` *instance* becomes a
   `ConstantInitializer`, and every index then shares one object — including
   its cached input values. The result is correct output at the last index
   evaluated and silently wrong derivatives everywhere else, with no error.
   Always pass `lambda b, t: _ExternalModelGreyBox(...)`.
2. **Deactivating the surrogate block does not deactivate its grey box.**
   Pyomo's `BlockData.deactivate()` does not recurse into child components:
   after `surrogate_block.deactivate()`, `surrogate_block.egb[t].active` is
   still `True`. `PyomoNLPWithGreyBoxBlocks` collects blocks with
   `component_objects(ExternalGreyBoxBlock, descend_into=True)` — note the
   missing `active=True` — and filters only on the *data* object's `active`
   flag. So a re-swapped relation leaves the stale grey box in the NLP with
   its link constraints deactivated: extra free variables and extra residual
   equations, no error, a different (and slower) problem than intended. This
   is why §"Files" instructs the narrow `swap_relation` change; a test must
   re-swap **and solve**, not merely assert a flag.
3. **dtype mismatch between validation and solve.** `torch`'s default dtype is
   float32; PyNumero hands the model float64. An `nn.Module` with float32
   weights passes a float32 smoke test and then raises
   `RuntimeError: expected scalar type Double` mid-solve — exactly the failure
   the smoke test exists to prevent. `TorchDriver` coerces once in `__init__`
   and uses `torch.float64` everywhere, validation included.
4. **`evaluate_hessian_outputs` must return the lower-triangular portion
   only**, with shape `(n_inputs, n_inputs)`. `_ExternalGreyBoxAsNLP` raises
   `ValueError('...must return lower triangular portion of the Hessian
   only')` on any entry with `row < col`, and a separate `ValueError` on a
   wrong shape. Easy to get wrong once since the failure only surfaces at
   solve time.
5. **The Jacobian and Hessian sparsity *pattern* must be constant across
   iterations, and `coo_matrix` built from a dense array drops zeros.**
   PyNumero caches `nnz` and the index arrays on the first evaluation and
   thereafter asserts `np.array_equal(jac.row, out.row)`. Combined with
   Pitfall 6 (first evaluation happens at the initial point), a model like
   `f(x) = x**3` evaluated at `x = 0` yields `f'(0) = f''(0) = 0`, an *empty*
   COO matrix, and a permanently-zero derivative or an `AssertionError` on the
   next iteration. This is why §4 builds both matrices from explicit index
   arrays including structurally-zero entries — never from
   `coo_matrix(dense)`.
6. **Grey-box input/output Vars initialize to `None` → 0.0, and must never be
   fixed.** `ExternalGreyBoxBlockData.set_external_model` creates
   `inputs`/`outputs` as plain `Var`s with no value and no bounds;
   `_ExternalGreyBoxAsNLP` substitutes `0.0` for a `None` value and evaluates
   the model, Jacobian and Hessian there before the solve starts. Zero is
   outside the domain of many ordinary models. Set real values in
   `finalize_block_construction` from `probe_point`. Separately,
   `PyomoNLPWithGreyBoxBlocks` raises `NotImplementedError` if any grey-box
   input or output Var is `fixed` — which is also why the unit's own Vars are
   joined by linking constraints rather than passed as
   `set_external_model(inputs=[...])`: a unit Var may legitimately be fixed.
7. **A non-differentiable forward pass breaks silently, but a non-finite
   gradient does not mean non-differentiable.** An in-place op, `.detach()`,
   `.item()`/`float()` cast, or `torch.no_grad()` anywhere in the wrapped
   model returns `None` or a disconnected graph rather than raising;
   `check_differentiable` turns that into a construction-time
   `FlexConfigError` instead of CyIpopt quietly "converging" at a
   non-stationary point. Do *not* extend the check to reject `inf`/`nan`
   gradients: `sqrt`, `log` and `1/x` have infinite derivatives at ordinary
   probe points and would be rejected wrongly. Warn, don't raise.
8. **`egb.inputs` carries no units.** `ExternalGreyBoxBlock`'s own input Vars
   are plain numeric Vars, so the linking constraint is the one place a
   conversion must happen explicitly — and the declared units must be
   *divided out*, as `MultilinearSurrogate.build` does
   (`pyunits.convert(var[t], u) / u`). Writing
   `egb[t].inputs[n] == pyunits.convert(var[t], u)` constructs fine and then
   fails `assert_units_consistent`, which the existing suite already applies
   to whole units.
9. **`model_path` executes imported code from config data.** Same trust
   model as every other `SurrogateSpec.data` (this repo's configs are
   already trusted as code, per `plan/00_conventions.md` §4) — state it in
   the class docstring, no sandboxing built here.
10. **`torch` is a heavy optional dependency.** Keep it out of core deps and
    out of `import flexops.surrogates` for anyone not using this surrogate
    (Specification §2/§5); `[greybox]` is the only place it's declared. Note
    also that the repo's own `.venv` is Python 3.14 while the `flex-pse`
    conda env is 3.13 — confirm a torch wheel exists for whichever
    interpreter you develop against before starting, and pin the CI job to
    3.13 (see "Files").
11. **Cost.** Every `evaluate_jacobian_outputs`/`evaluate_hessian_outputs`
    call re-traces the model, once per time index per IPOPT iteration. That
    is `O(T)` autograd traces per iteration with no caching or batching.
    Acceptable for v1 and small horizons; say so in the docs rather than
    optimizing speculatively.

## Tests

`src/flexops/tests/surrogates/test_external.py` — `unit`, no framework, no
solver (these run on a bare install and must not skip):

- `test_get_driver_resolves_pytorch` — returns the driver class (guard with
  `needs_torch`, since resolution imports it).
- `test_get_driver_reserved_frameworks_raise_not_implemented` — parametrized
  over `tensorflow`/`onnx`/`jax`.
- `test_get_driver_unknown_framework_raises_config_error`.
- `test_importing_flexops_surrogates_does_not_import_torch` — assert
  `"torch" not in sys.modules` after a fresh `import flexops.surrogates` in a
  subprocess. Guards DoD's bare-install line on every CI machine, framework
  installed or not.
- `test_get_regressor_external_model_raises_config_error` (may live beside the
  other `flexparameterize` registry tests instead) — guards §7's `KeyError`.

`src/flexops/tests/surrogates/test_grey_box.py` — `unit`, no solver. The
PyNumero-adapter tests use a **stub driver** (a two-line
`ExternalModelDriver` returning hand-written arrays) so they run without
`torch` and pin the sparse-matrix contract independently of any framework:

- `test_validate_rejects_multiple_output_variables`.
- `test_validate_rejects_probe_point_key_not_in_inputs`.
- `test_validate_resolves_model_path_eagerly` — a bad dotted path raises
  `FlexConfigError` at construction (`needs_torch`).
- `test_jacobian_has_fixed_sparsity_pattern_including_zeros` — a driver whose
  gradient is all zeros still yields `nnz == n_inputs`, with
  `row`/`col` equal to the pattern at a nonzero point. Guards Pitfall 5.
- `test_hessian_has_fixed_sparsity_pattern_including_zeros` — same, with
  `nnz == n*(n+1)//2`.
- `test_hessian_is_lower_triangular_only` — `np.all(H.row >= H.col)`. Guards
  Pitfall 4.
- `test_hessian_scales_with_output_multiplier` — set a multiplier, assert the
  returned data scales by it.
- `test_finalize_block_construction_initializes_inputs_and_outputs` — block
  Vars carry the probe point, and nothing is fixed. Guards Pitfall 6.
- `test_grey_box_models_are_distinct_per_time_index` — build over a 3-point
  index set, assert `egb[0].get_external_model() is not
  egb[1].get_external_model()`. Guards Pitfall 1.
- `test_input_link_constraints_are_units_consistent` —
  `assert_units_consistent` on the built block with a unit whose Var is in a
  different-but-compatible unit than declared. Guards Pitfall 8.

`unit`, `needs_torch`, no solver:

- `test_validate_raises_on_non_differentiable_model` — a model calling
  `.detach()`/`.item()` internally raises `FlexConfigError` at construction,
  not a downstream `None` gradient.
- `test_validate_accepts_model_with_infinite_gradient_at_probe` — e.g.
  `sqrt` at the default probe; must construct, not raise. Guards Pitfall 7's
  false positive.
- `test_float32_module_is_coerced_and_evaluates` — an `nn.Module` built with
  default float32 weights evaluates on float64 input without raising. Guards
  Pitfall 3.
- `test_jacobian_matches_analytic_derivative` — wrap `f(x) = x**3` (known
  closed-form derivative), assert the autograd Jacobian matches to
  floating-point precision. Evaluate away from zero.
- `test_hessian_matches_analytic_second_derivative` — same for the second
  derivative.

`component`, no `needs_` marker (pure Pyomo; this guard protects every user on
a normal install, so it must not be gated on optional packages — and it cannot
be `unit` tier, where the autouse fixture patches `get_solver` to raise):

- `test_get_solver_raises_clear_error_for_grey_box_model` — build a trivial
  model with one `ExternalGreyBoxBlock` and a stub external model; assert
  `FlexSolverError` mentioning `cyipopt`. Also assert that `classify` on that
  model returns `LP`, documenting *why* the guard has to precede it.

`component`, `needs_cyipopt`, `needs_torch`:

- `test_swap_relation_attaches_grey_box_and_solves` — small one-unit model
  (mirror `flexparameterize/tests/helpers.py::build_plant()`'s shape), swap
  to an `external_model` surrogate wrapping a simple known function, solve
  with `SolverFactory("cyipopt")`, assert the result matches a hand-computed
  optimum.
- `test_reswapping_a_grey_box_relation_solves_to_the_same_optimum` — swap
  twice, solve, and assert the objective equals the single-swap result. The
  stale grey box from the first swap must not contribute. Guards Pitfall 2;
  a flag assertion alone would pass while the bug is live.

## Definition of Done

- [ ] `ExternalModelSurrogate` exists, registered in `SURROGATES` under
      `SurrogateType.EXTERNAL_MODEL`, following the same
      validate-eagerly/one-output-variable conventions as every other
      surrogate class.
- [ ] `ExternalFramework`/`get_driver` resolve `pytorch` to a working driver
      and every other member to `NotImplementedError`, mirroring
      `get_regressor`.
- [ ] A bare install (no `[greybox]`) still imports `flexops.surrogates` and
      every other surrogate type cleanly, proven by a subprocess test that
      asserts `torch` is absent from `sys.modules`.
- [ ] Jacobian and Hessian are verified against a known analytic function to
      floating-point precision, not just "the solve converges."
- [ ] Jacobian and Hessian sparsity patterns are fixed and include
      structurally-zero entries; dedicated tests guard both.
- [ ] `evaluate_hessian_outputs` returns lower-triangular only and scales with
      the output multiplier; dedicated tests guard both.
- [ ] Grey-box input/output Vars are initialized from `probe_point` and never
      fixed.
- [ ] One `_ExternalModelGreyBox` per time index, guarded by a test.
- [ ] `check_differentiable` raises `FlexConfigError` (not a downstream
      `None`-gradient failure) on a non-differentiable model, and does **not**
      reject a model with an infinite gradient at the probe point.
- [ ] Re-swapping a grey-box relation solves to the same optimum as a single
      swap (no stale grey box left in the NLP).
- [ ] `get_solver(model=...)` raises a clear `FlexSolverError` for a model
      containing an `ExternalGreyBoxBlock`, naming `SolverFactory("cyipopt")`,
      and the test is not gated on an optional package.
- [ ] `get_regressor(SurrogateType.EXTERNAL_MODEL)` raises `FlexConfigError`,
      not `KeyError`.
- [ ] `cyipopt` and `torch` are both in the optional `[greybox]` extra, never
      core or `[solvers]` dependencies.
- [ ] `needs_cyipopt`/`needs_torch` actually skip — verified by running the
      suite once with the packages absent (tests skip, none error at
      collection) and once with them present (tests **run**; check the pytest
      summary counts, not just a green exit code).
- [ ] The coverage-floors CI job installs `[greybox]` and `src/flexops/*`
      still clears `--fail-under=94`.
- [ ] The exported JSON Schema is regenerated and committed.
- [ ] Reference docs updated for the new classes; `docs/conf.py` mocks
      `torch`; `sphinx-build -W` passes; CHANGELOG updated.
- [ ] plus the generic DoD in `CLAUDE.md`
