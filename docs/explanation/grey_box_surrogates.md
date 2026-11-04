# External models as grey-box surrogates

Some relationships have no closed form you can write as a Pyomo expression, but
do have code that can report a value and its derivatives at a point. A fitted
neural network is one example. A linearized CFD or process simulator, or a
vendor's sensitivity-aware solver, has the same shape.

{class}`~flexops.surrogates.grey_box.ExternalModelSurrogate` wraps that code as a
PyNumero `ExternalGreyBoxBlock`, so it can act as a unit's registered relation.
No expression is derived. The optimizer only sees numbers: the output and its
derivatives at whatever point it asks about.

Only cyipopt can call back into Python during a solve, so these models solve with
cyipopt. You need the `[greybox]` extra, described in
{doc}`../getting_started/installation`.

## Declaring one

```python
surrogate = ExternalModelSurrogate(
    {
        "framework": "pytorch",
        "model_path": "myproject.models.fouling_model",
        "input_variables": {"flow_in": "m^3/hr", "ambient_temperature": "degK"},
        "output_variables": {"fouling_rate": "1/hr"},
        "probe_point": {"flow_in": 40.0, "ambient_temperature": 298.15},
        "training_range": {"flow_in": [10.0, 80.0]},
    }
)
unit.swap_relation("fouling_relation", surrogate)
```

| Key | Required | Meaning |
|---|---|---|
| `framework` | Yes | The tool that evaluates the model. Only `"pytorch"` exists today. |
| `model_path` | Yes | Dotted import path to a fitted, callable model. |
| `input_variables` | Yes | `{name: units}` for each input, in the order the model expects. |
| `output_variables` | Yes | `{name: units}` with exactly one entry. |
| `probe_point` | No | Where to start and test the model, in declared units. |
| `training_range` | No | `{name: [lower, upper]}` the model was trained over. |

A surrogate like this registers no coefficients, because FlexParameterize can't
regress a wrapped model's internal weights.

```{warning}
**`model_path` runs code.** Building a model from a spec imports the module
`model_path` names, which runs its top-level code, and then calls the resolved
object. Only build specs you trust. Loading and validating a spec doesn't run
anything; only building it does. Spec extensions follow the same rule.
```

## Solving

Use {func}`~flexcore.solvers.get_solver`. It sees the live grey box and
selects cyipopt, whatever `prefer` says:

```python
solver = get_solver(model)
solver.solve(model)
```

A grey box counts as a nonlinear constraint, so
{func}`~flexcore.solvers.classify` reports `NLP`, or `MINLP` when the
model has unfixed integer variables.

### Models with binaries

cyipopt can't enforce integrality, and neither HiGHS, SCIP nor Gurobi can evaluate
a grey box. No solver available here handles both, so a true MIP solve isn't
possible. A direct solve with unfixed integers raises
{class}`~flexcore.exceptions.FlexSolverError`.

For binaries, opt into {class}`~flexcore.solvers.RelaxAndFix`:

```python
solver = get_solver(model, strategy=RelaxAndFix())
solver.solve(model)
```

With a grey box, this runs a dive:

1. Relax the binaries to `[0, 1]` and solve the NLP.
2. Fix every binary within `fix_tol` of 0 or 1. If none qualify, fix the one
   closest to an integer.
3. Re-solve, and repeat until every binary is fixed.

A constraint left with no free variables is set aside for that step, since IPOPT
would count it as one equation too many. If fixing has violated it, the step is
reported as infeasible without calling the solver.

This is a heuristic. It doesn't backtrack, so a poor early decision can end in an
infeasible step, which is reported rather than raised. Each pass is a full NLP
solve. The report's `bound` and `gap` are `None`, because a local NLP optimum
doesn't bound the true optimum. General integer variables aren't supported; fix
them first.

### Calling cyipopt directly

`SolverFactory("cyipopt")` still works but skips these protections:

- An unfixed integer variable ends the Python process, with nothing to catch.
- A grey box under a deactivated parent block stays in the NLP unless it's
  deactivated itself, because PyNumero only reads the grey box's own active flag.
- An infeasible solve overwrites the model's values.
- Nothing checks the training range unless you call
  {func}`~flexcore.solvers.check_training_range` yourself.

## Starting point and convergence

The solver starts each input at its `probe_point` value. Without one, it uses the
midpoint of `training_range`, and without that, 1.0. The surrogate also evaluates
the model at this point when it's constructed. A non-finite output, a wrong number
of inputs or a non-scalar output raises
{class}`~flexcore.exceptions.FlexConfigError`. A plain function that accepts any
input length can't be caught this way.

A value of 1.0 is often unphysical, such as 1 K, so set `probe_point` to a typical
operating point.

IPOPT finds local optima and assumes the model is twice differentiable. Two things
follow:

- **Prefer smooth activations.** ReLU networks have kinks and zero curvature, which
  can hurt convergence. Use `tanh`, `softplus` or similar.
- **Infeasible may not mean infeasible.** From a poor start, IPOPT can fail in its
  restoration phase and report a feasible model as infeasible. Try a better
  `probe_point` before concluding the model has no solution.

Two alternatives exist outside this surrogate. A trust-region filter, such as
Pyomo's `trustregion` (Eason and Biegler), has stronger convergence guarantees from
an arbitrary start and milder smoothness assumptions. For ReLU networks, OMLT's
exact MILP encoding avoids the smoothness problem and gives a global optimum when
the rest of the program is linear.

## Training range

A fitted model is only trustworthy near its training data. `training_range` records
that range, and after a facade solve, the final solution's inputs are checked
against it. Each input outside its range logs a warning and appears on
`solver.last_report.out_of_range`.

The range is not a constraint. The optimizer can still leave it, and the warning
tells you that it did. To forbid extrapolation, add bounds to the linked unit
variables.

## How the model is evaluated

A **driver** evaluates one model and its first two derivatives at a point. The
`framework` key selects it. PyTorch, through `torch.autograd`, is the only driver
today, but the abstraction isn't limited to neural networks: a CFD model exposing
its own adjoint computation would be a new driver, not a new kind of surrogate.
A driver's framework is imported only when it's needed, so importing
`flexops.surrogates` never imports `torch`.

The PyTorch driver works on a float64, eval-mode copy of an `nn.Module`, so your
module keeps its dtype, training mode and batch-norm statistics. A plain function
is called as is, so any modules it captures are your responsibility to put in eval
mode.

The copy stays on the device your model is on. To run on GPU, call `model.to("cuda")`
on the object `model_path` names. Results always return to CPU, so the optimization
itself stays on CPU.

Each time step holds its own grey box, which remembers its evaluations at the
current point and recomputes only when the point changes. The model still runs once
per time step per iteration, so a GPU mainly pays off for a large model.

## Known limits

- **One output per grey box.** A multi-output model has to be wrapped once per
  output and evaluated separately for each.
- **One grey box per time step, with no batching.** Solve time grows with horizon
  length, and long horizons may be dominated by model evaluation.
- **macOS with conda.** Two OpenMP runtimes can end up loaded, from conda and from
  the `torch` wheel. Importing `torch` before `flexcore` or `flexops` can then crash
  the solve with a segmentation fault, and a model using `BatchNorm` can hang during
  its derivative pass. Import `flexops` first, and set `OMP_NUM_THREADS=1` if a hang
  remains. If `torch` fails to start with `OMP: Error #15`, also set
  `KMP_DUPLICATE_LIB_OK=TRUE`.
