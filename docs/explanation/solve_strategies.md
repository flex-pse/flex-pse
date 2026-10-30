# Solve strategies

A scheduling model with on/off decisions is a mixed-integer program. A month
of 15-minute steps across a handful of units holds tens of thousands of binary
variables, and solving that exactly can take hours. Most of those binaries are
not real choices, though. A unit that runs flat out through every off-peak
night is not "deciding" anything at 03:00. The decisions that matter cluster
around tariff boundaries and demand swings.

A **solve strategy** is how flex-pse exploits that structure without asking
you to rewrite the model. You choose one when you get the solver:

```python
from flexcore.solvers import RelaxAndFix, get_solver

solver = get_solver(model=m, strategy=RelaxAndFix())
results = solver.solve(m, mip_gap=0.002, time_limit=1800)
print(solver.last_report.gap)
```

The default strategy, `Direct`, solves the model once, exactly as built.

## Relax and fix

`RelaxAndFix` runs in three stages.

1. **Relax.** Every binary that isn't already fixed may take any value between
   0 and 1, and the model is solved. This is a linear program (or a nonlinear
   one if the model has nonlinear constraints) and it solves quickly. Its cost
   is a **lower bound** on the true cost, because it optimizes over a superset of
   the real schedules.
2. **Fix.** A binary that the relaxation put within `fix_tol` of 0 or 1 is
   treated as decided and fixed there. Those are the long flat runs.
3. **Re-solve.** Integrality is restored and the much smaller mixed-integer
   program is solved over the binaries that came back fractional. Those are the
   steps where the plant is genuinely trading a unit's output against the tariff.

When the solve finishes, the model is left as you built it: every binary is
binary again and nothing the strategy fixed stays fixed. Only the solution
values remain.

### The fixing ladder

Fixing can rule out every feasible schedule. For example, pinning a unit on at
one step can forbid the restart that was the only way to meet demand. An
infeasible re-solve is therefore not accepted as the answer. The strategy
retries with tolerances `fix_tol`, `fix_tol/2` and `fix_tol/4`, then `0`. Each
retry starts from the same relaxed values and hands more decisions back to the
solver. The last rung fixes nothing, so it is the exact problem. Every rung is
recorded in `solver.last_report.steps`.

### How good is the answer?

Relax and fix is a heuristic. The schedule it returns satisfies every
constraint, but it is not proven optimal. Use `last_report.gap` to judge it:
the distance between the returned cost and the relaxation's lower bound. The
true optimum lies somewhere between the two. The solver's own MIP gap from the
final stage measures something different. It only covers the decisions left
free after fixing, and says nothing about the ones that were fixed.

`fix_tol` trades speed against quality. With a larger tolerance the relaxation
decides more of the schedule, so the final solve is faster, but it can lock in
choices a fractional schedule liked and a real one would not. The default of
`0.1` is a reasonable starting point for unit commitment.

If the relaxation is nonlinear and solved by a local solver such as IPOPT, its
cost is a local optimum and **not** a guaranteed lower bound, so the reported
gap is only indicative. Global solvers such as Gurobi do not have this
problem.

## Warm starting

`solve(m, warmstart=True)` gives the solver the model's current values as a
starting incumbent. It pays off when the previous solution is nearly right
for the new problem: after retargeting a demand parameter, moving a slider, or
stepping a rolling horizon. It does nothing for a first solve, which has no
previous solution to start from. A solver that can't warm start logs a warning
and solves cold. A warm start only speeds up the search; the problem being
solved is the same, so falling back never changes the answer.

`RelaxAndFix` does not warm start. Its final stage starts from relaxed,
fractional values, which make a poor starting incumbent.

## Adding a strategy

A strategy is a small frozen dataclass whose `solve` method receives the model,
its problem class and a `run_step` function. `run_step` solves the model once as
a given problem class, using the facade's solver, settings and options. The
strategy decides what to change between steps and returns the final results
together with a `SolveReport`. Strategies never choose solvers or options
themselves, which keeps new ones (rolling horizon, decomposition) short.
