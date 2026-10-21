# Relaxation policies

**Classify loudly, never transform silently.** The solver facade
(`flexcore.solvers.get_solver`) classifies a model into a `ProblemClass` and
selects the best available solver for that class. It **never** relaxes
integrality, decomposes the problem, or sets up trust regions on its own. A
relaxed MIP schedule sent to a real plant is a correctness hazard. Any such
transformation must be requested explicitly. Nothing gets applied behind
your back.

The default open source solver stack routes each class to a genuine solver
for that class. HiGHS handles LP, **SCIP** handles MILP and MINLP, and
IPOPT (built from idaes with the HSL ``ma27`` linear solver when available)
handles NLP. Solving a MINLP directly with a MINLP capable solver isn't the
silent transformation this facade refuses to do. It's the sanctioned path
("install a MINLP capable solver"). What the facade will never do on its
own is relax integrality to make a MINLP fit a MILP solver.

Concretely, when a model classifies as MINLP and **no** MINLP capable
solver is installed (SCIP absent and none registered), `get_solver` raises
`FlexSolverError` instead of quietly relaxing.

> this model is MINLP; compose a `flexschedule.SolveSequence` (relax -> MIP ->
> fix -> NLP) or install a MINLP-capable solver.

When a MINLP capable solver *is* available, it gets used directly.

Relaxing integrality is something you ask for, never a default. The
`RelaxAndFix` strategy (`get_solver(..., strategy=RelaxAndFix())`) solves the
relaxation, fixes the binaries it already decided, then **restores integrality**
and solves the rest exactly. What it returns is an integer-feasible schedule,
not a relaxed one, along with its distance from the relaxation's lower bound.
See [Solve strategies](solve_strategies.md).

The `ProblemClass.MINLP_OA` (outer approximation) and `ProblemClass.MINLP_TR`
(trust region) enum members are reserved slots for future strategies. They're
documented but deliberately unimplemented today, and `classify` never returns
them.
