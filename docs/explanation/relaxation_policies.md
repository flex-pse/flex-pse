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

When a MINLP capable solver *is* available, it gets used directly. For
cases where a direct MINLP solve is intractable, an explicit relax, solve
MIP, fix, solve NLP pipeline is planned as part of the rolling horizon
scheduling engine, which doesn't exist yet. The `ProblemClass.MINLP_OA`
(outer approximation) and `ProblemClass.MINLP_TR` (trust region) enum
members are reserved strategy slots for that pipeline. They're documented
but deliberately unimplemented today, and `classify` never returns them.
