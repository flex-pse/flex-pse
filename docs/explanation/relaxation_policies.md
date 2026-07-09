# Relaxation policies

**Classify loudly, never transform silently (decision R5).** The solver facade
(`flexcore.solvers.get_solver`) classifies a model into a `ProblemClass` and
selects the best available solver for that class. It **never** relaxes
integrality, decomposes the problem, or sets up trust regions on its own. A
relaxed MIP schedule sent to a real plant is a correctness hazard, so any such
transformation must be requested explicitly, not applied behind the user's back.

The default open-source solver stack routes each class to a genuine solver for
that class — HiGHS for LP, **SCIP** for MILP and MINLP, and IPOPT (built from
idaes with the HSL ``ma27`` linear solver when available) for NLP. Solving a
MINLP directly with a MINLP-capable solver is not the silent transformation R5
forbids; it is the sanctioned path ("install a MINLP-capable solver"). What R5
rules out is the facade *itself* relaxing integrality to make a MINLP fit a MILP
solver.

Concretely, when a model classifies as MINLP and **no** MINLP-capable solver is
installed (SCIP absent and none registered), `get_solver` raises
`FlexSolverError` rather than quietly relaxing:

> this model is MINLP; compose a `flexschedule.SolveSequence` (relax -> MIP ->
> fix -> NLP) or install a MINLP-capable solver.

When a MINLP-capable solver *is* available it is used directly, and relaxation
strategies remain available **only** in `flexschedule.SolveSequence` (milestone
M12) — for cases where a direct MINLP solve is intractable — where a relax →
solve-MIP → fix → solve-NLP sequence is an explicit, inspectable pipeline. The `ProblemClass.MINLP_OA` (outer approximation) and
`ProblemClass.MINLP_TR` (trust region) enum members are reserved strategy slots,
documented but deliberately unimplemented in v0; `classify` never returns them.

M12 extends this page with the concrete `SolveSequence` steps and per-step
failure policies.
