# Time and dynamics (decision R2)

flex-pse models time as a **discrete, ordered set of integer indices**, never as
a continuous DAE. A {py:class}`~flexops.core.time_block.TimeBlock` builds an
ordered Pyomo `Set` of indices `0..N-1`, a unit-carrying step size `dt`, and a
`pandas.DatetimeIndex` mirror for aligning tariffs and other time series. Time
points are interval starts: point `i` is the timestamp `start_date + i * dt`,
and `end_date` is the exclusive end of the horizon. A single `TimeBlock` spans
at most one calendar month; longer studies are composed from several `TimeBlock`s
by the rolling-horizon driver or the design-mode wrapper.

**Why discrete, and never `dynamic=True` / Pyomo.DAE?** The scheduling problems
flex-pse targets are unit-commitment problems: they carry binary status
variables and constraints expressed over integer index arithmetic — minimum
up/down (dwell) times, startup delays measured in a whole number of steps, and
rolling-horizon windows that slice the index set. These formulations need `t`,
`t-1`, and `t+k` to be plain integers so that difference equations and dwell
counters are exact. Instead, every dynamic relationship in flex-pse — tank holdup, 
battery state of charge — is written by hand as a *difference equation* against
`time_block.dt`, always in the **backward (implicit)** form: the state ending
period `t` in terms of the rates sampled at `t`, indexed `t = 1 … N-1`, with the
initial condition as its own constraint. For tank holdup that is
`V[t] = V[t-1] + dt * (inflow[t] - outflow[t])`. One direction, applied
everywhere (conventions §2), so no difference equation in the codebase has to be
re-derived to be read. This keeps the discrete index arithmetic that unit
commitment depends on, keeps the model a clean MILP/MINLP for the solver facade
to classify, and makes the rolling-horizon hooks (`register_initial_state`,
`window`) straightforward.

## Composition inherits the same time set

Both composition levels — a `PlantBlock` (a collection of **units**) and a
`NetworkBlock` (a composition of **plants**, R7) — are thin IDAES
`FlowsheetBlockData` subclasses built with `dynamic=False` and the `TimeBlock`'s
ordered integer `Set` installed as their time domain *by reference*, so
`plant.time is time_block.time_index`. Nothing in flex-pse ever constructs a
`ContinuousSet`.

Both take the TimeBlock **explicitly**:

```python
m.waterfacility = fo.PlantBlock(time_block=m.time_block)
```

This is a deliberate correction to the original slide API, which left the time
domain implicit (architecture §3.3). Omitting `time_block=` is a documented
convenience, not the primary path: it works when the model carries exactly one
`TimeBlock` and raises `FlexConfigError` on zero or several, telling you to name
the one you meant. A design-mode study composes *several* models, each with its
own `TimeBlock`, which is precisely the case where implicit discovery would pick
the wrong one.

To model systems that strongly depend on higher-order PDEs, flexOps will relax by
linearization and solve iteratively to converge. While this does not guarantee 
convergence to a globally optimal solution in all cases, we find that it does 
result in a stable solution for many cases of practical interest. 
