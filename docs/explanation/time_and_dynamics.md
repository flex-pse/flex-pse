# Time and dynamics

flex-pse models time as a **discrete, ordered set of integer indices**. It
never uses a continuous DAE. A {py:class}`~flexops.core.time_block.TimeBlock`
builds an ordered Pyomo `Set` of indices `0..N-1`, a step size `dt` that
carries units, and a `pandas.DatetimeIndex` mirror for aligning tariffs and
other time series. Time points mark interval starts. Point `i` is the
timestamp `start_date + i * dt`, and `end_date` is the exclusive end of the
horizon. A single `TimeBlock` spans at most one calendar month. Longer
studies compose several `TimeBlock`s through the rolling horizon driver or
the design mode wrapper.

**Why discrete, and never `dynamic=True` / Pyomo.DAE?** The scheduling
problems flex-pse targets are unit commitment problems. They carry binary
status variables and constraints expressed over integer index arithmetic.
Minimum up and down (dwell) times, startup delays measured in a whole
number of steps, and rolling horizon windows that slice the index set all
need `t`, `t-1`, and `t+k` to be plain integers, so that difference
equations and dwell counters stay exact. So every dynamic relationship in
flex-pse, tank holdup, battery state of charge, gets written by hand as a
*difference equation* against `time_block.dt`. It always takes the
**backward (implicit)** form. The state ending period `t` is written in
terms of the rates sampled at `t`, indexed `t = 1 … N-1`, with the initial
condition as its own constraint. For tank holdup that reads
`V[t] = V[t-1] + dt * (inflow[t] - outflow[t])`. One direction, applied
everywhere. No difference equation in the codebase needs to be re-derived
just to be read. This keeps the discrete index arithmetic that unit
commitment depends on, keeps the model a clean MILP/MINLP for the solver
facade to classify, and keeps the rolling horizon hooks
(`register_initial_state`, `window`) simple.

## Composition inherits the same time set

Both composition levels, a `PlantBlock` (a collection of **units**) and a
`NetworkBlock` (a composition of **plants**), are thin IDAES
`FlowsheetBlockData` subclasses. Both build with `dynamic=False` and install
the `TimeBlock`'s ordered integer `Set` as their time domain *by reference*,
so `plant.time is time_block.time_index`. Nothing in flex-pse ever
constructs a `ContinuousSet`.

Both take the TimeBlock **explicitly**.

```python
m.waterfacility = fo.PlantBlock(time_block=m.time_block)
```

Passing `time_block=` explicitly is deliberate. Omitting it is a documented
convenience, not the primary path. It works when the model carries exactly
one `TimeBlock`, and raises `FlexConfigError` on zero or several, telling
you to name the one you meant. A design mode study composes *several*
models, each with its own `TimeBlock`. That's exactly the case where
implicit discovery would pick the wrong one.

Some systems depend heavily on higher order PDEs. For those, flex-pse
relaxes the problem through linearization and solves it iteratively toward
convergence. This doesn't guarantee a globally optimal solution in every
case. In practice, though, it produces a stable solution for most cases
you'll run into.
