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
`time_block.dt`, for example `V[t+1] = V[t] + dt * (inflow[t] - outflow[t])`.
This keeps the discrete index arithmetic that unit commitment depends on, keeps
the model a clean MILP/MINLP for the solver facade to classify, and makes the
rolling-horizon hooks (`register_initial_state`, `window`) straightforward. 

To model systems that strongly depend on higher-order PDEs, flexOps will relax by
linearization and solve iteratively to converge. While this does not guarantee 
convergence to a globally optimal solution in all cases, we find that it does 
result in a stable solution for many cases of practical interest. 
