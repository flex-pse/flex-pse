# Reactor models

flex-pse has two dynamic reactor models that share one base class. Which one
you pick depends on what your data can observe.

| | {py:class}`~flexops.unit_models.reactor.species.SpeciesReactor` | {py:class}`~flexops.unit_models.reactor.lumped.LumpedReactor` |
|---|---|---|
| Use when | every feed and product species is measured or simulated | only flows, level, temperature, pressure, power, and a few outlet compositions are measured |
| Typical data | a detailed reactor model run as step tests, or lab data with full compositions | plant historian tags |
| Accumulation | species holdup in a series of compartments | total volume holdup (level) |
| Fitted structure | residence-time distribution, rate constants, yields | outlet composition relations |

Both are linear when their parameters are fixed, so a schedule that contains
them stays an LP or MILP. Both are written as backward-Euler difference
equations against the time block's `dt`, like every other dynamic unit (see
[Time and dynamics](time_and_dynamics.md)).

## Shared base

{py:class}`~flexops.unit_models.reactor.base.ReactorBase` owns everything the
two models share:

- **Ports.** It builds one inlet port per name in `inlet_names` and one outlet
  port per name in `outlet_names`. The first name in each list is the
  reference port.
- **Intensive states.** Every other inlet's pressure and temperature are held
  at the reference inlet's, and those values pass through to every outlet.
- **Power.** Electrical power is a constant intensity times the reference
  outlet flow, built as the swappable `power_electrical_relation`.
- **Thermal lag (optional).** With `has_thermal_dynamics=True`, the reactor
  carries one lumped temperature that relaxes toward a steady-state value
  with a first-order time constant:

  ```
  T_ss[t] = T_in[t] + thermal_gain * P[t]
  T[t]    = T[t-1] + dt / tau_th * (T_ss[t] - T[t])
  ```

  This is the lumped-capacitance energy balance used for boiler and
  catalyst-bed heating surfaces, `tau dT/dt = -(T - T_ss)`. The steady-state
  relation is registered, so a richer fit can replace it, for example one
  that adds a reaction-heat term. Every outlet leaves at the reactor
  temperature.

## Species reactor: residence-time compartments

The species reactor follows the residence-time-distribution reactor model.
The real vessel is replaced by a plug-flow path discretized into `N` equal,
well-mixed compartments, and the feed enters through side feeds along that
path:

- Feed that enters early, in compartment 1, travels every compartment and
  represents **back-mixing**.
- Feed that enters late, in compartment `N`, leaves almost at once and
  represents **channeling**.

The split of the feed across compartments is therefore the residence-time
distribution itself. Feed species are `inlet_names`, product species are
`outlet_names`, and each species `s` has a holdup in each compartment `i`:

```
holdup[s,i,t] = holdup[s,i,t-1] + dt * ( side_feed[s,i,t]
              + (holdup[s,i-1,t] - holdup[s,i,t]) / tau_c
              + reaction_rate[s,i,t] )
```

- **Outlets.** Each product leaves its own outlet at `holdup[j,N,t] / tau_c`.
  Unreacted feed leaves through the reference outlet.
- **Feed closure.** The last compartment takes whatever feed the others do
  not, so the distribution always conserves feed.

### What is fitted, and what can be swapped

The holdup balance, the feed closure, and the outlet flows are conservation
laws. They are never registered and can never be swapped. Everything the data
has to teach the model is a registered relation:

| Relation | Baseline | What it represents |
|---|---|---|
| `side_feed_<k>_<i>_relation` | `rtd_fraction_<i> * inlet flow` | residence-time distribution |
| `reaction_rate_<k>_<i>_relation` | `-rate_constant_<k> * holdup` | consumption of feed `k` |
| `reaction_rate_<j>_<i>_relation` | `sum_k yield_<j>_<k> * rate_constant_<k> * holdup` | formation of product `j` |

The baseline's scalars (`rtd_fraction_<i>`, `compartment_time`,
`rate_constant_<k>`, `yield_<j>_<k>`) are fixed, regressable parameters. To fit
them, unfix them and minimize the outlet-flow error against data. That makes
the problem an NLP, so solve it with IPOPT. The product of a yield and a rate
constant can only be separated when the unreacted feed on the reference outlet
is measured.

Because each relation is registered per species and compartment, a richer
model can replace any of them in place through `swap_relation`:

- **Nonlinear kinetics.** A fitted rate law, or a neural network of the
  compartment holdups and temperature, can be applied with the same weights
  in every compartment.
- **Flow-dependent distribution.** A side-feed relation of the form
  `h(F) * F` makes the distribution depend on throughput, as it does at part
  load.

Either swap makes the model nonlinear.

## Lumped reactor: volume holdup and composition relations

When the species inside the reactor cannot be measured, the lumped reactor
models what can be:

- **Volume.** Total volume accumulates like a tank:

  ```
  volume[t] = volume[t-1] + dt * (sum of inflows[t] - sum of outflows[t])
  ```

  `level = volume / capacity` is bounded.
- **Flows.** The inlet flows and the reference outlet flow are dispatch
  inputs. Every other outlet takes a fixed, regressable fraction of the total
  outflow.
- **Compositions.** Each configured composition becomes its own output
  `outlet_composition_<name>[t]`, set by the constant relation
  `outlet_composition_<name>_relation`.

That relation is the hook for plant data. A regression of the composition
against measured states, such as reactor temperature, level, or feed flow, is
swapped in with `swap_relation`, and the rest of the model is untouched. A
transfer-function form with lagged terms will use the same hook.

## Choosing a workflow

- **From a detailed model.** Simulate step changes in each feed with the
  detailed model and record every outlet species. Then fit a
  `SpeciesReactor`'s distribution, rate constants, and yields to those
  responses. Space the step changes by a few residence times so the fit sees
  both the transient and the steady state.
- **From plant data.** Map flow, level, temperature, and power tags onto a
  `LumpedReactor`, enable `has_thermal_dynamics` if temperature is measured,
  and fit each composition relation from the tags.
