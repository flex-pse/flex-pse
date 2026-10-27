# Data center and cooling models

This page sets out the equations behind the server and cooling unit
models, {class}`~flexops.unit_models.datacenter.server.Server`,
{class}`~flexops.unit_models.cooling.space_cooling.SpaceCooling` and
{class}`~flexops.unit_models.cooling.chiller.Chiller`, what problem class they
produce, and the assumptions that limit them.

## What the units represent

A data center is a plant of parallel servers plus the cooling that removes their
heat. Each server is its own `Server` unit, so servers can differ in hardware,
utilization and on/off status. Each cooling unit is its own `SpaceCooling` or
`Chiller`. None of them has fluid ports, and each registers its own
`power_electrical[t]`, so the plant total is the sum of the server and cooling
draws.

The units do not reference each other. A flowsheet ties them together by
setting each cooling unit's `cooling_load` input, for example:

```python
m.site.r1 = fo.Server(utilization=u1, idle_power=4 * pyunits.kW, ...)
m.site.r2 = fo.Server(utilization=u2, idle_power=5 * pyunits.kW, ...)
m.site.crah = fo.SpaceCooling(cop=4.0, ancillary_intensity=0.08)
m.site.chiller = fo.Chiller(cop=5.0)

@m.site.Constraint(m.time_block.time_index)
def room_heat(b, t):
    return b.crah.cooling_load[t] == b.r1.air_heat[t] + b.r2.air_heat[t]

@m.site.Constraint(m.time_block.time_index)
def liquid_heat(b, t):
    return b.chiller.cooling_load[t] == b.r1.liquid_heat[t] + b.r2.liquid_heat[t]
```

A complete facility, including UPS and transformer losses and miscellaneous
load, is built in the `flex-pse-examples` repository.

Every time-indexed quantity is the **mean over one TimeBlock step**, and
`FlexCosting` multiplies mean power by the step for energy.

## Nomenclature

| Symbol | Meaning | Units | Type | Model name |
|---|---|---|---|---|
| {math}`u_t` | server utilization | – | input (data or live component) | `utilization[t]` |
| {math}`P^{idle}, P^{peak}` | IT power at zero and full utilization | kW | fitted parameter | `idle_power`, `peak_power` |
| {math}`\gamma` | server power curve exponent | – | parameter | `utilization_exponent` |
| {math}`\dot m_t` | server air mass flow | kg/s | data | `airflow[t]` |
| {math}`P^{fan}_{rated}, \dot m_{rated}` | server fan power at its rated airflow | kW, kg/s | parameter | `fan_rated_power`, `rated_airflow` |
| {math}`\lambda` | PDU loss per kW of IT power | – | fitted parameter | `pdu_loss_fraction` |
| {math}`\chi` | share of server heat to the liquid loop | – | fitted parameter | `liquid_fraction` |
| {math}`P^{IT}_t` | server IT power while running | kW | derived | `it_power[t]` |
| {math}`Q_t` | heat the server releases | kW | derived | `heat_output[t]` |
| {math}`L_t` | cooling load | kW | input | `cooling_load[t]` |
| {math}`\mathrm{COP}` | coefficient of performance | – | fitted parameter | `cop` |
| {math}`\kappa` | fan and pump power per kW of cooling load | – | fitted parameter | `ancillary_intensity` |

"Fitted parameter" means a fixed value in a schedule optimization that
parameter estimation can regress from plant data.

## Server

### IT power

```{math}
P^{IT}_t = P^{idle} + (P^{peak} - P^{idle})\,g_t + P^{fan}_t
```

With {math}`\gamma = 1` the curve term is the utilization itself,
{math}`g_t = u_t`. This is the linear idle-to-busy server power model reported
by Fan, Weber and Barroso. Any other exponent is a power-law generalization,
{math}`g_t = u_t^{\gamma}`, computed from data when the model is built
(`utilization_curve[t]`). Server fan power follows the cube law:

```{math}
P^{fan}_t = P^{fan}_{rated}\,(\dot m_t / \dot m_{rated})^3
```

`it_power_relation` is registered, so it can be swapped for a fitted surrogate.

### Draw and heat

The server draws its IT power plus a linear PDU loss, and all IT power leaves as
heat, split between room air and the liquid loop:

```{math}
P_t = (1 + \lambda)\,P^{IT}_t, \qquad Q_t = P^{IT}_t,
\qquad Q^{air}_t = (1 - \chi)\,Q_t, \qquad Q^{liq}_t = \chi\,Q_t
```

### Utilization: data or a live component

`utilization` takes one of two forms:

- **Data**, a length-T sequence. `utilization[t]` is a fixed Var holding it.
  Change it in place with `set_value`. With an exponent other than 1, call
  `update_parameters` afterwards so `utilization_curve` is recomputed.
- **A live time-indexed component** (Param, Var or Expression) built elsewhere
  on the model, such as a curtailment or load-shifting decision. The server leaves
  `utilization[t]` free and adds `utilization_definition[t]` tying it to the
  source. The server bounds utilization to [0, 1], so an out-of-range source is
  infeasible rather than clipped. A live source needs
  `utilization_exponent == 1`, because the nonlinear curve is computed from
  data.

`emit_model` writes a live source's current values, so the rebuilt spec gives a
data server with that snapshot and warns that the live link itself can't be
written.

Lowering utilization reduces only the dynamic part of IT power. Idle power and
fans keep running. To shut a server down, use its status.

### On/off status

`unit_commitment=UnitCommitmentConfig(status=True)` (with `rated_power`) gives a
server a status binary through the standard `add_status`, so `relax`/`unrelax`
and `add_startup_shutdown` work unchanged. Big-M links with
{math}`M = P^{rated}` gate the whole server draw and the heat it releases:

```{math}
P_t \le P^{run}_t, \quad P_t \ge P^{run}_t - M(1 - s_t), \quad 0 \le P_t \le M s_t
```

```{math}
Q_t \le P^{IT}_t, \quad Q_t \ge P^{IT}_t - M(1 - s_t), \quad Q_t \le M s_t
```

So an off server draws nothing and heats nothing, and an on server is identical to
the ungated build. The unit keeps `allows_status_element = False`, since a spec
`status` element would gate `power_electrical` without the heat links.

## Cooling units

Both cooling units remove their cooling load at a constant COP, plus fan and
pump power proportional to the load:

```{math}
P^{comp}_t\,\mathrm{COP} = L_t, \qquad P_t = P^{comp}_t + \kappa\,L_t
```

`power_compressor_relation` is registered, so a fitted COP curve can replace
it. `SpaceCooling` (for example a CRAH or CRAC) defaults to a COP of 3 and
`Chiller` to a COP of 5, both with {math}`\kappa = 0.05`. These defaults are
illustrative placeholders, not values from a source; calibrate them to the
equipment.

## Problem class

With a linear server curve or data utilization, every unit is an **LP**.
Server status makes the model a MILP.

## Assumptions and limits

- Server power, heat and cooling are quasi-steady at the time step; thermal
  inertia of room air and equipment is neglected.
- Within-step variation is not represented. A nonlinear curve (power-law
  exponent, fan cube law) applied to a step mean is biased low
  when its input varies inside the step, so prefer short steps or near-linear
  curves when inputs fluctuate.
- Airflows are exogenous, and COP does not vary with temperature or part load.
- No server inlet temperature, air mixing or leakage power; no economizer,
  chiller staging or thermal storage.

## Worked example

Two servers with idle and peak power of 4 and 12 kW, 0.3 kW fans rated at
1 kg/s, 0.9 kg/s airflow, utilization 0.5 and 0.7, and a 2% PDU loss. All their
heat goes to a `SpaceCooling` unit with COP 3 and {math}`\kappa = 0.05`. For
one step:

| Quantity | Value |
|---|---|
| Server fan power, {math}`0.3 \times 0.9^3` | 0.2187 kW |
| Server IT power, {math}`4 + 8u + 0.2187` | 8.2187, 9.8187 kW |
| Server draw, IT × 1.02 | 8.3831, 10.0151 kW |
| Cooling load = total IT power | 18.0374 kW |
| Compressor power, load / 3 | 6.0125 kW |
| Fans and pumps, 0.05 × load | 0.9019 kW |
| Site total, and PUE = total / IT | 25.3126 kW, 1.40 |

## References

- X. Fan, W.-D. Weber and L. A. Barroso, "Power provisioning for a
  warehouse-sized computer," *Proc. 34th ACM International Symposium on
  Computer Architecture (ISCA)*, San Diego, 2007. Linear idle-to-busy server
  power model.
