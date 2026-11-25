# Time points, periods, and measured data

flex-pse is a scheduler. At each time point it asks what the plant should do
over the coming step so that it ends the step in the best state. Its time
points are defined around that question, so they don't read like sensor
readings. This page covers what a time point means, how the storage equations
use it, and how to line measured data up with it.

## A time point is the step that starts there

A {py:class}`~flexops.core.time_block.TimeBlock` with `start_date`
`2025-07-08 17:00` and a one-hour `time_step` has these points:

| Index `t` | Timestamp | Covers           |
|-----------|-----------|------------------|
| `-1`      | (none)    | before the horizon; the initial state |
| `0`       | 17:00     | 17:00 to 18:00   |
| `1`       | 18:00     | 18:00 to 19:00   |
| `2`       | 19:00     | 19:00 to 20:00   |

Point `t` is stamped `start_date + t * dt` and covers the step
`[start_date + t * dt, start_date + (t + 1) * dt)`. The last point starts one
step before `end_date`.

What a value at point `t` means depends on the kind of quantity:

- A setpoint or decision at `t` (pump flow, battery charge power, unit on or
  off) is what the plant is told to do now, at 17:00. It holds for the whole
  step, until 18:00.
- A rate at `t` (power, flow, fuel use) is the average over the step, 17:00
  to 18:00. Because the setpoint is held, the average and the setpoint are the
  same number.
- A state at `t` (tank volume, battery charge, state of charge) is the level
  at the *end* of the step: at 18:00, once the step's flows have acted.
- A price or tariff charge at `t` is the price over the step, 17:00 to 18:00.

```text
   initial state             volume[0]           volume[1]
   (volume[-1])                  |                   |
        |                        v                   v
  ------+------------------------+-------------------+------->
      17:00                    18:00               19:00
        |<---- step t = 0 ------>|<--- step t = 1 -->|
         flow[0], power[0],        flow[1], power[1],
         price[0]: held / averaged price[1]
```

## How the backward difference uses this

Every storage balance in flex-pse has the same form. For a tank:

```text
volume[t] = volume[t-1] + dt * (flow_in[t] - flow_out[t])     for t = 0 ... N-1
volume[-1] = initial_volume
```

The level at the end of step `t` is the level at the end of the step before,
plus what the step's flows add. For the horizon above:

- `initial_volume` is the level at 17:00, when the schedule starts. It stands
  in for `volume[-1]`, the end of the step before the horizon.
- `flow_in[0]` and `flow_out[0]` are the flows the schedule sets at 17:00,
  held until 18:00.
- `volume[0]` is the level at 18:00, after that hour of flow, not the level
  at 17:00.

The battery's `charge[t]` works the same way, with `charge_init` in place of
`charge[-1]`. Because the balance covers `t = 0`, the first step's flows are
always counted.

The name *backward* difference comes from relating `volume[t]` to the point
before it, `t - 1`, through the rates sampled at `t`. This is the implicit
(backward Euler) form with rates held constant over each step. The timestamps
themselves look forward: the rates at `t` describe the step that starts at
the timestamp.

## Why flex-pse defines time this way

Several scheduling and tariff tools use the same convention.

A schedule is a sequence of setpoints. The one stamped 17:00 is sent at 17:00
and runs the plant until 18:00, and its effect on the level and the cost shows
up by the end of the step. The optimizer picks today's setpoints for their
effect on the future. Model predictive control and a rolling-horizon scheduler
ask the same question at each step, so the first point of each window is the
decision to apply now.

Retail tariffs are billed the same way. Time-of-use windows are clock spans
such as "peak is 16:00 to 21:00". EECO, which prices flex-pse's electricity
and gas, puts a step in a window when its start hour falls in it
(`hour_start <= hour < hour_end`). The steps stamped 16:00 through 20:00 are
therefore billed as peak, which covers exactly 16:00 to 21:00. Green Button
(NAESB ESPI), the format for utility interval meter data, also stamps each
reading with its interval's start time and duration.

Other energy-system optimization models write storage the same way. PyPSA
sets the level at snapshot `t` from the level at `t - 1` plus that snapshot's
dispatch, with the initial level at `t = -1`, and UnitCommitment.jl's storage
transition has the same form. In both, the level at a step is the level once
that step's flows have acted.

`FlexCosting` passes EECO the `TimeBlock`'s own start, end, and step, so
EECO's charge array at position `t` prices `power[t]`.
`flexops/tests/costing/test_time_alignment.py` checks the stamps EECO bills
on, the peak window's edges at 15:00, 16:00, 20:00, and 21:00, and the bill
for a one-hour load at each of those steps: energy charges after the fact and
in a solved model, and a peak demand charge after the fact.

### Supplying prices that line up

Where you can, describe pricing as an EECO tariff (`tariff` or `tariff_file`
on `FlexCosting`). Time-of-use energy charges, demand charges, and export
charges are then written as clock windows, and EECO places every step by its
start using the same rule, so the windows line up with the `TimeBlock` without
any shifting on your part. The tests cover energy and demand charges at the
window edges.

A price series you pass yourself as `energy_prices` or `export_prices` gets no
such check. flex-pse reads it by position, one value per time point, and
ignores any timestamps it carries, so the series has to be aligned before you
pass it. This matters most for wholesale electricity prices, which markets
usually label by the hour's end. In ISO and RTO data such as CAISO's, "hour
ending 17" (HE17) is 16:00 to 17:00, which in flex-pse is the step stamped
16:00. Move a series labeled by hour ending one step earlier, then select the
horizon's steps from it:

```python
prices.index = prices.index - dt
energy_prices = {"electrical": prices.reindex(time_block.datetime_index).to_numpy()}
```

Demand response is not covered by either route yet. `dr_event_file` loads a
program into a container, but flex-pse builds no demand-response constraints
or credits from it, and EECO 0.4.2 has no demand-response API.

## Lining measured data up with time points

Sensors and historians usually report either what is happening at an instant
("the tank is at 300 m³ at 17:00") or the average over the interval that just
ended ("the 17:00 power reading is the average from 16:00 to 17:00"). Neither
is what a flex-pse time point holds, so convert the data first. If you don't,
every value lands one step late against the tariff, and a fitted model
predicts the wrong hour.

The timeline below sets one hour of a historian next to the flex-pse step
that covers it:

```text
  clock              16:00                         17:00
  -------------------+-----------------------------+------------------->
  flex-pse           |<---- step stamped 16:00 --->|
                     |  power[t]: held/averaged    | charge[t]: the level
                     |  over 16:00 to 17:00        | here, at 17:00
                     |                             |
  sensor, instant    o                             o
                     reading at 16:00:             reading at 17:00:
                     the level when the step       the level when the step
                     starts = charge[t-1]          ends = charge[t]
                     |                             |
  historian average  |.... samples averaged ......>*
  (hour ending)      |                             stamped 17:00, but it is
                                                   the 16:00 step's power[t]
```

The historian's 17:00 average and the sensor's 17:00 reading carry the same
stamp but describe different things. The average belongs to the flex-pse step
stamped 16:00. The reading is the level at the end of that step, `charge[t]`
for the step stamped 16:00, which is also the level the 17:00 step starts
from.

### Example: a battery through a peak

The figure follows one afternoon of a battery (100kW capacity 400kWh charge/discharge rate) operating under a 16:00
to 21:00 peak pricing structure. The schedule makes the battery charge at 100 kW for the three hours before the
peak, then discharges it at 60 kW through the peak. 

Note: The data is synthetic imitating a sensor that
sampled every minute and tracks each setpoint with a three-minute lag and
some noise, and a historian that keeps hourly averages of power stamped at
each hour's end, plus an on-the-hour reading of the charge.

```{image} ../_static/figures/time_alignment_battery_light.svg
---
class: only-light
alt: >-
  Three panels sharing a clock-time axis from 13:00 to 23:00. Price steps
  from $0.09 to $0.18 at 16:00 and back at 21:00. In the power panel the
  flex-pse schedule steps from +100 kW to -60 kW at 16:00 and to 0 at 21:00,
  with a dot at the start of each step. The historian's hourly averages,
  stamped at each hour's end, sit one hour to the right. Used unshifted, they
  trace the same steps one hour late, so the 16:00 to 17:00 hour shows +100 kW
  instead of -60 kW. In the charge panel the flex-pse charge is drawn as
  steps. Each step holds the level at its end, so its right edge meets the
  historian's on-the-hour reading: the step stamped 15:00 holds 400 kWh and
  ends on the 16:00 reading.
---
```

```{image} ../_static/figures/time_alignment_battery_dark.svg
---
class: only-dark
alt: >-
  The same three panels in dark colors: price, battery power with the
  flex-pse schedule and the hour-ending historian averages, and battery charge
  with the flex-pse steps ending on the historian's readings.
---
```

In the price panel, the steps stamped 16:00 to 20:00 are billed as peak,
which covers exactly 16:00 to 21:00 time window.

In the power panel, the blue dot at 16:00 and the orange diamond at 17:00
describe the same hour. If you put the historian's averages into flex-pse
unshifted (the dashed line), the whole series runs one hour late: the model
would think the battery was still charging from 16:00 to 17:00, at the peak
price, and kept discharging after the peak ended. Shifted back one step, the
series lies on the schedule.

In the charge panel, `charge[t]` is drawn as steps, like power, because
flex-pse keeps one value per step. That value is the level the step ends at,
so each step's right end lands on the historian's reading at
`timestamp + dt`. The step stamped 15:00 holds 400 kWh and ends on the 16:00
reading. The steps show how the model records the level; the thin sensor line
shows the real level ramping across each step. Compared stamp for stamp, the
offset runs the other way from power, with the flex-pse charge one hour
ahead: at 15:00 the model holds 400 kWh while the historian reads 300. Compare
`charge[t]` with the reading at `timestamp + dt`, and leave the readings
unshifted. The 13:00 reading is the schedule's `charge_init`, the level before
the first step.

`docs/figures/time_alignment_battery.py` draws the figure. Rerun it after
changing the example.

### Converting each kind of data

The examples below use an hourly `dt = pd.Timedelta(hours=1)`.

#### Interval averages stamped at the interval's end

SCADA systems, meter exports, and hour-ending market data often stamp
averages this way. The 17:00 value belongs to the step stamped 16:00:

```python
series.index = series.index - dt
```

#### Interval averages stamped at the interval's start

Green Button and most utility interval data already match flex-pse, so use
them as they are.

#### Fast instantaneous samples of a rate

For power or flow sampled every minute or second, average every sample within
a step and label the result with the step's start:

```python
hourly = samples.resample("1h", label="left", closed="left").mean()
```

A single instantaneous reading taken at 17:00 is not a step average. Use it
for the step stamped 17:00 only as an approximation, and only if the rate
changes slowly.

#### Instantaneous readings of a state

For tank level or state of charge, the reading at the schedule's start is the
initial state, so set `initial_volume` or `initial_soc` from it. The reading
at the end of step `t` is `volume[t]`. To compare measured levels with a
solved schedule, read each `volume[t]` against the measurement at
`timestamp + dt`:

```python
measured_at_step_end = level.reindex(time_block.datetime_index + dt)
```

The same rules apply to the inputs and outputs of a surrogate fitted with
`flexparameterize`, and to an ARIMA surrogate's history. A surrogate fitted on
data stamped by interval end predicts each step one step late, so shift the
training data before fitting.
