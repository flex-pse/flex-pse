# How demand-response revenue enters the cost

An incentive-based demand-response (DR) program pays a site for using less
power than it otherwise would have during an event the utility calls. flex-pse
models that payment with EECO's demand-response module and subtracts it from the
operating cost. When the plant decides when to run, it then weighs the energy
and demand charges it saves against the DR payment it can earn.

## Two questions every program answers

Every DR program answers two independent questions, and the spec configures them
separately.

1. **What would the site have drawn?** This is the *baseline*, the
   counterfactual load. The payment is based on the *reduction*, which is
   baseline minus actual power, averaged over each settlement interval of the
   event.
   - `similar_days` averages the event window over the most recent eligible
     days. It skips weekends and holidays by default, and it can scale the
     result by a day-of adjustment computed from the hours just before the
     event. Its defaults follow PG&E's Capacity Bidding Program.
   - `top_usage_days` is the same average, taken over the days with the highest
     use instead of the most recent ones.
   - `fixed_level` is a contracted firm service level in kW.
   - `unilateral_interruption` covers programs where the utility, not the site,
     cuts load. It caps site power at the interruption level for the whole
     event, as a hard constraint. Pair it with a payout-only payment.
2. **How is the reduction paid?** This is the *payment*.
   - `capacity` maps the *delivered ratio* (reduction divided by bid capacity)
     through a piecewise-linear schedule of regions to a fraction of
     `capacity_price × bid_capacity_kW`.
   - `capacity_energy` adds a flat payment per kWh curtailed.

   Either kind can also pay a flat participation `payout` per kW of bid.
   `payment_basis` sets whether the capacity price is paid per event, per event
   hour, or as a share of a monthly payment.

The baseline needs meter history from before the horizon. The spec carries it in
`historical_power` as a start time, a time step and a list of kW values. That
list is usually long, so it is normally kept in a `$source` file beside the
spec. Baseline days that fall inside the optimization horizon are taken from the
model's own power instead, so the plan can't inflate its baseline.

## Where the revenue appears

`FlexCosting` builds the revenue on `opex.dr` from the site's net electrical
power and exposes it as `opex.dr_revenue`, in the costing block's currency.
`opex.total_operating_cost` is electricity + fuel + fixed + scalar cost minus
`dr_revenue`. The annualized and design-mode totals pick it up from there.

Like the tariff bill, the revenue in a report is recomputed after the solve.
`report_cost` settles every event from scratch with EECO on the realized power
plus the meter history. It never reads `opex.dr_revenue`.

## Payment regions make the model a MILP

The payment schedule usually isn't concave. For example, a program may pay
nothing below 60 % delivery, jump at 60 %, and jump again at 75 %. To capture
those jumps exactly, EECO picks the active region with one binary variable per
region, for every settlement interval of every event. Any program with a regions
schedule therefore turns the operating model into a MILP.

When you already know which region an event will land in, for instance full
delivery, set that event's `region_x1` to the region's lower bound. EECO then
fixes the region's binaries and the model stays an LP. The trade-off is that a
fixed region is a constraint: the plan must deliver within that region, and it
becomes infeasible if it can't.

## A minimal program

```json
"dr": {
  "events": [{"event_date": "2025-07-08", "start_hour": 16, "duration_hours": 3,
              "baseline_days": ["2025-07-07"], "bid_capacity_kW": 100,
              "capacity_price": 10}],
  "historical_power": {"start": "2025-06-16T00:00", "time_step_minutes": 60,
                       "values_kW": {"$source": "data/meter.json.gz"}},
  "baseline": {"method": "fixed_level", "firm_level_kW": 150},
  "payment": {"kind": "capacity",
              "regions": [{"x1": 0, "x2": 1, "y1": 0, "y2": 1},
                          {"x1": 1, "x2": null, "y1": 1, "y2": 1}]}
}
```

A region bound of `null` is unbounded, since JSON has no infinity. Above, a
region with an unbounded `x2` pays a flat `y1`, so delivering more than the bid
earns nothing extra.
