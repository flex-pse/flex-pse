# How design studies annualize cost

A design study minimizes one number, the annualized cost: a constant yearly payment equivalent to the study's operating and capital costs.

## The formulas

With effective annual rate {math}`r`, and study years counted from 1 at the end of each year:

```{math}
\mathrm{CRF}(r, L) = \frac{r(1+r)^L}{(1+r)^L - 1}, \qquad \mathrm{CRF}(0, L) = \frac{1}{L}
```

```{math}
\mathrm{PW}_y = \sum_{k \in K_y} (1+r)^{-k}, \qquad \mathrm{AF} = \sum_y \mathrm{PW}_y
```

where {math}`K_y` holds the calendar years that modeled year {math}`y` represents. Then:

```{math}
\text{annual opex}_y = \sum_m w_m \, s_{y,m} \, \text{bill}_{y,m}
```

```{math}
\text{annualized opex} = \frac{1}{\mathrm{AF}} \sum_y \mathrm{PW}_y \, \text{annual opex}_y
```

```{math}
\text{annualized capital} = \sum_d \mathrm{CRF}(r, L_d) \, c_d \, \text{size}_d
```

Here {math}`w_m` is the month's weight, {math}`c_d` is overnight cost per unit, and {math}`L_d` is the decision's lifetime. {math}`s_{y,m}` is 1 for a whole month, or the days in that month divided by `days` for a prorated one. In a one-year study {math}`\mathrm{AF}` cancels, leaving that year's operating cost plus capital times CRF.

## Lifetime and study length

Capital is charged as a constant annuity over each year of the study, which has a useful consequence when a lifetime differs from the study length:

- **An asset outliving the study** pays only for the years used. A 25-year solar array in a 10-year study is charged for 10 years of its annuity, which implicitly credits its remaining value.
- **A shorter-lived asset** keeps being charged, which implicitly renews it at today's cost. There's no replacement schedule, so a 15-year battery in a 20-year study has no separate replacement event or cost.

Capacity is built once, at the start of the study. Later years change operating conditions only.

## The rate

The rate comes from the costing element, and every horizon must share it. With only `discount_rate` set, that is the rate. With `interest_rate` also set, the rate is {math}`(1 + \text{interest}) / (1 + \text{discount}) - 1`, a real rate, so prices should be in constant dollars. The default `discount_rate` still applies if you set only `interest_rate`, which can produce a low or negative rate, so set both deliberately.

## Representative months and bills

Weights apply to whole bills, which is valid because a whole calendar month bills a real monthly demand charge. A month shortened with `days` is scaled up to the full month. Its monthly charges follow `prorate_monthly_charges`, and its peak only reflects the days modeled.

## Reading the report

`design_results(m).summary()` lists each capacity with its size, overnight cost, lifetime and annualized cost, then operating cost by modeled year, then the annualized totals.

Don't read `total_cost` or `annualized_cost` from a horizon's own costing block. Those carry that block's empty capital placeholder, and the costing element's `lifetime_years` isn't used by a study.

## NPV and IRR

`investment_metrics` reports NPV and IRR for a solved design. It only reads the results and doesn't change the model or what it optimizes. It builds explicit yearly cash flows, using the same timing as the annualization above:

- **Year 0** pays the total overnight capital.
- **Years 1 to H** earn the savings, baseline operating cost minus the design's, for the modeled year that stands in for that calendar year. Years past the study repeat the last modeled year.
- **Replacement.** An asset with lifetime {math}`L` is bought again at its overnight cost in year {math}`\lceil nL \rceil` for each {math}`n \ge 1` with {math}`nL < H`.
- **Salvage.** In year {math}`H`, each asset returns the unused share of its last purchase, straight-line: {math}`c_d \, \mathrm{size}_d \, (\lceil H/L \rceil L - H) / L`.

NPV discounts these flows at the study's effective rate. IRR is the rate that makes NPV zero, computed with `pyxirr`. It is `None` when the cash flows have no sign change. Replacement outflows can give more than one IRR, and only one is reported.

Unlike the annuity treatment, these flows have explicit replacement and salvage events, so NPV divided by the annuity factor won't exactly match the annualized cost difference.
