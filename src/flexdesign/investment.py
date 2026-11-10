"""NPV and IRR of a solved design's savings against a baseline."""

import math
from dataclasses import dataclass

import numpy_financial as npf

from flexdesign.results import DesignResults


@dataclass(frozen=True)
class InvestmentMetrics:
    """Cash flows, NPV and IRR of a design against a baseline."""

    horizon_years: int
    rate: float
    currency: str
    cash_flows: list[float]
    npv: float
    irr: float | None

    def summary(self) -> str:
        """Return a plain-text table of the yearly cash flows, NPV and IRR.

        Returns:
            The table, with amounts in the study's currency.
        """
        irr = "none" if self.irr is None else f"{self.irr:.2%}"
        lines = [
            f"{self.horizon_years}-year horizon at a {self.rate:.2%} rate, "
            f"amounts in {self.currency}",
            "",
            f"{'Year':>6}{'Cash flow':>16}",
        ]
        lines += [f"{k:>6}{flow:>16,.2f}" for k, flow in enumerate(self.cash_flows)]
        lines += ["", f"NPV:  {self.npv:,.2f}", f"IRR:  {irr}"]
        return "\n".join(lines)


def investment_metrics(
    results: DesignResults,
    baseline_operating_cost: dict[int, float],
    horizon_years: int | None = None,
) -> InvestmentMetrics:
    """Return the NPV and IRR of a design's operating savings net of capital spending.

    Args:
        results: Results from :func:`~flexdesign.results.design_results`.
        baseline_operating_cost: Annual operating cost without the investment,
            keyed by the same modeled years as the results.
        horizon_years: Years to evaluate; None uses the study length. Years past
            the study repeat the last modeled year.

    Returns:
        The metrics, with cash flows indexed by year and year 0 the initial build.

    Raises:
        ValueError: If the baseline years differ from the modeled years, or
            ``horizon_years`` is below 1.

    Warning:
        Replacement outflows can give more than one IRR; only one is reported.
    """
    design_cost = results.annual_operating_cost_by_year
    if baseline_operating_cost.keys() != design_cost.keys():
        raise ValueError(
            f"Baseline years {sorted(baseline_operating_cost)} must match the "
            f"modeled years {sorted(design_cost)}."
        )
    horizon = results.study_years if horizon_years is None else horizon_years
    if horizon < 1:
        raise ValueError(f"horizon_years must be at least 1, got {horizon}.")

    first_year = min(design_cost)
    cash_flows = [-results.total_capital_cost]
    for k in range(1, horizon + 1):
        year = max(y for y in design_cost if y <= first_year + k - 1)
        cash_flows.append(baseline_operating_cost[year] - design_cost[year])
    for capacity in results.capacities.values():
        life = capacity.lifetime_years
        installs = math.ceil(horizon / life)
        for n in range(1, installs):
            cash_flows[math.ceil(n * life)] -= capacity.capital_cost
        remaining = installs * life - horizon
        cash_flows[horizon] += capacity.capital_cost * remaining / life

    irr = float(npf.irr(cash_flows))

    return InvestmentMetrics(
        horizon_years=horizon,
        rate=results.effective_rate,
        currency=results.currency,
        cash_flows=cash_flows,
        npv=float(npf.npv(results.effective_rate, cash_flows)),
        irr=None if math.isnan(irr) else irr,
    )
