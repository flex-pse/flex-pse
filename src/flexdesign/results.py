"""Read capacities and annualized costs off a solved design model."""

from dataclasses import dataclass

import pyomo.environ as pyo
from pyomo.environ import units as pyunits


@dataclass(frozen=True)
class CapacityResult:
    """One sized capacity and its costs."""

    size: float
    units: str
    at_upper_bound: bool
    capital_cost: float
    lifetime_years: float
    annualized_capital_cost: float


@dataclass(frozen=True)
class DesignResults:
    """Solved capacities and annualized costs of a design model."""

    study_years: int
    effective_rate: float
    currency: str
    capacities: dict[str, CapacityResult]
    annual_operating_cost_by_year: dict[int, float]
    operating_cost_by_horizon: dict[tuple[int, int], float]
    annualized_operating_cost: float
    annualized_capital_cost: float
    annualized_cost: float
    total_capital_cost: float

    def summary(self) -> str:
        """Return a plain-text table of the capacities and annualized costs.

        Returns:
            The table, with costs in the study's currency.
        """
        lines = [
            f"{self.study_years}-year study at a {self.effective_rate:.2%} rate, "
            f"costs in {self.currency}",
            "",
            f"{'Capacity':<20}{'Size':>12} {'Units':<8}{'Overnight':>14}"
            f"{'Life (yr)':>11}{'Annualized':>14}",
        ]
        for name, capacity in self.capacities.items():
            flag = "  (at upper bound)" if capacity.at_upper_bound else ""
            lines.append(
                f"{name:<20}{capacity.size:>12,.2f} {capacity.units:<8}"
                f"{capacity.capital_cost:>14,.2f}{capacity.lifetime_years:>11g}"
                f"{capacity.annualized_capital_cost:>14,.2f}{flag}"
            )
        lines += ["", "Annual operating cost by modeled year"]
        for year, cost in self.annual_operating_cost_by_year.items():
            lines.append(f"  {year}: {cost:,.2f}")
        lines += [
            "",
            f"Total overnight capital:    {self.total_capital_cost:,.2f}",
            f"Annualized operating cost:  {self.annualized_operating_cost:,.2f}",
            f"Annualized capital cost:    {self.annualized_capital_cost:,.2f}",
            f"Annualized cost:            {self.annualized_cost:,.2f}",
        ]
        return "\n".join(lines)


def design_results(m: pyo.ConcreteModel) -> DesignResults:
    """Return the capacities and annualized costs of a solved design model.

    Args:
        m: A solved model from :func:`~flexdesign.build.build_design_model`.

    Returns:
        The results, with costs in the base currency.
    """
    spec = m._design_spec
    costing_name = spec.base.of_kind("costing")[0].name
    capacities = {}
    for decision in spec.capacities:
        block = m.capacity[decision.name]
        size = pyo.value(block.size)
        capacities[decision.name] = CapacityResult(
            size=size,
            units=str(pyunits.get_units(block.size)),
            at_upper_bound=size >= decision.upper - 1e-6,
            capital_cost=pyo.value(block.capital_cost),
            lifetime_years=decision.lifetime_years,
            annualized_capital_cost=pyo.value(block.annualized_capital_cost),
        )
    costing = {key: m.horizon[key].find_component(costing_name) for key in m.horizon}
    first = next(iter(costing.values()))
    return DesignResults(
        study_years=sum(year.represents for year in spec.years),
        effective_rate=pyo.value(first.effective_rate),
        currency=str(pyunits.get_units(first.aggregate_operating_cost)),
        capacities=capacities,
        annual_operating_cost_by_year={
            year: pyo.value(m.annual_operating_cost[year])
            for year in m.annual_operating_cost
        },
        operating_cost_by_horizon={
            key: pyo.value(c.aggregate_operating_cost) for key, c in costing.items()
        },
        annualized_operating_cost=pyo.value(m.annualized_operating_cost),
        annualized_capital_cost=pyo.value(m.annualized_capital_cost),
        annualized_cost=pyo.value(m.annualized_cost),
        total_capital_cost=sum(c.capital_cost for c in capacities.values()),
    )
