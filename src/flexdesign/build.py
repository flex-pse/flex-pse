"""Build one merged design model from a base flowsheet and a study."""

import calendar
import itertools
from datetime import date, timedelta

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

from flexcore.config.design import (
    CapacitySpec,
    DesignSpec,
    PlanningYear,
    RepresentativeMonth,
)
from flexcore.config.spec import FlowsheetSpec
from flexcore.exceptions import FlexConfigError
from flexops import build_model
from flexops.costing import capital_recovery_factor


def _deep_merge(target: dict, patch: dict) -> None:
    """Merge ``patch`` into ``target`` in place, recursing into nested dicts.

    Args:
        target: The dictionary to change.
        patch: The values to merge in; a non-dict value replaces the target's.
    """
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value


def expand_horizons(spec: DesignSpec) -> dict[tuple[int, int], FlowsheetSpec]:
    """Return one validated flowsheet spec per year and month.

    Args:
        spec: The design study.

    Returns:
        Specs keyed by ``(year, month)``, each carrying the base's directory.
    """
    expanded = {}
    for year, month in itertools.product(spec.years, spec.months):
        data = spec.base.model_dump(by_alias=True)
        named = {el["name"]: el for el in data["elements"] if el.get("name")}
        for change in spec.changes:
            if change.years is not None and year.year not in change.years:
                continue
            if change.months is not None and month.month not in change.months:
                continue
            _deep_merge(named[change.element], change.values)
        start = date(year.year, month.month, 1)
        end = (
            (start + timedelta(days=32)).replace(day=1)
            if month.days is None
            else start + timedelta(days=month.days)
        )
        time = next(el for el in data["elements"] if el["kind"] == "time")
        time.update(start_date=start.isoformat(), end_date=end.isoformat())
        horizon = FlowsheetSpec.model_validate(data)
        horizon._base_dir = spec.base._base_dir
        expanded[year.year, month.month] = horizon
    return expanded


def build_horizons(spec: DesignSpec) -> dict[tuple[int, int], pyo.ConcreteModel]:
    """Build one standalone operations model per year and month.

    Args:
        spec: The design study.

    Returns:
        Built models keyed by ``(year, month)``.

    Raises:
        FlexConfigError: If a model holds a fixed initial state.
    """
    models = {key: build_model(s) for key, s in expand_horizons(spec).items()}
    for key, model in models.items():
        states = model.time_block.initial_state_params
        if states:
            raise FlexConfigError(
                f"Horizon {key} has fixed initial states "
                f"{[p.name for p in states]}; set fix_initial_state=False on storage.",
                field="base",
            )
    return models


def present_worth_factors(years: list[PlanningYear], rate: float) -> dict[int, float]:
    """Return each modeled year's sum of discount factors over the years it represents.

    Args:
        years: The modeled years, in order.
        rate: Effective annual rate, as a fraction.

    Returns:
        The factors keyed by calendar year.
    """
    factors, elapsed = {}, 0
    for year in years:
        factors[year.year] = sum(
            (1 + rate) ** -k for k in range(elapsed + 1, elapsed + year.represents + 1)
        )
        elapsed += year.represents
    return factors


def _month_scale(month: RepresentativeMonth, year: int) -> float:
    """Return the factor taking a horizon's bill to its share of the year.

    Args:
        month: The representative month.
        year: The calendar year modeled.

    Returns:
        The month's weight, scaled up from ``days`` to the full month if prorated.
    """
    if month.days is None:
        return month.weight
    return month.weight * calendar.monthrange(year, month.month)[1] / month.days


def _capacity_var(block: pyo.Block, decision: CapacitySpec) -> pyo.Var:
    """Return the scalar Var a decision sizes on one horizon.

    Args:
        block: The horizon block.
        decision: The capacity decision.

    Returns:
        The Var.

    Raises:
        FlexConfigError: If the path doesn't name a scalar Var, or the decision's
            bounds exceed the Var's own.
    """
    var = block.find_component(decision.variable)
    if var is None:
        problem = "was not found"
    elif var.ctype is not pyo.Var:
        problem = "is not a Var"
    elif var.is_indexed():
        problem = "must be a scalar Var"
    elif (var.ub is not None and decision.upper > var.ub) or (
        var.lb is not None and decision.lower < var.lb
    ):
        problem = f"has bounds ({var.lb}, {var.ub}) narrower than the decision's"
    else:
        return var
    raise FlexConfigError(
        f"Capacity {decision.name!r}: {decision.variable!r} {problem}.",
        field="variable",
        value=decision.variable,
    )


def _link_rule(variables: dict[tuple[int, int], pyo.Var]):
    """Return a Constraint rule equating each horizon's Var with the shared size.

    Args:
        variables: The sized Var of each horizon, keyed by ``(year, month)``.

    Returns:
        The rule.
    """

    def rule(block, year, month):
        return variables[year, month] == block.size

    return rule


def build_design_model(spec: DesignSpec) -> pyo.ConcreteModel:
    """Merge every horizon into one model with shared sizes and an annualized objective.

    Args:
        spec: The design study.

    Returns:
        A model with ``horizon[year, month]``, ``capacity[name]``,
        ``annual_operating_cost[year]``, ``annualized_operating_cost``,
        ``annualized_capital_cost`` and the objective ``annualized_cost``.

    Raises:
        FlexConfigError: If a capacity path is not a scalar Var, its bounds exceed the
            Var's own, or horizons disagree on the effective rate.
    """
    models = build_horizons(spec)
    years = [y.year for y in spec.years]
    months = [month.month for month in spec.months]
    m = pyo.ConcreteModel(name=spec.name)
    m.horizon = pyo.Block(years, months)
    for key, model in models.items():
        m.horizon[key].transfer_attributes_from(model)
        for objective in m.horizon[key].component_data_objects(
            pyo.Objective, active=True
        ):
            objective.deactivate()

    costing_name = spec.base.of_kind("costing")[0].name
    costing = {key: m.horizon[key].find_component(costing_name) for key in models}
    rates = {pyo.value(c.effective_rate) for c in costing.values()}
    if len(rates) != 1:
        raise FlexConfigError(
            "All horizons must share one effective rate.", field="changes"
        )
    rate = rates.pop()
    currency = pyunits.get_units(next(iter(costing.values())).aggregate_operating_cost)

    m.capacity = pyo.Block([c.name for c in spec.capacities])
    for decision in spec.capacities:
        block = m.capacity[decision.name]
        variables = {key: _capacity_var(m.horizon[key], decision) for key in models}
        var_units = pyunits.get_units(next(iter(variables.values())))
        block.size = pyo.Var(
            bounds=(decision.lower, decision.upper),
            initialize=decision.upper,
            units=var_units,
        )
        for var in variables.values():
            var.unfix()
        block.link = pyo.Constraint(m.horizon.index_set(), rule=_link_rule(variables))
        block.capital_cost = pyo.Expression(
            expr=decision.capital_cost * currency / var_units * block.size
        )
        block.annualized_capital_cost = pyo.Expression(
            expr=capital_recovery_factor(rate, decision.lifetime_years)
            * block.capital_cost
        )

    m.annual_operating_cost = pyo.Expression(
        years,
        rule=lambda _, y: sum(
            _month_scale(month, y) * costing[y, month.month].aggregate_operating_cost
            for month in spec.months
        ),
    )
    factors = present_worth_factors(spec.years, rate)
    m.annualized_operating_cost = pyo.Expression(
        expr=sum(factors[y] * m.annual_operating_cost[y] for y in years)
        / sum(factors.values())
    )
    m.annualized_capital_cost = pyo.Expression(
        expr=sum(m.capacity[c.name].annualized_capital_cost for c in spec.capacities)
    )
    m.annualized_cost = pyo.Objective(
        expr=m.annualized_operating_cost + m.annualized_capital_cost
    )
    m._design_spec = spec
    return m
