"""Optional piece: a priced degradation (wear) penalty on a unit's variables.

Each term charges a linear wear driver of one variable beyond a free deadband,
summed into a cost-rate series. See ``docs/explanation/degradation.md`` for the
formulation.
"""

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

from flexcore.config.schema import DegradationSpec, DegradationTerm, DegradationTermSpec
from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.logic.status import RollingStateKind, _register_rolling_state

_log = get_logger(__name__)


def add_degradation(
    unit, spec: DegradationSpec, *, costing=None
) -> tuple[pyo.Var, pyo.Var]:
    """Attach the degradation penalty ``spec`` to ``unit``, billed through ``costing``.

    Args:
        unit: The unit block whose variables the terms name.
        spec: The validated penalty (terms, covered cost and budget).
        costing: A FlexCosting block to bill the charged wear cost to as the
            scalar cost ``f"{unit.local_name}_{spec.name}"``, or ``None``.

    Returns:
        ``(rate, total)``: the ``<name>_rate[t]`` wear-cost rate Var (1/hr, in
        the model currency) and the ``<name>_total`` horizon wear-cost Var.

    Raises:
        FlexConfigError: If ``unit`` already has a penalty named ``spec.name``,
            or a term names a variable that is not on ``unit``.

    Warning:
        The hinge Vars equal the wear only when a price or budget pushes them
        down; without ``costing`` or ``horizon_budget`` their values are loose.
    """
    name = spec.name
    if unit.find_component(f"{name}_rate") is not None:
        raise FlexConfigError(
            f"{unit.name!r} already has a degradation penalty named {name!r}.",
            field="name",
            value=name,
        )
    tb = unit._find_time_block()
    time = tb.time_index
    per_hour = 1 / pyunits.hr
    terms = [
        _add_term(unit, f"{name}_{k}", term, tb) for k, term in enumerate(spec.terms)
    ]

    unit.add_component(
        f"{name}_rate",
        pyo.Var(time, initialize=0.0, units=per_hour, doc="Wear cost rate."),
    )
    rate = unit.find_component(f"{name}_rate")

    def _rate_rule(_b, t):
        charged = sum(
            pyunits.convert(
                price * excess[t] / (tb.dt if per_step else 1), to_units=per_hour
            )
            for price, excess, per_step in terms
            if t in excess
        )
        return rate[t] == charged

    unit.add_component(
        f"{name}_rate_relation",
        pyo.Constraint(time, rule=_rate_rule, doc=f"Defines {name}_rate[t]."),
    )
    unit.register_relation(unit.find_component(f"{name}_rate_relation"), rate)

    unit.add_component(
        f"{name}_total",
        pyo.Var(initialize=0.0, units=pyunits.dimensionless, doc="Wear cost."),
    )
    total = unit.find_component(f"{name}_total")
    unit.add_component(
        f"{name}_total_relation",
        pyo.Constraint(
            expr=total
            == sum(
                pyunits.convert(rate[t] * tb.dt, to_units=pyunits.dimensionless)
                for t in time
            )
        ),
    )

    horizon_hours = len(time) * pyo.value(pyunits.convert(tb.dt, pyunits.hr))
    scale = 1.0 if spec.period_hours is None else horizon_hours / spec.period_hours
    covered = _add_param(unit, f"{name}_covered_cost", spec.covered_cost)
    unit.add_component(
        f"{name}_billable",
        pyo.Var(
            domain=pyo.NonNegativeReals,
            initialize=0.0,
            units=pyunits.dimensionless,
            doc="Wear cost beyond the covered cost.",
        ),
    )
    billable = unit.find_component(f"{name}_billable")
    unit.add_component(
        f"{name}_billable_floor",
        pyo.Constraint(expr=billable >= total - scale * covered),
    )
    if spec.horizon_budget is not None:
        budget = _add_param(unit, f"{name}_horizon_budget", spec.horizon_budget)
        unit.add_component(
            f"{name}_budget", pyo.Constraint(expr=total <= scale * budget)
        )

    unit.add_component(
        f"{name}_billed_rate",
        pyo.Expression(
            time,
            rule=lambda _b, t: billable / (horizon_hours * pyunits.hr),
            doc="Billable wear cost spread evenly over the horizon.",
        ),
    )
    if costing is not None:
        costing.register_scalar_cost(
            f"{unit.local_name}_{name}",
            unit.find_component(f"{name}_billed_rate"),
            price=1.0,
            quantity_units=per_hour,
            unit=unit,
        )
    elif spec.horizon_budget is None:
        _log.warning(
            "add_degradation: %s.%s is neither priced nor budgeted, so its wear "
            "values are not driven tight.",
            unit.name,
            name,
        )
    return rate, total


def _add_term(unit, prefix: str, term: DegradationTermSpec, tb):
    """Build one term's tracked quantity, parameters, and hinge constraints.

    Args:
        unit: The unit block the term lives on.
        prefix: Component-name stem for the term.
        term: The validated term.
        tb: The unit's TimeBlock.

    Returns:
        ``(price, excess, per_step)``: the price Param, the hinge Var, and
        whether the excess is a per-step amount (variation) or a level.
    """
    var = unit.resolve_variable(term.variable, field="variable")
    var_units = pyunits.get_units(var)
    if var_units is None:
        var_units = pyunits.dimensionless
    time = tb.time_index
    per_step = term.kind is DegradationTerm.VARIATION

    unit.add_component(
        f"{prefix}_quantity",
        pyo.Var(time, units=var_units, doc=f"Tracked {term.variable} for wear."),
    )
    q = unit.find_component(f"{prefix}_quantity")
    unit.add_component(
        f"{prefix}_quantity_relation",
        pyo.Constraint(time, rule=lambda _b, t: q[t] == var[t]),
    )
    unit.register_relation(unit.find_component(f"{prefix}_quantity_relation"), q)

    price_units = 1 / var_units if per_step else 1 / (var_units * pyunits.hr)
    price = _add_param(unit, f"{prefix}_price", term.price, price_units)
    if term.kind is not DegradationTerm.EXCEEDANCE:
        a = _add_param(unit, f"{prefix}_deadband", term.deadband, var_units)
    w = term.window
    index = [t for t in time if t >= w] if per_step else list(time)
    unit.add_component(
        f"{prefix}_excess",
        pyo.Var(index, domain=pyo.NonNegativeReals, initialize=0.0, units=var_units),
    )
    e = unit.find_component(f"{prefix}_excess")

    if term.kind is DegradationTerm.VARIATION:
        bodies = {
            "up": lambda t: q[t] - q[t - w] - a,
            "down": lambda t: q[t - w] - q[t] - a,
        }
        _register_rolling_state(unit, q, w, RollingStateKind.DEGRADATION)
    elif term.kind is DegradationTerm.DEVIATION:
        ref = _add_param(unit, f"{prefix}_reference", term.reference, var_units)
        bodies = {"up": lambda t: q[t] - ref - a, "down": lambda t: ref - q[t] - a}
    elif term.kind is DegradationTerm.EXCEEDANCE:
        bodies = {}
        if term.upper is not None:
            upper = _add_param(unit, f"{prefix}_upper", term.upper, var_units)
            bodies["up"] = lambda t: q[t] - upper
        if term.lower is not None:
            lower = _add_param(unit, f"{prefix}_lower", term.lower, var_units)
            bodies["down"] = lambda t: lower - q[t]
    else:
        bodies = {"up": lambda t: q[t] - a}

    for side, body in bodies.items():
        unit.add_component(
            f"{prefix}_{side}",
            pyo.Constraint(index, rule=lambda _b, t, f=body: e[t] >= f(t)),
        )
    return price, e, per_step


def _add_param(unit, name: str, value: float, units=pyunits.dimensionless):
    """Add a mutable, registered, non-regressable Param to ``unit``.

    Args:
        unit: The unit block.
        name: The Param's component name.
        value: Its initial value, in ``units``.
        units: Its Pyomo units.

    Returns:
        The Param.
    """
    unit.add_component(name, pyo.Param(initialize=value, mutable=True, units=units))
    param = unit.find_component(name)
    unit.register_process_parameter(param, regressable=False)
    return param
