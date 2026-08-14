"""Apply a fit to a live model — the mutate-in-place direction.

The FlexParameterize -> FlexOps half of the two-way coupling (architecture §1,
§5): given a **built** FlexOps model, tabular data and a
:class:`~flexparameterize.tags.TagMap`, :func:`apply_to_model` fits each unit's
registered parameters and writes the result straight into the model. No config
file is produced; its twin :func:`~flexparameterize.emit.emit_model_config`
covers that direction, and both consume the same ``SurrogateSpec``, so they
cannot disagree about what was fitted.

Two things happen per unit, and both are in-place — flex-pse never deletes a
built component (conventions §9):

* a ``constant_intensity`` relationship fixes the unit's registered intensity
  parameter at the fitted value, dropping the model's degrees of freedom;
* a richer relationship deactivates the unit's ``power_electrical_relation``
  (or any other relation the unit registered — see
  :meth:`~flexops.core.ops_block.OpsBlockData.register_relation`) and attaches
  an equality built from the ``SurrogateSpec``, on the same unit object,
  reusing the same registered IO variables — ports and arcs are untouched, so
  there is nothing to reconnect.

Unlike :func:`~flexparameterize.validate.check_sufficiency`, which only
reports, this function mutates a real model and so raises
:class:`~flexcore.exceptions.FlexDataError` up front when the data does not
determine the fit.
"""

from dataclasses import dataclass, field

import pandas as pd
from idaes.core.util.model_statistics import degrees_of_freedom

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexops.core.registration import iter_io_registry
from flexops.core.time_block import find_time_block
from flexparameterize.regression import (
    COEFFICIENT_NAME,
    ConstantIntensityRegressor,
    constant_intensity_coefficient,
)
from flexparameterize.tags import TagMap, model_alias
from flexparameterize.validate import DEFAULT_MIN_ROWS, check_sufficiency


@dataclass
class ApplyReport:
    """What :func:`apply_to_model` changed on the model.

    Attributes:
        fixed_parameters: Per unit name, the process parameters that were fixed
            and the values they were fixed at.
        swapped_relations: Per unit name, the relation names whose Constraint
            was swapped for a richer fitted one (the unit's own energy relation
            unless ``surrogates`` named a different one).
        dof_before: Degrees of freedom of the model before applying.
        dof_after: Degrees of freedom after; each fixed parameter removes one.
    """

    fixed_parameters: dict[str, dict[str, float]] = field(default_factory=dict)
    swapped_relations: dict[str, list[str]] = field(default_factory=dict)
    dof_before: int = 0
    dof_after: int = 0

    def __str__(self) -> str:
        """Render the fixed parameters, the swaps, and the DOF change."""
        touched = len(self.fixed_parameters) + len(self.swapped_relations)
        lines = [f"Applied to {touched} unit(s):"]
        for unit, values in self.fixed_parameters.items():
            fixed = ", ".join(f"{name}={value:g}" for name, value in values.items())
            lines.append(f"  fixed    {unit}: {fixed}")
        for unit, relations in self.swapped_relations.items():
            for relation in relations:
                lines.append(f"  swapped  {unit}: {relation} -> fitted")
        lines.append(f"  degrees of freedom: {self.dof_before} -> {self.dof_after}")
        return "\n".join(lines)


def _registered_alias(basis, registry) -> str:
    """Return the alias of the registered IO variable ``basis`` reads from.

    A unit's named flow (``flow_out``, ``permeate``) is a ``Reference`` into a
    state block, while the registered IO variable — and so the data column — is
    the state variable itself. The two share their underlying ``VarData``, so
    matching on identity maps the one onto the other.

    Args:
        basis: The unit's named flow component.
        registry: The unit's :class:`~flexops.core.registration.IORegistry`.

    Returns:
        The registered variable's alias, or the basis component's own alias
        when it is not a view onto a registered one.
    """
    members = {id(data) for data in basis.values()}
    for record in registry.io_variables:
        if any(id(data) in members for data in record.var.values()):
            return model_alias(record.var)
    return model_alias(basis)


def _basis_aliases(unit, registry) -> list[str]:
    """Return the alias(es) of the flow ``unit``'s intensity is metered on.

    The unit's own relation divides power by its **product** flow, so the fit
    has to regress against that same stream or the coefficient it recovers
    means something the model does not. Units built through
    ``add_constant_intensity_relation`` record the basis on their registry,
    which settles it exactly even for a unit with two outlets. A unit that
    hand-rolled its relation falls back to its registered output variables
    other than the power draw, which is unambiguous for a single-outlet unit.

    Args:
        unit: The built unit being parameterized.
        registry: Its :class:`~flexops.core.registration.IORegistry`.

    Returns:
        The candidate aliases — one when the basis is determined.
    """
    recorded = registry.intensity_basis.get(nm.PowerKind.ELECTRICAL)
    if recorded is not None:
        return [_registered_alias(unit.find_component(recorded), registry)]
    powers = {id(record.var) for record in registry.power}
    return [
        model_alias(record.var)
        for record in registry.io_variables
        if record.role == "output" and id(record.var) not in powers
    ]


def _fit_unit(unit, registry, data: pd.DataFrame) -> ConstantIntensityRegressor:
    """Fit ``unit``'s energy intensity from its own columns of ``data``.

    Args:
        unit: The built unit being parameterized.
        registry: Its :class:`~flexops.core.registration.IORegistry`.
        data: The aliased data frame.

    Returns:
        The fitted regressor.

    Raises:
        FlexDataError: If the unit does not present exactly one basis flow and
            one registered electrical power draw — the constant-intensity fit
            is a one-in, one-out problem.
    """
    basis = _basis_aliases(unit, registry)
    powers = [
        model_alias(record.var)
        for record in registry.power
        if record.kind is nm.PowerKind.ELECTRICAL
    ]
    if len(basis) != 1 or len(powers) != 1:
        raise FlexDataError(
            f"The constant-intensity fit needs exactly one basis flow and one "
            f"electrical power draw for {unit.name!r}; found flows {basis} and "
            f"power {powers}. Supply this unit's relationship through "
            "surrogates= instead.",
            field=unit.name,
        )
    return ConstantIntensityRegressor().fit(data[[basis[0]]], data[[powers[0]]])


POWER_ELECTRICAL_RELATION = f"{nm.POWER_ELECTRICAL}_relation"
"""str: name of the relation the fit-from-data path swaps -- the unit's own
energy relationship. A unit's other relations are only reachable by naming
them explicitly through ``surrogates={unit: {relation_name: spec}}``."""


def _attach_surrogate(unit, registry, surrogate) -> tuple[bool, dict[str, float]]:
    """Write ``surrogate`` into the live ``unit`` as its energy relationship.

    The single attachment path, taken by a fitted and a hand-supplied spec
    alike: the unit's default ``constant_intensity`` form is expressed by fixing
    its intensity parameter, and any richer form by swapping
    :data:`POWER_ELECTRICAL_RELATION`.

    Args:
        unit: The built unit to mutate.
        registry: Its :class:`~flexops.core.registration.IORegistry`.
        surrogate: The :class:`~flexcore.config.schema.SurrogateSpec` to attach.

    Returns:
        ``(swapped, fixed values)`` — whether the Constraint was swapped, and
        the parameters that were fixed (empty when it was).

    Raises:
        FlexConfigError: If a ``constant_intensity`` spec carries no
            coefficient under that name, or the unit does not register one as a
            regressable process parameter.
    """
    if surrogate.functional_form != "constant_intensity":
        unit.swap_relation(POWER_ELECTRICAL_RELATION, surrogate)
        return True, {}

    coefficient = constant_intensity_coefficient(surrogate)
    regressable = {
        record.name: record.param
        for record in registry.parameters
        if record.regressable
    }
    if COEFFICIENT_NAME not in regressable:
        raise FlexConfigError(
            f"A 'constant_intensity' relationship determines "
            f"{COEFFICIENT_NAME!r}, which {unit.name!r} does not register as a "
            f"regressable process parameter (it registers "
            f"{sorted(regressable)}). Supply this unit's relationship through "
            "surrogates= instead.",
            field=COEFFICIENT_NAME,
            value=unit.name,
        )

    unit.update_parameters({COEFFICIENT_NAME: coefficient})
    parameter = regressable[COEFFICIENT_NAME]
    if parameter.is_variable_type():
        parameter.fix()
    return False, {COEFFICIENT_NAME: coefficient}


def _require_sufficient_data(model, data, unit_names, min_rows: int) -> None:
    """Raise unless ``data`` determines the fit for every unit in ``unit_names``.

    Args:
        model: The built model being parameterized.
        data: The aliased data frame.
        unit_names: Names of the units that will be fitted; units whose
            relationship was supplied are not checked, having no data to check.
        min_rows: Minimum non-null rows a column must carry.

    Raises:
        FlexDataError: If any checked IO pair, or the data index, falls short.
    """
    report = check_sufficiency(model, data, find_time_block(model), min_rows=min_rows)
    short = [
        pair for pair in report.pairs if pair.unit in unit_names and not pair.sufficient
    ]
    if short or not report.index_ok:
        raise FlexDataError(
            "apply_to_model mutates a built model, so it will not proceed on "
            f"insufficient data.\n{report}",
            field=short[0].missing[0] if short and short[0].missing else None,
        )


def apply_to_model(
    model,
    data: pd.DataFrame,
    tagmap: TagMap,
    surrogates: dict | None = None,
    *,
    min_rows: int = DEFAULT_MIN_ROWS,
) -> ApplyReport:
    """Fit a built model's registered parameters from data and mutate it in place.

    Every unit that registers a regressable process parameter is fitted from
    ``data`` and updated. A unit named in ``surrogates`` is not fitted at all —
    its relationship is already known in closed form — so no data is required
    for it and no sufficiency check is run against it; the two paths mix freely
    across units in one call.

    Args:
        model: The built FlexOps model to mutate.
        data: Tabular plant data from any source, indexed by timestamp.
        tagmap: The :class:`~flexparameterize.tags.TagMap` renaming ``data``'s
            columns onto model aliases (pass ``TagMap({})`` if they already are).
        surrogates: Optional unit name (the unit's full model name, e.g.
            ``"facility.pump"``) -> either a hand-built
            :class:`~flexcore.config.schema.SurrogateSpec`, attached as the
            unit's own energy relationship, or a ``{relation_name: spec}``
            mapping naming one or more of the unit's *other* registered
            relations (see
            :meth:`~flexops.core.ops_block.OpsBlockData.register_relation`) —
            an RO skid's ``split_definition``, a tank's ``level_definition``.
        min_rows: Minimum non-null rows a data column must carry.

    Returns:
        The :class:`ApplyReport` describing what changed.

    Raises:
        FlexDataError: If the data does not determine the fit for a unit that
            has to be fitted; the model is left untouched.
        FlexConfigError: If a relationship cannot be attached to its unit.
    """
    supplied = dict(surrogates or {})
    aliased = tagmap.apply(data)
    units = [
        (unit, registry)
        for unit, registry in iter_io_registry(model)
        if unit.name in supplied
        or any(record.regressable for record in registry.parameters)
    ]

    to_fit = {unit.name for unit, _ in units if unit.name not in supplied}
    if to_fit:
        _require_sufficient_data(model, aliased, to_fit, min_rows)

    report = ApplyReport(dof_before=degrees_of_freedom(model))
    for unit, registry in units:
        supplied_for_unit = supplied.get(unit.name)
        if isinstance(supplied_for_unit, dict):
            for relation_name, spec in supplied_for_unit.items():
                unit.swap_relation(relation_name, spec)
                report.swapped_relations.setdefault(unit.name, []).append(relation_name)
            continue
        surrogate = supplied_for_unit
        if surrogate is None:
            surrogate = _fit_unit(unit, registry, aliased).to_surrogate_spec()
        swapped, fixed = _attach_surrogate(unit, registry, surrogate)
        if swapped:
            report.swapped_relations.setdefault(unit.name, []).append(
                POWER_ELECTRICAL_RELATION
            )
        if fixed:
            report.fixed_parameters[unit.name] = fixed
    report.dof_after = degrees_of_freedom(model)
    return report
