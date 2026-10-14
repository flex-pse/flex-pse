"""Emit a parameterized model as a config — the serializable direction.

The terminal stage of the FlexParameterize pipeline that produces an artifact
rather than a mutation: a fit (or a hand-built relationship) plus a unit's
identity become a :class:`~flexcore.config.schema.ModelConfig` that
``flexops.build_model`` rebuilds the parameterized model from. Its twin,
:func:`~flexparameterize.apply.apply_to_model`, writes the same fit straight
into a live model; the two agree by construction, because both consume the same
``SurrogateSpec``.

Serialization is not reimplemented here — callers persist the result with
:func:`flexcore.config.io.dump_model_config`.
"""

import datetime
from importlib.metadata import version

import pyomo.environ as pyo
from pyomo.environ import units as pyunits

from flexcore.config.schema import (
    CURRENT_SCHEMA_VERSION,
    CostingConfig,
    IOVariableSpec,
    ModelConfig,
    PlantConfig,
    PriceSpec,
    SurrogateSpec,
    SurrogateType,
    TimeConfig,
    UnitConfig,
)
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.core.registration import IORegistry, iter_io_registry
from flexops.core.time_block import find_time_block
from flexops.surrogates import surrogate_from_spec
from flexparameterize.regression import (
    COEFFICIENT_NAME,
    constant_intensity_coefficient,
)

INTENSITY_UNITS = "kWh/m^3"
"""str: units the fitted constant intensity is written out in."""

VERSIONED_PACKAGES = ("flex-pse", "pyomo", "pandas")
"""tuple: distributions whose versions every emitted provenance records."""


def _surrogate_and_fit_provenance(
    fit_result, spec_options: dict | None = None
) -> tuple[SurrogateSpec, dict]:
    """Split ``fit_result`` into its spec and the provenance of its fit.

    Args:
        fit_result: A fitted regressor (anything exposing
            ``to_surrogate_spec()`` plus the fit attributes), or an already
            built :class:`~flexcore.config.schema.SurrogateSpec` — a vendor
            curve or datasheet coefficient that was never fitted.
        spec_options: Keyword arguments for ``to_surrogate_spec()``.

    Returns:
        ``(surrogate, provenance)``; the provenance is empty for a
        hand-built spec, which has no metrics or data window to record.
    """
    if isinstance(fit_result, SurrogateSpec):
        if spec_options:
            raise FlexConfigError(
                "spec_options only apply to a fitted regressor; a hand-built "
                "SurrogateSpec is emitted as given.",
                field="spec_options",
                value=sorted(spec_options),
            )
        return fit_result, {}
    result = fit_result.to_fit_result()
    window = [
        value.isoformat() if hasattr(value, "isoformat") else value
        for value in result.data_window
    ]
    return fit_result.to_surrogate_spec(**(spec_options or {})), {
        "n_samples": result.n_samples,
        **result.metrics,
        "data_window": window,
    }


def _unit_model_class_name(unit_or_class) -> str:
    """Return the flexops unit-model class name to write into the config.

    Args:
        unit_or_class: A built unit or a unit-model class.

    Returns:
        The name resolvable against ``flexops.unit_models`` (what
        ``OpsBlockData.build_from_config`` looks the class up by).

    Raises:
        FlexConfigError: If nothing in the type's MRO is a flexops unit model.
    """
    from flexops import unit_models

    declared = unit_or_class if isinstance(unit_or_class, type) else type(unit_or_class)
    for cls in declared.__mro__:
        name = cls.__name__.removesuffix("Data")
        if getattr(unit_models, name, None) is not None:
            return name
    raise FlexConfigError(
        f"{unit_or_class!r} is not a flexops unit model (or an instance of "
        "one); emit_model_config needs a class it can name in the config.",
        field="unit_model_class",
        value=unit_or_class,
    )


def _surrogate_variables(
    surrogate: SurrogateSpec,
) -> tuple[dict[str, str], dict[str, str]]:
    """Return a surrogate's declared ``(input, output)`` name -> units mappings.

    ``constant_intensity`` has no surrogate class (see
    :mod:`flexops.surrogates`), so its ``data`` is read directly; every other
    type is realized through :func:`~flexops.surrogates.surrogates.surrogate_from_spec`,
    which validates it in the same step.

    Args:
        surrogate: The relationship being emitted.

    Returns:
        ``(input_variables, output_variables)``.
    """
    if surrogate.surrogate_type is SurrogateType.CONSTANT_INTENSITY:
        return (
            dict(surrogate.data.get("input_variables", {})),
            dict(surrogate.data.get("output_variables", {})),
        )
    built = surrogate_from_spec(surrogate)
    return built.input_variables, built.output_variables


def _io_variable_specs(unit_or_class, surrogate: SurrogateSpec) -> list[IOVariableSpec]:
    """Describe the unit's process IO variables.

    A built unit supplies them from its registry (with real units); a bare
    class has none, so the surrogate's own declared variables and units stand
    in.

    Args:
        unit_or_class: A built unit or a unit-model class.
        surrogate: The relationship being emitted.

    Returns:
        The IO variable specs.
    """
    if isinstance(unit_or_class, OpsBlockData):
        _, registry = next(iter_io_registry(unit_or_class), (None, IORegistry()))
        return [
            IOVariableSpec(
                name=record.name,
                role=record.role,
                units=record.units,
                tag_hint=record.tag_hint,
                time_indexed=record.time_indexed,
            )
            for record in registry.io_variables
        ]
    inputs, outputs = _surrogate_variables(surrogate)
    return [
        IOVariableSpec(name=name, role=role, units=units)
        for role, mapping in (("input", inputs), ("output", outputs))
        for name, units in mapping.items()
    ]


def _time_config(unit_or_class) -> TimeConfig:
    """Derive the horizon config from a built unit's TimeBlock.

    Args:
        unit_or_class: A built unit.

    Returns:
        The :class:`~flexcore.config.schema.TimeConfig`.

    Raises:
        FlexConfigError: If ``unit_or_class`` is a class, which carries no
            horizon of its own.
    """
    if not isinstance(unit_or_class, OpsBlockData):
        raise FlexConfigError(
            "emit_model_config cannot derive the horizon from a unit-model "
            "class; pass time=TimeConfig(...) explicitly.",
            field="time",
            value=unit_or_class,
        )
    tb = find_time_block(unit_or_class.model())
    start = tb.datetime_index[0]
    step_seconds = pyo.value(pyunits.convert(tb.dt, pyunits.s))
    end = start + datetime.timedelta(seconds=step_seconds) * tb.n_points
    return TimeConfig(
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        time_step=f"{pyo.value(tb.dt):g} {pyunits.get_units(tb.dt)}",
    )


def emit_model_config(
    unit_or_class,
    fit_result,
    provenance: dict | None = None,
    *,
    time: TimeConfig | None = None,
    costing: CostingConfig | None = None,
    construction_options: dict | None = None,
    plant_name: str | None = None,
    unit_name: str | None = None,
    spec_options: dict | None = None,
) -> ModelConfig:
    """Assemble the config that rebuilds a parameterized unit.

    The emitted ``ModelConfig`` holds one plant of one unit: its class name,
    its construction options (a constant-intensity fit is written out as the
    unit's ``energy_intensity`` option, so a rebuild carries the fitted value),
    its IO variable specs, the ``SurrogateSpec``, and provenance.

    Provenance always records the versions of ``flex-pse``, ``pyomo`` and
    ``pandas``, read at emit time, under ``"versions"``. A **fitted** result
    adds ``"n_samples"``, the regressor's metrics (``"r2"``, ``"rmse"``) and
    ``"data_window"`` (``[first, last]`` ISO-8601 timestamps). A hand-built
    ``SurrogateSpec`` has none of those — it documents its origin through
    ``provenance`` instead (e.g. ``{"source": "vendor_datasheet"}``).

    Args:
        unit_or_class: The built unit that was fitted (its class name, IO
            registry and horizon are read off it), or a unit-model class, in
            which case ``time`` is required and the IO specs come from the
            fit's column names.
        fit_result: A fitted regressor, or a hand-built
            :class:`~flexcore.config.schema.SurrogateSpec` for a relationship
            already known in closed form — no data, no sufficiency check and no
            regressor are involved on that path.
        provenance: Extra provenance entries, merged last so they win.
        time: The horizon config; defaults to the built unit's own TimeBlock.
        costing: The costing config; defaults to a **placeholder** priced at
            0 USD/kWh, since a unit carries no tariff. Pass the real one for a
            config that is meant to be solved.
        construction_options: Extra construction options for the unit (its
            non-default, non-fitted settings); the fitted coefficient is merged
            in on top.
        plant_name: Name of the emitted plant; defaults to the built unit's
            parent block name, else ``"plant"``.
        unit_name: Key the unit is stored under; defaults to the built unit's
            own name, else ``"unit"``.
        spec_options: Keyword arguments forwarded to the regressor's
            ``to_surrogate_spec()``, e.g. ``{"history_length": 0}`` to emit
            an ARIMA fit with only its terminal state.

    Returns:
        The :class:`~flexcore.config.schema.ModelConfig`. Persist it with
        :func:`flexcore.config.io.dump_model_config`.

    Raises:
        FlexConfigError: If ``unit_or_class`` is not a flexops unit model, or
            is a class and no ``time`` was given, or ``spec_options`` is
            given with a hand-built ``SurrogateSpec``.
    """
    surrogate, fit_provenance = _surrogate_and_fit_provenance(fit_result, spec_options)
    surrogate = surrogate.model_copy(
        update={
            "provenance": {
                **fit_provenance,
                "versions": {name: version(name) for name in VERSIONED_PACKAGES},
                **(provenance or {}),
            }
        }
    )

    options = dict(construction_options or {})
    if surrogate.surrogate_type is SurrogateType.CONSTANT_INTENSITY:
        options[COEFFICIENT_NAME] = {
            "value": constant_intensity_coefficient(surrogate),
            "units": INTENSITY_UNITS,
        }

    built = isinstance(unit_or_class, OpsBlockData)
    return ModelConfig(
        schema_version=CURRENT_SCHEMA_VERSION,
        time=time or _time_config(unit_or_class),
        costing=costing
        or CostingConfig(
            energy_prices={"electrical": PriceSpec(value=0.0, units="USD/kWh")}
        ),
        plant=PlantConfig(
            name=plant_name
            or (unit_or_class.parent_block().local_name if built else "plant"),
            units={
                unit_name
                or (unit_or_class.local_name if built else "unit"): UnitConfig(
                    unit_model_class=_unit_model_class_name(unit_or_class),
                    construction_options=options,
                    io_variables=_io_variable_specs(unit_or_class, surrogate),
                    surrogate=surrogate,
                )
            },
        ),
    )
