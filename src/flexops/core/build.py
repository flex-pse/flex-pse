"""build_model: construct a whole flex-pse model from one config.

The config-driven entry point. A single validated
:class:`~flexcore.config.schema.ModelConfig` yields the TimeBlock, the property
package, the ``FlexCosting`` block, the network/plant/unit tree, its arcs, any
external dispatch, and the objective — so nothing essential has to live in
imperative code. The frozen-API fixtures ``api_freeze.py`` and
``api_freeze_config.json`` (under ``flexops/tests/fixtures/api_freeze/``) are
the same model built each way, and a component test holds them to the same
solved objective.

**Units in a persisted config are data, not code.** A config cannot carry a
Pyomo expression, so a units-carrying quantity is written as
``{"value": 15, "units": "min"}`` (or, for the time step, the string
``"15 min"``) and :func:`parse_quantity` turns it into one at build time.
Anything else in a construction-options mapping passes through unchanged.
Arcs are declared but **not** expanded: applying
``TransformationFactory("network.expand_arcs")`` stays the caller's explicit
step, as in the frozen script.
"""

import json
from pathlib import Path

import pyomo.environ as pyo
from pyomo.network import Arc

from flexcore.config.io import load_model_config
from flexcore.config.schema import ModelConfig, PlantConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.network_block import NetworkBlock
from flexops.core.ops_block import OpsBlockData
from flexops.core.plant_block import PlantBlock
from flexops.core.time_block import TimeBlock
from flexops.core.units import parse_units
from flexops.costing import FlexCosting
from flexops.properties.simple_aqueous import SimpleAqueousFlow

_QUANTITY_KEYS = {"value", "units"}


def parse_quantity(value, *, strict: bool = True):
    """Turn a persisted units-carrying value into a Pyomo expression.

    Args:
        value: Either a ``{"value": ..., "units": ...}`` mapping, a
            ``"<number> <units>"`` string (the form ``TimeConfig.time_step``
            uses), or any other value, which is returned unchanged.
        strict: Whether a string with no numeric magnitude is an error. Pass
            ``False`` where a plain string is itself a legal value — a
            construction option naming an enum member (``"polarization"``) is
            not a botched quantity — and it is returned unchanged instead.

    Returns:
        A units-carrying Pyomo expression, or ``value`` itself.

    Raises:
        FlexConfigError: If ``strict`` and a quantity string has no numeric
            magnitude.
    """
    if isinstance(value, dict) and set(value) == _QUANTITY_KEYS:
        return value["value"] * parse_units(value["units"])
    if isinstance(value, str):
        magnitude, _, units = value.strip().partition(" ")
        try:
            return float(magnitude) * parse_units(units)
        except ValueError as exc:
            if not strict:
                return value
            raise FlexConfigError(
                f"Could not read {value!r} as a quantity; write it as a number "
                "and its units, e.g. '15 min'.",
                value=value,
            ) from exc
    return value


def build_model(config) -> pyo.ConcreteModel:
    """Build the whole Pyomo model described by a config.

    Args:
        config: A :class:`~flexcore.config.schema.ModelConfig`, or a path or
            mapping round-tripped through
            :func:`~flexcore.config.io.load_model_config` first (never used
            raw — anything a user configures must go through the validated
            schema, not an ad hoc dict).

    Returns:
        The constructed ``ConcreteModel``, carrying ``time_block``,
        ``properties``, ``costing``, the plant or network tree, and
        ``objective``. Arcs are declared but not expanded.

    Raises:
        FlexConfigError: If the config fails validation (the message names the
            offending field path, and ``__cause__`` is the underlying pydantic
            ``ValidationError``), or names an unknown unit-model class.
    """
    cfg = config if isinstance(config, ModelConfig) else load_model_config(config)

    model = pyo.ConcreteModel(name=(cfg.network or cfg.plant).name)
    model.time_block = TimeBlock(
        start_date=cfg.time.start_date,
        end_date=cfg.time.end_date,
        time_step=parse_quantity(cfg.time.time_step),
    )
    model.properties = SimpleAqueousFlow(**cfg.properties)
    model.costing = _build_costing(model, cfg)

    if cfg.network is not None:
        model.add_component(cfg.network.name, NetworkBlock(time_block=model.time_block))
        network = model.find_component(cfg.network.name)
        for name, plant_cfg in cfg.network.plants.items():
            _build_plant(network, name, plant_cfg, model)
        _build_arcs(network, cfg.network.arcs)
    else:
        _build_plant(model, cfg.plant.name, cfg.plant, model)

    model.costing.cost_process()
    model.objective = pyo.Objective(expr=model.costing.aggregate_operating_cost)
    return model


def _build_costing(model, cfg: ModelConfig):
    """Build the FlexCosting block from a ``CostingConfig``.

    Args:
        model: The model being built (supplies the TimeBlock).
        cfg: The validated whole-model config.

    Returns:
        The constructible ``FlexCosting`` block.
    """
    costing = cfg.costing
    prices = {
        name: parse_quantity({"value": spec.value, "units": spec.units})
        for name, spec in (costing.energy_prices or {}).items()
    }
    return FlexCosting(
        time_block=model.time_block,
        tariff_file=costing.tariff_source,
        energy_prices=prices or None,
        currency=costing.currency,
        dr_event_file=None if costing.dr is None else costing.dr.events_source,
        consumption_estimate=costing.consumption_estimate,
        fixed_operating_cost=costing.fixed_operating_cost,
        prorate_monthly_charges=costing.prorate_monthly_charges,
        lifetime_years=costing.lifetime_years,
        discount_rate=costing.discount_rate,
        interest_rate=costing.interest_rate,
    )


def _build_plant(parent, name: str, plant_cfg: PlantConfig, model) -> None:
    """Attach a PlantBlock named ``name`` to ``parent`` and populate it.

    Args:
        parent: The model or NetworkBlock the plant is attached to.
        name: The plant's attribute name on ``parent``.
        plant_cfg: The validated plant config.
        model: The whole model, supplying the TimeBlock, properties, costing.
    """
    parent.add_component(name, PlantBlock(time_block=model.time_block))
    plant = parent.find_component(name)
    for unit_name, unit_cfg in plant_cfg.units.items():
        plant.add_component(
            unit_name,
            OpsBlockData.build_from_config(
                unit_cfg,
                property_package=model.properties,
                costing_package=model.costing,
            ),
        )
        _apply_external_dispatch(plant.find_component(unit_name), unit_cfg)
    _build_arcs(plant, plant_cfg.arcs)


def _build_arcs(block, arcs) -> None:
    """Build the declared arcs on ``block`` as ``arc_0``, ``arc_1``, ....

    Args:
        block: The plant or network the arcs belong to.
        arcs: The validated :class:`~flexcore.config.schema.ArcSpec` list.

    Raises:
        FlexConfigError: If an endpoint does not resolve to a port on ``block``.
    """
    for index, arc in enumerate(arcs):
        endpoints = {}
        for role, path in (("source", arc.source), ("destination", arc.destination)):
            port = block.find_component(path)
            if port is None:
                raise FlexConfigError(
                    f"Arc {role} {path!r} is not a port on {block.name!r}. Write "
                    "it as 'unit.port' relative to the plant (or "
                    "'plant.unit.port' relative to the network).",
                    field=role,
                    value=path,
                )
            endpoints[role] = port
        block.add_component(f"arc_{index}", Arc(**endpoints))


def _apply_external_dispatch(unit, unit_cfg) -> None:
    """Fix a unit's actuator to a declared external (DERMS) command series.

    Args:
        unit: The built unit block.
        unit_cfg: Its validated ``UnitConfig``.

    Raises:
        FlexConfigError: If the declared variable is not on the unit, or the
            source file cannot be read as a time-indexed series.
    """
    spec = unit_cfg.external_dispatch
    if spec is None:
        return
    var = unit.find_component(spec.variable)
    if var is None:
        raise FlexConfigError(
            f"external_dispatch names variable {spec.variable!r}, which is not "
            f"on {unit.name!r}.",
            field="external_dispatch.variable",
            value=spec.variable,
        )
    try:
        raw = json.loads(Path(spec.source).read_text())
    except (OSError, ValueError) as exc:
        raise FlexConfigError(
            f"Could not read external-dispatch series {spec.source!r}: {exc}. "
            "Provide a JSON mapping of time index (or timestamp) to value.",
            field="external_dispatch.source",
            value=spec.source,
        ) from exc
    # JSON keys are always strings; integer time indices come back as "0".
    series = {
        int(key) if key.lstrip("-").isdigit() else key: value
        for key, value in raw.items()
    }
    unit.set_external_dispatch(var, series, fix=spec.fix)
