"""Convert the legacy nested ModelConfig into a flat FlowsheetSpec."""

import json
from pathlib import Path

from flexcore.config.io import load_surrogate_source, resolve_source_path
from flexcore.config.schema import ExternalDispatchSpec, ModelConfig, PlantConfig
from flexcore.config.spec import SCHEMA_VERSION, FlowsheetSpec
from flexcore.exceptions import FlexConfigError

CONSTRUCTOR_RELATION = {"Digestor": "biogas_relation"}
"""Unit classes whose constructor swaps a relation other than the power one."""


def _dispatch_series(spec: ExternalDispatchSpec, base_dir: Path | None) -> dict:
    """Read a legacy external-dispatch file into a time index -> value mapping.

    Args:
        spec: The legacy dispatch spec naming the file.
        base_dir: Directory a relative path resolves against.

    Returns:
        The series as written in the file.

    Raises:
        FlexConfigError: If the file cannot be read as a JSON mapping.
    """
    try:
        series = json.loads(resolve_source_path(spec.source, base_dir).read_text())
    except (OSError, ValueError) as exc:
        raise FlexConfigError(
            f"Could not read external-dispatch series {spec.source!r}: {exc}. "
            "Provide a JSON mapping of time index to value.",
            field="external_dispatch.source",
            value=spec.source,
        ) from exc
    if not isinstance(series, dict):
        raise FlexConfigError(
            f"External-dispatch series {spec.source!r} must be a JSON mapping of "
            f"time index to value, got {type(series).__name__}.",
            field="external_dispatch.source",
            value=spec.source,
        )
    return series


def _unit_elements(
    path: str, plant: PlantConfig, base_dir: Path | None
) -> tuple[list[dict], list[dict]]:
    """Return a nested plant's unit elements and its surrogate/dispatch elements."""
    units, extras = [], []
    for key, unit in plant.units.items():
        name = f"{path}.{key}"
        units.append(
            {
                "kind": "unit",
                "name": name,
                "unit_model_class": unit.unit_model_class,
                "construction_options": unit.construction_options,
                "property_package": unit.property_package,
                "costing_package": "auto" if unit.costing else None,
                "io_variables": [v.model_dump() for v in unit.io_variables],
                "unit_commitment": unit.unit_commitment
                and unit.unit_commitment.model_dump(),
            }
        )
        if unit.surrogate is not None:
            surrogate = load_surrogate_source(unit.surrogate, base_dir)
            extras.append(
                {
                    "kind": "surrogate",
                    "unit": name,
                    "relation": CONSTRUCTOR_RELATION.get(
                        unit.unit_model_class, "power_electrical_relation"
                    ),
                    "surrogate_type": surrogate.surrogate_type,
                    "data": surrogate.data,
                    "provenance": surrogate.provenance,
                }
            )
        if unit.external_dispatch is not None:
            extras.append(
                {
                    "kind": "dispatch",
                    "unit": name,
                    "variable": unit.external_dispatch.variable,
                    "values": _dispatch_series(unit.external_dispatch, base_dir),
                    "fix": unit.external_dispatch.fix,
                }
            )
    return units, extras


def nested_to_spec(cfg: ModelConfig) -> FlowsheetSpec:
    """Convert a legacy nested config into the equivalent flat spec.

    Args:
        cfg: The validated nested config.

    Returns:
        The flat spec that builds the same model.

    Raises:
        FlexConfigError: If an arc names a source unit its plant does not have,
            or a surrogate or external-dispatch file cannot be read.
    """
    elements: list[dict] = [{"kind": "time", **cfg.time.model_dump()}]
    for key, package in cfg.properties.items():
        elements.append(
            {"kind": "property_package", "name": key, **package.model_dump()}
        )
    costing = cfg.costing.model_dump(exclude={"objective", "solver"})
    elements.append({"kind": "costing", "name": "costing", **costing})
    elements.append({"kind": "objective", "expression": cfg.costing.objective})

    # Each arc list with the path its endpoints are relative to and how many
    # leading segments of the source name its unit.
    if cfg.network is None:
        name = cfg.plant.name
        plants = {name: cfg.plant}
        arc_lists = [(name, 1, cfg.plant.arcs)]
    else:
        name = cfg.network.name
        elements.append({"kind": "network", "name": name})
        plants = {f"{name}.{key}": plant for key, plant in cfg.network.plants.items()}
        arc_lists = [(name, 2, cfg.network.arcs)]
        arc_lists += [(path, 1, plant.arcs) for path, plant in plants.items()]

    units: dict[str, dict] = {}
    extras: list[dict] = []
    for path, plant in plants.items():
        plant_units, plant_extras = _unit_elements(path, plant, cfg._base_dir)
        elements.append({"kind": "plant", "name": path})
        units.update({unit["name"]: unit for unit in plant_units})
        extras += plant_extras
    for prefix, depth, arcs in arc_lists:
        for index, arc in enumerate(arcs):
            *unit_parts, port = arc.source.split(".", depth)
            source_unit = ".".join([prefix, *unit_parts])
            if source_unit not in units:
                raise FlexConfigError(
                    f"Arc source {arc.source!r} names no unit under {prefix!r}. "
                    f"Units: {sorted(units)}.",
                    field="source",
                    value=arc.source,
                )
            units[source_unit].setdefault("connections", []).append(
                {
                    "port": port,
                    "to": f"{prefix}.{arc.destination}",
                    "name": f"arc_{index}",
                }
            )

    spec = FlowsheetSpec.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "solver": cfg.costing.solver,
            "elements": [*elements, *units.values(), *extras],
        }
    )
    spec._base_dir = cfg._base_dir
    return spec
