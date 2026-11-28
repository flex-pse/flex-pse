"""Load, dump, and export the flat flowsheet spec.

JSON is the canonical on-disk format (pydantic stays the schema authority); an
already-parsed dict is accepted directly. Loading validates the version first
(a missing, malformed, or too-new ``schema_version`` is an error; older
versions step through :data:`MIGRATIONS`), then validates against
:class:`~flexcore.config.spec.FlowsheetSpec`, wrapping any pydantic error in a
:class:`~flexcore.exceptions.FlexConfigError` that preserves the offending field
path.
"""

import copy
import gzip
import json
import re
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from flexcore.config.layout import spec_file
from flexcore.config.schema import TimeConfig
from flexcore.config.spec import (
    KINDS,
    SCHEMA_VERSION,
    STAGE_ORDER,
    FlowsheetSpec,
    SourceRef,
    element_target,
)
from flexcore.config.surrogate_migration import (
    is_legacy_surrogate,
    migrate_surrogate_spec,
)
from flexcore.exceptions import FlexConfigError, FlexMigrationWarning


def _migrate_0_1_0(data: dict) -> dict:
    """Rename BatteryModel's ``capacity`` option and Var to ``energy_capacity``.

    Args:
        data: A parsed 0.1.0 spec; left unchanged.

    Returns:
        A copy of the spec at version 0.1.1.
    """

    migrated = copy.deepcopy(data)
    raw = migrated.get("elements")
    elements = (
        [el for el in raw if isinstance(el, dict)] if isinstance(raw, list) else []
    )

    batteries = set()
    for el in elements:
        if el.get("kind") == "unit" and el.get("unit_model_class") == "BatteryModel":
            batteries.add(el.get("name"))
            options = el.get("construction_options")
            if isinstance(options, dict) and "capacity" in options:
                options["energy_capacity"] = options.pop("capacity")

    for el in elements:
        variable = el.get("variable")
        if el.get("kind") in ("fix", "unfix") and isinstance(variable, str):
            owner, dot, leaf = variable.rpartition(".")
            if leaf == "capacity" and re.sub(r"\[.*?\]", "", owner) in batteries:
                el["variable"] = f"{owner}{dot}energy_capacity"

    warnings.warn(
        "The spec was migrated from schema_version 0.1.0 to 0.1.1. The "
        "BatteryModel option and variable 'capacity' were renamed to "
        "'energy_capacity'. Please update your config files accordingly.",
        FlexMigrationWarning,
        stacklevel=3,
    )

    return {**migrated, "schema_version": "0.1.1"}


def _migrate_0_1_1(data: dict) -> dict:
    """Split every surrogate element's ``data`` into ``config`` and ``parameters``.

    See :func:`~flexcore.config.surrogate_migration.migrate_surrogate_spec` for
    the per-type translation. The migrated spec builds the same model as
    before, with every coefficient fixed.

    Args:
        data: A parsed 0.1.1 spec; left unchanged.

    Returns:
        A copy of the spec at version 0.2.0.

    Raises:
        FlexConfigError: If a surrogate element cannot be translated
            faithfully; the message names the element.
    """
    migrated = copy.deepcopy(data)
    raw = migrated.get("elements")
    elements = raw if isinstance(raw, list) else []
    count = 0
    for position, el in enumerate(elements):
        if not (isinstance(el, dict) and el.get("kind") == "surrogate"):
            continue
        if not is_legacy_surrogate(el):
            continue
        try:
            elements[position] = migrate_surrogate_spec(el)
        except FlexConfigError as exc:
            target = (
                f"{el.get('unit')}.{el.get('relation', 'power_electrical_relation')}"
            )
            raise FlexConfigError(
                f"Cannot migrate the surrogate element on {target} to schema_version "
                f"0.2.0: {exc}",
                field=f"elements.{position}.{exc.field}" if exc.field else None,
                value=exc.value,
            ) from exc
        count += 1

    if count:
        warnings.warn(
            f"The spec was migrated from schema_version 0.1.1 to 0.2.0. {count} "
            "surrogate element(s) had their 'data' split into 'config' and "
            "'parameters' (each parameter now records its value and whether it "
            "is fixed). Please re-export your config files.",
            FlexMigrationWarning,
            stacklevel=3,
        )

    return {**migrated, "schema_version": "0.2.0"}


MIGRATIONS: dict[str, Callable[[dict], dict]] = {
    "0.1.0": _migrate_0_1_0,
    "0.1.1": _migrate_0_1_1,
}
"""Source version -> upgrade hook, applied in sequence on load. Each hook must
set the new ``schema_version`` on the dict it returns."""

TimeCheck = Callable[[TimeConfig, str], None]
"""Refuses (raises for) a series made for other time points than the model's."""

_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ValidationError with dotted field paths.

    Args:
        exc: The pydantic validation error.

    Returns:
        A multi-line message; each line names the dotted field path (e.g.
        ``elements.3.unit.io_variables.0.role``) and what was wrong.
    """
    lines = []
    for err in exc.errors():
        path = ".".join(str(part) for part in err["loc"])
        lines.append(f"{path}: {err['msg']}" if path else err["msg"])
    return "Invalid model config:\n" + "\n".join(lines)


def _parse_version(version, source: str) -> tuple[int, int, int]:
    """Parse an ``X.Y.Z`` schema version into a comparable tuple.

    Args:
        version: The declared ``schema_version`` value.
        source: Where the version came from, for the error message.

    Raises:
        FlexConfigError: If ``version`` is not a semantic-version string.
    """
    if not isinstance(version, str) or not _SEMVER.match(version):
        raise FlexConfigError(
            f"'schema_version' must be a semantic-version string like "
            f"{SCHEMA_VERSION!r}, got {version!r} in {source}.",
            field="schema_version",
            value=version,
        )
    major, minor, patch = version.split(".")
    return (int(major), int(minor), int(patch))


def _read(path: Path) -> dict:
    """Parse a JSON config file to a plain dict.

    Args:
        path: The config file path (``.json``).

    Returns:
        The parsed top-level mapping.

    Raises:
        FlexConfigError: For a non-``.json`` suffix or a non-mapping document.
    """
    if path.suffix.lower() != ".json":
        raise FlexConfigError(
            f"Unsupported config format {path.suffix!r} for {path}. Use a "
            ".json file.",
            value=str(path),
        )
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise FlexConfigError(
            f"Config file {path} must contain a mapping at the top level, got "
            f"{type(data).__name__}.",
            value=str(path),
        )
    return data


def resolve_source_path(source: str, base_dir) -> Path:
    """Resolve a config-relative path against the config file's directory.

    Args:
        source: The path as written in the config.
        base_dir: The config file's directory, or None for the working directory.

    Returns:
        The path itself if absolute or ``base_dir`` is None, else ``base_dir / source``.
    """
    path = Path(source)
    return path if base_dir is None or path.is_absolute() else Path(base_dir) / path


def read_source(source: str, base_dir):
    """Read the JSON data a ``$source`` reference names.

    Args:
        source: Path of a ``.json`` or ``.json.gz`` file.
        base_dir: Directory a relative path resolves against, or None.

    Returns:
        The parsed JSON data.

    Raises:
        FlexConfigError: If the file has another suffix or cannot be read.
    """
    path = resolve_source_path(source, base_dir)
    if not path.name.lower().endswith((".json", ".json.gz")):
        raise FlexConfigError(
            f"Unsupported $source format for {source!r}. Use a .json or .json.gz "
            "file.",
            field="$source",
            value=source,
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FlexConfigError(
            f"Could not read the $source file {path}: {exc.strerror}.",
            field="$source",
            value=source,
        ) from exc
    return json.loads(gzip.decompress(raw) if path.suffix.lower() == ".gz" else raw)


def _series_values(data, source: str, check_time: TimeCheck | None):
    """Return the values of a time series document, or other data unchanged.

    Args:
        data: The parsed JSON data of a ``$source`` file.
        source: The file's path as written, for error messages.
        check_time: Called with a document's time and ``source``, or None.

    Raises:
        FlexConfigError: If a document's ``time`` is not a valid time
            definition, or ``check_time`` refuses it.
    """
    if not (
        isinstance(data, dict)
        and {"time", "values"} <= set(data)
        and isinstance(data["time"], dict)
    ):
        return data
    try:
        time = TimeConfig.model_validate(data["time"])
    except ValidationError as exc:
        raise FlexConfigError(
            f"The 'time' of $source file {source!r} is not a time definition. "
            f"{_format_validation_error(exc)}",
            field="$source",
            value=source,
        ) from exc
    if check_time is not None:
        check_time(time, source)
    return data["values"]


def resolve_sources(value, base_dir, check_time: TimeCheck | None = None):
    """Replace every ``$source`` reference in ``value`` with the data it names.

    A file may hold a time series document, ``{"time": {...}, "values": [...]}``,
    whose ``time`` has the fields of a spec's time element. It is replaced by its
    ``values``; other keys, such as notes on where the data came from, are ignored.

    Args:
        value: A :class:`~flexcore.config.spec.SourceRef`, a
            ``{"$source": path}`` mapping, or any JSON data containing them.
        base_dir: Directory a relative path resolves against, or None.
        check_time: Called with each document's time and path, to refuse a series
            made for other time points. None skips the check.

    Returns:
        ``value`` with each reference replaced; the input is not mutated.

    Raises:
        FlexConfigError: If a referenced file cannot be read, or a document's
            ``time`` is invalid or refused.
    """
    if isinstance(value, SourceRef):
        value = {"$source": value.source}
    if isinstance(value, dict):
        if set(value) == {"$source"}:
            source = value["$source"]
            return _series_values(read_source(source, base_dir), source, check_time)
        return {
            key: resolve_sources(item, base_dir, check_time)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [resolve_sources(item, base_dir, check_time) for item in value]
    return value


def load_spec(source) -> FlowsheetSpec:
    """Load and validate a flat spec.

    Args:
        source: A ``.json`` path, a config folder holding ``config.json``
            (see :mod:`flexcore.config.layout`), a parsed mapping (not
            mutated), or a :class:`~flexcore.config.spec.FlowsheetSpec`
            (returned as is).

    Returns:
        The validated spec, with its base directory set for relative paths.

    Raises:
        FlexConfigError: If the document is unreadable, has a missing,
            malformed or too-new ``schema_version``, lacks a migration, or fails
            validation (the message names the bad field path).
    """
    if isinstance(source, FlowsheetSpec):
        return source
    if isinstance(source, Mapping):
        data, name, base_dir = dict(source), "the config dict", None
    else:
        path = Path(source)
        if path.is_dir():
            path = spec_file(path)
        data, name, base_dir = _read(path), str(path), path.parent

    version = data.get("schema_version")
    if version is None:
        raise FlexConfigError(
            f"Config {name} has no 'schema_version'. Every persisted config "
            f"must declare one (this build writes version {SCHEMA_VERSION!r}).",
            field="schema_version",
        )
    parsed = _parse_version(version, name)
    current = _parse_version(SCHEMA_VERSION, "this build")
    if parsed > current:
        raise FlexConfigError(
            f"Config {name} declares schema_version {version!r}, newer than "
            f"this build supports ({SCHEMA_VERSION!r}). Upgrade flex-pse.",
            field="schema_version",
            value=version,
        )
    while parsed < current:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise FlexConfigError(
                f"No migration registered from schema_version {version!r}; "
                f"cannot upgrade {name} to {SCHEMA_VERSION!r}.",
                field="schema_version",
                value=version,
            )
        data = migrate(data)
        version = data.get("schema_version")
        parsed = _parse_version(version, name)

    try:
        spec = FlowsheetSpec.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc
    spec._base_dir = base_dir
    return spec


def dumps_spec(spec: FlowsheetSpec) -> str:
    """Return a flat spec as indented JSON, elements in canonical order.

    Elements are sorted by build stage, kind and name so files diff cleanly;
    defaults are left out.

    Args:
        spec: The spec to serialize; it is not reordered.

    Returns:
        The text :func:`dump_spec` writes.
    """
    ordered = sorted(
        spec.elements,
        key=lambda el: (
            STAGE_ORDER.index(KINDS[el.kind][0]),
            el.kind,
            getattr(el, "name", None) or "",
            element_target(el),
        ),
    )
    return spec.model_copy(update={"elements": ordered}).model_dump_json(
        indent=2, by_alias=True, exclude_defaults=True
    )


def dump_spec(spec: FlowsheetSpec, path) -> None:
    """Write a flat spec to disk in the form :func:`dumps_spec` returns.

    Args:
        spec: The spec to serialize; it is not reordered.
        path: Destination path ending in ``.json``, or a config folder (a path
            with no suffix), which gets the spec as ``config.json`` and is
            created if missing.

    Raises:
        FlexConfigError: For any other suffix.
    """
    path = spec_file(path)
    if path.suffix.lower() != ".json":
        raise FlexConfigError(
            f"Unsupported config format {path.suffix!r} for {path}. Use a "
            ".json file.",
            value=str(path),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_spec(spec))


def _plain_descriptions(node) -> None:
    """Collapse every ``description`` in an exported schema to one line."""
    if isinstance(node, dict):
        description = node.get("description")
        if isinstance(description, str):
            node["description"] = " ".join(description.split())
        for value in node.values():
            _plain_descriptions(value)
    elif isinstance(node, list):
        for value in node:
            _plain_descriptions(value)


def export_json_schemas(
    directory, filename: str = "flowsheet_spec.schema.json"
) -> None:
    """Write the exported JSON Schema for the flat spec to ``directory``.

    Serializes :class:`~flexcore.config.spec.FlowsheetSpec`'s JSON Schema with
    ``indent=2`` and ``sort_keys=True`` so the checked-in schema diffs only on
    real schema changes. Descriptions are collapsed to single-line plain text.
    Run once and commit the result to ``src/flexcore/config/schemas/``.

    Args:
        directory: Destination directory for the schema file.
        filename: Output filename; override it to keep schemas for several
            versions side by side in one directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schema = FlowsheetSpec.model_json_schema()
    _plain_descriptions(schema)
    text = json.dumps(schema, indent=2, sort_keys=True)
    (directory / filename).write_text(text)
