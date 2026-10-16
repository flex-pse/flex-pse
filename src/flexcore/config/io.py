"""Load, dump, and export the flat flowsheet spec.

JSON is the canonical on-disk format (pydantic stays the schema authority); an
already-parsed dict is accepted directly. Loading validates the version first
(a missing, malformed, or too-new ``schema_version`` is an error; older
versions step through :data:`MIGRATIONS`), then validates against
:class:`~flexcore.config.spec.FlowsheetSpec`, wrapping any pydantic error in a
:class:`~flexcore.exceptions.FlexConfigError` that preserves the offending field
path.
"""

import gzip
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from flexcore.config.spec import (
    KINDS,
    SCHEMA_VERSION,
    STAGE_ORDER,
    FlowsheetSpec,
    SourceRef,
    element_target,
)
from flexcore.exceptions import FlexConfigError

MIGRATIONS: dict[str, Callable[[dict], dict]] = {}
"""Source version -> upgrade hook, applied in sequence on load. Each hook must
set the new ``schema_version`` on the dict it returns."""

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


def resolve_sources(value, base_dir):
    """Replace every ``$source`` reference in ``value`` with the data it names.

    Args:
        value: A :class:`~flexcore.config.spec.SourceRef`, a
            ``{"$source": path}`` mapping, or any JSON data containing them.
        base_dir: Directory a relative path resolves against, or None.

    Returns:
        ``value`` with each reference replaced; the input is not mutated.

    Raises:
        FlexConfigError: If a referenced file cannot be read.
    """
    if isinstance(value, SourceRef):
        return read_source(value.source, base_dir)
    if isinstance(value, dict):
        if set(value) == {"$source"}:
            return read_source(value["$source"], base_dir)
        return {key: resolve_sources(item, base_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_sources(item, base_dir) for item in value]
    return value


def load_spec(source) -> FlowsheetSpec:
    """Load and validate a flat spec.

    Args:
        source: A ``.json`` path, a parsed mapping (not mutated), or a
            :class:`~flexcore.config.spec.FlowsheetSpec` (returned as is).

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


def dump_spec(spec: FlowsheetSpec, path) -> None:
    """Write a flat spec to disk as indented JSON, elements in canonical order.

    Elements are sorted by build stage, kind and name so files diff cleanly;
    defaults are left out.

    Args:
        spec: The spec to serialize; it is not reordered.
        path: Destination path ending in ``.json``.

    Raises:
        FlexConfigError: For any other suffix.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise FlexConfigError(
            f"Unsupported config format {path.suffix!r} for {path}. Use a "
            ".json file.",
            value=str(path),
        )
    ordered = sorted(
        spec.elements,
        key=lambda el: (
            STAGE_ORDER.index(KINDS[el.kind][0]),
            el.kind,
            getattr(el, "name", None) or "",
            element_target(el),
        ),
    )
    text = spec.model_copy(update={"elements": ordered}).model_dump_json(
        indent=2, by_alias=True, exclude_defaults=True
    )
    path.write_text(text)


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
