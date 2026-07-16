"""Load, dump, migrate, and export the flex-pse config (R3).

JSON is the canonical on-disk format (pydantic stays the schema authority); an
already-parsed dict is accepted directly. Loading validates the version first
(a missing, malformed, or too-new ``schema_version`` is an error; older
versions step
through :data:`MIGRATIONS`), then validates against
:class:`~flexcore.config.schema.ModelConfig`, wrapping any pydantic error in a
:class:`~flexcore.exceptions.FlexConfigError` that preserves the offending field
path.
"""

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import ValidationError

from flexcore.config.schema import CURRENT_SCHEMA_VERSION, ModelConfig
from flexcore.exceptions import FlexConfigError

MIGRATIONS: dict[str, Callable[[dict], dict]] = {}
"""Source version -> upgrade hook, applied in sequence on load. Each hook must
set the new ``schema_version`` on the dict it returns. Empty at 0.0.1."""

_SCHEMA_FILENAME = "model_config.schema.json"
_SEMVER = re.compile(r"^\d+\.\d+\.\d+$")


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ValidationError with dotted field paths.

    Args:
        exc: The pydantic validation error.

    Returns:
        A multi-line message; each line names the dotted field path (e.g.
        ``plant.units.tank.io_variables.0.role``) and what was wrong.
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
            f"{CURRENT_SCHEMA_VERSION!r}, got {version!r} in {source}.",
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


def load_model_config(source) -> ModelConfig:
    """Load and validate a config file or dict into a model config.

    Args:
        source: Path to a ``.json`` config file, or an already-parsed config
            mapping (which is not mutated).

    Returns:
        The validated :class:`~flexcore.config.schema.ModelConfig`.

    Raises:
        FlexConfigError: If the format is unsupported, ``schema_version`` is
            missing, malformed, or newer than this build, a migration step is
            missing, or the config fails validation (the message names the bad
            field path).
    """
    if isinstance(source, Mapping):
        data, name = dict(source), "the config dict"
    else:
        path = Path(source)
        data, name = _read(path), str(path)

    version = data.get("schema_version")
    if version is None:
        raise FlexConfigError(
            f"Config {name} has no 'schema_version'. Every persisted config "
            f"must declare one (this build writes version "
            f"{CURRENT_SCHEMA_VERSION!r}).",
            field="schema_version",
        )
    parsed = _parse_version(version, name)
    current = _parse_version(CURRENT_SCHEMA_VERSION, "this build")
    if parsed > current:
        raise FlexConfigError(
            f"Config {name} declares schema_version {version!r}, newer than "
            f"this build supports ({CURRENT_SCHEMA_VERSION!r}). Upgrade "
            f"flex-pse.",
            field="schema_version",
            value=version,
        )
    while parsed < current:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise FlexConfigError(
                f"No migration registered from schema_version {version!r}; "
                f"cannot upgrade {name} to {CURRENT_SCHEMA_VERSION!r}.",
                field="schema_version",
                value=version,
            )
        data = migrate(data)
        new_version = data.get("schema_version")
        new_parsed = _parse_version(new_version, name)
        if new_parsed <= parsed:
            raise FlexConfigError(
                f"Migration from schema_version {version!r} did not advance "
                f"the version (got {new_version!r}).",
                field="schema_version",
                value=new_version,
            )
        version, parsed = new_version, new_parsed

    try:
        return ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc


def dump_model_config(cfg: ModelConfig, path) -> None:
    """Write a model config to disk as indented JSON.

    Args:
        cfg: The :class:`~flexcore.config.schema.ModelConfig` to serialize.
        path: Destination path with a ``.json`` suffix.

    Raises:
        FlexConfigError: For a non-``.json`` suffix.
    """
    path = Path(path)
    if path.suffix.lower() != ".json":
        raise FlexConfigError(
            f"Unsupported config format {path.suffix!r} for {path}. Use a "
            ".json file.",
            value=str(path),
        )
    path.write_text(cfg.model_dump_json(indent=2))


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


def export_json_schemas(directory, filename: str = _SCHEMA_FILENAME) -> None:
    """Write the exported JSON Schema for the model config to ``directory``.

    Serializes :class:`~flexcore.config.schema.ModelConfig`'s JSON Schema with
    ``indent=2`` and ``sort_keys=True`` so the checked-in schema diffs only on
    real schema changes (pitfall 7). Descriptions are collapsed to single-line
    plain text — line wrapping is the documentation builder's job, not the
    schema's. Run once and commit the result to
    ``src/flexcore/config/schemas/``.

    Args:
        directory: Destination directory for the schema file.
        filename: Output filename; override it to keep schemas for several
            versions side by side in one directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schema = ModelConfig.model_json_schema()
    _plain_descriptions(schema)
    text = json.dumps(schema, indent=2, sort_keys=True)
    (directory / filename).write_text(text)
