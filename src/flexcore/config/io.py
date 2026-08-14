"""Load, dump, migrate, and export the flex-pse config.

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

from flexcore.config.schema import CURRENT_SCHEMA_VERSION, ModelConfig, SurrogateSpec
from flexcore.exceptions import FlexConfigError


def _to_0_0_2(data: dict) -> dict:
    """Upgrade a 0.0.1 config to 0.0.2 by re-stamping its version.

    0.0.2 only widened ``SurrogateSpec``: ``functional_form`` became an open
    string and the optional ``source`` was added. Every 0.0.1 document is a
    valid 0.0.2 document, so there is no data to rewrite.

    Args:
        data: The parsed 0.0.1 config.

    Returns:
        The same mapping, stamped 0.0.2.
    """
    return {**data, "schema_version": "0.0.2"}


MIGRATIONS: dict[str, Callable[[dict], dict]] = {"0.0.1": _to_0_0_2}
"""Source version -> upgrade hook, applied in sequence on load. Each hook must
set the new ``schema_version`` on the dict it returns."""

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


_SURROGATE_SOURCE_FIELDS = ("coefficients", "input_variables", "output_variables")


def load_surrogate_source(spec: SurrogateSpec, base_dir=None) -> SurrogateSpec:
    """Fill a surrogate in from the sidecar file its ``source`` names.

    Lets a relationship live beside the config rather than inside it — a fitted
    curve with hundreds of terms, or one a tool wrote separately. The file is a
    JSON object holding any of ``coefficients``, ``input_variables`` and
    ``output_variables``; whatever it supplies replaces what the spec wrote
    inline. ``source`` is kept on the result, so the config still points at
    where the relationship came from.

    Args:
        spec: The :class:`~flexcore.config.schema.SurrogateSpec` to fill in;
            returned unchanged when it names no ``source``. Never mutated.
        base_dir: Directory a relative ``source`` resolves against (the
            directory of the config file that named it). None resolves against
            the current working directory.

    Returns:
        A filled-in copy of ``spec``.

    Raises:
        FlexConfigError: If the source is not a readable ``.json`` file, is not
            a JSON object, carries a key that is not a surrogate field, or
            supplies a value that fails validation.
    """
    if spec.source is None:
        return spec
    path = Path(spec.source)
    if base_dir is not None and not path.is_absolute():
        path = Path(base_dir) / path
    if path.suffix.lower() != ".json":
        raise FlexConfigError(
            f"Unsupported surrogate source format {path.suffix!r} for "
            f"{spec.source}. Use a .json file.",
            field="source",
            value=spec.source,
        )
    try:
        data = json.loads(path.read_text())
    except OSError as exc:
        raise FlexConfigError(
            f"Could not read the surrogate source {path}: {exc.strerror}.",
            field="source",
            value=spec.source,
        ) from exc
    if not isinstance(data, dict):
        raise FlexConfigError(
            f"Surrogate source {path} must contain a JSON object with any of "
            f"{', '.join(_SURROGATE_SOURCE_FIELDS)}, got "
            f"{type(data).__name__}.",
            field="source",
            value=spec.source,
        )
    unknown = sorted(set(data) - set(_SURROGATE_SOURCE_FIELDS))
    if unknown:
        raise FlexConfigError(
            f"Surrogate source {path} carries unknown key(s) {unknown}; it may "
            f"only supply {', '.join(_SURROGATE_SOURCE_FIELDS)}.",
            field="source",
            value=unknown,
        )
    try:
        return SurrogateSpec.model_validate({**spec.model_dump(), **data})
    except ValidationError as exc:
        raise FlexConfigError(
            f"Invalid surrogate source {path}:\n{_format_validation_error(exc)}",
            field="source",
            value=spec.source,
        ) from exc


def _resolve_surrogate_sources(cfg: ModelConfig, base_dir) -> ModelConfig:
    """Fill in every unit surrogate that names a sidecar file.

    Called once at the config boundary so nothing downstream ever sees a spec
    whose coefficients are still on disk (conventions §4).

    Args:
        cfg: The validated config, mutated in place.
        base_dir: Directory a relative source resolves against.

    Returns:
        ``cfg``.
    """
    plants = cfg.network.plants.values() if cfg.network is not None else [cfg.plant]
    for plant in plants:
        for unit in plant.units.values():
            if unit.surrogate is not None and unit.surrogate.source is not None:
                unit.surrogate = load_surrogate_source(unit.surrogate, base_dir)
    return cfg


def load_model_config(source) -> ModelConfig:
    """Load and validate a config file or dict into a model config.

    Any unit surrogate naming a ``source`` sidecar is filled in here, at the
    config boundary, so nothing downstream sees a half-loaded relationship. A
    relative source resolves against the config file's own directory, or the
    working directory when the config came in as a dict.

    Args:
        source: Path to a ``.json`` config file, or an already-parsed config
            mapping (which is not mutated).

    Returns:
        The validated :class:`~flexcore.config.schema.ModelConfig`.

    Raises:
        FlexConfigError: If the format is unsupported, ``schema_version`` is
            missing, malformed, or newer than this build, a migration step is
            missing, the config fails validation (the message names the bad
            field path), or a surrogate ``source`` cannot be loaded.
    """
    if isinstance(source, Mapping):
        data, name, base_dir = dict(source), "the config dict", None
    else:
        path = Path(source)
        data, name, base_dir = _read(path), str(path), path.parent

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
        cfg = ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc
    return _resolve_surrogate_sources(cfg, base_dir)


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
