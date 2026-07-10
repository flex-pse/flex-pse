"""Load, dump, migrate, and export the flex-pse config (R3).

YAML is the canonical on-disk format (pydantic stays the schema authority); JSON
is also accepted. The format is chosen by file suffix. Loading validates the
version first (missing or too-new ``schema_version`` is an error; older versions
step through :data:`MIGRATIONS`), then validates against
:class:`~flexcore.config.schema.ModelConfig`, wrapping any pydantic error in a
:class:`~flexcore.exceptions.FlexConfigError` that preserves the offending field
path.
"""

import json
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import ValidationError

from flexcore.config.schema import CURRENT_SCHEMA_VERSION, ModelConfig
from flexcore.exceptions import FlexConfigError

MIGRATIONS: dict[int, Callable[[dict], dict]] = {}
"""Version -> one-step upgrade hook, applied in sequence on load. Empty at v1."""

_SCHEMA_FILENAME = "model_config.schema.json"


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


def _read(path: Path) -> dict:
    """Parse a config file to a plain dict, dispatching on the suffix.

    Args:
        path: The config file path (``.yaml``/``.yml`` or ``.json``).

    Returns:
        The parsed top-level mapping.

    Raises:
        FlexConfigError: For an unsupported suffix or a non-mapping document.
    """
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise FlexConfigError(
            f"Unsupported config format {suffix!r} for {path}. Use a .yaml, "
            ".yml, or .json file.",
            value=str(path),
        )
    if not isinstance(data, dict):
        raise FlexConfigError(
            f"Config file {path} must contain a mapping at the top level, got "
            f"{type(data).__name__}.",
            value=str(path),
        )
    return data


def load_model_config(path) -> ModelConfig:
    """Load and validate a config file into a model config.

    Args:
        path: Path to a ``.yaml``/``.yml`` or ``.json`` config file.

    Returns:
        The validated :class:`~flexcore.config.schema.ModelConfig`.

    Raises:
        FlexConfigError: If the format is unsupported, ``schema_version`` is
            missing or newer than this build, a migration step is missing, or
            the config fails validation (the message names the bad field path).
    """
    path = Path(path)
    data = _read(path)

    version = data.get("schema_version")
    if version is None:
        raise FlexConfigError(
            f"Config {path} has no 'schema_version'. Every persisted config "
            f"must declare one (this build writes version "
            f"{CURRENT_SCHEMA_VERSION}).",
            field="schema_version",
        )
    if not isinstance(version, int) or isinstance(version, bool):
        raise FlexConfigError(
            f"'schema_version' must be an integer, got {version!r}.",
            field="schema_version",
            value=version,
        )
    if version > CURRENT_SCHEMA_VERSION:
        raise FlexConfigError(
            f"Config {path} declares schema_version {version}, newer than this "
            f"build supports ({CURRENT_SCHEMA_VERSION}). Upgrade flex-pse.",
            field="schema_version",
            value=version,
        )
    while version < CURRENT_SCHEMA_VERSION:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise FlexConfigError(
                f"No migration registered from schema_version {version} to "
                f"{version + 1}; cannot upgrade {path}.",
                field="schema_version",
                value=version,
            )
        data = migrate(data)
        version += 1
        data["schema_version"] = version

    try:
        return ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc


def dump_model_config(cfg: ModelConfig, path) -> None:
    """Write a model config to disk in the format its file suffix names.

    ``.yaml``/``.yml`` targets are written as YAML (the canonical format);
    ``.json`` targets as indented JSON. Ambiguous bare scalars (``no``/``on``/
    ``yes``) are quoted by ``yaml.safe_dump`` so the YAML "Norway problem"
    cannot bite on reload.

    Args:
        cfg: The :class:`~flexcore.config.schema.ModelConfig` to serialize.
        path: Destination path with a ``.yaml``/``.yml`` or ``.json`` suffix.

    Raises:
        FlexConfigError: For an unsupported suffix.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        text = yaml.safe_dump(
            cfg.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=True,
        )
        path.write_text(text)
    elif suffix == ".json":
        path.write_text(cfg.model_dump_json(indent=2))
    else:
        raise FlexConfigError(
            f"Unsupported config format {suffix!r} for {path}. Use a .yaml, "
            ".yml, or .json file.",
            value=str(path),
        )


def export_json_schemas(directory) -> None:
    """Write the exported JSON Schema for the model config to ``directory``.

    Serializes :class:`~flexcore.config.schema.ModelConfig`'s JSON Schema with
    ``indent=2`` and ``sort_keys=True`` so the
    checked-in schema diffs only on real schema changes (pitfall 7). Run once
    and commit the result to ``src/flexcore/config/schemas/``.

    Args:
        directory: Destination directory for ``model_config.schema.json``.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schema = ModelConfig.model_json_schema()
    text = json.dumps(schema, indent=2, sort_keys=True)
    (directory / _SCHEMA_FILENAME).write_text(text)
