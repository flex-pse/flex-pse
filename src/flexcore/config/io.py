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

import gzip
import json
import re
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel, ValidationError

from flexcore.config.schema import CURRENT_SCHEMA_VERSION, ModelConfig, SurrogateSpec
from flexcore.config.spec import (
    KINDS,
    SCHEMA_VERSION,
    STAGE_ORDER,
    FlowsheetSpec,
    SourceRef,
    element_target,
)
from flexcore.exceptions import FlexConfigError


def _to_0_0_2(data: dict, base_dir: Path | None) -> dict:
    """Upgrade a 0.0.1 config to 0.0.2 by re-stamping its version.

    0.0.2 only widened ``SurrogateSpec``: ``functional_form`` became an open
    string and the optional ``source`` was added. Every 0.0.1 document is a
    valid 0.0.2 document, so there is no data to rewrite.

    Args:
        data: The parsed 0.0.1 config.
        base_dir: Unused; every migration takes the document's directory.

    Returns:
        The same mapping, stamped 0.0.2.
    """
    return {**data, "schema_version": "0.0.2"}


def _iter_unit_configs(data: dict):
    """Yield every (name, unit config dict) in a raw (pre-validation) config.

    Args:
        data: The parsed config document.

    Yields:
        Each unit's attribute name and raw config mapping, wherever it sits (a
        bare plant, or every plant of a network).
    """
    network = data.get("network")
    plants = (
        network.get("plants", {}).values()
        if network is not None
        else [data.get("plant") or {}]
    )
    for plant in plants:
        yield from (plant or {}).get("units", {}).items()


def _to_0_0_3(data: dict, base_dir: Path | None) -> dict:
    """Reject a 0.0.2 config carrying a surrogate; re-stamp the rest.

    0.0.3 reshaped ``SurrogateSpec``: ``functional_form``/``coefficients``/
    ``input_variables``/``output_variables`` were replaced by a
    ``surrogate_type`` enum and an opaque ``data`` mapping (see
    ``flexops.surrogates``). The old shape cannot be mechanically translated
    into the new one (a coefficient key's units are no longer implicit), so a
    config naming a surrogate is rejected rather than guessed at.

    Args:
        data: The parsed 0.0.2 config.
        base_dir: Unused; every migration takes the document's directory.

    Returns:
        The same mapping, stamped 0.0.3, when it names no legacy-shaped
        surrogate.

    Raises:
        FlexConfigError: If any unit carries a surrogate in the old
            (``functional_form``) shape.
    """
    for _, unit in _iter_unit_configs(data):
        surrogate = unit.get("surrogate")
        if surrogate is not None and "surrogate_type" not in surrogate:
            raise FlexConfigError(
                "This 0.0.2 config carries a surrogate, which 0.0.3 cannot "
                "migrate automatically (the coefficient grammar was replaced "
                "by predefined, unit-declared surrogate classes). Re-emit "
                "this config with the current emit_model_config.",
                field="surrogate",
            )
    return {**data, "schema_version": "0.0.3"}


_UNBUILT_UC_DEFAULTS = {
    "startup_shutdown": False,
    "dwell": False,
    "min_up": None,
    "min_down": None,
    "delays": None,
    "conditional": None,
}


def _to_0_0_4(data: dict, base_dir: Path | None) -> dict:
    """Upgrade a 0.0.3 config to 0.0.4, rejecting unit-commitment fields nothing built.

    0.0.4 rejects the ``unit_commitment`` fields the builder never read, and
    turned ``properties`` from a dict of constructor kwargs into named
    ``PropertyPackageSpec`` entries. The old kwargs become the ``options`` of a
    single ``SimpleAqueousFlow`` package named ``properties``.

    Args:
        data: The parsed 0.0.3 config.
        base_dir: Unused; every migration takes the document's directory.

    Returns:
        A new mapping stamped 0.0.4; the input is not mutated.

    Raises:
        FlexConfigError: If any unit sets an unsupported ``unit_commitment``
            field to a non-default value.
    """
    for unit_name, unit in _iter_unit_configs(data):
        uc = unit.get("unit_commitment") or {}
        for name, default in _UNBUILT_UC_DEFAULTS.items():
            if name in uc and uc[name] != default:
                raise FlexConfigError(
                    f"Unit {unit_name!r} sets unit_commitment.{name}, which was never "
                    "built from config and is rejected as of 0.0.4. Remove it, "
                    "or build this logic in code with "
                    "flexops.logic.add_startup_shutdown.",
                    field=f"unit_commitment.{name}",
                    value=uc[name],
                )
    options = dict(data.get("properties") or {})
    return {
        **data,
        "schema_version": "0.0.4",
        "properties": {
            "properties": {"property_class": "SimpleAqueousFlow", "options": options}
        },
    }


def _to_0_1_0(data: dict, base_dir: Path | None) -> dict:
    """Upgrade a nested 0.0.4 config to the flat 0.1.0 flowsheet spec.

    Args:
        data: The parsed 0.0.4 config.
        base_dir: Directory the config's surrogate and dispatch files resolve
            against; their numbers are copied into the spec.

    Returns:
        The equivalent flat spec as a plain mapping.

    Raises:
        FlexConfigError: If the nested config fails validation or names a file
            that cannot be read.
    """
    # Local import: the converter reads legacy files through this module.
    from flexcore.config.convert import nested_to_spec

    try:
        cfg = ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc
    cfg._base_dir = base_dir
    return nested_to_spec(cfg).model_dump(mode="json")


MIGRATIONS: dict[str, Callable[[dict, Path | None], dict]] = {
    "0.0.1": _to_0_0_2,
    "0.0.2": _to_0_0_3,
    "0.0.3": _to_0_0_4,
    "0.0.4": _to_0_1_0,
}
"""Source version -> upgrade hook, applied in sequence on load. Each hook takes
the document and its directory, and must set the new ``schema_version`` on the
dict it returns. Versions up to 0.0.4 are the nested format; 0.1.0 is flat."""

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


_SURROGATE_SOURCE_FIELDS = ("data",)


def resolve_source_path(source: str, base_dir) -> Path:
    """Resolve a config-relative path against the config file's directory.

    Absolute paths are returned unchanged. A relative path resolves against
    ``base_dir``. If ``base_dir`` is None (the config came in as a dict), or the
    file does not exist there but does exist relative to the working directory,
    the working directory is used and a DeprecationWarning is emitted.

    Args:
        source: The path as written in the config.
        base_dir: The config file's directory, or None.

    Returns:
        The resolved path.

    Warns:
        DeprecationWarning: When the path falls back to the working directory.
    """
    path = Path(source)
    if path.is_absolute():
        return path
    if base_dir is not None and ((Path(base_dir) / path).exists() or not path.exists()):
        return Path(base_dir) / path
    warnings.warn(
        f"Path {source!r} resolved against the working directory; write it "
        "relative to the config file instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return path


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


def load_surrogate_source(spec: SurrogateSpec, base_dir=None) -> SurrogateSpec:
    """Fill a surrogate in from the sidecar file its ``source`` names.

    Lets a relationship live beside the config rather than inside it — a fitted
    curve with hundreds of terms, or one a tool wrote separately. The file is a
    JSON object holding ``data``, in the shape ``surrogate_type``'s class
    expects; it replaces what the spec wrote inline. ``source`` is kept on the
    result, so the config still points at where the relationship came from.

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
    path = resolve_source_path(spec.source, base_dir)
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
    whose coefficients are still on disk.

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


def _upgrade(data: dict, name: str, current: str, base_dir: Path | None) -> dict:
    """Step a parsed document up to ``current`` through ``MIGRATIONS``.

    Args:
        data: The parsed document, with a ``schema_version``.
        name: Where the document came from, for error messages.
        current: The version to reach.
        base_dir: The document's directory, passed to each migration.

    Returns:
        The document at ``current``.

    Raises:
        FlexConfigError: If the version is missing, malformed or newer than
            ``current``, a migration is missing, or one does not advance it.
    """
    version = data.get("schema_version")
    if version is None:
        raise FlexConfigError(
            f"Config {name} has no 'schema_version'. Every persisted config "
            f"must declare one (this build writes version {current!r}).",
            field="schema_version",
        )
    parsed = _parse_version(version, name)
    target = _parse_version(current, "this build")
    if parsed > target:
        raise FlexConfigError(
            f"Config {name} declares schema_version {version!r}, newer than "
            f"this build supports ({current!r}). Upgrade flex-pse.",
            field="schema_version",
            value=version,
        )
    while parsed < target:
        migrate = MIGRATIONS.get(version)
        if migrate is None:
            raise FlexConfigError(
                f"No migration registered from schema_version {version!r}; "
                f"cannot upgrade {name} to {current!r}.",
                field="schema_version",
                value=version,
            )
        data = migrate(data, base_dir)
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
    return data


def _is_flat_version(version) -> bool:
    """Return whether ``version`` is a valid version at or after the flat spec's."""
    return (
        isinstance(version, str)
        and bool(_SEMVER.match(version))
        and _parse_version(version, "") >= (0, 1, 0)
    )


def _source_document(source) -> tuple[dict, str, Path | None]:
    """Return a path or mapping as (parsed dict, display name, base directory)."""
    if isinstance(source, Mapping):
        return dict(source), "the config dict", None
    path = Path(source)
    return _read(path), str(path), path.parent


def load_model_config(source) -> ModelConfig:
    """Load and validate a legacy nested config file or dict (0.0.4 or older).

    Any unit surrogate naming a ``source`` sidecar is filled in here, at the
    config boundary, so nothing downstream sees a half-loaded relationship. A
    relative source resolves against the config file's own directory, or the
    working directory when the config came in as a dict.

    Args:
        source: Path to a ``.json`` config file, or an
            already-parsed config mapping (which is not mutated).

    Returns:
        The validated :class:`~flexcore.config.schema.ModelConfig`.

    Raises:
        FlexConfigError: If the format is unsupported, ``schema_version`` is
            missing, malformed, newer than this build, or belongs to a flat
            spec (use :func:`load_spec`), a migration step is missing, the
            config fails validation (the message names the bad field path), or
            a surrogate ``source`` cannot be loaded.
    """
    data, name, base_dir = _source_document(source)
    if _is_flat_version(data.get("schema_version")):
        raise FlexConfigError(
            f"Config {name} is a flat FlowsheetSpec (schema_version "
            f"{data['schema_version']!r}); load it with load_spec.",
            field="schema_version",
            value=data["schema_version"],
        )
    data = _upgrade(data, name, CURRENT_SCHEMA_VERSION, base_dir)
    try:
        cfg = ModelConfig.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc
    cfg._base_dir = None if base_dir is None else Path(base_dir)
    return _resolve_surrogate_sources(cfg, base_dir)


def load_spec(source) -> FlowsheetSpec:
    """Load any supported config as a validated flat spec.

    An older document, including a nested config at 0.0.4 or earlier, steps
    through :data:`MIGRATIONS` to the current version first.

    Args:
        source: A ``.json`` path, a parsed mapping (not mutated), a
            :class:`~flexcore.config.spec.FlowsheetSpec` (returned as is), or a
            nested :class:`~flexcore.config.schema.ModelConfig`.

    Returns:
        The validated flat spec, with its base directory set for relative paths.

    Raises:
        FlexConfigError: If the document is unreadable, has a missing,
            malformed or too-new ``schema_version``, lacks a migration, or fails
            validation (the message names the bad field path).
    """
    if isinstance(source, FlowsheetSpec):
        return source
    if isinstance(source, ModelConfig):
        data, name = source.model_dump(mode="json"), "the ModelConfig"
        base_dir = source._base_dir
    else:
        data, name, base_dir = _source_document(source)
    data = _upgrade(data, name, SCHEMA_VERSION, base_dir)
    try:
        spec = FlowsheetSpec.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc
    spec._base_dir = None if base_dir is None else Path(base_dir)
    return spec


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
    directory, filename: str = _SCHEMA_FILENAME, model: type[BaseModel] = ModelConfig
) -> None:
    """Write the exported JSON Schema for a config model to ``directory``.

    Serializes ``model``'s JSON Schema with
    ``indent=2`` and ``sort_keys=True`` so the checked-in schema diffs only on
    real schema changes (pitfall 7). Descriptions are collapsed to single-line
    plain text — line wrapping is the documentation builder's job, not the
    schema's. Run once and commit the result to
    ``src/flexcore/config/schemas/``.

    Args:
        directory: Destination directory for the schema file.
        filename: Output filename; override it to keep schemas for several
            versions side by side in one directory.
        model: The config model to export; the nested ``ModelConfig`` by
            default, or :class:`~flexcore.config.spec.FlowsheetSpec`.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    schema = model.model_json_schema()
    _plain_descriptions(schema)
    text = json.dumps(schema, indent=2, sort_keys=True)
    (directory / filename).write_text(text)
