"""The standard layout of a config folder: the spec and its data subfolders.

A config folder holds the spec as ``config.json`` and, beside it, one subfolder
per kind of data a ``$source`` can point to::

    config/
      config.json   the flat spec, fix and unfix values included
      price/        energy and export price series, tariff files, DR events
      opsdata/      historian series and tag maps
      sweep/        parameter sweep results and surrogate fit data

The layout is a convention. A ``$source`` path is still resolved relative to the
spec file, so a spec keeps loading wherever its files sit; ``emit_model`` and
:func:`~flexcore.config.io.dump_spec` are what write files into this layout.

A module that brings a new kind of data registers its own subfolder with
:func:`register_data_category`, instead of adding to the core list.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from flexcore.config.spec import KINDS
from flexcore.exceptions import FlexConfigError

SPEC_FILE = "config.json"
"""str: name of the spec file inside a config folder."""

DATA_CATEGORIES: dict[str, str] = {
    "price": "Energy and export price series, tariff files and demand-response "
    "events.",
    "opsdata": "Measured plant data: historian series and tag maps.",
    "sweep": "Parameter sweep results and the data surrogates are fitted from.",
}
"""Subfolder name -> what it holds."""

EMIT_CATEGORIES: dict[str, str] = {
    "costing": "price",
    "surrogate": "sweep",
}
"""Element kind -> subfolder ``emit_model`` writes that kind's long lists to.

A kind left out, such as ``fix``, ``unfix`` or ``dispatch``, keeps its values in
``config.json``.
"""

_FOLDER_NAME = re.compile(r"[a-z][a-z0-9_]*")


def register_data_category(
    name: str, description: str, kinds: tuple[str, ...] = ()
) -> None:
    """Add a data subfolder to the config layout.

    Registering the same name again with the same description is allowed, so a
    module can register at import time without guarding against re-imports.

    Args:
        name: The subfolder name: lowercase letters, digits and ``_``,
            starting with a letter.
        description: One line on what the subfolder holds.
        kinds: Element kinds whose long lists ``emit_model`` should write to
            this subfolder.

    Raises:
        FlexConfigError: If the name is not a valid folder name or is taken
            with another description, or a kind is unknown or already writes to
            another subfolder.
    """
    if not _FOLDER_NAME.fullmatch(name):
        raise FlexConfigError(
            f"Data category {name!r} is not a valid folder name. Use lowercase "
            "letters, digits and '_', starting with a letter.",
            value=name,
        )
    if DATA_CATEGORIES.get(name, description) != description:
        raise FlexConfigError(
            f"Data category {name!r} is already registered for "
            f"{DATA_CATEGORIES[name]!r}.",
            value=name,
        )
    for kind in kinds:
        if kind not in KINDS:
            raise FlexConfigError(
                f"Data category {name!r} names element kind {kind!r}, which is "
                "not in KINDS.",
                value=kind,
            )
        if EMIT_CATEGORIES.get(kind, name) != name:
            raise FlexConfigError(
                f"Element kind {kind!r} already writes its data to "
                f"{EMIT_CATEGORIES[kind]!r}.",
                value=kind,
            )
    DATA_CATEGORIES[name] = description
    for kind in kinds:
        EMIT_CATEGORIES[kind] = name


@dataclass(frozen=True)
class ConfigLayout:
    """Paths inside one config folder.

    Attributes:
        root: The config folder.
    """

    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    @property
    def spec_path(self) -> Path:
        """Path of the spec file, ``<root>/config.json``."""
        return self.root / SPEC_FILE

    def dir(self, category: str) -> Path:
        """Return the subfolder for a data category; it is not created.

        Args:
            category: A name in :data:`DATA_CATEGORIES`.

        Raises:
            FlexConfigError: For an unregistered category.
        """
        if category not in DATA_CATEGORIES:
            raise FlexConfigError(
                f"Unknown data category {category!r}. Registered categories: "
                f"{', '.join(sorted(DATA_CATEGORIES))}. A module adds its own "
                "with register_data_category.",
                value=category,
            )
        return self.root / category

    def emit_dir(self, kind: str) -> Path | None:
        """Return the subfolder ``emit_model`` writes ``kind``'s long lists to.

        Args:
            kind: An element kind.

        Returns:
            The subfolder, or None if the kind keeps its values in the spec.
        """
        category = EMIT_CATEGORIES.get(kind)
        return None if category is None else self.dir(category)


def spec_file(path) -> Path:
    """Return the spec file for ``path``: itself, or ``config.json`` in a folder.

    Args:
        path: A spec file path, or a config folder. A path with no suffix is
            taken to be a folder even if it does not exist yet.
    """
    path = Path(path)
    if path.is_dir() or not path.suffix:
        return ConfigLayout(path).spec_path
    return path
