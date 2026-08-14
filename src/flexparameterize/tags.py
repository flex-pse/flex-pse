"""Tag aliasing: the first stage of the FlexParameterize pipeline.

A :class:`TagMap` renames the columns of a tabular data set onto **model
aliases** — the dotted ``plant.unit.variable`` names by which the rest of
FlexParameterize refers to a built model's registered variables
(``plan/01_architecture.md`` §5).

FlexParameterize is data-source agnostic. The keys of a ``TagMap`` are whatever
the source called its columns — a historian tag (``"PIT-101.PV"``), a
spreadsheet header (``"Flow (gpm)"``), a CSV column, a database field. Nothing
in this module, or anything downstream of it, assumes or requires a historian
connection: once the columns are mapped, a hand-edited spreadsheet works
exactly like a historian export.
"""

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from flexcore.exceptions import FlexConfigError

_SUGGESTIONS_PER_COLUMN = 3


def model_alias(var) -> str:
    """Return the model alias of a registered Pyomo variable.

    The alias is the variable's fully qualified name on its model — e.g.
    ``"facility.plant.power_electrical"`` for a unit ``plant`` in a plant block
    ``facility``, and ``"facility.plant.inlet_state.flow_vol_phase"`` for a
    state-block variable. This is the dotted ``plant.unit.variable`` form a
    ``TagMap``'s values take, and it is unambiguous: an inlet and an outlet
    state variable share a local name but never an alias.

    Args:
        var: A Pyomo component attached to a model.

    Returns:
        The dotted alias.
    """
    return var.name


@dataclass
class TagReport:
    """Which source columns no ``TagMap`` entry claimed, and near misses for them.

    Attributes:
        unmapped_columns: Data columns that are neither a mapped tag nor an
            alias the map produces, in the order they appear in the frame.
        suggestions: Per unmapped column, the closest mapped tags
            (:func:`difflib.get_close_matches`, standard library only).
    """

    unmapped_columns: list[str] = field(default_factory=list)
    suggestions: dict[str, list[str]] = field(default_factory=dict)

    def __str__(self) -> str:
        """Render the unmapped columns and their suggestions, one per line."""
        if not self.unmapped_columns:
            return "All data columns are mapped."
        lines = [f"{len(self.unmapped_columns)} unmapped data column(s):"]
        for column in self.unmapped_columns:
            close = self.suggestions.get(column) or []
            hint = (
                f" — did you mean {', '.join(repr(c) for c in close)}?" if close else ""
            )
            lines.append(f"  {column!r}{hint}")
        lines.append("Add each column to the TagMap, or ignore it if it is not needed.")
        return "\n".join(lines)


class TagMap:
    """Maps a tabular source's column names onto dotted model aliases.

    Args:
        mapping: Source column name -> model alias (a dotted
            ``plant.unit.variable`` string, see :func:`model_alias`).

    Raises:
        FlexConfigError: If two tags map to the same alias — the resulting
            rename would silently drop one of the columns.

    Example:
        >>> import pandas as pd
        >>> tagmap = TagMap({"FT_0231.PV": "facility.pump.flow_in"})
        >>> tagmap.apply(pd.DataFrame({"FT_0231.PV": [1.0]})).columns.tolist()
        ['facility.pump.flow_in']
    """

    def __init__(self, mapping: dict[str, str]):
        duplicates = {
            alias
            for alias in mapping.values()
            if list(mapping.values()).count(alias) > 1
        }
        if duplicates:
            raise FlexConfigError(
                f"Several tags map to the same alias(es) {sorted(duplicates)}; "
                "each model alias may be fed by exactly one source column. "
                "Drop the duplicate entries.",
                field="mapping",
                value=sorted(duplicates),
            )
        self.mapping: dict[str, str] = dict(mapping)

    @classmethod
    def from_file(cls, path) -> "TagMap":
        """Load a tag map from a JSON file (the config format, architecture §2.3).

        Args:
            path: Path to a ``.json`` file holding a flat object of
                ``{"tag": "plant.unit.variable"}`` pairs.

        Returns:
            The loaded :class:`TagMap`.

        Raises:
            FlexConfigError: For a non-``.json`` suffix, a non-object document,
                or an entry whose value is not a string (the message names the
                offending key).
        """
        path = Path(path)
        if path.suffix.lower() != ".json":
            raise FlexConfigError(
                f"Unsupported tag-map format {path.suffix!r} for {path}. Use a "
                ".json file.",
                value=str(path),
            )
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise FlexConfigError(
                f"Tag-map file {path} must contain a flat object of "
                f"'tag': 'plant.unit.variable' pairs, got "
                f"{type(data).__name__}.",
                value=str(path),
            )
        for key, value in data.items():
            if not isinstance(value, str):
                raise FlexConfigError(
                    f"Tag-map entry {key!r} in {path} must be a dotted alias "
                    f"string like 'facility.pump.flow_in', got "
                    f"{type(value).__name__}.",
                    field=key,
                    value=value,
                )
        return cls(data)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``df`` with every mapped column renamed to its alias.

        Columns the map does not name are kept untouched — extra columns are
        legal, :func:`~flexparameterize.validate.check_sufficiency` decides what
        matters. The input frame is never mutated.

        Args:
            df: The source data frame.

        Returns:
            A renamed copy.
        """
        return df.rename(columns=self.mapping)

    def report_unmapped(self, df: pd.DataFrame) -> TagReport:
        """Report the columns of ``df`` this map does not account for.

        A column counts as accounted for if it is a mapped tag or is already
        one of the aliases the map produces (so a report on an
        already-:meth:`apply`-ed frame is empty). Suggestions come from
        :func:`difflib.get_close_matches` against the mapped tags.

        Args:
            df: The source data frame.

        Returns:
            The :class:`TagReport`.
        """
        known = set(self.mapping) | set(self.mapping.values())
        report = TagReport()
        for column in df.columns:
            if column in known:
                continue
            report.unmapped_columns.append(column)
            report.suggestions[column] = difflib.get_close_matches(
                column, list(self.mapping), n=_SUGGESTIONS_PER_COLUMN
            )
        return report
