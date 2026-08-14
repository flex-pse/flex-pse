"""Sufficiency validation: does the data determine the regression problem?

The second stage of the FlexParameterize pipeline. Given a built FlexOps model
(or a single :class:`~flexops.core.registration.IORegistry`) and an aliased
data frame, :func:`check_sufficiency` reports — per registered IO pair —
whether the data covers it, holds enough non-null rows, and is indexed on a
time axis the model recognizes.

This is what makes a fit a **zero-degree-of-freedom regression** (project
term): the registered IO pairs exactly determine the regression problem — no
unmapped inputs, no free parameters after fit.

The check reports; it never raises. Callers branch on
:attr:`SufficiencyReport.sufficient` and decide for themselves — as
:func:`~flexparameterize.apply.apply_to_model` does, because it mutates a real
model. Like the rest of FlexParameterize this module is data-source agnostic:
it sees a ``pandas`` frame and never a historian.
"""

from dataclasses import dataclass, field

import pandas as pd

from flexops.core.registration import IORegistry, iter_io_registry
from flexparameterize.tags import model_alias

DEFAULT_MIN_ROWS = 10
"""int: default minimum number of non-null rows a column must carry."""


@dataclass
class IOPairStatus:
    """Data coverage of one registered ``(input, output)`` variable pair.

    Attributes:
        unit: Name of the unit the pair belongs to (empty when
            :func:`check_sufficiency` was handed a bare registry, which has no
            owning block).
        variables: The pair's ``(input alias, output alias)``.
        present: Aliases of the pair found as data columns.
        missing: Aliases of the pair with no data column.
        non_null_counts: Non-null row count of each present column.
        min_rows: Minimum non-null rows each present column must carry.
    """

    unit: str
    variables: tuple[str, str]
    present: list[str]
    missing: list[str]
    non_null_counts: dict[str, int]
    min_rows: int

    @property
    def sufficient(self) -> bool:
        """bool: True if nothing is missing and every column has enough rows."""
        return not self.missing and all(
            count >= self.min_rows for count in self.non_null_counts.values()
        )

    def __str__(self) -> str:
        """Render this pair's status as one actionable line."""
        if self.sufficient:
            return f"  OK       {self.unit or '-'}: {' -> '.join(self.variables)}"
        problems = [f"missing column {alias!r}" for alias in self.missing]
        problems += [
            f"{alias!r} has {count} non-null row(s), needs {self.min_rows}"
            for alias, count in self.non_null_counts.items()
            if count < self.min_rows
        ]
        return (
            f"  MISSING  {self.unit or '-'}: {' -> '.join(self.variables)}\n"
            + "\n".join(f"             {problem}" for problem in problems)
        )


@dataclass
class SufficiencyReport:
    """Whether a data frame determines every registered regression problem.

    Attributes:
        pairs: One :class:`IOPairStatus` per registered IO pair.
        index_ok: Whether the frame's index is a usable time axis.
        index_message: What is right or wrong with the index, and what to do.
    """

    pairs: list[IOPairStatus] = field(default_factory=list)
    index_ok: bool = True
    index_message: str = ""

    @property
    def sufficient(self) -> bool:
        """bool: True if the index is usable and every IO pair is covered."""
        return self.index_ok and all(pair.sufficient for pair in self.pairs)

    def __str__(self) -> str:
        """Render the index verdict and every pair's status as a table."""
        header = "Data is sufficient." if self.sufficient else "Data is insufficient."
        lines = [header, f"  index: {self.index_message}"]
        lines += [str(pair) for pair in self.pairs]
        return "\n".join(lines)


def _iter_registries(registry):
    """Yield ``(unit name, registry)`` for a model or a single registry.

    Args:
        registry: An :class:`~flexops.core.registration.IORegistry`, or a built
            model to walk with
            :func:`~flexops.core.registration.iter_io_registry`.

    Yields:
        ``(unit name, IORegistry)`` pairs; the name is empty for a bare
        registry, which does not know its owning block.
    """
    if isinstance(registry, IORegistry):
        yield "", registry
    else:
        for block, block_registry in iter_io_registry(registry):
            yield block.name, block_registry


def _check_index(data: pd.DataFrame, time_block) -> tuple[bool, str]:
    """Check that ``data`` is indexed on a usable time axis.

    v0 requires a ``pandas.DatetimeIndex`` (a plain timestamp *column* is not
    enough — call ``set_index`` first). When ``time_block`` is given the index
    must additionally be monotonic and lie within the block's horizon; an exact
    match of the model's time grid is **not** required, so a finer or coarser
    sampling still passes.

    Args:
        data: The aliased data frame.
        time_block: The model's ``TimeBlockData``, or None to skip the
            horizon comparison.

    Returns:
        ``(ok, message)``.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        return False, (
            f"data.index is a {type(data.index).__name__}, not a pandas "
            "DatetimeIndex. Index the frame by its timestamp column first, e.g. "
            "df = df.set_index('timestamp')."
        )
    if data.empty:
        return False, "data has no rows; supply at least one timestamped sample."
    if not data.index.is_monotonic_increasing:
        return False, "data.index is not sorted; call df.sort_index() first."
    if time_block is None:
        return True, "DatetimeIndex, sorted (no time_block given to compare against)."
    horizon = time_block.datetime_index
    if data.index.min() < horizon[0] or data.index.max() > horizon[-1]:
        return False, (
            f"data spans {data.index.min()} to {data.index.max()}, outside the "
            f"model horizon {horizon[0]} to {horizon[-1]}. Slice the data to the "
            "horizon, or build the TimeBlock over the data's span."
        )
    return True, f"DatetimeIndex within the model horizon ({len(data)} rows)."


def check_sufficiency(
    registry, data: pd.DataFrame, time_block=None, *, min_rows: int = DEFAULT_MIN_ROWS
) -> SufficiencyReport:
    """Report whether ``data`` determines every registered regression problem.

    Walks each registered IO pair — the cartesian product of a unit's
    registered input and output variables, so a unit with several valid pairs
    gets each checked — and records which of the pair's aliases are present as
    data columns and how many non-null rows they carry. Together with the index
    check this is the **zero-degree-of-freedom regression** condition: the
    registered IO pairs exactly determine the regression problem — no unmapped
    inputs, no free parameters after fit.

    Insufficient data is reported, never raised: the caller decides (see
    :func:`~flexparameterize.apply.apply_to_model`, which does raise).

    Args:
        registry: A built model to walk, or a single
            :class:`~flexops.core.registration.IORegistry`.
        data: The data frame, with columns already renamed to model aliases by
            :meth:`~flexparameterize.tags.TagMap.apply`.
        time_block: The model's ``TimeBlockData``; when given, the data index
            is compared against its horizon.
        min_rows: Minimum non-null rows a present column must carry.

    Returns:
        The :class:`SufficiencyReport`.
    """
    index_ok, index_message = _check_index(data, time_block)
    report = SufficiencyReport(index_ok=index_ok, index_message=index_message)
    columns = set(data.columns)
    for unit_name, unit_registry in _iter_registries(registry):
        inputs = [r for r in unit_registry.io_variables if r.role == "input"]
        outputs = [r for r in unit_registry.io_variables if r.role == "output"]
        for input_record, output_record in ((i, o) for i in inputs for o in outputs):
            aliases = (model_alias(input_record.var), model_alias(output_record.var))
            present = [alias for alias in aliases if alias in columns]
            report.pairs.append(
                IOPairStatus(
                    unit=unit_name,
                    variables=aliases,
                    present=present,
                    missing=[alias for alias in aliases if alias not in columns],
                    non_null_counts={
                        alias: int(data[alias].count()) for alias in present
                    },
                    min_rows=min_rows,
                )
            )
    return report
