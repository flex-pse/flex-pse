"""Resolve ``$source`` files, refusing series made for other time points."""

import datetime
import math

from flexcore.config.io import TimeCheck, resolve_sources
from flexcore.config.schema import TimeConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.time_block import _parse_date, _step_seconds, find_time_block
from flexops.core.units import parse_quantity


def _time_check(time_block) -> TimeCheck:
    """Return the check that a series was made for ``time_block``'s time points.

    Args:
        time_block: The model's TimeBlock.

    Returns:
        A function of a series' time and file path that raises
        :class:`~flexcore.exceptions.FlexConfigError` if they differ.
    """
    step = time_block._step_seconds
    start = time_block.datetime_index[0].to_pydatetime()
    end = start + datetime.timedelta(seconds=step * time_block.n_points)

    def check(time: TimeConfig, source: str) -> None:
        series_step = _step_seconds(parse_quantity(time.time_step))
        if (
            _parse_date(time.start_date) == start
            and _parse_date(time.end_date) == end
            and math.isclose(series_step, step)
        ):
            return
        raise FlexConfigError(
            f"The series in {source!r} runs from {time.start_date} to "
            f"{time.end_date} in steps of {time.time_step}, but the model runs "
            f"from {start.isoformat()} to {end.isoformat()} in steps of "
            f"{step:g} s. Regenerate the series for the model's time element.",
            field="$source",
            value=source,
        )

    return check


def resolve_model_sources(model, value, base_dir):
    """Replace ``$source`` references, checking dated series against the model.

    Args:
        model: The model the data is for. Without a TimeBlock nothing is checked.
        value: Any data :func:`~flexcore.config.io.resolve_sources` accepts.
        base_dir: Directory a relative path resolves against, or None.

    Returns:
        ``value`` with each reference replaced.

    Raises:
        FlexConfigError: If a file cannot be read, or a time series document was
            made for other time points than the model's.
    """
    try:
        check = _time_check(find_time_block(model))
    except FlexConfigError:
        check = None
    return resolve_sources(value, base_dir, check)
