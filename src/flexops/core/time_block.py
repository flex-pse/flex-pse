"""Discrete-time substrate for flex-pse models.

``TimeBlock`` is the ordered integer time set every other flex-pse block
indexes against. Flex-pse uses discrete-time to form MIP unit-commitment logic
(dwell times, startup delays, rolling-horizon windows) which needs integer index
arithmetic. Every dynamic relationship (tank holdup, battery state of charge)
is a hand-written difference equation against ``time_block.dt``.
See ``docs/explanation/time_and_dynamics.md`` for the full rationale."""

import datetime
from dataclasses import dataclass

import pandas as pd
import pyomo.environ as pyo
from dateutil.relativedelta import relativedelta
from idaes.core import ProcessBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.core.base.units_container import InconsistentUnitsError
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError

_ISO_EXAMPLE = "2025-01-01"


def _parse_date(value: str | datetime.date) -> datetime.datetime:
    """Normalize a config date value to a naive ``datetime.datetime``.

    Args:
        value: An ISO-8601 date/datetime string, or a ``datetime.datetime``/
            ``datetime.date`` object.

    Returns:
        The corresponding naive ``datetime.datetime``.

    Raises:
        FlexConfigError: If ``value`` cannot be parsed, or is timezone-aware
            (flex-pse v0 uses naive local time throughout).
    """
    if isinstance(value, datetime.datetime):
        parsed = value
    elif isinstance(value, datetime.date):
        parsed = datetime.datetime(value.year, value.month, value.day)
    elif isinstance(value, str):
        try:
            parsed = datetime.datetime.fromisoformat(value)
        except ValueError as exc:
            raise FlexConfigError(
                f"Could not parse {value!r} as an ISO-8601 date/datetime "
                f"(example: {_ISO_EXAMPLE!r}).",
                value=value,
            ) from exc
    else:
        raise FlexConfigError(
            f"Expected an ISO-8601 string or a datetime/date object, got "
            f"{type(value).__name__}: {value!r}.",
            value=value,
        )
    if parsed.tzinfo is not None:
        raise FlexConfigError(
            "Timezone-aware dates are not supported; flex-pse v0 uses naive "
            f"local time throughout (got {value!r}). Strip the timezone "
            "before passing it in.",
            value=value,
        )
    return parsed


def _step_seconds(time_step) -> float:
    """Convert a ``pyunits``-carrying time-step expression to seconds.

    Args:
        time_step: A unit-carrying duration such as ``15 * pyunits.min``.

    Returns:
        The step length in seconds.

    Raises:
        FlexConfigError: If ``time_step`` is a bare number or does not carry
            units of time.
    """
    if isinstance(time_step, int | float):
        raise FlexConfigError(
            f"time_step={time_step!r} has no units. Multiply it by a "
            "pyomo.environ.units time unit, e.g. time_step=15 * pyunits.min.",
            field="time_step",
            value=time_step,
        )
    try:
        seconds = pyo.value(pyunits.convert(time_step, pyunits.s))
    except InconsistentUnitsError as exc:
        raise FlexConfigError(
            f"time_step={time_step!r} does not carry units of time. "
            "Multiply it by a pyomo.environ.units time unit, e.g. "
            "time_step=15 * pyunits.min.",
            field="time_step",
            value=time_step,
        ) from exc
    if seconds <= 0:
        raise FlexConfigError(
            f"time_step must be a positive duration, got {seconds} s.",
            field="time_step",
            value=time_step,
        )
    return seconds


def _describe_max_length(max_length: relativedelta) -> str:
    """Name the max-length limit for use in error messages.

    Args:
        max_length: The configured maximum horizon length.

    Returns:
        A hyphenated adjective phrase (e.g. ``"one-calendar-month"``) naming the
        limit; a generic label for any non-default value.
    """
    if max_length == relativedelta(months=1):
        return "one-calendar-month"
    return "configured maximum-length"


@dataclass(frozen=True)
class TimeWindow:
    """Metadata for a slice of a ``TimeBlock``'s horizon.

    Pure metadata (no Pyomo components) describing a contiguous run of time
    points, used by the rolling-horizon driver to step through a
    ``TimeBlock`` window by window.

    Attributes:
        start_index: First time-point index in the window.
        indices: The window's time-point indices, clipped to the block's
            horizon (shorter than requested if the window runs past the end).
        start_time: Timestamp of ``start_index``.
        end_time: Exclusive end timestamp of the window.
    """

    start_index: int
    indices: range
    start_time: pd.Timestamp
    end_time: pd.Timestamp


@declare_process_block_class("TimeBlock")
class TimeBlockData(ProcessBlockData):
    """A discrete, bounded time horizon at a configurable resolution.

    Holds the ordered integer time set (`time_index`), the elapsed-time Param
    (`time`, ``i*dt``), the unit-carrying step size (`dt`), datetime↔index
    utilities, and the rolling-horizon hooks
    (`register_initial_state`, `window`) that ``flexschedule`` drives. Time
    points are interval starts: point ``i`` is the timestamp
    ``start_date + i * dt``; ``end_date`` is the exclusive horizon end and is
    not itself a time point.

    A single ``TimeBlock`` spans at most ``max_length`` (default one calendar
    month; see decision R2); longer studies are composed from multiple
    ``TimeBlock`` blocks by the rolling-horizon driver or the design-mode
    wrapper.

    Example:
        >>> from pyomo.environ import units as pyunits
        >>> m = pyo.ConcreteModel()
        >>> m.time_block = TimeBlock(
        ...     start_date="2025-01-01", end_date="2025-01-31",
        ...     time_step=15 * pyunits.min,
        ... )
    """

    CONFIG = ProcessBlockData.CONFIG()
    CONFIG.declare(
        "start_date",
        ConfigValue(
            description="ISO-8601 string or datetime/date: inclusive start of "
            "the horizon."
        ),
    )
    CONFIG.declare(
        "end_date",
        ConfigValue(
            description="ISO-8601 string or datetime/date: exclusive end of "
            "the horizon. Must be within one calendar month of start_date."
        ),
    )
    CONFIG.declare(
        "time_step",
        ConfigValue(
            default=15 * pyunits.min,
            description="Time-step length as a pyunits-carrying expression, "
            "e.g. 15 * pyunits.min. Any positive duration is accepted.",
        ),
    )
    CONFIG.declare(
        "max_length",
        ConfigValue(
            default=relativedelta(months=1),
            description="Maximum horizon length as a dateutil.relativedelta, "
            "measured with calendar arithmetic from start_date. Defaults to one "
            "calendar month; widen it only if the rolling-horizon driver "
            "(flexschedule) or design-mode wrapper (flexops.design) cannot.",
        ),
    )

    def build(self) -> None:
        """Construct `time_index`, `time`, `dt`, and the datetime/state registries."""
        super().build()

        start = _parse_date(self.config.start_date)
        end = _parse_date(self.config.end_date)
        step_seconds = _step_seconds(self.config.time_step)

        if end <= start:
            raise FlexConfigError(
                f"end_date ({end.isoformat()}) must be after start_date "
                f"({start.isoformat()}).",
                field="end_date",
                value=self.config.end_date,
            )

        max_length = self.config.max_length
        max_end = start + max_length
        if end > max_end:
            raise FlexConfigError(
                f"Horizon {start.isoformat()} -> {end.isoformat()} exceeds "
                f"the {_describe_max_length(max_length)} limit (max end_date is "
                f"{max_end.isoformat()}). Use the rolling-horizon driver "
                "(flexschedule) or the design-mode wrapper (flexops.design) "
                "for longer studies.",
                field="end_date",
                value=self.config.end_date,
            )

        span = end - start
        step = datetime.timedelta(seconds=step_seconds)
        n, remainder = divmod(span, step)
        if remainder != datetime.timedelta(0):
            lower_end = start + n * step
            upper_end = start + (n + 1) * step
            raise FlexConfigError(
                f"Horizon span {span} is not an integer multiple of "
                f"time_step ({step}). The nearest valid end_dates are "
                f"{lower_end.isoformat()} and {upper_end.isoformat()}.",
                field="end_date",
                value=self.config.end_date,
            )
        if n <= 0:
            raise FlexConfigError(
                f"Horizon {start.isoformat()} -> {end.isoformat()} contains "
                "no whole time steps.",
                field="end_date",
                value=self.config.end_date,
            )

        self.time_index = pyo.Set(
            initialize=range(n),
            ordered=True,
            doc="Ordered integer time-point indices 0..N-1.",
        )
        step_value = pyo.value(self.config.time_step)
        step_units = pyunits.get_units(self.config.time_step)
        self.time = pyo.Param(
            self.time_index,
            initialize={i: i * step_value for i in range(n)},
            units=step_units,
            doc="Elapsed time at each point, i*dt, in the user's units.",
        )
        # dt keeps the user's units (e.g. 15 min), per the milestone spec;
        # step_seconds is the once-converted value used for date arithmetic.
        # Deviation from the spec's mutable=False: this pyomo version requires
        # unit-carrying Params to be mutable (it warns and silently upgrades
        # them otherwise). dt is still never mutated by this module; only the
        # rolling-horizon driver mutates state Params registered via
        # register_initial_state.
        self.dt = pyo.Param(
            initialize=pyo.value(self.config.time_step),
            units=pyunits.get_units(self.config.time_step),
            mutable=True,
            doc="Time-step length, in the units the user supplied.",
        )
        self._step_seconds = step_seconds
        self._datetime_index = pd.date_range(
            start, periods=n, freq=pd.Timedelta(seconds=step_seconds)
        )
        self._initial_state_params = []

    @property
    def n_points(self) -> int:
        """int: Number of time points, ``N``."""
        return len(self.time_index)

    @property
    def horizon(self):
        """Total horizon length as a unit-carrying expression, ``n_points * dt``."""
        return self.n_points * self.dt

    @property
    def datetime_index(self) -> pd.DatetimeIndex:
        """pandas.DatetimeIndex: timestamps of every time point."""
        return self._datetime_index

    @property
    def initial_state_params(self) -> tuple:
        """tuple: the registered rolling-horizon initial-state Params."""
        return tuple(self._initial_state_params)

    def index_of(self, timestamp) -> int:
        """Return the time-point index for a timestamp.

        Args:
            timestamp: A ``pd.Timestamp``, ``datetime``, or ISO-8601 string
                that must fall exactly on a grid point.

        Returns:
            The integer time-point index.

        Raises:
            FlexConfigError: If ``timestamp`` is off-grid or outside the
                horizon.
        """
        ts = pd.Timestamp(timestamp)
        try:
            return self._datetime_index.get_loc(ts)
        except KeyError as exc:
            raise FlexConfigError(
                f"Timestamp {ts} is not a time point of this TimeBlock "
                f"(step={self._step_seconds} s, horizon="
                f"[{self._datetime_index[0]}, {self._datetime_index[-1]}]).",
                value=timestamp,
            ) from exc

    def timestamp_of(self, i: int) -> pd.Timestamp:
        """Return the timestamp for a time-point index.

        Args:
            i: A time-point index.

        Returns:
            The corresponding ``pd.Timestamp``.

        Raises:
            FlexConfigError: If ``i`` is outside ``[0, n_points)``.
        """
        if not 0 <= i < self.n_points:
            raise FlexConfigError(
                f"Time-point index {i} is out of range [0, {self.n_points}).",
                value=i,
            )
        return self._datetime_index[i]

    def register_initial_state(self, param) -> None:
        """Register a mutable Param as rolling-horizon initial state.

        Args:
            param: A mutable Pyomo ``Param`` (e.g. tank level, battery SOC,
                on/off status) that the rolling-horizon driver mutates
                between windows.

        Raises:
            FlexConfigError: If ``param`` is not a Pyomo ``Param``, or was
                declared with ``mutable=False``.
        """
        if not isinstance(param, pyo.Param):
            raise FlexConfigError(
                f"register_initial_state expects a Pyomo Param, got "
                f"{type(param).__name__}.",
                value=param,
            )
        if not param.mutable:
            raise FlexConfigError(
                "Initial-state Params must be declared with mutable=True.",
                value=param,
            )
        self._initial_state_params.append(param)

    def window(self, start: int, length: int) -> TimeWindow:
        """Return metadata for a contiguous slice of the horizon.

        Args:
            start: First time-point index of the window.
            length: Requested number of time points; clipped if the window
                would run past the horizon.

        Returns:
            The window metadata.

        Raises:
            FlexConfigError: If ``start`` is outside ``[0, n_points)``.
        """
        n = self.n_points
        if not 0 <= start < n:
            raise FlexConfigError(
                f"window start index {start} is out of range [0, {n}).",
                value=start,
            )
        stop = min(start + length, n)
        indices = range(start, stop)
        start_time = self.timestamp_of(start)
        end_time = start_time + len(indices) * datetime.timedelta(
            seconds=self._step_seconds
        )
        return TimeWindow(
            start_index=start,
            indices=indices,
            start_time=start_time,
            end_time=end_time,
        )
