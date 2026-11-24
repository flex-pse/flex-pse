"""Demand-response revenue: the bridge to EECO's demand-response module.

See :doc:`/explanation/demand_response` for how DR revenue enters the cost.
"""

import contextlib
import math

import pandas as pd

# Soft import, as in opex.py: this module imports without EECO's demand-response
# module (eeco releases up to 0.4.2 have none), and the first use raises a
# FlexConfigError. The ImportError is kept, not dropped, so a failure inside
# eeco.demandresponse (not only its absence) reaches the user with its cause.
try:
    import eeco.demandresponse as _eeco_dr
except ImportError as exc:  # pragma: no cover - depends on the installed eeco
    _eeco_dr = None
    _EECO_IMPORT_ERROR = exc
else:
    _EECO_IMPORT_ERROR = None

from flexcore.config.schema import (
    DRBaseline,
    DRBaselineMethod,
    DRConfig,
    DRPayment,
    DRPaymentKind,
    HistoricalPower,
)
from flexcore.exceptions import FlexConfigError


def _require_eeco_dr() -> None:
    """Fail if EECO's demand-response module could not be imported.

    Raises:
        FlexConfigError: If ``eeco.demandresponse`` is missing (an eeco release
            older than 0.5.0) or exists but failed to import. The import error is
            chained as the cause.
    """
    if _eeco_dr is not None:
        return
    error = _EECO_IMPORT_ERROR
    is_missing = isinstance(error, ModuleNotFoundError) and error.name in (
        "eeco",
        "eeco.demandresponse",
    )
    if is_missing:
        message = (
            "Demand response needs eeco 0.5.0 or newer, which ships the "
            "demandresponse module; install it or omit demand_response."
        )
    else:
        message = (
            f"eeco.demandresponse failed to import ({error}); fix that import "
            "error or omit demand_response."
        )
    raise FlexConfigError(message, field="demand_response") from error


@contextlib.contextmanager
def _eeco_rejection(what: str):
    """Re-raise a ValueError from EECO as a FlexConfigError naming what it rejected.

    EECO checks its own inputs, and those checks stay the single source of truth:
    flex-pse does not copy them, it only reports them as configuration errors.

    Args:
        what: What was being built, for the message (for example ``"baseline"``).

    Raises:
        FlexConfigError: If the wrapped EECO call raises a ``ValueError``.
    """
    try:
        yield
    except ValueError as exc:
        raise FlexConfigError(
            f"EECO rejected the demand-response {what}: {exc}",
            field="demand_response",
        ) from exc


def dr_baseline_method(spec: DRBaseline):
    """Build the EECO baseline method a DR baseline spec describes.

    Args:
        spec: The baseline spec.

    Returns:
        An ``eeco.demandresponse.BaselineMethod`` instance.

    Raises:
        FlexConfigError: If eeco has no demand-response module, or EECO rejects
            the baseline options.
    """
    _require_eeco_dr()
    with _eeco_rejection("baseline"):
        if spec.method is DRBaselineMethod.FIXED_LEVEL:
            return _eeco_dr.FixedLevelBaseline(spec.firm_level_kW, spec.resolution)
        if spec.method is DRBaselineMethod.UNILATERAL_INTERRUPTION:
            return _eeco_dr.UnilateralInterruptionBaseline(
                spec.interruption_level_kW, spec.resolution
            )
        similar_days = (
            _eeco_dr.TopUsageDaysBaseline
            if spec.method is DRBaselineMethod.TOP_USAGE_DAYS
            else _eeco_dr.SimilarDaysBaseline
        )
        return similar_days(
            n_baseline_days=spec.n_baseline_days,
            adjustment_offset_hours=spec.adjustment_offset_hours,
            adjustment_duration_hours=spec.adjustment_duration_hours,
            adjustment_clip=spec.adjustment_clip,
            exclude_weekends=spec.exclude_weekends,
            exclude_holidays=spec.exclude_holidays,
            holiday_country=spec.holiday_country,
            holiday_subdiv=spec.holiday_subdiv,
            holiday_dates=spec.holiday_dates or None,
            resolution=spec.resolution,
        )


def dr_payment_structure(spec: DRPayment):
    """Build the EECO payment structure a DR payment spec describes.

    Args:
        spec: The payment spec; an open region bound becomes infinite.

    Returns:
        An ``eeco.demandresponse.PaymentStructure`` instance.

    Raises:
        FlexConfigError: If eeco has no demand-response module.
    """
    _require_eeco_dr()
    regions = [
        {
            "x1": -math.inf if region.x1 is None else region.x1,
            "x2": math.inf if region.x2 is None else region.x2,
            "y1": region.y1,
            "y2": region.y2,
        }
        for region in spec.regions or []
    ]
    options = {
        "settlement": spec.settlement,
        "payment_basis": spec.payment_basis,
        "payout": spec.payout,
        "payout_basis": spec.payout_basis,
    }
    if spec.kind is DRPaymentKind.CAPACITY_ENERGY:
        return _eeco_dr.CapacityEnergyPayment(regions, spec.energy_price, **options)
    return _eeco_dr.PaymentStructure(regions, **options)


def historical_power_series(spec: HistoricalPower) -> pd.Series:
    """Return meter history as a kW series on a naive datetime index.

    Args:
        spec: The meter history.

    Returns:
        A float ``pd.Series`` of kW indexed from ``spec.start``.
    """
    index = pd.date_range(
        spec.start,
        periods=len(spec.values_kW),
        freq=pd.Timedelta(minutes=spec.time_step_minutes),
    )
    return pd.Series(spec.values_kW, index=index, dtype=float)


def check_dr_resolution(config: DRConfig, dt_minutes: float) -> None:
    """Check that the meter history and model steps fit EECO's settlement interval.

    EECO settles at ``baseline.resolution`` when it is set, and otherwise at the
    model's time step. It averages power over each settlement interval, so a step
    that does not fit the interval evenly leaves intervals without data, and a
    history at a different step than the model would be settled differently ex
    post than it was optimized.

    Args:
        config: The demand-response program.
        dt_minutes: The model's time step in minutes.

    Raises:
        FlexConfigError: If ``baseline.resolution`` is not a time interval, or is
            unset and the history step differs from the model step, or is set and
            the history step or the model step does not divide it evenly.
    """
    history_minutes = config.historical_power.time_step_minutes
    resolution = config.baseline.resolution
    if resolution is None:
        if not math.isclose(history_minutes, dt_minutes):
            raise FlexConfigError(
                f"historical_power.time_step_minutes is {history_minutes:g}, but "
                f"the model time step is {dt_minutes:g} minutes. With no "
                "baseline.resolution, EECO settles at the model time step, so the "
                "two must match. Resample the meter history to the model step, or "
                "set baseline.resolution so that both steps divide it.",
                field="demand_response.historical_power.time_step_minutes",
                value=history_minutes,
            )
        return

    try:
        interval_minutes = pd.Timedelta(resolution).total_seconds() / 60
    except ValueError as exc:
        raise FlexConfigError(
            f"baseline.resolution {resolution!r} is not a time interval such as "
            "'15m' or '1h'.",
            field="demand_response.baseline.resolution",
            value=resolution,
        ) from exc
    for label, minutes, field in (
        (
            "historical_power.time_step_minutes",
            history_minutes,
            "demand_response.historical_power.time_step_minutes",
        ),
        ("the model time step", dt_minutes, "time_block"),
    ):
        ratio = interval_minutes / minutes
        if round(ratio) < 1 or not math.isclose(ratio, round(ratio)):
            raise FlexConfigError(
                f"{label} is {minutes:g} minutes, which does not divide the "
                f"settlement interval baseline.resolution={resolution!r} "
                f"({interval_minutes:g} minutes) evenly. Choose a baseline.resolution "
                "that is a whole multiple of it, or change the step.",
                field=field,
                value=minutes,
            )


def check_dr_history(
    config: DRConfig, datetime_index: pd.DatetimeIndex, dt_minutes: float
) -> None:
    """Check that every baseline day has data, from the history or the model.

    EECO reads a baseline day from the meter history, except when that day's
    event window lies wholly inside the model horizon, where it uses the model's
    own power. A day that is in neither has no data and fails late inside EECO, so
    this reports it at build time, naming the event and the day.

    Args:
        config: The demand-response program.
        datetime_index: Naive timestamps of the model horizon.
        dt_minutes: The model's time step in minutes.

    Raises:
        FlexConfigError: If an event's baseline day is covered by neither the
            meter history nor the model horizon.
    """
    history = config.historical_power
    history_start = pd.Timestamp(history.start)
    history_end = history_start + pd.Timedelta(
        minutes=history.time_step_minutes * len(history.values_kW)
    )
    horizon_start = datetime_index.min()
    horizon_end = datetime_index.max() + pd.Timedelta(minutes=dt_minutes)
    for event in config.events:
        for day in event.baseline_days:
            start = pd.Timestamp(day) + pd.Timedelta(hours=event.start_hour)
            end = start + pd.Timedelta(hours=event.duration_hours)
            in_history = history_start <= start and end <= history_end
            in_horizon = horizon_start <= start and end <= horizon_end
            if not (in_history or in_horizon):
                raise FlexConfigError(
                    f"baseline day {day.isoformat()} of the event on "
                    f"{event.event_date.isoformat()} needs data from {start} to "
                    f"{end}, but historical_power covers {history_start} to "
                    f"{history_end} and the model horizon is {horizon_start} to "
                    f"{horizon_end}. Extend historical_power or remove that "
                    "baseline day.",
                    field="demand_response.events.baseline_days",
                    value=day.isoformat(),
                )


def _eeco_events(config: DRConfig) -> list[dict]:
    """Return the config's events as EECO event dicts (validated by EECO).

    Raises:
        FlexConfigError: If EECO rejects an event.
    """
    events = None
    with _eeco_rejection("events"):
        for event in config.events:
            events = _eeco_dr.add_event(
                events, **event.model_dump(exclude={"region_x1"})
            )
    return events


def add_dr_revenue(block, power_kW, datetime_index: pd.DatetimeIndex, config: DRConfig):
    """Build DR revenue for every event on ``block`` and return its expression.

    Args:
        block: The Pyomo block EECO adds its baseline and payment components to.
        power_kW: Time-indexed site power Var as a bare kW number (no units).
        datetime_index: Naive timestamps aligned with ``power_kW``.
        config: The demand-response program.

    Returns:
        The total revenue expression, a bare USD number.

    Raises:
        FlexConfigError: If eeco has no demand-response module, or EECO rejects
            the program (for example, an event window outside the horizon).
    """
    _require_eeco_dr()
    region_x1s = {
        event.event_date.isoformat(): event.region_x1
        for event in config.events
        if event.region_x1 is not None
    }
    with _eeco_rejection("revenue build"):
        revenue, _ = _eeco_dr.calculate_dr_revenue(
            power_kW,
            _eeco_events(config),
            dr_baseline_method(config.baseline),
            dr_payment_structure(config.payment),
            historical_power_kW=historical_power_series(config.historical_power),
            datetime_index=datetime_index,
            model=block,
            region_x1s=region_x1s or None,
        )
    return revenue


def evaluate_dr_revenue(
    realized_kW: pd.Series, config: DRConfig
) -> tuple[pd.DataFrame, float]:
    """Settle DR revenue ex post on realized power plus the meter history.

    Args:
        realized_kW: Realized site power in kW on a naive datetime index;
            it overrides the history wherever both have a value.
        config: The demand-response program.

    Returns:
        EECO's per-event revenue table and the total revenue in USD.

    Raises:
        FlexConfigError: If eeco has no demand-response module, or EECO rejects
            the settlement (for example, a baseline day with no data).
    """
    _require_eeco_dr()
    power = realized_kW.combine_first(historical_power_series(config.historical_power))
    baseline = dr_baseline_method(config.baseline)
    with _eeco_rejection("settlement"):
        if config.baseline.method is DRBaselineMethod.UNILATERAL_INTERRUPTION:
            # EECO's interruption baseline exists only inside a model: the flat
            # interruption level plus a hard ``power <= level`` constraint in the
            # event. Realized power already satisfies that constraint, so a fixed
            # baseline at the same level settles to the revenue the optimizer saw
            # (test_in_model_and_ex_post_revenue_agree checks every baseline).
            baseline = _eeco_dr.FixedLevelBaseline(
                config.baseline.interruption_level_kW, config.baseline.resolution
            )
        per_event, total = _eeco_dr.calculate_itemized_dr_revenue(
            power,
            _eeco_events(config),
            baseline,
            dr_payment_structure(config.payment),
        )
    return per_event, float(total)
