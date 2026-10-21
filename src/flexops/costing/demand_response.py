"""Demand-response revenue: the bridge to EECO's demand-response module.

See :doc:`/explanation/demand_response` for how DR revenue enters the cost.
"""

import math

import pandas as pd

# Soft import: eeco releases up to 0.4.2 have no demandresponse module.
try:
    from eeco import demandresponse as _eeco_dr
except ImportError:  # pragma: no cover - depends on the installed eeco
    _eeco_dr = None

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
    """Fail if the installed eeco has no demand-response module.

    Raises:
        FlexConfigError: If ``eeco.demandresponse`` cannot be imported.
    """
    if _eeco_dr is None:
        raise FlexConfigError(
            "Demand response needs an eeco release with the demandresponse "
            "module (newer than 0.4.2); upgrade eeco or omit demand_response.",
            field="demand_response",
        )


def dr_baseline_method(spec: DRBaseline):
    """Build the EECO baseline method a DR baseline spec describes.

    Args:
        spec: The baseline spec.

    Returns:
        An ``eeco.demandresponse.BaselineMethod`` instance.

    Raises:
        FlexConfigError: If eeco has no demand-response module.
    """
    _require_eeco_dr()
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


def _eeco_events(config: DRConfig) -> list[dict]:
    """Return the config's events as EECO event dicts (validated by EECO)."""
    events = None
    for event in config.events:
        events = _eeco_dr.add_event(events, **event.model_dump(exclude={"region_x1"}))
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
        FlexConfigError: If eeco has no demand-response module.
    """
    _require_eeco_dr()
    region_x1s = {
        event.event_date.isoformat(): event.region_x1
        for event in config.events
        if event.region_x1 is not None
    }
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
        FlexConfigError: If eeco has no demand-response module.
    """
    _require_eeco_dr()
    power = realized_kW.combine_first(historical_power_series(config.historical_power))
    baseline = dr_baseline_method(config.baseline)
    if config.baseline.method is DRBaselineMethod.UNILATERAL_INTERRUPTION:
        # EECO has no ex-post interruption baseline; the level is a fixed one
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
