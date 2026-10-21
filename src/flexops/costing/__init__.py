"""flex-pse costing: the thin interface around the external EECO package.

The tariff/operating-cost engine is EECO (``eeco`` on PyPI); this package wraps
it. Tariff ``eeco`` calls are collected in :mod:`flexops.costing.opex` and
demand-response calls in :mod:`flexops.costing.demand_response`.
"""

from flexops.costing.demand_response import (
    add_dr_revenue,
    dr_baseline_method,
    dr_payment_structure,
    evaluate_dr_revenue,
    historical_power_series,
)
from flexops.costing.flex_costing import (
    CapitalCostBreakdown,
    CostReport,
    FlexCosting,
    OperatingCostBreakdown,
    ScalarCostSpec,
)
from flexops.costing.opex import (
    OperatingCostHandles,
    add_electricity_cost,
    add_fuel_cost,
    add_operating_cost,
    currency_units,
    evaluate_cost,
    evaluate_fuel_cost,
    is_peak,
    load_tariff,
    merge_tariffs,
    monthly_scale_factor,
    peak_windows,
    price_gradient,
    price_series,
    tariff_csv_to_dict,
    tariff_currency_units,
)

__all__ = [
    "load_tariff",
    "merge_tariffs",
    "tariff_csv_to_dict",
    "tariff_currency_units",
    "currency_units",
    "monthly_scale_factor",
    "price_series",
    "is_peak",
    "peak_windows",
    "price_gradient",
    "add_operating_cost",
    "add_electricity_cost",
    "add_fuel_cost",
    "OperatingCostHandles",
    "evaluate_cost",
    "evaluate_fuel_cost",
    "add_dr_revenue",
    "evaluate_dr_revenue",
    "dr_baseline_method",
    "dr_payment_structure",
    "historical_power_series",
    "FlexCosting",
    "CostReport",
    "OperatingCostBreakdown",
    "CapitalCostBreakdown",
    "ScalarCostSpec",
]
