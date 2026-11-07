"""flex-pse costing: the thin interface around the external EECO package.

The tariff/operating-cost engine is EECO (``eeco`` on PyPI); this package wraps
it. All ``eeco`` calls are collected in :mod:`flexops.costing.opex` (the sole
import point).
"""

from flexops.costing.flex_costing import (
    CapitalCostBreakdown,
    CostReport,
    FlexCosting,
    OperatingCostBreakdown,
    ScalarCostSpec,
    capital_recovery_factor,
)
from flexops.costing.opex import (
    DRConfig,
    OperatingCostHandles,
    add_electricity_cost,
    add_fuel_cost,
    add_operating_cost,
    currency_units,
    evaluate_cost,
    evaluate_fuel_cost,
    is_peak,
    load_dr_program,
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
    "load_dr_program",
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
    "DRConfig",
    "FlexCosting",
    "capital_recovery_factor",
    "CostReport",
    "OperatingCostBreakdown",
    "CapitalCostBreakdown",
    "ScalarCostSpec",
]
