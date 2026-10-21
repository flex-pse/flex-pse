"""The versioned flex-pse config schema (pydantic v2, the schema authority).

These are the shared building blocks of the flat
:class:`~flexcore.config.spec.FlowsheetSpec`; JSON is the canonical and only
on-disk format (see :mod:`flexcore.config.io`).

Every field carries a ``description`` (it renders into the docs) and every model
forbids unknown keys (an undocumented key does not get to exist). Only the
persisted (Layer-1) config lives here; runtime Pyomo
``ConfigDict`` options are a separate layer and are never serialized.

Class docstrings and field descriptions here are exported verbatim into the
JSON Schema, so keep them plain text: no section signs, RST markup, or manual
line-break art — rendering is the documentation builder's job.
"""

import datetime
import enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, model_validator


class _StrictModel(BaseModel):
    """Base for every config model: reject undocumented keys (an undocumented
    key does not get to exist)."""

    model_config = ConfigDict(extra="forbid")


class IOVariableSpec(_StrictModel):
    """A declared process input or output variable of a unit."""

    name: str = Field(description="Local variable name on the unit block.")
    role: Literal["input", "output"] = Field(
        description="Whether the variable is a process input or output."
    )
    units: str = Field(description="Units of the variable as a string, e.g. 'm^3/hr'.")
    tag_hint: str | None = Field(
        default=None,
        description="Optional historian-tag hint for FlexParameterize aliasing.",
    )
    time_indexed: bool = Field(
        default=True,
        description="Whether the variable is indexed over the time set.",
    )


class SurrogateType(enum.StrEnum):
    """Which predefined surrogate class a SurrogateSpec's data describes.

    Each member names a class in flexops.surrogates that validates and builds
    the relationship; CONSTANT_INTENSITY is the one exception -- it has no
    class, since it fixes a process parameter rather than swapping a
    Constraint (see flexops.core.ops_block.OpsBlockData.swap_relation).
    """

    CONSTANT_INTENSITY = "constant_intensity"
    MULTILINEAR = "multilinear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"


class SurrogateSpec(_StrictModel):
    """A fitted (or default) energy/IO relationship for a unit."""

    surrogate_type: SurrogateType = Field(
        description="Which predefined surrogate class this relationship is. "
        "The class (flexops.surrogates) validates and builds data; a class "
        "not yet implemented raises NotImplementedError when the model is "
        "built, not when the config is validated."
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="The relationship's data, in the shape surrogate_type's "
        "class defines and validates (e.g. a multilinear surrogate's "
        "coefficients and its input/output variables with their units). "
        "Opaque to the schema; see the flexops.surrogates reference page for "
        "each class's data contract.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form fit metadata (metrics, data window, versions).",
    )


# Architecture references: external dispatch is plan/01_architecture.md §3.2.
class ExternalDispatchSpec(_StrictModel):
    """Declares an external (DERMS) command source for a controllable variable."""

    variable: str = Field(
        description="Name of the controllable variable to drive externally."
    )
    source: str = Field(
        description="File or tag pointing at the time-indexed command series."
    )
    fix: bool = Field(
        default=True,
        description="Whether to fix the variable to the series (remove its DOF).",
    )


# Architecture references: unit commitment is plan/01_architecture.md §3.5.
class UnitCommitmentConfig(_StrictModel):
    """Per-unit unit-commitment configuration, a validated container.
    Every piece is optional except status."""

    status: bool = Field(
        default=True,
        description="Whether the unit has an on/off status binary (a tank sets "
        "this False).",
    )


class PropertyPackageSpec(_StrictModel):
    """A property package the model builds, by class name and options."""

    property_class: str = Field(
        description="Name of a flexops property-package class, e.g. "
        "'SimpleAqueousFlow' or 'SimpleGasFlow'."
    )
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword options passed to the property-package constructor, "
        "e.g. {'has_pressure': true}.",
    )


class UnitConfig(_StrictModel):
    """A single unit model: its class, construction options, and IO/logic."""

    unit_model_class: str = Field(
        description="Name of the flexops unit-model class to construct."
    )
    construction_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword options passed to the unit-model constructor.",
    )
    io_variables: list[IOVariableSpec] = Field(
        default_factory=list,
        description="Declared process input/output variables of the unit.",
    )
    surrogate: SurrogateSpec | None = Field(
        default=None,
        description="Optional fitted energy/IO relationship for the unit.",
    )
    unit_commitment: UnitCommitmentConfig | None = Field(
        default=None,
        description="Per-unit unit-commitment configuration. Unset (null) "
        "leaves the unit model's own default in force (a battery or tank has "
        "no status binary unless asked for one).",
    )
    external_dispatch: ExternalDispatchSpec | None = Field(
        default=None,
        description="Optional external (DERMS) dispatch source for the unit.",
    )


# Architecture references: the time horizon is plan/01_architecture.md §3.1.
class TimeConfig(_StrictModel):
    """The discrete-time horizon specification."""

    start_date: str = Field(description="Inclusive ISO-8601 start of the horizon.")
    end_date: str = Field(description="Exclusive ISO-8601 end of the horizon.")
    time_step: str = Field(
        description="Time-step as a units-carrying expression string, e.g. "
        "'15 min'; parsed at build time."
    )


class DRBaselineMethod(enum.StrEnum):
    """How a demand-response baseline (the load the site would have drawn) is set."""

    SIMILAR_DAYS = "similar_days"
    TOP_USAGE_DAYS = "top_usage_days"
    FIXED_LEVEL = "fixed_level"
    UNILATERAL_INTERRUPTION = "unilateral_interruption"


class DRPaymentKind(enum.StrEnum):
    """How a demand-response program pays for a delivered reduction."""

    CAPACITY = "capacity"
    CAPACITY_ENERGY = "capacity_energy"


class DRSettlement(enum.StrEnum):
    """Whether a payment settles each interval or the event-mean reduction."""

    INTERVAL = "interval"
    AVERAGE = "average"


class DRPaymentBasis(enum.StrEnum):
    """Whether a price is paid per event, per event hour, or as a monthly share."""

    PER_EVENT = "per_event"
    PER_HOUR = "per_hour"
    MONTHLY_SHARE = "monthly_share"


class DREvent(_StrictModel):
    """One demand-response event, in local wall-clock time."""

    event_date: datetime.date = Field(description="Calendar date of the event.")
    start_hour: float = Field(description="Hour of day (0-24) the event begins.")
    duration_hours: float = Field(description="Length of the event in hours.")
    notification_hours: float = Field(
        default=0.0, description="Advance notice before the event, in hours."
    )
    baseline_days: list[datetime.date] = Field(
        min_length=1,
        description="Candidate days the baseline may average over; exclude other "
        "event days.",
    )
    bid_capacity_kW: float = Field(description="Bid capacity in kW.")
    capacity_price: float = Field(
        description="Capacity price in USD/kW, or USD/kW-hour with the per_hour "
        "basis, or USD/kW-month with the monthly_share basis."
    )
    adjustment_factor: float | None = Field(
        default=None,
        description="Day-of adjustment factor to apply as is; omit to compute it "
        "from meter history.",
    )
    monthly_called_capacity_kWh: float | None = Field(
        default=None,
        description="Called capacity summed over every event hour in the month, for "
        "the monthly_share basis; omit to sum it over the given events.",
    )
    region_x1: float | None = Field(
        default=None,
        description="Lower delivered-ratio bound of the payment region to fix for "
        "this event, which removes its region-choice binaries; omit to let the "
        "optimizer choose.",
    )


class DRBaseline(_StrictModel):
    """How the demand-response baseline is computed."""

    method: DRBaselineMethod = Field(description="The baseline method.")
    resolution: str | None = Field(
        default=None,
        description="Settlement interval width as an EECO resolution string, e.g. "
        "'15m' or '1h'; omit to infer it from the data.",
    )
    n_baseline_days: int = Field(
        default=10,
        ge=0,
        description="Similar-days methods: number of eligible days to average; 0 "
        "gives a zero baseline.",
    )
    adjustment_offset_hours: int | None = Field(
        default=4,
        description="Similar-days methods: hours before the event start that the "
        "day-of adjustment window begins; null turns the adjustment off.",
    )
    adjustment_duration_hours: int = Field(
        default=3,
        description="Similar-days methods: length of the day-of adjustment window "
        "in hours.",
    )
    adjustment_clip: tuple[float, float] = Field(
        default=(0.8, 1.2),
        description="Similar-days methods: lower and upper bounds on the day-of "
        "adjustment factor.",
    )
    exclude_weekends: bool = Field(
        default=True, description="Similar-days methods: drop Saturdays and Sundays."
    )
    exclude_holidays: bool = Field(
        default=True, description="Similar-days methods: drop holidays."
    )
    holiday_country: str | None = Field(
        default="US",
        description="Similar-days methods: ISO 3166-1 country code of the holiday "
        "calendar; null uses holiday_dates only.",
    )
    holiday_subdiv: str | None = Field(
        default=None,
        description="Similar-days methods: holiday-calendar subdivision, e.g. 'CA'.",
    )
    holiday_dates: list[datetime.date] = Field(
        default_factory=list,
        description="Similar-days methods: extra dates to treat as holidays.",
    )
    firm_level_kW: float | None = Field(
        default=None,
        description="fixed_level method: the contracted firm service level in kW.",
    )
    interruption_level_kW: float = Field(
        default=0.0,
        description="unilateral_interruption method: the level in kW load is held "
        "at or below during an event.",
    )

    @model_validator(mode="after")
    def _fixed_level_has_a_level(self) -> "DRBaseline":
        """Require firm_level_kW for the fixed_level method."""
        if self.method is DRBaselineMethod.FIXED_LEVEL and self.firm_level_kW is None:
            raise ValueError("the fixed_level baseline needs firm_level_kW.")
        return self


class DRPaymentRegion(_StrictModel):
    """One region of a payment schedule, mapping delivered ratio to payment ratio."""

    x1: float | None = Field(
        default=None,
        description="Lower delivered-ratio bound (reduction over bid capacity); "
        "null means unbounded below.",
    )
    x2: float | None = Field(
        default=None,
        description="Upper delivered-ratio bound; null means unbounded above, where "
        "the region pays a flat y1.",
    )
    y1: float = Field(description="Payment ratio at x1.")
    y2: float = Field(description="Payment ratio at x2.")


class DRPayment(_StrictModel):
    """How a demand-response program pays for the delivered reduction."""

    kind: DRPaymentKind = Field(description="The payment structure.")
    regions: list[DRPaymentRegion] | None = Field(
        default=None,
        description="Piecewise-linear payment schedule; omit for a payout-only "
        "program. Each region adds binaries to the model unless the event fixes "
        "region_x1.",
    )
    settlement: DRSettlement = Field(
        default=DRSettlement.INTERVAL,
        description="Settle each interval's reduction, or the event-mean reduction.",
    )
    payment_basis: DRPaymentBasis = Field(
        default=DRPaymentBasis.PER_EVENT,
        description="Basis the capacity price is paid on.",
    )
    payout: float = Field(
        default=0.0,
        description="Flat participation payment in USD per kW of bid capacity.",
    )
    payout_basis: DRPaymentBasis = Field(
        default=DRPaymentBasis.PER_EVENT,
        description="Basis the payout is paid on.",
    )
    energy_price: float | None = Field(
        default=None,
        description="capacity_energy kind: payment in USD/kWh for energy curtailed.",
    )

    @model_validator(mode="after")
    def _energy_payment_has_a_price(self) -> "DRPayment":
        """Require energy_price for the capacity_energy kind."""
        if self.kind is DRPaymentKind.CAPACITY_ENERGY and self.energy_price is None:
            raise ValueError("the capacity_energy payment needs energy_price.")
        return self


class HistoricalPower(_StrictModel):
    """Evenly spaced meter history the baseline is computed from."""

    start: datetime.datetime = Field(
        description="Naive local timestamp of the first value."
    )
    time_step_minutes: float = Field(
        gt=0, description="Spacing between values in minutes."
    )
    values_kW: list[float] = Field(min_length=1, description="Site power in kW.")


class DRConfig(_StrictModel):
    """A demand-response program: its events, baseline, payment, and meter history."""

    events: list[DREvent] = Field(min_length=1, description="The program's events.")
    historical_power: HistoricalPower = Field(
        description="Meter history the baseline is computed from."
    )
    baseline: DRBaseline = Field(description="How the baseline is computed.")
    payment: DRPayment = Field(description="How the reduction is paid.")


class PriceSpec(_StrictModel):
    """A native price for one energy carrier or fuel: a number and its units."""

    value: float | list[float] = Field(
        description="The numeric price, in the units below. One number is a flat "
        "price over the whole horizon; a list is one price per time point and must "
        "have exactly as many entries as the horizon has time points, which is "
        "checked when the model is built."
    )
    units: str = Field(
        description="Units of the price as a string, e.g. 'USD/kWh' for "
        "electricity or 'USD/m^3' for a fuel."
    )

    @model_validator(mode="after")
    def _series_is_not_empty(self) -> "PriceSpec":
        """Reject an empty price list: it can never align with a horizon."""
        if isinstance(self.value, list) and not self.value:
            raise ValueError("a price list needs one entry per time point, not zero.")
        return self


class CostingConfig(_StrictModel):
    """Tariff, price, demand-response, and solve/objective options for a run."""

    tariff_source: str | list[str] | dict[str, str] | None = Field(
        default=None,
        description="Where the EECO tariff comes from: one file or tag, a list of "
        "them to merge, or a mapping of EECO utility ('electric'/'gas') to file to "
        "merge and also assign each file to a utility. Omit to price every carrier "
        "from energy_prices instead.",
    )
    energy_prices: dict[str, PriceSpec] | None = Field(
        default=None,
        description="Optional native prices keyed by carrier or fuel name "
        "('electrical', or a fuel such as 'natural_gas'). A carrier priced here is "
        "billed at that price instead of through the tariff, so a run with every "
        "carrier priced needs no tariff at all. Each price is flat over the horizon "
        "or given per time point.",
    )
    export_prices: dict[str, PriceSpec] | None = Field(
        default=None,
        description="Optional prices paid for exports, keyed like energy_prices "
        "('electrical' or a fuel name). Each key needs a matching energy_prices "
        "entry and must not exceed it at any time point. A carrier left out "
        "credits exports at its import price.",
    )
    decomposition_type: str | None = Field(
        default=None,
        description="Optional EECO decomposition of net electrical power and fuel "
        "usage into imports and exports on the tariff path (e.g. 'linear'), so the "
        "tariff's export charges apply. Passed straight to EECO.",
    )
    currency: str = Field(
        default="USD",
        description="Currency basis to use when no tariff is given; a tariff's own "
        "currency basis always wins.",
    )
    dr: DRConfig | None = Field(
        default=None,
        description="Optional demand-response program, whose revenue is "
        "subtracted from the operating cost.",
    )
    consumption_estimate: dict[Literal["electric", "gas"], NonNegativeFloat] | None = (
        Field(
            default=None,
            description="Estimated total consumption over the horizon, keyed by EECO "
            "utility ('electric'/'gas'; kWh / m^3). EECO needs it to price any tier "
            "other than a flat top tier.",
        )
    )
    fixed_operating_cost: float = Field(
        default=0.0,
        description="Fixed operating cost over the horizon, in the currency basis "
        "in force (a tariff's own, else 'currency'): non-tariff costs such as "
        "maintenance, labor, and chemicals. Distinct from the "
        "tariff's own fixed charge (which EECO includes in the electricity cost).",
    )
    prorate_monthly_charges: bool = Field(
        default=True,
        description="Prorate a tariff's monthly-assessed demand charge and fixed "
        "(customer) charge to the horizon length when the horizon is shorter than "
        "the calendar month it starts in. False bills the full monthly charges.",
    )
    lifetime_years: float = Field(
        default=20.0,
        description="Plant lifetime in years (> 0), used with the effective rate "
        "to form the capital recovery factor that annualizes capital cost.",
    )
    discount_rate: float = Field(
        default=0.08,
        description="Annual discount rate (fraction, e.g. 0.08 = 8%). Used alone "
        "as the effective rate when interest_rate is unset; otherwise it deflates "
        "interest_rate into a real effective rate. An effective rate of 0 falls "
        "back to straight-line 1/lifetime.",
    )
    interest_rate: float | None = Field(
        default=None,
        description="Optional annual cost of capital (fraction, e.g. 0.06). When "
        "given, the capital recovery factor uses the effective rate "
        "(1 + interest_rate) / (1 + discount_rate) - 1. Omit to use discount_rate "
        "alone.",
    )
    objective: Literal["cost"] = Field(
        default="cost",
        description="Objective to minimize (only tariff cost in v0).",
    )
    solver: str | None = Field(
        default=None,
        description="Optional explicit solver name; None lets the facade pick.",
    )

    @model_validator(mode="after")
    def _some_pricing_source(self) -> "CostingConfig":
        """Require a tariff or at least one flat price, so pricing is never empty."""
        if self.tariff_source is None and not self.energy_prices:
            raise ValueError(
                "costing needs a pricing source: set tariff_source, or give at "
                "least one entry in energy_prices."
            )
        return self
