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

import enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    model_serializer,
    model_validator,
)
from typing_extensions import TypeAliasType


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
    """Which predefined surrogate class a SurrogateSpec describes.

    Each member names a class in flexops.surrogates that defines the
    surrogate's config, validates its parameters, and builds the relationship.
    """

    CONSTANT_INTENSITY = "constant_intensity"
    MULTILINEAR = "multilinear"
    QUADRATIC = "quadratic"
    EXPONENTIAL = "exponential"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"


_Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]

ParameterValue = TypeAliasType("ParameterValue", _Number | list["ParameterValue"])
"""A parameter value: a number, a list of values, or nested lists (a matrix)."""


class ParameterState(_StrictModel):
    """One surrogate parameter's value and whether the model holds it fixed.

    A parameter is a scalar or an array (a list, or nested lists for a matrix);
    the surrogate class declares each parameter's shape. fixed applies to the
    whole parameter. A free parameter may omit value, in which case the
    surrogate starts it from its documented initial guess.
    """

    value: ParameterValue | None = Field(
        default=None,
        description="The parameter's value: a number, a list, or nested lists. "
        "Required when fixed is true.",
    )
    fixed: Annotated[bool, Field(strict=True)] = Field(
        default=True,
        description="Whether the model holds the parameter fixed (true) or "
        "leaves it free for estimation (false). Always written on emit.",
    )

    @model_validator(mode="after")
    def _fixed_needs_value(self) -> "ParameterState":
        if self.fixed and self.value is None:
            raise ValueError(
                "a fixed parameter needs a value; give one, or set fixed to "
                "false to start it from the surrogate's initial guess"
            )
        return self

    @model_serializer(mode="wrap")
    def _always_write_fixed(self, handler):
        dumped = handler(self)
        dumped["fixed"] = self.fixed
        return dumped


class _SurrogateFields(_StrictModel):
    """The fields every surrogate spec carries, wherever it is placed."""

    surrogate_type: SurrogateType = Field(
        description="Which predefined surrogate class this relationship is. "
        "The class (flexops.surrogates) validates config and parameters and "
        "builds the relationship; a class not yet implemented raises "
        "NotImplementedError when the model is built, not when the config is "
        "validated."
    )
    config_version: int = Field(
        default=1,
        strict=True,
        ge=1,
        description="Version of this surrogate type's config format. Each "
        "surrogate class migrates older versions of its own config when it is "
        "built. Always written on emit.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Everything the surrogate needs besides its parameters "
        "(variables and units, structure, options, state such as an ARIMA "
        "history), in the shape surrogate_type's class defines and validates. "
        "Opaque to the schema; see the flexops.surrogates reference page for "
        "each class's config.",
    )
    parameters: dict[str, ParameterState] = Field(
        default_factory=dict,
        description="The surrogate's fitted values, keyed by the names its "
        "class registers (for example 'intercept', 'ar.L1', 'exog.feed'). The "
        "names must match the class's parameters exactly.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form fit metadata (metrics, data window, versions).",
    )

    def free_parameters(self) -> list[str]:
        """Return the names of the parameters that are not fixed, in order."""
        return [name for name, state in self.parameters.items() if not state.fixed]


class SurrogateSpec(_SurrogateFields):
    """A fitted (or default) energy/IO relationship for a unit.

    The same object is produced by a fit (flexparameterize), read back from a
    live surrogate block (flexops), and placed into a flowsheet spec as a
    surrogate element.
    """

    @model_serializer(mode="wrap")
    def _always_write_version(self, handler):
        dumped = handler(self)
        dumped["config_version"] = self.config_version
        return dumped


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


class DegradationTerm(enum.StrEnum):
    """Which wear driver a DegradationTermSpec charges for."""

    VARIATION = "variation"
    DEVIATION = "deviation"
    EXCEEDANCE = "exceedance"
    THROUGHPUT = "throughput"


class DegradationTermSpec(_StrictModel):
    """One priced wear driver of a unit variable, valued in that variable's units."""

    kind: DegradationTerm = Field(
        description="Wear driver: variation (change from window steps earlier), "
        "deviation (distance from reference), exceedance (excursion outside "
        "lower/upper), or throughput (the variable itself)."
    )
    variable: str = Field(
        description="Name of the time-indexed variable on the unit (dotted names "
        "into a sub-block resolve)."
    )
    price: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Cost in the model currency per unit of change for variation, "
        "or per unit of the variable held for one hour for the other kinds.",
    )
    deadband: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Free amount per step, in the variable's units; only the "
        "excess beyond it is charged (a deadband). Not used by exceedance.",
    )
    window: int = Field(
        default=1,
        ge=1,
        description="Variation only: number of steps back the change is measured. "
        "Each move falls in window overlapping windows, so the charge is divided "
        "by window to keep price per unit of change.",
    )
    reference: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Deviation only (required): the target value.",
    )
    lower: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Exceedance only: the charge-free lower bound.",
    )
    upper: float | None = Field(
        default=None,
        allow_inf_nan=False,
        description="Exceedance only: the charge-free upper bound.",
    )

    @model_validator(mode="after")
    def _check_kind_fields(self) -> "DegradationTermSpec":
        """Require each kind's own fields and reject the fields of other kinds."""
        is_kind = {
            "reference": self.kind is DegradationTerm.DEVIATION,
            "lower": self.kind is DegradationTerm.EXCEEDANCE,
            "upper": self.kind is DegradationTerm.EXCEEDANCE,
        }
        for field, allowed in is_kind.items():
            if not allowed and getattr(self, field) is not None:
                raise ValueError(f"{field} is not used by a {self.kind} term.")
        if self.kind is DegradationTerm.DEVIATION and self.reference is None:
            raise ValueError("A deviation term requires a reference.")
        if self.kind is DegradationTerm.EXCEEDANCE:
            if self.lower is None and self.upper is None:
                raise ValueError("An exceedance term requires lower and/or upper.")
            if self.deadband:
                raise ValueError("An exceedance term's bounds are its deadband.")
            if None not in (self.lower, self.upper) and self.lower > self.upper:
                raise ValueError("An exceedance term's lower exceeds its upper.")
        if self.kind is not DegradationTerm.VARIATION and self.window != 1:
            raise ValueError("window is only used by a variation term.")
        return self


class DegradationSpec(_StrictModel):
    """A named, priced degradation penalty on a unit: a sum of wear-driver terms."""

    name: str = Field(description="Name of the penalty, unique on its unit.")
    terms: list[DegradationTermSpec] = Field(
        min_length=1, description="The priced wear drivers that are summed."
    )
    covered_cost: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Free wear cost per period, in the model currency; only cost "
        "beyond it is charged.",
    )
    horizon_budget: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Optional hard cap on wear cost per period, in the model "
        "currency.",
    )
    period_hours: float | None = Field(
        default=None,
        gt=0,
        allow_inf_nan=False,
        description="Length of the period covered_cost and horizon_budget "
        "are stated for (e.g. 730 for a month); they are prorated to the modeled "
        "horizon. None means they are stated for the modeled horizon.",
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


# Architecture references: demand response is plan/01_architecture.md §2.4/§3.6.
class DRConfig(_StrictModel):
    """Demand-response container slot, a placeholder in v0: DR containers exist
    so wiring is stable, but no DR constraints are built. Turning DR on later
    is additive."""

    events_source: str | None = Field(
        default=None,
        description="Optional file or tag pointing at demand-response events.",
    )


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
        description="Optional demand-response container (containers-only in v0).",
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
