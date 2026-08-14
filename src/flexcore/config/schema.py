"""The versioned flex-pse config schema (pydantic v2, the schema authority).

A whole flex-pse model and run are built from a single version-controlled
config artifact (``flexops.build_model(config)``): the TimeBlock,
properties, costing, and the network/plant/unit tree all come from one
:class:`ModelConfig`. These pydantic models are the **authority** for that
config (``plan/01_architecture.md`` §2.3); JSON is the canonical
and only on-disk format (see :mod:`flexcore.config.io`).

Every field carries a ``description`` (it renders into the docs) and every model
forbids unknown keys (``plan/00_conventions.md`` §4: an undocumented key does
not get to exist). Only the persisted (Layer-1) config lives here; runtime Pyomo
``ConfigDict`` options are a separate layer and are never serialized.

Class docstrings and field descriptions here are exported verbatim into the
JSON Schema, so keep them plain text: no section signs, RST markup, or manual
line-break art — rendering is the documentation builder's job.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CURRENT_SCHEMA_VERSION = "0.0.2"
"""str: the semantic schema version this build writes and validates against."""


class _StrictModel(BaseModel):
    """Base for every config model: reject undocumented keys (conventions §4)."""

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


class SurrogateSpec(_StrictModel):
    """A fitted (or default) energy/IO relationship for a unit."""

    functional_form: str = Field(
        description="Name of the relationship builder to use, e.g. "
        "'constant_intensity', 'linear', 'quadratic', 'bilinear'. Deliberately "
        "an open string rather than a fixed list: builders are registered in "
        "code, so a new functional form never needs a schema revision. A name "
        "no builder is registered for is rejected when the model is built, not "
        "when the config is validated."
    )
    coefficients: dict[str, float] = Field(
        default_factory=dict,
        description="Coefficients of the relationship, keyed by the term each "
        "multiplies. A term is a '*'-separated product of input variable names, "
        "each optionally raised to an integer power with '^': 'flow_out', "
        "'flow_out^2', 'flow_out*outlet_state.pressure'. The reserved key "
        "'intercept' is the constant term. Each coefficient is read in kW over "
        "the product of its factors' own units.",
    )
    source: str | None = Field(
        default=None,
        description="Optional path to a JSON file supplying this "
        "relationship's coefficients, input_variables, and output_variables, "
        "for a relationship too large to inline or not expressible as "
        "coefficients at all. A relative path resolves against the directory of "
        "the config file that names it. Anything the file supplies replaces "
        "what is written inline here.",
    )
    input_variables: list[str] = Field(
        default_factory=list,
        description="Names of the relationship's input variables, each "
        "resolvable on the unit; the coefficient terms name these.",
    )
    output_variables: list[str] = Field(
        default_factory=list,
        description="Names of the relationship's output variables.",
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
    startup_shutdown: bool = Field(
        default=False,
        description="Whether to build startup/shutdown transition logic.",
    )
    dwell: bool = Field(
        default=False,
        description="Whether to build minimum up/down-time (dwell) constraints.",
    )
    min_up: int | None = Field(
        default=None,
        description="Minimum number of steps the unit must stay up once started.",
    )
    min_down: int | None = Field(
        default=None,
        description="Minimum number of steps the unit must stay down once stopped.",
    )
    delays: dict[str, Any] | None = Field(
        default=None,
        description="Upstream-linked startup-delay specification.",
    )
    conditional: dict[str, Any] | None = Field(
        default=None,
        description="Conditional status implications between units.",
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
    unit_commitment: UnitCommitmentConfig = Field(
        default_factory=UnitCommitmentConfig,
        description="Per-unit unit-commitment configuration.",
    )
    external_dispatch: ExternalDispatchSpec | None = Field(
        default=None,
        description="Optional external (DERMS) dispatch source for the unit.",
    )


class ArcSpec(_StrictModel):
    """A directed connection between two unit (or plant) ports."""

    source: str = Field(description="Source endpoint as a 'unit.port' string.")
    destination: str = Field(
        description="Destination endpoint as a 'unit.port' string."
    )


class PlantConfig(_StrictModel):
    """A named collection of unit models and the arcs between them."""

    name: str = Field(description="Human-readable plant name.")
    units: dict[str, UnitConfig] = Field(
        description="Units of the plant, keyed by their attribute name."
    )
    arcs: list[ArcSpec] = Field(
        default_factory=list,
        description="Arcs connecting unit ports within the plant.",
    )


class NetworkConfig(_StrictModel):
    """A named collection of plants and the inter-plant arcs between them."""

    name: str = Field(description="Human-readable network name.")
    plants: dict[str, PlantConfig] = Field(
        description="Plants of the network, keyed by their attribute name."
    )
    arcs: list[ArcSpec] = Field(
        default_factory=list,
        description="Arcs connecting plant ports across the network.",
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
    currency: str = Field(
        default="USD",
        description="Currency basis to use when no tariff is given; a tariff's own "
        "currency basis always wins.",
    )
    dr: DRConfig | None = Field(
        default=None,
        description="Optional demand-response container (containers-only in v0).",
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


# Architecture references: the config artifact is plan/01_architecture.md §2.3.
class ModelConfig(_StrictModel):
    """The top-level config artifact the whole model and run are built from."""

    schema_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic schema version of this config, an X.Y.Z string; "
        "mandatory, no default.",
    )
    time: TimeConfig = Field(description="The discrete-time horizon.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Property-package specification (kept loose at schema "
        "version 0.0.1).",
    )
    costing: CostingConfig = Field(description="Tariff/DR/solve options.")
    network: NetworkConfig | None = Field(
        default=None,
        description="A network of plants; mutually exclusive with 'plant'.",
    )
    plant: PlantConfig | None = Field(
        default=None,
        description="A single plant of units; mutually exclusive with 'network'.",
    )

    @model_validator(mode="after")
    def _exactly_one_topology(self) -> "ModelConfig":
        """Require exactly one of ``network`` or ``plant`` (pitfall 10)."""
        if (self.network is None) == (self.plant is None):
            raise ValueError(
                "Set exactly one of 'network' or 'plant' on ModelConfig "
                f"(got network={self.network is not None}, "
                f"plant={self.plant is not None})."
            )
        return self
