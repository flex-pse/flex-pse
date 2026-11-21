"""Configuration for fitting surrogate specs from parameter sweeps of a Pyomo model.

Every field carries a ``description`` and every model forbids unknown keys and
is frozen, so a config hashes to the same value for as long as it exists.
"""

import enum
import hashlib

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.units import parse_units


class _StrictModel(BaseModel):
    """Base for every parameterize config model: unknown keys rejected, frozen."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_units(text: str) -> str:
    """Return ``text`` unchanged if it parses as a units string.

    Raises:
        ValueError: If ``parse_units`` rejects it.
    """
    try:
        parse_units(text)
    except FlexConfigError as exc:
        raise ValueError(str(exc)) from exc
    return text


class SamplerType(enum.StrEnum):
    """How sweep points are drawn from the input ranges."""

    GRID = "grid"
    LHS = "lhs"


class SweepInput(_StrictModel):
    """One swept model input."""

    path: str = Field(
        description="Component path on the built model, e.g. 'fs.feed.flow_vol'. "
        "Must be a fixed Var or a mutable Param."
    )
    units: str = Field(
        description="Units of the component on the model, checked when sweeping."
    )
    range: tuple[float, float] = Field(
        description="Lower and upper sweep bound, in the component's units."
    )

    @field_validator("units")
    @classmethod
    def _parsable_units(cls, value: str) -> str:
        """Reject a units string parse_units cannot read."""
        return _check_units(value)

    @field_validator("range")
    @classmethod
    def _ordered(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Reject a range whose lower bound is not below its upper bound."""
        if value[0] >= value[1]:
            raise ValueError(f"range lower bound must be below upper, got {value}.")
        return value


class SweepOutput(_StrictModel):
    """One model output to record and fit to a flex-pse unit's relation."""

    path: str = Field(description="Component path of the output on the built model.")
    units: str = Field(
        description="Units of the output on the model, checked when sweeping."
    )
    unit: str = Field(
        description="Full name of the flex-pse unit the spec is applied to, "
        "e.g. 'facility.ro'."
    )
    relation: str = Field(
        default="power_electrical_relation",
        description="Name of the unit's relation the spec replaces.",
    )
    output_units: str = Field(
        description="Units of the fitted spec's output; values are converted "
        "from units."
    )
    input_names: dict[str, str] = Field(
        description="Config input name to the flex-pse unit variable it maps to, "
        "e.g. {'feed': 'flow_in'}. Needed for every input used in terms."
    )
    surrogate: SurrogateType = Field(
        default=SurrogateType.MULTILINEAR,
        description="Surrogate type to fit. Only multilinear is supported.",
    )
    terms: list[str] = Field(
        min_length=1,
        description="Terms to fit, as config input names or '*'-separated "
        "products of distinct input names, e.g. ['feed', 'recovery', "
        "'feed*recovery']. The intercept is always fitted.",
    )

    @field_validator("units", "output_units")
    @classmethod
    def _parsable_units(cls, value: str) -> str:
        """Reject a units string parse_units cannot read."""
        return _check_units(value)

    @field_validator("surrogate")
    @classmethod
    def _multilinear_only(cls, value: SurrogateType) -> SurrogateType:
        """Reject surrogate types sweeps cannot fit yet."""
        if value is not SurrogateType.MULTILINEAR:
            raise ValueError(
                f"surrogate {value.value!r} is not yet supported by sweeps; "
                "use 'multilinear'."
            )
        return value

    @field_validator("terms")
    @classmethod
    def _plain_products(cls, value: list[str]) -> list[str]:
        """Reject exponents and repeated factors in a term."""
        for term in value:
            factors = term.split("*")
            if any("^" in factor for factor in factors):
                raise ValueError(f"term {term!r} uses '^'; write products only.")
            if len(set(factors)) != len(factors):
                raise ValueError(f"term {term!r} repeats a factor.")
        return value

    @model_validator(mode="after")
    def _names_cover_terms(self) -> "SweepOutput":
        """Require an input_names entry for every factor and distinct targets."""
        factors = {f for term in self.terms for f in term.split("*")}
        missing = sorted(factors - set(self.input_names))
        if missing:
            raise ValueError(f"input_names has no entry for term factors {missing}.")
        if len(set(self.input_names.values())) != len(self.input_names):
            raise ValueError(
                f"input_names maps two inputs to the same variable: {self.input_names}."
            )
        return self


class SamplerConfig(_StrictModel):
    """How many sweep points to draw, and how."""

    type: SamplerType = Field(description="Grid or Latin hypercube sampling.")
    n: int = Field(
        ge=2,
        description="Points per input for a grid (n^d in total), or total "
        "points for a Latin hypercube.",
    )
    seed: int = Field(description="Random seed passed to the sweep.")


class SolverConfig(_StrictModel):
    """The solver handed to the optimize function."""

    name: str = Field(default="ipopt", description="Pyomo SolverFactory name.")
    options: dict[str, float | int | str] = Field(
        default_factory=dict, description="Solver options set before every solve."
    )


class HoldoutConfig(_StrictModel):
    """Which rows are held out of the fit: those at position p with p % k == r."""

    k: int = Field(ge=2, description="Holdout period.")
    r: int = Field(ge=0, description="Holdout remainder, below k.")

    @model_validator(mode="after")
    def _remainder_below_period(self) -> "HoldoutConfig":
        """Reject a remainder that selects no rows."""
        if self.r >= self.k:
            raise ValueError(f"holdout r must be below k, got r={self.r}, k={self.k}.")
        return self


class AcceptanceConfig(_StrictModel):
    """Limits the holdout rows must meet for a fit to be returned."""

    max_rel_error: float = Field(
        default=0.1,
        gt=0.0,
        description="Maximum holdout |predicted - actual| / |actual|.",
    )
    max_rel_rmse: float = Field(
        default=0.1,
        gt=0.0,
        description="Maximum holdout RMSE divided by the mean |actual|.",
    )
    r2_min: float | None = Field(
        default=None, description="Minimum holdout R^2, or None to skip the check."
    )


class ParameterizeConfig(_StrictModel):
    """Everything a sweep and its fit need: inputs, outputs, sampling and limits."""

    inputs: dict[str, SweepInput] = Field(
        min_length=1, description="Swept inputs, keyed by config input name."
    )
    context: dict[str, float] = Field(
        default_factory=dict,
        description="Component path to the value it is fixed at after the model "
        "is built, in the component's units.",
    )
    sampler: SamplerConfig = Field(description="Sampling of the inputs.")
    solver: SolverConfig = Field(
        default_factory=SolverConfig, description="Solver for every point."
    )
    outputs: dict[str, SweepOutput] = Field(
        min_length=1, description="Recorded outputs, keyed by output name."
    )
    max_failed_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description="Largest fraction of failed points a fit accepts.",
    )
    holdout: HoldoutConfig = Field(description="Rows held out of the fit.")
    acceptance: AcceptanceConfig = Field(
        default_factory=AcceptanceConfig, description="Limits on the holdout rows."
    )

    @model_validator(mode="after")
    def _outputs_match_inputs(self) -> "ParameterizeConfig":
        """Require term factors to be inputs and one output per unit relation."""
        targets = set()
        for name, output in self.outputs.items():
            unknown = sorted(set(output.input_names) - set(self.inputs))
            if unknown:
                raise ValueError(
                    f"output {name!r} names inputs {unknown} that are not in "
                    f"inputs {sorted(self.inputs)}."
                )
            target = (output.unit, output.relation)
            if target in targets:
                raise ValueError(f"two outputs target {output.unit} {output.relation}.")
            targets.add(target)
        return self


def config_hash(config: ParameterizeConfig) -> str:
    """Return the SHA-256 hex digest of the config's JSON dump.

    Args:
        config: The config to hash.

    Returns:
        A 64-character hex string.
    """
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()
