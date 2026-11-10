"""DesignSpec: a base flowsheet, the months and years to run it over, what to size."""

import itertools
import math
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, model_validator

from flexcore.config.io import _format_validation_error, _read, load_spec
from flexcore.config.schema import _StrictModel
from flexcore.config.spec import FlowsheetSpec
from flexcore.exceptions import FlexConfigError


class PlanningYear(_StrictModel):
    """One modeled year of the study."""

    year: int = Field(description="Calendar year to model.")
    represents: int = Field(
        default=1, ge=1, description="Calendar years it stands in for."
    )


class RepresentativeMonth(_StrictModel):
    """One calendar month modeled in every planning year."""

    month: int = Field(ge=1, le=12, description="Calendar month, 1 to 12.")
    weight: float = Field(gt=0, description="Months of the year it stands in for.")
    days: int | None = Field(
        default=None,
        ge=1,
        le=28,
        description="Model only the first N days and scale the bill up to the full "
        "month; None models the whole month.",
    )


class ParameterChange(_StrictModel):
    """A change to one base element for the selected years and months."""

    element: str = Field(description="Name of the base element to change.")
    values: dict[str, Any] = Field(description="Fields to deep-merge into the element.")
    years: list[int] | None = Field(
        default=None, min_length=1, description="Years it applies to; None is all."
    )
    months: list[int] | None = Field(
        default=None, min_length=1, description="Months it applies to; None is all."
    )


class CapacitySpec(_StrictModel):
    """One capacity decision shared by every horizon."""

    name: str = Field(description="Decision name, e.g. 'battery_energy'.")
    variable: str = Field(
        description="Scalar Var path, e.g. 'battery.energy_capacity'."
    )
    capital_cost: float = Field(ge=0, description="Overnight cost per Var unit.")
    lifetime_years: float = Field(
        gt=0, description="Asset lifetime used for annualization."
    )
    lower: float = Field(default=0.0, ge=0, description="Lower bound, in Var units.")
    upper: float = Field(description="Required upper bound, in Var units.")

    @model_validator(mode="after")
    def _bounds_are_ordered(self) -> "CapacitySpec":
        """Require lower <= upper."""
        if self.lower > self.upper:
            raise ValueError(f"Capacity {self.name!r} has lower > upper.")
        return self


class DesignSpec(_StrictModel):
    """A base flowsheet, the years and months to run it over, and what to size."""

    schema_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$", description="Semantic version of this study."
    )
    name: str = Field(description="Name of the merged ConcreteModel.")
    base: FlowsheetSpec = Field(description="The flowsheet each horizon starts from.")
    years: list[PlanningYear] = Field(
        min_length=1, description="Modeled years, ascending."
    )
    months: list[RepresentativeMonth] = Field(
        min_length=1, description="Representative months."
    )
    changes: list[ParameterChange] = Field(
        default_factory=list, description="Changes to base elements, applied in order."
    )
    capacities: list[CapacitySpec] = Field(
        min_length=1, description="Decisions to size."
    )
    capex_budget: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Optional cap on total capital cost, in the model " "currency.",
    )

    @model_validator(mode="after")
    def _study_is_consistent(self) -> "DesignSpec":
        """Require consecutive years, weights summing to 12, and one costing element."""
        for this, following in itertools.pairwise(self.years):
            if following.year != this.year + this.represents:
                raise ValueError(
                    f"Year {following.year} must follow {this.year} + represents."
                )
        if not math.isclose(sum(m.weight for m in self.months), 12.0):
            raise ValueError("Month weights must sum to 12.")
        months = [m.month for m in self.months]
        names = [c.name for c in self.capacities]
        if len(set(months)) != len(months) or len(set(names)) != len(names):
            raise ValueError("Months and capacity names must be unique.")
        if len(self.base.of_kind("costing")) != 1:
            raise ValueError("The base spec needs exactly one costing element.")
        return self

    @model_validator(mode="after")
    def _changes_match_the_study(self) -> "DesignSpec":
        """Require each change to target a real element, year and month."""
        elements = {el.name for el in self.base.elements if getattr(el, "name", None)}
        years = {y.year for y in self.years}
        months = {m.month for m in self.months}
        for change in self.changes:
            if change.element not in elements:
                raise ValueError(f"Change targets unknown element {change.element!r}.")
            if {"name", "kind"} & change.values.keys():
                raise ValueError(
                    f"Change to {change.element!r} can't edit name or kind."
                )
            if not set(change.years or years) <= years:
                raise ValueError(
                    f"Change to {change.element!r} selects an unmodeled year."
                )
            if not set(change.months or months) <= months:
                raise ValueError(
                    f"Change to {change.element!r} selects an unmodeled month."
                )
        return self


def load_design_spec(path) -> DesignSpec:
    """Load a design study from JSON, with ``base`` inline or as a relative path.

    Args:
        path: The study's ``.json`` file.

    Returns:
        The validated spec.

    Raises:
        FlexConfigError: If the file or its base spec can't be read or validated.
    """
    path = Path(path)
    data = _read(path)
    base = data.get("base")
    if isinstance(base, str):
        data["base"] = load_spec(path.parent / base)
    elif base is not None:
        data["base"] = load_spec(base)
        data["base"]._base_dir = path.parent
    try:
        return DesignSpec.model_validate(data)
    except ValidationError as exc:
        raise FlexConfigError(_format_validation_error(exc)) from exc
