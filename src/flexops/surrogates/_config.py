"""Config building blocks shared by the surrogate classes' contracts."""

from pydantic import ConfigDict, field_validator

from flexcore.config.schema import _StrictModel
from flexcore.exceptions import FlexConfigError
from flexops.core.units import parse_units


def _check_units(mapping: dict[str, str]) -> dict[str, str]:
    for name, units in mapping.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid variable name {name!r}")
        if not isinstance(units, str) or not units:
            raise ValueError(f"{name!r} must map to a non-empty units string")
        try:
            parse_units(units)
        except FlexConfigError as exc:
            raise ValueError(f"{name!r} has unparsable units {units!r}: {exc}") from exc
    return mapping


class VariablesConfig(_StrictModel):
    """Input and output variables, each mapped to the units a fit declared.

    Each variable is resolved on the unit at build time and converted from its
    own units into the declared ones, so values are read in the basis the fit
    used, whatever units the model happens to carry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_variables: dict[str, str]
    output_variables: dict[str, str]

    @field_validator("input_variables")
    @classmethod
    def _inputs(cls, value: dict[str, str]) -> dict[str, str]:
        return _check_units(value)

    @field_validator("output_variables")
    @classmethod
    def _one_output(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) != 1:
            raise ValueError(
                "must be exactly one {name: units} entry -- a registered relation "
                f"has one target -- got {len(value)}"
            )
        return _check_units(value)
