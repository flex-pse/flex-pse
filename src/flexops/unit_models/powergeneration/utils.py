"""Fuel-source config validators shared by fuel-burning unit models."""

from collections.abc import Mapping, Sequence

from flexcore.exceptions import FlexConfigError


def inlet_names_domain(value) -> tuple[str, ...] | None:
    """ConfigValue domain: coerce inlet names to a tuple, or accept None.

    Args:
        value: The configured inlet names, or ``None`` for no inlets.

    Returns:
        The names as a tuple, or ``None``.
    """
    if value is None:
        return None
    return tuple(value)


def heating_values_domain(value) -> dict | None:
    """ConfigValue domain: ``None``, or a mapping of fuel name to heating value.

    Args:
        value: The configured heating values.

    Returns:
        The mapping as a plain dict, or ``None``.

    Raises:
        FlexConfigError: If ``value`` is not a mapping.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FlexConfigError(
            "heating_values must be a mapping of inlet name to heating "
            f"value, got {type(value).__name__}.",
            field="heating_values",
            value=value,
        )
    return dict(value)


def utility_fuel_source_domain(value) -> tuple[str, ...] | None:
    """ConfigValue domain: None, a single fuel name, or a tuple of fuel names.

    Args:
        value: The configured utility fuel name(s).

    Returns:
        The names as a tuple, or ``None``.

    Raises:
        FlexConfigError: If ``value`` is not a string or a list of non-empty strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        result = tuple(value)
        if not all(isinstance(v, str) and v for v in result):
            raise FlexConfigError(
                "utility_fuel_source must contain non-empty strings; "
                f"got {result!r}.",
                field="utility_fuel_source",
                value=value,
            )
        return result
    raise FlexConfigError(
        "utility_fuel_source must be None, a string, or a list/tuple of "
        f"strings; got {type(value).__name__}.",
        field="utility_fuel_source",
        value=value,
    )


def validate_fuel_sources(
    inlet_names: Sequence[str],
    utility_names: Sequence[str] | None,
    *,
    inlet_field: str,
    require_any: bool,
) -> None:
    """Check fuel inlet and utility names are valid, unique, and disjoint.

    Args:
        inlet_names: Names of the fuel inlet ports.
        utility_names: Names of the utility fuels, or ``None``.
        inlet_field: Config field name carrying ``inlet_names``, for errors.
        require_any: Whether at least one fuel source is required.

    Raises:
        FlexConfigError: If a name is empty or repeated, the two lists
            overlap, or no source is given when one is required.
    """
    utility_names = utility_names or ()
    if not inlet_names and not utility_names and require_any:
        raise FlexConfigError(
            f"{inlet_field} is empty and no utility_fuel_source is given; "
            "pass utility_fuel_source to run without inlet ports, or "
            "pass one or more inlet names.",
            field=inlet_field,
            value=inlet_names,
        )
    if not all(isinstance(n, str) and n for n in inlet_names):
        raise FlexConfigError(
            f"{inlet_field} must be one or more non-empty strings, got "
            f"{inlet_names!r}.",
            field=inlet_field,
            value=inlet_names,
        )
    if len(set(inlet_names)) != len(inlet_names):
        raise FlexConfigError(
            f"{inlet_field} must be unique, got {inlet_names!r}.",
            field=inlet_field,
            value=inlet_names,
        )
    if len(set(utility_names)) != len(utility_names):
        raise FlexConfigError(
            "utility_fuel_source names must be unique, "
            f"got {list(utility_names)!r}.",
            field="utility_fuel_source",
            value=utility_names,
        )
    overlap = set(utility_names) & set(inlet_names)
    if overlap:
        raise FlexConfigError(
            f"utility_fuel_source names must not overlap with {inlet_field}; "
            f"duplicate(s): {sorted(overlap)}.",
            field="utility_fuel_source",
            value=utility_names,
        )


def heating_value_mismatch(
    heating_values: Mapping | None, fuel_names: set[str]
) -> tuple[list[str], list[str]]:
    """Return the fuels missing a heating value and the unknown names given one.

    Args:
        heating_values: Mapping of fuel name to heating value, or ``None``.
        fuel_names: Every configured fuel source name.

    Returns:
        ``(missing, unknown)``, each sorted.
    """
    given = set(heating_values or {})
    return sorted(fuel_names - given), sorted(given - fuel_names)
