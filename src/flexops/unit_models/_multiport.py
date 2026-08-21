"""Shared checks for the config-driven-arity flow junctions (architecture §3.4).

:class:`~flexops.unit_models.mixer.Mixer` and
:class:`~flexops.unit_models.splitter.Splitter` both take their port count from
a config option and both share one property package across every port, so both
need the same two checks: that the configured port role names are usable, and
that the package carries exactly one phase for the flow balance to sum over.

Neither check can live in a ``ConfigValue`` domain. Pyomo's ``ConfigValue``
wraps any exception a domain raises into a plain ``ValueError``, discarding the
``field``/``value`` a :class:`~flexcore.exceptions.FlexConfigError` carries — so
these are called from ``build()`` instead, and the port-name config domains do
type coercion only.
"""

from collections.abc import Sequence

from flexcore.exceptions import FlexConfigError


def validate_port_names(names: Sequence[str], field: str) -> None:
    """Reject empty, non-string, or duplicated port role names.

    Args:
        names: The configured role names; each becomes one port.
        field: The config option's name, reported on the raised error.

    Raises:
        FlexConfigError: If ``names`` is empty, holds a non-string or
            empty-string entry, or repeats a name.
    """
    if not names or not all(isinstance(name, str) and name for name in names):
        raise FlexConfigError(
            f"{field} must be one or more non-empty strings, got {names!r}.",
            field=field,
            value=names,
        )
    if len(set(names)) != len(names):
        raise FlexConfigError(
            f"{field} must be unique, got {names!r}; each name becomes its own "
            "port, so two ports cannot share one.",
            field=field,
            value=names,
        )


def single_flow_phase(pkg, class_name: str) -> str:
    """Return the one phase ``pkg`` carries, for the flow balance to sum over.

    Args:
        pkg: The unit's configured property (parameter) block.
        class_name: The calling unit model's class name, for the error message.

    Returns:
        The name of the package's single phase.

    Raises:
        FlexConfigError: If ``pkg`` does not have exactly one phase — a
            multi-phase basis needs a per-phase balance this junction does not
            write.
    """
    phases = list(pkg.phase_list)
    if len(phases) != 1:
        raise FlexConfigError(
            f"{class_name} requires a property_package with exactly one phase "
            f"(its balance sums a single flow per stream); got "
            f"phase_list={phases!r}.",
            field="property_package",
            value=pkg,
        )
    return phases[0]
