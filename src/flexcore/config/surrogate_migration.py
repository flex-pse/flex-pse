"""Translate pre-0.2.0 surrogate specs to the ``config`` / ``parameters`` form.

Before schema 0.2.0 a surrogate carried one ``data`` dict whose layout each
surrogate type chose for itself, with coefficient values mixed into it and no
record of which coefficients were fixed. From 0.2.0 every surrogate carries
``config`` (everything except fitted values, defined by its class) and
``parameters`` (``{name: {"value": ..., "fixed": ...}}``, keyed by the names its
class registers on the built block).

This module holds the one-time translation of the three surrogate types that
existed before 0.2.0. It reproduces what the old build did, so a migrated spec
builds the same model: every coefficient the old build created is listed,
fixed, at the value the old build gave it (including regression-mode initial
guesses). It deliberately knows nothing about later formats; each surrogate
class in ``flexops.surrogates`` migrates its own ``config`` between
``config_version`` values.
"""

import copy
from typing import Any

from flexcore.exceptions import FlexConfigError

_INTERCEPT = "intercept"
_INTENSITY = "energy_intensity"


def _fixed(value: Any) -> dict[str, Any]:
    return {"value": value, "fixed": True}


def _variables(data: dict) -> dict[str, Any]:
    return {
        key: copy.deepcopy(data[key])
        for key in ("input_variables", "output_variables")
        if key in data
    }


def _multilinear(data: dict) -> tuple[dict, dict]:
    coefficients = data.get("coefficients")
    if not isinstance(coefficients, dict):
        return {**_variables(data), "terms": []}, {}
    terms = [key for key in coefficients if key != _INTERCEPT]
    parameters = {_INTERCEPT: _fixed(coefficients.get(_INTERCEPT, 0.0))}
    parameters.update({key: _fixed(coefficients[key]) for key in terms})
    return {**_variables(data), "terms": terms}, parameters


def _constant_intensity(data: dict) -> tuple[dict, dict]:
    coefficients = data.get("coefficients")
    parameters = {}
    if isinstance(coefficients, dict) and _INTENSITY in coefficients:
        parameters[_INTENSITY] = _fixed(coefficients[_INTENSITY])
    return _variables(data), parameters


def _arima_exog(value: Any, inputs: list[str]) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [(name, value[name]) for name in inputs if name in value]
    if isinstance(value, list):
        if len(inputs) > 1:
            suggestion = dict(zip(inputs, value, strict=False))
            raise FlexConfigError(
                f"Cannot migrate the ARIMA 'exog_coefs' list {value!r}: with "
                f"{len(inputs)} inputs a list pairs coefficients with inputs by "
                f"the key order of 'input_variables', which JSON does not "
                f"preserve. Rewrite it as a mapping from input name to "
                f"coefficient; under the current key order that is "
                f"{suggestion!r}.",
                field="data.coefficients.exog_coefs",
                value=value,
            )
        return list(zip(inputs, value, strict=False))
    return []


def _arima(data: dict) -> tuple[dict, dict]:
    coefficients = data.get("coefficients")
    if not isinstance(coefficients, dict):
        coefficients = {}
    order = coefficients.get("order")
    if (
        not isinstance(order, (list, tuple))
        or len(order) != 3
        or not all(type(value) is int and value >= 0 for value in order)
    ):
        raise FlexConfigError(
            "Cannot migrate an ARIMA surrogate whose coefficients['order'] is "
            f"not [p, d, q] with three non-negative integers; got {order!r}.",
            field="data.coefficients.order",
            value=order,
        )
    p, _d, q = order
    inputs = data.get("input_variables")
    input_names = list(inputs) if isinstance(inputs, dict) else []
    regression_mode = "history" not in data

    deterministic = next(
        (key for key in ("intercept", "drift") if key in coefficients), None
    )
    config: dict[str, Any] = {"order": list(order), "deterministic": deterministic}
    config.update(_variables(data))
    if not regression_mode:
        config["history"] = copy.deepcopy(data["history"])

    # The old build fixed every coefficient, using these initial guesses for
    # the ones regression mode left out.
    ar_coefs = coefficients.get("ar_coefs", [1.0] * p if regression_mode else [])
    ma_coefs = coefficients.get("ma_coefs", [1.0] * q if regression_mode else [])
    exog_coefs = coefficients.get(
        "exog_coefs", dict.fromkeys(input_names, 1.0) if regression_mode else {}
    )

    parameters: dict[str, Any] = {}
    if deterministic is not None:
        parameters[deterministic] = _fixed(coefficients[deterministic])
    for prefix, values in (("ar", ar_coefs), ("ma", ma_coefs)):
        if isinstance(values, list):
            for lag, value in enumerate(values, start=1):
                parameters[f"{prefix}.L{lag}"] = _fixed(value)
    for name, value in _arima_exog(exog_coefs, input_names):
        parameters[f"exog.{name}"] = _fixed(value)
    return config, parameters


_TRANSLATORS = {
    "multilinear": _multilinear,
    "constant_intensity": _constant_intensity,
    "arima": _arima,
}


def is_legacy_surrogate(spec: dict) -> bool:
    """Return whether ``spec`` uses the pre-0.2.0 ``data`` layout.

    Args:
        spec: A surrogate spec or surrogate element, as a parsed dict.

    Returns:
        ``True`` when it carries ``data`` instead of ``config``/``parameters``.
    """
    return isinstance(spec, dict) and "data" in spec


def migrate_surrogate_spec(spec: dict) -> dict:
    """Return a pre-0.2.0 surrogate spec or element in the 0.2.0 form.

    ``data`` is split into ``config`` and ``parameters`` and
    ``config_version`` is set to 1; every other key (``surrogate_type``,
    ``provenance``, and an element's ``kind``, ``unit``, ``relation`` and
    ``name``) is kept. A spec already in the 0.2.0 form is returned as an
    unchanged copy, so the function is safe to call on either. Types without
    a pre-0.2.0 implementation keep their ``data`` as ``config``.

    Args:
        spec: A surrogate spec (e.g. saved from ``to_surrogate_spec()``) or a
            flowsheet surrogate element, as a parsed dict. Not modified.

    Returns:
        The migrated copy.

    Raises:
        FlexConfigError: If the old data cannot be translated faithfully, such
            as an ARIMA ``exog_coefs`` list with several inputs, which cannot
            be paired with its inputs reliably.
    """
    migrated = copy.deepcopy(spec)
    if not is_legacy_surrogate(migrated):
        return migrated
    data = migrated.pop("data")
    if not isinstance(data, dict):
        raise FlexConfigError(
            f"Cannot migrate a surrogate whose 'data' is not a mapping; got {data!r}.",
            field="data",
            value=data,
        )
    translate = _TRANSLATORS.get(migrated.get("surrogate_type"))
    if translate is None:
        config, parameters = copy.deepcopy(data), {}
    else:
        config, parameters = translate(data)
    migrated["config_version"] = 1
    migrated["config"] = config
    migrated["parameters"] = parameters
    return migrated
