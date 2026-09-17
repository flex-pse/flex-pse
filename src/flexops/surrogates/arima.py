"""Pyomo ARIMA/ARIMAX surrogate with explicit innovations and local state.

The surrogate implements one non-seasonal ``ARIMA(p, d, q)`` equation for
``d`` equal to zero or one. Its persisted data contains fitted coefficients
and only the state needed immediately before the local Pyomo horizon::

    {
        "input_variables": {"feed": "kg/hr"},
        "output_variables": {"production": "m^3/hr"},
        "coefficients": {
            "order": [p, d, q],
            "intercept": 0.0,       # optional, d=0 only
            "drift": 0.0,           # optional, d=1 only
            "ar_coefs": [...],      # length p
            "ma_coefs": [...],      # length q
            "exog_coefs": [...],    # one per input variable
        },
        "history": {
            "start_date": "2025-01-01T00:00:00",
            "time_step_seconds": 900.0,
            "y_values": [...],
            "eps_values": [...],
        },
    }

For ``z[t] = y[t]`` when ``d=0`` and
``z[t] = y[t] - y[t-1]`` when ``d=1``, the equation is::

    z[t] = deterministic
           + sum(phi[i] * z[t-i])
           + sum(theta[j] * eps[t-j])
           + sum(beta[k] * x[t,k])
           + eps[t]

``OpsBlock.swap_relation`` supplies the single ``target[t] == body(t)``
constraint. Consequently, ``body(t)`` includes the current innovation and
this surrogate adds no separate residual constraint.

Coefficient values are stored as dimensionless magnitudes in their declared
data basis, matching other flex-pse surrogates. ``intercept`` and ``drift``
therefore represent output-unit magnitudes; each exogenous coefficient
represents output units per declared input unit.
"""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Mapping
from numbers import Real
from types import MethodType
from typing import ClassVar

import pyomo.environ as pyo
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import CoefficientRegistry
from flexops.core.time_block import find_time_block
from flexops.core.units import parse_units
from flexops.surrogates.base import Surrogate

_DATA_KEYS = (
    "input_variables",
    "output_variables",
    "coefficients",
    "history",
)
_REQUIRED_DATA_KEYS = _DATA_KEYS[:3]
_COEFFICIENT_KEYS = {
    "order",
    "intercept",
    "drift",
    "ar_coefs",
    "ma_coefs",
    "exog_coefs",
}
_HISTORY_KEYS = {
    "start_date",
    "time_step_seconds",
    "y_values",
    "eps_values",
}


def _require_numeric_list(
    value: object,
    *,
    field: str,
    expected_length: int,
) -> list[float]:
    """Validate and normalize a fixed-length list of numeric magnitudes.

    Args:
        value: The candidate list.
        field: Field name used in error messages.
        expected_length: Exact required length.

    Returns:
        The values as a list of floats.

    Raises:
        FlexConfigError: If ``value`` is not a list of exactly
            ``expected_length`` numbers.
    """
    if not isinstance(value, list) or len(value) != expected_length:
        raise FlexConfigError(
            f"ARIMA {field!r} must be a list of {expected_length} numbers, "
            f"got {value!r}.",
            field=field,
            value=value,
        )
    result: list[float] = []
    for index, item in enumerate(value):
        try:
            result.append(float(item))
        except (TypeError, ValueError) as exc:
            raise FlexConfigError(
                f"ARIMA {field}[{index}] must be a number, got {item!r}.",
                field=field,
                value=item,
            ) from exc
    return result


def _scale_for_time(block, innovation_scale, t):
    """Return one positive innovation scale carrying declared output units.

    Args:
        block: Bound ARIMA surrogate block, used for its output units.
        innovation_scale: A scalar, a Pyomo component, or a mapping/indexed
            component keyed by time. Bare numbers are read as magnitudes in
            the declared output units.
        t: Time index to resolve, used for indexed scales and messages.

    Returns:
        The scale as a quantity in the block's declared output units.

    Raises:
        FlexConfigError: If an indexed scale has no entry for ``t``, or the
            resolved scale is unit-incompatible, nonnumeric, or nonpositive.
    """
    indexed = isinstance(innovation_scale, Mapping) or (
        hasattr(innovation_scale, "is_indexed") and innovation_scale.is_indexed()
    )
    if indexed:
        try:
            raw_scale = innovation_scale[t]
        except (KeyError, IndexError, TypeError) as exc:
            raise FlexConfigError(
                f"ARIMA innovation_scale has no value for time index {t!r}.",
                field="innovation_scale",
                value=t,
            ) from exc
    else:
        raw_scale = innovation_scale

    output_units = block._output_units
    if isinstance(raw_scale, Real):
        scale = float(raw_scale) * output_units
    else:
        try:
            scale = pyunits.convert(raw_scale, output_units)
        except (TypeError, ValueError, UnitsError) as exc:
            raise FlexConfigError(
                f"ARIMA innovation_scale at time {t!r} must have units "
                f"compatible with {output_units!s}, got {raw_scale!r}.",
                field="innovation_scale",
                value=raw_scale,
            ) from exc

    try:
        magnitude = float(pyo.value(scale / output_units))
    except (TypeError, ValueError) as exc:
        raise FlexConfigError(
            f"ARIMA innovation_scale at time {t!r} must have a numeric value, "
            f"got {raw_scale!r}.",
            field="innovation_scale",
            value=raw_scale,
        ) from exc
    if magnitude <= 0:
        raise FlexConfigError(
            f"ARIMA innovation_scale at time {t!r} must be positive, "
            f"got {magnitude}.",
            field="innovation_scale",
            value=magnitude,
        )
    return scale


def _get_regression_objective(
    block,
    innovation_scale=None,
    time_index=None,
):
    """Return this block's dimensionless innovation SSE expression.

    Args:
        block: Bound ARIMA surrogate block.
        innovation_scale: Positive scalar or time-indexed scale. Bare numbers
            use the declared output units. ``None`` defaults to one output
            unit and emits ``UserWarning``.
        time_index: Optional subset of local time indices. Defaults to every
            time index on the block.

    Returns:
        A composable Pyomo numeric expression. No ``Objective`` is created and
        no variable fixation changes.

    Raises:
        FlexConfigError: If an index is unknown or a scale is missing,
            nonpositive, nonnumeric, or unit-incompatible.
    """
    selected = list(block._time_values if time_index is None else time_index)
    if not selected:
        raise FlexConfigError(
            "ARIMA regression time_index must contain at least one index.",
            field="time_index",
            value=selected,
        )
    unknown = [t for t in selected if t not in block.time]
    if unknown:
        raise FlexConfigError(
            f"ARIMA regression time_index contains unknown indices {unknown}; "
            f"known indices are {list(block._time_values)}.",
            field="time_index",
            value=unknown,
        )
    if innovation_scale is None:
        warnings.warn(
            "ARIMA regression objective is defaulting innovation_scale to "
            f"1.0 {block._output_units!s}. This is a numerical weighting, not "
            "an estimated innovation standard deviation. Pass "
            "innovation_scale explicitly to control its weight relative to "
            "other objective terms.",
            UserWarning,
            stacklevel=2,
        )
        innovation_scale = 1.0

    return sum(
        (block.eps[t] / _scale_for_time(block, innovation_scale, t)) ** 2
        for t in selected
    )


class ArimaSurrogate(Surrogate):
    """Non-seasonal ARIMA/ARIMAX relationship with explicit innovations."""

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.ARIMA

    def __init__(self, data: dict, *, max_ar_coeff: float | None = None) -> None:
        """Store and validate ``data``, with an optional AR stability bound.

        Args:
            data: The relationship's data (see the module docstring).
            max_ar_coeff: Bounds every ``ar_coefs`` entry to
                ``[-max_ar_coeff, max_ar_coeff]``. A supplied (fixed)
                coefficient that already exceeds this raises immediately;
                an unfixed one (regression mode) is bounded during solving.
                AR coefficients near the unit root make this surrogate's
                difference-equation form numerically unstable. ``None``
                (the default) applies no bound.

        Raises:
            FlexConfigError: If ``data`` does not match this class's
                contract, or ``max_ar_coeff`` is not a number in ``(0, 1]``.
        """
        if max_ar_coeff is not None and (
            not isinstance(max_ar_coeff, (int, float))
            or max_ar_coeff <= 0
            or max_ar_coeff > 1
        ):
            raise FlexConfigError(
                f"ArimaSurrogate max_ar_coeff must be a number in (0, 1] or "
                f"None, got {max_ar_coeff!r}.",
                field="max_ar_coeff",
                value=max_ar_coeff,
            )
        self._max_ar_coeff = max_ar_coeff
        super().__init__(data)

    def _validate(self) -> None:
        """Validate structural coefficients, units, and pre-horizon state."""
        unknown = sorted(set(self.data) - set(_DATA_KEYS))
        if unknown:
            raise FlexConfigError(
                f"ARIMA surrogate data carries unknown key(s) {unknown}; "
                f"it may only have {_DATA_KEYS}.",
                field="data",
                value=unknown,
            )
        missing = sorted(set(_REQUIRED_DATA_KEYS) - set(self.data))
        if missing:
            raise FlexConfigError(
                f"ARIMA surrogate data is missing key(s) {missing}; it must "
                f"have {_REQUIRED_DATA_KEYS}.",
                field="data",
                value=missing,
            )

        inputs = self.data["input_variables"]
        if not isinstance(inputs, dict):
            raise FlexConfigError(
                "ARIMA surrogate 'input_variables' must be a {name: units} "
                f"mapping, got {inputs!r}.",
                field="input_variables",
                value=inputs,
            )
        outputs = self.data["output_variables"]
        if not isinstance(outputs, dict) or len(outputs) != 1:
            raise FlexConfigError(
                "ARIMA surrogate 'output_variables' must be one {name: units} "
                f"entry, got {outputs!r}.",
                field="output_variables",
                value=outputs,
            )
        for field, mapping in (
            ("input_variables", inputs),
            ("output_variables", outputs),
        ):
            for name, units in mapping.items():
                if not isinstance(name, str) or not name:
                    raise FlexConfigError(
                        f"ARIMA surrogate {field!r} has invalid name {name!r}.",
                        field=field,
                        value=name,
                    )
                if not isinstance(units, str) or not units:
                    raise FlexConfigError(
                        f"ARIMA surrogate {field!r} entry {name!r} must map "
                        f"to a non-empty units string, got {units!r}.",
                        field=field,
                        value=units,
                    )
                parse_units(units)

        coefficients = self.data["coefficients"]
        if not isinstance(coefficients, dict):
            raise FlexConfigError(
                "ARIMA surrogate 'coefficients' must be a mapping, "
                f"got {coefficients!r}.",
                field="coefficients",
                value=coefficients,
            )
        unknown_coefficients = sorted(set(coefficients) - _COEFFICIENT_KEYS)
        if unknown_coefficients:
            raise FlexConfigError(
                f"ARIMA coefficients carry unknown key(s) "
                f"{unknown_coefficients}; allowed keys are "
                f"{sorted(_COEFFICIENT_KEYS)}.",
                field="coefficients",
                value=unknown_coefficients,
            )

        order = coefficients.get("order")
        if (
            not isinstance(order, (list, tuple))
            or len(order) != 3
            or not all(type(value) is int and value >= 0 for value in order)
        ):
            raise FlexConfigError(
                "ARIMA coefficients['order'] must be [p, d, q] with three "
                f"non-negative integers, got {order!r}.",
                field="coefficients.order",
                value=order,
            )
        p, d, q = order
        if d not in (0, 1):
            raise FlexConfigError(
                f"ARIMA surrogate supports only d=0 or d=1, got d={d}.",
                field="coefficients.order",
                value=order,
            )
        if d == 1 and "intercept" in coefficients:
            raise FlexConfigError(
                "ARIMA coefficient 'intercept' is invalid for d=1; use "
                "'drift' for a constant in the differenced equation.",
                field="coefficients.intercept",
                value=coefficients["intercept"],
            )
        if d == 0 and "drift" in coefficients:
            raise FlexConfigError(
                "ARIMA coefficient 'drift' is invalid for d=0; use "
                "'intercept' for the level equation.",
                field="coefficients.drift",
                value=coefficients["drift"],
            )
        deterministic = "intercept" if d == 0 else "drift"
        if deterministic in coefficients:
            try:
                float(coefficients[deterministic])
            except (TypeError, ValueError) as exc:
                raise FlexConfigError(
                    f"ARIMA coefficient {deterministic!r} must be a number, "
                    f"got {coefficients[deterministic]!r}.",
                    field=f"coefficients.{deterministic}",
                    value=coefficients[deterministic],
                ) from exc

        regression_mode = "history" not in self.data
        ar_coefs = _require_numeric_list(
            coefficients.get("ar_coefs", [1.0] * p if regression_mode else []),
            field="ar_coefs",
            expected_length=p,
        )
        if self._max_ar_coeff is not None and "ar_coefs" in coefficients:
            # Only explicitly supplied (fixed) coefficients are checked here;
            # regression mode's placeholder initial guess is unfixed and
            # bounded during solving instead (see build()).
            for value in ar_coefs:
                if abs(value) > self._max_ar_coeff:
                    raise FlexConfigError(
                        f"ARIMA ar_coefs entry {value} exceeds "
                        f"max_ar_coeff={self._max_ar_coeff}; AR coefficients "
                        f"near the unit root are unstable in this "
                        f"surrogate's difference-equation form.",
                        field="coefficients.ar_coefs",
                        value=value,
                    )
        _require_numeric_list(
            coefficients.get("ma_coefs", [1.0] * q if regression_mode else []),
            field="ma_coefs",
            expected_length=q,
        )
        _require_numeric_list(
            coefficients.get(
                "exog_coefs", [1.0] * len(inputs) if regression_mode else []
            ),
            field="exog_coefs",
            expected_length=len(inputs),
        )

        if "history" in self.data:
            history = self.data["history"]
            if not isinstance(history, dict):
                raise FlexConfigError(
                    "ARIMA history must be a mapping.",
                    field="history",
                    value=history,
                )
            missing_history = sorted(_HISTORY_KEYS - set(history))
            unknown_history = sorted(set(history) - _HISTORY_KEYS)
            if missing_history or unknown_history:
                raise FlexConfigError(
                    "ARIMA history must contain exactly the required keys; "
                    f"missing={missing_history}, unknown={unknown_history}.",
                    field="history",
                    value=history,
                )
            try:
                datetime.datetime.fromisoformat(history["start_date"])
                step_seconds = float(history["time_step_seconds"])
            except (TypeError, ValueError) as exc:
                raise FlexConfigError(
                    "ARIMA history start_date must be ISO-8601 and "
                    "time_step_seconds must be numeric.",
                    field="history",
                    value=history,
                ) from exc
            if step_seconds <= 0:
                raise FlexConfigError(
                    "ARIMA history time_step_seconds must be positive.",
                    field="history.time_step_seconds",
                    value=step_seconds,
                )
            _require_numeric_list(
                history["y_values"],
                field="history.y_values",
                expected_length=len(history["y_values"]),
            )
            _require_numeric_list(
                history["eps_values"],
                field="history.eps_values",
                expected_length=len(history["eps_values"]),
            )
            # Both series are stored with the same `max(p + d, q)` prefix
            # so that a shared offset indexes into either one; build() slices
            # `p + d` levels and `q` innovations out of that prefix.
            history_prefix = max(p + d, q)
            for field_name in ("y_values", "eps_values"):
                if len(history[field_name]) < history_prefix:
                    raise FlexConfigError(
                        f"ARIMA history.{field_name} must include at least "
                        f"max(p+d, q)={history_prefix} value(s) before the "
                        f"first modeled point, got "
                        f"{len(history[field_name])}.",
                        field=f"history.{field_name}",
                        value=len(history[field_name]),
                    )

    @property
    def input_variables(self) -> dict[str, str]:
        """Return exogenous input names and declared units."""
        return dict(self.data["input_variables"])

    @property
    def output_variables(self) -> dict[str, str]:
        """Return the single output name and declared units."""
        return dict(self.data["output_variables"])

    def build(self, unit, target):
        """Build coefficient, innovation, and initial-state components.

        The returned ``body(t)`` is the complete right-hand side consumed by
        :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`. All
        coefficients and current innovations are fixed at build time. When
        ``history`` is supplied, the pre-horizon state it implies is also
        fixed. When ``history`` is omitted, the pre-horizon state is
        initialized to zero and left free for in-model estimation. Regression
        callers explicitly unfix any other quantities they intend to estimate.

        Args:
            unit: Unit owning the variables named by ``input_variables``.
            target: Time-indexed output variable representing ``y[t]``.

        Returns:
            ``(block, body)`` where ``body(t)`` carries declared output units.

        Raises:
            FlexConfigError: If ``history`` does not align with the
                ``TimeBlock`` start date on the model's step grid, if its
                ``y_values`` and ``eps_values`` imply different numbers of
                modeled points, or if an input variable's model units are
                incompatible with the units it declares.
        """
        coefficients = self.data["coefficients"]
        p, d, q = (int(value) for value in coefficients["order"])
        state_provided = "history" in self.data
        history = self.data.get("history")
        if history is not None:
            time_block = find_time_block(unit.model())
            current_start = time_block.datetime_index[0].to_pydatetime()
            training_start = datetime.datetime.fromisoformat(history["start_date"])
            step_seconds = float(history["time_step_seconds"])
            offset = (current_start - training_start).total_seconds() / step_seconds
            offset_index = round(offset)
            if offset < 0 or abs(offset - offset_index) > 1e-8:
                raise FlexConfigError(
                    "ARIMA history does not align with the TimeBlock start date; "
                    f"got {current_start.isoformat()} for training start "
                    f"{training_start.isoformat()} and step {step_seconds}s.",
                    field="history.start_date",
                    value=current_start,
                )
            history_prefix = max(p + d, q)
            y_values = [float(value) for value in history["y_values"]]
            eps_values = [float(value) for value in history["eps_values"]]
            actual_count = len(y_values) - history_prefix
            if len(eps_values) - history_prefix != actual_count:
                raise FlexConfigError(
                    "ARIMA history y_values and eps_values must contain the "
                    "same number of modeled points.",
                    field="history",
                    value=history,
                )
            if offset_index > actual_count:
                intercept = float(coefficients.get("intercept", 0.0))
                drift = float(coefficients.get("drift", 0.0))
                ar = [float(value) for value in coefficients.get("ar_coefs", [])]
                ma = [float(value) for value in coefficients.get("ma_coefs", [])]
                for position in range(actual_count, offset_index):
                    y_index = history_prefix + position
                    eps_index = history_prefix + position
                    ar_part = sum(
                        (
                            ar[lag - 1] * y_values[y_index - lag]
                            if d == 0
                            else ar[lag - 1]
                            * (y_values[y_index - lag] - y_values[y_index - lag - 1])
                        )
                        for lag in range(1, p + 1)
                    )
                    ma_part = sum(
                        ma[lag - 1] * eps_values[eps_index - lag]
                        for lag in range(1, q + 1)
                    )
                    mean = (intercept if d == 0 else drift) + ar_part + ma_part
                    y_values.append(mean if d == 0 else y_values[-1] + mean)
                    eps_values.append(0.0)
            y_history = [
                float(value)
                for value in y_values[
                    history_prefix
                    + offset_index
                    - (p + d) : history_prefix
                    + offset_index
                ]
            ]
            eps_history = [
                float(value)
                for value in eps_values[
                    history_prefix + offset_index - q : history_prefix + offset_index
                ]
            ]
        else:
            y_history = [0.0] * (p + d)
            eps_history = [0.0] * q
        exog_names = list(self.input_variables)
        exog_coefs = [
            float(value)
            for value in coefficients.get(
                "exog_coefs", [] if state_provided else [1.0] * len(exog_names)
            )
        ]
        output_units = parse_units(next(iter(self.output_variables.values())))
        time_values = list(find_time_block(unit.model()).time_index)
        time_positions = {value: position for position, value in enumerate(time_values)}

        exogenous = []
        for name, units_string in self.input_variables.items():
            exogenous.append(
                (
                    unit.resolve_variable(name, field="input_variables"),
                    parse_units(units_string),
                )
            )

        block = pyo.Block(concrete=True)
        block.time = pyo.Set(initialize=time_values, ordered=True)
        coefficient_vars: dict[str, object] = {}

        deterministic_name = "intercept" if d == 0 else "drift"
        if deterministic_name in coefficients:
            block.add_component(
                deterministic_name,
                pyo.Var(
                    initialize=float(coefficients[deterministic_name]),
                    doc=(
                        "ARIMA level intercept magnitude"
                        if d == 0
                        else "ARIMA constant drift magnitude in differenced output"
                    ),
                ),
            )
            deterministic_var = block.find_component(deterministic_name)
            deterministic_var.fix(float(coefficients[deterministic_name]))
            coefficient_vars[deterministic_name] = deterministic_var

        if p > 0:
            ar_default = (
                1.0 if self._max_ar_coeff is None else min(1.0, self._max_ar_coeff)
            )
            ar_values = [
                float(value)
                for value in coefficients.get(
                    "ar_coefs", [] if state_provided else [ar_default] * p
                )
            ]
            block.ar_coefs = pyo.Var(
                range(1, p + 1),
                initialize={index: ar_values[index - 1] for index in range(1, p + 1)},
                bounds=(
                    (None, None)
                    if self._max_ar_coeff is None
                    else (-self._max_ar_coeff, self._max_ar_coeff)
                ),
                doc="Dimensionless autoregressive coefficients",
            )
            for index, value in enumerate(ar_values, start=1):
                block.ar_coefs[index].fix(value)
            coefficient_vars["ar_coefs"] = block.ar_coefs

        if q > 0:
            ma_values = [
                float(value)
                for value in coefficients.get(
                    "ma_coefs", [] if state_provided else [1.0] * q
                )
            ]
            block.ma_coefs = pyo.Var(
                range(1, q + 1),
                initialize={index: ma_values[index - 1] for index in range(1, q + 1)},
                doc="Dimensionless moving-average coefficients",
            )
            for index, value in enumerate(ma_values, start=1):
                block.ma_coefs[index].fix(value)
            coefficient_vars["ma_coefs"] = block.ma_coefs

        if exog_names:
            block.exog_coefs = pyo.Var(
                range(1, len(exog_names) + 1),
                initialize={
                    index: exog_coefs[index - 1]
                    for index in range(1, len(exog_names) + 1)
                },
                doc="Exogenous coefficient magnitudes in declared data units",
            )
            for index, value in enumerate(exog_coefs, start=1):
                block.exog_coefs[index].fix(value)
            coefficient_vars["exog_coefs"] = block.exog_coefs

        block.coefficient_vars = coefficient_vars
        block.coefficients = CoefficientRegistry()
        if deterministic_name in coefficient_vars:
            block.coefficients.register_coefficient(
                deterministic_name, coefficient_vars[deterministic_name]
            )
        if p > 0:
            for index in range(1, p + 1):
                block.coefficients.register_coefficient(
                    f"ar.L{index}", block.ar_coefs[index]
                )
        if q > 0:
            for index in range(1, q + 1):
                block.coefficients.register_coefficient(
                    f"ma.L{index}", block.ma_coefs[index]
                )
        for index, name in enumerate(exog_names, start=1):
            block.coefficients.register_coefficient(
                f"exog.{name}", block.exog_coefs[index]
            )

        block.y_history_index = pyo.Set(initialize=range(len(y_history)), ordered=True)
        block.initial_y_history = pyo.Var(
            block.y_history_index,
            initialize={index: value for index, value in enumerate(y_history)},
            units=output_units,
            doc="Pre-horizon output levels, oldest to newest",
        )
        if state_provided:
            for index, value in enumerate(y_history):
                block.initial_y_history[index].fix(value)

        block.eps_history_index = pyo.Set(
            initialize=range(len(eps_history)), ordered=True
        )
        block.initial_eps_history = pyo.Var(
            block.eps_history_index,
            initialize={index: value for index, value in enumerate(eps_history)},
            units=output_units,
            doc="Pre-horizon innovations, oldest to newest",
        )
        if state_provided:
            for index, value in enumerate(eps_history):
                block.initial_eps_history[index].fix(value)

        block.eps = pyo.Var(
            block.time,
            initialize=0.0,
            units=output_units,
            doc="Current-horizon ARIMA innovations",
        )
        block.eps.fix(0.0)

        block._order = (p, d, q)
        block._exog_names = exog_names
        block._output_units = output_units
        block._time_values = tuple(time_values)
        block.get_regression_objective = MethodType(_get_regression_objective, block)

        def exog_at(position: int):
            """Return the level exogenous regression contribution."""
            t = time_values[position]
            val = 0.0 * output_units
            for index, (variable, declared_units) in enumerate(exogenous, start=1):
                try:
                    normalized_input = (
                        pyunits.convert(variable[t], declared_units) / declared_units
                    )
                except UnitsError as exc:
                    raise FlexConfigError(
                        f"ARIMA input {exog_names[index - 1]!r} declares "
                        f"{declared_units!s}, incompatible with its model units "
                        f"{pyunits.get_units(variable[t])!s}.",
                        field="input_variables",
                        value=exog_names[index - 1],
                    ) from exc
                val += block.exog_coefs[index] * normalized_input * output_units
            return val

        def eta_at(position: int):
            """Return disturbance level from local horizon or pre-horizon state."""
            if position >= 0:
                y_val = pyunits.convert(target[time_values[position]], output_units)
                return y_val - exog_at(position)
            history_index = len(y_history) + position
            if history_index < 0:
                raise FlexConfigError(
                    f"ARIMA disturbance lag at local position {position} exceeds "
                    f"the {len(y_history)} stored y_history values.",
                    field="history.y_values",
                    value=position,
                )
            return block.initial_y_history[history_index]

        def difference_at(position: int):
            """Return first difference of disturbance at one position."""
            return eta_at(position) - eta_at(position - 1)

        def innovation_at(position: int):
            """Return an innovation from the horizon or pre-horizon state."""
            if position >= 0:
                return block.eps[time_values[position]]
            history_index = len(eps_history) + position
            if history_index < 0:
                raise FlexConfigError(
                    f"ARIMA innovation lag at local position {position} "
                    f"exceeds the {len(eps_history)} stored eps_history values.",
                    field="history.eps_values",
                    value=position,
                )
            return block.initial_eps_history[history_index]

        def body(t):
            position = time_positions[t]
            current_exog = exog_at(position)
            mean = 0.0 * output_units
            if deterministic_name in coefficient_vars:
                mean += coefficient_vars[deterministic_name] * output_units
            if p > 0:
                for lag in range(1, p + 1):
                    lagged = (
                        eta_at(position - lag)
                        if d == 0
                        else difference_at(position - lag)
                    )
                    mean += block.ar_coefs[lag] * lagged
            if q > 0:
                for lag in range(1, q + 1):
                    mean += block.ma_coefs[lag] * innovation_at(position - lag)

            current = block.eps[t]
            if d == 0:
                return current_exog + mean + current
            return current_exog + eta_at(position - 1) + mean + current

        return block, body

    def get_surrogate_spec(self, block, target) -> dict:
        """Extract coefficients and the complete dated solved history.

        The returned ``history`` spans the whole local horizon.

        Args:
            block: The block returned by :meth:`build`, after a solve.
            target: The time-indexed output variable passed to
                :meth:`build`.

        Returns:
            A ``data`` dict in this class's persisted contract, suitable for
            constructing a fresh :class:`ArimaSurrogate` for the next
            horizon.
        """
        p, d, q = block._order
        coefficients: dict[str, object] = {"order": [p, d, q]}
        if p > 0:
            coefficients["ar_coefs"] = [
                float(pyo.value(block.ar_coefs[index])) for index in range(1, p + 1)
            ]
        if q > 0:
            coefficients["ma_coefs"] = [
                float(pyo.value(block.ma_coefs[index])) for index in range(1, q + 1)
            ]
        if len(block._exog_names) > 0:
            coefficients["exog_coefs"] = [
                float(pyo.value(block.exog_coefs[index]))
                for index in range(1, len(block._exog_names) + 1)
            ]
        deterministic_name = "intercept" if d == 0 else "drift"
        deterministic_var = block.find_component(deterministic_name)
        if deterministic_var is not None:
            coefficients[deterministic_name] = float(pyo.value(deterministic_var))

        output_units = block._output_units
        unit = target.parent_block()
        exogenous = [
            (
                unit.resolve_variable(name, field="input_variables"),
                parse_units(self.input_variables[name]),
            )
            for name in block._exog_names
        ]

        def exog_val(t):
            val = 0.0 * output_units
            for index, (variable, declared_units) in enumerate(exogenous, start=1):
                normalized_input = (
                    pyunits.convert(variable[t], declared_units) / declared_units
                )
                val += block.exog_coefs[index] * normalized_input * output_units
            return val

        y_values = [
            float(
                pyo.value(
                    pyunits.convert(target[t] - exog_val(t), output_units)
                    / output_units
                )
            )
            for t in block._time_values
        ]
        previous_y = [
            float(pyo.value(block.initial_y_history[index] / output_units))
            for index in block.y_history_index
        ]
        eps_values = [
            float(pyo.value(block.eps[t] / output_units)) for t in block._time_values
        ]
        previous_eps = [
            float(pyo.value(block.initial_eps_history[index] / output_units))
            for index in block.eps_history_index
        ]
        history_prefix = max(p + d, q)
        if previous_y:
            previous_y = [previous_y[0]] * (
                history_prefix - len(previous_y)
            ) + previous_y
        else:
            previous_y = [0.0] * history_prefix
        previous_eps = [0.0] * (history_prefix - len(previous_eps)) + previous_eps
        time_block = find_time_block(target.model())
        history = {
            "start_date": time_block.datetime_index[0].isoformat(),
            "time_step_seconds": float(
                pyo.value(
                    pyunits.convert(time_block.config.time_step, to_units=pyunits.s)
                )
            ),
            "y_values": previous_y + y_values,
            "eps_values": previous_eps + eps_values,
        }

        return {
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "coefficients": coefficients,
            "history": history,
        }
