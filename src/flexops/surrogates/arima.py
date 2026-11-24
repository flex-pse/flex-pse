"""Pyomo ARIMA/ARIMAX surrogate with explicit innovations and local state.

The surrogate implements one non-seasonal ``ARIMA(p, d, q)`` equation for
``d`` equal to zero or one. Its spec carries the structure, options and the
state needed immediately before the local Pyomo horizon in ``config``, and its
fitted values in ``parameters``::

    {
        "surrogate_type": "arima",
        "config_version": 1,
        "config": {
            "order": [p, d, q],
            "deterministic": "intercept",   # "intercept" (d=0), "drift" (d=1) or null
            "input_variables": {"feed": "kg/hr"},
            "output_variables": {"production": "m^3/hr"},
            "max_ar_coeff": null,           # optional AR stability bound in (0, 1]
            "history": {                    # optional; omitted in regression mode
                "start_date": "2025-01-01T00:00:00",
                "time_step_seconds": 900.0,
                "eta_values": [...],        # disturbance y - x @ beta
                "eps_values": [...],
            },
        },
        "parameters": {
            "intercept": {"value": 0.0, "fixed": true},
            "ar.L1": {"value": 0.5, "fixed": true},     # one per AR lag, p in all
            "ma.L1": {"value": 0.2, "fixed": true},     # one per MA lag, q in all
            "exog.feed": {"value": 0.8, "fixed": true}, # one per input, by name
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

Exogenous coefficients are named ``exog.<input>``, so each one stays paired
with its input even if a JSON tool reorders ``input_variables``.

After a solve, :meth:`ArimaSurrogate.update_state` (through
``unit.update_surrogate_state()``) rewrites ``history`` from the solved
horizon, so the spec initializes the next horizon from it.
"""

from __future__ import annotations

import datetime
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MethodType
from typing import Annotated, Any, ClassVar, Literal

import pyomo.environ as pyo
from pydantic import Field, field_validator, model_validator
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits

from flexcore.config.schema import SurrogateSpec, SurrogateType, _StrictModel
from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.core.registration import CoefficientRegistry
from flexops.core.time_block import find_time_block
from flexops.core.units import parse_units
from flexops.surrogates._config import VariablesConfig
from flexops.surrogates.base import Surrogate

_log = get_logger(__name__)

_Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
_Order = Annotated[int, Field(strict=True, ge=0)]


def _history_prefix(p: int, d: int, q: int) -> int:
    """Return how many pre-modeled values each stored history series carries.

    Both series share the same ``max(p + d, q)`` prefix so a single offset
    indexes into either one; :meth:`ArimaSurrogate.build` slices ``p + d``
    levels and ``q`` innovations out of it.
    """
    return max(p + d, q)


def arima_parameter_names(
    order: tuple[int, int, int] | list[int],
    deterministic: str | None,
    input_names: list[str],
) -> list[str]:
    """Return an ARIMA surrogate's parameter names, in registration order.

    The single definition of ARIMA parameter naming, shared with
    flexparameterize's fitter so a fit's coefficient names and a built
    block's registry can't drift apart.

    Args:
        order: ``(p, d, q)``.
        deterministic: ``"intercept"``, ``"drift"`` or ``None``.
        input_names: Exogenous input names, in order.

    Returns:
        ``deterministic`` (when set), ``ar.L1..ar.Lp``, ``ma.L1..ma.Lq``, then
        ``exog.<name>`` per input.
    """
    p, _d, q = order
    names = [] if deterministic is None else [deterministic]
    names += [f"ar.L{lag}" for lag in range(1, p + 1)]
    names += [f"ma.L{lag}" for lag in range(1, q + 1)]
    return names + [f"exog.{name}" for name in input_names]


class ArimaHistory(_StrictModel):
    """Pre-horizon ARIMA state: disturbances and innovations on a dated grid.

    Both series begin with ``max(p + d, q)`` values before the first modeled
    point, which is ``start_date``.
    """

    start_date: str
    time_step_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    eta_values: list[_Number]
    eps_values: list[_Number]

    @field_validator("start_date")
    @classmethod
    def _iso(cls, value: str) -> str:
        try:
            datetime.datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"must be an ISO-8601 date, got {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _same_length(self) -> ArimaHistory:
        if len(self.eta_values) != len(self.eps_values):
            raise ValueError(
                "eta_values and eps_values must contain the same number of points"
            )
        return self

    @property
    def start(self) -> datetime.datetime:
        """Return ``start_date`` as a datetime."""
        return datetime.datetime.fromisoformat(self.start_date)


class ArimaConfig(VariablesConfig):
    """An ARIMA surrogate's config: order, variables, options and history."""

    order: Annotated[list[_Order], Field(min_length=3, max_length=3)]
    deterministic: Literal["intercept", "drift"] | None = None
    max_ar_coeff: Annotated[float, Field(gt=0, le=1)] | None = None
    history: ArimaHistory | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ArimaConfig:
        p, d, q = self.order
        if d not in (0, 1):
            raise ValueError(f"only d=0 or d=1 is supported, got d={d}")
        if d == 1 and self.deterministic == "intercept":
            raise ValueError(
                "'intercept' is invalid for d=1; use deterministic='drift' for a "
                "constant in the differenced equation"
            )
        if d == 0 and self.deterministic == "drift":
            raise ValueError(
                "'drift' is invalid for d=0; use deterministic='intercept' for "
                "the level equation"
            )
        if self.history is not None:
            prefix = _history_prefix(p, d, q)
            if len(self.history.eta_values) < prefix:
                raise ValueError(
                    f"history.eta_values must include at least max(p+d, q)="
                    f"{prefix} value(s) before the first modeled point, got "
                    f"{len(self.history.eta_values)}"
                )
        return self

    @property
    def parameter_names(self) -> list[str]:
        """Return this config's parameter names, in registration order."""
        return arima_parameter_names(
            self.order, self.deterministic, list(self.input_variables)
        )


@dataclass(frozen=True)
class _Coefficients:
    """The values an ARIMA block is built with, by role."""

    deterministic: float | None
    ar: tuple[float, ...]
    ma: tuple[float, ...]
    exog: tuple[float, ...]


def _extend_history(
    order: tuple[int, int, int],
    coefficients: _Coefficients,
    eta_values: list[float],
    eps_values: list[float],
    count: int,
) -> None:
    """Append ``count`` zero-innovation forecast steps to the history lists.

    ``eta_values`` holds disturbance levels (output minus exogenous term), so
    the forecast needs no exogenous inputs. Each step is the conditional
    mean of the ARIMA equation with the new innovation set to zero.
    """
    p, d, q = order
    constant = coefficients.deterministic or 0.0
    for _ in range(count):
        ar_part = sum(
            coefficients.ar[lag - 1]
            * (eta_values[-lag] if d == 0 else eta_values[-lag] - eta_values[-lag - 1])
            for lag in range(1, p + 1)
        )
        ma_part = sum(
            coefficients.ma[lag - 1] * eps_values[-lag] for lag in range(1, q + 1)
        )
        mean = constant + ar_part + ma_part
        eta_values.append(mean if d == 0 else eta_values[-1] + mean)
        eps_values.append(0.0)


def _exog_contribution(block, exogenous, t):
    """Return the exogenous regression term at ``t`` in output units.

    Raises:
        FlexConfigError: If an input's model units are incompatible with
            the units it declares.
    """
    output_units = block._output_units
    val = 0.0 * output_units
    for index, (variable, declared_units) in enumerate(exogenous, start=1):
        try:
            normalized_input = (
                pyunits.convert(variable[t], declared_units) / declared_units
            )
        except UnitsError as exc:
            name = block._exog_names[index - 1]
            raise FlexConfigError(
                f"ARIMA input {name!r} declares {declared_units!s}, "
                f"incompatible with its model units "
                f"{pyunits.get_units(variable[t])!s}.",
                field="input_variables",
                value=name,
            ) from exc
        val += block.exog_coefs[index] * normalized_input * output_units
    return val


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
            f"ARIMA innovation_scale at time {t!r} must be positive, got {magnitude}.",
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
    """Non-seasonal ARIMA/ARIMAX relationship with explicit innovations.

    See the module docstring for the spec. With ``config.history`` the
    pre-horizon state it implies is fixed; without it (regression mode) that
    state starts at zero and is left free for in-model estimation.

    A free parameter given without a value starts from these guesses: ``0.0``
    for the intercept or drift, ``min(1.0, max_ar_coeff)`` for AR lags, and
    ``1.0`` for MA lags and exogenous coefficients.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.ARIMA
    config_model: ClassVar[type[ArimaConfig]] = ArimaConfig

    @classmethod
    def parameter_shapes(cls, config: ArimaConfig) -> dict[str, tuple]:
        """Return the deterministic term, one per lag and one per input."""
        return {name: () for name in config.parameter_names}

    @classmethod
    def initial_guess(cls, name: str, config: ArimaConfig) -> float:
        """Return a free parameter's starting value (see the class docstring)."""
        if name.startswith("ar."):
            return 1.0 if config.max_ar_coeff is None else min(1.0, config.max_ar_coeff)
        if name.startswith(("ma.", "exog.")):
            return 1.0
        return 0.0

    def _validate(self, config: ArimaConfig, spec: SurrogateSpec) -> None:
        """Reject a fixed AR coefficient outside ``max_ar_coeff``.

        A free one is bounded during solving instead (see :meth:`_build`).

        Raises:
            FlexConfigError: If a fixed ``ar.L*`` value exceeds the bound.
        """
        bound = config.max_ar_coeff
        if bound is None:
            return
        for name, state in spec.parameters.items():
            if (
                name.startswith("ar.")
                and state.fixed
                and state.value is not None
                and abs(state.value) > bound
            ):
                raise FlexConfigError(
                    f"ARIMA {name} = {state.value} exceeds max_ar_coeff={bound}; AR "
                    "coefficients near the unit root are unstable in this "
                    "surrogate's difference-equation form.",
                    field=f"parameters.{name}",
                    value=state.value,
                )

    @classmethod
    def make_spec(
        cls,
        *,
        order: tuple[int, int, int] | list[int],
        input_variables: dict[str, str],
        output_variables: dict[str, str],
        values: dict[str, float] | None = None,
        deterministic: Literal["intercept", "drift"] | None = None,
        fixed: dict[str, bool] | None = None,
        history: dict[str, Any] | None = None,
        max_ar_coeff: float | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> SurrogateSpec:
        """Return a validated ARIMA spec.

        Args:
            order: ``(p, d, q)``.
            input_variables: ``{name: units}`` of each exogenous input, in
                order.
            output_variables: ``{name: units}`` of the one output.
            values: ``{parameter: value}`` by the names
                :func:`arima_parameter_names` gives. A free parameter may be
                left out to start from its initial guess.
            deterministic: ``"intercept"`` (``d=0``), ``"drift"`` (``d=1``),
                or ``None`` for no constant term.
            fixed: ``{parameter: False}`` for each parameter to leave free;
                every other parameter is fixed.
            history: The pre-horizon state; ``None`` for regression mode.
            max_ar_coeff: Optional bound on every AR coefficient's magnitude.
            provenance: Free-form fit metadata.

        Returns:
            The spec.

        Raises:
            FlexConfigError: If the result does not match the contract.
        """
        config: dict[str, Any] = {
            "order": list(order),
            "deterministic": deterministic,
            "input_variables": dict(input_variables),
            "output_variables": dict(output_variables),
        }
        if max_ar_coeff is not None:
            config["max_ar_coeff"] = max_ar_coeff
        if history is not None:
            config["history"] = dict(history)
        return cls._make(
            config=config, values=dict(values or {}), fixed=fixed, provenance=provenance
        )

    @property
    def input_variables(self) -> dict[str, str]:
        """Return exogenous input names and declared units."""
        return dict(self.config.input_variables)

    @property
    def output_variables(self) -> dict[str, str]:
        """Return the single output name and declared units."""
        return dict(self.config.output_variables)

    @property
    def order(self) -> tuple[int, int, int]:
        """Return ``(p, d, q)``."""
        p, d, q = self.config.order
        return p, d, q

    def _coefficients(self) -> _Coefficients:
        p, _d, q = self.order
        deterministic = self.config.deterministic
        return _Coefficients(
            deterministic=(
                None if deterministic is None else float(self.value_of(deterministic))
            ),
            ar=tuple(float(self.value_of(f"ar.L{lag}")) for lag in range(1, p + 1)),
            ma=tuple(float(self.value_of(f"ma.L{lag}")) for lag in range(1, q + 1)),
            exog=tuple(
                float(self.value_of(f"exog.{name}")) for name in self.input_variables
            ),
        )

    def _initial_state(self, time_block) -> tuple[list[float], list[float]]:
        """Return the pre-horizon ``(eta, eps)`` lags ``history`` implies.

        When ``history`` ends before the ``TimeBlock`` starts, the missing
        steps are forecast with zero innovations and a warning is logged.

        Args:
            time_block: The model's ``TimeBlock``.

        Returns:
            The ``p + d`` disturbance lags and ``q`` innovation lags immediately
            preceding the first modeled point, oldest to newest.

        Raises:
            FlexConfigError: If ``history`` does not align with the
                ``TimeBlock`` start date on the model's step grid.
        """
        p, d, q = self.order
        history = self.config.history
        prefix = _history_prefix(p, d, q)
        current_start = time_block.datetime_index[0].to_pydatetime()
        offset = (current_start - history.start).total_seconds() / (
            history.time_step_seconds
        )
        offset_index = round(offset)
        if offset < 0 or abs(offset - offset_index) > 1e-8:
            reason = (
                "starts before the stored history, which begins at "
                f"{history.start.isoformat()}; re-export the spec with a "
                "longer history_length, or history_length=None for the full "
                "training series"
                if offset < 0
                else "is off the history's step grid starting at "
                f"{history.start.isoformat()} with step "
                f"{history.time_step_seconds}s"
            )
            raise FlexConfigError(
                f"ARIMA TimeBlock start date {current_start.isoformat()} {reason}.",
                field="config.history.start_date",
                value=current_start,
            )
        eta_values = list(history.eta_values)
        eps_values = list(history.eps_values)
        missing = offset_index - (len(eta_values) - prefix)
        if missing > 0:
            _log.warning(
                "ARIMA history ends %d step(s) before the TimeBlock start "
                "date %s; forecasting the missing pre-horizon state with "
                "zero innovations.",
                missing,
                current_start.isoformat(),
            )
            _extend_history(
                self.order, self._coefficients(), eta_values, eps_values, missing
            )
        end = prefix + offset_index
        return (
            eta_values[end - (p + d) : end],
            eps_values[end - q : end],
        )

    def _build(self, unit, target):
        """Build coefficient, innovation, and initial-state components.

        The returned ``body(t)`` is the complete right-hand side consumed by
        :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`. Each
        coefficient is fixed or free as its parameter says (applied by
        :meth:`~flexops.surrogates.base.Surrogate.build`), and current innovations are
        fixed. When ``history`` is supplied, the pre-horizon state it implies
        is also fixed. When
        ``history`` is omitted, the pre-horizon state is initialized to zero
        and left free for in-model estimation. Regression callers explicitly
        unfix any other quantities they intend to estimate.

        Args:
            unit: Unit owning the variables named by ``input_variables``.
            target: Time-indexed output variable representing ``y[t]``.

        Returns:
            ``(block, body)`` where ``body(t)`` carries declared output units.

        Raises:
            FlexConfigError: If ``history`` does not align with the
                ``TimeBlock`` start date on the model's step grid, or if an
                input variable's model units are incompatible with the units
                it declares.
        """
        p, d, q = self.order
        coefficients = self._coefficients()
        state_provided = self.config.history is not None
        time_block = find_time_block(unit.model())
        if state_provided:
            eta_history, eps_history = self._initial_state(time_block)
        else:
            eta_history = [0.0] * (p + d)
            eps_history = [0.0] * q
        exog_names = list(self.input_variables)
        output_units = parse_units(next(iter(self.output_variables.values())))
        time_values = list(time_block.time_index)
        time_positions = {value: position for position, value in enumerate(time_values)}

        exogenous = [
            (unit.resolve_variable(name, field="input_variables"), parse_units(units))
            for name, units in self.input_variables.items()
        ]

        block = pyo.Block(concrete=True)
        block.time = pyo.Set(initialize=time_values, ordered=True)
        block.coefficient_vars: dict[str, object] = {}
        block.coefficients = CoefficientRegistry()

        deterministic_name = self.config.deterministic
        if deterministic_name is not None:
            block.add_component(
                deterministic_name,
                pyo.Var(
                    initialize=coefficients.deterministic,
                    doc=(
                        "ARIMA level intercept magnitude"
                        if d == 0
                        else "ARIMA constant drift magnitude in differenced output"
                    ),
                ),
            )
            deterministic_var = block.find_component(deterministic_name)
            block.coefficient_vars[deterministic_name] = deterministic_var
            block.coefficients.register_coefficient(
                deterministic_name, deterministic_var
            )

        if p > 0:
            bound = self.config.max_ar_coeff
            block.ar_coefs = pyo.Var(
                range(1, p + 1),
                initialize={lag: coefficients.ar[lag - 1] for lag in range(1, p + 1)},
                bounds=(None, None) if bound is None else (-bound, bound),
                doc="Dimensionless autoregressive coefficients",
            )
            block.coefficient_vars["ar_coefs"] = block.ar_coefs
            for lag in range(1, p + 1):
                block.coefficients.register_coefficient(
                    f"ar.L{lag}", block.ar_coefs[lag]
                )

        if q > 0:
            block.ma_coefs = pyo.Var(
                range(1, q + 1),
                initialize={lag: coefficients.ma[lag - 1] for lag in range(1, q + 1)},
                doc="Dimensionless moving-average coefficients",
            )
            block.coefficient_vars["ma_coefs"] = block.ma_coefs
            for lag in range(1, q + 1):
                block.coefficients.register_coefficient(
                    f"ma.L{lag}", block.ma_coefs[lag]
                )

        if exog_names:
            block.exog_coefs = pyo.Var(
                range(1, len(exog_names) + 1),
                initialize={
                    index: coefficients.exog[index - 1]
                    for index in range(1, len(exog_names) + 1)
                },
                doc="Exogenous coefficient magnitudes in declared data units",
            )
            block.coefficient_vars["exog_coefs"] = block.exog_coefs
            for index, name in enumerate(exog_names, start=1):
                block.coefficients.register_coefficient(
                    f"exog.{name}", block.exog_coefs[index]
                )

        block.eta_history_index = pyo.Set(
            initialize=range(len(eta_history)), ordered=True
        )
        block.initial_eta_history = pyo.Var(
            block.eta_history_index,
            initialize={index: value for index, value in enumerate(eta_history)},
            units=output_units,
            doc=(
                "Pre-horizon disturbance levels (output minus exogenous "
                "term), oldest to newest"
            ),
        )
        if state_provided:
            for index, value in enumerate(eta_history):
                block.initial_eta_history[index].fix(value)

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
            return _exog_contribution(block, exogenous, time_values[position])

        def eta_at(position: int):
            """Return disturbance level from local horizon or pre-horizon state."""
            if position >= 0:
                y_val = pyunits.convert(target[time_values[position]], output_units)
                return y_val - exog_at(position)
            history_index = len(eta_history) + position
            if history_index < 0:
                raise FlexConfigError(
                    f"ARIMA disturbance lag at local position {position} exceeds "
                    f"the {len(eta_history)} stored eta_history values.",
                    field="config.history.eta_values",
                    value=position,
                )
            return block.initial_eta_history[history_index]

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
                    field="config.history.eps_values",
                    value=position,
                )
            return block.initial_eps_history[history_index]

        def body(t):
            position = time_positions[t]
            current_exog = exog_at(position)
            mean = 0.0 * output_units
            if deterministic_name is not None:
                mean += block.coefficient_vars[deterministic_name] * output_units
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

    def solved_history(self, block, target) -> dict[str, Any]:
        """Return the history a solve implies, without recording it.

        The history begins with the ``max(p + d, q)`` initialization lags the
        block was built with (the solved values in regression mode), followed
        by the solved horizon's disturbances and innovations, and starts at the
        horizon's first point. Rebuilding the same horizon from it reproduces
        the same initial state; a later horizon starts from the solved values.

        Args:
            block: The block :meth:`~flexops.surrogates.base.Surrogate.build` returned,
                after a solve.
            target: The time-indexed output variable passed to
                :meth:`~flexops.surrogates.base.Surrogate.build`.

        Returns:
            A ``history`` mapping for ``config``.

        Raises:
            FlexConfigError: If the block has values missing (it has not been
                solved).
        """
        p, d, q = block._order
        output_units = block._output_units
        unit = target.parent_block()
        exogenous = [
            (
                unit.resolve_variable(name, field="input_variables"),
                parse_units(self.input_variables[name]),
            )
            for name in block._exog_names
        ]
        unsolved = [
            component.name
            for component in (target, block.eps, block.initial_eta_history)
            if any(component[index].value is None for index in component)
        ]
        if unsolved:
            raise FlexConfigError(
                f"Cannot record ARIMA state: {unsolved} have no values yet. Call "
                "update_surrogate_state() after a successful solve.",
                field="history",
                value=unsolved,
            )

        def magnitude(expression) -> float:
            return float(pyo.value(expression / output_units))

        eta_values = [
            magnitude(
                pyunits.convert(
                    target[t] - _exog_contribution(block, exogenous, t), output_units
                )
            )
            for t in block._time_values
        ]
        eps_values = [magnitude(block.eps[t]) for t in block._time_values]
        previous_eta = [
            magnitude(block.initial_eta_history[index])
            for index in block.eta_history_index
        ]
        previous_eps = [
            magnitude(block.initial_eps_history[index])
            for index in block.eps_history_index
        ]
        prefix = _history_prefix(p, d, q)
        if previous_eta:
            previous_eta = [previous_eta[0]] * (
                prefix - len(previous_eta)
            ) + previous_eta
        else:
            previous_eta = [0.0] * prefix
        previous_eps = [0.0] * (prefix - len(previous_eps)) + previous_eps
        time_block = find_time_block(target.model())
        return {
            "start_date": time_block.datetime_index[0].isoformat(),
            "time_step_seconds": float(
                pyo.value(
                    pyunits.convert(time_block.config.time_step, to_units=pyunits.s)
                )
            ),
            "eta_values": previous_eta + eta_values,
            "eps_values": previous_eps + eps_values,
        }

    def update_state(self, block, target) -> None:
        """Rewrite ``config.history`` from the solved horizon.

        See :meth:`solved_history`. The spec then initializes the next horizon
        from the solve. Parameters are untouched; read them with
        :meth:`~flexops.surrogates.base.Surrogate.read_spec`.

        Args:
            block: The block :meth:`~flexops.surrogates.base.Surrogate.build` returned,
                after a solve.
            target: The time-indexed output variable passed to
                :meth:`~flexops.surrogates.base.Surrogate.build`.

        Raises:
            FlexConfigError: If the block has not been solved.
        """
        config = {**self.spec.config, "history": self.solved_history(block, target)}
        self.config = self.config_model.model_validate(config)
        self.spec = self.spec.model_copy(update={"config": config})
