"""Wrap an arbitrary external differentiable model as a unit's relation."""

import enum
import importlib
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
    ExternalGreyBoxBlock,
    ExternalGreyBoxModel,
)
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits
from scipy.sparse import coo_matrix

from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.units import parse_units
from flexops.surrogates.base import Surrogate


class ExternalModelDataKey(enum.StrEnum):
    """The keys an external-model surrogate's ``data`` may hold."""

    FRAMEWORK = "framework"
    MODEL_PATH = "model_path"
    INPUT_VARIABLES = "input_variables"
    OUTPUT_VARIABLES = "output_variables"
    PROBE_POINT = "probe_point"
    TRAINING_RANGE = "training_range"


class ExternalFramework(enum.StrEnum):
    """Which tool a driver evaluates an external model through."""

    PYTORCH = "pytorch"


class ExternalModelDriver(ABC):
    """Evaluate one external model and its first two derivatives at a point.

    Attributes:
        framework: The :class:`ExternalFramework` this driver implements.
    """

    framework: ClassVar[ExternalFramework]

    def __init__(self, model, n_inputs: int) -> None:
        """Store the model and its input dimension.

        Args:
            model: The fitted, callable external model.
            n_inputs: Number of scalar inputs the model takes.
        """
        self._model = model
        self._n_inputs = n_inputs

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> float:
        """Return the model's scalar output at ``x``."""

    @abstractmethod
    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Dense gradient, shape ``(n_inputs,)``."""

    @abstractmethod
    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Dense *full symmetric* Hessian, shape ``(n_inputs, n_inputs)``."""

    @abstractmethod
    def check_differentiable(self, x: np.ndarray) -> None:
        """Raise FlexConfigError if the model is not differentiable at ``x``."""


def get_driver(framework: ExternalFramework | str) -> type[ExternalModelDriver]:
    """Resolve a framework to its driver class, importing it lazily.

    Args:
        framework: An :class:`ExternalFramework` member or its string value.

    Returns:
        The driver class.

    Raises:
        FlexConfigError: If ``framework`` is not a known ``ExternalFramework``
            value.
    """
    try:
        ExternalFramework(framework)
    except ValueError as exc:
        known = ", ".join(repr(m.value) for m in ExternalFramework)
        raise FlexConfigError(
            f"{framework!r} is not a known ExternalFramework. Known: {known}.",
            field="framework",
            value=framework,
        ) from exc
    # Local import: torch is optional and must not load with this module.
    from flexops.surrogates.drivers.torch_driver import TorchDriver

    return TorchDriver


class _ExternalModelGreyBox(ExternalGreyBoxModel):
    """Adapts an :class:`ExternalModelDriver` to PyNumero, one instance per time index.

    Attributes:
        training_range: ``{input name: (lower, upper)}`` the model was trained over.
    """

    def __init__(
        self,
        driver,
        input_names: list[str],
        output_name: str,
        probe: dict,
        training_range: dict[str, tuple[float, float]] | None = None,
    ):
        """Store the driver and this instance's own input and evaluation cache.

        Args:
            driver: The driver evaluating the wrapped model.
            input_names: Ordered input names.
            output_name: The one output name.
            probe: ``{input name: start value}``, in declared units.
            training_range: ``{input name: (lower, upper)}``, in declared units.
        """
        self._driver = driver
        self._input_names = list(input_names)
        self._output_name = output_name
        self._probe = probe
        self._n = len(self._input_names)
        self._x = np.array([probe[name] for name in self._input_names], dtype=float)
        self._multiplier = 0.0
        self.training_range = training_range or {}
        self._clear_cache()

    def _clear_cache(self) -> None:
        """Forget the evaluations made at the previous input point."""
        self._value = None
        self._jacobian = None
        self._hessian = None

    def input_names(self):
        """Return the ordered input names."""
        return self._input_names

    def output_names(self):
        """Return the one-element list of output names."""
        return [self._output_name]

    def set_input_values(self, input_values) -> None:
        """Store a copy of the input point, clearing the cache if it changed."""
        # A copy, since PyNumero may reuse this buffer for the next point.
        x = np.array(input_values, dtype=float)
        if not np.array_equal(x, self._x):
            self._clear_cache()
        self._x = x

    def evaluate_outputs(self) -> np.ndarray:
        """Return the model's output at the cached input point."""
        if self._value is None:
            self._value = self._driver.evaluate(self._x)
        return np.array([self._value])

    def evaluate_jacobian_outputs(self) -> coo_matrix:
        """Dense-pattern Jacobian, shape ``(1, n_inputs)``, zeros included."""
        row = np.zeros(self._n, dtype=int)
        col = np.arange(self._n)
        if self._jacobian is None:
            self._jacobian = np.asarray(self._driver.jacobian(self._x), dtype=float)
        return coo_matrix((self._jacobian, (row, col)), shape=(1, self._n))

    def set_output_constraint_multipliers(self, output_con_multiplier_values) -> None:
        """Cache the output-constraint multiplier for the Hessian scale."""
        self._multiplier = float(output_con_multiplier_values[0])

    def evaluate_hessian_outputs(self) -> coo_matrix:
        """Lower-triangular Hessian scaled by the multiplier, zeros included."""
        if self._hessian is None:
            self._hessian = np.asarray(self._driver.hessian(self._x), dtype=float)
        hessian = self._multiplier * self._hessian
        rows, cols = np.tril_indices(self._n)
        return coo_matrix((hessian[rows, cols], (rows, cols)), shape=(self._n, self._n))

    def finalize_block_construction(self, pyomo_block) -> None:
        """Initialize input/output Vars from the probe point; never fix them."""
        for name in self._input_names:
            pyomo_block.inputs[name].set_value(self._probe[name])
        pyomo_block.outputs[self._output_name].set_value(self.evaluate_outputs()[0])


class ExternalModelSurrogate(Surrogate):
    """Wraps a fitted external differentiable model as a unit's registered relation."""

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.EXTERNAL_MODEL

    def _validate(self) -> None:
        """Validate the external-model ``data`` and check the model at its start point.

        Raises:
            FlexConfigError: If a key is missing/unknown, a units string does
                not parse, ``output_variables`` does not have exactly one
                entry, ``probe_point`` or ``training_range`` is malformed,
                ``framework`` is unknown, ``model_path`` does not resolve, the
                resolved model fails ``check_differentiable``, or its output
                at the start point is not finite.
        """
        allowed = [str(key) for key in ExternalModelDataKey]
        required = sorted(
            set(allowed)
            - {ExternalModelDataKey.PROBE_POINT, ExternalModelDataKey.TRAINING_RANGE}
        )
        unknown = sorted(set(self.data) - set(allowed))
        if unknown:
            raise FlexConfigError(
                f"external_model surrogate data carries unknown key(s) "
                f"{unknown}; it may only have {allowed}.",
                field="data",
                value=unknown,
            )
        missing = sorted(set(required) - set(self.data))
        if missing:
            raise FlexConfigError(
                f"external_model surrogate data is missing key(s) {missing}; "
                f"it must have {required}.",
                field="data",
                value=missing,
            )

        inputs = self.data["input_variables"]
        if not isinstance(inputs, dict) or not inputs:
            raise FlexConfigError(
                "external_model surrogate 'input_variables' must be a "
                f"non-empty {{name: units}} mapping, got {inputs!r}.",
                field="input_variables",
                value=inputs,
            )
        outputs = self.data["output_variables"]
        if not isinstance(outputs, dict) or len(outputs) != 1:
            raise FlexConfigError(
                "external_model surrogate 'output_variables' must be a "
                f"single {{name: units}} entry -- a registered relation has "
                f"one target -- got {outputs!r}.",
                field="output_variables",
                value=outputs,
            )
        for field, mapping in (
            ("input_variables", inputs),
            ("output_variables", outputs),
        ):
            for name, units in mapping.items():
                if not name or not isinstance(units, str) or not units:
                    raise FlexConfigError(
                        f"external_model surrogate {field!r} entry {name!r} "
                        f"must map to a non-empty units string, got "
                        f"{units!r}.",
                        field=field,
                        value=units,
                    )
                parse_units(units)

        probe_point = self.data.get("probe_point", {})
        if not isinstance(probe_point, dict):
            raise FlexConfigError(
                "external_model surrogate 'probe_point' must be a mapping, "
                f"got {probe_point!r}.",
                field="probe_point",
                value=probe_point,
            )
        unknown_probe = sorted(set(probe_point) - set(inputs))
        if unknown_probe:
            raise FlexConfigError(
                f"external_model surrogate 'probe_point' names {unknown_probe}, "
                f"not in 'input_variables' ({sorted(inputs)}).",
                field="probe_point",
                value=unknown_probe,
            )
        for name, value in probe_point.items():
            try:
                float(value)
            except (TypeError, ValueError) as exc:
                raise FlexConfigError(
                    f"external_model surrogate 'probe_point' entry {name!r} "
                    f"must be a number, got {value!r}.",
                    field="probe_point",
                    value=name,
                ) from exc

        training_range = self.data.get("training_range", {})
        if not isinstance(training_range, dict) or set(training_range) - set(inputs):
            raise FlexConfigError(
                "external_model surrogate 'training_range' must be a mapping of "
                f"names in 'input_variables' ({sorted(inputs)}) to [lower, upper], "
                f"got {training_range!r}.",
                field="training_range",
                value=training_range,
            )
        self._training_range = {}
        for name, bounds in training_range.items():
            try:
                lower, upper = (float(bound) for bound in bounds)
            except (TypeError, ValueError) as exc:
                raise FlexConfigError(
                    f"external_model surrogate 'training_range' entry {name!r} "
                    f"must be [lower, upper], got {bounds!r}.",
                    field="training_range",
                    value=name,
                ) from exc
            if not lower < upper:
                raise FlexConfigError(
                    f"external_model surrogate 'training_range' entry {name!r} "
                    f"must have lower < upper, got {bounds!r}.",
                    field="training_range",
                    value=name,
                )
            self._training_range[name] = (lower, upper)

        driver_class = get_driver(self.data["framework"])

        model_path = self.data["model_path"]
        module_name, _, attr_name = model_path.rpartition(".")
        if not module_name:
            raise FlexConfigError(
                f"'model_path' {model_path!r} must be a dotted path "
                "'module.attribute'.",
                field="model_path",
                value=model_path,
            )
        try:
            module = importlib.import_module(module_name)
            model = getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            raise FlexConfigError(
                f"Could not resolve 'model_path' {model_path!r}: {exc}.",
                field="model_path",
                value=model_path,
            ) from exc

        # Start from the probe point, then the range's midpoint, then 1.0.
        midpoints = {name: sum(pair) / 2 for name, pair in self._training_range.items()}
        self._probe_point = {
            name: float(probe_point.get(name, midpoints.get(name, 1.0)))
            for name in inputs
        }
        probe_vector = np.array(
            [self._probe_point[name] for name in inputs], dtype=float
        )
        self._driver = driver_class(model, n_inputs=len(inputs))
        self._driver.check_differentiable(probe_vector)
        if not np.isfinite(self._driver.evaluate(probe_vector)):
            raise FlexConfigError(
                f"model {model_path!r} has a non-finite output at {self._probe_point}; "
                "set 'probe_point' to a point where the model is defined.",
                field="probe_point",
                value=self._probe_point,
            )

    @property
    def input_variables(self) -> dict[str, str]:
        """Return the declared input variable names and their units."""
        return dict(self.data["input_variables"])

    @property
    def output_variables(self) -> dict[str, str]:
        """Return the one declared output variable name and its units."""
        return dict(self.data["output_variables"])

    def build(self, unit, target):
        """Return ``(block, body(t))`` wrapping the external model as a grey box.

        Args:
            unit: The unit the relationship is built on.
            target: The Var/Reference the relationship determines; its
                ``index_set()`` (time) indexes the ``ExternalGreyBoxBlock``.

        Returns:
            A tuple ``(block, body)``: ``block`` carries the
            ``ExternalGreyBoxBlock`` and its input-linking constraints;
            ``body`` reads the grey box's output for each time index, in the
            declared output units.
        """
        index_set = target.index_set()
        input_names = list(self.input_variables)
        output_name = next(iter(self.output_variables))
        output_units = parse_units(next(iter(self.output_variables.values())))

        declared = {
            name: (
                unit.resolve_variable(name, field="input_variables"),
                parse_units(units),
            )
            for name, units in self.input_variables.items()
        }
        driver = self._driver
        probe = self._probe_point
        training_range = self._training_range

        def _make_grey_box(b, t):
            del b, t
            return _ExternalModelGreyBox(
                driver, input_names, output_name, probe, training_range
            )

        block = pyo.Block(concrete=True)
        block.egb = ExternalGreyBoxBlock(index_set, external_model=_make_grey_box)

        def _link_rule(b, t, name):
            del b
            var, units = declared[name]
            try:
                converted = pyunits.convert(var[t], units)
            except UnitsError as exc:
                raise FlexConfigError(
                    f"external_model surrogate declares {name!r} in "
                    f"{units!s}, incompatible with its actual units "
                    f"{pyunits.get_units(var[t])!s} on the unit.",
                    field="input_variables",
                    value=name,
                ) from exc
            return block.egb[t].inputs[name] == converted / units

        block.input_links = pyo.Constraint(index_set, input_names, rule=_link_rule)

        def body(t):
            return block.egb[t].outputs[output_name] * output_units

        return block, body
