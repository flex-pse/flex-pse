"""Wrap an arbitrary external differentiable model as a unit's relation.

The general idea this module implements: some relationships have no closed
form a modeler can write as a Pyomo expression, but *do* have external code
that can report a numeric value and its derivatives at a point -- a fitted
neural network, a linearized CFD or process simulator, a vendor's own
sensitivity-aware solver, anything with that shape. Rather than deriving a
closed-form expression, this module wraps that external code opaquely as a
PyNumero ``ExternalGreyBoxBlock``: only its numeric output and derivatives
are used, evaluated at whatever point the optimizer asks for. Building a unit
with this surrogate therefore requires solving with ``SolverFactory("cyipopt")``,
the one Pyomo-side solver that can call back into such a block; see
:mod:`flexcore.solvers.facade` for the guard that raises otherwise.

Which tool actually evaluates the wrapped model is a declared ``framework``
field on the surrogate's data, resolved to a **driver** -- a small object
evaluating one model and its first two derivatives at a point
(:class:`ExternalModelDriver`). PyTorch (wrapping a fitted ``nn.Module`` or a
plain callable, via ``torch.autograd``) is the only driver implemented today,
but the driver abstraction exists for future non-neural-network cases too --
e.g. a CFD model that exposes its own adjoint/sensitivity computation would
be a new driver here, not a new kind of surrogate. Resolving a framework
(:func:`get_driver`) is the only place a driver's own heavy dependency
(``torch``, ...) is imported, and that happens lazily, at call time -- this
is what keeps ``import flexops.surrogates`` clean on a bare install.

``pyomo.contrib.pynumero.interfaces.external_grey_box`` is core Pyomo,
importable with neither ``cyipopt`` nor a driver's framework (``torch``)
installed -- only :func:`get_driver` (called from
:meth:`ExternalModelSurrogate._validate`) imports one.
"""

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

_DATA_KEYS = ("framework", "model_path", "input_variables", "output_variables")
_OPTIONAL_KEYS = ("probe_point",)


class ExternalFramework(enum.StrEnum):
    """Which tool a driver evaluates an external model through.

    Only ``PYTORCH`` is implemented; add a member here only when a concrete
    new driver is actually being built (e.g. a CFD-model driver), not
    speculatively ahead of one.
    """

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


_DRIVERS: dict[ExternalFramework, str] = {
    ExternalFramework.PYTORCH: "flexops.surrogates.drivers.torch_driver.TorchDriver",
}
"""dict: framework -> dotted path of its driver class. The extension point
for a new driver: add a member to :class:`ExternalFramework` and an entry
here -- no other change is needed to resolve it."""


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
        member = ExternalFramework(framework)
    except ValueError as exc:
        known = ", ".join(repr(m.value) for m in ExternalFramework)
        raise FlexConfigError(
            f"{framework!r} is not a known ExternalFramework. Known: {known}.",
            field="framework",
            value=framework,
        ) from exc
    dotted = _DRIVERS[member]
    module_name, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class _ExternalModelGreyBox(ExternalGreyBoxModel):
    """Adapts an :class:`ExternalModelDriver` into PyNumero's
    ``ExternalGreyBoxModel`` interface. One instance per time index -- never
    shared (see ``ExternalModelSurrogate.build`` pitfall 1).
    """

    def __init__(self, driver, input_names: list[str], output_name: str, probe: dict):
        """Store the driver and this instance's own input/probe cache.

        Args:
            driver: The driver evaluating the wrapped model.
            input_names: Ordered input names.
            output_name: The one output name.
            probe: ``{input name: probe value}``, in declared units.
        """
        self._driver = driver
        self._input_names = list(input_names)
        self._output_name = output_name
        self._probe = probe
        self._n = len(self._input_names)
        self._x = np.array([probe[name] for name in self._input_names], dtype=float)
        self._multiplier = 0.0

    def input_names(self):
        """Return the ordered input names."""
        return self._input_names

    def output_names(self):
        """Return the one-element list of output names."""
        return [self._output_name]

    def set_input_values(self, input_values) -> None:
        """Cache the current input point."""
        self._x = np.asarray(input_values, dtype=float)

    def evaluate_outputs(self) -> np.ndarray:
        """Return the model's output at the cached input point."""
        return np.array([self._driver.evaluate(self._x)])

    def evaluate_jacobian_outputs(self) -> coo_matrix:
        """Dense-pattern Jacobian, shape ``(1, n_inputs)``, zeros included."""
        row = np.zeros(self._n, dtype=int)
        col = np.arange(self._n)
        data = np.asarray(self._driver.jacobian(self._x), dtype=float)
        return coo_matrix((data, (row, col)), shape=(1, self._n))

    def set_output_constraint_multipliers(self, output_con_multiplier_values) -> None:
        """Cache the output-constraint multiplier for the Hessian scale."""
        self._multiplier = float(output_con_multiplier_values[0])

    def evaluate_hessian_outputs(self) -> coo_matrix:
        """Lower-triangular Hessian, shape ``(n_inputs, n_inputs)``, scaled by
        the cached multiplier, zeros included."""
        hessian = self._multiplier * np.asarray(
            self._driver.hessian(self._x), dtype=float
        )
        rows, cols = np.tril_indices(self._n)
        return coo_matrix((hessian[rows, cols], (rows, cols)), shape=(self._n, self._n))

    def finalize_block_construction(self, pyomo_block) -> None:
        """Initialize input/output Vars from the probe point; never fix them."""
        for name in self._input_names:
            pyomo_block.inputs[name].set_value(self._probe[name])
        probe_vector = np.array(
            [self._probe[name] for name in self._input_names], dtype=float
        )
        pyomo_block.outputs[self._output_name].set_value(
            self._driver.evaluate(probe_vector)
        )


class ExternalModelSurrogate(Surrogate):
    """Wraps a fitted external differentiable model (e.g. a PyTorch
    ``nn.Module``) as a unit's registered relation, solved via
    ``SolverFactory("cyipopt")``.

    ``data`` is::

        {"framework": "pytorch",
         "model_path": "myproject.models.my_fitted_model",
         "input_variables": {"flow_in": "m^3/hr", "ambient_temperature": "degK"},
         "output_variables": {"fouling_rate": "1/hr"},
         "probe_point": {"flow_in": 1.0, "ambient_temperature": 298.15}}

    ``model_path`` is a dotted import path (``importlib``-resolved) to an
    already-fitted, callable model; this is trusted-as-code data, the same
    trade-off every other surrogate's ``data`` makes (see
    ``plan/00_conventions.md`` §4). ``probe_point`` (optional, defaults to
    ``1.0`` for every input) is the point the grey-box block's input Vars are
    initialized to and the point ``check_differentiable`` is smoke-tested at.

    GPU placement (for the PyTorch driver) is controlled entirely by the
    model object itself -- call ``model.to("cuda")`` before it is ever named
    by ``model_path`` -- not by any key here; there is no ``"device"`` field.

    Registers no coefficients: a wrapped model's internal weights are not
    something FlexParameterize can regress.
    """

    surrogate_type: ClassVar[SurrogateType] = SurrogateType.EXTERNAL_MODEL

    def _validate(self) -> None:
        """Validate the external-model ``data`` contract (see class docstring).

        Raises:
            FlexConfigError: If a key is missing/unknown, a units string does
                not parse, ``output_variables`` does not have exactly one
                entry, ``probe_point`` names a key not in ``input_variables``,
                ``framework`` is unknown, ``model_path`` does not resolve, or
                the resolved model fails ``check_differentiable``.
        """
        unknown = sorted(set(self.data) - set(_DATA_KEYS) - set(_OPTIONAL_KEYS))
        if unknown:
            raise FlexConfigError(
                f"external_model surrogate data carries unknown key(s) "
                f"{unknown}; it may only have {_DATA_KEYS + _OPTIONAL_KEYS}.",
                field="data",
                value=unknown,
            )
        missing = sorted(set(_DATA_KEYS) - set(self.data))
        if missing:
            raise FlexConfigError(
                f"external_model surrogate data is missing key(s) {missing}; "
                f"it must have {_DATA_KEYS}.",
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

        self._probe_point = {name: float(probe_point.get(name, 1.0)) for name in inputs}
        probe_vector = np.array(
            [self._probe_point[name] for name in inputs], dtype=float
        )
        self._driver = driver_class(model, n_inputs=len(inputs))
        self._driver.check_differentiable(probe_vector)

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

        def _make_grey_box(b, t):
            del b, t
            return _ExternalModelGreyBox(driver, input_names, output_name, probe)

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
