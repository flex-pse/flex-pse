"""Surrogate: the base class every predefined relationship shape subclasses.

A surrogate's contract is its spec: a
:class:`~flexcore.config.schema.SurrogateSpec` whose ``config`` the class
defines (variables, structure, options, state) and whose ``parameters`` are
the values it fits, keyed by the names it registers on its built block. The
class validates both when it is constructed, so a malformed relationship fails
at construction, not mid-solve, and without building any Pyomo objects.

Developing a surrogate starts here, with its contract:

1. Subclass :class:`Surrogate`, set :attr:`~Surrogate.surrogate_type` and
   :attr:`~Surrogate.config_model` (a pydantic model of ``config``).
2. Implement :meth:`~Surrogate.parameter_shapes` (which parameters a config
   implies, and their shapes) and a typed ``make_spec`` builder.
3. Implement ``_build``, registering every parameter in a
   :class:`~flexops.core.registration.CoefficientRegistry`. The base class
   then checks the registry against the contract and applies each
   parameter's value and fixed flag.
4. Override :meth:`~Surrogate.update_state` and the in-place hooks only when
   the type carries state that evolves with a solve.

Regression tooling (flexparameterize) comes after, and produces specs only
through ``make_spec``.

:meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` is the only caller
of :meth:`Surrogate.build`.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.core.registration import value_leaves, value_shape

_log = get_logger(__name__)


def log_parameter_state(target: str, spec: SurrogateSpec, action: str) -> None:
    """Log which of a surrogate's parameters are free.

    INFO when any parameter is free, since a free parameter changes what a
    solve does; DEBUG when all are fixed, the expected case. An array
    parameter is shown with its shape.

    Args:
        target: ``unit.relation`` the surrogate is on.
        spec: The surrogate's spec.
        action: What happened, e.g. ``"built"`` or ``"written"``.
    """

    def label(name: str) -> str:
        shape = value_shape(spec.parameters[name].value)
        return f"{name} {shape}" if shape else name

    free = spec.free_parameters()
    total = len(spec.parameters)
    kind = spec.surrogate_type.value
    if free:
        _log.info(
            "%s: %s surrogate %s, %d of %d parameters free: %s",
            target,
            kind,
            action,
            len(free),
            total,
            ", ".join(label(name) for name in free),
        )
    else:
        _log.debug(
            "%s: %s surrogate %s, all %d parameters fixed", target, kind, action, total
        )


def _config_error(type_name: str, exc: ValidationError) -> FlexConfigError:
    """Turn a config ValidationError into a FlexConfigError naming each field."""
    lines = []
    first_field = None
    for error in exc.errors():
        loc = ".".join(str(part) for part in ("config", *error["loc"]))
        first_field = first_field or loc
        lines.append(f"{loc}: {error['msg']}")
    return FlexConfigError(
        f"Invalid {type_name} surrogate config:\n" + "\n".join(lines),
        field=first_field,
    )


class Surrogate(ABC):
    """Base class of every predefined surrogate-structure class.

    A leaf class in ``flexops`` (imports only ``flexcore``, pydantic and
    Pyomo), so ``flexops.core.ops_block`` can import the registry that
    resolves to it with no cross-package import.

    Attributes:
        surrogate_type: The :class:`~flexcore.config.schema.SurrogateType`
            this class implements.
        config_model: The pydantic model that validates ``config``.
        config_version: The current version of this type's ``config`` format.
            A spec at an older version is migrated by :meth:`migrate_config`
            on construction.
        replaces_relation: ``True`` when :meth:`build` swaps a relation;
            ``False`` for a surrogate that instead sets a parameter the unit
            already has (``constant_intensity``).
        spec: The validated spec, at the current ``config_version``. Its
            ``config`` reflects any :meth:`update_state` since construction;
            its ``parameters`` are the values the surrogate was built with.
            Read the live values with :meth:`read_spec`.
        config: ``spec.config`` parsed by :attr:`config_model`.
    """

    surrogate_type: ClassVar[SurrogateType]
    config_model: ClassVar[type[BaseModel]]
    config_version: ClassVar[int] = 1
    replaces_relation: ClassVar[bool] = True

    def __init__(self, spec: SurrogateSpec | dict) -> None:
        """Migrate and validate ``spec`` against this class's contract.

        Args:
            spec: A :class:`~flexcore.config.schema.SurrogateSpec`, or a dict
                that validates as one, whose ``surrogate_type`` is this
                class's. Build one with this class's ``make_spec``.

        Raises:
            FlexConfigError: If ``spec`` is for another type, uses the
                pre-0.2.0 ``data`` layout, has a newer ``config_version`` than
                this build supports, or does not match this class's contract.
            NotImplementedError: If this class is not yet implemented.
        """
        spec = self._coerce(spec)
        if spec.surrogate_type is not self.surrogate_type:
            raise FlexConfigError(
                f"{type(self).__name__} implements surrogate_type "
                f"{self.surrogate_type.value!r}, but the spec is for "
                f"{spec.surrogate_type.value!r}; use surrogate_from_spec to pick "
                "the class from the spec.",
                field="surrogate_type",
                value=spec.surrogate_type,
            )
        spec = self._migrate(spec)
        try:
            config = self.config_model.model_validate(spec.config)
        except ValidationError as exc:
            raise _config_error(self.surrogate_type.value, exc) from exc
        self._check_parameters(config, spec)
        self._validate(config, spec)
        self.spec = spec
        self.config = config

    @classmethod
    def _coerce(cls, spec: SurrogateSpec | dict) -> SurrogateSpec:
        if isinstance(spec, SurrogateSpec):
            return spec
        if isinstance(spec, dict) and "data" in spec:
            raise FlexConfigError(
                f"{cls.__name__} was given a spec in the pre-0.2.0 layout, with "
                "'data'. Surrogate specs now carry 'config' and 'parameters'; "
                "upgrade a saved spec with "
                "flexcore.config.surrogate_migration.migrate_surrogate_spec, or "
                f"build a new one with {cls.__name__}.make_spec.",
                field="data",
            )
        if isinstance(spec, dict):
            try:
                return SurrogateSpec.model_validate(
                    {"surrogate_type": cls.surrogate_type, **spec}
                )
            except ValidationError as exc:
                raise FlexConfigError(
                    f"Invalid {cls.surrogate_type.value} surrogate spec: {exc}"
                ) from exc
        raise FlexConfigError(
            f"{cls.__name__} takes a SurrogateSpec (see {cls.__name__}.make_spec), "
            f"got {type(spec).__name__}.",
            value=spec,
        )

    @classmethod
    def _migrate(cls, spec: SurrogateSpec) -> SurrogateSpec:
        version = spec.config_version
        if version > cls.config_version:
            raise FlexConfigError(
                f"The {cls.surrogate_type.value} surrogate spec has "
                f"config_version {version}, newer than this build supports "
                f"({cls.config_version}). Upgrade flex-pse.",
                field="config_version",
                value=version,
            )
        if version == cls.config_version:
            return spec
        config = copy.deepcopy(spec.config)
        while version < cls.config_version:
            config = cls.migrate_config(version, config)
            version += 1
        return spec.model_copy(
            update={"config": config, "config_version": cls.config_version}
        )

    @classmethod
    def migrate_config(cls, version: int, config: dict) -> dict:
        """Return ``config`` upgraded from ``version`` to ``version + 1``.

        Override when a type's config format changes: bump
        :attr:`config_version` and handle the old version here.

        Args:
            version: The version ``config`` is at.
            config: The config at that version (a copy; may be modified).

        Returns:
            The config at ``version + 1``.

        Raises:
            FlexConfigError: If this type has no migration from ``version``.
        """
        raise FlexConfigError(
            f"No {cls.surrogate_type.value} config migration is registered from "
            f"config_version {version}.",
            field="config_version",
            value=version,
        )

    @classmethod
    @abstractmethod
    def parameter_shapes(cls, config: BaseModel) -> dict[str, tuple[int, ...]]:
        """Return every parameter a config implies and its shape.

        Args:
            config: A validated :attr:`config_model` instance.

        Returns:
            ``{name: shape}`` in registration order, ``()`` for a scalar,
            ``(n,)`` for a vector, ``(n, m)`` for a matrix. A spec's
            ``parameters`` must name exactly these, and ``_build`` must
            register exactly these.

        Raises:
            NotImplementedError: If this class is not yet implemented.
        """

    @classmethod
    def initial_guess(cls, name: str, config: BaseModel) -> Any:
        """Return the starting value of a free parameter given without a value.

        Override to document a type's guesses; the default is ``0.0`` for
        every entry.

        Args:
            name: The parameter.
            config: The validated config.

        Returns:
            A number, or nested lists matching the parameter's shape.
        """
        shape = cls.parameter_shapes(config)[name]
        value: Any = 0.0
        for size in reversed(shape):
            value = [copy.deepcopy(value) for _ in range(size)]
        return value

    @classmethod
    def _check_parameters(cls, config: BaseModel, spec: SurrogateSpec) -> None:
        shapes = cls.parameter_shapes(config)
        given = list(spec.parameters)
        missing = [name for name in shapes if name not in spec.parameters]
        unknown = [name for name in given if name not in shapes]
        if missing or unknown:
            raise FlexConfigError(
                f"{cls.surrogate_type.value} surrogate parameters must name "
                f"exactly {list(shapes)}; missing={missing}, unknown={unknown}.",
                field="parameters",
                value=given,
            )
        for name, state in spec.parameters.items():
            if state.value is None:
                continue
            actual = value_shape(state.value)
            if actual != shapes[name]:
                raise FlexConfigError(
                    f"{cls.surrogate_type.value} surrogate parameter {name!r} "
                    f"must have shape {shapes[name]}, got "
                    f"{'a ragged nesting' if actual is None else actual}.",
                    field=f"parameters.{name}",
                    value=state.value,
                )

    def _validate(  # noqa: B027 - optional hook, empty by design
        self, config: BaseModel, spec: SurrogateSpec
    ) -> None:
        """Type-specific checks beyond the config model and parameter shapes.

        Args:
            config: The validated config.
            spec: The migrated spec.

        Raises:
            FlexConfigError: If the spec breaks a type-specific rule.
        """

    @classmethod
    def validate_spec(cls, spec: SurrogateSpec | dict) -> SurrogateSpec:
        """Return ``spec`` migrated to the current ``config_version``, validated.

        No Pyomo objects are built.

        Args:
            spec: The spec to check.

        Returns:
            The migrated, validated spec.

        Raises:
            FlexConfigError: If it does not match this class's contract.
        """
        return cls(spec).spec

    @classmethod
    def _make(
        cls,
        config: dict,
        values: dict[str, Any],
        fixed: dict[str, bool] | None = None,
        provenance: dict | None = None,
    ) -> SurrogateSpec:
        """Assemble and validate a spec; the shared body of every ``make_spec``.

        Args:
            config: The config.
            values: ``{parameter: value}``. A free parameter may be left out.
            fixed: ``{parameter: False}`` for each free parameter; every
                other parameter is fixed. ``None`` fixes everything.
            provenance: Free-form fit metadata.

        Returns:
            The validated spec.

        Raises:
            FlexConfigError: If ``fixed`` names an unknown parameter, or the
                spec does not match this class's contract.
        """
        fixed = dict(fixed or {})
        names = list(values) + [name for name in fixed if name not in values]
        parameters = {}
        for name in names:
            state: dict[str, Any] = {"fixed": fixed.get(name, True)}
            if name in values:
                state["value"] = values[name]
            parameters[name] = state
        try:
            spec = SurrogateSpec(
                surrogate_type=cls.surrogate_type,
                config_version=cls.config_version,
                config=config,
                parameters=parameters,
                provenance=dict(provenance or {}),
            )
        except ValidationError as exc:
            raise FlexConfigError(
                f"Invalid {cls.surrogate_type.value} surrogate spec: {exc}",
                field="parameters",
            ) from exc
        return cls.validate_spec(spec)

    def value_of(self, name: str) -> Any:
        """Return the value a parameter is built with: given, or the guess."""
        state = self.spec.parameters[name]
        if state.value is not None:
            return state.value
        return self.initial_guess(name, self.config)

    @property
    @abstractmethod
    def input_variables(self) -> dict[str, str]:
        """Return this relationship's input variable names and declared units."""

    @property
    @abstractmethod
    def output_variables(self) -> dict[str, str]:
        """Return this relationship's output variable names and declared units."""

    def build(self, unit, target) -> tuple[Any | None, Any]:
        """Return ``(block, body(t))``, with every parameter set from the spec.

        Calls ``_build``, checks that the block's coefficient registry
        holds exactly the contract's parameters with their shapes, then sets
        each parameter's value and fixes or frees it as the spec says.

        Args:
            unit: The :class:`~flexops.core.ops_block.OpsBlockData` the
                relationship is built on.
            target: The Var/Reference the relationship determines.

        Returns:
            ``(block, body)``; see ``_build``.

        Raises:
            FlexConfigError: If the built registry does not match the
                contract (a bug in the surrogate class).
        """
        block, body = self._build(unit, target)
        registry = getattr(block, "coefficients", None)
        expected = self.parameter_shapes(self.config)
        if expected or registry is not None:
            built = {} if registry is None else {n: registry.shape(n) for n in registry}
            if built != expected:
                raise FlexConfigError(
                    f"{type(self).__name__}._build registered {built}, but its "
                    f"contract declares {expected}; the class's build and "
                    "parameter_shapes disagree.",
                    field="parameters",
                )
            self.apply_parameters(registry, self.spec.parameters)
        return block, body

    def apply_parameters(self, registry, parameters) -> None:
        """Set each parameter's value and fixed flag on a built registry.

        Args:
            registry: The block's
                :class:`~flexops.core.registration.CoefficientRegistry`.
            parameters: ``{name: ParameterState}``; a free parameter without
                a value keeps its current value.
        """
        for name, state in parameters.items():
            if state.value is not None:
                registry.set_value(name, state.value)
            if state.fixed:
                registry.fix(name)
            else:
                registry.unfix(name)

    @abstractmethod
    def _build(self, unit, target) -> tuple[Any | None, Any]:
        """Build the relationship; called by :meth:`build`.

        Create the parameter Vars (initialized from :meth:`value_of`) and
        register each in a ``CoefficientRegistry`` at ``block.coefficients``;
        :meth:`build` sets their values and fixed flags afterwards.

        Args:
            unit: The :class:`~flexops.core.ops_block.OpsBlockData` the
                relationship is built on; resolve input variables through
                ``unit.resolve_variable``.
            target: The Var/Reference the relationship determines.

        Returns:
            A tuple ``(block, body)`` where ``block`` is a Pyomo ``Block`` the
            surrogate constructs (or ``None`` if this surrogate needs no
            auxiliary Vars). ``swap_relation`` adds the block to ``unit``
            itself; this method must not touch ``unit.add_component``.
            ``body`` is a callable taking a time index and returning a
            units-carrying Pyomo expression in this relationship's declared
            output units, or ``pyomo.environ.Constraint.Skip`` to omit that
            index.
        """

    def read_spec(self, block, target) -> SurrogateSpec:
        """Return this surrogate's current spec, read from its built block.

        ``config`` is :attr:`spec`'s (including any :meth:`update_state`);
        each parameter's value and fixed flag come from the block's registry.
        This changes nothing.

        Args:
            block: The block :meth:`build` returned.
            target: The target passed to :meth:`build`.

        Returns:
            The spec, which rebuilds this surrogate as it stands.
        """
        del target
        registry = getattr(block, "coefficients", None)
        parameters = {}
        for name in self.parameter_shapes(self.config):
            value = registry.value(name)
            fixed = registry.is_fixed(name)
            if None in value_leaves(value):
                # A free Var no solve has set: it has no value to write, so
                # a rebuild starts it from the initial guess again.
                parameters[name] = {"fixed": False}
            else:
                parameters[name] = {"value": value, "fixed": fixed}
        return self.spec.model_copy(
            update={
                "config": copy.deepcopy(self.spec.config),
                "parameters": SurrogateSpec.model_validate(
                    {"surrogate_type": self.surrogate_type, "parameters": parameters}
                ).parameters,
            }
        )

    def update_state(self, block, target) -> None:
        """Record solved state into :attr:`spec`'s ``config``; call after a solve.

        The default does nothing: a type without time-dependent state has
        nothing to record. A type with state (ARIMA's history) rewrites its
        config so the spec initializes the next model from the solve.

        Args:
            block: The block :meth:`build` returned, after a solve.
            target: The target passed to :meth:`build`.
        """
        del block, target

    def can_update_in_place(self, spec: SurrogateSpec) -> bool:
        """Return whether ``spec`` can be applied to the built block in place.

        The default allows it when ``spec`` has this surrogate's type, config
        version and config, so only parameter values and fixed flags differ.

        Args:
            spec: A validated spec for the same relation.

        Returns:
            ``True`` to update in place; ``False`` to swap a new block in.
        """
        return (
            spec.surrogate_type is self.surrogate_type
            and spec.config_version == self.spec.config_version
            and spec.config == self.spec.config
        )

    def update_in_place(self, block, spec: SurrogateSpec) -> None:
        """Apply ``spec``'s parameters to the built block.

        Called only when :meth:`can_update_in_place` is true.

        Args:
            block: The block :meth:`build` returned.
            spec: The validated spec to apply.
        """
        self.apply_parameters(block.coefficients, spec.parameters)
        self.spec = self.spec.model_copy(
            update={"parameters": spec.parameters, "provenance": spec.provenance}
        )
