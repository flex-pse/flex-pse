"""Registration records and model-wide discovery for flex-pse units.

Every :class:`~flexops.core.ops_block.OpsBlockData` holds an :class:`IORegistry`
of what it exposes to FlexParameterize and the docs generator: its process IO
variables, its regressable parameters, its power-draw variables (kW), its
fuel-usage variables (volumetric flows, m³/hr), and the boundary flows it meters
into or out of the facility. The record
dataclasses hold **live** Pyomo references (typed ``Any`` — a Pyomo component
has no useful static type here). :func:`iter_io_registry` walks a whole model to
find every block that registered something.
"""

import enum
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from flexcore.exceptions import FlexConfigError
from flexcore.nomenclature import PowerKind


def value_leaves(value: Any) -> list[Any]:
    """Return the scalars of a number or nested list, row-major."""
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in value_leaves(item)]
    return [value]


def value_shape(value: Any) -> tuple[int, ...] | None:
    """Return the shape of a number or rectangular nested list, else ``None``.

    Args:
        value: A number, a list, or nested lists.

    Returns:
        ``()`` for a number, ``(n,)`` for a list of numbers, ``(n, m)`` for a
        list of ``n`` equal-length lists, and so on; ``None`` for a ragged or
        empty-inner nesting, whose shape is ambiguous.
    """
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    shapes = {value_shape(item) for item in value}
    if len(shapes) != 1 or None in shapes:
        return None
    return (len(value), *shapes.pop())


def _nest(flat: list[Any], shape: tuple[int, ...]) -> Any:
    """Fold a row-major flat list into nested lists of ``shape``."""
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    stride = len(flat) // shape[0]
    return [
        _nest(flat[i * stride : (i + 1) * stride], shape[1:]) for i in range(shape[0])
    ]


class CoefficientRegistry:
    """A dict-like ``name -> Var`` container for a surrogate block's coefficients.

    Each registered name is one surrogate parameter, the unit a spec's
    ``parameters`` addresses. A parameter is either a scalar (a scalar Var or
    one entry of an indexed Var) or an array (a whole indexed Var, one name
    for many values, e.g. an ML weight matrix). An array's values map onto the
    Var's index set in its iteration order, read row-major for nested lists;
    one ``fixed`` flag covers the whole array.

    To register every entry of an indexed Var as its own scalar parameter,
    named by its index, pass it to :meth:`register_coefficients`.

    The registry dynamically expands as coefficients are registered. It is
    attached to every surrogate block as ``block.coefficients`` before
    ``build()`` returns, so a developer can call ``register_coefficient``
    multiple times in a single ``build()`` method — useful when coefficients
    come from several independent Var groups (e.g. separate low-flow and
    high-flow regimes).
    """

    def __init__(self) -> None:
        self._vars: dict[Any, Any] = {}
        self._shapes: dict[Any, tuple[int, ...]] = {}

    def register_coefficient(
        self, name: str, var: Any, *, shape: tuple[int, ...] | None = None
    ) -> None:
        """Add one named parameter: a scalar Var entry, or a whole indexed Var.

        Args:
            name: The parameter name used by specs, ``update_parameters`` and
                ``register_surrogate_coefficients``.
            var: A scalar Pyomo Var or one entry of an indexed Var (a scalar
                parameter), or a whole indexed Var (an array parameter).
            shape: An array parameter's shape, e.g. ``(3, 2)``. Defaults to
                ``(len(var),)`` for an indexed Var; must be ``()`` or omitted
                for a scalar.

        Raises:
            FlexConfigError: If ``name`` is already registered, ``var`` is
                not a Pyomo Var, or ``shape`` does not match ``var``'s size.
        """
        import math

        import pyomo.environ as pyo
        from pyomo.core.base.var import VarData

        if name in self._vars:
            raise FlexConfigError(
                f"Coefficient {name!r} is already registered on this block."
            )
        if not isinstance(var, (pyo.Var, VarData)):
            raise FlexConfigError(
                f"Coefficient must be a pyo.Var or pyo.VarData, "
                f"got {type(var).__name__}."
            )
        if var.is_indexed():
            shape = (len(var),) if shape is None else tuple(shape)
            if not shape or math.prod(shape) != len(var):
                raise FlexConfigError(
                    f"Array coefficient {name!r} has {len(var)} entries, which "
                    f"does not fill shape {shape}.",
                    field=name,
                    value=shape,
                )
        elif shape not in (None, ()):
            raise FlexConfigError(
                f"Coefficient {name!r} is a scalar Var, so it cannot have "
                f"shape {shape}.",
                field=name,
                value=shape,
            )
        self._vars[name] = var
        self._shapes[name] = shape if var.is_indexed() else ()

    def register_coefficients(self, mapping) -> None:
        """Bulk-add scalar parameters from a name->Var mapping or an indexed Var.

        Args:
            mapping: Either a dict of parameter name to Pyomo Var, or a single
                indexed ``pyo.Var`` whose entries each become a scalar
                parameter named by its index.

        Raises:
            FlexConfigError: If any name is already registered or any value
                is not a Pyomo Var or Var entry.
        """
        if hasattr(mapping, "is_indexed") and mapping.is_indexed():
            for idx in mapping.index_set():
                self.register_coefficient(idx, mapping[idx])
            return
        for name, var in mapping.items():
            self.register_coefficient(name, var)

    def items(self):
        """Return the registered (name, Var) pairs; an array's Var is indexed."""
        return self._vars.items()

    def names(self) -> list[str]:
        """Return the registered parameter names, in registration order."""
        return list(self._vars)

    def shape(self, name: str) -> tuple[int, ...]:
        """Return a parameter's shape: ``()`` for a scalar."""
        return self._shapes[name]

    def _entries(self, name: str) -> list[Any]:
        var = self._vars[name]
        return [var[index] for index in var] if var.is_indexed() else [var]

    def value(self, name: str) -> Any:
        """Return a parameter's current value: a float, or nested lists of floats.

        An entry whose Var has no value yet reads as ``None``.
        """
        flat = [
            None if entry.value is None else float(entry.value)
            for entry in self._entries(name)
        ]
        return _nest(flat, self._shapes[name])

    def set_value(self, name: str, value: Any) -> None:
        """Set a parameter's value without changing whether it is fixed.

        Args:
            name: A registered parameter.
            value: A number for a scalar; for an array, nested lists matching
                its shape.

        Raises:
            FlexConfigError: If ``value``'s shape does not match the
                parameter's.
        """
        shape = self._shapes[name]
        if value_shape(value) != shape:
            raise FlexConfigError(
                f"Parameter {name!r} has shape {shape}, but the value given has "
                f"shape {value_shape(value)}.",
                field=name,
                value=value,
            )
        for entry, leaf in zip(self._entries(name), value_leaves(value), strict=True):
            entry.set_value(leaf)

    def is_fixed(self, name: str) -> bool:
        """Return whether every entry of a parameter is fixed."""
        return all(entry.fixed for entry in self._entries(name))

    def is_partially_fixed(self, name: str) -> bool:
        """Return whether an array parameter has some entries fixed and some free."""
        states = {entry.fixed for entry in self._entries(name)}
        return len(states) > 1

    def __getitem__(self, name: str) -> Any:
        return self._vars[name]

    def __contains__(self, name: str) -> bool:
        return name in self._vars

    def __iter__(self):
        return iter(self._vars)

    def __len__(self) -> int:
        return len(self._vars)

    def unfix(self, name: str | None = None) -> None:
        """Unfix one parameter, or every registered parameter when ``name`` is None."""
        for key in self._vars if name is None else [name]:
            for entry in self._entries(key):
                entry.unfix()

    def fix(self, name: str | None = None) -> None:
        """Fix one parameter, or every one when ``name`` is None, at its value."""
        for key in self._vars if name is None else [name]:
            for entry in self._entries(key):
                entry.fix()


class BoundaryKind(enum.StrEnum):
    """Which side of a facility boundary a registered flow crosses."""

    FEED = "feed"
    PRODUCT = "product"


@dataclass
class IOVariableRecord:
    """A registered process input/output variable.

    Attributes:
        var: The live Pyomo ``Var``.
        name: The variable's local name on its unit block.
        role: ``"input"`` or ``"output"``.
        tag_hint: Optional historian-tag hint for FlexParameterize aliasing.
        units: The variable's units as a string.
        time_indexed: Whether the variable is indexed over the time set.
    """

    var: Any
    name: str
    role: str
    tag_hint: str | None
    units: str
    time_indexed: bool


@dataclass
class ParameterRecord:
    """A registered design/regression parameter.

    Attributes:
        param: The live Pyomo ``Param`` or ``Var``.
        name: The parameter's local name on its unit block.
        regressable: Whether FlexParameterize may fit this parameter.
        relation_name: The relation this parameter belongs to, or ``None`` for
            unit-level process parameters that are not tied to a specific
            swapped relation.
    """

    param: Any
    name: str
    regressable: bool
    relation_name: str | None = None


@dataclass
class PowerRecord:
    """A registered power-draw variable.

    Attributes:
        var: The live Pyomo ``Var`` (kW).
        name: The nomenclature constant value (e.g. ``"power_electrical"``).
        kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.
        temperature: The heat duty's temperature (a unit-carrying value); set
            only when ``kind is PowerKind.THERMAL``, else ``None``.
    """

    var: Any
    name: str
    kind: PowerKind
    temperature: Any | None = None


@dataclass
class FuelUsageRecord:
    """A registered fuel-usage variable — a volumetric flow, not a power.

    Attributes:
        var: The live Pyomo ``Var`` (a volumetric rate, convertible to m³/hr).
        name: The variable's local name on its unit block.
        fuel_name: The fuel's name (e.g. ``"natural_gas"``), the key its flow
            aggregates and bills under.
    """

    var: Any
    name: str
    fuel_name: str


@dataclass
class RelationRecord:
    """A non-conservation relationship a unit has declared swappable.

    Registered via
    :meth:`~flexops.core.ops_block.OpsBlockData.register_relation` — an
    unregistered constraint (a mass balance, a conservation law) can never be
    swapped, so this list is what
    :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` may act on and
    what :func:`iter_swapped_relations` reports over.

    Attributes:
        constraint: The live, originally-built Constraint.
        name: Its local name — the string ``swap_relation`` is called with.
        target: The live Var/Reference the relationship determines.
        target_name: ``target``'s local name.
        fitted: The Constraint a swap attached, replacing ``constraint``;
            ``None`` until a swap has happened.
        components: Any Vars/Constraints a builder attached while fitting
            ``fitted`` (e.g. an auxiliary variable a state-space or big-M form
            needs); deactivated on the next swap, alongside ``fitted`` itself.
        swap_count: How many times this relation has been swapped; used to
            keep each successive ``fitted`` Constraint's name unique (flex-pse
            never deletes a component, so a second swap cannot reuse the first
            fitted Constraint's name).
        surrogate_block: The currently active surrogate sub-block (a Pyomo
            Block carrying coefficient Vars and the fitted constraint), or
            ``None`` when no surrogate has been swapped for this relation.
        surrogate_blocks: Every surrogate block ever built for this relation,
            oldest first. The list is append-only; deactivated blocks remain
            here so ``switch_surrogate_block`` can reactivate them.
        surrogate_instances: The :class:`~flexops.surrogates.base.Surrogate`
            that built each entry of ``surrogate_blocks``, index-aligned with
            it, so ``switch_surrogate_block`` can restore the matching
            ``surrogate_instance``.
        surrogate_instance: The :class:`~flexops.surrogates.base.Surrogate`
            instance currently active for this relation, or ``None`` when
            ``surrogate_block`` is ``None`` (no swap yet, or the active
            surrogate built no block). Used by
            :meth:`~flexops.core.ops_block.OpsBlockData.get_surrogate_spec`
            to read its spec back (``read_spec``) without depending on block
            attributes; it is the one source of the relation's spec.
    """

    constraint: Any
    name: str
    target: Any
    target_name: str
    fitted: Any = None
    components: list = field(default_factory=list)
    swap_count: int = 0
    surrogate_block: Any = None
    surrogate_blocks: list = field(default_factory=list)
    surrogate_instances: list = field(default_factory=list)
    surrogate_instance: Any = None


@dataclass
class BoundaryRecord:
    """A registered boundary flow — a resource entering or leaving a facility.

    Attributes:
        var: The live Pyomo ``Var`` carrying the total flow across the
            boundary, indexed over the time set.
        name: The variable's local name on its unit block.
        resource: The resource's name (e.g. ``"raw_water"``, ``"brine"``), the
            key its flow aggregates under. Two blocks sharing one name sum into
            a single row.
        kind: Whether the flow enters (``BoundaryKind.FEED``) or leaves
            (``BoundaryKind.PRODUCT``) the facility.
    """

    var: Any
    name: str
    resource: str
    kind: BoundaryKind


@dataclass
class IORegistry:
    """Container for everything a unit block registers.

    Attributes:
        io_variables: Registered process IO variables.
        parameters: Registered design/regression parameters.
        power: Registered power-draw variables (kW).
        fuel: Registered fuel-usage variables (volumetric).
        intensity_basis: Per :class:`~flexcore.nomenclature.PowerKind`, the
            local name of the product flow the unit's constant-intensity
            relation meters against. FlexParameterize reads it to regress the
            intensity against the same stream the model divides by; without it
            a unit with several flows would have to be guessed at.
        relations: Registered swappable relationships (see
            :class:`RelationRecord`).
        boundary: Registered boundary flows (feeds and products).
        applied_surrogates: Per relation name, the surrogate applied without a
            swap (a ``constant_intensity`` that sets the unit's own intensity),
            so its spec can be read back; swapped surrogates live on their
            :class:`RelationRecord` instead.
    """

    io_variables: list[IOVariableRecord] = field(default_factory=list)
    parameters: list[ParameterRecord] = field(default_factory=list)
    power: list[PowerRecord] = field(default_factory=list)
    fuel: list[FuelUsageRecord] = field(default_factory=list)
    intensity_basis: dict[PowerKind, str] = field(default_factory=dict)
    relations: list[RelationRecord] = field(default_factory=list)
    boundary: list[BoundaryRecord] = field(default_factory=list)
    applied_surrogates: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True if nothing has been registered on this block."""
        return not (
            self.io_variables
            or self.parameters
            or self.power
            or self.fuel
            or self.relations
            or self.boundary
        )


def iter_io_registry(model) -> Iterator[tuple[Any, IORegistry]]:
    """Yield every block on ``model`` that exposes a non-empty registry.

    Walks ``model`` and all its sub-blocks and yields ``(block, registry)`` for
    each block carrying a non-empty ``_io_registry`` attribute, giving
    FlexParameterize and the docs generator model-wide discoverability.

    Args:
        model: The Pyomo model (or block) to walk.

    Yields:
        ``(block, registry)`` pairs, each block yielded at most once.
    """
    seen: set[int] = set()
    blocks = [model, *model.block_data_objects(descend_into=True)]
    for block in blocks:
        if id(block) in seen:
            continue
        seen.add(id(block))
        registry = getattr(block, "_io_registry", None)
        if isinstance(registry, IORegistry) and not registry.is_empty():
            yield block, registry


def iter_swapped_relations(model) -> Iterator[tuple[Any, RelationRecord]]:
    """Yield every relation on ``model`` that has actually been swapped.

    A debugging and reporting aid, not a required step on any build or apply
    path: it answers "what in this model differs from its defaults?" for a
    model built any way — from config, by hand, or by
    ``flexparameterize.apply_to_model`` — not only for the call that changed
    it. Cheap by construction: it reads the ``fitted`` field
    :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation` already set on
    each :class:`RelationRecord`, rather than re-deriving anything from
    constraint bodies.

    Deliberately out of scope: detecting a relationship altered some other
    way (hand-editing a constraint's rule, rebuilding a component outside
    ``swap_relation``) would mean constructing a shadow default unit and
    diffing constraint bodies — a "deep audit" left to a future milestone.

    Args:
        model: The Pyomo model (or block) to walk.

    Yields:
        ``(block, RelationRecord)`` pairs for every swapped relation.
    """
    for block, registry in iter_io_registry(model):
        for record in registry.relations:
            if record.fitted is not None:
                yield block, record
