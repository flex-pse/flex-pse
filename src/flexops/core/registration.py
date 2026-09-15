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


class CoefficientRegistry:
    """A dict-like container for a surrogate block's coefficient Vars.

    Supports both scalar ``pyo.Var`` objects and indexed ``pyo.Var`` objects.
    For indexed Vars, the registry yields individual index entries in
    ``items()``, ``__iter__``, and ``__getitem__`` so callers see a flat
    ``name -> Var`` view regardless of how the underlying Var is stored.

    The registry dynamically expands as coefficients are registered. It is
    attached to every surrogate block as ``block.coefficients`` before
    ``build()`` returns, so a developer can call ``register_coefficient``
    multiple times in a single ``build()`` method — useful when coefficients
    come from several independent Var groups (e.g. separate low-flow and
    high-flow regimes).

    Attributes:
        _scalar_vars: Mapping of coefficient name -> scalar pyo.Var.
        _indexed_vars: Mapping of coefficient name -> indexed pyo.Var whose
            index set holds the individual coefficient entries.
    """

    def __init__(self) -> None:
        self._scalar_vars: dict[str, Any] = {}
        self._indexed_vars: dict[str, Any] = {}

    def _check_var(self, var: Any) -> None:
        import pyomo.environ as pyo
        from pyomo.core.base.var import VarData

        if not isinstance(var, (pyo.Var, VarData)):
            raise FlexConfigError(
                f"Coefficient must be a pyo.Var or pyo.VarData, "
                f"got {type(var).__name__}."
            )

    def register_coefficient(self, name: str, var: Any) -> None:
        """Add a single named coefficient Var.

        Args:
            name: The coefficient name used by ``body(t)`` and
                ``register_surrogate_coefficients``.
            var: The Pyomo Var carrying this coefficient's value. May be a
                scalar Var or an indexed Var; in the latter case the
                individual index entries are surfaced through this registry.

        Raises:
            FlexConfigError: If ``name`` is already registered or ``var``
                is not a ``pyo.Var``.
        """
        if name in self._scalar_vars or name in self._indexed_vars:
            raise FlexConfigError(
                f"Coefficient {name!r} is already registered on this block."
            )
        self._check_var(var)
        if var.is_indexed():
            self._indexed_vars[name] = var
        else:
            self._scalar_vars[name] = var

    def register_coefficients(self, mapping) -> None:
        """Bulk-add coefficients from a name->Var mapping or a single indexed Var.

        Args:
            mapping: Either a dict of coefficient name to Pyomo Var, or a
                single indexed ``pyo.Var`` whose index set supplies the
                coefficient names.

        Raises:
            FlexConfigError: If any name is already registered or any value
                is not a ``pyo.Var``.
        """
        if hasattr(mapping, "is_indexed") and mapping.is_indexed():
            for idx in mapping.index_set():
                self.register_coefficient(idx, mapping[idx])
            return
        for name, var in mapping.items():
            self.register_coefficient(name, var)

    def items(self):
        """Return the registered (name, Var) pairs."""
        yield from self._scalar_vars.items()
        for _parent_name, indexed_var in self._indexed_vars.items():
            for idx in indexed_var.index_set():
                yield idx, indexed_var[idx]

    def __getitem__(self, name: str) -> Any:
        if name in self._scalar_vars:
            return self._scalar_vars[name]
        if name in self._indexed_vars:
            return self._indexed_vars[name]
        for indexed_var in self._indexed_vars.values():
            if name in indexed_var.index_set():
                return indexed_var[name]
        raise KeyError(name)

    def __contains__(self, name: str) -> bool:
        if name in self._scalar_vars:
            return True
        if name in self._indexed_vars:
            return True
        return any(name in iv.index_set() for iv in self._indexed_vars.values())

    def __iter__(self):
        yield from self._scalar_vars
        for indexed_var in self._indexed_vars.values():
            yield from indexed_var.index_set()

    def __len__(self) -> int:
        return len(self._scalar_vars) + sum(
            len(iv.index_set()) for iv in self._indexed_vars.values()
        )

    def unfix(self) -> None:
        """Unfix every registered coefficient Var."""
        for var in self._scalar_vars.values():
            if var.is_variable_type() and var.is_fixed():
                var.unfix()
        for indexed_var in self._indexed_vars.values():
            for idx in indexed_var.index_set():
                entry = indexed_var[idx]
                if entry.is_variable_type() and entry.is_fixed():
                    entry.unfix()

    def fix(self) -> None:
        """Fix every registered coefficient Var at its current value."""
        for var in self._scalar_vars.values():
            if var.is_variable_type():
                var.fix()
        for indexed_var in self._indexed_vars.values():
            for idx in indexed_var.index_set():
                entry = indexed_var[idx]
                if entry.is_variable_type():
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
        surrogate_instance: The :class:`~flexops.surrogates.base.Surrogate`
            instance currently active for this relation, or ``None`` when no
            surrogate has been swapped. Used by
            :meth:`~flexops.core.ops_block.OpsBlockData.get_surrogate_spec`
            to delegate spec extraction without depending on block attributes.
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
    """

    io_variables: list[IOVariableRecord] = field(default_factory=list)
    parameters: list[ParameterRecord] = field(default_factory=list)
    power: list[PowerRecord] = field(default_factory=list)
    fuel: list[FuelUsageRecord] = field(default_factory=list)
    intensity_basis: dict[PowerKind, str] = field(default_factory=dict)
    relations: list[RelationRecord] = field(default_factory=list)
    boundary: list[BoundaryRecord] = field(default_factory=list)

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
