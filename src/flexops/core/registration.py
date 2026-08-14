"""Registration records and model-wide discovery for flex-pse units.

Every :class:`~flexops.core.ops_block.OpsBlockData` holds an :class:`IORegistry`
of what it exposes to FlexParameterize and the docs generator: its process IO
variables, its regressable parameters, its power-draw variables (kW), and its
fuel-usage variables (volumetric flows, m³/hr). The record
dataclasses hold **live** Pyomo references (typed ``Any`` — a Pyomo component
has no useful static type here). :func:`iter_io_registry` walks a whole model to
find every block that registered something.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from flexcore.nomenclature import PowerKind


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
    """

    param: Any
    name: str
    regressable: bool


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
    """

    constraint: Any
    name: str
    target: Any
    target_name: str
    fitted: Any = None
    components: list = field(default_factory=list)
    swap_count: int = 0


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
    """

    io_variables: list[IOVariableRecord] = field(default_factory=list)
    parameters: list[ParameterRecord] = field(default_factory=list)
    power: list[PowerRecord] = field(default_factory=list)
    fuel: list[FuelUsageRecord] = field(default_factory=list)
    intensity_basis: dict[PowerKind, str] = field(default_factory=dict)
    relations: list[RelationRecord] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True if nothing has been registered on this block."""
        return not (
            self.io_variables
            or self.parameters
            or self.power
            or self.fuel
            or self.relations
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
