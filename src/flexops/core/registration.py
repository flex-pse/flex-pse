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
class IORegistry:
    """Container for everything a unit block registers.

    Attributes:
        io_variables: Registered process IO variables.
        parameters: Registered design/regression parameters.
        power: Registered power-draw variables (kW).
        fuel: Registered fuel-usage variables (volumetric).
    """

    io_variables: list[IOVariableRecord] = field(default_factory=list)
    parameters: list[ParameterRecord] = field(default_factory=list)
    power: list[PowerRecord] = field(default_factory=list)
    fuel: list[FuelUsageRecord] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return True if nothing has been registered on this block."""
        return not (self.io_variables or self.parameters or self.power or self.fuel)


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
