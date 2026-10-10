"""The flat, order-agnostic flowsheet spec (pydantic v2).

A flowsheet is one flat list of elements. An element's ``name`` is the full
dotted path of the Pyomo component it creates, and elements refer to each other
by name, so the order of the list never matters: the assembler decides the build
order from each element's kind. The nested
:class:`~flexcore.config.schema.ModelConfig` is the legacy input format; see
:mod:`flexcore.config.convert`.

Class docstrings and field descriptions here are exported verbatim into the
JSON Schema, so keep them plain text.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from flexcore.config.schema import (
    CostingConfig,
    DRConfig,
    IOVariableSpec,
    PriceSpec,
    PropertyPackageSpec,
    SurrogateType,
    TimeConfig,
    UnitCommitmentConfig,
    _StrictModel,
)

SCHEMA_VERSION = "0.1.0"
"""str: the schema version this build writes and validates against."""

KINDS: dict[str, tuple[str, bool]] = {
    "time": ("declare", False),
    "property_package": ("declare", False),
    "costing": ("declare", False),
    "network": ("topology", False),
    "plant": ("topology", False),
    "set": ("topology", False),
    "unit": ("topology", False),
    "surrogate": ("surrogates", True),
    "dispatch": ("state", True),
    "objective": ("costing", False),
}
"""Element kind -> (build stage, whether it can change on a live model)."""

STAGE_ORDER = tuple(dict.fromkeys(stage for stage, _ in KINDS.values()))
"""Build stages in the order the registry lists them."""

AUTO = "auto"
"""Value of a unit's package field that means the model's only package."""

PACKAGE_REF = "$package"
"""Key of a construction option ``{"$package": name}`` naming a package element."""


def _like(model: type[_StrictModel], name: str):
    """Return a Field with the default and description of ``model``'s field."""
    info = model.model_fields[name]
    return Field(
        default=info.get_default(call_default_factory=True),
        description=info.description,
    )


class SourceRef(_StrictModel):
    """Data kept in a JSON or gzipped JSON file beside the spec."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    source: str = Field(
        alias="$source",
        description="Path of a .json or .json.gz file, relative to the spec file.",
    )


class SourcedPrice(PriceSpec):
    """A native price whose values may sit in a file beside the spec."""

    value: float | list[float] | SourceRef = Field(
        description="The numeric price: one number, one per time point, or a "
        "$source reference to a file holding that list."
    )


class TimeElement(TimeConfig):
    """The discrete-time horizon; builds the model's time_block."""

    kind: Literal["time"]


class PropertyPackageElement(PropertyPackageSpec):
    """A property package, built at the model root under its name."""

    kind: Literal["property_package"]
    name: str = Field(description="Name of the package on the model.")


class CostingElement(_StrictModel):
    """The costing package: tariff, prices, demand response and finance terms."""

    kind: Literal["costing"]
    name: str = Field(description="Name of the costing block on the model.")
    tariff_source: str | list[str] | dict[str, str] | None = _like(
        CostingConfig, "tariff_source"
    )
    energy_prices: dict[str, SourcedPrice] | None = _like(
        CostingConfig, "energy_prices"
    )
    currency: str = _like(CostingConfig, "currency")
    dr: DRConfig | None = _like(CostingConfig, "dr")
    consumption_estimate: dict[Literal["electric", "gas"], float] | None = _like(
        CostingConfig, "consumption_estimate"
    )
    fixed_operating_cost: float = _like(CostingConfig, "fixed_operating_cost")
    prorate_monthly_charges: bool = _like(CostingConfig, "prorate_monthly_charges")
    lifetime_years: float = _like(CostingConfig, "lifetime_years")
    discount_rate: float = _like(CostingConfig, "discount_rate")
    interest_rate: float | None = _like(CostingConfig, "interest_rate")

    @model_validator(mode="after")
    def _some_pricing_source(self) -> "CostingElement":
        """Require a tariff or at least one price."""
        if self.tariff_source is None and not self.energy_prices:
            raise ValueError(
                f"costing {self.name!r} needs a pricing source: set "
                "tariff_source, or give at least one entry in energy_prices."
            )
        return self


class NetworkElement(_StrictModel):
    """A network of plants."""

    kind: Literal["network"]
    name: str = Field(description="Dotted path of the network on the model.")


class PlantElement(_StrictModel):
    """A plant of units."""

    kind: Literal["plant"]
    name: str = Field(description="Dotted path of the plant on the model.")


class SetElement(_StrictModel):
    """A set of members that units and connections can be indexed over."""

    kind: Literal["set"]
    name: str = Field(description="Dotted path of the set on the model.")
    values: list[int | str] = Field(
        description="The set's members, in order; non-empty and unique."
    )

    @model_validator(mode="after")
    def _values_nonempty_and_unique(self) -> "SetElement":
        """Reject an empty or repeating member list."""
        if not self.values or len(set(self.values)) != len(self.values):
            raise ValueError(
                f"set {self.name!r} needs a non-empty list of unique values, "
                f"got {self.values}."
            )
        return self


class Connection(_StrictModel):
    """An arc from a port of the unit that owns it to another unit's port."""

    port: str = Field(description="Port on the owning unit, e.g. 'outlet'.")
    to: str = Field(
        description="Destination as '<unit element name>.<port path>'; may "
        "contain '{i}' for an indexed connection."
    )
    name: str | None = Field(
        default=None,
        description="Name of the arc on the model; omitted: derived from the "
        "endpoints.",
    )
    doc: str | None = Field(default=None, description="Optional arc description.")
    index: str | None = Field(
        default=None,
        description="Name of a set element to index the arc over; omitted: the "
        "owning unit's index, if it has one.",
    )
    directed: bool = Field(
        default=True,
        description="False builds an undirected arc between the two ports.",
    )


class UnitElement(_StrictModel):
    """A unit model and the connections that leave it."""

    kind: Literal["unit"]
    name: str = Field(description="Dotted path of the unit on the model.")
    unit_model_class: str = Field(
        description="Name of the flexops unit-model class to construct."
    )
    construction_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword options passed to the unit-model constructor. A "
        "value {'$package': name} passes the named property_package element.",
    )
    property_package: str | None = Field(
        default=AUTO,
        description="Name of a property_package element. 'auto' means the "
        "model's only one (an error if it has several); null gives the unit "
        "none.",
    )
    costing_package: str | None = Field(
        default=AUTO,
        description="Name of a costing element. 'auto' means the model's only "
        "one; null leaves the unit uncosted.",
    )
    index: str | None = Field(
        default=None,
        description="Name of a set element; the unit is built once per member.",
    )
    connections: list[Connection] = Field(
        default_factory=list,
        description="Arcs leaving this unit. Each arc is written once, on its "
        "source unit.",
    )
    io_variables: list[IOVariableSpec] = Field(
        default_factory=list,
        description="Declared process input/output variables of the unit.",
    )
    unit_commitment: UnitCommitmentConfig | None = Field(
        default=None,
        description="Per-unit unit-commitment configuration. Unset (null) "
        "leaves the unit model's own default in force.",
    )


class SurrogateElement(_StrictModel):
    """A fitted relationship swapped into a unit after it is built."""

    kind: Literal["surrogate"]
    unit: str = Field(description="Name of the unit element it applies to.")
    relation: str = Field(
        default="power_electrical_relation",
        description="Name of the unit's relation to replace.",
    )
    surrogate_type: SurrogateType = Field(
        description="Which predefined surrogate class the data describes."
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="The relationship's data, in the shape surrogate_type's "
        "class defines.",
    )
    provenance: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form fit metadata (metrics, data window, versions).",
    )
    name: str | None = Field(
        default=None, description="Optional label; not used to build anything."
    )


class DispatchElement(_StrictModel):
    """Values an external controller sets for one variable of a unit."""

    kind: Literal["dispatch"]
    unit: str = Field(description="Name of the unit element it applies to.")
    variable: str = Field(description="Name of the time-indexed variable on the unit.")
    values: list[float] | dict[str, float] | SourceRef = Field(
        description="One value per time point as a list, a mapping of time "
        "index to value, or a $source reference to a file holding either."
    )
    fix: bool = Field(
        default=True,
        description="Whether to fix the variable to the values (removing its "
        "degrees of freedom).",
    )
    name: str | None = Field(
        default=None, description="Optional label; not used to build anything."
    )


class ObjectiveElement(_StrictModel):
    """The objective the model optimizes."""

    kind: Literal["objective"]
    name: str = Field(default="objective", description="Name on the model.")
    expression: Literal["cost"] = Field(
        default="cost", description="Quantity to optimize (only cost today)."
    )
    costing_package: str = Field(
        default=AUTO,
        description="Name of the costing element whose cost is optimized; "
        "'auto' means the model's only one.",
    )
    sense: Literal["minimize", "maximize"] = Field(
        default="minimize", description="Direction of optimization."
    )


Element = Annotated[
    TimeElement
    | PropertyPackageElement
    | CostingElement
    | NetworkElement
    | PlantElement
    | SetElement
    | UnitElement
    | SurrogateElement
    | DispatchElement
    | ObjectiveElement,
    Field(discriminator="kind"),
]
"""One entry of a spec's element list, told apart by its ``kind``."""

NAMED_KINDS = (
    "property_package",
    "costing",
    "network",
    "plant",
    "set",
    "unit",
    "objective",
)
"""Kinds whose element name must be unique."""

PARENT_KINDS: dict[str, tuple[str, ...]] = {
    "network": (),
    "plant": ("network",),
    "set": ("plant", "network"),
    "unit": ("plant", "network"),
}
"""Kinds of element each kind may sit under; the model root is always allowed."""


def element_target(element) -> tuple[str, ...]:
    """Return what a surrogate or dispatch sets, ``(unit, relation|variable)``.

    Args:
        element: Any spec element.

    Returns:
        The tuple for a surrogate or dispatch, an empty tuple for other kinds.
    """
    if element.kind == "surrogate":
        return (element.unit, element.relation)
    if element.kind == "dispatch":
        return (element.unit, element.variable)
    return ()


def package_refs(value) -> list[str]:
    """Return every package name a construction-option value references.

    Args:
        value: A construction-option value (any JSON data).

    Returns:
        The names in each ``{"$package": name}`` found at any depth.
    """
    if isinstance(value, dict):
        if set(value) == {PACKAGE_REF}:
            return [value[PACKAGE_REF]]
        return [name for item in value.values() for name in package_refs(item)]
    if isinstance(value, list):
        return [name for item in value for name in package_refs(item)]
    return []


def parent_name(name: str) -> str:
    """Return everything before the last dot of ``name`` ('' for a root name)."""
    return name.rpartition(".")[0]


def find_unit(units: dict, path: str) -> tuple[str, str] | None:
    """Split ``path`` into its unit element name and the port path after it.

    Args:
        units: Unit element names (any container supporting iteration).
        path: A dotted path such as ``'plant.ro_bypass.inlet'``.

    Returns:
        ``(unit name, port path)`` for the longest unit name that prefixes the
        path before a ``.`` or ``[``, or None if there is none.
    """
    for name in sorted(units, key=len, reverse=True):
        if path.startswith((f"{name}.", f"{name}[")):
            return name, path[len(name) :].lstrip(".")
    return None


class FlowsheetSpec(_StrictModel):
    """A flowsheet as a flat list of elements, in any order."""

    schema_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+$",
        description="Semantic schema version of this spec, an X.Y.Z string; "
        "mandatory, no default.",
    )
    name: str = Field(description="Name of the Pyomo ConcreteModel.")
    solver: str | None = Field(
        default=None,
        description="Optional solver name; None lets the facade pick.",
    )
    elements: list[Element] = Field(
        description="Every element of the flowsheet; order is irrelevant."
    )

    _base_dir: Path | None = PrivateAttr(default=None)

    def of_kind(self, *kinds: str) -> list:
        """Return the elements of the given kinds, in file order."""
        return [el for el in self.elements if el.kind in kinds]

    @model_validator(mode="after")
    def _exactly_one_time(self) -> "FlowsheetSpec":
        """Require exactly one time element."""
        count = len(self.of_kind("time"))
        if count != 1:
            raise ValueError(f"A spec needs exactly one 'time' element, found {count}.")
        return self

    @model_validator(mode="after")
    def _names_are_unique_and_well_formed(self) -> "FlowsheetSpec":
        """Require unique, well-formed names, with root-only kinds undotted."""
        seen = set()
        for el in self.of_kind(*NAMED_KINDS):
            label = f"{el.kind} {el.name!r}"
            if el.name in seen or el.name == "time_block":
                raise ValueError(
                    f"{label}: this name is already used by another element."
                )
            seen.add(el.name)
            segments = el.name.split(".")
            if not all(segments) or any(c in el.name for c in "[]{} "):
                raise ValueError(
                    f"{label}: names are dotted paths of plain segments, e.g. "
                    "'plant.tank'; remove empty segments, spaces and brackets."
                )
            if el.kind in ("property_package", "costing", "objective") and (
                len(segments) > 1
            ):
                raise ValueError(
                    f"{label}: this kind lives at the model root; remove the dot."
                )
        return self

    @model_validator(mode="after")
    def _parents_exist(self) -> "FlowsheetSpec":
        """Require each parent to be an element of an allowed kind, or the root."""
        kinds = {el.name: el.kind for el in self.of_kind(*NAMED_KINDS)}
        for el in self.of_kind(*PARENT_KINDS):
            parent = parent_name(el.name)
            if parent and kinds.get(parent) not in PARENT_KINDS[el.kind]:
                allowed = " or ".join(PARENT_KINDS[el.kind]) or "none (root only)"
                raise ValueError(
                    f"{el.kind} {el.name!r}: parent {parent!r} is "
                    f"{kinds.get(parent, 'not an element')}; allowed parent "
                    f"kinds: {allowed}."
                )
        return self

    def _check_package(self, label: str, choice: str | None, kind: str) -> None:
        """Raise unless ``choice`` ('auto', None or a name) resolves among ``kind``."""
        names = [el.name for el in self.of_kind(kind)]
        if choice == AUTO and len(names) != 1:
            raise ValueError(
                f"{label}: {kind} 'auto' needs exactly one {kind} element, but "
                f"there are {len(names)} ({names}). Name one explicitly"
                f"{', or set it to null' if kind != 'objective' else ''}."
            )
        if choice not in (None, AUTO) and choice not in names:
            raise ValueError(
                f"{label}: {kind} {choice!r} is not an element. Available: {names}."
            )

    @model_validator(mode="after")
    def _packages_resolve(self) -> "FlowsheetSpec":
        """Require each unit's and the objective's packages to resolve."""
        for el in self.of_kind("unit"):
            label = f"unit {el.name!r}"
            self._check_package(label, el.property_package, "property_package")
            self._check_package(label, el.costing_package, "costing")
        for el in self.of_kind("objective"):
            self._check_package(f"objective {el.name!r}", el.costing_package, "costing")
        return self

    @model_validator(mode="after")
    def _package_refs_resolve(self) -> "FlowsheetSpec":
        """Require each $package construction option to name a package element."""
        names = {el.name for el in self.of_kind("property_package")}
        for el in self.of_kind("unit"):
            for ref in package_refs(el.construction_options):
                if ref not in names:
                    raise ValueError(
                        f"unit {el.name!r}: construction option {{'$package': "
                        f"{ref!r}}} is not a property_package element. Available: "
                        f"{sorted(names)}."
                    )
        return self

    @model_validator(mode="after")
    def _indexes_name_sets(self) -> "FlowsheetSpec":
        """Require every unit and connection index to name a set element."""
        sets = {el.name for el in self.of_kind("set")}
        for el in self.of_kind("unit"):
            for index in [el.index, *(c.index for c in el.connections)]:
                if index is not None and index not in sets:
                    raise ValueError(
                        f"unit {el.name!r}: index {index!r} is not a set element. "
                        f"Available: {sorted(sets)}."
                    )
        return self

    @model_validator(mode="after")
    def _connections_are_consistent(self) -> "FlowsheetSpec":
        """Require resolvable destinations, valid placeholders and one arc per port."""
        units = {el.name: el for el in self.of_kind("unit")}
        sources, destinations = set(), set()
        for unit in units.values():
            label = f"unit {unit.name!r}"
            for conn in unit.connections:
                if find_unit(units, conn.to) is None:
                    raise ValueError(
                        f"{label}: connection to {conn.to!r} does not start with "
                        f"a unit element name. Units: {sorted(units)}."
                    )
                indexed = conn.index is not None or unit.index is not None
                if not indexed and "{i}" in conn.port + conn.to:
                    raise ValueError(
                        f"{label}: '{{i}}' in connection {conn.port!r} -> "
                        f"{conn.to!r} needs an index on the connection or the unit."
                    )
                source = (unit.name, conn.port)
                if source in sources or conn.to in destinations:
                    raise ValueError(
                        f"{label}: port {conn.port!r} or destination {conn.to!r} "
                        "is already used by another connection; one arc per port."
                    )
                sources.add(source)
                destinations.add(conn.to)
        return self

    @model_validator(mode="after")
    def _surrogates_and_dispatches_name_units(self) -> "FlowsheetSpec":
        """Require unit references to resolve and no (unit, target) to repeat."""
        units = {el.name for el in self.of_kind("unit")}
        seen = set()
        for el in self.of_kind("surrogate", "dispatch"):
            target = el.relation if el.kind == "surrogate" else el.variable
            if el.unit not in units:
                raise ValueError(
                    f"{el.kind} for unit {el.unit!r}: no such unit element. "
                    f"Units: {sorted(units)}."
                )
            if (el.kind, el.unit, target) in seen:
                raise ValueError(
                    f"{el.kind} for unit {el.unit!r}: {target!r} is set twice; "
                    "keep one."
                )
            seen.add((el.kind, el.unit, target))
        return self

    @model_validator(mode="after")
    def _at_most_one_objective(self) -> "FlowsheetSpec":
        """Allow at most one objective element."""
        names = [el.name for el in self.of_kind("objective")]
        if len(names) > 1:
            raise ValueError(
                f"At most one 'objective' element is allowed, found {names}."
            )
        return self
