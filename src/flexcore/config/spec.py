"""The flat, order-agnostic flowsheet spec (pydantic v2).

A flowsheet is one flat list of elements. An element's ``name`` is the full
dotted path of the Pyomo component it creates, and elements refer to each other
by name, so the order of the list never matters: the assembler decides the build
order from each element's kind.

Class docstrings and field descriptions here are exported verbatim into the
JSON Schema, so keep them plain text.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from flexcore.config.schema import (
    CostingConfig,
    DRConfig,
    HistoricalPower,
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
    "ramp": ("ramping", False),
    "status": ("logic", False),
    "startup_shutdown": ("logic", False),
    "dwell": ("logic", False),
    "bypass": ("logic", False),
    "startup_delay": ("logic", False),
    "conditional": ("logic", False),
    "parallel_group": ("logic", False),
    "dispatch": ("state", True),
    "fix": ("state", True),
    "unfix": ("state", True),
    "extension": ("extensions", False),
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


class SourcedHistoricalPower(HistoricalPower):
    """Meter history whose values may sit in a file beside the spec."""

    values_kW: list[float] | SourceRef = Field(
        description="Site power in kW, or a $source reference to a file holding it."
    )


class SourcedDRConfig(DRConfig):
    """A demand-response program whose meter history may sit in a file."""

    historical_power: SourcedHistoricalPower = Field(
        description="Meter history the baseline is computed from."
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
    export_prices: dict[str, SourcedPrice] | None = _like(
        CostingConfig, "export_prices"
    )
    decomposition_type: str | None = _like(CostingConfig, "decomposition_type")
    currency: str = _like(CostingConfig, "currency")
    dr: SourcedDRConfig | None = _like(CostingConfig, "dr")
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


FixPolicy = Literal["error", "warn", "skip"]
"""What to do with a fix value outside its variable's bounds or domain."""


def _variable_path(value: str) -> str:
    """Reject an empty path or one holding whitespace."""
    if not value or any(c.isspace() for c in value):
        raise ValueError(
            f"variable {value!r} must be a non-empty path with no spaces, e.g. "
            "'plant.pump.flow_in[3]'."
        )
    return value


class FixElement(_StrictModel):
    """Fix one or more model variables, named by their path from the model root."""

    kind: Literal["fix"]
    name: str | None = Field(
        default=None, description="Optional id, used in error messages."
    )
    variable: str = Field(
        description="Pyomo ComponentUID path from the model root, e.g. "
        "'plant.pump.flow_in', 'plant.pump.flow_in[3]', 'plant.ro[*].status' or "
        "'plant.u.flow_vol_phase[*,Liq]'. A star (*) matches every index in that "
        "position."
    )
    value: float | list[float] | dict[str, float] | SourceRef = Field(
        description="One number for every matched point; a list with one number "
        "per time point (or per entry of at); a {time index: value} mapping; or a "
        "$source file holding either."
    )
    units: str | None = Field(
        default=None,
        description="Units of value; omitted means each variable's own units.",
    )
    at: list[int] | None = Field(
        default=None,
        description="Only fix these time indices; omitted means every matched point.",
    )
    on_violation: FixPolicy | None = Field(
        default=None,
        description="What to do when a value is outside the variable's bounds or "
        "domain: 'error' raises and fixes nothing; 'warn' fixes out-of-bounds "
        "points anyway and warns; 'skip' leaves violating points unfixed, fixes "
        "the rest, and warns. Omitted means the spec's fix_on_violation.",
    )
    tolerance: float | None = Field(
        default=None,
        ge=0,
        description="Absolute tolerance for the bounds check, in the variable's "
        "units. Omitted means the spec's fix_tolerance.",
    )

    @field_validator("variable")
    @classmethod
    def _variable_is_a_path(cls, value: str) -> str:
        """Reject an empty path or one holding whitespace."""
        return _variable_path(value)

    @model_validator(mode="after")
    def _list_matches_at(self) -> "FixElement":
        """Require a list value to have one entry per entry of ``at``."""
        if isinstance(self.value, list) and self.at is not None:
            if len(self.value) != len(self.at):
                raise ValueError(
                    f"fix {self.variable!r}: value has {len(self.value)} entries "
                    f"but at has {len(self.at)}; give one value per entry of at."
                )
        return self


class UnfixElement(_StrictModel):
    """Unfix one or more model variables (e.g. a parameter a unit fixes by default)."""

    kind: Literal["unfix"]
    name: str | None = Field(
        default=None, description="Optional id, used in error messages."
    )
    variable: str = Field(
        description="ComponentUID path from the model root, as for fix."
    )

    @field_validator("variable")
    @classmethod
    def _variable_is_a_path(cls, value: str) -> str:
        """Reject an empty path or one holding whitespace."""
        return _variable_path(value)


class Quantity(_StrictModel):
    """A number with units, e.g. {"value": 0.5, "units": "kWh/m^3"}."""

    value: float = Field(description="The magnitude.")
    units: str = Field(description="Units as parse_units reads them, e.g. 'kW'.")


def _unit_field():
    return Field(
        description="Name of the unit element, or 'name[i]' for one member of an "
        "indexed unit."
    )


def _variable_field():
    return Field(
        description="Path of the time-indexed variable relative to the unit, e.g. "
        "'flow_in'."
    )


def _label_field():
    return Field(
        default=None, description="Optional label; not used to build anything."
    )


class StatusElement(_StrictModel):
    """An on/off status binary gating a unit's output between two limits."""

    kind: Literal["status"]
    unit: str = _unit_field()
    variable: str = _variable_field()
    min_output: float | Quantity = Field(
        description="Output when on, lower limit: a number or a quantity."
    )
    max_output: float | Quantity = Field(
        description="Output when on, upper limit: a number or a quantity."
    )
    name: str | None = _label_field()


class StartupShutdownElement(_StrictModel):
    """Startup and shutdown binaries with minimum up and down times."""

    kind: Literal["startup_shutdown"]
    unit: str = _unit_field()
    min_up: int = Field(
        default=1, ge=1, description="Minimum steps on after a startup."
    )
    min_down: int = Field(
        default=1, ge=1, description="Minimum steps off after a shutdown."
    )
    name: str | None = _label_field()


class DwellElement(_StrictModel):
    """Hold a continuous variable constant within blocks of k steps."""

    kind: Literal["dwell"]
    unit: str = _unit_field()
    variable: str = _variable_field()
    k: int = Field(ge=1, description="Block length in steps.")
    name: str | None = _label_field()


class BypassElement(_StrictModel):
    """Let part of a flow bypass the unit's energy relation."""

    kind: Literal["bypass"]
    unit: str = _unit_field()
    variable: str = _variable_field()
    max: float | Quantity = Field(
        description="Upper limit on the bypassed flow: a number or a quantity."
    )
    name: str | None = _label_field()


class RampElement(_StrictModel):
    """A limit on how fast a variable may change."""

    kind: Literal["ramp"]
    unit: str = _unit_field()
    variable: str = _variable_field()
    ramp_up: float | None = Field(
        default=None, ge=0, description="Largest increase per step, in units."
    )
    ramp_down: float | None = Field(
        default=None, ge=0, description="Largest decrease per step, in units."
    )
    units: str = Field(description="Units of ramp_up and ramp_down, e.g. 'kW/h'.")
    window: int = Field(
        default=1, ge=1, description="Number of steps the ramp is measured across."
    )
    component_name: str | None = Field(
        default=None,
        description="Stem of the built component names; omitted: '<variable>_ramp'.",
    )
    name: str | None = _label_field()

    @model_validator(mode="after")
    def _some_direction(self) -> "RampElement":
        """Require at least one of ramp_up and ramp_down."""
        if self.ramp_up is None and self.ramp_down is None:
            raise ValueError(
                f"ramp on {self.unit!r}.{self.variable}: give ramp_up, ramp_down "
                "or both."
            )
        return self


class StartupDelayElement(_StrictModel):
    """Allow a unit on only once an upstream unit has been on for k steps."""

    kind: Literal["startup_delay"]
    unit: str = _unit_field()
    upstream: str = Field(description="Name of the unit element that must run first.")
    k: int = Field(ge=0, description="Steps the unit lags behind the upstream unit.")
    initially_on: bool = Field(
        default=True,
        description="Whether the upstream unit counts as already warmed up at the "
        "start of the horizon.",
    )
    name: str | None = _label_field()


class ConditionalElement(_StrictModel):
    """When a unit is on, another unit is on (or off)."""

    kind: Literal["conditional"]
    unit: str = _unit_field()
    implies: str = Field(description="Name of the unit element whose state follows.")
    state: Literal["on", "off"] = Field(
        default="on", description="State of the implied unit while this one is on."
    )
    name: str | None = _label_field()


class ParallelGroupElement(_StrictModel):
    """Interchangeable units ordered so the first listed runs first."""

    kind: Literal["parallel_group"]
    units: list[str] = Field(
        min_length=2, description="Unit element names, in priority order."
    )
    variables: list[str] = Field(
        default_factory=list,
        description="Variables ordered the same way across the group.",
    )
    order_status: bool = Field(
        default=True, description="Whether to order the units' status binaries."
    )
    name: str | None = _label_field()


class ExtensionElement(_StrictModel):
    """A user function that adds constraints or an objective to the model.

    The spec only names the function, so a spec that uses one builds only where
    the package defining it is installed.
    """

    kind: Literal["extension"]
    name: str = Field(description="Name of this call; unique among extensions.")
    entry_point: str = Field(
        description="'package.module:function', or the name of an entry point in "
        "the 'flexpse.extensions' group."
    )
    options: dict[str, Any] = Field(
        default_factory=dict, description="Keyword options passed to the function."
    )


class ObjectiveElement(_StrictModel):
    """The objective the model optimizes."""

    kind: Literal["objective"]
    name: str = Field(default="objective", description="Name on the model.")
    expression: Literal["cost", "extension"] = Field(
        default="cost",
        description="'cost' optimizes a costing element's cost; 'extension' means "
        "an extension adds the objective.",
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
    | RampElement
    | StatusElement
    | StartupShutdownElement
    | DwellElement
    | BypassElement
    | StartupDelayElement
    | ConditionalElement
    | ParallelGroupElement
    | DispatchElement
    | FixElement
    | UnfixElement
    | ExtensionElement
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


LOGIC_KINDS = (
    "ramp",
    "status",
    "startup_shutdown",
    "dwell",
    "bypass",
    "startup_delay",
    "conditional",
    "parallel_group",
)
"""Kinds that add operating logic to units they name."""

TARGET_FIELDS: dict[str, tuple[str, ...]] = {
    "surrogate": ("unit", "relation"),
    "dispatch": ("unit", "variable"),
    "fix": ("variable",),
    "unfix": ("variable",),
    "ramp": ("unit", "variable", "component_name"),
    "status": ("unit",),
    "startup_shutdown": ("unit",),
    "dwell": ("unit", "variable"),
    "bypass": ("unit",),
    "startup_delay": ("unit",),
    "conditional": ("unit",),
    "parallel_group": ("units",),
}
"""Kind -> the fields that say what an element sets; two equal targets clash."""


def element_target(element) -> tuple[str, ...]:
    """Return what an element sets, e.g. ``(unit, relation)`` for a surrogate.

    Args:
        element: Any spec element.

    Returns:
        The values of the kind's ``TARGET_FIELDS`` as strings, or an empty tuple
        for kinds that are identified by name.
    """
    values = [getattr(element, f) for f in TARGET_FIELDS.get(element.kind, ())]
    return tuple(",".join(v) if isinstance(v, list) else str(v or "") for v in values)


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
    fix_on_violation: FixPolicy = Field(
        default="error",
        description="Default bounds/domain policy for every fix element that "
        "doesn't set its own.",
    )
    fix_tolerance: float = Field(
        default=1e-9,
        ge=0,
        description="Default absolute tolerance for the fix bounds check.",
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
            if el.expression == "cost":
                label = f"objective {el.name!r}"
                self._check_package(label, el.costing_package, "costing")
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

    def _members(self, ref: str) -> list[str]:
        """Return the member names a unit reference covers, or raise naming it."""
        units = {el.name: el for el in self.of_kind("unit")}
        sets = {el.name: [str(v) for v in el.values] for el in self.of_kind("set")}
        base, _, member = ref.partition("[")
        unit = units.get(base)
        if unit is None:
            raise ValueError(f"{ref!r}: no such unit element. Units: {sorted(units)}.")
        if unit.index is None:
            if member:
                raise ValueError(f"{ref!r}: unit {base!r} is not indexed.")
            return [ref]
        values = sets[unit.index]
        if not member:
            return [f"{base}[{v}]" for v in values]
        if member.removesuffix("]") not in values:
            raise ValueError(f"{ref!r}: {member[:-1]!r} is not in set {unit.index!r}.")
        return [ref]

    @model_validator(mode="after")
    def _logic_is_consistent(self) -> "FlowsheetSpec":
        """Require logic to name units and never repeat."""

        def refs(el) -> list[str]:
            if el.kind == "parallel_group":
                return el.units
            extra = {"conditional": "implies", "startup_delay": "upstream"}.get(el.kind)
            return [el.unit] + ([getattr(el, extra)] if extra else [])

        logic = self.of_kind(*LOGIC_KINDS)
        members = {}
        for el in logic:
            for ref in refs(el):
                try:
                    members[ref] = self._members(ref)
                except ValueError as exc:
                    raise ValueError(f"{el.kind}: {exc}") from None

        seen = set()
        for el in logic:
            if el.kind == "parallel_group":
                continue
            rest = element_target(el)[1:]
            for member in members[el.unit]:
                key = (el.kind, member, *rest)
                if key in seen:
                    raise ValueError(f"{el.kind} on {member!r} is set twice; keep one.")
                seen.add(key)
        return self

    @model_validator(mode="after")
    def _extension_names_are_unique(self) -> "FlowsheetSpec":
        """Require every extension name to be unique."""
        names = [el.name for el in self.of_kind("extension")]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"extension names {repeated} are used twice; keep one.")
        return self
