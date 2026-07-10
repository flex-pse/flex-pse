"""OpsBlock: the base class of every flex-pse unit model (§3.2, decision R1).

``OpsBlockData`` inherits IDAES ``UnitModelBlockData`` for its ConfigBlock, Port,
and costing-registration machinery, but uses **no ControlVolumes** (R1):
subclasses hand-write their 1-3 balance constraints. It provides the
registration API that FlexParameterize and the docs generator consume
(:meth:`OpsBlockData.register_io_variable`,
:meth:`~OpsBlockData.register_process_parameter`,
:meth:`~OpsBlockData.register_power`), the base power Vars
(:meth:`~OpsBlockData.declare_power`), the external-dispatch hook
(:meth:`~OpsBlockData.set_external_dispatch`), and the config slots
(``unit_commitment``, ``relaxation``, ``allow_bypass``, ``external_dispatch``)
that the M08 logic layer will consume.

The block-replacement helper :func:`replace_unit` is a free function, not a
method: the block that *holds* units is a ``FlowsheetBlockData`` container, a
different IDAES hierarchy, so the surgery is deliberately hierarchy-agnostic
(§5/R10). ``PlantBlock``/``NetworkBlock`` expose it as a thin ``.replace_unit``
wrapper in M09.
"""

import enum
import numbers

import pyomo.environ as pyo
from idaes.core import UnitModelBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits
from pyomo.network import Arc, Port

from flexcore.config.schema import (
    ExternalDispatchSpec,
    UnitCommitmentConfig,
    UnitConfig,
)
from flexcore.exceptions import FlexConfigError
from flexcore.nomenclature import (
    ELECTRICAL_POWER,
    THERMAL_POWER,
    PowerKind,
)
from flexops.core.registration import (
    IORegistry,
    IOVariableRecord,
    ParameterRecord,
    PowerRecord,
)
from flexops.core.time_block import TimeBlockData

_POWER_VARS = {
    PowerKind.ELECTRICAL.value: (ELECTRICAL_POWER, "Electrical draw of the unit"),
    PowerKind.THERMAL.value: (THERMAL_POWER, "Thermal/gas-driven duty of the unit"),
}


class RelaxationPolicy(enum.StrEnum):
    """How a unit's discrete structure is relaxed (config slot only in M03).

    The switching logic that consumes this is the M08 logic layer; M03 only
    validates and stores the choice.
    """

    EXACT = "exact"
    RELAXED = "relaxed"


def _unit_config_domain(value):
    """ConfigValue domain: accept only a validated ``UnitConfig`` or ``None``."""
    if value is None or isinstance(value, UnitConfig):
        return value
    raise FlexConfigError(
        "flexops_config must be a validated flexcore.config.schema.UnitConfig "
        f"instance (or None), not {type(value).__name__}; never pass a raw "
        "dict (conventions §4).",
        field="flexops_config",
        value=value,
    )


def _unit_commitment_domain(value):
    """ConfigValue domain: coerce ``None`` to an all-defaults UC config."""
    if value is None:
        return UnitCommitmentConfig()
    if isinstance(value, UnitCommitmentConfig):
        return value
    raise FlexConfigError(
        "unit_commitment must be a UnitCommitmentConfig instance (or None), "
        f"not {type(value).__name__}.",
        field="unit_commitment",
        value=value,
    )


def _external_dispatch_domain(value):
    """ConfigValue domain: accept only an ``ExternalDispatchSpec`` or ``None``."""
    if value is None or isinstance(value, ExternalDispatchSpec):
        return value
    raise FlexConfigError(
        "external_dispatch must be an ExternalDispatchSpec instance (or None), "
        f"not {type(value).__name__}.",
        field="external_dispatch",
        value=value,
    )


def _relaxation_domain(value):
    """ConfigValue domain: coerce to a :class:`RelaxationPolicy`."""
    try:
        return RelaxationPolicy(value)
    except ValueError as exc:
        allowed = ", ".join(repr(p.value) for p in RelaxationPolicy)
        raise FlexConfigError(
            f"relaxation must be one of {allowed}, got {value!r}.",
            field="relaxation",
            value=value,
        ) from exc


@declare_process_block_class("OpsBlock")
class OpsBlockData(UnitModelBlockData):
    """Base class of every flex-pse unit model (see module docstring)."""

    CONFIG = UnitModelBlockData.CONFIG()
    # A unit lives on a bare ConcreteModel or a dynamic=False flowsheet, never a
    # DAE flowsheet (R2). Fix the inherited defaults to False so build() does not
    # try to resolve them from a parent flowsheet (pitfall 1).
    CONFIG.get("dynamic").set_default_value(False)
    CONFIG.get("has_holdup").set_default_value(False)
    CONFIG.declare(
        "property_package",
        ConfigValue(
            default=None,
            description="Property (parameter) block carrying flow between "
            "units; None for units that need no fluid properties.",
        ),
    )
    CONFIG.declare(
        "flexops_config",
        ConfigValue(
            default=None,
            domain=_unit_config_domain,
            description="Optional already-validated UnitConfig for this unit "
            "(never a raw dict, conventions §4).",
        ),
    )
    CONFIG.declare(
        "unit_commitment",
        ConfigValue(
            default=UnitCommitmentConfig(),
            domain=_unit_commitment_domain,
            description="Per-unit unit-commitment sub-config (§3.5). Validated "
            "and stored in M03; its constraints are built in M08.",
        ),
    )
    CONFIG.declare(
        "external_dispatch",
        ConfigValue(
            default=None,
            domain=_external_dispatch_domain,
            description="Optional external (DERMS) dispatch source consumed by "
            "set_external_dispatch.",
        ),
    )
    CONFIG.declare(
        "relaxation",
        ConfigValue(
            default=RelaxationPolicy.EXACT,
            domain=_relaxation_domain,
            description="Discrete-structure relaxation policy (config slot only "
            "in M03; the switching logic is built in M08).",
        ),
    )
    CONFIG.declare(
        "allow_bypass",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether a bypass stream is allowed (config slot only in "
            "M03; the bypass constraints are built in M08).",
        ),
    )

    def build(self) -> None:
        """Set up dynamics defaults and the empty IO registry (no constraints)."""
        super().build()
        self._io_registry = IORegistry()

    # -- time access ------------------------------------------------------

    def _find_time_block(self) -> TimeBlockData:
        """Return the unique TimeBlock on this unit's model.

        Interim time access until the ``flowsheet()`` chain arrives with
        PlantBlock in M09: search the model for exactly one TimeBlock. The
        result is not cached on the block — assigning a Pyomo component to an
        attribute would trip ``Block.__setattr__`` (pitfall 2).

        Returns:
            The model's ``TimeBlockData``.

        Raises:
            FlexConfigError: If the model has zero or several TimeBlocks.
        """
        model = self.model()
        found = [
            b
            for b in model.component_data_objects(pyo.Block, descend_into=True)
            if isinstance(b, TimeBlockData)
        ]
        if len(found) != 1:
            raise FlexConfigError(
                f"Expected exactly one TimeBlock on model {model.name!r}, found "
                f"{len(found)}. Build a TimeBlock on the model first.",
            )
        return found[0]

    # -- registration API -------------------------------------------------

    @staticmethod
    def _units_str(var) -> str:
        """Return a variable's units as a string (first data object if indexed)."""
        ref = next(iter(var.values())) if var.is_indexed() else var
        return str(pyunits.get_units(ref))

    def register_io_variable(
        self, var, role: str = "input", tag_hint: str | None = None
    ) -> None:
        """Register a process input/output variable (fixed during regression).

        Args:
            var: The Pyomo ``Var`` to register.
            role: ``"input"`` or ``"output"``.
            tag_hint: Optional historian-tag hint for FlexParameterize.

        Raises:
            FlexConfigError: If ``role`` is not ``"input"`` or ``"output"``.
        """
        if role not in ("input", "output"):
            raise FlexConfigError(
                f"IO variable role must be 'input' or 'output', got {role!r} "
                f"for variable {var.local_name!r}.",
                field="role",
                value=role,
            )
        self._io_registry.io_variables.append(
            IOVariableRecord(
                var=var,
                name=var.local_name,
                role=role,
                tag_hint=tag_hint,
                units=self._units_str(var),
                time_indexed=var.is_indexed(),
            )
        )

    def register_process_parameter(
        self, param_or_var, regressable: bool = True
    ) -> None:
        """Register a design/regression parameter (found during regression).

        Args:
            param_or_var: The Pyomo ``Param`` or ``Var`` to register.
            regressable: Whether FlexParameterize may fit this parameter.
        """
        self._io_registry.parameters.append(
            ParameterRecord(
                param=param_or_var,
                name=param_or_var.local_name,
                regressable=regressable,
            )
        )

    def register_power(self, var, kind: str = "electrical") -> None:
        """Register a power-draw variable for plant/costing aggregation.

        Args:
            var: The Pyomo ``Var`` (kW) to register.
            kind: A :class:`~flexcore.nomenclature.PowerKind` value.

        Raises:
            FlexConfigError: If ``kind`` is not a valid ``PowerKind`` value.
        """
        kind_value = kind.value if isinstance(kind, PowerKind) else kind
        if kind_value not in _POWER_VARS:
            allowed = ", ".join(repr(k) for k in _POWER_VARS)
            raise FlexConfigError(
                f"Power kind must be one of {allowed}, got {kind!r}.",
                field="kind",
                value=kind,
            )
        self._io_registry.power.append(
            PowerRecord(var=var, name=var.local_name, kind=kind_value)
        )

    def declare_power(self, kind: str = "electrical"):
        """Create, register, and return this unit's power-draw Var (kW).

        Creates ``electrical_power[t]`` (resp. ``thermal_power[t]``) indexed over
        the time set, attaches it under the nomenclature constant name, registers
        it, and returns it.

        Args:
            kind: A :class:`~flexcore.nomenclature.PowerKind` value.

        Returns:
            The created, time-indexed ``Var`` in kW.

        Raises:
            FlexConfigError: If ``kind`` is not a valid ``PowerKind`` value.
        """
        kind_value = kind.value if isinstance(kind, PowerKind) else kind
        if kind_value not in _POWER_VARS:
            allowed = ", ".join(repr(k) for k in _POWER_VARS)
            raise FlexConfigError(
                f"Power kind must be one of {allowed}, got {kind!r}.",
                field="kind",
                value=kind,
            )
        name, doc = _POWER_VARS[kind_value]
        tb = self._find_time_block()
        setattr(
            self,
            name,
            pyo.Var(tb.time_index, initialize=0.0, units=pyunits.kW, doc=doc),
        )
        var = getattr(self, name)
        self.register_power(var, kind=kind_value)
        return var

    # -- external dispatch (DERMS, §3.2) ----------------------------------

    def _resolve_dispatch_series(self, series, tb) -> dict[int, float]:
        """Map a dispatch series to ``{time_index: value}``.

        Accepts a mapping or pandas Series keyed by integer time index or by
        timestamp (coerced through ``TimeBlock.index_of``).

        Raises:
            FlexConfigError: If ``series`` is not a mapping/Series, or a key is
                an out-of-range integer index.
        """
        if not hasattr(series, "items"):
            raise FlexConfigError(
                "External-dispatch series must be a mapping or pandas Series "
                "keyed by time index or timestamp.",
                value=series,
            )
        resolved: dict[int, float] = {}
        for key, val in series.items():
            if isinstance(key, numbers.Integral) and not isinstance(key, bool):
                idx = int(key)
                if not 0 <= idx < tb.n_points:
                    raise FlexConfigError(
                        f"External-dispatch index {idx} is out of range "
                        f"[0, {tb.n_points}).",
                        value=idx,
                    )
            else:
                idx = tb.index_of(key)
            resolved[idx] = float(pyo.value(val))
        return resolved

    def set_external_dispatch(self, var, series, *, fix: bool = True) -> None:
        """Drive a controllable var from an external time-indexed command series.

        For each time point ``t``, sets ``var[t]`` to ``series[t]`` and, when
        ``fix`` is True, fixes it — removing the dispatch degree of freedom while
        leaving sizing free (the DERMS/aggregator case, §3.2). Available on all
        units; first-classed on ``BatteryModel`` in M08.

        Args:
            var: A time-indexed ``Var`` on this unit.
            series: A mapping or pandas Series aligned to the time set, keyed by
                integer index or timestamp.
            fix: Whether to fix each ``var[t]`` after setting it.

        Raises:
            FlexConfigError: If ``var`` is not indexed, or ``series`` does not
                cover exactly the time set.
        """
        if not var.is_indexed():
            raise FlexConfigError(
                f"set_external_dispatch requires a time-indexed Var; "
                f"{var.local_name!r} is not indexed.",
                value=var.local_name,
            )
        tb = self._find_time_block()
        resolved = self._resolve_dispatch_series(series, tb)
        required = set(tb.time_index)
        if set(resolved) != required:
            missing = sorted(required - set(resolved))
            extra = sorted(set(resolved) - required)
            raise FlexConfigError(
                f"External-dispatch series for {var.local_name!r} does not align "
                f"with the TimeBlock's {tb.n_points} time points "
                f"(missing indices {missing}, unexpected {extra}).",
                value=var.local_name,
            )
        for t in tb.time_index:
            var[t].set_value(resolved[t])
            if fix:
                var[t].fix()

    # -- config-driven construction (M09) ---------------------------------

    @classmethod
    def from_config(cls, cfg: UnitConfig, **kwargs):
        """Construct a unit from a validated ``UnitConfig`` (deferred to M09).

        Raises:
            NotImplementedError: Always; config-driven construction and the
                whole-model ``flexops.build_model`` land in M09.
        """
        raise NotImplementedError(
            "Config-driven construction lands in M09. Build units directly for "
            "now; whole-model construction is flexops.build_model in M09."
        )


def _within(component, block) -> bool:
    """Return True if ``component`` is ``block`` or nested under it."""
    node = component.parent_block()
    while node is not None:
        if node is block:
            return True
        node = node.parent_block()
    return False


def _matching_port(new_block, rel_name: str, name: str) -> Port:
    """Return the port on ``new_block`` matching ``rel_name`` or error.

    Args:
        new_block: The replacement block.
        rel_name: The old port's name relative to the old block.
        name: The child attribute name being replaced (for error text).

    Raises:
        FlexConfigError: If ``new_block`` has no matching ``Port``.
    """
    port = new_block.find_component(rel_name)
    if not isinstance(port, Port):
        raise FlexConfigError(
            f"replace_unit: the replacement for {name!r} has no port "
            f"{rel_name!r} to reconnect the arc to (port-topology mismatch).",
            field=name,
        )
    return port


def replace_unit(parent, name: str, new_block) -> None:
    """Swap child block ``name`` on ``parent`` and re-point arcs at its ports.

    Deletes the old child block, attaches ``new_block`` under ``name``, and
    re-points any arcs that referenced the old block's ports at the new block's
    matching ports (re-expanding when the originals were expanded). This is the
    raw, hierarchy-agnostic in-place rewire (§5/R10); surrogate construction and
    the ``SurrogateSpec``-driven reconnection are FlexParameterize's job (M10).

    Args:
        parent: The block that holds the child.
        name: Attribute name of the child block on ``parent``.
        new_block: The replacement block (constructed on attach).

    Raises:
        FlexConfigError: If ``parent`` has no child ``name``, or the replacement
            lacks a port an arc needs (port-topology mismatch).
    """
    old_block = parent.component(name)
    if old_block is None:
        raise FlexConfigError(
            f"replace_unit: parent block {parent.name!r} has no child named "
            f"{name!r}.",
            field=name,
        )

    root = parent.model()
    affected = []
    for arc in list(root.component_data_objects(Arc, descend_into=True)):
        src, dst = arc.source, arc.destination
        src_in = src is not None and _within(src, old_block)
        dst_in = dst is not None and _within(dst, old_block)
        if not (src_in or dst_in):
            continue
        affected.append(
            {
                "arc": arc,
                "parent": arc.parent_block(),
                "local_name": arc.local_name,
                "src": src,
                "dst": dst,
                "src_in": src_in,
                "dst_in": dst_in,
                "src_rel": src.getname(relative_to=old_block) if src_in else None,
                "dst_rel": dst.getname(relative_to=old_block) if dst_in else None,
                "expanded": getattr(arc, "expanded_block", None) is not None,
            }
        )

    for info in affected:
        arc = info["arc"]
        expanded = getattr(arc, "expanded_block", None)
        info["parent"].del_component(arc)
        if expanded is not None:
            expanded.parent_block().del_component(expanded)

    parent.del_component(name)
    setattr(parent, name, new_block)

    # Resolve every new endpoint before rebuilding, so a topology mismatch is
    # reported without leaving half-rewired arcs behind.
    rewired = []
    any_expanded = False
    for info in affected:
        src = (
            _matching_port(new_block, info["src_rel"], name)
            if info["src_in"]
            else info["src"]
        )
        dst = (
            _matching_port(new_block, info["dst_rel"], name)
            if info["dst_in"]
            else info["dst"]
        )
        rewired.append((info, src, dst))
        any_expanded = any_expanded or info["expanded"]

    for info, src, dst in rewired:
        setattr(
            info["parent"],
            info["local_name"],
            Arc(source=src, destination=dst, doc="Reconnected by replace_unit"),
        )

    if any_expanded:
        pyo.TransformationFactory("network.expand_arcs").apply_to(root)
