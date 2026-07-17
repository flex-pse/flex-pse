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

flex-pse **never deletes** model components (blocks, Vars, Params,
constraints): anything else on the model that referenced a deleted component —
an aggregated power constraint, an expanded arc — would silently keep the stale
reference. A built model is updated only by mutating parameter values in place
(:meth:`~OpsBlockData.update_parameters`) or by adding/deactivating
constraints; FlexParameterize drives this in M10.
"""

import enum
import numbers
from collections.abc import Sequence

import pyomo.environ as pyo
from idaes.core import UnitModelBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.config.schema import (
    ExternalDispatchSpec,
    UnitCommitmentConfig,
    UnitConfig,
)
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import (
    IORegistry,
    IOVariableRecord,
    ParameterRecord,
    PowerRecord,
)
from flexops.core.time_block import TimeBlockData

_POWER_VARS = {
    nm.PowerKind.ELECTRICAL: (nm.POWER_ELECTRICAL, "Electrical draw of the unit"),
    nm.PowerKind.THERMAL: (nm.POWER_THERMAL, "Thermal/gas-driven duty of the unit"),
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
        # Pyomo skips a ConfigValue's domain for an explicit None, so the
        # None -> all-defaults coercion has to happen here, not in the domain.
        if self.config.unit_commitment is None:
            self.config.unit_commitment = UnitCommitmentConfig()
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

    @staticmethod
    def _check_power_kind(kind) -> None:
        """Raise unless ``kind`` is a :class:`~flexcore.nomenclature.PowerKind`.

        Args:
            kind: The value to check; even a valid-value plain string (e.g.
                ``"electrical"``) is rejected — pass the enum member.

        Raises:
            FlexConfigError: If ``kind`` is not a ``PowerKind`` member.
        """
        if not isinstance(kind, nm.PowerKind):
            allowed = ", ".join(f"PowerKind.{k.name}" for k in nm.PowerKind)
            raise FlexConfigError(
                f"Power kind must be a PowerKind member ({allowed}), got " f"{kind!r}.",
                field="kind",
                value=kind,
            )

    def register_power(self, var, kind: nm.PowerKind = nm.PowerKind.ELECTRICAL) -> None:
        """Register a power-draw variable for plant/costing aggregation.

        Args:
            var: The Pyomo ``Var`` (kW) to register.
            kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.

        Raises:
            FlexConfigError: If ``kind`` is not a ``PowerKind`` member.
        """
        self._check_power_kind(kind)
        self._io_registry.power.append(
            PowerRecord(var=var, name=var.local_name, kind=kind)
        )

    def declare_power(self, kind: nm.PowerKind = nm.PowerKind.ELECTRICAL):
        """Create, register, and return this unit's power-draw Var (kW).

        Creates ``power_electrical[t]`` (resp. ``power_thermal[t]``) indexed over
        the time set, attaches it under the nomenclature constant name, registers
        it, and returns it.

        Args:
            kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.

        Returns:
            The created, time-indexed ``Var`` in kW.

        Raises:
            FlexConfigError: If ``kind`` is not a ``PowerKind`` member.
        """
        self._check_power_kind(kind)
        name, doc = _POWER_VARS[kind]
        tb = self._find_time_block()
        self.add_component(
            name,
            pyo.Var(tb.time_index, initialize=0.0, units=pyunits.kW, doc=doc),
        )
        var = self.find_component(name)
        self.register_power(var, kind=kind)
        return var

    # -- stream state blocks + ports (property package, §3.7) --------------

    def add_stream_ports(
        self,
        inlet_ports: Sequence[str] = ("inlet",),
        outlet_ports: Sequence[str] = ("outlet",),
        io_vars: Sequence[str] = ("flow_vol_phase",),
    ) -> None:
        """Build the configured stream state blocks and expose them as ports.

        For each requested port, constructs a scalar state block from this
        unit's configured ``property_package`` (its state variables indexed over
        the time set), registers the named ``io_vars`` on that block as process
        IO variables — inlet ports as ``"input"``, outlet ports as
        ``"output"`` — and builds the IDAES port from the state block via the
        inherited ``add_inlet_port``/``add_outlet_port`` helpers. Each state
        block is named ``"{port}_state"`` (e.g. ``inlet_state``), and each
        registered IO variable is the live state-block ``Var`` itself (e.g.
        ``inlet_state.flow_vol_phase``), never a ``Reference`` or slice. The
        extensive/intensive split of the states a port carries is applied when
        the topology layer wires the ports onto arcs in M09.

        Args:
            inlet_ports: Names of the inlet ports to build (default one
                ``"inlet"``).
            outlet_ports: Names of the outlet ports to build (default one
                ``"outlet"``).
            io_vars: State-block variable names to register as process IO on
                every port (default the volumetric ``"flow_vol_phase"``). Which
                variables are the meaningful IO is property-package dependent,
                so the caller chooses them.

        Raises:
            FlexConfigError: If the unit has no ``property_package`` configured.
        """
        pkg = self.config.property_package
        if pkg is None:
            raise FlexConfigError(
                "add_stream_ports requires a property_package on the unit; none "
                "was configured.",
                field="property_package",
                value=None,
            )
        tb = self._find_time_block()
        for port_name in inlet_ports:
            self._add_stream_port(pkg, tb, port_name, io_vars, role="input")
        for port_name in outlet_ports:
            self._add_stream_port(pkg, tb, port_name, io_vars, role="output")

    def _add_stream_port(self, pkg, tb, port_name, io_vars, role) -> None:
        """Build one ``{port_name}_state`` block, register its IO, add the port.

        Args:
            pkg: The unit's configured property (parameter) block.
            tb: The model's ``TimeBlockData``, supplying the time index.
            port_name: The port (and ``"{port_name}_state"`` block) name.
            io_vars: State-block variable names to register as process IO.
            role: ``"input"`` for an inlet port, ``"output"`` for an outlet port.
        """
        state_name = f"{port_name}_state"
        self.add_component(state_name, pkg.build_state_block(time_index=tb.time_index))
        state = self.find_component(state_name)
        for var_name in io_vars:
            self.register_io_variable(getattr(state, var_name), role=role)
        if role == "input":
            self.add_inlet_port(name=port_name, block=state, doc=f"{port_name} stream")
        else:
            self.add_outlet_port(name=port_name, block=state, doc=f"{port_name} stream")

    # -- in-place parameter updates (FlexParameterize 2-way, §5) -----------

    def update_parameters(self, values: dict) -> None:
        """Update registered process parameters in place, by name.

        This is flex-pse's only sanctioned way to change a built model's
        parameters: mutate the existing (mutable) ``Param``/``Var`` so every
        constraint and expression that references it sees the new value. Never
        delete and rebuild components — anything else on the model holding a
        reference to the old component would silently keep the stale one.

        Args:
            values: Mapping of registered parameter name (as returned by
                ``register_process_parameter``) to its new value. Values may
                carry Pyomo units; bare numbers are taken in the component's
                declared units.

        Raises:
            FlexConfigError: If a name is not a registered process parameter.
        """
        registered = {rec.name: rec.param for rec in self._io_registry.parameters}
        for name, value in values.items():
            if name not in registered:
                known = ", ".join(repr(n) for n in registered) or "none"
                raise FlexConfigError(
                    f"{name!r} is not a registered process parameter on "
                    f"{self.name!r} (registered: {known}).",
                    field=name,
                    value=value,
                )
            registered[name].set_value(value)

    # -- external dispatch (DERMS, §3.2) ----------------------------------

    def _resolve_dispatch_series(self, series, tb) -> dict[int, float]:
        """Map a dispatch series to ``{time_index: value}``.

        Accepts a mapping or pandas Series keyed by integer time index or by
        timestamp (coerced through ``TimeBlock.index_of``).

        Args:
            series: The mapping or pandas Series of dispatch values, keyed by
                integer time index or timestamp.
            tb: The model's ``TimeBlockData``, used to bound integer keys and
                resolve timestamps.

        Returns:
            The resolved ``{time_index: float value}`` mapping.

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
    def build_from_config(cls, cfg: UnitConfig, **kwargs):
        """Construct a unit from a validated ``UnitConfig`` (deferred to M09).

        Raises:
            NotImplementedError: Always; config-driven construction and the
                whole-model ``flexops.build_model`` land in M09.
        """
        raise NotImplementedError(
            "Config-driven construction lands in M09. Build units directly for "
            "now; whole-model construction is flexops.build_model in M09."
        )
