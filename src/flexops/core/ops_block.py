"""OpsBlock: the base class of every flex-pse unit model (§3.2).

``OpsBlockData`` inherits IDAES ``UnitModelBlockData`` for its ConfigBlock, Port,
and costing-registration machinery, but uses **no ControlVolumes**:
subclasses hand-write their 1-3 balance constraints. It provides the
registration API that FlexParameterize and the docs generator consume
(:meth:`OpsBlockData.register_io_variable`,
:meth:`~OpsBlockData.register_process_parameter`,
:meth:`~OpsBlockData.register_power`,
:meth:`~OpsBlockData.register_fuel_usage`), the base power Vars
(:meth:`~OpsBlockData.declare_power`), the external-dispatch hook
(:meth:`~OpsBlockData.set_external_dispatch`), and the config slots
(``unit_commitment``, ``relaxation``, ``allow_bypass``, ``external_dispatch``)
that the logic layer will consume.

flex-pse **never deletes** model components (blocks, Vars, Params,
constraints): anything else on the model that referenced a deleted component —
an aggregated power constraint, an expanded arc — would silently keep the stale
reference. A built model is updated only by mutating parameter values in place
(:meth:`~OpsBlockData.update_parameters`) or by adding/deactivating
constraints; FlexParameterize drives this.
"""

import enum
import numbers
from collections.abc import Sequence

import pyomo.environ as pyo
from idaes.core import UnitModelBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits
from pyomo.network import Port

from flexcore import nomenclature as nm
from flexcore.config.schema import (
    ExternalDispatchSpec,
    UnitCommitmentConfig,
    UnitConfig,
)
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import (
    FuelUsageRecord,
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
    """How a unit's discrete structure is relaxed."""

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


def _costing_package_domain(value):
    """ConfigValue domain: accept ``None`` or a costing package (duck-typed).

    A costing package is any object exposing ``register_unit_power``
    (``FlexCosting`` block). Duck-typed rather than isinstance-checked so
    ``flexops.core`` need not import ``flexops.costing`` (which imports core).
    """
    if value is None or hasattr(value, "register_unit_power"):
        return value
    raise FlexConfigError(
        "costing_package must be a FlexCosting block (an object exposing "
        f"register_unit_power) or None, not {type(value).__name__}.",
        field="costing_package",
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
    # DAE flowsheet. Fix the inherited defaults to False so build() does not
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
            description="Per-unit unit-commitment sub-config.",
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
            description="Discrete-structure relaxation policy.",
        ),
    )
    CONFIG.declare(
        "allow_pass_through",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether add_bypass_constraints() builds inlet-to-outlet "
            "pass-through equalities for this unit's non-excluded state "
            "variables. SISOBlock (and its subclasses Pump/Tank) "
            "override the base default to True so the flow-topology units are "
            "well-posed out of the box; the base OpsBlock default stays False.",
        ),
    )
    CONFIG.declare(
        "costing_package",
        ConfigValue(
            default=None,
            domain=_costing_package_domain,
            description="Optional FlexCosting block this unit associates "
            "with: register_power forwards the unit's power draw to it. None for "
            "standalone units; the forwarding is strictly conditional.",
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

        Interim time access until the ``flowsheet()``: search the model for
        exactly one TimeBlock. The result is not cached on the block —
        assigning a Pyomo component to an attribute would trip
        ``Block.__setattr__`` (pitfall 2).

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

    @staticmethod
    def _check_power_metadata(kind, temperature) -> None:
        """Validate ``temperature`` against ``kind``.

        A thermal draw must carry a temperature (heat duties at different
        temperatures are never aggregated together); an electrical draw must not.

        Args:
            kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.
            temperature: The heat-duty temperature (unit-carrying), or ``None``.

        Raises:
            FlexConfigError: If the metadata does not match ``kind``.
        """
        if kind is nm.PowerKind.THERMAL:
            if temperature is None:
                raise FlexConfigError(
                    "A PowerKind.THERMAL draw requires a temperature (a "
                    "unit-carrying value, e.g. 350 * pyunits.K).",
                    field="temperature",
                    value=temperature,
                )
        elif temperature is not None:
            raise FlexConfigError(
                f"A PowerKind.{kind.name} draw takes no temperature.",
                field="temperature",
                value=temperature,
            )

    def register_power(
        self,
        var,
        kind: nm.PowerKind = nm.PowerKind.ELECTRICAL,
        *,
        temperature=None,
    ) -> None:
        """Register a power-draw variable for plant/costing aggregation.

        Args:
            var: The Pyomo ``Var`` (kW) to register.
            kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.
            temperature: The heat-duty temperature, a unit-carrying value
                (required only for ``PowerKind.THERMAL``).

        Raises:
            FlexConfigError: If ``kind`` is not a ``PowerKind`` member, or the
                ``temperature`` metadata does not match ``kind``.
        """
        self._check_power_kind(kind)
        self._check_power_metadata(kind, temperature)
        self._io_registry.power.append(
            PowerRecord(
                var=var,
                name=var.local_name,
                kind=kind,
                temperature=temperature,
            )
        )
        # Forward the association to a FlexCosting block when one was given
        costing_package = self.config.costing_package
        if costing_package is not None:
            costing_package.register_unit_power(self, var, kind)

    def declare_power(
        self,
        kind: nm.PowerKind = nm.PowerKind.ELECTRICAL,
        *,
        temperature=None,
    ):
        """Create, register, and return this unit's power-draw Var (kW).

        Creates ``power_electrical[t]`` (resp. ``power_thermal[t]``) indexed over
        the time set, attaches it under its nomenclature name, registers it (with
        its temperature metadata), and returns it.

        Args:
            kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.
            temperature: The heat-duty temperature, a unit-carrying value
                (required only for ``PowerKind.THERMAL``).

        Returns:
            The created, time-indexed ``Var`` in kW.

        Raises:
            FlexConfigError: If ``kind`` is not a ``PowerKind`` member, or the
                ``temperature`` metadata does not match ``kind``.
        """
        self._check_power_kind(kind)
        self._check_power_metadata(kind, temperature)
        name, doc = _POWER_VARS[kind]
        tb = self._find_time_block()
        self.add_component(
            name,
            pyo.Var(tb.time_index, initialize=0.0, units=pyunits.kW, doc=doc),
        )
        var = self.find_component(name)
        self.register_power(var, kind=kind, temperature=temperature)
        return var

    def register_fuel_usage(self, var, fuel_name: str) -> None:
        """Register a fuel-usage variable — a volumetric flow — for costing.

        Fuel is metered and billed on **volume**, not as a kW power draw, so a
        unit that burns fuel registers its usage flow here rather than through
        :meth:`register_power`. ``var`` is typically an existing process
        variable: for a stream built from
        :class:`~flexops.properties.simple_gas.SimpleGasFlow` (whose
        ``flow_vol_phase`` is already m³/hr), that is
        ``pyo.Reference(self.inlet_state.flow_vol_phase[:, "Vap"])``.

        FlexCosting pulls every registered flow from the model and sums it into
        ``aggregate_fuel_usage[t, fuel_name]`` in EECO's m³/hr, converting with
        ``pyunits.convert`` — so a ``var`` that is not a volumetric rate fails
        loudly there, the same contract a power draw has. flex-pse applies no
        heating value; energy-basis tariff conversion is EECO's job.

        Args:
            var: The time-indexed Pyomo ``Var``/``Reference`` carrying the fuel's
                volumetric flow (convertible to m³/hr).
            fuel_name: The fuel's name (e.g. ``"natural_gas"``), the key its flow
                aggregates and bills under.

        Raises:
            FlexConfigError: If ``fuel_name`` is empty.
        """
        if not fuel_name:
            raise FlexConfigError(
                "register_fuel_usage requires a non-empty fuel_name (e.g. "
                "'natural_gas'); it is the key the flow aggregates and bills "
                "under.",
                field="fuel_name",
                value=fuel_name,
            )
        self._io_registry.fuel.append(
            FuelUsageRecord(var=var, name=var.local_name, fuel_name=fuel_name)
        )

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
        the topology layer wires the ports onto arcs.

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

    def add_pass_through_constraints(
        self,
        inlet: Port,
        outlet: Port,
        *,
        exclude_vars: Sequence[str] = (),
    ) -> None:
        """Pass non-excluded, non-fixed inlet state variables straight to the outlet.

        For each state-variable name exposed by ``inlet`` and not in
        ``exclude_vars``, adds an equality Constraint ``outlet_var[idx] ==
        inlet_var[idx]`` over every index the variable carries -- unless every
        entry of that variable is already ``fixed`` (e.g. ``dens_mass`` under
        ``fixed_density=True``), in which case building a redundant constraint
        is skipped. This is the generic "everything not otherwise governed
        flows straight through" wiring: ``SISOBlock`` calls it with no
        exclusions (the flow pass-through is itself a pass-through equality);
        ``Tank`` excludes the flow-basis variable because its holdup
        equation governs flow instead. Note this is intra-unit property
        copying, distinct from a physical bypass *stream* that diverts flow
        around a unit (see ``flexops.logic.bypass.add_bypass``).

        Gated by ``self.config.allow_pass_through``: when it is ``False`` this
        method builds **no** constraints (a developer wires the missing
        relationship by hand, and the model's degrees of freedom reflect the
        unlinked state variables); when ``True`` it builds them.

        Args:
            inlet: The unit's inlet Port carrying the source state variables.
            outlet: The unit's outlet Port receiving the passed-through values.
            exclude_vars: State-variable names to leave unlinked (e.g. a
                topology's flow-basis variable when the unit governs flow
                itself).

        Raises:
            FlexConfigError: If a name in ``exclude_vars`` is not a state
                variable exposed by ``inlet``, or ``inlet``/``outlet`` were
                not built by :meth:`add_stream_ports` (no sibling
                ``"{port_name}_state"`` block).

        Note:
            Ports built via ``add_inlet_port``/``add_outlet_port`` expose
            their members as auto-generated ``Reference`` objects
            (``port.vars[name]``), which carry an extra leading
            ``UnindexedComponent_set`` dimension and are awkward to index
            directly. This method instead resolves each port's sibling
            ``"{port_name}_state"`` block (the convention
            :meth:`add_stream_ports` establishes) and builds constraints
            directly against its state variables, so indices stay exactly
            ``inlet_var.index_set()`` (e.g. ``(t, phase)``), with no leaked
            reference dimension.
        """
        if not self.config.allow_pass_through:
            return
        inlet_state = inlet.parent_block().find_component(f"{inlet.local_name}_state")
        outlet_state = outlet.parent_block().find_component(
            f"{outlet.local_name}_state"
        )
        if inlet_state is None or outlet_state is None:
            raise FlexConfigError(
                "add_pass_through_constraints requires ports built by "
                "add_stream_ports (each needs a sibling "
                f"'{{port_name}}_state' block); got inlet={inlet.local_name!r}, "
                f"outlet={outlet.local_name!r}.",
                field="inlet/outlet",
                value=(inlet.local_name, outlet.local_name),
            )
        inlet_vars = inlet_state.define_state_vars()
        outlet_vars = outlet_state.define_state_vars()

        known = set(inlet_vars)
        unknown = set(exclude_vars) - known
        if unknown:
            raise FlexConfigError(
                f"add_pass_through_constraints: exclude_vars {sorted(unknown)} are "
                f"not state variables on this port; known: {sorted(known)}.",
                field="exclude_vars",
                value=sorted(unknown),
            )
        for name, inlet_var in inlet_vars.items():
            if name in exclude_vars:
                continue
            if all(inlet_var[idx].fixed for idx in inlet_var):
                continue
            outlet_var = outlet_vars[name]

            def _pass_through_rule(_b, *idx_parts, _in=inlet_var, _out=outlet_var):
                idx = idx_parts[0] if len(idx_parts) == 1 else idx_parts
                return _out[idx] == _in[idx]

            self.add_component(
                f"pass_through_{name}_eq",
                pyo.Constraint(
                    inlet_var.index_set(),
                    rule=_pass_through_rule,
                    doc=f"Pass-through: outlet {name} equals inlet {name}.",
                ),
            )

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
        units; first-classed on ``BatteryModel``.

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

    # -- config-driven construction ---------------------------------

    @classmethod
    def build_from_config(cls, cfg: UnitConfig, **kwargs):
        """Construct a unit from a validated ``UnitConfig``.

        Raises:
            NotImplementedError: Always; config-driven construction and the
                whole-model ``flexops.build_model``.
        """
        # TODO: whoever implements this JSON-to-model bridge must first parse the
        # units that persisted configs carry as plain text into Pyomo units --
        # CostingConfig.energy_prices entries ({"value": 0.12, "units": "USD/kWh"})
        # and TimeConfig.time_step ("15 min"). No such parser exists anywhere yet;
        # the runtime APIs take units-carrying Pyomo expressions directly, so
        # nothing has needed one. Building the model straight from those strings
        # without converting them would silently mis-scale prices and timesteps.
        # See architecture §2.3 (the config artifact) and conventions §4 (the two
        # config layers, which "never mix").
        raise NotImplementedError(
            "build_from_config is not implemented; build units directly."
        )
