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
    RelationRecord,
)
from flexops.core.time_block import TimeBlockData, find_time_block


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


_INTERCEPT = "intercept"
"""str: reserved coefficient key naming a relationship's constant term."""

POLYNOMIAL_FORMS: dict[str, int | None] = {
    "linear": 1,
    "quadratic": 2,
    "bilinear": 2,
    "polynomial": None,
}
"""dict: functional-form name -> highest term degree it admits (None: any).

Every one of these is built by the same polynomial rule; the names differ only
in what they promise, and the degree bound is what makes a mislabelled
relationship an error rather than a silent surprise. ``"bilinear"`` is the
*expanded* form -- a constant, each input on its own, and their cross term.
"""


def _polynomial_body(unit, surrogate, target):
    """Return ``body(t)`` evaluating a polynomial relationship at time ``t``.

    Args:
        unit: The unit the relationship is built on.
        surrogate: The :class:`~flexcore.config.schema.SurrogateSpec`.
        target: The Var/Reference the relationship determines. Unused here —
            a polynomial reads its own units from each factor — but part of
            every builder's signature (see ``_RELATION_BUILDERS``), since a
            builder for a different functional form may need it.

    Returns:
        A callable taking a time index and returning a **dimensionless**
        expression; :meth:`OpsBlockData.swap_relation` multiplies it by
        ``target``'s own units.
    """
    terms = unit._polynomial_terms(
        surrogate, POLYNOMIAL_FORMS[surrogate.functional_form]
    )
    intercept = surrogate.coefficients.get(_INTERCEPT, 0.0)

    def body(t):
        total = intercept
        for coefficient, factors in terms:
            term = coefficient
            for var, exponent in factors:
                term = term * (var[t] / pyunits.get_units(var[t])) ** exponent
            total = total + term
        return total

    return body


_RELATION_BUILDERS = {name: _polynomial_body for name in POLYNOMIAL_FORMS}
"""dict: functional-form name -> the builder that turns a spec into a body.

The extension point for a new relationship shape: register a builder here and
the config can name it immediately, because
:class:`~flexcore.config.schema.SurrogateSpec.functional_form` is an open
string rather than a fixed schema enumeration. A builder is
``(unit, surrogate, target) -> body``, where ``body(t)`` returns either a
**dimensionless** expression for that time point or ``pyomo.environ.Constraint
.Skip`` to omit it (a lagged/state-space form uses this to skip the horizon
points where its lag does not exist). A builder may attach its own components
to ``unit`` (an auxiliary Var/Constraint a big-M or state-space form needs);
:meth:`OpsBlockData.swap_relation` finds and tracks them itself, so nothing
needs to be returned besides ``body``. Nothing here is polynomial-specific:
a softplus/ICNN forward pass, a ratio, or any other Pyomo-expressible function
registers exactly the same way.
"""


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

    #: Default role -> Pyomo component name mapping. Each IO-topology base sets
    #: its own generic vocabulary here; a physical subclass renames any subset by
    #: passing ``naming_dict`` to the base ``build()``, which registers it.
    _component_names: dict[str, str] = {}

    def create_stream_naming_convention(self, naming_dict: dict[str, str]) -> None:
        """Register this unit's role -> Pyomo component name mapping.

        Called once from an IO-topology base's ``build()``, before any named
        component is built, with either that topology's default vocabulary or
        the ``naming_dict`` a physical subclass passed up. Only the components
        are renamed; ports never are.

        Args:
            naming_dict: The complete role -> component name mapping this unit
                builds and looks names up through.
        """
        self._component_names = dict(naming_dict)

    def _named(self, role: str) -> str:
        """Return the Pyomo component name registered for `role`.

        Args:
            role: A logical role in this unit's naming convention (e.g.
                ``"flow_in"``, ``"split_fraction"``).

        Returns:
            The Pyomo component name to build/look up for that role.
        """
        return self._component_names[role]

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

        Delegates to :func:`~flexops.core.time_block.find_time_block`, the one
        auto-discovery implementation shared with ``PlantBlock``/
        ``NetworkBlock``. The result is not cached on the block — assigning a
        Pyomo component to an attribute would trip ``Block.__setattr__``
        (pitfall 2).

        Returns:
            The model's ``TimeBlockData``.

        Raises:
            FlexConfigError: If the model has zero or several TimeBlocks.
        """
        return find_time_block(self.model())

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
        name, doc = nm.POWER_VARS[kind]
        tb = self._find_time_block()
        self.add_component(
            name,
            pyo.Var(tb.time_index, initialize=0.0, units=pyunits.kW, doc=doc),
        )
        var = self.find_component(name)
        self.register_power(var, kind=kind, temperature=temperature)
        return var

    def declare_process_parameter(
        self,
        name: str,
        value,
        units,
        doc: str,
        *,
        bounds: tuple[float | None, float | None] | None = None,
        regressable: bool = True,
    ):
        """Create, fix, register, and return a scalar process-parameter Var.

        The parameter twin of :meth:`declare_power`, and the one way flex-pse
        declares a design or fitted coefficient: a **scalar Var fixed at its
        configured value** rather than a ``Param``, so a design mode or a
        regression can unfix it in place without any component being replaced
        (conventions §9). Because it is registered, the value is reachable
        afterwards through :meth:`update_parameters`.

        Args:
            name: The component name to attach the Var under.
            value: The initial value, either a units-carrying Pyomo expression
                (converted into ``units``) or a bare number, taken to be in
                ``units`` already.
            units: The Var's Pyomo units.
            doc: The Var's ``doc=`` string; required, since the generated docs
                render it.
            bounds: Optional ``(lower, upper)`` bounds, in ``units``. Give them
                whenever unfixing the parameter would otherwise admit a
                physically meaningless value (an efficiency above one, say);
                ``None`` leaves it unbounded.
            regressable: Whether FlexParameterize may fit this parameter.

        Returns:
            The created, fixed scalar ``Var``.
        """
        magnitude = (
            float(value)
            if isinstance(value, numbers.Real) and not isinstance(value, bool)
            else float(pyo.value(pyunits.convert(value, units)))
        )
        self.add_component(
            name,
            pyo.Var(initialize=magnitude, bounds=bounds, units=units, doc=doc),
        )
        var = self.find_component(name)
        var.fix(magnitude)
        self.register_process_parameter(var, regressable=regressable)
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
        name_prefix: str = "pass_through",
    ) -> None:
        """Pass non-excluded, non-fixed inlet state variables straight to the outlet.

        For each state-variable name exposed by ``inlet`` and not in
        ``exclude_vars``, adds an equality Constraint ``outlet_var[idx] ==
        inlet_var[idx]`` over every index the variable carries -- unless every
        entry of that variable is already ``fixed`` (e.g. an inlet held at a
        known pressure), in which case building a redundant constraint
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
            name_prefix: Prefix of the Constraint names this builds
                (``"{name_prefix}_{state_var}_eq"``). A multi-outlet topology
                calls this once per outlet and must vary the prefix, since two
                calls would otherwise collide on one component name.

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
                f"{name_prefix}_{name}_eq",
                pyo.Constraint(
                    inlet_var.index_set(),
                    rule=_pass_through_rule,
                    doc=f"Pass-through: outlet {name} equals inlet {name}.",
                ),
            )

    # -- flow <-> energy relationship (the one place it is built) ----------

    def add_constant_intensity_relation(
        self,
        flow,
        *,
        kind: nm.PowerKind = nm.PowerKind.ELECTRICAL,
        intensity,
        temperature=None,
    ) -> None:
        """Declare a power draw and tie it to ``flow`` by a constant intensity.

        Every flex-pse unit's flow-to-energy relationship is built here (there
        is no separate regression unit class): it declares ``power_<kind>[t]``,
        an intensity Var (``energy_intensity`` for an electrical draw,
        ``thermal_intensity`` for a heat duty) fixed at ``intensity`` and
        registered as a regressable process parameter, and the equality

        .. math:: P_{kind}[t] = \\text{intensity} \\cdot \\dot{V}[t]

        as a Constraint named ``power_<kind>_relation``. **That name is the
        swap contract**: it is registered (see :meth:`register_relation`) so
        FlexParameterize can deactivate exactly it and attach a fitted
        replacement (see :meth:`swap_relation`). When this unit was built from
        a config whose ``SurrogateSpec`` asks for a richer functional form, the
        swap is applied here, at construction.

        ``flow`` is the unit's **product** stream — what it delivers, not what
        it takes in — so the intensity reads as energy per unit of output (a
        reverse-osmosis skid's kWh per m³ of permeate, the industry's specific
        energy consumption). For a unit that passes its flow straight through
        the two coincide; for one with a recovery or a loss they do not, and
        the product is the meaningful denominator. The basis is recorded on the
        unit's registry so FlexParameterize regresses against the same stream.

        Args:
            flow: The unit's time-indexed product flow Var/Reference
                (``flow[t]``) the draw scales with.
            kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.
            intensity: Energy per unit volume (e.g. ``0.5 * pyunits.kWh /
                pyunits.m**3``), converted to kWh/m³.
            temperature: The heat-duty temperature (required for
                ``PowerKind.THERMAL``).
        """
        tb = self._find_time_block()
        power = self.declare_power(kind, temperature=temperature)
        self.register_io_variable(power, role="output")
        self._io_registry.intensity_basis[kind] = flow.local_name

        intensity_var = self.declare_process_parameter(
            nm.INTENSITY_VARS[kind],
            intensity,
            pyunits.kWh / pyunits.m**3,
            f"{kind.value.capitalize()} energy per unit volume processed.",
        )

        relation_name = f"{nm.POWER_VARS[kind][0]}_relation"
        self.add_component(
            relation_name,
            pyo.Constraint(
                tb.time_index,
                rule=lambda b, t: power[t]
                == pyunits.convert(intensity_var * flow[t], pyunits.kW),
                doc=f"{relation_name}: power == {nm.INTENSITY_VARS[kind]} * "
                "flow. kWh/m^3 * m^3/hr = kW "
                "exactly, no fudge factor. FlexParameterize swaps this "
                "Constraint in place when it fits a richer relationship.",
            ),
        )
        self.register_relation(self.find_component(relation_name), target=power)

        surrogate = getattr(self.config.flexops_config, "surrogate", None)
        if surrogate is not None and surrogate.functional_form != "constant_intensity":
            self.swap_relation(relation_name, surrogate)

    def _resolve_input(self, name: str, field: str):
        """Return the variable ``name`` refers to on this unit.

        Args:
            name: A component name, possibly dotted into a sub-block (e.g.
                ``"outlet_state.pressure"``).
            field: The surrogate field the name came from, for the error.

        Returns:
            The resolved Pyomo component.

        Raises:
            FlexConfigError: If the name is not a component on this unit.
        """
        var = self.find_component(name)
        if var is None:
            raise FlexConfigError(
                f"Fitted relationship names input {name!r}, which is not a "
                f"variable on {self.name!r}.",
                field=field,
                value=name,
            )
        return var

    def _polynomial_terms(self, surrogate, max_degree: int | None):
        """Parse ``surrogate.coefficients`` into resolved polynomial terms.

        A coefficient key is a ``*``-separated product of input-variable names,
        each optionally raised to an integer power with ``^``, so one grammar
        spans every polynomial relationship: ``"flow_out"`` is linear,
        ``"flow_out^2"`` quadratic, ``"flow_out*outlet_state.pressure"`` the
        cross term of an expanded bilinear form.

        Args:
            surrogate: The :class:`~flexcore.config.schema.SurrogateSpec`.
            max_degree: Highest total degree the functional form allows, or
                None for no bound.

        Returns:
            ``[(coefficient, [(var, exponent), ...]), ...]``, excluding the
            constant ``"intercept"`` term.

        Raises:
            FlexConfigError: If a term's exponent is not a positive integer, a
                factor is not a variable on this unit, or a term's total degree
                exceeds ``max_degree``.
        """
        terms = []
        for key, coefficient in surrogate.coefficients.items():
            if key == _INTERCEPT:
                continue
            factors = []
            for token in key.split("*"):
                name, _, exponent = token.strip().partition("^")
                if exponent and not exponent.isdigit():
                    raise FlexConfigError(
                        f"Coefficient term {key!r} has exponent {exponent!r}; "
                        "write a term as 'a', 'a^2', or 'a*b'.",
                        field="coefficients",
                        value=key,
                    )
                factors.append(
                    (self._resolve_input(name, "coefficients"), int(exponent or 1))
                )
            degree = sum(exponent for _, exponent in factors)
            if max_degree is not None and degree > max_degree:
                raise FlexConfigError(
                    f"Term {key!r} is degree {degree}, but functional form "
                    f"{surrogate.functional_form!r} allows at most "
                    f"{max_degree}. Label the relationship with a form that "
                    "admits it (e.g. 'polynomial').",
                    field="coefficients",
                    value=key,
                )
            terms.append((coefficient, factors))
        return terms

    def register_relation(self, constraint, target) -> None:
        """Declare ``constraint`` swappable: a relationship determining ``target``.

        Only a *registered* relationship may ever be swapped by
        :meth:`swap_relation` — a mass balance or other conservation law that
        is never registered can never be swapped, by construction (there is no
        naming convention to accidentally satisfy).
        :meth:`add_constant_intensity_relation` registers the relation it
        builds; a unit with its own performance relationship (an RO skid's
        ``split_definition``, a tank's ``level_definition``) registers it the
        same way.

        Args:
            constraint: The live, time-indexed Constraint to declare swappable.
            target: The live Var/Reference it determines. Must be indexed over
                exactly one dimension (time); a target indexed over more (a
                per-component flux, say) is out of scope for this milestone.

        Raises:
            FlexConfigError: If ``target``'s index set is not one-dimensional.
        """
        if target.index_set().dimen != 1:
            raise FlexConfigError(
                f"register_relation requires a one-dimensional target, got "
                f"{target.local_name!r} with dimen={target.index_set().dimen}. "
                "A relationship over a multi-dimensional target (e.g. per "
                "component) is out of scope for this milestone — see "
                "M10b_parameterize_multicomponent.",
                field="target",
                value=target.local_name,
            )
        self._io_registry.relations.append(
            RelationRecord(
                constraint=constraint,
                name=constraint.local_name,
                target=target,
                target_name=target.local_name,
            )
        )

    def swap_relation(self, relation_name: str, surrogate) -> None:
        """Replace a registered relationship in place, from a fitted spec.

        Deactivates the named relation (never deletes it — conventions §9),
        deactivates whatever an earlier swap's builder attached, and adds
        ``{relation_name}_fitted`` built from ``surrogate``, reusing the unit's
        existing registered variables so ports and arcs are untouched. One
        implementation, every caller: construction-time
        (:meth:`add_constant_intensity_relation`, via a config's
        ``SurrogateSpec``), runtime (``flexparameterize.apply_to_model``), and
        any unit that hand-builds and registers its own relationship.

        Every polynomial form shares one builder:

        .. math::

            y[t] = c_0 + \\sum_k c_k \\prod_j x_{kj}[t]^{n_{kj}}

        where ``y`` is the registered target and each term comes from a
        coefficient key — :data:`POLYNOMIAL_FORMS` lists the names that select
        it and the degree each admits, so ``"linear"`` takes
        ``{"intercept": 5, "flow_out": 0.4}`` and ``"bilinear"`` additionally
        takes ``{"outlet_state.pressure": 1e-5,
        "flow_out*outlet_state.pressure": 2e-6}``. But the builder registry is
        not limited to polynomials: a builder is
        ``(unit, surrogate, target) -> body``, where ``body(t)`` is any
        Pyomo-expressible, dimensionless function of the unit's own variables
        (a ratio, a softplus/ICNN forward pass, a lag polynomial) — see
        ``_RELATION_BUILDERS``'s own docstring for the full contract. Every
        factor a body reads is normalized by its own units, and the
        result is multiplied by ``target``'s own units here, so a coefficient
        is read in the target's units over the product of its factors' units.
        Registering another form is an entry in ``_RELATION_BUILDERS``, never
        a config-schema change.

        Args:
            relation_name: Local name of a relation this unit registered via
                :meth:`register_relation` (e.g. ``"power_electrical_relation"``,
                ``"split_definition"``).
            surrogate: The fitted
                :class:`~flexcore.config.schema.SurrogateSpec`.

        Raises:
            FlexConfigError: If ``relation_name`` was never registered, no
                builder is registered for the functional form, a named input
                variable is not found on the unit, or a term exceeds the
                form's degree.
        """
        record = next(
            (r for r in self._io_registry.relations if r.name == relation_name), None
        )
        if record is None:
            known = (
                ", ".join(repr(r.name) for r in self._io_registry.relations) or "none"
            )
            raise FlexConfigError(
                f"{relation_name!r} is not a registered relation on "
                f"{self.name!r} (registered: {known}).",
                field="relation_name",
                value=relation_name,
            )
        build = _RELATION_BUILDERS.get(surrogate.functional_form)
        if build is None:
            known_forms = ", ".join(repr(name) for name in sorted(_RELATION_BUILDERS))
            raise FlexConfigError(
                f"No relationship builder is registered for functional form "
                f"{surrogate.functional_form!r}. Known forms: {known_forms}.",
                field="functional_form",
                value=surrogate.functional_form,
            )
        for name in surrogate.input_variables:
            self._resolve_input(name, "input_variables")

        (record.fitted if record.fitted is not None else record.constraint).deactivate()
        for component in record.components:
            deactivate = getattr(component, "deactivate", None)
            if deactivate is not None:
                deactivate()

        before = set(self.component_map())
        body = build(self, surrogate, record.target)
        record.components = [
            self.find_component(name) for name in set(self.component_map()) - before
        ]

        record.swap_count += 1
        suffix = "" if record.swap_count == 1 else f"_{record.swap_count}"
        fitted_name = f"{relation_name}_fitted{suffix}"
        target = record.target

        def _rule(b, t, _body=body, _target=target):
            expr = _body(t)
            if expr is pyo.Constraint.Skip:
                return pyo.Constraint.Skip
            return _target[t] == expr * pyunits.get_units(_target[t])

        self.add_component(
            fitted_name,
            pyo.Constraint(
                target.index_set(),
                rule=_rule,
                doc=f"Fitted relationship ({surrogate.functional_form}), "
                f"replacing the deactivated {relation_name}. Coefficients are "
                f"read in {target.local_name!r}'s own units over the product "
                "of each term's factors' own units.",
            ),
        )
        record.fitted = self.find_component(fitted_name)

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

        The per-unit primitive behind
        :func:`~flexops.core.build.build_model`. A path or mapping is
        round-tripped through the pydantic schema first, so a raw dict is never
        passed onward (conventions §4) and an invalid one raises
        ``pydantic.ValidationError`` naming the offending field path (the error
        is deliberately **not** wrapped — the field path is the contract).

        The unit-model class always comes from ``cfg.unit_model_class``,
        resolved against ``flexops.unit_models``. Persisted construction
        options carry their units as data: a value written
        ``{"value": 0.5, "units": "kWh/m^3"}`` becomes a units-carrying Pyomo
        expression, anything else passes through unchanged -- including a plain
        string naming an enum member (``"detail": "polarization"``), which the
        receiving ``ConfigValue``'s own domain then validates. Runtime-only
        options that cannot be serialized (``property_package``,
        ``costing_package``) are supplied by the caller through ``kwargs``,
        which wins over the config on a key collision.

        Args:
            cfg: A ``UnitConfig``, or a mapping/path to validate into one.
            **kwargs: Extra runtime construction options.

        Returns:
            The constructible unit block, to be assigned onto a model
            (``m.u = OpsBlockData.build_from_config(cfg, ...)``).

        Raises:
            pydantic.ValidationError: If ``cfg`` fails schema validation.
            FlexConfigError: If ``cfg.unit_model_class`` is not a flexops
                unit-model class.
        """
        # Local imports: both modules import this one, so the unit-model
        # registry and the units parser are only reachable at call time.
        from flexops import unit_models
        from flexops.core.build import parse_quantity

        if not isinstance(cfg, UnitConfig):
            cfg = UnitConfig.model_validate(cfg)
        block_class = getattr(unit_models, cfg.unit_model_class, None)
        if block_class is None:
            known = ", ".join(sorted(unit_models.__all__))
            raise FlexConfigError(
                f"Unknown unit_model_class {cfg.unit_model_class!r}. Known "
                f"flexops unit models: {known}.",
                field="unit_model_class",
                value=cfg.unit_model_class,
            )
        options = {
            name: parse_quantity(value, strict=False)
            for name, value in cfg.construction_options.items()
        }
        return block_class(
            **options,
            unit_commitment=cfg.unit_commitment,
            external_dispatch=cfg.external_dispatch,
            flexops_config=cfg,
            **kwargs,
        )
