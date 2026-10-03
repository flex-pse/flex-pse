"""ReactorBase(OpsBlockData): shared ports, intensive states, power, and thermal lag.

The abstract parent of :class:`~flexops.unit_models.reactor.species.SpeciesReactor`
and :class:`~flexops.unit_models.reactor.lumped.LumpedReactor`. It builds the
named inlet/outlet ports, ties every inlet's intensive states to the reference
inlet, passes them through to every outlet, declares the electrical power
relation, and optionally a lumped thermal lag. Each subclass writes its own
flow accumulation in :meth:`ReactorBaseData._build_holdup`. See
``docs/explanation/reactor_models.md`` for the equations.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.unit_models._multiport import single_flow_phase, validate_port_names


def _names_domain(value) -> tuple[str, ...]:
    """ConfigValue domain: coerce port names to a tuple (validated in ``build``)."""
    return tuple(value)


@declare_process_block_class("ReactorBase")
class ReactorBaseData(OpsBlockData):
    """Abstract reactor with named ports, power, and an optional thermal lag.

    Config:
        ``inlet_names`` (default ``("feed",)``) and ``outlet_names`` (default
        ``("product",)``) name the ports; the first of each is the reference.
        ``energy_intensity`` (default 0 kWh/m^3) scales power with the
        reference outlet flow. ``has_thermal_dynamics`` (default False) adds
        ``reactor_temperature`` with ``thermal_time_constant`` (default 1 hr),
        ``thermal_gain`` (default 0 K/kW), and ``initial_temperature``
        (default 298.15 K).

    Raises:
        NotImplementedError: If built directly; build a subclass.
        FlexConfigError: If port names are invalid, the package is not
            single-phase, or thermal dynamics are requested on a package with
            no temperature state.
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "inlet_names",
        ConfigValue(
            default=("feed",),
            domain=_names_domain,
            description="Inlet port names, built as f'inlet_{name}'. The first "
            "is the reference inlet whose intensive states every other inlet "
            "and every outlet is held at.",
        ),
    )
    CONFIG.declare(
        "outlet_names",
        ConfigValue(
            default=("product",),
            domain=_names_domain,
            description="Outlet port names, built as f'outlet_{name}'. The "
            "first is the reference outlet, the power-intensity basis.",
        ),
    )
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.0 * pyunits.kWh / pyunits.m**3,
            description="Electrical energy per unit volume leaving the "
            "reference outlet (a fixed, regressable Var once built), kWh/m^3.",
        ),
    )
    CONFIG.declare(
        "has_thermal_dynamics",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether to build the lumped reactor_temperature state "
            "(a first-order lag toward a steady-state temperature).",
        ),
    )
    CONFIG.declare(
        "thermal_time_constant",
        ConfigValue(
            default=1.0 * pyunits.hr,
            description="Thermal lag time constant (a fixed, regressable Var).",
        ),
    )
    CONFIG.declare(
        "thermal_gain",
        ConfigValue(
            default=0.0 * pyunits.K / pyunits.kW,
            description="Steady-state temperature rise per kW of electrical "
            "power (a fixed, regressable Var).",
        ),
    )
    CONFIG.declare(
        "initial_temperature",
        ConfigValue(
            default=298.15 * pyunits.K,
            description="Reactor temperature at the first time point (a "
            "rolling-horizon initial-state Param).",
        ),
    )

    def build(self) -> None:
        """Build ports, the subclass holdup, intensive states, power, and thermal."""
        super().build()
        validate_port_names(self.config.inlet_names, "inlet_names")
        validate_port_names(self.config.outlet_names, "outlet_names")
        self._phase = single_flow_phase(
            self.config.property_package, type(self).__name__
        )
        self.add_stream_ports(
            inlet_ports=tuple(f"inlet_{name}" for name in self.config.inlet_names),
            outlet_ports=tuple(f"outlet_{name}" for name in self.config.outlet_names),
        )
        self._check_thermal_package()
        self._build_flow_references()
        self._register_stream_states()
        self._build_holdup()
        self._tie_inlet_states()
        self._build_outlet_states()
        self.add_constant_intensity_relation(
            self._flow_out(self.config.outlet_names[0]),
            intensity=self.config.energy_intensity,
        )
        if self.config.has_thermal_dynamics:
            self._build_thermal()

    def _build_holdup(self) -> None:
        """Build the subclass's flow accumulation.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            "ReactorBase is abstract; build SpeciesReactor or LumpedReactor."
        )

    def _state(self, port_name: str):
        """Return the state block behind the port named ``port_name``."""
        return self.find_component(f"{port_name}_state")

    def _flow_in(self, name: str):
        """Return the time-indexed flow Reference of inlet ``name``."""
        return self.find_component(f"flow_in_{name}")

    def _flow_out(self, name: str):
        """Return the time-indexed flow Reference of outlet ``name``."""
        return self.find_component(f"flow_out_{name}")

    def _flow_basis_name(self) -> str:
        """Return the property package's flow state-variable name."""
        return self.config.property_package.get_flow_basis_var_name()

    def _check_thermal_package(self) -> None:
        """Reject thermal dynamics on a package with no temperature state.

        Raises:
            FlexConfigError: If ``has_thermal_dynamics`` is on and the outlet
                state carries no ``temperature``.
        """
        if not self.config.has_thermal_dynamics:
            return
        outlet_vars = self._state(f"outlet_{self.config.outlet_names[0]}")
        if "temperature" not in outlet_vars.define_state_vars():
            raise FlexConfigError(
                "has_thermal_dynamics=True needs a property_package carrying a "
                "temperature state; build it with has_temperature=True or use "
                "SimpleGasFlow.",
                field="has_thermal_dynamics",
                value=True,
            )

    def _build_flow_references(self) -> None:
        """Expose each port's flow as ``flow_in_<name>``/``flow_out_<name>``."""
        flow_name = self._flow_basis_name()
        for prefix, names in (
            ("in", self.config.inlet_names),
            ("out", self.config.outlet_names),
        ):
            port_prefix = "inlet" if prefix == "in" else "outlet"
            for name in names:
                state = self._state(f"{port_prefix}_{name}")
                self.add_component(
                    f"flow_{prefix}_{name}",
                    pyo.Reference(state.find_component(flow_name)[:, self._phase]),
                )

    def _register_stream_states(self) -> None:
        """Register reference-inlet intensive states as inputs, outlets' as outputs."""
        flow_name = self._flow_basis_name()
        reference = self._state(f"inlet_{self.config.inlet_names[0]}")
        for name, var in reference.define_state_vars().items():
            if name != flow_name:
                self.register_io_variable(var, role="input")
        for outlet in self.config.outlet_names:
            for name, var in (
                self._state(f"outlet_{outlet}").define_state_vars().items()
            ):
                if name != flow_name:
                    self.register_io_variable(var, role="output")

    def _tie_inlet_states(self) -> None:
        """Hold every non-reference inlet's intensive states at the reference's."""
        others = self.config.inlet_names[1:]
        if not others:
            return
        tb = self._find_time_block()
        reference = f"inlet_{self.config.inlet_names[0]}"
        flow_name = self._flow_basis_name()
        for state_var in self._state(reference).define_state_vars():
            if state_var == flow_name:
                continue

            def _equality_rule(b, t, name, _v=state_var):
                other = b._state(f"inlet_{name}").find_component(_v)
                return other[t] == b._state(reference).find_component(_v)[t]

            self.add_component(
                f"inlet_state_equality_{state_var}",
                pyo.Constraint(
                    tb.time_index,
                    others,
                    rule=_equality_rule,
                    doc=f"Inlet {state_var} equals the reference inlet's.",
                ),
            )

    def _build_outlet_states(self) -> None:
        """Pass the reference inlet's intensive states through to every outlet."""
        exclude_vars = [self._flow_basis_name()]
        if self.config.has_thermal_dynamics:
            exclude_vars.append("temperature")
        reference = self.find_component(f"inlet_{self.config.inlet_names[0]}")
        for name in self.config.outlet_names:
            self.add_pass_through_constraints(
                reference,
                self.find_component(f"outlet_{name}"),
                exclude_vars=exclude_vars,
                name_prefix=f"pass_through_{name}",
            )

    def _build_thermal(self) -> None:
        """Build the lumped thermal lag and tie every outlet temperature to it."""
        tb = self._find_time_block()
        initial = pyo.value(pyunits.convert(self.config.initial_temperature, pyunits.K))
        tau = self.declare_process_parameter(
            "thermal_time_constant",
            self.config.thermal_time_constant,
            pyunits.hr,
            "Thermal lag time constant.",
            bounds=(1e-6, None),
        )
        gain = self.declare_process_parameter(
            "thermal_gain",
            self.config.thermal_gain,
            pyunits.K / pyunits.kW,
            "Steady-state temperature rise per kW of electrical power.",
        )
        self.initial_temperature = pyo.Param(
            initialize=initial,
            mutable=True,
            units=pyunits.K,
            doc="Reactor temperature at the first time point.",
        )
        tb.register_initial_state(self.initial_temperature)
        self.register_process_parameter(self.initial_temperature, regressable=False)

        self.reactor_temperature = pyo.Var(
            tb.time_index,
            initialize=initial,
            units=pyunits.K,
            doc="Lumped reactor temperature, shared by every outlet.",
        )
        self.temperature_steady_state = pyo.Var(
            tb.time_index,
            initialize=initial,
            units=pyunits.K,
            doc="Temperature the reactor relaxes toward at each time point.",
        )
        self.register_io_variable(self.reactor_temperature, role="output")
        inlet_temperature = self._state(
            f"inlet_{self.config.inlet_names[0]}"
        ).temperature

        @self.Constraint(
            tb.time_index,
            doc="Steady-state temperature: reference inlet temperature plus "
            "thermal_gain * power_electrical. Swappable.",
        )
        def thermal_steady_state_relation(b, t):
            return b.temperature_steady_state[t] == inlet_temperature[
                t
            ] + pyunits.convert(gain * b.power_electrical[t], pyunits.K)

        self.register_relation(
            self.thermal_steady_state_relation, target=self.temperature_steady_state
        )

        @self.Constraint(
            doc="Initial condition: temperature[0] == initial_temperature."
        )
        def initial_temperature_eq(b):
            return b.reactor_temperature[0] == b.initial_temperature

        @self.Constraint(
            list(tb.time_index)[1:],
            doc="Thermal lag (backward Euler): T[t] = T[t-1] + dt/tau*(T_ss - T[t]).",
        )
        def thermal_balance(b, t):
            rate = pyunits.convert(tb.dt / tau, pyunits.dimensionless)
            return b.reactor_temperature[t] == b.reactor_temperature[t - 1] + rate * (
                b.temperature_steady_state[t] - b.reactor_temperature[t]
            )

        @self.Constraint(
            tb.time_index,
            self.config.outlet_names,
            doc="Every outlet leaves at the reactor temperature.",
        )
        def outlet_temperature_eq(b, t, name):
            return b._state(f"outlet_{name}").temperature[t] == b.reactor_temperature[t]
