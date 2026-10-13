"""Boiler(OpsBlockData): fired or heat-recovery steam raising with a thermal lag."""

import enum

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.logic.status import add_status
from flexops.unit_models.powergeneration.utils import (
    heating_value_mismatch,
    heating_values_domain,
    inlet_names_domain,
    utility_fuel_source_domain,
    validate_fuel_sources,
)


class BoilerType(enum.StrEnum):
    """Where a :class:`Boiler` gets its heat."""

    FIRED = "fired"
    HEAT_RECOVERY = "heat_recovery"


class BoilerInletName(enum.StrEnum):
    """Inlet names a :class:`Boiler` builds itself, so no fuel may take them."""

    FEEDWATER = "feedwater"
    HOT_GAS = "hot_gas"


class FiringOption(enum.StrEnum):
    """:class:`Boiler` config options that need at least one fuel source."""

    EFFICIENCY = "efficiency"
    MAX_FIRING_RATE = "max_firing_rate"


class HeatRecoveryOption(enum.StrEnum):
    """:class:`Boiler` config options that need ``BoilerType.HEAT_RECOVERY``."""

    HOT_GAS_PROPERTY_PACKAGE = "hot_gas_property_package"
    GAS_HEAT_CONTENT = "gas_heat_content"
    STACK_TEMPERATURE = "stack_temperature"


@declare_process_block_class("Boiler")
class BoilerData(OpsBlockData):
    """Steam boiler: fuel and/or hot gas in, steam out, with a first-order lag.

    Config:
        ``boiler_type``: a :class:`BoilerType`. ``fuel_inlet_names`` /
        ``utility_fuel_source``: fuel burned from inlet ports / from
        utilities, each needing a ``heating_values`` entry. ``efficiency``,
        ``max_firing_rate``: firing options, only with fuel. ``gas_heat_content``,
        ``stack_temperature``: heat-recovery options. ``no_load_loss``,
        ``min_heat_output``, ``max_heat_output``, ``time_constant``,
        ``initial_heat_output``, ``steam_heat_content``,
        ``steam_to_water_density_ratio``, ``steam_pressure``,
        ``steam_temperature``, ``aux_energy_intensity``: shared. One property
        package per stream: ``fuel_property_package``,
        ``hot_gas_property_package``, ``feedwater_property_package``,
        ``steam_property_package``.
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "boiler_type",
        ConfigValue(
            default=BoilerType.FIRED,
            description="Heat source: BoilerType.FIRED burns fuel; "
            "BoilerType.HEAT_RECOVERY recovers heat from a hot-gas inlet, "
            "optionally with duct firing.",
        ),
    )
    CONFIG.declare(
        "fuel_property_package",
        ConfigValue(
            default=None,
            description="Single-phase gas package for fuel inlet ports. "
            "Required when fuel_inlet_names is given.",
        ),
    )
    CONFIG.declare(
        "hot_gas_property_package",
        ConfigValue(
            default=None,
            description="Single-phase gas package for the hot-gas inlet and "
            "cooled-gas outlet. Required for BoilerType.HEAT_RECOVERY.",
        ),
    )
    CONFIG.declare(
        "feedwater_property_package",
        ConfigValue(
            default=None,
            description="Single-phase liquid package for the feedwater inlet.",
        ),
    )
    CONFIG.declare(
        "steam_property_package",
        ConfigValue(
            default=None,
            description="Single-phase gas package for the steam outlet.",
        ),
    )
    CONFIG.declare(
        "fuel_inlet_names",
        ConfigValue(
            default=None,
            domain=inlet_names_domain,
            description="Fuels fed through inlet ports; fuel i is built as port "
            "f'inlet_{name}'. Names must be unique and may not be 'feedwater' or "
            "'hot_gas'.",
        ),
    )
    CONFIG.declare(
        "utility_fuel_source",
        ConfigValue(
            default=None,
            domain=utility_fuel_source_domain,
            description="Fuels pulled from utilities, each a utility_flow_{name} "
            "Var (m^3/hr) registered for costing. Must not overlap "
            "fuel_inlet_names.",
        ),
    )
    CONFIG.declare(
        "heating_values",
        ConfigValue(
            default=None,
            domain=heating_values_domain,
            description="Mapping of every fuel name to its lower heating value "
            "per unit volume (kWh/m^3).",
        ),
    )
    CONFIG.declare(
        "efficiency",
        ConfigValue(
            default=0.9,
            description="Fraction of fuel heat input absorbed by the steam, in "
            "(0, 1]. Only with fuel sources.",
        ),
    )
    CONFIG.declare(
        "max_firing_rate",
        ConfigValue(
            default=13000 * pyunits.kW,
            description="Burner capacity: maximum total fuel heat input (kW). "
            "Only with fuel sources.",
        ),
    )
    CONFIG.declare(
        "gas_heat_content",
        ConfigValue(
            default=0.15 * pyunits.kWh / pyunits.m**3,
            description="Heat recovered per unit volume of hot gas (kWh/m^3). "
            "BoilerType.HEAT_RECOVERY only.",
        ),
    )
    CONFIG.declare(
        "stack_temperature",
        ConfigValue(
            default=423 * pyunits.K,
            description="Cooled-gas outlet temperature (K). "
            "BoilerType.HEAT_RECOVERY only.",
        ),
    )
    CONFIG.declare(
        "no_load_loss",
        ConfigValue(
            default=0 * pyunits.kW,
            description="Standing heat loss while on (kW).",
        ),
    )
    CONFIG.declare(
        "min_heat_output",
        ConfigValue(
            default=4000 * pyunits.kW,
            description="Minimum stable steam heat output while on (kW).",
        ),
    )
    CONFIG.declare(
        "max_heat_output",
        ConfigValue(
            default=10000 * pyunits.kW,
            description="Rated steam heat output (kW).",
        ),
    )
    CONFIG.declare(
        "time_constant",
        ConfigValue(
            default=300 * pyunits.s,
            description="First-order lag of steam heat output behind absorbed "
            "heat (s). Zero makes the boiler static.",
        ),
    )
    CONFIG.declare(
        "initial_heat_output",
        ConfigValue(
            default=0 * pyunits.kW,
            description="Steam heat output before the first time point (kW); the "
            "rolling-horizon initial state.",
        ),
    )
    CONFIG.declare(
        "steam_heat_content",
        ConfigValue(
            default=3.4 * pyunits.kWh / pyunits.m**3,
            description="Heat carried per unit volume of outlet steam, relative "
            "to feedwater (kWh/m^3).",
        ),
    )
    CONFIG.declare(
        "steam_to_water_density_ratio",
        ConfigValue(
            default=0.0054,
            description="Outlet steam density over feedwater density "
            "(dimensionless); sets feedwater volume per steam volume.",
        ),
    )
    CONFIG.declare(
        "steam_pressure",
        ConfigValue(
            default=10 * pyunits.bar,
            description="Steam outlet pressure (Pa).",
        ),
    )
    CONFIG.declare(
        "steam_temperature",
        ConfigValue(
            default=453 * pyunits.K,
            description="Steam outlet temperature (K); also the thermal export's "
            "temperature.",
        ),
    )
    CONFIG.declare(
        "aux_energy_intensity",
        ConfigValue(
            default=0 * pyunits.kWh / pyunits.m**3,
            description="Auxiliary electrical draw per unit volume of steam "
            "(kWh/m^3).",
        ),
    )

    def build(self) -> None:
        """Validate the config, then build heat input, lag, steam, and power."""
        super().build()
        self._validate_config()
        self._build_fuel()
        self._build_hot_gas()
        self._build_heat_balance()
        self._build_steam()
        self._build_power()

    # -- config --------------------------------------------------------------

    def _fail(self, field: str, message: str) -> None:
        """Raise a :class:`FlexConfigError` naming ``field``.

        Raises:
            FlexConfigError: Always.
        """
        raise FlexConfigError(
            message, field=field, value=self.config._data[field].value()
        )

    def _validate_config(self) -> None:
        """Check type, fuel sources, packages, and value ranges.

        Raises:
            FlexConfigError: If any option is invalid for the chosen type.
        """
        config = self.config
        if not isinstance(config.boiler_type, BoilerType):
            self._fail("boiler_type", "boiler_type must be a BoilerType member.")
        if config.fuel_inlet_names is None:
            config._data["fuel_inlet_names"].set_value(())
        heat_recovery = config.boiler_type is BoilerType.HEAT_RECOVERY
        user_set = {v.name() for v in config.user_values()}

        validate_fuel_sources(
            config.fuel_inlet_names,
            config.utility_fuel_source,
            inlet_field="fuel_inlet_names",
            require_any=not heat_recovery,
        )
        reserved = set(config.fuel_inlet_names) & {n.value for n in BoilerInletName}
        if reserved:
            self._fail("fuel_inlet_names", f"Reserved fuel name(s) {sorted(reserved)}.")
        missing, unknown = heating_value_mismatch(
            config.heating_values, self._fuel_names()
        )
        if missing or unknown:
            self._fail(
                "heating_values",
                f"heating_values is missing {missing} and has unknown {unknown}.",
            )

        if not self._fuel_names():
            for option in FiringOption:
                if option.value in user_set:
                    self._fail(
                        option.value, f"{option} needs at least one fuel source."
                    )
        if not heat_recovery:
            for option in HeatRecoveryOption:
                if option.value in user_set:
                    self._fail(
                        option.value, f"{option} needs BoilerType.HEAT_RECOVERY."
                    )

        required = ["feedwater_property_package", "steam_property_package"]
        if config.fuel_inlet_names:
            required.append("fuel_property_package")
        if heat_recovery:
            required.append("hot_gas_property_package")
        for option in required:
            package = config._data[option].value()
            if package is None or len(list(package.phase_list)) != 1:
                self._fail(option, f"{option} must be a single-phase package.")

        if not 0 < config.efficiency <= 1:
            self._fail("efficiency", "efficiency must be in (0, 1].")
        if pyo.value(pyunits.convert(config.time_constant, pyunits.s)) < 0:
            self._fail("time_constant", "time_constant must be non-negative.")
        if pyo.value(pyunits.convert(config.min_heat_output, pyunits.kW)) > pyo.value(
            pyunits.convert(config.max_heat_output, pyunits.kW)
        ):
            self._fail("min_heat_output", "min_heat_output exceeds max_heat_output.")

    def _fuel_names(self) -> set[str]:
        """Return every configured fuel source name."""
        return set(self.config.fuel_inlet_names) | set(
            self.config.utility_fuel_source or ()
        )

    # -- heat input ------------------------------------------------------------

    def _build_fuel(self) -> None:
        """Build fuel inlets, utility fuel Vars, heating values, and burner limit."""
        tb = self._find_time_block()
        self._fuel_flows = {}
        for name in self.config.fuel_inlet_names:
            pkg = self.config.fuel_property_package
            self._add_stream_port(
                pkg, tb, f"inlet_{name}", ("flow_vol_phase",), role="input"
            )
            state = self.find_component(f"inlet_{name}_state")
            phase = next(iter(pkg.phase_list))
            self.add_component(
                f"flow_in_{name}", pyo.Reference(state.flow_vol_phase[:, phase])
            )
            self._fuel_flows[name] = self.find_component(f"flow_in_{name}")
        for name in self.config.utility_fuel_source or ():
            self.add_component(
                f"utility_flow_{name}",
                pyo.Var(
                    tb.time_index,
                    bounds=(0.0, None),
                    units=pyunits.m**3 / pyunits.hr,
                    doc=f"Utility fuel '{name}' burned (m^3/hr).",
                ),
            )
            flow = self.find_component(f"utility_flow_{name}")
            self.register_fuel_usage(flow, fuel_name=name)
            self.register_io_variable(flow, role="input")
            self._fuel_flows[name] = flow
        if not self._fuel_flows:
            return

        heating_values = {
            name: self.declare_process_parameter(
                f"heating_value_{name}",
                self.config.heating_values[name],
                pyunits.kWh / pyunits.m**3,
                f"Lower heating value of fuel '{name}'.",
                bounds=(0.0, None),
            )
            for name in self._fuel_flows
        }
        self.declare_process_parameter(
            "efficiency",
            self.config.efficiency,
            pyunits.dimensionless,
            "Fraction of fuel heat input absorbed by the steam.",
            bounds=(0.0, 1.0),
        )
        self.declare_process_parameter(
            "max_firing_rate",
            self.config.max_firing_rate,
            pyunits.kW,
            "Burner capacity: maximum total fuel heat input.",
            bounds=(0.0, None),
        )

        @self.Expression(tb.time_index, doc="Total fuel heat input (kW).")
        def firing_rate(b, t):
            return pyunits.convert(
                sum(heating_values[n] * f[t] for n, f in self._fuel_flows.items()),
                pyunits.kW,
            )

        @self.Constraint(tb.time_index, doc="Fuel heat input <= max_firing_rate.")
        def firing_rate_limit(b, t):
            return b.firing_rate[t] <= b.max_firing_rate

    def _build_hot_gas(self) -> None:
        """Build the hot-gas inlet, cooled-gas outlet, and recovered-heat relation."""
        if self.config.boiler_type is not BoilerType.HEAT_RECOVERY:
            return
        tb = self._find_time_block()
        pkg = self.config.hot_gas_property_package
        phase = next(iter(pkg.phase_list))
        states = ("flow_vol_phase", "pressure", "temperature")
        self._add_stream_port(pkg, tb, "inlet_hot_gas", states, role="input")
        self._add_stream_port(pkg, tb, "outlet_gas", states, role="output")
        self.flow_hot_gas = pyo.Reference(
            self.inlet_hot_gas_state.flow_vol_phase[:, phase]
        )
        gas_heat_content = self.declare_process_parameter(
            "gas_heat_content",
            self.config.gas_heat_content,
            pyunits.kWh / pyunits.m**3,
            "Heat recovered per unit volume of hot gas.",
            bounds=(0.0, None),
        )
        stack_temperature = self.declare_process_parameter(
            "stack_temperature",
            self.config.stack_temperature,
            pyunits.K,
            "Cooled-gas outlet temperature.",
            bounds=(0.0, None),
        )

        @self.Constraint(tb.time_index, doc="Cooled-gas flow equals hot-gas flow.")
        def gas_flow_balance(b, t):
            return b.outlet_gas_state.flow_vol_phase[t, phase] == b.flow_hot_gas[t]

        self.add_pass_through_constraints(
            self.inlet_hot_gas,
            self.outlet_gas,
            exclude_vars=["flow_vol_phase", "temperature"],
            name_prefix="gas_pass_through",
        )

        @self.Constraint(tb.time_index, doc="Cooled-gas temperature is fixed.")
        def stack_temperature_eq(b, t):
            return b.outlet_gas_state.temperature[t] == stack_temperature

        self.heat_recovered = pyo.Var(
            tb.time_index,
            bounds=(0.0, None),
            units=pyunits.kW,
            doc="Heat recovered from the hot gas (kW).",
        )
        self.register_io_variable(self.heat_recovered, role="output")

        @self.Constraint(tb.time_index, doc="heat_recovered = gas_heat_content * gas.")
        def heat_recovered_relation(b, t):
            return b.heat_recovered[t] == pyunits.convert(
                gas_heat_content * b.flow_hot_gas[t], pyunits.kW
            )

        self.register_relation(self.heat_recovered_relation, self.heat_recovered)

    def _build_heat_balance(self) -> None:
        """Build heat output, status, absorbed heat, and the thermal lag."""
        tb = self._find_time_block()
        min_output = pyunits.convert(self.config.min_heat_output, pyunits.kW)
        max_output = pyunits.convert(self.config.max_heat_output, pyunits.kW)
        uc_on = self.config.unit_commitment.status
        self.heat_output = pyo.Var(
            tb.time_index,
            bounds=(0.0 if uc_on else pyo.value(min_output), pyo.value(max_output)),
            units=pyunits.kW,
            doc="Steam heat output (kW).",
        )
        self.register_io_variable(self.heat_output, role="output")
        if uc_on:
            add_status(self, self.heat_output, min_output, max_output)

        self.heat_absorbed = pyo.Var(
            tb.time_index, units=pyunits.kW, doc="Heat absorbed by the water (kW)."
        )
        self.register_io_variable(self.heat_absorbed, role="output")
        self.heat_dissipated = pyo.Var(
            tb.time_index,
            bounds=(0.0, None),
            units=pyunits.kW,
            doc="Heat vented or lost on ramp-down and shutdown (kW).",
        )
        no_load_loss = self.declare_process_parameter(
            "no_load_loss",
            self.config.no_load_loss,
            pyunits.kW,
            "Standing heat loss while on.",
            bounds=(0.0, None),
        )

        @self.Constraint(
            tb.time_index,
            doc="heat_absorbed = efficiency * firing_rate + heat_recovered - "
            "no_load_loss * status.",
        )
        def heat_absorbed_relation(b, t):
            total = -no_load_loss * (b.status[t] if uc_on else 1)
            if self._fuel_flows:
                total += b.efficiency * b.firing_rate[t]
            if self.config.boiler_type is BoilerType.HEAT_RECOVERY:
                total += b.heat_recovered[t]
            return b.heat_absorbed[t] == total

        self.register_relation(self.heat_absorbed_relation, self.heat_absorbed)

        time_constant = self.declare_process_parameter(
            "time_constant",
            self.config.time_constant,
            pyunits.s,
            "First-order lag of heat output behind absorbed heat.",
            bounds=(0.0, None),
        )
        self.initial_heat_output = pyo.Param(
            initialize=pyo.value(
                pyunits.convert(self.config.initial_heat_output, pyunits.kW)
            ),
            mutable=True,
            units=pyunits.kW,
            doc="Heat output before the first time point (rolling-horizon state).",
        )
        tb.register_initial_state(self.initial_heat_output)
        self.register_process_parameter(self.initial_heat_output, regressable=False)
        first = tb.time_index.first()

        @self.Constraint(
            tb.time_index,
            doc="Backward-Euler lag: time_constant * (Q[t] - Q[t-1]) = "
            "dt * (heat_absorbed - Q - heat_dissipated).",
        )
        def thermal_lag(b, t):
            previous = b.initial_heat_output if t == first else b.heat_output[t - 1]
            stored = pyunits.convert(
                time_constant * (b.heat_output[t] - previous), pyunits.kWh
            )
            supplied = pyunits.convert(
                tb.dt * (b.heat_absorbed[t] - b.heat_output[t] - b.heat_dissipated[t]),
                pyunits.kWh,
            )
            return stored == supplied

    # -- streams and power -------------------------------------------------------

    def _build_steam(self) -> None:
        """Build the feedwater inlet, steam outlet, and their relations."""
        tb = self._find_time_block()
        water_pkg = self.config.feedwater_property_package
        steam_pkg = self.config.steam_property_package
        self._add_stream_port(water_pkg, tb, "inlet_feedwater", (), role="input")
        self._add_stream_port(
            steam_pkg,
            tb,
            "outlet_steam",
            ("flow_vol_phase", "pressure", "temperature"),
            role="output",
        )
        water_phase = next(iter(water_pkg.phase_list))
        steam_phase = next(iter(steam_pkg.phase_list))
        self.flow_feedwater = pyo.Reference(
            self.inlet_feedwater_state.flow_vol_phase[:, water_phase]
        )
        self.register_io_variable(
            self.inlet_feedwater_state.flow_vol_phase, role="output"
        )
        self.flow_out = pyo.Reference(
            self.outlet_steam_state.flow_vol_phase[:, steam_phase]
        )
        self.pressure_out = pyo.Reference(self.outlet_steam_state.pressure)

        steam_heat_content = self.declare_process_parameter(
            "steam_heat_content",
            self.config.steam_heat_content,
            pyunits.kWh / pyunits.m**3,
            "Heat carried per unit volume of outlet steam.",
            bounds=(0.0, None),
        )
        density_ratio = self.declare_process_parameter(
            "steam_to_water_density_ratio",
            self.config.steam_to_water_density_ratio,
            pyunits.dimensionless,
            "Outlet steam density over feedwater density.",
            bounds=(0.0, None),
        )
        steam_pressure = self.declare_process_parameter(
            "steam_pressure",
            self.config.steam_pressure,
            pyunits.Pa,
            "Steam outlet pressure.",
            bounds=(0.0, None),
        )
        steam_temperature = self.declare_process_parameter(
            "steam_temperature",
            self.config.steam_temperature,
            pyunits.K,
            "Steam outlet temperature.",
            bounds=(0.0, None),
        )

        @self.Constraint(tb.time_index, doc="steam_heat_content * flow_out = Q.")
        def steam_flow_relation(b, t):
            return (
                pyunits.convert(steam_heat_content * b.flow_out[t], pyunits.kW)
                == b.heat_output[t]
            )

        self.register_relation(self.steam_flow_relation, self.flow_out)

        @self.Constraint(tb.time_index, doc="Feedwater volume = ratio * steam volume.")
        def feedwater_balance(b, t):
            return b.flow_feedwater[t] == pyunits.convert(
                density_ratio * b.flow_out[t], pyunits.get_units(b.flow_feedwater[t])
            )

        @self.Constraint(tb.time_index, doc="Steam outlet pressure = steam_pressure.")
        def steam_pressure_relation(b, t):
            return b.pressure_out[t] == steam_pressure

        self.register_relation(self.steam_pressure_relation, self.pressure_out)

        @self.Constraint(tb.time_index, doc="Steam outlet temperature is fixed.")
        def steam_temperature_eq(b, t):
            return b.outlet_steam_state.temperature[t] == steam_temperature

    def _build_power(self) -> None:
        """Declare the thermal export and the auxiliary electrical draw."""
        tb = self._find_time_block()
        power = self.declare_power(
            nm.PowerKind.THERMAL, temperature=self.config.steam_temperature
        )
        for t in tb.time_index:
            power[t].setub(0.0)

        @self.Constraint(tb.time_index, doc="Export sign: power_thermal = -Q.")
        def power_thermal_sign(b, t):
            return power[t] == -b.heat_output[t]

        self.add_constant_intensity_relation(
            self.flow_out,
            kind=nm.PowerKind.ELECTRICAL,
            intensity=self.config.aux_energy_intensity,
        )
