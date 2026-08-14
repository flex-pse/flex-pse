r"""Combustor(OpsBlockData): N gas inlets mixed into one flue-gas outlet (§3.2, §3.4).

Every unit model shipped so far moves water: each is built on a
single-``property_package`` IO-topology base (``SISOBlock``/``SIDOBlock``/
``DIDOBlock``) that hardcodes the liquid phase. A combustor takes an
**arbitrary number** of fuel-gas inlets -- natural gas, digester gas,
supplemental gas streams -- burned into one flue-gas outlet, so no fixed-arity
topology base fits (its port
count is a config option, §3.4's "choosing a base class" question). It
subclasses :class:`~flexops.core.ops_block.OpsBlockData` directly instead,
hand-writing its ports and balances, the way
:class:`~flexops.unit_models.storage.battery.BatteryModel` does.

Two flow-to-power relations, selected automatically (never configured
directly) from whether every inlet was given a heating value:

* **Heating value** -- when every inlet in ``inlet_names`` has an entry in
  ``heating_values``:

  .. math::

      P_{elec}[t] = -\eta \sum_i \text{HV}_i \cdot \dot{V}_i[t]

* **Constant intensity** -- when no inlet has a heating value:

  .. math::

      P_{elec}[t] = -\text{energy\_intensity} \sum_i \dot{V}_i[t]

Both are dimensionally exact (kWh/m^3 * m^3/hr = kW, no fudge factor) and
**negative**: like a discharging :class:`BatteryModel`, a combustor *exports*
electrical power, so ``power_electrical[t]`` is upper-bounded at 0 and plant
aggregation (a plain sum) nets the export against load with no per-unit sign
flipping.

A partial ``heating_values`` mapping -- some but not all inlets named -- is
rejected rather than silently falling back to the constant-intensity relation,
and an option the resolved relation would ignore (``efficiency`` under
constant intensity, ``energy_intensity`` under heating value) is rejected too.

**Combustion air is not a modeled inlet.** Every configured inlet is a fuel-gas
stream; an IC unit entrains atmospheric air at roughly its design air-to-fuel
ratio, so the flue-gas volume is the fuel burned scaled up by the air that came
with it:

.. math::

    \dot{V}_{flue}[t] = (1 + \text{air\_to\_gas\_ratio})
                        \sum_i \dot{V}_i[t]

``air_to_fuel_ratio`` is an **IC-design property**, not something derived here:
estimate it or model it externally, then supply it (or regress it). Keeping it a
multiplier — rather than a fuel/air reciprocal — keeps the balance linear even
once a design mode or regression unfixes it. Real combustion also changes moles,
which this volumetric balance does not track.

.. note::
   Under the constant-intensity relation, ``energy_intensity`` is per unit
   **total inlet volume**.

.. note::
   ``efficiency * heating_value_i`` is a product of fixed scalar Vars: linear
   while both factors stay fixed, **NLP** once a design mode or regression
   unfixes one -- the same caveat
   :class:`~flexops.unit_models.storage.tank.Tank` documents for
   ``capacity * level``.
"""

import enum
from collections.abc import Mapping

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData

_HEATING_VALUE_UNITS = pyunits.kWh / pyunits.m**3


class CombustorPowerRelation(enum.StrEnum):
    """Which flow-to-power relation a :class:`Combustor` resolved to.

    Not a config option: resolved once, in ``build()``, from whether every
    configured inlet carries a heating value.
    """

    HEATING_VALUE = "heating_value"
    CONSTANT_INTENSITY = "constant_intensity"


def _inlet_names_domain(value) -> tuple[str, ...]:
    """ConfigValue domain: coerce to a tuple.

    Only coerces the type; emptiness/uniqueness/non-empty-string checks
    happen in :meth:`CombustorData._validate_inlet_names` instead, so they
    raise :class:`FlexConfigError` directly rather than the ``ValueError``
    Pyomo's ``ConfigValue`` wraps every domain-raised exception into.
    """
    return tuple(value)


def _heating_values_domain(value):
    """ConfigValue domain: ``None``, or a mapping of inlet name to a quantity."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise FlexConfigError(
            "heating_values must be a mapping of inlet name to heating "
            f"value, got {type(value).__name__}.",
            field="heating_values",
            value=value,
        )
    return dict(value)


def _air_to_fuel_ratio_domain(value):
    """ConfigValue domain: the air-to-gas ratio must be a non-negative float."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    raise FlexConfigError(
        f"air_to_fuel_ratio must be a non-negative float, got {value!r}.",
        field="air_to_fuel_ratio",
        value=value,
    )


def _efficiency_domain(value):
    """ConfigValue domain: efficiency must be a fraction in (0, 1]."""
    if isinstance(value, (int, float)) and 0 < value <= 1:
        return float(value)
    raise FlexConfigError(
        f"efficiency must be a float in (0, 1], got {value!r}.",
        field="efficiency",
        value=value,
    )


@declare_process_block_class("Combustor")
class CombustorData(OpsBlockData):
    r"""N fuel-gas inlets burned into one flue-gas outlet, exporting power.

    See the module docstring for both flow-to-power relations and the
    documented simplifications. ``inlet_names`` sets the inlet count and their
    port names (``f"inlet_{name}"``); ``heating_values`` (a mapping of inlet
    name to a heating value) selects between them.

    Config:
        ``property_package`` (inherited): a single-phase
        :class:`~flexops.properties.simple_gas.SimpleGasFlow`-shaped package
        shared by every port. ``inlet_names`` (default ``("fuel",)``):
        the inlets' role/port names. ``heating_values`` (default ``None``):
        mapping of inlet name to its lower heating value per unit volume;
        supplying one for every inlet selects the heating-value relation,
        ``None``/empty selects constant intensity, anything in between raises.
        ``efficiency`` (default 0.35): electrical conversion efficiency,
        heating-value relation only. ``energy_intensity`` (default 2.0
        kWh/m^3): electrical output per unit total inlet volume, constant-
        intensity relation only. ``air_to_fuel_ratio`` (default 9.5): volumes of
        combustion air entrained per unit volume of fuel gas, which sets the
        flue-gas volume. ``flue_gas_temperature`` (default 750 K): the outlet
        temperature.

    Example:
        >>> from pyomo.environ import units as pyunits
        >>> from flexops.testing import dummy_gas_time_block
        >>> from flexops.unit_models import Combustor
        >>> m = dummy_gas_time_block(3)
        >>> m.chp = Combustor(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     inlet_names=("digester_gas", "natural_gas"),
        ...     heating_values={
        ...         "digester_gas": 6.0 * pyunits.kWh / pyunits.m**3,
        ...         "natural_gas": 10.5 * pyunits.kWh / pyunits.m**3,
        ...     },
        ...     efficiency=0.35,
        ... )
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "inlet_names",
        ConfigValue(
            default=("fuel",),
            domain=_inlet_names_domain,
            description="Role names of the combustor's gas inlets; inlet i is "
            "built as port f'inlet_{name}'. Must be unique and non-empty.",
        ),
    )
    CONFIG.declare(
        "heating_values",
        ConfigValue(
            default=None,
            domain=_heating_values_domain,
            description="Mapping of inlet name to its lower heating value per "
            "unit volume (a fixed, regressable Var per inlet once built, "
            "kWh/m^3). A value for every inlet in inlet_names selects the "
            "heating-value power relation; None or empty selects the "
            "constant-intensity relation; a partial mapping is rejected.",
        ),
    )
    CONFIG.declare(
        "efficiency",
        ConfigValue(
            default=0.35,
            domain=_efficiency_domain,
            description="Electrical conversion efficiency, a dimensionless "
            "fraction in (0, 1] (a fixed, regressable Var once built). Used "
            "only under the heating-value power relation.",
        ),
    )
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=2.0 * pyunits.kWh / pyunits.m**3,
            description="Electrical output per unit total inlet volume (a "
            "fixed, regressable Var once built), kWh/m^3. Used only under "
            "the constant-intensity power relation.",
        ),
    )
    CONFIG.declare(
        "air_to_fuel_ratio",
        ConfigValue(
            default=9.5,
            domain=_air_to_fuel_ratio_domain,
            description="Volumes of combustion air entrained per unit volume of "
            "fuel gas (a fixed, regressable Var once built), dimensionless. "
            "Sets the flue-gas volume, since combustion air is not a modeled "
            "inlet. An IC-design property: estimate it or model it externally "
            "rather than deriving it here. The default ~9.5 is roughly "
            "stoichiometric for natural gas.",
        ),
    )
    CONFIG.declare(
        "flue_gas_temperature",
        ConfigValue(
            default=750 * pyunits.K,
            description="Flue-gas outlet temperature (a fixed, regressable "
            "Var once built), K.",
        ),
    )

    def build(self) -> None:
        """Validate the config, build ports/balances, then the power relation."""
        super().build()
        self._validate_inlet_names()
        self._power_relation = self._resolve_power_relation()
        self.add_stream_ports(
            inlet_ports=self._inlet_port_names(), outlet_ports=("outlet",)
        )
        self._register_stream_states()
        self._build_mass_balance()
        self._build_outlet_state()
        self._build_power_relation()

    # -- config resolution --------------------------------------------------

    def _inlet_port_names(self) -> tuple[str, ...]:
        """Return the ``f"inlet_{name}"`` port names, in ``inlet_names`` order."""
        return tuple(f"inlet_{name}" for name in self.config.inlet_names)

    def _reference_inlet_name(self) -> str:
        """Return the first configured inlet name -- the mixed-stream reference."""
        return self.config.inlet_names[0]

    def _validate_inlet_names(self) -> None:
        """Reject empty, non-string, or duplicate ``inlet_names``.

        Raises:
            FlexConfigError: If ``inlet_names`` is empty, contains a
                non-string or empty-string entry, or contains a duplicate.
        """
        names = self.config.inlet_names
        if not names or not all(isinstance(n, str) and n for n in names):
            raise FlexConfigError(
                f"inlet_names must be one or more non-empty strings, got "
                f"{names!r}.",
                field="inlet_names",
                value=names,
            )
        if len(set(names)) != len(names):
            raise FlexConfigError(
                f"inlet_names must be unique, got {names!r}.",
                field="inlet_names",
                value=names,
            )

    def _flow_phase(self) -> str:
        """Return the property package's one phase.

        Raises:
            FlexConfigError: If the package does not have exactly one phase.
        """
        phases = list(self.config.property_package.phase_list)
        if len(phases) != 1:
            raise FlexConfigError(
                "Combustor requires a property_package with exactly one "
                f"phase (a single-phase gas basis); got phase_list={phases!r}.",
                field="property_package",
                value=self.config.property_package,
            )
        return phases[0]

    def _resolve_power_relation(self) -> "CombustorPowerRelation":
        """Derive the power relation from ``heating_values`` and gate options.

        Returns:
            The resolved :class:`CombustorPowerRelation`.

        Raises:
            FlexConfigError: If ``heating_values`` names some but not all
                inlets (or an inlet not in ``inlet_names``), or if an option
                the resolved relation ignores was explicitly set.
        """
        heating_values = self.config.heating_values or {}
        inlet_names = set(self.config.inlet_names)
        hv_names = set(heating_values)
        user_set = {v.name() for v in self.config.user_values()}

        if not hv_names:
            relation = CombustorPowerRelation.CONSTANT_INTENSITY
        elif hv_names == inlet_names:
            relation = CombustorPowerRelation.HEATING_VALUE
        else:
            missing = sorted(inlet_names - hv_names)
            unknown = sorted(hv_names - inlet_names)
            detail = ", ".join(
                part
                for part in (
                    f"missing a heating value for {missing}" if missing else "",
                    f"names unknown inlet(s) {unknown}" if unknown else "",
                )
                if part
            )
            raise FlexConfigError(
                "heating_values must supply a heating value for every inlet "
                f"in inlet_names or none at all ({detail}); supply one for "
                "every inlet to select the heating-value relation, or drop "
                "heating_values entirely to fall back to constant_intensity.",
                field="heating_values",
                value=self.config.heating_values,
            )

        if relation is CombustorPowerRelation.CONSTANT_INTENSITY:
            if "efficiency" in user_set:
                raise FlexConfigError(
                    "efficiency has no effect under the constant_intensity "
                    "power relation (selected because heating_values was not "
                    "given for every inlet); remove efficiency, or supply "
                    "heating_values for every inlet in inlet_names.",
                    field="efficiency",
                    value=self.config.efficiency,
                )
        elif "energy_intensity" in user_set:
            raise FlexConfigError(
                "energy_intensity has no effect under the heating_value "
                "power relation (selected because every inlet has a heating "
                "value); remove energy_intensity, or drop a heating_values "
                "entry to fall back to constant_intensity.",
                field="energy_intensity",
                value=self.config.energy_intensity,
            )
        return relation

    # -- ports, mass balance, outlet state -----------------------------------

    def _register_stream_states(self) -> None:
        """Register flow (every inlet) and the reference inlet's other states.

        ``add_stream_ports`` registers only ``flow_vol_phase``.
        ``SimpleGasFlow`` carries three more always-on state variables, and
        registering the *reference* inlet's -- not every inlet's -- keeps the
        model well-posed: the mixing equalities below already pin every other
        inlet's intensive state to the reference's.
        """
        flow_name = self.config.property_package.get_flow_basis_var_name()
        ref_state = self.find_component(f"inlet_{self._reference_inlet_name()}_state")
        for name, var in ref_state.define_state_vars().items():
            if name == flow_name:
                continue
            self.register_io_variable(var, role="input")
        for name, var in self.outlet_state.define_state_vars().items():
            if name == flow_name:
                continue
            self.register_io_variable(var, role="output")

    def _build_mass_balance(self) -> None:
        """Build per-inlet flow References, the mixing balance, and state ties."""
        tb = self._find_time_block()
        phase = self._flow_phase()
        flow_name = self.config.property_package.get_flow_basis_var_name()
        inlet_names = self.config.inlet_names
        ref_name = self._reference_inlet_name()

        flows = {}
        for name in inlet_names:
            state = self.find_component(f"inlet_{name}_state")
            self.add_component(
                f"flow_in_{name}", pyo.Reference(state.flow_vol_phase[:, phase])
            )
            flows[name] = getattr(self, f"flow_in_{name}")
        self.add_component(
            "flow_out", pyo.Reference(self.outlet_state.flow_vol_phase[:, phase])
        )
        flow_out = self.flow_out

        air_to_fuel_ratio = self.declare_process_parameter(
            "air_to_fuel_ratio",
            self.config.air_to_fuel_ratio,
            pyunits.dimensionless,
            "Volumes of combustion air entrained per unit volume of fuel gas.",
            bounds=(0.0, None),
        )

        @self.Constraint(
            tb.time_index,
            doc="Flue gas: outlet flow == (1 + air_to_fuel_ratio) * total fuel "
            "flow — the fuel burned plus the combustion air it entrains.",
        )
        def mixing_mass_balance(b, t):
            return flow_out[t] == (1 + air_to_fuel_ratio) * sum(
                flows[name][t] for name in inlet_names
            )

        other_names = inlet_names[1:]
        if not other_names:
            return
        ref_state = self.find_component(f"inlet_{ref_name}_state")
        state_vars = [v for v in ref_state.define_state_vars() if v != flow_name]
        for state_var in state_vars:

            def _equality_rule(b, t, name, _v=state_var, _ref=ref_name):
                other_state = b.find_component(f"inlet_{name}_state")
                ref_state_ = b.find_component(f"inlet_{_ref}_state")
                return getattr(other_state, _v)[t] == getattr(ref_state_, _v)[t]

            self.add_component(
                f"inlet_state_equality_{state_var}",
                pyo.Constraint(
                    tb.time_index,
                    other_names,
                    rule=_equality_rule,
                    doc=f"Mixing node: inlet {state_var} equals the reference "
                    f"inlet's {state_var}.",
                ),
            )

    def _build_outlet_state(self) -> None:
        """Pass pressure through from the reference inlet; fix the temperature."""
        tb = self._find_time_block()
        flow_name = self.config.property_package.get_flow_basis_var_name()
        ref_name = self._reference_inlet_name()
        ref_port = self.find_component(f"inlet_{ref_name}")

        self.add_pass_through_constraints(
            ref_port,
            self.outlet,
            exclude_vars=[flow_name, "temperature"],
        )

        flue_gas_temperature = self.declare_process_parameter(
            "flue_gas_temperature",
            self.config.flue_gas_temperature,
            pyunits.K,
            "Flue-gas outlet temperature.",
            bounds=(0.0, None),
        )

        @self.Constraint(
            tb.time_index, doc="Outlet temperature is fixed at flue_gas_temperature."
        )
        def outlet_temperature_eq(b, t):
            return b.outlet_state.temperature[t] == flue_gas_temperature

    # -- power relation -------------------------------------------------------

    def _build_power_relation(self) -> None:
        """Declare the electrical export and its resolved flow-to-power relation."""
        tb = self._find_time_block()
        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.register_io_variable(power, role="output")
        for t in tb.time_index:
            power[t].setub(0.0)

        inlet_names = self.config.inlet_names
        flows = {name: getattr(self, f"flow_in_{name}") for name in inlet_names}

        if self._power_relation is CombustorPowerRelation.HEATING_VALUE:
            heating_values = {}
            for name in inlet_names:
                heating_values[name] = self.declare_process_parameter(
                    f"heating_value_{name}",
                    self.config.heating_values[name],
                    _HEATING_VALUE_UNITS,
                    f"Lower heating value of inlet '{name}' per unit volume.",
                    bounds=(0.0, None),
                )
            efficiency = self.declare_process_parameter(
                "efficiency",
                self.config.efficiency,
                pyunits.dimensionless,
                "Electrical conversion efficiency.",
                bounds=(0.0, 1.0),
            )

            @self.Constraint(
                tb.time_index,
                doc="power_electrical == -efficiency * sum(heating_value_i * "
                "flow_in_i); an export (kW).",
            )
            def power_electrical_relation(b, t):
                return -power[t] == pyunits.convert(
                    efficiency
                    * sum(
                        heating_values[name] * flows[name][t] for name in inlet_names
                    ),
                    pyunits.kW,
                )

        else:
            energy_intensity = self.declare_process_parameter(
                "energy_intensity",
                self.config.energy_intensity,
                _HEATING_VALUE_UNITS,
                "Electrical output per unit total inlet volume.",
            )

            @self.Constraint(
                tb.time_index,
                doc="power_electrical == -energy_intensity * total inlet flow; "
                "an export (kW).",
            )
            def power_electrical_relation(b, t):
                return -power[t] == pyunits.convert(
                    energy_intensity * sum(flows[name][t] for name in inlet_names),
                    pyunits.kW,
                )

        self.register_relation(self.power_electrical_relation, target=power)
        surrogate = getattr(self.config.flexops_config, "surrogate", None)
        if surrogate is not None and surrogate.functional_form != "constant_intensity":
            self.swap_relation("power_electrical_relation", surrogate)
