r"""Combustor(OpsBlockData): N fuel sources burned into one flue-gas outlet.

Every unit model shipped so far moves water: each is built on a
single-``property_package`` IO-topology base (``SISOBlock``/``SIDOBlock``/
``DIDOBlock``) that hardcodes the liquid phase. A combustor takes an
**arbitrary number** of fuel sources -- connected via inlet ports, pulled
from utilities, or both -- burned into one flue-gas outlet, so no fixed-arity
topology base fits (its port count is a config option, and picking a base
class asks whether every port shares one property_package -- here they
don't). It subclasses :class:`~flexops.core.ops_block.OpsBlockData` directly instead,
hand-writing its ports and balances, the way
:class:`~flexops.unit_models.storage.battery.BatteryModel` does.

Two flow-to-power relations, selected automatically (never configured
directly) from whether every fuel source was given a heating value. Both determine
``power_generated[t]``, the **non-negative** generation magnitude:

* **Heating value** -- when every fuel source (inlet and utility) has an entry in
  ``heating_values``:

  .. math::

      P_{gen}[t] = \eta \sum_i \text{HV}_i \cdot \dot{V}_i[t]

* **Constant intensity** -- when no fuel source has a heating value:

  .. math::

      P_{gen}[t] = \text{energy\_intensity} \sum_i \dot{V}_i[t]

Both are dimensionally exact (kWh/m^3 * m^3/hr = kW, no fudge factor). The
sign is a separate constraint,

.. math::

    P_{elec}[t] = -P_{gen}[t]

so that like a discharging :class:`BatteryModel`, a combustor *exports*
electrical power: ``power_electrical[t]`` is upper-bounded at 0 and plant
aggregation (a plain sum) nets the export against load with no per-unit sign
flipping.

Splitting the magnitude from the sign is what makes that convention survive a
surrogate swap. Only ``power_electrical_relation`` is registered, so only it
can be swapped (see
:meth:`~flexops.core.ops_block.OpsBlockData.register_relation`); the sign
constraint cannot be. Its target ``power_generated`` carries the lower bound
of 0, so a fitted relationship that predicts a negative export violates a
bound naming the quantity it got wrong, rather than silently conflicting with
``power_electrical``'s upper bound. Surrogates for this unit are therefore
fitted in generation-magnitude space -- positive, the convention generation
data is normally logged in -- which is also why ``power_generated`` (not
``power_electrical``) is the unit's registered IO output.

A partial ``heating_values`` mapping -- some but not all fuel sources named -- is
rejected rather than silently falling back to the constant-intensity relation,
and an option the resolved relation would ignore (``efficiency`` under
constant intensity, ``energy_intensity`` under heating value) is rejected too.

**Combustion air is not a modeled inlet.** Every configured fuel source is a
fuel-gas stream; an IC unit entrains atmospheric air at roughly its design air-to-fuel
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
   **total fuel volume** (inlet + utility).

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
from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.surrogates import surrogate_from_spec
from flexops.unit_models.powergeneration.utils import (
    heating_value_mismatch,
    heating_values_domain,
    inlet_names_domain,
    utility_fuel_source_domain,
    validate_fuel_sources,
)

_HEATING_VALUE_UNITS = pyunits.kWh / pyunits.m**3


class CombustorPowerRelation(enum.StrEnum):
    """Which flow-to-power relation a :class:`Combustor` resolved to.

    Not a config option: resolved once, in ``build()``, from whether every
    configured inlet carries a heating value.
    """

    HEATING_VALUE = "heating_value"
    CONSTANT_INTENSITY = "constant_intensity"


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


def _blend_ratio_domain(value):
    """ConfigValue domain: None, a single ratio dict, or a list of ratio dicts.

    Each dict must be of the form ``{fuel_a: fuel_b, ratio: r}``.
    """
    if value is None:
        return None
    if isinstance(value, Mapping):
        return (_validate_blend_ratio_mapping(value),)
    if isinstance(value, (list, tuple)):
        if not value:
            raise FlexConfigError(
                "blend_ratio must contain at least one ratio mapping; "
                "got an empty list.",
                field="blend_ratio",
                value=value,
            )
        result = []
        seen_pairs = set()
        for idx, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise FlexConfigError(
                    "blend_ratio entries must be mappings; "
                    f"entry {idx} is {type(item).__name__}.",
                    field="blend_ratio",
                    value=value,
                )
            validated = _validate_blend_ratio_mapping(item)
            fuel_a = next(k for k in item if k != "ratio")
            fuel_b = item[fuel_a]
            pair = (fuel_a, fuel_b)
            if pair in seen_pairs:
                raise FlexConfigError(
                    "blend_ratio contains duplicate pair "
                    f"'{fuel_a}' -> '{fuel_b}'; each pair must be "
                    "unique.",
                    field="blend_ratio",
                    value=item,
                )
            seen_pairs.add(pair)
            result.append(validated)
        return tuple(result)
    raise FlexConfigError(
        "blend_ratio must be None, a mapping, or a list/tuple of mappings; "
        f"got {type(value).__name__}.",
        field="blend_ratio",
        value=value,
    )


def _validate_blend_ratio_mapping(item):
    """Validate one blend_ratio mapping and return it as a plain dict."""
    keys = set(item.keys())
    if "ratio" not in keys:
        raise FlexConfigError(
            "Each blend_ratio mapping must contain a 'ratio' key.",
            field="blend_ratio",
            value=item,
        )
    non_ratio = keys - {"ratio"}
    if len(non_ratio) != 1:
        raise FlexConfigError(
            "Each blend_ratio mapping must contain exactly one fuel pair "
            "(one key mapping to the other fuel); got "
            f"{len(non_ratio)} non-ratio entries: {sorted(non_ratio)}.",
            field="blend_ratio",
            value=item,
        )
    fuel_a = next(iter(non_ratio))
    fuel_b = item[fuel_a]
    if not isinstance(fuel_b, str) or not fuel_b:
        raise FlexConfigError(
            f"blend_ratio value for '{fuel_a}' must be a non-empty fuel name.",
            field="blend_ratio",
            value=item,
        )
    ratio = item["ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio <= 0:
        raise FlexConfigError(
            "blend_ratio 'ratio' must be a positive number; " f"got {ratio!r}.",
            field="blend_ratio",
            value=item,
        )
    return dict(item)


@declare_process_block_class("Combustor")
class CombustorData(OpsBlockData):
    r"""N fuel sources burned into one flue-gas outlet, exporting power.

    See the module docstring for both flow-to-power relations and the
    documented simplifications. ``inlet_names`` sets the inlet port count and their
    port names (``f"inlet_{name}"``); ``utility_fuel_source`` names fuels pulled
    from utilities rather than connected via inlet ports. ``heating_values`` maps
    every fuel name -- inlet or utility -- to a heating value and selects the
    relation.

    Config:
        ``property_package`` (inherited): a single-phase
        :class:`~flexops.properties.simple_gas.SimpleGasFlow`-shaped package
        shared by every port. ``inlet_names`` (default ``None``): the inlets'
        role/port names. Pass ``None`` or ``()`` for no inlets (requires
        ``utility_fuel_source``); pass a non-empty tuple of unique strings
        to enable inlet-connected combustion. ``heating_values`` (default
        ``None``): mapping of fuel name to its lower heating value per unit
        volume; supplying one for every fuel source (inlet and utility) selects the
        heating-value relation, ``None``/empty selects constant intensity,
        anything in between raises. ``efficiency`` (default 0.35): electrical
        conversion efficiency, heating-value relation only.
        ``energy_intensity`` (default 2.0 kWh/m^3): electrical output per
        unit total fuel volume, constant-intensity relation only.
        ``air_to_fuel_ratio`` (default 9.5): volumes of combustion air
        entrained per unit volume of fuel gas, which sets the flue-gas
        volume. ``flue_gas_temperature`` (default 750 K): the outlet
        temperature. ``utility_fuel_source`` (default ``None``): fuel names
        pulled from utilities rather than connected via inlet ports; accepts
        a single name or a list of names. These names are independent of
        ``inlet_names`` and must not overlap with them. Their heating values
        are declared in the same ``heating_values`` mapping. ``blend_ratio``
        (default ``None``): optional dict or list of dicts of the form
        ``{fuel_a: fuel_b, ratio: r}`` enforcing
        ``flow_fuel_a[t] == flow_fuel_b[t] * r``; both fuel names must be
        known inlets or utility fuel sources, and ``r`` must be positive.
        Pass a list to define multiple independent ratio constraints among
        the same fuels.

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
            default=None,
            domain=inlet_names_domain,
            description="Role names of the combustor's gas inlets; inlet i is "
            "built as port f'inlet_{name}'. Must be unique and non-empty when "
            "given. Defaults to None, meaning no inlet ports are built; "
            "provide at least one name to enable inlet-connected combustion.",
        ),
    )
    CONFIG.declare(
        "heating_values",
        ConfigValue(
            default=None,
            domain=heating_values_domain,
            description="Mapping of fuel name to its lower heating value per "
            "unit volume (a fixed, regressable Var per fuel once built, "
            "kWh/m^3). A value for every fuel source -- both inlets in "
            "``inlet_names`` and names in ``utility_fuel_source`` -- selects "
            "the heating-value power relation; None or empty selects the "
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
            description="Electrical output per unit total fuel volume (a "
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
    CONFIG.declare(
        "utility_fuel_source",
        ConfigValue(
            default=None,
            domain=utility_fuel_source_domain,
            description="Fuel names pulled from utilities rather than connected "
            "via inlet ports. Accepts a single name or a list of names; these "
            "names are independent of ``inlet_names`` and must not overlap with "
            "it. Each utility fuel gets its own ``utility_flow_{name}`` Var "
            "(m^3/hr) registered via ``register_fuel_usage`` for costing. "
            "If None, no utility fuel is used.",
        ),
    )
    CONFIG.declare(
        "blend_ratio",
        ConfigValue(
            default=None,
            domain=_blend_ratio_domain,
            description="Optional fixed volume ratio between two fuel sources. "
            "Accepts a single dict or a list of dicts, each of the form "
            "{fuel_a: fuel_b, ratio: r}, which enforces "
            "flow_fuel_a[t] == flow_fuel_b[t] * r. Both fuel names must be "
            "known inlets or utility fuel sources, and r must be positive. "
            "The ratio is a fixed, regressable process parameter. Pass a "
            "list to define multiple independent ratio constraints.",
        ),
    )

    def build(self) -> None:
        """Validate the config, build ports/balances, then the power relation."""
        super().build()
        self._validate_inlet_names()
        self._validate_fuel_config()
        self._power_relation = self._resolve_power_relation()
        self.add_stream_ports(
            inlet_ports=self._inlet_port_names(), outlet_ports=("outlet",)
        )
        self._register_stream_states()
        self._build_fuel_usage()
        self._build_mass_balance()
        self._build_blend_ratio()
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
        """Validate ``inlet_names``.

        ``None`` is allowed and means "no inlet ports"; it requires
        ``utility_fuel_source`` to be set. An explicit empty tuple has the
        same meaning and requirement.
        """
        if self.config.inlet_names is None:
            self.config._data["inlet_names"].set_value(())
        validate_fuel_sources(
            self.config.inlet_names,
            self.config.utility_fuel_source,
            inlet_field="inlet_names",
            require_any=True,
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
                fuel sources (inlets and utility fuel sources), or if an
                option the resolved relation ignores was explicitly set.
        """
        all_fuel_names = set(self.config.inlet_names) | set(
            self.config.utility_fuel_source or []
        )
        missing, unknown = heating_value_mismatch(
            self.config.heating_values, all_fuel_names
        )
        user_set = {v.name() for v in self.config.user_values()}

        if not self.config.heating_values:
            relation = CombustorPowerRelation.CONSTANT_INTENSITY
        elif not missing and not unknown:
            relation = CombustorPowerRelation.HEATING_VALUE
        else:
            detail = ", ".join(
                part
                for part in (
                    f"missing a heating value for {missing}" if missing else "",
                    f"names unknown fuel source(s) {unknown}" if unknown else "",
                )
                if part
            )
            raise FlexConfigError(
                "heating_values must supply a heating value for every fuel "
                f"source (inlet_names and utility_fuel_source) or none at "
                f"all ({detail}); supply one for every fuel source to "
                "select the heating-value relation, or drop heating_values "
                "entirely to fall back to constant_intensity.",
                field="heating_values",
                value=self.config.heating_values,
            )

        if relation is CombustorPowerRelation.CONSTANT_INTENSITY:
            if "efficiency" in user_set:
                raise FlexConfigError(
                    "efficiency has no effect under the constant_intensity "
                    "power relation (selected because heating_values was not "
                    "given for every fuel source); remove efficiency, or supply "
                    "heating_values for every fuel source (inlet_names and "
                    "utility_fuel_source).",
                    field="efficiency",
                    value=self.config.efficiency,
                )
        elif "energy_intensity" in user_set:
            raise FlexConfigError(
                "energy_intensity has no effect under the heating_value "
                "power relation (selected because every fuel source has a "
                "heating value); remove energy_intensity, or drop a "
                "heating_values entry to fall back to constant_intensity.",
                field="energy_intensity",
                value=self.config.energy_intensity,
            )
        return relation

    # -- fuel usage and blend ratio -------------------------------------------

    def _validate_fuel_config(self) -> None:
        """Validate blend_ratio against the known fuel sources."""
        utility_fuels = self.config.utility_fuel_source
        blend_ratio = self.config.blend_ratio

        if blend_ratio is not None:
            all_names = set(self.config.inlet_names) | set(utility_fuels or [])
            for item in blend_ratio:
                fuel_a = next(k for k in item if k != "ratio")
                fuel_b = item[fuel_a]
                if fuel_a not in all_names:
                    raise FlexConfigError(
                        f"blend_ratio fuel '{fuel_a}' is not a known inlet "
                        "or utility fuel source.",
                        field="blend_ratio",
                        value=item,
                    )
                if fuel_b not in all_names:
                    raise FlexConfigError(
                        f"blend_ratio fuel '{fuel_b}' is not a known inlet "
                        "or utility fuel source.",
                        field="blend_ratio",
                        value=item,
                    )

    def _build_fuel_usage(self) -> None:
        """Register fuel usage for utility fuel sources.

        Each utility fuel source gets its own time-indexed volumetric flow
        ``Var`` on the combustor block and is registered via
        :meth:`register_fuel_usage` for costing.
        """
        utility_fuels = self.config.utility_fuel_source
        if utility_fuels is None:
            return

        tb = self._find_time_block()
        for name in utility_fuels:
            flow_var = pyo.Var(
                tb.time_index,
                units=pyunits.m**3 / pyunits.hr,
                bounds=(0.0, None),
                doc=f"Volumetric flow of utility fuel '{name}' burned by "
                f"this combustor (m^3/hr).",
            )
            self.add_component(f"utility_flow_{name}", flow_var)
            self.register_fuel_usage(flow_var, fuel_name=name)

    def _build_blend_ratio(self) -> None:
        """Optionally enforce fixed volume ratios between fuel sources.

        ``blend_ratio`` may be a single mapping or a list/tuple of mappings,
        each of the form ``{fuel_a: fuel_b, ratio: r}``. For every mapping
        a fixed process parameter and a time-indexed equality constraint
        are built.
        """
        blend_ratio = self.config.blend_ratio
        if blend_ratio is None:
            return

        tb = self._find_time_block()
        inlet_names = set(self.config.inlet_names)

        def _flow_var(name):
            if name in inlet_names:
                return getattr(self, f"flow_in_{name}")
            return getattr(self, f"utility_flow_{name}")

        for item in blend_ratio:
            fuel_a = next(k for k in item if k != "ratio")
            fuel_b = item[fuel_a]
            ratio_value = item["ratio"]

            flow_a = _flow_var(fuel_a)
            flow_b = _flow_var(fuel_b)

            ratio_var = self.declare_process_parameter(
                f"blend_ratio_{fuel_a}_{fuel_b}",
                ratio_value,
                pyunits.dimensionless,
                f"Volume ratio of {fuel_a} to {fuel_b} in the fuel blend.",
                bounds=(0.0, None),
            )

            def _blend_rule(b, t, _a=flow_a, _b=flow_b, _r=ratio_var):
                return _a[t] == _b[t] * _r

            self.add_component(
                f"blend_ratio_{fuel_a}_{fuel_b}_constraint",
                pyo.Constraint(
                    tb.time_index,
                    rule=_blend_rule,
                    doc=f"Blend ratio: {fuel_a} flow == {fuel_b} flow * {ratio_value}.",
                ),
            )

    # -- ports, mass balance, outlet state -----------------------------------

    def _register_stream_states(self) -> None:
        """Register flow (every inlet) and the reference inlet's other states.

        ``add_stream_ports`` registers only ``flow_vol_phase``.
        ``SimpleGasFlow`` carries three more always-on state variables, and
        registering the *reference* inlet's -- not every inlet's -- keeps the
        model well-posed: the mixing equalities below already pin every other
        inlet's intensive state to the reference's. When there are no inlets
        (utility-only mode), no inlet states are registered.
        """
        flow_name = self.config.property_package.get_flow_basis_var_name()
        if self.config.inlet_names:
            ref_state = self.find_component(
                f"inlet_{self._reference_inlet_name()}_state"
            )
            for name, var in ref_state.define_state_vars().items():
                if name == flow_name:
                    continue
                self.register_io_variable(var, role="input")
        for name, var in self.outlet_state.define_state_vars().items():
            if name == flow_name:
                continue
            self.register_io_variable(var, role="output")

    def _build_mass_balance(self) -> None:
        """Build per-inlet flow References, the mixing balance."""
        tb = self._find_time_block()
        phase = self._flow_phase()
        inlet_names = self.config.inlet_names

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
            return flow_out[t] == (1 + air_to_fuel_ratio) * (
                sum(flows[name][t] for name in inlet_names)
                + sum(
                    getattr(b, f"utility_flow_{name}")[t]
                    for name in b.config.utility_fuel_source or []
                )
            )

    def _build_outlet_state(self) -> None:
        """Pass pressure through from the reference inlet; fix the temperature."""
        tb = self._find_time_block()
        flow_name = self.config.property_package.get_flow_basis_var_name()
        if self.config.inlet_names:
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
        """Declare the export, its generation magnitude, and their relation."""
        tb = self._find_time_block()
        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        for t in tb.time_index:
            power[t].setub(0.0)

        inlet_names = self.config.inlet_names
        utility_names = self.config.utility_fuel_source or ()
        flows = {name: getattr(self, f"flow_in_{name}") for name in inlet_names}
        utility_flows = {
            name: getattr(self, f"utility_flow_{name}") for name in utility_names
        }

        self.power_generated = pyo.Var(
            tb.time_index,
            bounds=(0.0, None),
            initialize=0.0,
            units=pyunits.kW,
            doc="Gross electrical generation magnitude (kW, non-negative). The "
            "relation's target: its lower bound is what keeps a swapped-in "
            "surrogate from predicting a negative export.",
        )
        self.register_io_variable(self.power_generated, role="output")

        @self.Constraint(
            tb.time_index,
            doc="Sign convention: an export is a negative draw, "
            "power_electrical == -power_generated. Deliberately not registered, "
            "so no surrogate swap can replace it.",
        )
        def power_electrical_sign(b, t):
            return power[t] == -b.power_generated[t]

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
            for name in utility_names:
                heating_values[name] = self.declare_process_parameter(
                    f"heating_value_{name}",
                    self.config.heating_values[name],
                    _HEATING_VALUE_UNITS,
                    f"Lower heating value of utility fuel '{name}' per unit " "volume.",
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
                doc="power_generated == efficiency * sum(heating_value_i * "
                "flow_in_i); the export magnitude (kW).",
            )
            def power_electrical_relation(b, t):
                total = 0.0 * pyunits.kWh / pyunits.hr
                for name in inlet_names:
                    total += heating_values[name] * flows[name][t]
                for name in utility_names:
                    total += heating_values[name] * utility_flows[name][t]
                return b.power_generated[t] == pyunits.convert(
                    efficiency * total,
                    pyunits.kW,
                )

        else:
            energy_intensity = self.declare_process_parameter(
                "energy_intensity",
                self.config.energy_intensity,
                _HEATING_VALUE_UNITS,
                "Electrical output per unit total inlet volume.",
                bounds=(0.0, None),
            )

            @self.Constraint(
                tb.time_index,
                doc="power_generated == energy_intensity * total inlet flow; "
                "the export magnitude (kW).",
            )
            def power_electrical_relation(b, t):
                total = 0.0 * pyunits.m**3 / pyunits.hr
                for name in inlet_names:
                    total += flows[name][t]
                for name in utility_names:
                    total += utility_flows[name][t]
                return b.power_generated[t] == pyunits.convert(
                    energy_intensity * total,
                    pyunits.kW,
                )

        self.register_relation(
            self.power_electrical_relation, target=self.power_generated
        )
        spec = getattr(self.config.flexops_config, "surrogate", None)
        if spec is not None and (
            spec.surrogate_type is not SurrogateType.CONSTANT_INTENSITY
        ):
            self.swap_relation("power_electrical_relation", surrogate_from_spec(spec))
