"""Electrolyzer unit models: a stack plus its gas-liquid separators."""

import dataclasses
import enum

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.surrogates import surrogate_from_spec
from flexops.unit_models._multiport import single_flow_phase

FARADAY = 96485.33212 * pyunits.C / pyunits.mol
WATER_MOLAR_MASS = 18.015 * pyunits.g / pyunits.mol


class ProductPhase(enum.StrEnum):
    """The phase a product leaves the separators in."""

    GAS = "gas"
    LIQUID = "liquid"


class ElectrolyzerTechnology(enum.StrEnum):
    """Water-electrolysis technologies with built-in defaults."""

    PEM = "pem"
    AEM = "aem"


@dataclasses.dataclass(frozen=True)
class ElectrochemicalProduct:
    """One cathode product and its overall-reaction stoichiometry.

    Attributes:
        name: Product name; a Python identifier, used in component names.
        electrons: Electrons transferred per mole of product.
        faradaic_efficiency: Fraction of the stack charge producing this product.
        phase: Phase the product leaves in (gas or liquid outlet).
        co2_per_mol: Moles of CO2 consumed per mole of product.
        water_per_mol: Moles of water consumed per mole of product.
        molar_mass: Molar mass with units (used for liquid products).
    """

    name: str
    electrons: int
    faradaic_efficiency: float
    phase: ProductPhase
    co2_per_mol: float
    water_per_mol: float
    molar_mass: object

    def __post_init__(self) -> None:
        """Coerce ``phase`` from its string value."""
        object.__setattr__(self, "phase", ProductPhase(self.phase))


HYDROGEN = ElectrochemicalProduct(
    name="H2",
    electrons=2,
    faradaic_efficiency=1.0,
    phase=ProductPhase.GAS,
    co2_per_mol=0.0,
    water_per_mol=1.0,
    molar_mass=2.016 * pyunits.g / pyunits.mol,
)


def _products_domain(value) -> tuple[ElectrochemicalProduct, ...]:
    """Coerce a list of products or product dicts into products.

    Args:
        value: Products, or dicts of their fields.

    Returns:
        The products as a tuple.
    """
    return tuple(
        p if isinstance(p, ElectrochemicalProduct) else ElectrochemicalProduct(**p)
        for p in value
    )


def _magnitude(value, units) -> float:
    """Return ``value`` as a float in ``units``.

    Args:
        value: A units-carrying value, or a bare number already in ``units``.
        units: The units to convert to.

    Returns:
        The magnitude in ``units``.
    """
    if isinstance(value, (int, float)):
        return float(value)
    return float(pyo.value(pyunits.convert(value, units)))


@declare_process_block_class("Electrolyzer")
class ElectrolyzerData(OpsBlockData):
    """An electrolyzer stack and its gas-liquid separators over a product table.

    Config:
        Inherits the OpsBlock config; each added option is documented on its
        ``CONFIG`` entry.
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "liquid_property_package",
        ConfigValue(
            default=None,
            description="Single-phase property package for the liquid make-up "
            "inlet and the liquid outlet.",
        ),
    )
    CONFIG.declare(
        "gas_property_package",
        ConfigValue(
            default=None,
            description="Single-phase property package for every gas port "
            "not given its own package below.",
        ),
    )
    CONFIG.declare(
        "cathode_gas_property_package",
        ConfigValue(
            default=None,
            description="Single-phase property package for the cathode gas "
            "outlet; defaults to gas_property_package.",
        ),
    )
    CONFIG.declare(
        "anode_gas_property_package",
        ConfigValue(
            default=None,
            description="Single-phase property package for the anode gas "
            "outlet; defaults to gas_property_package.",
        ),
    )
    CONFIG.declare(
        "n_cells",
        ConfigValue(default=100, domain=int, description="Cells in series."),
    )
    CONFIG.declare(
        "rated_current",
        ConfigValue(
            default=2000 * pyunits.A,
            description="Maximum stack current, the upper bound on current[t], A.",
        ),
    )
    CONFIG.declare(
        "cell_voltage",
        ConfigValue(
            default=1.9 * pyunits.V,
            description="Cell voltage (a fixed, regressable Var once built), V.",
        ),
    )
    CONFIG.declare(
        "thermoneutral_voltage",
        ConfigValue(
            default=1.48 * pyunits.V,
            description="Thermoneutral cell voltage; the voltage above it is "
            "released as stack heat (waste_heat), V.",
        ),
    )
    CONFIG.declare(
        "bop_fraction",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Balance-of-plant electrical draw as a fraction of stack "
            "power (a fixed, regressable Var once built).",
        ),
    )
    CONFIG.declare(
        "operating_temperature",
        ConfigValue(
            default=333.15 * pyunits.K,
            description="Separator temperature; the gas outlets leave at it, K.",
        ),
    )
    CONFIG.declare(
        "operating_pressure",
        ConfigValue(
            default=101325.0 * pyunits.Pa,
            description="Separator pressure; the gas outlets leave at it, Pa.",
        ),
    )
    CONFIG.declare(
        "products",
        ConfigValue(
            default=(HYDROGEN,),
            domain=_products_domain,
            description="Cathode products, as ElectrochemicalProduct entries (or "
            "dicts of their fields). Faradaic efficiencies may sum to at most 1.",
        ),
    )
    CONFIG.declare(
        "has_liquid_outlet",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether to build the liquid outlet, carrying the "
            "electrolyte bleed and any liquid products.",
        ),
    )
    CONFIG.declare(
        "bleed_fraction",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Liquid bled from the loop per unit volume of water "
            "consumed (a fixed, regressable Var once built). Requires "
            "has_liquid_outlet.",
        ),
    )
    CONFIG.declare(
        "liquid_density",
        ConfigValue(
            default=1000.0 * pyunits.kg / pyunits.m**3,
            description="Density converting consumed water and liquid products "
            "to volume, kg/m^3.",
        ),
    )
    CONFIG.declare(
        "separator_volume",
        ConfigValue(
            default=1.0 * pyunits.m**3,
            description="Liquid capacity of the separators, m^3.",
        ),
    )
    CONFIG.declare(
        "initial_liquid_volume",
        ConfigValue(
            default=0.5 * pyunits.m**3,
            description="Liquid inventory at the first time point, m^3 (a "
            "rolling-horizon initial state).",
        ),
    )
    CONFIG.declare(
        "liquid_level_min",
        ConfigValue(
            default=0.1,
            domain=float,
            description="Minimum fill fraction of separator_volume.",
        ),
    )
    CONFIG.declare(
        "liquid_level_max",
        ConfigValue(
            default=0.9,
            domain=float,
            description="Maximum fill fraction of separator_volume.",
        ),
    )

    def build(self) -> None:
        """Build ports, electrochemistry, gas outlets, and the liquid loop."""
        super().build()
        self._resolve_config()
        self._validate_config()
        self._gas_phases = {
            field: single_flow_phase(self._gas_package(field), type(self).__name__)
            for field in self._gas_package_fields()
        }
        self._liquid_phase = single_flow_phase(
            self.config.liquid_property_package, type(self).__name__
        )
        self._build_ports()
        self._build_electrochemistry()
        self._build_gas_outlets()
        self._build_liquid_loop()
        self._register_io()

    def _products(self) -> tuple[ElectrochemicalProduct, ...]:
        """Return the configured product slate.

        Returns:
            The configured products.
        """
        return self.config.products

    def _gas_package_fields(self) -> tuple[str, ...]:
        """Return the config fields naming each gas port's property package.

        Returns:
            One field per gas port.
        """
        return ("cathode_gas_property_package", "anode_gas_property_package")

    def _gas_package(self, field: str):
        """Return the package for one gas port, else ``gas_property_package``.

        Args:
            field: One of :meth:`_gas_package_fields`.

        Returns:
            The property (parameter) block the port is built from.
        """
        pkg = self.config[field]
        return self.config.gas_property_package if pkg is None else pkg

    # -- validation ------------------------------------------------------------

    def _resolve_config(self) -> None:
        """Fill config options defaulted at build time (none in the base class)."""

    def _validate_config(self) -> None:
        """Validate the config options.

        Raises:
            FlexConfigError: On the first invalid option found.
        """
        if self.config.liquid_property_package is None:
            raise FlexConfigError(
                "liquid_property_package is required; pass a single-phase "
                "property package.",
                field="liquid_property_package",
                value=None,
            )
        for field in self._gas_package_fields():
            if self._gas_package(field) is None:
                raise FlexConfigError(
                    f"{field} is required; pass it or gas_property_package, "
                    "a single-phase property package.",
                    field=field,
                    value=None,
                )
        products = self._products()
        names = [p.name for p in products]
        if not products or not all(n.isidentifier() for n in names):
            raise FlexConfigError(
                f"products must be one or more products named by Python "
                f"identifiers, got {names!r}.",
                field="products",
                value=names,
            )
        if len(set(names)) != len(names):
            raise FlexConfigError(
                f"product names must be unique, got {names!r}.",
                field="products",
                value=names,
            )
        for p in products:
            if p.electrons <= 0 or not 0.0 <= p.faradaic_efficiency <= 1.0:
                raise FlexConfigError(
                    f"product {p.name!r} needs electrons > 0 and a faradaic "
                    f"efficiency in [0, 1], got {p.electrons} and "
                    f"{p.faradaic_efficiency}.",
                    field="products",
                    value=p,
                )
            if p.phase is ProductPhase.LIQUID and not self.config.has_liquid_outlet:
                raise FlexConfigError(
                    f"liquid product {p.name!r} needs a liquid outlet; set "
                    "has_liquid_outlet=True.",
                    field="has_liquid_outlet",
                    value=False,
                )
        total = sum(p.faradaic_efficiency for p in products)
        if total > 1.0 + 1e-9:
            raise FlexConfigError(
                f"faradaic efficiencies sum to {total}, above 1.",
                field="products",
                value=names,
            )
        low, high = self.config.liquid_level_min, self.config.liquid_level_max
        capacity = _magnitude(self.config.separator_volume, pyunits.m**3)
        initial = _magnitude(self.config.initial_liquid_volume, pyunits.m**3)
        if not 0.0 <= low < high <= 1.0 or not low * capacity <= initial <= high * (
            capacity
        ):
            raise FlexConfigError(
                f"initial_liquid_volume ({initial} m^3) must lie within the fill "
                f"window [{low}, {high}] of separator_volume ({capacity} m^3), "
                "with 0 <= liquid_level_min < liquid_level_max <= 1.",
                field="initial_liquid_volume",
                value=initial,
            )

    # -- ports -----------------------------------------------------------------

    def _add_port(self, port_name: str, pkg, inlet: bool) -> None:
        """Build ``{port_name}_state`` from ``pkg`` and expose it as a port.

        Args:
            port_name: Name of the port.
            pkg: Property package to build the state block from.
            inlet: True for an inlet port, False for an outlet.
        """
        tb = self._find_time_block()
        self.add_component(
            f"{port_name}_state", pkg.build_state_block(time_index=tb.time_index)
        )
        state = self.find_component(f"{port_name}_state")
        add = self.add_inlet_port if inlet else self.add_outlet_port
        add(name=port_name, block=state, doc=f"{port_name} stream")

    def _build_ports(self) -> None:
        """Build the make-up inlet, both gas outlets, and the optional liquid outlet."""
        liquid = self.config.liquid_property_package
        self._add_port("inlet_water", liquid, inlet=True)
        self._add_port(
            "outlet_cathode_gas",
            self._gas_package("cathode_gas_property_package"),
            inlet=False,
        )
        self._add_port(
            "outlet_anode_gas",
            self._gas_package("anode_gas_property_package"),
            inlet=False,
        )
        if self.config.has_liquid_outlet:
            self._add_port("outlet_liquid", liquid, inlet=False)

    # -- electrochemistry ------------------------------------------------------

    def _build_electrochemistry(self) -> None:
        """Build current, Faraday's law per product, power, and waste heat."""
        tb = self._find_time_block()
        n_cells = self.config.n_cells

        self.current = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0.0, _magnitude(self.config.rated_current, pyunits.A)),
            units=pyunits.A,
            doc="Stack current: the unit's operating variable, A.",
        )
        self.electron_flow = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: pyunits.convert(
                n_cells * b.current[t] / FARADAY, pyunits.mol / pyunits.s
            ),
            doc="Molar flow of electrons through the stack, N_cells * I / F.",
        )

        for p in self._products():
            fe = self.declare_process_parameter(
                f"faradaic_efficiency_{p.name}",
                p.faradaic_efficiency,
                pyunits.dimensionless,
                f"Fraction of the stack charge producing {p.name}.",
                bounds=(0.0, 1.0),
            )
            production = pyo.Var(
                tb.time_index,
                initialize=0.0,
                domain=pyo.NonNegativeReals,
                units=pyunits.mol / pyunits.s,
                doc=f"Molar production rate of {p.name}.",
            )
            self.add_component(f"production_{p.name}", production)
            self.add_component(
                f"faradaic_relation_{p.name}",
                pyo.Constraint(
                    tb.time_index,
                    rule=lambda b, t, _n=production, _fe=fe, _z=p.electrons: _n[t]
                    == _fe * b.electron_flow[t] / _z,
                    doc=f"Faraday's law: production_{p.name} == faradaic "
                    f"efficiency * electron_flow / {p.electrons}.",
                ),
            )
            self.register_relation(
                self.find_component(f"faradaic_relation_{p.name}"), target=production
            )

        cell_voltage = self.declare_process_parameter(
            "cell_voltage",
            self.config.cell_voltage,
            pyunits.V,
            "Cell voltage at the operating point.",
            bounds=(0.0, None),
        )
        bop = self.declare_process_parameter(
            "bop_fraction",
            self.config.bop_fraction,
            pyunits.dimensionless,
            "Balance-of-plant draw as a fraction of stack power.",
            bounds=(0.0, None),
        )
        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.power_electrical_relation = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: power[t]
            == pyunits.convert(
                n_cells * cell_voltage * b.current[t] * (1 + bop), pyunits.kW
            ),
            doc="Electrical draw: N_cells * cell_voltage * current * "
            "(1 + bop_fraction). Swapped in place for a fitted polarization curve.",
        )
        self.register_relation(self.power_electrical_relation, target=power)

        v_tn = _magnitude(self.config.thermoneutral_voltage, pyunits.V) * pyunits.V
        self.waste_heat = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: pyunits.convert(
                n_cells * b.current[t] * (cell_voltage - v_tn), pyunits.kW
            ),
            doc="Stack heat released above the thermoneutral voltage, kW.",
        )

        spec = getattr(self.config.flexops_config, "surrogate", None)
        if spec is not None and (
            spec.surrogate_type is not SurrogateType.CONSTANT_INTENSITY
        ):
            self.swap_relation("power_electrical_relation", surrogate_from_spec(spec))

    # -- gas outlets -----------------------------------------------------------

    def _molar_volume(self):
        """Return the ideal-gas molar volume at the operating point.

        Returns:
            R*T/P as a constant expression, m^3/mol.
        """
        temperature = _magnitude(self.config.operating_temperature, pyunits.K)
        pressure = _magnitude(self.config.operating_pressure, pyunits.Pa)
        return pyunits.convert(
            pyunits.R * temperature * pyunits.K / (pressure * pyunits.Pa),
            pyunits.m**3 / pyunits.mol,
        )

    def _cathode_gas_extra(self, t):
        """Return the non-product molar flow leaving with the cathode gas.

        Args:
            t: Time point.

        Returns:
            The molar flow, mol/s.
        """
        return 0 * pyunits.mol / pyunits.s

    def _anode_gas_extra(self, t):
        """Return the non-oxygen molar flow leaving with the anode gas.

        Args:
            t: Time point.

        Returns:
            The molar flow, mol/s.
        """
        return 0 * pyunits.mol / pyunits.s

    def _build_gas_outlets(self) -> None:
        """Tie each gas outlet's flow to its molar flow and fix its T and P."""
        tb = self._find_time_block()
        cathode_phase = self._gas_phases["cathode_gas_property_package"]
        anode_phase = self._gas_phases["anode_gas_property_package"]
        molar_volume = self._molar_volume()
        cathode = self.outlet_cathode_gas_state
        anode = self.outlet_anode_gas_state
        gas_products = [
            self.find_component(f"production_{p.name}")
            for p in self._products()
            if p.phase is ProductPhase.GAS
        ]

        @self.Constraint(
            tb.time_index,
            doc="Cathode gas volume: gas products plus any unreacted feed, at "
            "the operating temperature and pressure.",
        )
        def cathode_gas_balance(b, t):
            moles = sum(n[t] for n in gas_products) + b._cathode_gas_extra(t)
            return cathode.flow_vol_phase[t, cathode_phase] == pyunits.convert(
                molar_volume * moles, pyunits.m**3 / pyunits.hr
            )

        @self.Constraint(
            tb.time_index,
            doc="Anode gas volume: oxygen (electron_flow / 4) plus any crossover, "
            "at the operating temperature and pressure.",
        )
        def anode_gas_balance(b, t):
            moles = b.electron_flow[t] / 4 + b._anode_gas_extra(t)
            return anode.flow_vol_phase[t, anode_phase] == pyunits.convert(
                molar_volume * moles, pyunits.m**3 / pyunits.hr
            )

        temperature = _magnitude(self.config.operating_temperature, pyunits.K)
        pressure = _magnitude(self.config.operating_pressure, pyunits.Pa)
        for state in (cathode, anode):
            state.temperature.fix(temperature)
            state.pressure.fix(pressure)

    # -- liquid loop -----------------------------------------------------------

    def _build_liquid_loop(self) -> None:
        """Build water consumption, the separator holdup, and the liquid outlet."""
        tb = self._find_time_block()
        phase = self._liquid_phase
        density = _magnitude(self.config.liquid_density, pyunits.kg / pyunits.m**3)
        density = density * pyunits.kg / pyunits.m**3
        products = self._products()

        self.water_consumption = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: pyunits.convert(
                sum(
                    p.water_per_mol * b.find_component(f"production_{p.name}")[t]
                    for p in products
                )
                * WATER_MOLAR_MASS
                / density,
                pyunits.m**3 / pyunits.hr,
            ),
            doc="Volume of water the cell reactions consume.",
        )

        bleed = 0.0
        if self.config.has_liquid_outlet:
            bleed = self.declare_process_parameter(
                "bleed_fraction",
                self.config.bleed_fraction,
                pyunits.dimensionless,
                "Liquid bled per unit volume of water consumed.",
                bounds=(0.0, None),
            )
            liquid_products = [p for p in products if p.phase is ProductPhase.LIQUID]
            outlet = self.outlet_liquid_state

            @self.Constraint(
                tb.time_index,
                doc="Liquid outlet: bleed_fraction * water_consumption plus the "
                "volume of liquid products drawn off.",
            )
            def liquid_outlet_balance(b, t):
                drawn = sum(
                    pyunits.convert(
                        b.find_component(f"production_{p.name}")[t]
                        * p.molar_mass
                        / density,
                        pyunits.m**3 / pyunits.hr,
                    )
                    for p in liquid_products
                )
                return (
                    outlet.flow_vol_phase[t, phase]
                    == bleed * b.water_consumption[t] + drawn
                )

            self.add_pass_through_constraints(
                self.inlet_water,
                self.outlet_liquid,
                exclude_vars=[
                    self.config.liquid_property_package.get_flow_basis_var_name()
                ],
                name_prefix="pass_through_liquid",
            )

        capacity = _magnitude(self.config.separator_volume, pyunits.m**3)
        self.liquid_volume = pyo.Var(
            tb.time_index,
            initialize=_magnitude(self.config.initial_liquid_volume, pyunits.m**3),
            bounds=(
                self.config.liquid_level_min * capacity,
                self.config.liquid_level_max * capacity,
            ),
            units=pyunits.m**3,
            doc="Separator liquid inventory, bounded by the fill window.",
        )
        self.initial_liquid_volume = pyo.Param(
            initialize=_magnitude(self.config.initial_liquid_volume, pyunits.m**3),
            mutable=True,
            units=pyunits.m**3,
            doc="Liquid inventory at the first time point (rolling-horizon "
            "initial state).",
        )
        tb.register_initial_state(self.initial_liquid_volume)
        self.register_process_parameter(self.initial_liquid_volume, regressable=False)

        @self.Constraint(doc="Initial condition: liquid_volume[0] == initial.")
        def initial_liquid_volume_eq(b):
            return b.liquid_volume[0] == b.initial_liquid_volume

        make_up = self.inlet_water_state.flow_vol_phase

        @self.Constraint(
            list(tb.time_index)[1:],
            doc="Separator holdup (backward): liquid_volume[t] = "
            "liquid_volume[t-1] + dt * (make-up - (1 + bleed) * consumption).",
        )
        def liquid_holdup(b, t):
            net = make_up[t, phase] - (1 + bleed) * b.water_consumption[t]
            return b.liquid_volume[t] == b.liquid_volume[t - 1] + pyunits.convert(
                tb.dt * net, pyunits.m**3
            )

    # -- IO registration -------------------------------------------------------

    def _register_io(self) -> None:
        """Register current and make-up as inputs; power, gas, inventory as outputs."""
        self.register_io_variable(self.current, role="input")
        for var in self.inlet_water_state.define_state_vars().values():
            self.register_io_variable(var, role="input")
        self.register_io_variable(self.power_electrical, role="output")
        self.register_io_variable(self.liquid_volume, role="output")
        outlets = ["outlet_cathode_gas_state", "outlet_anode_gas_state"]
        if self.config.has_liquid_outlet:
            outlets.append("outlet_liquid_state")
        for name in outlets:
            self.register_io_variable(
                self.find_component(name).flow_vol_phase, role="output"
            )


_TECHNOLOGY_DEFAULTS = {
    ElectrolyzerTechnology.PEM: {
        "cell_voltage": 1.9 * pyunits.V,
        "operating_temperature": 333.15 * pyunits.K,
        "operating_pressure": 30e5 * pyunits.Pa,
    },
    ElectrolyzerTechnology.AEM: {
        "cell_voltage": 2.0 * pyunits.V,
        "operating_temperature": 323.15 * pyunits.K,
        "operating_pressure": 35e5 * pyunits.Pa,
    },
}


@declare_process_block_class("WaterElectrolyzer")
class WaterElectrolyzerData(ElectrolyzerData):
    """A water electrolyzer producing hydrogen, with defaults set by technology.

    Config:
        Inherits the Electrolyzer config without ``products``; adds
        ``technology``.
    """

    CONFIG = ElectrolyzerData.CONFIG()
    del CONFIG["products"]  # fixed to hydrogen
    for _name in ("cell_voltage", "operating_temperature", "operating_pressure"):
        CONFIG.get(_name).set_default_value(None)
    CONFIG.declare(
        "technology",
        ConfigValue(
            default=ElectrolyzerTechnology.PEM,
            domain=ElectrolyzerTechnology,
            description="Electrolysis technology; selects the defaults of "
            "cell_voltage, operating_temperature and operating_pressure.",
        ),
    )

    def _resolve_config(self) -> None:
        """Fill unset options from the technology defaults."""
        for name, value in _TECHNOLOGY_DEFAULTS[self.config.technology].items():
            if self.config[name] is None:
                self.config[name] = value

    def _products(self) -> tuple[ElectrochemicalProduct, ...]:
        """Return the fixed hydrogen product slate.

        Returns:
            A one-product tuple, :data:`HYDROGEN`.
        """
        return (HYDROGEN,)


@declare_process_block_class("CO2Electrolyzer")
class CO2ElectrolyzerData(ElectrolyzerData):
    """A CO2 reduction electrolyzer with a CO2 feed inlet.

    Config:
        Inherits the Electrolyzer config; adds ``single_pass_conversion``,
        ``co2_crossover_per_electron``, and ``co2_property_package``.
    """

    CONFIG = ElectrolyzerData.CONFIG()
    CONFIG.get("products").set_default_value(
        (
            ElectrochemicalProduct(
                name="CO",
                electrons=2,
                faradaic_efficiency=0.9,
                phase=ProductPhase.GAS,
                co2_per_mol=1.0,
                water_per_mol=0.0,
                molar_mass=28.01 * pyunits.g / pyunits.mol,
            ),
            dataclasses.replace(HYDROGEN, faradaic_efficiency=0.1),
        )
    )
    CONFIG.get("cell_voltage").set_default_value(3.0 * pyunits.V)
    CONFIG.get("thermoneutral_voltage").set_default_value(1.47 * pyunits.V)
    CONFIG.get("operating_temperature").set_default_value(298.15 * pyunits.K)
    CONFIG.declare(
        "single_pass_conversion",
        ConfigValue(
            default=0.5,
            domain=float,
            description="Fraction of the CO2 fed that is consumed or crosses "
            "over in one pass, in (0, 1] (a fixed, regressable Var once built).",
        ),
    )
    CONFIG.declare(
        "co2_crossover_per_electron",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Moles of CO2 carried to the anode as carbonate per mole "
            "of electrons (a fixed, regressable Var once built).",
        ),
    )
    CONFIG.declare(
        "co2_property_package",
        ConfigValue(
            default=None,
            description="Single-phase property package for the CO2 inlet; "
            "defaults to gas_property_package.",
        ),
    )

    def _gas_package_fields(self) -> tuple[str, ...]:
        """Return the gas package fields, including the CO2 inlet's.

        Returns:
            One field per gas port.
        """
        return (*super()._gas_package_fields(), "co2_property_package")

    def _validate_config(self) -> None:
        """Also validate the CO2 conversion and crossover options.

        Raises:
            FlexConfigError: On the first invalid option found.
        """
        super()._validate_config()
        conversion = self.config.single_pass_conversion
        if not 0.0 < conversion <= 1.0:
            raise FlexConfigError(
                f"single_pass_conversion must lie in (0, 1], got {conversion}.",
                field="single_pass_conversion",
                value=conversion,
            )
        if self.config.co2_crossover_per_electron < 0.0:
            raise FlexConfigError(
                "co2_crossover_per_electron must be non-negative, got "
                f"{self.config.co2_crossover_per_electron}.",
                field="co2_crossover_per_electron",
                value=self.config.co2_crossover_per_electron,
            )

    def _build_ports(self) -> None:
        """Add the CO2 feed inlet to the base ports."""
        super()._build_ports()
        self._add_port(
            "inlet_co2", self._gas_package("co2_property_package"), inlet=True
        )

    def _build_electrochemistry(self) -> None:
        """Also build CO2 consumption, crossover, and the CO2 feed."""
        super()._build_electrochemistry()
        tb = self._find_time_block()
        products = self._products()
        conversion = self.declare_process_parameter(
            "single_pass_conversion",
            self.config.single_pass_conversion,
            pyunits.dimensionless,
            "Fraction of the CO2 fed that is consumed or crosses over per pass.",
            bounds=(1e-6, 1.0),
        )
        crossover = self.declare_process_parameter(
            "co2_crossover_per_electron",
            self.config.co2_crossover_per_electron,
            pyunits.dimensionless,
            "Moles of CO2 carried to the anode as carbonate per mole of electrons.",
            bounds=(0.0, None),
        )
        self.co2_consumption = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: sum(
                p.co2_per_mol * b.find_component(f"production_{p.name}")[t]
                for p in products
            ),
            doc="Molar CO2 consumed by the cell reactions.",
        )
        self.co2_crossover = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: crossover * b.electron_flow[t],
            doc="Molar CO2 carried to the anode as carbonate.",
        )
        self.co2_unreacted = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: (b.co2_consumption[t] + b.co2_crossover[t])
            * (1 / conversion - 1),
            doc="Molar CO2 leaving unreacted with the cathode gas.",
        )
        feed = self.inlet_co2_state.flow_vol_phase
        molar_volume = self._molar_volume()

        @self.Constraint(
            tb.time_index,
            doc="CO2 feed: (consumption + crossover) / single_pass_conversion.",
        )
        def co2_feed_balance(b, t):
            return feed[t, self._gas_phases["co2_property_package"]] == pyunits.convert(
                molar_volume * (b.co2_consumption[t] + b.co2_crossover[t]) / conversion,
                pyunits.m**3 / pyunits.hr,
            )

    def _cathode_gas_extra(self, t):
        """Return the unreacted CO2 leaving with the cathode gas.

        Args:
            t: Time point.

        Returns:
            The molar flow, mol/s.
        """
        return self.co2_unreacted[t]

    def _anode_gas_extra(self, t):
        """Return the crossover CO2 leaving with the anode gas.

        Args:
            t: Time point.

        Returns:
            The molar flow, mol/s.
        """
        return self.co2_crossover[t]

    def _register_io(self) -> None:
        """Also register the CO2 inlet conditions as inputs and its flow as output."""
        super()._register_io()
        state = self.inlet_co2_state
        self.register_io_variable(state.pressure, role="input")
        self.register_io_variable(state.temperature, role="input")
        self.register_io_variable(state.flow_vol_phase, role="output")
