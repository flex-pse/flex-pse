"""Electrolyzer unit models: a stack plus its gas-liquid separators."""

import dataclasses
import enum

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import Bool, ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.config.schema import SurrogateType
from flexcore.exceptions import FlexConfigError
from flexcore.logger import get_logger
from flexops.core.ops_block import OpsBlockData
from flexops.surrogates import surrogate_from_spec
from flexops.unit_models._multiport import single_flow_phase

FARADAY = 96485.33212 * pyunits.C / pyunits.mol
WATER_MOLAR_MASS = 18.015 * pyunits.g / pyunits.mol

_log = get_logger(__name__)


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
        molar_mass: Molar mass with units (used for liquid products).
        reactants: Moles of each reactant consumed per mole of product, keyed
            by reactant name; negative values are produced.
    """

    name: str
    electrons: int
    faradaic_efficiency: float
    phase: ProductPhase
    molar_mass: object
    reactants: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce ``phase`` from its string value."""
        object.__setattr__(self, "phase", ProductPhase(self.phase))


HYDROGEN = ElectrochemicalProduct(
    name="H2",
    electrons=2,
    faradaic_efficiency=1.0,
    phase=ProductPhase.GAS,
    molar_mass=2.016 * pyunits.g / pyunits.mol,
    reactants={"H2O": 1.0},
)

OXYGEN = ElectrochemicalProduct(
    name="O2",
    electrons=4,
    faradaic_efficiency=1.0,
    phase=ProductPhase.GAS,
    molar_mass=32.00 * pyunits.g / pyunits.mol,
)


def _products_domain(value) -> tuple[ElectrochemicalProduct, ...]:
    """Coerce a list of products or product dicts into products.

    Args:
        value: Products, or dicts of their fields, e.g.
            ``[HYDROGEN, {"name": "CO", "electrons": 2,
            "faradaic_efficiency": 0.9, "phase": "gas",
            "molar_mass": 28.01 * pyunits.g / pyunits.mol,
            "reactants": {"CO2": 1.0}}]``.

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
        "electrode_area",
        ConfigValue(
            default=1000.0 * pyunits.cm**2,
            description="Active electrode area of one cell, cm^2.",
        ),
    )
    CONFIG.declare(
        "rated_current_density",
        ConfigValue(
            default=2.0 * pyunits.A / pyunits.cm**2,
            description="Maximum current density, the upper bound on "
            "current_density[t], A/cm^2.",
        ),
    )
    CONFIG.declare(
        "cell_voltage",
        ConfigValue(
            default=1.9 * pyunits.V,
            description="Cell voltage before ohmic loss (a fixed, regressable Var "
            "once built), V.",
        ),
    )
    CONFIG.declare(
        "ohmic_loss",
        ConfigValue(
            default=False,
            domain=Bool,
            description="If True, add area_specific_resistance * current_density "
            "to cell_voltage, making power quadratic in current density.",
        ),
    )
    CONFIG.declare(
        "area_specific_resistance",
        ConfigValue(
            default=0.15 * pyunits.ohm * pyunits.cm**2,
            description="Area-specific cell resistance, used when ohmic_loss is "
            "True (a fixed, regressable Var once built), ohm*cm^2.",
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
        "anode_products",
        ConfigValue(
            default=(OXYGEN,),
            domain=_products_domain,
            description="Anode gas products, as ElectrochemicalProduct entries (or "
            "dicts of their fields). Faradaic efficiencies may sum to at most 1.",
        ),
    )
    CONFIG.declare(
        "balance_product",
        ConfigValue(
            default=None,
            description="Name of the cathode product whose rate closes the charge "
            "balance (the electron flow left after the other cathode products); "
            "its configured faradaic efficiency is ignored and its relation is not "
            "swappable. None gives every product its own Faradaic relation.",
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

    def _anode_products(self) -> tuple[ElectrochemicalProduct, ...]:
        """Return the configured anode product slate.

        Returns:
            The configured anode products.
        """
        return self.config.anode_products

    def _gas_package_fields(self) -> tuple[str, ...]:
        """Return the config fields naming each gas port's property package.

        Returns:
            One field per gas port.
        """
        return ("cathode_gas_property_package", "anode_gas_property_package")

    def _reactant_names(self) -> tuple[str, ...]:
        """Return every reactant the products consume, always including water.

        Returns:
            Reactant names, each built as a ``consumption_{name}`` Expression.
        """
        products = (*self._products(), *self._anode_products())
        names = dict.fromkeys(["H2O", *(r for p in products for r in p.reactants)])
        return tuple(names)

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
        tables = {
            "products": self._products(),
            "anode_products": self._anode_products(),
        }
        for field, products in tables.items():
            names = [p.name for p in products]
            if not products or not all(n.isidentifier() for n in names):
                raise FlexConfigError(
                    f"{field} must be one or more products named by Python "
                    f"identifiers, got {names!r}.",
                    field=field,
                    value=names,
                )
            total = sum(p.faradaic_efficiency for p in products)
            if total > 1.0 + 1e-9:
                raise FlexConfigError(
                    f"{field} faradaic efficiencies sum to {total}, above 1.",
                    field=field,
                    value=names,
                )
        names = [p.name for products in tables.values() for p in products]
        if len(set(names)) != len(names):
            raise FlexConfigError(
                f"product names must be unique across cathode and anode, "
                f"got {names!r}.",
                field="products",
                value=names,
            )
        for p in (*self._products(), *self._anode_products()):
            if not all(r.isidentifier() for r in p.reactants):
                raise FlexConfigError(
                    f"product {p.name!r} reactant names must be Python "
                    f"identifiers, got {list(p.reactants)!r}.",
                    field="products",
                    value=p,
                )
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
        balance = self.config.balance_product
        if balance is not None and balance not in [p.name for p in self._products()]:
            raise FlexConfigError(
                f"balance_product {balance!r} must name a cathode product, got "
                f"{[p.name for p in self._products()]!r}; pass None to give every "
                "product its own Faradaic relation.",
                field="balance_product",
                value=balance,
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

        j_units = pyunits.A / pyunits.cm**2
        self.current_density = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0.0, _magnitude(self.config.rated_current_density, j_units)),
            units=j_units,
            doc="Current density: the unit's operating variable, A/cm^2.",
        )
        area = _magnitude(self.config.electrode_area, pyunits.cm**2) * pyunits.cm**2
        self.current = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: pyunits.convert(b.current_density[t] * area, pyunits.A),
            doc="Stack current, current_density * electrode_area, A.",
        )
        self.electron_flow = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: pyunits.convert(
                n_cells * b.current[t] / FARADAY, pyunits.mol / pyunits.s
            ),
            doc="Molar flow of electrons through the stack, N_cells * I / F.",
        )

        products = (*self._products(), *self._anode_products())
        balance = self.config.balance_product
        for p in products:
            if p.name != balance:
                self._build_faradaic(p)
        for p in self._products():
            if p.name == balance:
                self._build_charge_balance(p)
        for r in self._reactant_names():
            self.add_component(
                f"consumption_{r}",
                pyo.Expression(
                    tb.time_index,
                    rule=lambda b, t, _r=r: sum(
                        p.reactants.get(_r, 0.0)
                        * b.find_component(f"production_{p.name}")[t]
                        for p in products
                    ),
                    doc=f"Molar {r} consumed by the cell reactions.",
                ),
            )

        cell_voltage = self.declare_process_parameter(
            "cell_voltage",
            self.config.cell_voltage,
            pyunits.V,
            "Cell voltage at the operating point.",
            bounds=(0.0, None),
        )
        asr = None
        if self.config.ohmic_loss:
            _log.warning(
                "%s: ohmic_loss=True makes power quadratic in current "
                "density; the schedule needs a nonlinear solver and solves slower.",
                self.name,
            )
            asr = self.declare_process_parameter(
                "area_specific_resistance",
                self.config.area_specific_resistance,
                pyunits.ohm * pyunits.cm**2,
                "Area-specific cell resistance.",
                bounds=(0.0, None),
            )
        self.operating_voltage = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: (
                cell_voltage
                if asr is None
                else cell_voltage
                + pyunits.convert(asr * b.current_density[t], pyunits.V)
            ),
            doc="Cell voltage plus any ohmic loss, V.",
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
                n_cells * b.operating_voltage[t] * b.current[t] * (1 + bop),
                pyunits.kW,
            ),
            doc="Electrical draw: N_cells * operating_voltage * current * "
            "(1 + bop_fraction). Swapped in place for a fitted polarization curve.",
        )
        self.register_relation(self.power_electrical_relation, target=power)

        v_tn = _magnitude(self.config.thermoneutral_voltage, pyunits.V) * pyunits.V
        self.waste_heat = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: pyunits.convert(
                n_cells * b.current[t] * (b.operating_voltage[t] - v_tn), pyunits.kW
            ),
            doc="Stack heat released above the thermoneutral voltage, kW.",
        )

        spec = getattr(self.config.flexops_config, "surrogate", None)
        if spec is not None and (
            spec.surrogate_type is not SurrogateType.CONSTANT_INTENSITY
        ):
            self.swap_relation("power_electrical_relation", surrogate_from_spec(spec))

    def _build_faradaic(self, p: ElectrochemicalProduct) -> None:
        """Build one product's production rate and its Faradaic relation.

        Args:
            p: The product.
        """
        tb = self._find_time_block()
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

    def _build_charge_balance(self, p: ElectrochemicalProduct) -> None:
        """Build the balance product's rate from the charge the other products leave.

        Args:
            p: The balance product, one of the cathode products.
        """
        tb = self._find_time_block()
        others = [q for q in self._products() if q.name != p.name]
        production = pyo.Var(
            tb.time_index,
            initialize=0.0,
            domain=pyo.NonNegativeReals,
            units=pyunits.mol / pyunits.s,
            doc=f"Molar production rate of {p.name}.",
        )
        self.add_component(f"production_{p.name}", production)
        self.add_component(
            f"charge_balance_{p.name}",
            pyo.Constraint(
                tb.time_index,
                rule=lambda b, t: p.electrons * production[t]
                == b.electron_flow[t]
                - sum(
                    q.electrons * b.find_component(f"production_{q.name}")[t]
                    for q in others
                ),
                doc=f"Charge balance: {p.electrons} * production_{p.name} == "
                "electron_flow minus the other cathode products' charge.",
            ),
        )

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
        """Return the non-product molar flow leaving with the anode gas.

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
        anode_products = [
            self.find_component(f"production_{p.name}")
            for p in self._anode_products()
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
            doc="Anode gas volume: anode products plus any crossover, at the "
            "operating temperature and pressure.",
        )
        def anode_gas_balance(b, t):
            moles = sum(n[t] for n in anode_products) + b._anode_gas_extra(t)
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
        products = (*self._products(), *self._anode_products())

        self.water_consumption = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: pyunits.convert(
                b.consumption_H2O[t] * WATER_MOLAR_MASS / density,
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
        """Register j and make-up as inputs; power, gas, inventory as outputs."""
        self.register_io_variable(self.current_density, role="input")
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
        Inherits the Electrolyzer config without ``products`` or
        ``anode_products``; adds
        ``technology``.
    """

    CONFIG = ElectrolyzerData.CONFIG()
    del CONFIG["products"]  # fixed to hydrogen
    del CONFIG["anode_products"]  # fixed to oxygen
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

    def _anode_products(self) -> tuple[ElectrochemicalProduct, ...]:
        """Return the fixed oxygen anode product slate.

        Returns:
            A one-product tuple, :data:`OXYGEN`.
        """
        return (OXYGEN,)


@declare_process_block_class("CO2Electrolyzer")
class CO2ElectrolyzerData(ElectrolyzerData):
    """A CO2 reduction electrolyzer with a CO2 feed inlet.

    Config:
        Inherits the Electrolyzer config with ``balance_product`` defaulting to
        ``"H2"``; adds ``single_pass_conversion``, ``co2_crossover_per_electron``,
        ``co2_dissolved_fraction``, ``mass_transfer_coefficient``,
        ``co2_bulk_concentration``, and ``co2_property_package``.
    """

    # TODO: replace the fixed dissolved fraction with linearized CO2 kinetics.
    # TODO: make mass_transfer_coefficient a decision variable set by electrolyte flow.

    CONFIG = ElectrolyzerData.CONFIG()
    CONFIG.get("products").set_default_value(
        (
            ElectrochemicalProduct(
                name="CO",
                electrons=2,
                faradaic_efficiency=0.9,
                phase=ProductPhase.GAS,
                molar_mass=28.01 * pyunits.g / pyunits.mol,
                reactants={"CO2": 1.0},
            ),
            dataclasses.replace(HYDROGEN, faradaic_efficiency=0.1),
        )
    )
    CONFIG.get("cell_voltage").set_default_value(3.0 * pyunits.V)
    CONFIG.get("thermoneutral_voltage").set_default_value(1.47 * pyunits.V)
    CONFIG.get("operating_temperature").set_default_value(298.15 * pyunits.K)
    CONFIG.get("balance_product").set_default_value("H2")
    CONFIG.declare(
        "mass_transfer_coefficient",
        ConfigValue(
            default=None,
            description="CO2 mass-transfer coefficient to the cathode, the CO2 "
            "diffusivity over the boundary-layer thickness (a fixed, regressable "
            "Var once built), m/s. None builds no transport limit.",
        ),
    )
    CONFIG.declare(
        "co2_bulk_concentration",
        ConfigValue(
            default=34.0 * pyunits.mol / pyunits.m**3,
            description="CO2 concentration in the bulk electrolyte, used with "
            "mass_transfer_coefficient (a fixed, regressable Var once built), "
            "mol/m^3.",
        ),
    )
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
        "co2_dissolved_fraction",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Fraction of the unreacted CO2 leaving dissolved in the "
            "electrolyte rather than with the cathode gas, in [0, 1] (a fixed, "
            "regressable Var once built).",
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

    def _reactant_names(self) -> tuple[str, ...]:
        """Return the reactant names, always including CO2.

        Returns:
            Reactant names, each built as a ``consumption_{name}`` Expression.
        """
        return tuple(dict.fromkeys([*super()._reactant_names(), "CO2"]))

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
        dissolved = self.config.co2_dissolved_fraction
        if not 0.0 <= dissolved <= 1.0:
            raise FlexConfigError(
                f"co2_dissolved_fraction must lie in [0, 1], got {dissolved}.",
                field="co2_dissolved_fraction",
                value=dissolved,
            )
        if self.config.co2_crossover_per_electron < 0.0:
            raise FlexConfigError(
                "co2_crossover_per_electron must be non-negative, got "
                f"{self.config.co2_crossover_per_electron}.",
                field="co2_crossover_per_electron",
                value=self.config.co2_crossover_per_electron,
            )
        k_m = self.config.mass_transfer_coefficient
        if k_m is not None and _magnitude(k_m, pyunits.m / pyunits.s) <= 0.0:
            raise FlexConfigError(
                f"mass_transfer_coefficient must be positive, got {k_m}.",
                field="mass_transfer_coefficient",
                value=k_m,
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
        self.co2_crossover = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: crossover * b.electron_flow[t],
            doc="Molar CO2 carried to the anode as carbonate.",
        )
        self.co2_unreacted = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: (b.consumption_CO2[t] + b.co2_crossover[t])
            * (1 / conversion - 1),
            doc="Molar CO2 leaving the stack unreacted.",
        )
        dissolved = self.declare_process_parameter(
            "co2_dissolved_fraction",
            self.config.co2_dissolved_fraction,
            pyunits.dimensionless,
            "Fraction of the unreacted CO2 leaving dissolved in the electrolyte.",
            bounds=(0.0, 1.0),
        )
        self.co2_dissolved = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: dissolved * b.co2_unreacted[t],
            doc="Molar unreacted CO2 leaving dissolved in the electrolyte.",
        )
        feed = self.inlet_co2_state.flow_vol_phase
        molar_volume = self._molar_volume()

        @self.Constraint(
            tb.time_index,
            doc="CO2 feed: (consumption + crossover) / single_pass_conversion.",
        )
        def co2_feed_balance(b, t):
            return feed[t, self._gas_phases["co2_property_package"]] == pyunits.convert(
                molar_volume * (b.consumption_CO2[t] + b.co2_crossover[t]) / conversion,
                pyunits.m**3 / pyunits.hr,
            )

        if self.config.mass_transfer_coefficient is not None:
            self._build_co2_transport_limit()

    def _build_co2_transport_limit(self) -> None:
        """Build the local CO2 concentration and cap the CO2 flux at transport."""
        tb = self._find_time_block()
        k_m = self.declare_process_parameter(
            "mass_transfer_coefficient",
            self.config.mass_transfer_coefficient,
            pyunits.m / pyunits.s,
            "CO2 mass-transfer coefficient: diffusivity over boundary-layer thickness.",
            bounds=(0.0, None),
        )
        c_bulk = self.declare_process_parameter(
            "co2_bulk_concentration",
            self.config.co2_bulk_concentration,
            pyunits.mol / pyunits.m**3,
            "CO2 concentration in the bulk electrolyte.",
            bounds=(0.0, None),
        )
        area = (
            self.config.n_cells
            * _magnitude(self.config.electrode_area, pyunits.m**2)
            * pyunits.m**2
        )
        self.co2_local_concentration = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: c_bulk
            - pyunits.convert(
                (b.consumption_CO2[t] + b.co2_crossover[t]) / (k_m * area),
                pyunits.mol / pyunits.m**3,
            ),
            doc="CO2 concentration at the cathode: bulk minus the CO2 flux over "
            "mass_transfer_coefficient, mol/m^3.",
        )
        self.co2_transport_limit = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: b.consumption_CO2[t] + b.co2_crossover[t]
            <= pyunits.convert(k_m * c_bulk * area, pyunits.mol / pyunits.s),
            doc="CO2 consumed and crossed over cannot exceed what transport "
            "delivers: mass_transfer_coefficient * co2_bulk_concentration * area.",
        )

    def _cathode_gas_extra(self, t):
        """Return the undissolved unreacted CO2 leaving with the cathode gas.

        Args:
            t: Time point.

        Returns:
            The molar flow, mol/s.
        """
        return self.co2_unreacted[t] - self.co2_dissolved[t]

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
