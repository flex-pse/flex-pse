"""``Pump``: a SISO pump with a selectable energy relation (§3.4, LP)."""

import enum

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from idaes.core.util.constants import Constants
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexcore.nomenclature import PowerKind
from flexops.unit_models.base.siso import SISOBlockData


class PumpEnergyRelation(enum.StrEnum):
    """Which electrical-power relation a ``Pump`` builds."""

    ENERGY_INTENSITY = "energy_intensity"
    EFFICIENCY_HEAD = "efficiency_head"


def _energy_relation(value) -> PumpEnergyRelation:
    """Coerce a config value to a :class:`PumpEnergyRelation`.

    Called from ``build()`` rather than installed as the ConfigValue domain:
    Pyomo wraps domain exceptions in its own ``ValueError``, whereas a build
    exception surfaces as-is, keeping the project ``FlexConfigError`` contract.

    Raises:
        FlexConfigError: If ``value`` is not a valid relation name.
    """
    try:
        return PumpEnergyRelation(value)
    except ValueError as exc:
        allowed = ", ".join(repr(r.value) for r in PumpEnergyRelation)
        raise FlexConfigError(
            f"energy_relation must be one of {allowed}, got {value!r}.",
            field="energy_relation",
            value=value,
        ) from exc


@declare_process_block_class("Pump")
class PumpData(SISOBlockData):
    """A pump with a constant-parameter electrical-power relation (LP).

    Inherits the inlet/outlet ports, the pass-through mass balance, and the
    ``flow_vol`` handle from
    :class:`~flexops.unit_models.base.siso.SISOBlockData` and adds only the
    electrical-power relationship, selected by the ``energy_relation`` config:

    ``"energy_intensity"`` (default):

    .. math::

        \\text{electrical\\_power}[t] = e \\cdot \\text{flow\\_vol}[t]

    where :math:`e` is the ``energy_intensity`` config (a mutable, regressable
    Param, default 0.5 kWh/m³). **Unit algebra:** ``flow_vol`` is m³/hr, so
    kWh/m³ × m³/hr = kW — dimensionally exact with no conversion factor.

    ``"efficiency_head"``:

    .. math::

        \\text{electrical\\_power}[t] = \\frac{\\rho \\, g \\, H \\,
        \\text{flow\\_vol}[t]}{\\eta}

    where :math:`\\rho` is the property package's fixed density, :math:`g`
    the gravitational acceleration, :math:`H` the ``head`` config (required,
    m), and :math:`\\eta` the ``efficiency`` config (default 0.7,
    dimensionless). ``head`` and ``efficiency`` are mutable, regressable
    Params — FlexParameterize may later swap the efficiency for a fitted
    function, but v0 keeps it a constant parameter. The hydraulic product is
    converted to kW with ``pyunits.convert``.

    Config: the SISO/OpsBlock config plus ``energy_relation``,
    ``energy_intensity``, ``efficiency``, and ``head``.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Pump
        >>> m = dummy_time_block(3)
        >>> m.pump = Pump(property_package=m.properties, energy_intensity=0.7)
        >>> m.head_pump = Pump(
        ...     property_package=m.properties,
        ...     energy_relation="efficiency_head", efficiency=0.8, head=50.0,
        ... )
    """

    CONFIG = SISOBlockData.CONFIG()
    CONFIG.declare(
        "energy_relation",
        ConfigValue(
            default=PumpEnergyRelation.ENERGY_INTENSITY,
            description="Which electrical-power relation to build: "
            "'energy_intensity' (power = intensity * flow, the default) or "
            "'efficiency_head' (power = density * g * head * flow / "
            "efficiency).",
        ),
    )
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.5,
            domain=float,
            description="Electrical energy intensity in kWh per m^3 pumped "
            "(default 0.5, an ordinary transfer-pump duty). Used only by the "
            "'energy_intensity' relation.",
        ),
    )
    CONFIG.declare(
        "efficiency",
        ConfigValue(
            default=0.7,
            domain=float,
            description="Overall pump efficiency, dimensionless in (0, 1] "
            "(default 0.7). Used only by the 'efficiency_head' relation.",
        ),
    )
    CONFIG.declare(
        "head",
        ConfigValue(
            default=None,
            description="Pump head in m; required by the 'efficiency_head' "
            "relation.",
        ),
    )

    def build(self) -> None:
        """Build the SISO base and the selected electrical-power relation.

        Raises:
            FlexConfigError: If the ``efficiency_head`` relation is selected
                without a ``head``, or on a property package without a fixed
                density.
        """
        super().build()
        time_block = self._find_time_block()
        relation = _energy_relation(self.config.energy_relation)
        electrical_power = self.declare_power(PowerKind.ELECTRICAL)

        if relation is PumpEnergyRelation.ENERGY_INTENSITY:
            self.energy_intensity = pyo.Param(
                initialize=self.config.energy_intensity,
                mutable=True,
                units=pyunits.kWh / pyunits.m**3,
                doc="Electrical energy intensity per unit volume pumped",
            )
            self.register_process_parameter(self.energy_intensity, regressable=True)

            @self.Constraint(
                time_block.time_index,
                doc="Electrical draw: electrical_power[t] = energy_intensity "
                "* flow_vol[t]; kWh/m^3 x m^3/hr = kW, dimensionally exact "
                "with no conversion factor",
            )
            def electrical_power_eq(blk, t):
                return electrical_power[t] == blk.energy_intensity * blk.flow_vol[t]

        else:
            if self.config.head is None:
                raise FlexConfigError(
                    "The 'efficiency_head' relation requires a head (in m); "
                    "e.g. Pump(..., energy_relation='efficiency_head', "
                    "head=50.0).",
                    field="head",
                )
            properties = self.config.property_package
            if not hasattr(properties, "dens_mass"):
                raise FlexConfigError(
                    "The 'efficiency_head' relation needs a property package "
                    "with a fixed density; build it with fixed_density=True.",
                    field="property_package",
                )
            self.efficiency = pyo.Param(
                initialize=self.config.efficiency,
                mutable=True,
                units=pyunits.dimensionless,
                doc="Overall pump efficiency (hydraulic over electrical power)",
            )
            self.head = pyo.Param(
                initialize=self.config.head,
                mutable=True,
                units=pyunits.m,
                doc="Pump head",
            )
            self.register_process_parameter(self.efficiency, regressable=True)
            self.register_process_parameter(self.head, regressable=True)

            @self.Constraint(
                time_block.time_index,
                doc="Electrical draw: electrical_power[t] = density * g * "
                "head * flow_vol[t] / efficiency, converted to kW",
            )
            def electrical_power_eq(blk, t):
                hydraulic = (
                    properties.dens_mass
                    * Constants.acceleration_gravity
                    * blk.head
                    * blk.flow_vol[t]
                )
                return electrical_power[t] == pyunits.convert(
                    hydraulic / blk.efficiency, to_units=pyunits.kW
                )

        self.register_io_variable(self.flow_vol, role="input")
        self.register_io_variable(electrical_power, role="output")
