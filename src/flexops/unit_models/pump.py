"""``Pump``: constant-energy-intensity SISO unit (architecture §3.4, LP)."""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.nomenclature import PowerKind
from flexops.unit_models.base.siso import SISOBlockData


@declare_process_block_class("Pump")
class PumpData(SISOBlockData):
    """A pump with a constant electrical energy intensity (LP).

    Inherits the inlet/outlet ports, the pass-through mass balance, and the
    ``flow_vol`` handle from :class:`~flexops.unit_models.base.siso.SISOBlockData`
    and adds only the electrical-work relationship:

    .. math::

        \\text{electrical\\_power}[t] = e \\cdot \\text{flow\\_vol}[t]

    where :math:`e` is the ``energy_intensity`` config (a mutable, regressable
    Param, default 0.5 kWh/m³). **Unit algebra:** ``flow_vol`` is m³/hr, so
    kWh/m³ × m³/hr = kW — dimensionally exact with no conversion factor.

    Config: the SISO/OpsBlock config plus ``energy_intensity``.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Pump
        >>> m = dummy_time_block(3)
        >>> m.pump = Pump(property_package=m.properties, energy_intensity=0.7)
    """

    CONFIG = SISOBlockData.CONFIG()
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.5,
            domain=float,
            description="Electrical energy intensity in kWh per m^3 pumped "
            "(default 0.5, an ordinary transfer-pump duty).",
        ),
    )

    def build(self) -> None:
        """Build the SISO base, the intensity Param, and the power relation."""
        super().build()
        time_block = self._find_time_block()

        self.energy_intensity = pyo.Param(
            initialize=self.config.energy_intensity,
            mutable=True,
            units=pyunits.kWh / pyunits.m**3,
            doc="Electrical energy intensity per unit volume pumped",
        )
        self.register_process_parameter(self.energy_intensity, regressable=True)

        electrical_power = self.declare_power(PowerKind.ELECTRICAL)

        @self.Constraint(
            time_block.time_index,
            doc="Electrical draw: electrical_power[t] = energy_intensity * "
            "flow_vol[t]; kWh/m^3 x m^3/hr = kW, dimensionally exact with no "
            "conversion factor",
        )
        def electrical_power_eq(blk, t):
            return electrical_power[t] == blk.energy_intensity * blk.flow_vol[t]

        self.register_io_variable(self.flow_vol, role="input")
        self.register_io_variable(electrical_power, role="output")
