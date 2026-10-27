"""SpaceCooling(OpsBlockData): a cooling load met at a COP plus ancillary draw."""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.config.schema import UnitCommitmentConfig
from flexops.core.ops_block import OpsBlockData


@declare_process_block_class("SpaceCooling")
class SpaceCoolingData(OpsBlockData):
    """A cooling unit whose draw is its load over a COP plus an ancillary share.

    Config:
        Inherits the OpsBlock config; each added option is documented on its
        ``CONFIG`` entry.
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("unit_commitment").set_default_value(UnitCommitmentConfig(status=False))
    CONFIG.declare(
        "cop",
        ConfigValue(
            default=3.0,
            domain=float,
            description="Coefficient of performance: heat removed per unit of "
            "compressor power.",
        ),
    )
    CONFIG.declare(
        "ancillary_intensity",
        ConfigValue(
            default=0.05,
            domain=float,
            description="Fan and pump power per kW of cooling load.",
        ),
    )

    def build(self) -> None:
        """Build the cooling load input, the COP and the electrical draw."""
        super().build()
        tb = self._find_time_block()
        self.cooling_load = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Step-mean heat the unit removes.",
        )
        self.register_io_variable(self.cooling_load, role="input")
        self.declare_process_parameter(
            "cop",
            self.config.cop,
            pyunits.dimensionless,
            "Coefficient of performance.",
            bounds=(0.1, None),
        )
        self.declare_process_parameter(
            "ancillary_intensity",
            self.config.ancillary_intensity,
            pyunits.dimensionless,
            "Fan and pump power per kW of cooling load.",
            bounds=(0, 1),
        )
        self.power_compressor = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Step-mean compressor power.",
        )
        self.power_compressor_relation = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: b.power_compressor[t] * b.cop == b.cooling_load[t],
            doc="Compressor power * COP == cooling load.",
        )
        self.register_relation(
            self.power_compressor_relation, target=self.power_compressor
        )
        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.register_io_variable(power, role="output")
        self.power_electrical_definition = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: power[t]
            == b.power_compressor[t] + b.ancillary_intensity * b.cooling_load[t],
            doc="Draw: compressor power plus ancillary_intensity * cooling load.",
        )
