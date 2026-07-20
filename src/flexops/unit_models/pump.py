"""Pump(SISOBlock): constant electrical energy-intensity pump (architecture §3.4).

Inherits its inlet/outlet ports and pass-through mass balance from
:class:`~flexops.unit_models.base.siso.SISOBlockData`; adds only the
flow-to-power relationship.

.. todo::
    Post-v0: add a detailed pump power law (e.g.
    ``power ~ density * flowrate * head / efficiency``) as an alternative to
    the constant energy-intensity relationship below. Not implemented here --
    this is a placeholder for future work, not a spec for this milestone.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexops.unit_models.base.siso import SISOBlockData


@declare_process_block_class("Pump")
class PumpData(SISOBlockData):
    r"""A pump with a constant electrical energy intensity.

    .. math::

        P_{elec}[t] = \text{energy\_intensity} \cdot \dot{V}_{in}[t]

    ``energy_intensity`` is in kWh/m^3 and the inlet ``flow_vol_phase`` in
    m^3/hr, so kWh/m^3 * m^3/hr = kWh/hr = kW -- dimensionally exact with no
    fudge factor; the ``pyunits.convert`` to kW below applies a factor of 1.

    Config:
        Inherits the SISO/OpsBlock config; adds ``energy_intensity``
        (default 0.5 kWh/m^3).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Pump
        >>> m = dummy_time_block(3)
        >>> m.pump = Pump(property_package=m.properties)  # doctest: +SKIP
    """

    CONFIG = SISOBlockData.CONFIG()
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.5 * pyunits.kWh / pyunits.m**3,
            description="Electrical energy per unit volume pumped (a mutable "
            "Param once built), kWh/m^3.",
        ),
    )

    def build(self) -> None:
        """Build the SISO base, then the energy_intensity Param and power draw."""
        super().build()
        tb = self._find_time_block()

        self.energy_intensity = pyo.Param(
            initialize=pyo.value(
                pyunits.convert(
                    self.config.energy_intensity, pyunits.kWh / pyunits.m**3
                )
            ),
            mutable=True,
            units=pyunits.kWh / pyunits.m**3,
            doc="Electrical energy per unit volume pumped.",
        )
        self.register_process_parameter(self.energy_intensity, regressable=True)

        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.register_io_variable(power, role="output")

        @self.Constraint(
            tb.time_index,
            doc="power_electrical = energy_intensity * inlet flow; "
            "kWh/m^3 * m^3/hr = kW exactly, no fudge factor.",
        )
        def power_eq(b, t):
            return power[t] == pyunits.convert(
                b.energy_intensity * b.inlet_state.flow_vol_phase[t, "Liq"],
                pyunits.kW,
            )
