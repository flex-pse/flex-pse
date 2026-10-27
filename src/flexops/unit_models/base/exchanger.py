r"""Exchanger(DIDOBlock): two coupled streams exchanging mass/energy.

Inherits its two inlet/two outlet ports, ``transfer_fraction``, and the coupled
per-stream mass balances from
:class:`~flexops.unit_models.base.dido.DIDOBlockData`; adds only the electrical
draw (auxiliaries — pumps, fans, controls — scaling with the primary stream).
This could eventually become a template for heat exchanger or flow-through
electrolyzer.
"""

from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexops.unit_models.base.dido import DIDOBlockData


@declare_process_block_class("Exchanger")
class ExchangerData(DIDOBlockData):
    r"""An exchanger: two coupled streams, constant electrical intensity.

    .. math::

        \dot{V}_{out,a}[t] &= (1 - f) \cdot \dot{V}_{in,a}[t] \\
        \dot{V}_{out,b}[t] &= \dot{V}_{in,b}[t] + f \cdot \dot{V}_{in,a}[t] \\
        P_{elec}[t] &= \text{energy\_intensity} \cdot \dot{V}_{in,a}[t]

    with :math:`f` the ``transfer_fraction``. The energy relation is the
    Constraint ``power_electrical_relation`` (the swap contract).

    Config:
        Inherits the DIDO/OpsBlock config (``transfer_fraction``); adds
        ``energy_intensity`` (default 0.1 kWh/m^3).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Exchanger
        >>> m = dummy_time_block(3)
        >>> m.unit = Exchanger(property_package=m.properties)  # doctest: +SKIP
    """

    CONFIG = DIDOBlockData.CONFIG()
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.1 * pyunits.kWh / pyunits.m**3,
            description="Electrical energy per unit volume delivered at outlet "
            "a (a fixed, regressable Var once built), kWh/m^3.",
        ),
    )

    def build(self) -> None:
        """Build the DIDO base, then the constant-intensity electrical relation."""
        super().build()
        self.add_constant_intensity_relation(
            self.flow_out_a,
            kind=nm.PowerKind.ELECTRICAL,
            intensity=self.config.energy_intensity,
        )
