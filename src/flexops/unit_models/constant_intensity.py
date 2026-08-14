r"""ConstantEnergyIntensityModel(SISOBlock): the generic surrogate unit.

FlexOps' one generic building block for anything without a bespoke physical
topology — a whole treatment plant modeled as a single surrogate, as in the
api-freeze script's ``waterfacility.plant``. There is deliberately **no** separate
regression unit class: every unit defaults to a constant energy intensity, and
FlexParameterize later upgrades that relationship by swapping this unit's
``power_electrical_relation`` Constraint in place.
"""

from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexops.unit_models.base.siso import SISOBlockData


@declare_process_block_class("ConstantEnergyIntensityModel")
class ConstantEnergyIntensityModelData(SISOBlockData):
    r"""A generic "energy factor times flow" unit.

    Inherits its inlet/outlet ports and pass-through mass balance from
    :class:`~flexops.unit_models.base.siso.SISOBlockData`, and adds only

    .. math::

        P_{elec}[t] = \text{energy\_intensity} \cdot \dot{V}_{out}[t]

    as the Constraint ``power_electrical_relation`` — the **swap contract**
    FlexParameterize deactivates and replaces when it fits a richer
    relationship (see
    :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`). The draw is
    metered on what the unit **delivers**, so the intensity reads as energy per
    unit of product. ``energy_intensity`` is in kWh/m^3 and the outlet
    ``flow_vol_phase`` in m^3/hr, so kWh/m^3 * m^3/hr = kW exactly.

    ``Pump`` stays an independent class rather than a subclass of this one:
    the two diverge as soon as ``Pump`` grows pressure terms, and its
    hydraulic power law is not an energy intensity at all.

    Config:
        Inherits the SISO/OpsBlock config; adds ``energy_intensity`` (default
        0.5 kWh/m^3).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import ConstantEnergyIntensityModel
        >>> m = dummy_time_block(3)
        >>> m.plant = ConstantEnergyIntensityModel(  # doctest: +SKIP
        ...     property_package=m.properties
        ... )
    """

    CONFIG = SISOBlockData.CONFIG()
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.5 * pyunits.kWh / pyunits.m**3,
            description="Electrical energy per unit volume delivered at the "
            "outlet (a fixed, regressable Var once built), kWh/m^3.",
        ),
    )

    def build(self) -> None:
        """Build the SISO base, then the constant-intensity energy relation."""
        super().build()
        self.add_constant_intensity_relation(
            self.flow_out,
            kind=nm.PowerKind.ELECTRICAL,
            intensity=self.config.energy_intensity,
        )
