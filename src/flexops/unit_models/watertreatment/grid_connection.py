"""GridConnection(OpsBlockData): grid import and export with per-period limits."""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.config.schema import UnitCommitmentConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.core.plant_block import PlantBlockData
from flexops.unit_models.base._boundary import add_time_limits


@declare_process_block_class("GridConnection")
class GridConnectionData(OpsBlockData):
    """Splits its plant's net electrical draw into non-negative grid import and export.

    Config:
        ``min_import`` / ``max_import`` / ``min_export`` / ``max_export``
        (default ``None``, unlimited): per-period limits in power units, each a
        scalar, a length-T sequence, or a time-indexed Pyomo component.

    Raises:
        FlexConfigError: If the unit is not built directly on a ``PlantBlock``,
            or a limit has the wrong units or length.

    Example:
        >>> import flexops as fo
        >>> from pyomo.environ import units as pyunits
        >>> from flexops.testing import dummy_time_block
        >>> m = dummy_time_block(3)
        >>> m.plant = fo.PlantBlock(time_block=m.time_block)
        >>> m.plant.grid = fo.GridConnection(  # doctest: +SKIP
        ...     max_import=5 * pyunits.MW,
        ...     max_export=[1 * pyunits.MW, 2 * pyunits.MW, 0 * pyunits.MW],
        ... )
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("unit_commitment").set_default_value(UnitCommitmentConfig(status=False))
    CONFIG.declare(
        "min_import",
        ConfigValue(
            default=None,
            description="Per-period lower limit on grid_import (e.g. a contracted "
            "minimum take): a power scalar, a length-T series, or a time-indexed "
            "Pyomo component. None means no limit.",
        ),
    )
    CONFIG.declare(
        "max_import",
        ConfigValue(
            default=None,
            description="Per-period upper limit on grid_import (the interconnection's "
            "maximum net import): a power scalar, a length-T series, or a "
            "time-indexed Pyomo component. None means no limit.",
        ),
    )
    CONFIG.declare(
        "min_export",
        ConfigValue(
            default=None,
            description="Per-period lower limit on grid_export: a power scalar, a "
            "length-T series, or a time-indexed Pyomo component. None means no "
            "limit.",
        ),
    )
    CONFIG.declare(
        "max_export",
        ConfigValue(
            default=None,
            description="Per-period upper limit on grid_export (the interconnection's "
            "maximum net export): a power scalar, a length-T series, or a "
            "time-indexed Pyomo component. None means no limit.",
        ),
    )

    def build(self) -> None:
        """Build the import/export Vars, the net-power balance, and the limits."""
        super().build()
        plant = self.parent_block()
        if not isinstance(plant, PlantBlockData):
            raise FlexConfigError(
                f"GridConnection {self.name!r} must be built directly on a "
                "PlantBlock, since it balances that plant's net electrical draw.",
                field="parent",
                value=None if plant is None else plant.name,
            )
        # Creates the plant total now; cost_process() refreshes its body in place.
        plant._build_aggregates()
        net_power = plant.component(nm.TOTAL_POWER_VARS[nm.PowerKind.ELECTRICAL][0])
        tb = self._find_time_block()

        self.grid_import = pyo.Var(
            tb.time_index,
            bounds=(0.0, None),
            initialize=0.0,
            units=pyunits.kW,
            doc="Electrical power imported from the grid (kW).",
        )
        self.grid_export = pyo.Var(
            tb.time_index,
            bounds=(0.0, None),
            initialize=0.0,
            units=pyunits.kW,
            doc="Electrical power exported to the grid (kW).",
        )
        self.register_io_variable(self.grid_import, role="output")
        self.register_io_variable(self.grid_export, role="output")

        @self.Constraint(
            tb.time_index,
            doc="grid_import - grid_export equals the plant's net electrical draw.",
        )
        def eq_net_power(b, t):
            return b.grid_import[t] - b.grid_export[t] == net_power[t]

        add_time_limits(
            self,
            self.grid_import,
            "grid_import",
            lower=self.config.min_import,
            upper=self.config.max_import,
            fields=("min_import", "max_import"),
        )
        add_time_limits(
            self,
            self.grid_export,
            "grid_export",
            lower=self.config.min_export,
            upper=self.config.max_export,
            fields=("min_export", "max_export"),
        )
