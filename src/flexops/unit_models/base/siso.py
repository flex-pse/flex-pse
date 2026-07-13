"""``SISOBlock``: the single-inlet/single-outlet IO-topology base (§3.4, R6).

The first of the IO-topology base classes. It owns port construction (inlet and
outlet built from ``SimpleAqueousFlow``-style state blocks — no ControlVolumes,
decision R1), the per-stream mass balance, and the energy-registration wiring
inherited from :class:`~flexops.core.ops_block.OpsBlockData`. Physical
subclasses (``Pump``, ``StorageTank``) add only the flow↔energy relationship
and any bounds; ``SISOBlock`` itself registers no energy.

Logic/unit-commitment (the base ``status`` capability, §3.5) is available via
the inherited ``unit_commitment`` config but not built here; a physical
subclass may disable it entirely (``StorageTank`` — the canonical R6 example).
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class

from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.core.time_block import TimeBlockData


@declare_process_block_class("SISOBlock")
class SISOBlockData(OpsBlockData):
    """Single-inlet/single-outlet topology base for flex-pse unit models.

    Builds one ``inlet`` and one ``outlet`` Port from the configured property
    package's state blocks (indexed by the model's time points), the
    per-stream pass-through mass balance, and the convenience ``flow_vol``
    Reference to the inlet state's flow. Subclasses that need a different
    balance (e.g. the tank's holdup difference equation) override
    :meth:`_build_mass_balance`.

    Config: the base OpsBlock config (``unit_commitment``, ``relaxation``,
    ``allow_bypass``, ``external_dispatch``) plus a required
    ``property_package``.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models.base import SISOBlock
        >>> m = dummy_time_block(3)
        >>> m.unit = SISOBlock(property_package=m.properties)
    """

    CONFIG = OpsBlockData.CONFIG()

    def build(self) -> None:
        """Build the inlet/outlet state blocks, ports, and mass balance.

        Raises:
            FlexConfigError: If no ``property_package`` was configured.
        """
        super().build()
        if self.config.property_package is None:
            raise FlexConfigError(
                f"{type(self).__name__} requires a property_package (e.g. "
                "SimpleAqueousFlow); pass property_package=m.properties.",
                field="property_package",
            )
        time_block = self._find_time_block()

        self.properties_in = self.config.property_package.build_state_block(
            time_block.time_index, doc="Inlet stream state"
        )
        self.properties_out = self.config.property_package.build_state_block(
            time_block.time_index, doc="Outlet stream state"
        )
        self.add_port(name="inlet", block=self.properties_in, doc="Inlet port")
        self.add_port(name="outlet", block=self.properties_out, doc="Outlet port")

        self.flow_vol = pyo.Reference(self.properties_in[:].flow_vol)
        self.flow_vol.doc = "Volumetric flow through the unit (the inlet flow)"

        self._build_mass_balance(time_block)

    def _build_mass_balance(self, time_block: TimeBlockData) -> None:
        """Write the per-stream pass-through mass balance, indexed by ``t``.

        Subclasses whose inlet and outlet flows differ (e.g. ``StorageTank``,
        which stores the difference) override this instead of deactivating the
        base constraint.

        Args:
            time_block: The model's TimeBlock.
        """

        @self.Constraint(
            time_block.time_index,
            doc="Per-stream mass balance: outlet flow_vol equals inlet "
            "flow_vol at every time point",
        )
        def mass_balance(blk, t):
            return blk.properties_out[t].flow_vol == blk.properties_in[t].flow_vol
