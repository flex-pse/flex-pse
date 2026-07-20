"""SISOBlock: the single-inlet/single-outlet IO-topology base (architecture §3.4, R6).

The first of the IO-topology base classes: owns port construction (via the
inherited :meth:`~flexops.core.ops_block.OpsBlockData.add_stream_ports`) and
the per-stream mass balance, so physical subclasses
(:class:`~flexops.unit_models.pump.Pump`,
:class:`~flexops.unit_models.storage_tank.StorageTank`) only add the
flow<->energy relationship (or, when their flows genuinely differ, replace
the mass balance -- see :meth:`SISOBlockData._build_mass_balance`). Registers
no power itself; a bare ``SISOBlock`` declares neither ``power_electrical``
nor ``power_thermal``.
"""

from idaes.core import declare_process_block_class

from flexops.core.ops_block import OpsBlockData


@declare_process_block_class("SISOBlock")
class SISOBlockData(OpsBlockData):
    """One inlet, one outlet ``SimpleAqueousFlow`` port pair with a pass-through
    balance.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models.base import SISOBlock
        >>> m = dummy_time_block(3)
        >>> m.unit = SISOBlock(property_package=m.properties)  # doctest: +SKIP
    """

    def build(self) -> None:
        """Build the inlet/outlet ports and the per-stream mass balance."""
        super().build()
        self.add_stream_ports()
        self._build_mass_balance()

    def _build_mass_balance(self) -> None:
        """Per-stream pass-through balance: outlet flow equals inlet flow.

        Subclasses whose flows genuinely differ (e.g. ``StorageTank``'s
        holdup) override this instead of building a second, conflicting
        balance alongside the inherited ports (never re-declare the ports or
        the balance in a subclass).
        """
        tb = self._find_time_block()

        @self.Constraint(
            tb.time_index,
            doc="Per-stream mass balance: outlet flow equals inlet flow.",
        )
        def mass_balance(b, t):
            return (
                b.outlet_state.flow_vol_phase[t, "Liq"]
                == b.inlet_state.flow_vol_phase[t, "Liq"]
            )
