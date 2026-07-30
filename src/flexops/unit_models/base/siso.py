"""SISOBlock: the single-inlet/single-outlet IO-topology base (architecture §3.4).

The first of the IO-topology base classes: owns port construction (via the
inherited :meth:`~flexops.core.ops_block.OpsBlockData.add_stream_ports`) and
the per-stream mass balance, so physical subclasses
(:class:`~flexops.unit_models.pump.Pump`,
:class:`~flexops.unit_models.storage_tank.Tank`) only add the
flow<->energy relationship (or, when their flows genuinely differ, replace
the mass balance -- see :meth:`SISOBlockData._build_mass_balance`). Registers
no power itself; a bare ``SISOBlock`` declares neither ``power_electrical``
nor ``power_thermal``.

The pass-through balance is built via the inherited
:meth:`~flexops.core.ops_block.OpsBlockData.add_pass_through_constraints`: every
state variable the inlet/outlet ports expose (flow included) flows straight
through. ``SISOBlock`` overrides the base ``allow_pass_through`` default to
``True`` so the topology is well-posed (DoF == 0) out of the box; pass
``allow_pass_through=False`` to leave the state variables unlinked and wire a
custom relationship instead.
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

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)

    def build(self) -> None:
        """Build the inlet/outlet ports and the per-stream mass balance."""
        super().build()
        self.add_stream_ports()
        self._build_mass_balance()

    def _build_mass_balance(self) -> None:
        """Per-stream pass-through balance: every inlet state var equals outlet.

        Delegates with no exclusions to
        :meth:`~flexops.core.ops_block.OpsBlockData.add_pass_through_constraints`
        -- flow's pass-through *is* a pass-through equality here.
        Subclasses whose flow genuinely differs (e.g. ``Tank``'s
        holdup) override this instead of building a second, conflicting
        balance alongside the inherited ports (never re-declare the ports or
        the balance in a subclass).
        """
        self.add_pass_through_constraints(self.inlet, self.outlet, exclude_vars=())
