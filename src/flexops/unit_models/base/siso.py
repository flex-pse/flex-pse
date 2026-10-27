"""SISOBlock: the single-inlet/single-outlet IO-topology base.

The first of the IO-topology base classes: owns port construction (via the
inherited :meth:`~flexops.core.ops_block.OpsBlockData.add_stream_ports`) and
the per-stream mass balance, so physical subclasses
(:class:`~flexops.unit_models.general.pump.Pump`,
:class:`~flexops.unit_models.general.tank.Tank`) only add the
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

import pyomo.environ as pyo
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

    _component_names = {"flow_in": "flow_in", "flow_out": "flow_out"}

    def build(self, naming_dict: dict[str, str] | None = None) -> None:
        """Build the inlet/outlet ports and the per-stream mass balance.

        Args:
            naming_dict: Complete role -> component name mapping for this unit,
                passed up by a physical subclass renaming the generic roles
                (spread ``SISOBlockData._component_names`` and override the
                subset it renames). None uses this topology's own vocabulary.
        """
        super().build()
        self.create_stream_naming_convention(naming_dict or self._component_names)
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

        Also builds ``flow_in``/``flow_out`` (or a subclass's renamed
        equivalents) as named references into the inlet/outlet flow, purely
        for named access -- the pass-through constraint below is what ties
        them together.
        """
        self.add_component(
            self._named("flow_in"),
            pyo.Reference(self.inlet_state.flow_vol_phase[:, "Liq"]),
        )
        self.add_component(
            self._named("flow_out"),
            pyo.Reference(self.outlet_state.flow_vol_phase[:, "Liq"]),
        )
        self.add_pass_through_constraints(self.inlet, self.outlet, exclude_vars=())
