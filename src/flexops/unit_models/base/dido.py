"""DIDOBlock: the double-inlet/double-outlet IO-topology base.

The third IO-topology base class: two inlet and two outlet ports, with the two
per-stream mass balances **coupled** by a single transfer term — the fraction of
stream a's feed that crosses over into stream b:

.. math::

    \\dot{V}_{out,a}[t] &= (1 - f) \\cdot \\dot{V}_{in,a}[t] \\\\
    \\dot{V}_{out,b}[t] &= \\dot{V}_{in,b}[t] + f \\cdot \\dot{V}_{in,a}[t]

``transfer_fraction`` (:math:`f`) is a fixed, regressable scalar Var, so both
balances stay linear and the coupling is the natural regression target. Setting
it to zero makes the two streams independent pass-throughs (a pure heat
exchanger); leaving it free of a physical interpretation is the point — physical
subclasses (:class:`~flexops.unit_models.base.exchanger.Exchanger` and what derives
from it) add the flow-to-energy relationship on top. Registers no power itself.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexops.core.ops_block import OpsBlockData


@declare_process_block_class("DIDOBlock")
class DIDOBlockData(OpsBlockData):
    """Two inlet, two outlet ports with coupled mass balances (module docstring).

    Config:
        Inherits the OpsBlock config; adds ``transfer_fraction`` (default 0.0).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models.base import DIDOBlock
        >>> m = dummy_time_block(3)
        >>> m.unit = DIDOBlock(property_package=m.properties)  # doctest: +SKIP
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "transfer_fraction",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Fraction of stream a's inlet flow that crosses into "
            "stream b (a fixed, regressable Var once built). 0.0 leaves the two "
            "streams independent.",
        ),
    )

    _component_names = {
        "flow_in_a": "flow_in_a",
        "flow_in_b": "flow_in_b",
        "flow_out_a": "flow_out_a",
        "flow_out_b": "flow_out_b",
        "transfer_fraction": "transfer_fraction",
    }

    def build(self, naming_dict: dict[str, str] | None = None) -> None:
        """Build the two-inlet/two-outlet ports and the coupled mass balances.

        Args:
            naming_dict: Complete role -> component name mapping for this unit,
                passed up by a physical subclass renaming the generic roles
                (spread ``DIDOBlockData._component_names`` and override the
                subset it renames). None uses this topology's own vocabulary.
        """
        super().build()
        self.create_stream_naming_convention(naming_dict or self._component_names)
        self.add_stream_ports(
            inlet_ports=("inlet_a", "inlet_b"), outlet_ports=("outlet_a", "outlet_b")
        )
        self._build_mass_balance()

    def _build_mass_balance(self) -> None:
        """Build ``transfer_fraction`` and the two coupled per-stream balances."""
        tb = self._find_time_block()
        for port, role in (
            ("inlet_a", "flow_in_a"),
            ("inlet_b", "flow_in_b"),
            ("outlet_a", "flow_out_a"),
            ("outlet_b", "flow_out_b"),
        ):
            state = self.find_component(f"{port}_state")
            self.add_component(
                self._named(role), pyo.Reference(state.flow_vol_phase[:, "Liq"])
            )
        flow_in_a = self.find_component(self._named("flow_in_a"))
        flow_in_b = self.find_component(self._named("flow_in_b"))
        flow_out_a = self.find_component(self._named("flow_out_a"))
        flow_out_b = self.find_component(self._named("flow_out_b"))

        transfer = self.declare_process_parameter(
            self._named("transfer_fraction"),
            self.config.transfer_fraction,
            pyunits.dimensionless,
            "Fraction of stream a's inlet flow crossing into stream b. "
            "Fixed at the configured value; FlexParameterize may regress it.",
            bounds=(0.0, 1.0),
        )

        @self.Constraint(
            tb.time_index,
            doc="Stream a balance: outlet_a == (1 - transfer_fraction) * inlet_a.",
        )
        def mass_balance_a(b, t):
            return flow_out_a[t] == (1 - transfer) * flow_in_a[t]

        @self.Constraint(
            tb.time_index,
            doc="Stream b balance: outlet_b == inlet_b + transfer_fraction * inlet_a.",
        )
        def mass_balance_b(b, t):
            return flow_out_b[t] == flow_in_b[t] + transfer * flow_in_a[t]

        # Flow is governed above; each stream passes everything else through.
        flow_name = self.config.property_package.get_flow_basis_var_name()
        for suffix in ("a", "b"):
            self.add_pass_through_constraints(
                self.find_component(f"inlet_{suffix}"),
                self.find_component(f"outlet_{suffix}"),
                exclude_vars=[flow_name],
                name_prefix=f"pass_through_{suffix}",
            )
