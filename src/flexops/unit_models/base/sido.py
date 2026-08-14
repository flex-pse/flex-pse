"""SIDOBlock: the single-inlet/double-outlet (split) IO-topology base (§3.4).

The second IO-topology base class: owns port construction (via the inherited
:meth:`~flexops.core.ops_block.OpsBlockData.add_stream_ports`) and the split
mass balance, so a physical subclass only adds the flow-to-energy
relationship. Registers no power itself.

**The mass balance is two constraints.** Conservation
``flow_in[t] == flow_out_a[t] + flow_out_b[t]`` alone leaves the split
undetermined, so the base also fixes *where* the feed goes with
``flow_out_a[t] == split_fraction * flow_in[t]``. ``split_fraction`` is a fixed,
regressable scalar Var (not time-indexed), which keeps both constraints linear —
the topology stays LP-representable — and makes the split the natural regression
target for FlexParameterize. Everything a stream carries other than flow (e.g.
pressure/temperature, when the property package has them) passes straight from
the inlet to **both** outlets.

``split_definition`` (determining ``flow_out_a``) is registered as a swappable
relation (see
:meth:`~flexops.core.ops_block.OpsBlockData.register_relation`) — a richer
recovery/flux model (a function of feed pressure, temperature, fouling) can
replace it in place via
:meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`. Conservation
(``split_mass_balance``) is deliberately **not** registered and so can never be
swapped.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexops.core.ops_block import OpsBlockData


@declare_process_block_class("SIDOBlock")
class SIDOBlockData(OpsBlockData):
    """One inlet, two outlet ports with a split mass balance (module docstring).

    Config:
        Inherits the OpsBlock config; adds ``split_fraction`` (default 0.5),
        bounded here only to a physical fraction. A physical subclass may
        rename the components (via ``build(naming_dict=...)``), rename the
        option itself, and narrow that window (see
        :meth:`_split_parameter_value` and :meth:`_split_parameter_bounds`).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models.base import SIDOBlock
        >>> m = dummy_time_block(3)
        >>> m.unit = SIDOBlock(property_package=m.properties)  # doctest: +SKIP
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "split_fraction",
        ConfigValue(
            default=0.5,
            domain=float,
            description="Fraction of the inlet flow leaving through outlet_a "
            "(a fixed, regressable Var once built); the remainder leaves "
            "through outlet_b.",
        ),
    )

    _component_names = {
        "flow_in": "flow_in",
        "flow_out_a": "flow_out_a",
        "flow_out_b": "flow_out_b",
        "split_fraction": "split_fraction",
    }

    def build(self, naming_dict: dict[str, str] | None = None) -> None:
        """Build the inlet/two-outlet ports and the split mass balance.

        Args:
            naming_dict: Complete role -> component name mapping for this unit,
                passed up by a physical subclass renaming the generic roles
                (spread ``SIDOBlockData._component_names`` and override the
                subset it renames, as
                :class:`~flexops.unit_models.reverseosmosis.ReverseOsmosis`
                does). None uses this topology's own vocabulary.
        """
        super().build()
        self.create_stream_naming_convention(naming_dict or self._component_names)
        self.add_stream_ports(outlet_ports=("outlet_a", "outlet_b"))
        self._build_mass_balance()

    def _split_parameter_bounds(self) -> tuple[float | None, float | None]:
        """Return the bounds for the split-parameter Var.

        The generic topology admits any physical fraction. A subclass that
        exposes a configurable window overrides this to narrow it, and is the
        right place to reject an unusable window before the Var is built.

        Returns:
            The ``(lower, upper)`` bounds of the split parameter.
        """
        return (0.0, 1.0)

    def _split_parameter_value(self) -> float:
        """Return the configured value of the split parameter.

        Read through a method because ``component_names`` renames the Pyomo
        component only — the config option carrying the value is a separate,
        class-fixed name, which a subclass that renames the option itself (as
        ``ReverseOsmosis`` does with ``recovery``) overrides here.

        Returns:
            The configured split fraction.
        """
        return self.config.split_fraction

    def _build_mass_balance(self) -> None:
        """Build the split parameter, the split definition, and conservation."""
        tb = self._find_time_block()
        self.add_component(
            self._named("flow_in"),
            pyo.Reference(self.inlet_state.flow_vol_phase[:, "Liq"]),
        )
        self.add_component(
            self._named("flow_out_a"),
            pyo.Reference(self.outlet_a_state.flow_vol_phase[:, "Liq"]),
        )
        self.add_component(
            self._named("flow_out_b"),
            pyo.Reference(self.outlet_b_state.flow_vol_phase[:, "Liq"]),
        )
        flow_in = self.find_component(self._named("flow_in"))
        flow_out_a = self.find_component(self._named("flow_out_a"))
        flow_out_b = self.find_component(self._named("flow_out_b"))

        name = self._named("split_fraction")
        split = self.declare_process_parameter(
            name,
            self._split_parameter_value(),
            pyunits.dimensionless,
            f"Fraction of the inlet flow leaving through outlet_a ({name}). "
            "Fixed at the configured value; FlexParameterize may regress it.",
            bounds=self._split_parameter_bounds(),
        )

        @self.Constraint(
            tb.time_index,
            doc=f"Split definition: outlet_a flow == {name} * inlet flow.",
        )
        def split_definition(b, t):
            return flow_out_a[t] == split * flow_in[t]

        self.register_relation(self.split_definition, target=flow_out_a)

        @self.Constraint(
            tb.time_index,
            doc="Conservation: inlet flow == outlet_a flow + outlet_b flow.",
        )
        def split_mass_balance(b, t):
            return flow_in[t] == flow_out_a[t] + flow_out_b[t]

        # Flow is governed above; everything else the streams carry passes
        # through to BOTH outlets (a distinct name_prefix per outlet, or the
        # two calls would collide on one component name).
        flow_name = self.config.property_package.get_flow_basis_var_name()
        for suffix, outlet in (("a", self.outlet_a), ("b", self.outlet_b)):
            self.add_pass_through_constraints(
                self.inlet,
                outlet,
                exclude_vars=[flow_name],
                name_prefix=f"pass_through_{suffix}",
            )
