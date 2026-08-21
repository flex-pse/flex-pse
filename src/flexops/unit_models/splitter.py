r"""Splitter(OpsBlockData): one inlet fanned out to N named outlets (§3.2, §3.4).

The mirror of :class:`~flexops.unit_models.mixer.Mixer`: one stream divided
among an arbitrary number of outlets. Its port count is a config option, so no
fixed-arity IO-topology base fits — it subclasses
:class:`~flexops.core.ops_block.OpsBlockData` directly and hand-writes its ports
and balance.

**Density is constant across the junction**, so volume is conserved directly
and the whole flow model is one linear equality:

.. math::

    \dot{V}_{in}[t] = \sum_i \dot{V}_{out,i}[t]

There is no energy input: a splitter declares neither ``power_electrical`` nor
``power_thermal``, and registers no power.

**The split is a decision, not a parameter.** Conservation alone leaves
:math:`(N-1)` free outlet flows per time point, and that is deliberate: routing
is exactly the kind of operational freedom this platform exists to optimize
over, so the enclosing model's objective (or a caller fixing outlet flows)
picks the split. A unit whose split is instead *prescribed* is a different
model: use :class:`~flexops.unit_models.base.sido.SIDOBlock`, whose fixed and
regressable ``split_fraction`` closes those degrees of freedom, or
:class:`~flexops.unit_models.reverseosmosis.ReverseOsmosis` on top of it.

Every stream gets its **own port**. Pyomo's ``Port`` members are equalities
here, not extensive quantities apportioned across several arcs, so N outlets
means N ports each carrying exactly one arc — never one outlet port feeding N
arcs.

Everything the stream carries other than flow (pressure and temperature, when
the property package has them) passes straight from the inlet to **every**
outlet: dividing a stream changes how much goes each way, not what it is.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue

from flexops.core.ops_block import OpsBlockData
from flexops.unit_models._multiport import single_flow_phase, validate_port_names


def _outlet_names_domain(value) -> tuple[str, ...]:
    """ConfigValue domain: coerce to a tuple.

    Only coerces the type; emptiness/uniqueness/non-empty-string checks happen
    in ``build()`` via
    :func:`~flexops.unit_models._multiport.validate_port_names`, so they raise
    :class:`~flexcore.exceptions.FlexConfigError` directly rather than the
    ``ValueError`` Pyomo's ``ConfigValue`` wraps every domain-raised exception
    into.
    """
    return tuple(value)


@declare_process_block_class("Splitter")
class SplitterData(OpsBlockData):
    r"""One inlet stream divided among N named outlets, at constant density.

    See the module docstring for the balance and, in particular, for why this
    unit is deliberately left with :math:`(N-1)` degrees of freedom per time
    point. ``outlet_names`` sets the outlet count and their port names
    (``f"outlet_{name}"``).

    Config:
        ``property_package`` (inherited): a single-phase package shared by
        every port. ``outlet_names`` (default ``("a", "b")``): the outlets'
        role/port names, which must be unique and non-empty.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Splitter
        >>> m = dummy_time_block(3)
        >>> m.junction = Splitter(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     outlet_names=("treatment", "bypass"),
        ... )
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("allow_pass_through").set_default_value(True)
    CONFIG.declare(
        "outlet_names",
        ConfigValue(
            default=("a", "b"),
            domain=_outlet_names_domain,
            description="Role names of the splitter's outlets; outlet i is "
            "built as port f'outlet_{name}'. Must be unique and non-empty. No "
            "split fraction accompanies them: conservation is the only flow "
            "constraint, so the routing stays a decision variable.",
        ),
    )

    def build(self) -> None:
        """Validate the config, then build the ports and the split balance."""
        super().build()
        validate_port_names(self.config.outlet_names, "outlet_names")
        self._phase = single_flow_phase(self.config.property_package, "Splitter")
        self.add_stream_ports(
            inlet_ports=("inlet",), outlet_ports=self._outlet_port_names()
        )
        self._register_stream_states()
        self._build_mass_balance()

    # -- config resolution --------------------------------------------------

    def _outlet_port_names(self) -> tuple[str, ...]:
        """Return the ``f"outlet_{name}"`` port names, in ``outlet_names`` order."""
        return tuple(f"outlet_{name}" for name in self.config.outlet_names)

    def _flow_basis_name(self) -> str:
        """Return the property package's extensive flow state-variable name."""
        return self.config.property_package.get_flow_basis_var_name()

    def _outlet_state(self, name: str):
        """Return the state block behind the outlet named ``name``."""
        return self.find_component(f"outlet_{name}_state")

    # -- ports and balance --------------------------------------------------

    def _register_stream_states(self) -> None:
        """Register the non-flow states beyond the flows ``add_stream_ports`` did.

        The inlet's intensive states are the junction's boundary conditions;
        every outlet's are results of the pass-through equalities.
        """
        flow_name = self._flow_basis_name()
        for name, var in self.inlet_state.define_state_vars().items():
            if name != flow_name:
                self.register_io_variable(var, role="input")
        for outlet_name in self.config.outlet_names:
            state = self._outlet_state(outlet_name)
            for name, var in state.define_state_vars().items():
                if name != flow_name:
                    self.register_io_variable(var, role="output")

    def _build_mass_balance(self) -> None:
        """Build the flow References, conservation, and the per-outlet pass-through."""
        tb = self._find_time_block()
        flow_name = self._flow_basis_name()
        outlet_names = self.config.outlet_names

        self.add_component(
            "flow_in",
            pyo.Reference(self.inlet_state.find_component(flow_name)[:, self._phase]),
        )
        flow_in = self.flow_in
        flows = {}
        for name in outlet_names:
            state = self._outlet_state(name)
            self.add_component(
                f"flow_out_{name}",
                pyo.Reference(state.find_component(flow_name)[:, self._phase]),
            )
            flows[name] = self.find_component(f"flow_out_{name}")

        @self.Constraint(
            tb.time_index,
            doc="Conservation: inlet flow == the sum of the outlet flows. This "
            "is the only flow constraint — no split fractions — so the routing "
            "stays a decision the enclosing model's objective makes.",
        )
        def split_mass_balance(b, t):
            return flow_in[t] == sum(flows[name][t] for name in outlet_names)

        # Flow is governed above; everything else the stream carries passes
        # through to EVERY outlet (a distinct name_prefix per outlet, or the
        # calls would collide on one component name).
        for name in outlet_names:
            self.add_pass_through_constraints(
                self.inlet,
                self.find_component(f"outlet_{name}"),
                exclude_vars=[flow_name],
                name_prefix=f"pass_through_{name}",
            )
