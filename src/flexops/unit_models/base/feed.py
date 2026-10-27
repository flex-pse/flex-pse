r"""Feed(OpsBlockData): a boundary source with no inlets and N named outlets.

This models the supply side of the facility. A ``Feed`` represents a stream
entering the plant from outside: it carries the stream's state (composition,
pressure, temperature — whatever the property package tracks), measures the
**total** amount of resource crossing the boundary, limits that withdrawal
over time if configured to, and can optionally add its cost to opex.

The number of outlets is a config option, so ``Feed`` can't reuse one of the
fixed-inlet/fixed-outlet base classes — it subclasses
:class:`~flexops.core.ops_block.OpsBlockData` directly and builds its own
ports and balance equation, the same way
:class:`~flexops.unit_models.base.splitter.Splitter` does.

The metered withdrawal is just the sum of the flow leaving through every
outlet:

.. math::

    \dot{W}[t] = \sum_i \dot{V}_{out,i}[t]

A feed uses no energy: it declares neither ``power_electrical`` nor
``power_thermal``, and doesn't register any power use.

**One block, one resource.** ``resource_name`` is the key that the
surrounding :class:`~flexops.core.plant_block.PlantBlockData` groups blocks
under when it adds up ``total_feed``. This name doesn't have to match the
Pyomo block's own name: two ``Feed`` blocks with different resource names
show up as two separate rows in ``total_feed``, while two blocks that share a
resource name get summed into one row — this is how you'd model the same
resource entering the plant at two different points. If a plant brings in
several different resources, use one ``Feed`` block per resource, not one
block that tracks several resources at once.

Everything about the stream other than flow — its composition, pressure,
temperature — is treated as the boundary condition. That condition is set on
the **reference** outlet (the first name in ``outlet_names``), and every
other outlet is forced to match it: there's one source, so there's one
condition.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.core.registration import BoundaryKind
from flexops.unit_models.base._boundary import (
    LimitBasis,
    add_horizon_limits,
    add_time_limits,
    resolve_basis,
)
from flexops.unit_models.base._multiport import (
    single_flow_phase,
    validate_port_names,
)


def _outlet_names_domain(value) -> tuple[str, ...]:
    """ConfigValue domain: turn the given value into a tuple.

    This only fixes the type. Checks like "not empty," "no duplicate
    names," or "no blank names" happen later in ``build()``, through
    :func:`~flexops.unit_models.base._multiport.validate_port_names`. Doing it
    there means those checks can raise
    :class:`~flexcore.exceptions.FlexConfigError` directly, instead of the
    plain ``ValueError`` that Pyomo's ``ConfigValue`` wraps every domain
    error into.
    """
    return tuple(value)


@declare_process_block_class("Feed")
class FeedData(OpsBlockData):
    r"""A boundary source: no inlets, N named outlets, one metered withdrawal.

    See the module docstring above for the balance equation, how outlet
    states are handled, and why ``resource_name`` doesn't have to match the
    block name. ``outlet_names`` decides how many outlets there are and what
    their port names will be (``f"outlet_{name}"``). The first name in the
    list is the reference outlet, which carries the boundary conditions.

    Config:
        ``property_package`` (inherited): a single-phase package shared by
        every port.

        ``outlet_names`` (default ``("a",)``): the outlets' names, used as
        their port names. Names must be unique and non-empty.

        ``resource_name`` (default ``None``): the key this block's
        withdrawal is grouped under. Defaults to the block's own Pyomo name.

        ``min_withdrawal`` / ``max_withdrawal`` (default ``None``): scalar
        limits on the withdrawal, with units. Interpreted according to
        ``withdrawal_basis`` (default ``LimitBasis.PERIOD``) as either a
        rate that must hold every period, or a total amount over the whole
        horizon.
        On the period basis a limit may also be a length-T series.

        ``price`` (default ``None``): a flat price per unit withdrawn,
        billed through ``costing_package`` (inherited) — which must be set
        if you use this.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Feed
        >>> m = dummy_time_block(3)
        >>> m.raw_water = Feed(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     outlet_names=("north", "south"),
        ... )
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.declare(
        "outlet_names",
        ConfigValue(
            default=("a",),
            domain=_outlet_names_domain,
            description="Role names of the feed's outlets; outlet i is built "
            "as port f'outlet_{name}'. Must be unique and non-empty. The first "
            "name is the reference outlet, whose non-flow states are the "
            "boundary conditions every other outlet is held at.",
        ),
    )
    CONFIG.declare(
        "resource_name",
        ConfigValue(
            default=None,
            description="The withdrawn resource's name (e.g. 'raw_water'), the "
            "key it aggregates under in the plant's total_feed. Defaults to "
            "the block's own Pyomo name; set it explicitly to override that — "
            "to merge two blocks under one key, or when the block name is not "
            "the name wanted in total_feed.",
        ),
    )
    CONFIG.declare(
        "max_withdrawal",
        ConfigValue(
            default=None,
            description="Units-carrying upper limit on the withdrawal, a scalar or "
            "(on the period basis) a length-T series "
            "(e.g. a permitted abstraction rate, or a monthly permit volume). "
            "Its dimension follows withdrawal_basis: a rate per period, or a "
            "quantity over the horizon. Builds a mutable Param and its "
            "Constraint; rewrite the Param to move the bound.",
        ),
    )
    CONFIG.declare(
        "min_withdrawal",
        ConfigValue(
            default=None,
            description="Units-carrying lower limit on the withdrawal, a scalar or "
            "(on the period basis) a length-T series "
            "(e.g. a take-or-pay minimum). Its dimension follows "
            "withdrawal_basis: a rate per period, or a quantity over the "
            "horizon. Builds a mutable Param and its Constraint.",
        ),
    )
    CONFIG.declare(
        "withdrawal_basis",
        ConfigValue(
            default=LimitBasis.PERIOD,
            description="Whether min_withdrawal/max_withdrawal bind in every "
            "period ('period', the default -- the limits are rates such as "
            "m**3/hr) or on the horizon total ('horizon' -- the limits are "
            "quantities such as m**3, bounding the scalar withdrawal_total and "
            "leaving the optimizer to shape the profile that reaches it). A "
            "limit whose units contradict the basis is rejected. On its own, "
            "with no limit configured, this builds nothing.",
        ),
    )
    CONFIG.declare(
        "price",
        ConfigValue(
            default=None,
            description="Flat price per unit withdrawn, in the costing "
            "currency basis; positive is a cost. Requires costing_package, and "
            "is billed as a scalar operating cost (price * sum_t flow[t] * dt).",
        ),
    )

    def build(self) -> None:
        """Validate the config, then build the ports, withdrawal, limits, and cost."""
        super().build()
        validate_port_names(self.config.outlet_names, "outlet_names")
        basis = resolve_basis(self.config.withdrawal_basis, "withdrawal_basis")
        self._phase = single_flow_phase(self.config.property_package, "Feed")
        self._resource = (
            self.local_name
            if self.config.resource_name is None
            else self.config.resource_name
        )
        self.add_stream_ports(inlet_ports=(), outlet_ports=self._outlet_port_names())
        self._register_stream_states()
        self._tie_outlet_states()
        self._build_withdrawal()
        add_limits = (
            add_horizon_limits if basis is LimitBasis.HORIZON else add_time_limits
        )
        add_limits(
            self,
            self.withdrawal,
            "withdrawal",
            lower=self.config.min_withdrawal,
            upper=self.config.max_withdrawal,
            fields=("min_withdrawal", "max_withdrawal"),
        )
        self.register_boundary_flow(
            self.withdrawal, resource=self._resource, kind=BoundaryKind.FEED
        )
        self._register_cost()

    # -- config resolution --------------------------------------------------

    def _outlet_port_names(self) -> tuple[str, ...]:
        """Return the ``f"outlet_{name}"`` port names, in ``outlet_names`` order."""
        return tuple(f"outlet_{name}" for name in self.config.outlet_names)

    def _flow_basis_name(self) -> str:
        """Return the name of the flow variable that the property package uses."""
        return self.config.property_package.get_flow_basis_var_name()

    def _outlet_state(self, name: str):
        """Return the state block behind the outlet named ``name``."""
        return self.find_component(f"outlet_{name}_state")

    def _flow_units(self):
        """Return the units used by the property package's flow variable."""
        tb = self._find_time_block()
        reference = self._outlet_state(self.config.outlet_names[0])
        flow = reference.find_component(self._flow_basis_name())
        return pyunits.get_units(flow[tb.time_index.first(), self._phase])

    # -- ports, balance, limits ---------------------------------------------

    def _register_stream_states(self) -> None:
        """Register every state besides flow, which ``add_stream_ports`` already
        handled.

        The reference outlet's states (other than flow) are the boundary
        conditions. Every other outlet's states are just copies of the
        reference, enforced by the equality constraints built in
        :meth:`_tie_outlet_states`.
        """
        flow_name = self._flow_basis_name()
        reference_name = self.config.outlet_names[0]
        for outlet_name in self.config.outlet_names:
            role = "input" if outlet_name == reference_name else "output"
            state = self._outlet_state(outlet_name)
            for name, var in state.define_state_vars().items():
                if name != flow_name:
                    self.register_io_variable(var, role=role)

    def _tie_outlet_states(self) -> None:
        """Force every outlet except the reference to match the reference's state.

        A boundary stream only has one composition, one pressure, and one
        temperature — no matter how many outlets it's split across. Without
        this constraint, the extra outlets' states would be free variables
        with nothing pinning them down.
        """
        other_names = self.config.outlet_names[1:]
        if not other_names:
            return
        tb = self._find_time_block()
        flow_name = self._flow_basis_name()
        reference_name = self.config.outlet_names[0]
        tied = [
            name
            for name in self._outlet_state(reference_name).define_state_vars()
            if name != flow_name
        ]
        for state_var in tied:

            def _equality_rule(b, t, name, _v=state_var, _ref=reference_name):
                # Some state-block properties are built lazily (IDAES
                # "on-demand" properties): they're only created the first
                # time you access them through getattr. find_component
                # would return None here because nothing has triggered that
                # creation yet.
                other = getattr(b._outlet_state(name), _v)
                reference = getattr(b._outlet_state(_ref), _v)
                return other[t] == reference[t]

            self.add_component(
                f"outlet_state_equality_{state_var}",
                pyo.Constraint(
                    tb.time_index,
                    other_names,
                    rule=_equality_rule,
                    doc=f"Boundary stream: outlet {state_var} equals the "
                    f"reference outlet's {state_var}.",
                ),
            )

    def _build_withdrawal(self) -> None:
        """Build the per-outlet flow References and the metered withdrawal."""
        tb = self._find_time_block()
        flow_name = self._flow_basis_name()
        outlet_names = self.config.outlet_names

        flows = {}
        for name in outlet_names:
            state = self._outlet_state(name)
            self.add_component(
                f"flow_out_{name}",
                pyo.Reference(state.find_component(flow_name)[:, self._phase]),
            )
            flows[name] = self.find_component(f"flow_out_{name}")

        # This is a Var defined by an equality constraint, not a plain
        # Expression, because the withdrawal needs to support bounds, be
        # fixable through set_external_dispatch, and be usable by costing.
        self.withdrawal = pyo.Var(
            tb.time_index,
            initialize=0.0,
            units=self._flow_units(),
            doc="Total resource withdrawn across the boundary in each period.",
        )

        @self.Constraint(
            tb.time_index,
            doc="Metering: the withdrawal equals the sum of the outlet flows.",
        )
        def eq_withdrawal(b, t):
            return b.withdrawal[t] == sum(flows[name][t] for name in outlet_names)

    def _register_cost(self) -> None:
        """Add the withdrawal's cost to opex, if a price was configured.

        Raises:
            FlexConfigError: If ``price`` was set but no ``costing_package``
                was given to bill it through.
        """
        price = self.config.price
        if price is None:
            return
        costing = self.config.costing_package
        if costing is None:
            raise FlexConfigError(
                f"price={price!r} was set on {self.name!r} but no "
                "costing_package was given, so the cost could not be billed; "
                "pass costing_package=<your FlexCosting block>, or drop price.",
                field="price",
                value=price,
            )
        costing.register_scalar_cost(
            # Inside a plant, the block's Pyomo name contains dots, but this
            # name becomes a component name on the opex block, so the dots
            # must be replaced first.
            name=self.name.replace(".", "_"),
            quantity=self.withdrawal,
            price=price,
            quantity_units=self._flow_units(),
            unit=self,
        )
