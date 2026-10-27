r"""Product(OpsBlockData): a boundary sink with N named inlets and no outlets.

This is the mirror image of :class:`~flexops.unit_models.base.feed.Feed`, and it
models the delivery side of the facility. A ``Product`` adds up everything
arriving through its inlets, can limit that delivery over time, and can
optionally price it — a positive price is a cost (like paying to dispose of
brine), while a negative price is revenue (like selling potable water).

The number of inlets is a config option, so ``Product`` can't reuse one of
the fixed-inlet/fixed-outlet base classes — it subclasses
:class:`~flexops.core.ops_block.OpsBlockData` directly and builds its own
ports and balance equation.

.. math::

    \dot{D}[t] = \sum_i \dot{V}_{in,i}[t]

A product uses no energy: it declares neither ``power_electrical`` nor
``power_thermal``, and doesn't register any power use.

Two things are deliberately different from ``Feed``:

**It doesn't blend.** Each inlet's state (other than flow) comes from
whatever's connected upstream and is left as-is — there's no constraint
tying the inlets' states together. A ``Product`` only adds up *flow*.
Blending composition, temperature, or pressure together is
:class:`~flexops.unit_models.base.mixer.Mixer`'s job, and that blending math is
bilinear (harder for the solver). If you want one blended stream, put a
``Mixer`` upstream of the ``Product``.

**Limits on composition are set per inlet.** If you want to limit something
about what arrives (like pressure), you limit one specific inlet's state —
there's no blended stream to limit::

    from flexops.unit_models.base._boundary import add_time_limits

    add_time_limits(
        product, product.inlet_a_state.pressure, "inlet_a_pressure",
        upper=200000 * pyunits.Pa,
    )

**One block, one resource.** ``resource_name`` is the key that the
surrounding :class:`~flexops.core.plant_block.PlantBlockData` groups blocks
under when it adds up ``total_product``. This doesn't have to match the
Pyomo block's own name: two ``Product`` blocks with different resource names
show up as two separate rows in ``total_product``, while two blocks sharing a
resource name get summed into one row.
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


def _inlet_names_domain(value) -> tuple[str, ...]:
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


@declare_process_block_class("Product")
class ProductData(OpsBlockData):
    r"""A boundary sink: N named inlets, no outlets, one delivery total.

    See the module docstring above for the balance equation and the two
    deliberate differences from :class:`~flexops.unit_models.base.feed.Feed`: no
    blending, and composition limits set per inlet. ``inlet_names`` decides
    how many inlets there are and what their port names will be
    (``f"inlet_{name}"``).

    Config:
        ``property_package`` (inherited): a single-phase package shared by
        every port.

        ``inlet_names`` (default ``("a",)``): the inlets' names, used as
        their port names. Names must be unique and non-empty.

        ``resource_name`` (default ``None``): the key this block's delivery
        is grouped under. Defaults to the block's own Pyomo name.

        ``min_demand`` / ``max_demand`` (default ``None``): scalar limits on
        the delivery, with units. Interpreted according to ``demand_basis``
        (default ``LimitBasis.PERIOD``) as either a rate that must hold
        every period, or a total amount over the whole horizon.
        On the period basis a limit may also be a length-T series.

        ``price`` (default ``None``): a flat price per unit delivered —
        positive is a cost, negative is revenue — billed through
        ``costing_package`` (inherited), which must be set if you use this.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Product
        >>> m = dummy_time_block(3)
        >>> m.potable = Product(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     inlet_names=("municipal", "reuse"),
        ...     price=-2.0,
        ... )
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.declare(
        "inlet_names",
        ConfigValue(
            default=("a",),
            domain=_inlet_names_domain,
            description="Role names of the product's inlets; inlet i is built "
            "as port f'inlet_{name}'. Must be unique and non-empty. Each "
            "inlet's intensive states arrive from its own arc and are left "
            "independent: a Product aggregates flow and does not blend.",
        ),
    )
    CONFIG.declare(
        "resource_name",
        ConfigValue(
            default=None,
            description="The delivered resource's name (e.g. "
            "'potable_water'), the key it aggregates under in the plant's "
            "total_product. Defaults to the block's own Pyomo name; set it "
            "explicitly to override that — to merge two blocks under one key, "
            "or when the block name is not the name wanted in total_product.",
        ),
    )
    CONFIG.declare(
        "max_demand",
        ConfigValue(
            default=None,
            description="Units-carrying upper limit on the delivery, a scalar or "
            "(on the period basis) a length-T series "
            "(e.g. a disposal permit). Its dimension follows demand_basis: a "
            "rate per period, or a quantity over the horizon. Builds a mutable "
            "Param and its Constraint; rewrite the Param to move the bound.",
        ),
    )
    CONFIG.declare(
        "min_demand",
        ConfigValue(
            default=None,
            description="Units-carrying lower limit on the delivery, a scalar or "
            "(on the period basis) a length-T series "
            "(e.g. a contracted demand that must be met). Its dimension "
            "follows demand_basis: a rate per period, or a quantity over the "
            "horizon. Builds a mutable Param and its Constraint.",
        ),
    )
    CONFIG.declare(
        "demand_basis",
        ConfigValue(
            default=LimitBasis.PERIOD,
            description="Whether min_demand/max_demand bind in every period "
            "('period', the default -- the limits are rates such as m**3/hr) "
            "or on the horizon total ('horizon' -- the limits are quantities "
            "such as m**3, bounding the scalar delivery_total and leaving the "
            "optimizer to shape the profile that reaches it). A limit whose "
            "units contradict the basis is rejected. On its own, with no limit "
            "configured, this builds nothing.",
        ),
    )
    CONFIG.declare(
        "price",
        ConfigValue(
            default=None,
            description="Flat price per unit delivered, in the costing "
            "currency basis: positive is a cost (brine disposal), negative is "
            "revenue (potable water sold). Requires costing_package, and is "
            "billed as a scalar operating cost (price * sum_t flow[t] * dt).",
        ),
    )

    def build(self) -> None:
        """Validate the config, then build the ports, delivery, limits, and cost."""
        super().build()
        validate_port_names(self.config.inlet_names, "inlet_names")
        basis = resolve_basis(self.config.demand_basis, "demand_basis")
        self._phase = single_flow_phase(self.config.property_package, "Product")
        self._resource = (
            self.local_name
            if self.config.resource_name is None
            else self.config.resource_name
        )
        self.add_stream_ports(inlet_ports=self._inlet_port_names(), outlet_ports=())
        self._register_stream_states()
        self._build_delivery()
        add_limits = (
            add_horizon_limits if basis is LimitBasis.HORIZON else add_time_limits
        )
        add_limits(
            self,
            self.delivery,
            "delivery",
            lower=self.config.min_demand,
            upper=self.config.max_demand,
            fields=("min_demand", "max_demand"),
        )
        self.register_boundary_flow(
            self.delivery, resource=self._resource, kind=BoundaryKind.PRODUCT
        )
        self._register_cost()

    # -- config resolution --------------------------------------------------

    def _inlet_port_names(self) -> tuple[str, ...]:
        """Return the ``f"inlet_{name}"`` port names, in ``inlet_names`` order."""
        return tuple(f"inlet_{name}" for name in self.config.inlet_names)

    def _flow_basis_name(self) -> str:
        """Return the name of the flow variable that the property package uses."""
        return self.config.property_package.get_flow_basis_var_name()

    def _inlet_state(self, name: str):
        """Return the state block behind the inlet named ``name``."""
        return self.find_component(f"inlet_{name}_state")

    def _flow_units(self):
        """Return the units used by the property package's flow variable."""
        tb = self._find_time_block()
        state = self._inlet_state(self.config.inlet_names[0])
        flow = state.find_component(self._flow_basis_name())
        return pyunits.get_units(flow[tb.time_index.first(), self._phase])

    # -- ports, balance, limits ---------------------------------------------

    def _register_stream_states(self) -> None:
        """Register every inlet's non-flow states as results, not inputs.

        Unlike a feed's outlets, no inlet here acts as a reference: each
        incoming stream is already fully determined by whatever is
        connected upstream. That's why every one is registered with
        ``role="output"``, and none is tied to any other.
        """
        flow_name = self._flow_basis_name()
        for inlet_name in self.config.inlet_names:
            state = self._inlet_state(inlet_name)
            for name, var in state.define_state_vars().items():
                if name != flow_name:
                    self.register_io_variable(var, role="output")

    def _build_delivery(self) -> None:
        """Build the per-inlet flow References and the aggregated delivery."""
        tb = self._find_time_block()
        flow_name = self._flow_basis_name()
        inlet_names = self.config.inlet_names

        flows = {}
        for name in inlet_names:
            state = self._inlet_state(name)
            self.add_component(
                f"flow_in_{name}",
                pyo.Reference(state.find_component(flow_name)[:, self._phase]),
            )
            flows[name] = self.find_component(f"flow_in_{name}")

        # This is a Var defined by an equality constraint, not a plain
        # Expression, because the delivery needs to support bounds, be
        # fixable through set_external_dispatch, and be usable by costing.
        self.delivery = pyo.Var(
            tb.time_index,
            initialize=0.0,
            units=self._flow_units(),
            doc="Total resource delivered across the boundary in each period.",
        )

        @self.Constraint(
            tb.time_index,
            doc="Metering: the delivery equals the sum of the inlet flows.",
        )
        def eq_delivery(b, t):
            return b.delivery[t] == sum(flows[name][t] for name in inlet_names)

    def _register_cost(self) -> None:
        """Add the delivery's cost to opex, if a price was configured.

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
            quantity=self.delivery,
            price=price,
            quantity_units=self._flow_units(),
            unit=self,
        )
