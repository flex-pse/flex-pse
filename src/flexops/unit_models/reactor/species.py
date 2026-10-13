"""SpeciesReactor(ReactorBase): RTD compartments with swappable feed and kinetics.

For reactors whose feed and product species are all observable. The reactor is
a series of ``n_compartments`` well-mixed volumes; each feed species enters
through side feeds whose split is the residence-time distribution, reacts with
first-order kinetics, and leaves the last compartment. The side-feed and
reaction-rate relations are registered per (species, compartment) so
``swap_relation`` can replace them; the holdup balance, feed closure, and
outlet flows are conservation and can never be swapped. See
``docs/explanation/reactor_models.md`` for the equations.
"""

import math

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.unit_models.reactor.base import ReactorBaseData


def _fractions_domain(value) -> tuple[float, ...] | None:
    """ConfigValue domain: coerce RTD fractions to a tuple of floats."""
    return None if value is None else tuple(float(v) for v in value)


@declare_process_block_class("SpeciesReactor")
class SpeciesReactorData(ReactorBaseData):
    """Compartment-series reactor tracking every feed and product species.

    Feed species are ``inlet_names`` and product species are ``outlet_names``;
    each outlet carries its own product, and the reference outlet also
    carries unreacted feed. ``yield_<j>_<k> * rate_constant_<k>`` is only
    separately identifiable when that unreacted feed is measured.

    Config:
        Inherits the ReactorBase config; adds ``n_compartments`` (default 3),
        ``residence_time`` (default 1 hr, split evenly across compartments),
        ``rtd_fractions`` (default all feed into compartment 1; one per
        compartment, summing to 1), ``rate_constants`` (default 1/hr per feed
        species), and ``yields`` (``{(product, feed): value}``; default 1.0
        from every feed to the reference outlet's product, any pair left out
        is 0).

    Raises:
        FlexConfigError: If a species is both an inlet and an outlet,
            ``n_compartments`` is below 1, ``rtd_fractions`` has the wrong
            length, a negative entry, or does not sum to 1, or
            ``rate_constants``/``yields`` names an unknown species.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import SpeciesReactor
        >>> m = dummy_time_block(3)
        >>> m.unit = SpeciesReactor(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     inlet_names=("co2", "h2"),
        ...     outlet_names=("purge", "ch4"),
        ...     rtd_fractions=(0.6, 0.3, 0.1),
        ... )
    """

    CONFIG = ReactorBaseData.CONFIG()
    CONFIG.declare(
        "n_compartments",
        ConfigValue(
            default=3,
            domain=int,
            description="Number of well-mixed compartments in series.",
        ),
    )
    CONFIG.declare(
        "residence_time",
        ConfigValue(
            default=1.0 * pyunits.hr,
            description="Total residence time; compartment_time starts at "
            "residence_time / n_compartments (a fixed, regressable Var).",
        ),
    )
    CONFIG.declare(
        "rtd_fractions",
        ConfigValue(
            default=None,
            domain=_fractions_domain,
            description="Fraction of each feed entering each compartment,"
            "one per compartment, summing to 1. Feed entering later compartments"
            "bypasses the earlier ones (channeling). None sends all feed"
            "into compartment 1.",
        ),
    )
    CONFIG.declare(
        "rate_constants",
        ConfigValue(
            default={},
            domain=dict,
            description="First-order consumption constant per feed species, "
            "1/hr. Species left out default to 1/hr.",
        ),
    )
    CONFIG.declare(
        "yields",
        ConfigValue(
            default={},
            domain=dict,
            description="{(product, feed): volume of product formed per volume "
            "of feed consumed}. Empty yields 1.0 from every feed to the "
            "reference outlet's product; otherwise pairs left out are 0.",
        ),
    )

    def _build_holdup(self) -> None:
        """Build the compartment holdups, conservation, and swappable relations."""
        self._validate_species()
        self._build_parameters()
        self._build_variables()
        self._build_conservation()
        self._build_side_feed_relations()
        self._build_reaction_relations()

    def _validate_species(self) -> None:
        """Validate species names, compartment count, fractions, and parameters.

        Raises:
            FlexConfigError: On any invalid species or parameter config.
        """
        feeds, products = self.config.inlet_names, self.config.outlet_names
        shared = sorted(set(feeds) & set(products))
        if shared:
            raise FlexConfigError(
                f"species {shared} are both inlets and outlets; each species "
                "must be a feed or a product, not both.",
                field="outlet_names",
                value=shared,
            )
        n = self.config.n_compartments
        if n < 1:
            raise FlexConfigError(
                f"n_compartments must be at least 1, got {n}.",
                field="n_compartments",
                value=n,
            )
        fractions = self._rtd_fractions()
        if (
            len(fractions) != n
            or min(fractions) < 0
            or not math.isclose(sum(fractions), 1.0, abs_tol=1e-8)
        ):
            raise FlexConfigError(
                f"rtd_fractions must be {n} non-negative values summing to 1, "
                f"got {fractions!r}.",
                field="rtd_fractions",
                value=fractions,
            )
        unknown_rates = sorted(set(self.config.rate_constants) - set(feeds))
        if unknown_rates:
            raise FlexConfigError(
                f"rate_constants names {unknown_rates}, not feed species {feeds}.",
                field="rate_constants",
                value=unknown_rates,
            )
        unknown_yields = [
            pair
            for pair in self.config.yields
            if pair[0] not in products or pair[1] not in feeds
        ]
        if unknown_yields:
            raise FlexConfigError(
                f"yields keys must be (product, feed) pairs from {products} x "
                f"{feeds}, got {unknown_yields}.",
                field="yields",
                value=unknown_yields,
            )

    def _rtd_fractions(self) -> tuple[float, ...]:
        """Return the configured RTD fractions, defaulting to all in compartment 1."""
        if self.config.rtd_fractions is not None:
            return self.config.rtd_fractions
        return (1.0,) + (0.0,) * (self.config.n_compartments - 1)

    def _yield(self, product: str, feed: str) -> float:
        """Return the configured yield of ``product`` per ``feed`` consumed."""
        if not self.config.yields:
            return 1.0 if product == self.config.outlet_names[0] else 0.0
        return self.config.yields.get((product, feed), 0.0)

    def _build_parameters(self) -> None:
        """Declare compartment_time, RTD fractions, rate constants, and yields."""
        n = self.config.n_compartments
        self.declare_process_parameter(
            "compartment_time",
            self.config.residence_time / n,
            pyunits.hr,
            "Residence time of one compartment.",
            bounds=(1e-6, None),
        )
        for i, fraction in enumerate(self._rtd_fractions()[:-1], start=1):
            self.declare_process_parameter(
                f"rtd_fraction_{i}",
                fraction,
                pyunits.dimensionless,
                f"Fraction of each feed entering compartment {i}.",
                bounds=(0.0, 1.0),
            )
        for feed in self.config.inlet_names:
            self.declare_process_parameter(
                f"rate_constant_{feed}",
                self.config.rate_constants.get(feed, 1.0),
                1 / pyunits.hr,
                f"First-order consumption constant of {feed}.",
                bounds=(0.0, None),
            )
            for product in self.config.outlet_names:
                self.declare_process_parameter(
                    f"yield_{product}_{feed}",
                    self._yield(product, feed),
                    pyunits.dimensionless,
                    f"Volume of {product} formed per volume of {feed} consumed.",
                    bounds=(0.0, None),
                )

    def _build_variables(self) -> None:
        """Declare holdups, reaction rates, side feeds, and their 1-D slices."""
        tb = self._find_time_block()
        feeds, products = self.config.inlet_names, self.config.outlet_names
        self.feed_species = pyo.Set(initialize=feeds, doc="Feed species.")
        self.species = pyo.Set(initialize=feeds + products, doc="All species.")
        self.compartments = pyo.RangeSet(
            self.config.n_compartments, doc="Compartments."
        )

        self.holdup = pyo.Var(
            self.species,
            self.compartments,
            tb.time_index,
            initialize=0.0,
            bounds=(0.0, None),
            units=pyunits.m**3,
            doc="Volume of each species held in each compartment.",
        )
        self.reaction_rate = pyo.Var(
            self.species,
            self.compartments,
            tb.time_index,
            initialize=0.0,
            units=pyunits.m**3 / pyunits.hr,
            doc="Net generation rate of each species in each compartment.",
        )
        self.side_feed = pyo.Var(
            self.feed_species,
            self.compartments,
            tb.time_index,
            initialize=0.0,
            bounds=(0.0, None),
            units=pyunits.m**3 / pyunits.hr,
            doc="Feed flow of each feed species entering each compartment.",
        )
        self.initial_holdup = pyo.Param(
            self.species,
            self.compartments,
            initialize=0.0,
            mutable=True,
            units=pyunits.m**3,
            doc="Holdup at the first time point (rolling-horizon initial state).",
        )
        tb.register_initial_state(self.initial_holdup)
        self.register_process_parameter(self.initial_holdup, regressable=False)

        for s in self.species:
            for i in self.compartments:
                self.add_component(
                    f"holdup_{s}_{i}", pyo.Reference(self.holdup[s, i, :])
                )
                self.add_component(
                    f"reaction_rate_{s}_{i}", pyo.Reference(self.reaction_rate[s, i, :])
                )
        for k in self.feed_species:
            for i in self.compartments:
                self.add_component(
                    f"side_feed_{k}_{i}", pyo.Reference(self.side_feed[k, i, :])
                )

    def _build_conservation(self) -> None:
        """Build the holdup balance, feed closure, and outlet flows."""
        tb = self._find_time_block()
        n = self.config.n_compartments
        tau = self.compartment_time
        feeds = self.config.inlet_names
        reference_outlet = self.config.outlet_names[0]

        @self.Constraint(
            self.species,
            self.compartments,
            doc="Initial condition: holdup[0] equals initial_holdup.",
        )
        def initial_holdup_eq(b, s, i):
            return b.holdup[s, i, 0] == b.initial_holdup[s, i]

        @self.Constraint(
            self.species,
            self.compartments,
            list(tb.time_index)[1:],
            doc="Holdup balance (backward Euler): side feed, transport from the "
            "previous compartment, and reaction. Never swappable.",
        )
        def holdup_balance(b, s, i, t):
            transport = -b.holdup[s, i, t]
            if i > 1:
                transport = transport + b.holdup[s, i - 1, t]
            rate = transport / tau + b.reaction_rate[s, i, t]
            if s in b.feed_species:
                rate = rate + b.side_feed[s, i, t]
            return b.holdup[s, i, t] == b.holdup[s, i, t - 1] + pyunits.convert(
                tb.dt * rate, pyunits.m**3
            )

        @self.Constraint(
            self.feed_species,
            tb.time_index,
            doc="Feed closure: the last compartment takes whatever feed the "
            "others do not, so the RTD always conserves feed.",
        )
        def feed_closure(b, k, t):
            return b.side_feed[k, n, t] == pyunits.convert(
                b._flow_in(k)[t], pyunits.m**3 / pyunits.hr
            ) - sum(b.side_feed[k, i, t] for i in range(1, n))

        @self.Constraint(
            self.config.outlet_names,
            tb.time_index,
            doc="Each outlet carries its product leaving the last compartment; "
            "the reference outlet also carries unreacted feed.",
        )
        def outlet_flow_eq(b, j, t):
            leaving = b.holdup[j, n, t]
            if j == reference_outlet:
                leaving = leaving + sum(b.holdup[k, n, t] for k in feeds)
            flow = b._flow_out(j)[t]
            return flow == pyunits.convert(leaving / tau, pyunits.get_units(flow))

    def _build_side_feed_relations(self) -> None:
        """Build a swappable ``side_feed_<k>_<i>_relation`` per feed and compartment."""
        tb = self._find_time_block()
        for k in self.config.inlet_names:
            flow_in = self._flow_in(k)
            for i in range(1, self.config.n_compartments):
                target = self.find_component(f"side_feed_{k}_{i}")
                fraction = self.find_component(f"rtd_fraction_{i}")

                def _rule(b, t, _target=target, _fraction=fraction, _flow=flow_in):
                    return _target[t] == pyunits.convert(
                        _fraction * _flow[t], pyunits.m**3 / pyunits.hr
                    )

                name = f"side_feed_{k}_{i}_relation"
                self.add_component(
                    name,
                    pyo.Constraint(
                        tb.time_index,
                        rule=_rule,
                        doc=f"Side feed of {k} into compartment {i}: "
                        f"rtd_fraction_{i} * inlet flow. Swappable.",
                    ),
                )
                self.register_relation(self.find_component(name), target=target)

    def _build_reaction_relations(self) -> None:
        """Build a swappable reaction-rate relation per species and compartment."""
        tb = self._find_time_block()
        feeds = self.config.inlet_names
        for s in self.species:
            for i in self.compartments:
                target = self.find_component(f"reaction_rate_{s}_{i}")
                if s in self.feed_species:
                    body = {s: -1.0}
                    doc = f"Consumption of {s}: -rate_constant_{s} * holdup."
                else:
                    body = {k: self.find_component(f"yield_{s}_{k}") for k in feeds}
                    doc = f"Generation of {s}: sum of yield * rate_constant * holdup."

                def _rule(b, t, _target=target, _body=body, _i=i):
                    return _target[t] == sum(
                        coefficient
                        * b.find_component(f"rate_constant_{k}")
                        * b.holdup[k, _i, t]
                        for k, coefficient in _body.items()
                    )

                name = f"reaction_rate_{s}_{i}_relation"
                self.add_component(
                    name,
                    pyo.Constraint(tb.time_index, rule=_rule, doc=f"{doc} Swappable."),
                )
                self.register_relation(self.find_component(name), target=target)
