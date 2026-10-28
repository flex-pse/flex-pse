"""LumpedReactor(ReactorBase): volume holdup with swappable outlet compositions.

For reactors whose species are not observable: only flows, level, temperature,
pressure, power, and the outlet compositions of interest are measured. Flow
accumulates in a tank-like volume holdup. Each outlet composition is a constant
relation registered so ``swap_relation`` can replace it with a fitted
relationship (for example a transfer function of the measured states). See
``docs/explanation/reactor_models.md`` for the equations.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError
from flexops.unit_models._multiport import validate_port_names
from flexops.unit_models.reactor.base import ReactorBaseData


@declare_process_block_class("LumpedReactor")
class LumpedReactorData(ReactorBaseData):
    """Volume-holdup reactor with constant, swappable outlet compositions.

    The inlet flows and the reference outlet flow are dispatch inputs; every
    other outlet takes ``split_fraction_<name>`` of the total outflow. Volume,
    level, and each ``outlet_composition_<name>`` are outputs.

    Config:
        Inherits the ReactorBase config; adds ``max_volume`` (default
        1000 m^3), ``initial_volume`` (default 500 m^3), ``level_min``
        (default 0.0), ``level_max`` (default 1.0), ``split_fractions``
        (``{outlet: fraction}`` naming every non-reference outlet; default an
        equal split), and ``compositions`` (``{name: constant fraction}``;
        default none).

    Raises:
        FlexConfigError: If ``split_fractions`` names the reference or an
            unknown outlet, misses a non-reference outlet, or sums above 1, or
            if ``compositions`` names are empty or duplicated.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import LumpedReactor
        >>> m = dummy_time_block(3)
        >>> m.unit = LumpedReactor(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     outlet_names=("biogas", "digestate"),
        ...     compositions={"ch4": 0.6},
        ... )
    """

    CONFIG = ReactorBaseData.CONFIG()
    CONFIG.declare(
        "max_volume",
        ConfigValue(
            default=1000 * pyunits.m**3,
            description="Maximum possible reactor volume; the bound on "
            "capacity and its fixed value outside design mode.",
        ),
    )
    CONFIG.declare(
        "initial_volume",
        ConfigValue(
            default=500 * pyunits.m**3,
            description="Volume at the first time point (a rolling-horizon "
            "initial-state Param).",
        ),
    )
    CONFIG.declare(
        "level_min",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Minimum fractional fill, volume / capacity.",
        ),
    )
    CONFIG.declare(
        "level_max",
        ConfigValue(
            default=1.0,
            domain=float,
            description="Maximum fractional fill, volume / capacity.",
        ),
    )
    CONFIG.declare(
        "split_fractions",
        ConfigValue(
            default={},
            domain=dict,
            description="{outlet: fraction of total outflow} for every "
            "non-reference outlet (fixed, regressable Vars). The reference "
            "outlet takes the rest. Empty splits the outflow equally.",
        ),
    )
    CONFIG.declare(
        "compositions",
        ConfigValue(
            default={},
            domain=dict,
            description="{name: constant fraction} of outlet compositions to "
            "track; each becomes outlet_composition_<name>[t] with a "
            "swappable relation.",
        ),
    )

    def _build_holdup(self) -> None:
        """Build the volume holdup, outlet split, and composition relations."""
        self._build_volume()
        self._build_split()
        self._build_compositions()

    def _build_volume(self) -> None:
        """Build volume, capacity, level, and the holdup difference equation."""
        tb = self._find_time_block()
        max_volume = pyo.value(pyunits.convert(self.config.max_volume, pyunits.m**3))
        initial_volume = pyo.value(
            pyunits.convert(self.config.initial_volume, pyunits.m**3)
        )
        self.volume = pyo.Var(
            tb.time_index,
            initialize=initial_volume,
            bounds=(0.0, max_volume),
            units=pyunits.m**3,
            doc="Reactor holdup volume.",
        )
        self.capacity = pyo.Var(
            initialize=max_volume,
            bounds=(0.0, max_volume),
            units=pyunits.m**3,
            doc="Chosen reactor volume; fixed at max_volume outside design mode.",
        )
        self.capacity.fix(max_volume)
        self.level = pyo.Var(
            tb.time_index,
            initialize=initial_volume / max_volume,
            bounds=(self.config.level_min, self.config.level_max),
            units=pyunits.dimensionless,
            doc="Fractional fill, volume / capacity.",
        )
        self.initial_volume = pyo.Param(
            initialize=initial_volume,
            mutable=True,
            units=pyunits.m**3,
            doc="Volume at the first time point (rolling-horizon initial state).",
        )
        tb.register_initial_state(self.initial_volume)
        self.register_process_parameter(self.initial_volume, regressable=False)

        @self.Constraint(doc="Initial condition: volume[0] equals initial_volume.")
        def initial_volume_eq(b):
            return b.volume[0] == b.initial_volume

        @self.Constraint(
            list(tb.time_index)[1:],
            doc="Holdup (backward Euler): volume[t] = volume[t-1] + dt*(total "
            "inflow - total outflow). Never swappable.",
        )
        def holdup(b, t):
            net = sum(b._flow_in(name)[t] for name in b.config.inlet_names) - sum(
                b._flow_out(name)[t] for name in b.config.outlet_names
            )
            return b.volume[t] == b.volume[t - 1] + pyunits.convert(
                tb.dt * net, pyunits.m**3
            )

        @self.Constraint(tb.time_index, doc="Volume never exceeds capacity.")
        def capacity_limit(b, t):
            return b.volume[t] <= b.capacity

        @self.Constraint(
            tb.time_index, doc="Defines level: volume == level * capacity."
        )
        def level_definition(b, t):
            return b.volume[t] == b.level[t] * b.capacity

        self.register_relation(self.level_definition, target=self.level)

        reference_flow = self._state(
            f"outlet_{self.config.outlet_names[0]}"
        ).flow_vol_phase
        self._io_registry.io_variables = [
            rec
            for rec in self._io_registry.io_variables
            if rec.var is not reference_flow
        ]
        self.register_io_variable(reference_flow, role="input")
        self.register_io_variable(self.volume, role="output")
        self.register_io_variable(self.level, role="output")

    def _split_fractions(self) -> dict[str, float]:
        """Return validated split fractions for every non-reference outlet.

        Raises:
            FlexConfigError: If the fractions name the reference or an unknown
                outlet, miss a non-reference outlet, or sum outside [0, 1].
        """
        others = self.config.outlet_names[1:]
        splits = self.config.split_fractions or {
            name: 1.0 / len(self.config.outlet_names) for name in others
        }
        if (
            set(splits) != set(others)
            or min(splits.values(), default=0.0) < 0
            or sum(splits.values()) > 1.0 + 1e-8
        ):
            raise FlexConfigError(
                f"split_fractions must give every non-reference outlet {others} "
                f"a non-negative fraction summing to at most 1, got {splits!r}.",
                field="split_fractions",
                value=splits,
            )
        return splits

    def _build_split(self) -> None:
        """Build ``split_fraction_<name>`` and the outlet split constraints."""
        splits = self._split_fractions()
        if not splits:
            return
        tb = self._find_time_block()
        for name, fraction in splits.items():
            self.declare_process_parameter(
                f"split_fraction_{name}",
                fraction,
                pyunits.dimensionless,
                f"Fraction of total outflow leaving through outlet {name}.",
                bounds=(0.0, 1.0),
            )

        @self.Constraint(
            list(splits),
            tb.time_index,
            doc="Each non-reference outlet takes its split fraction of total outflow.",
        )
        def outlet_split(b, name, t):
            total = sum(b._flow_out(outlet)[t] for outlet in b.config.outlet_names)
            return b._flow_out(name)[t] == (
                b.find_component(f"split_fraction_{name}") * total
            )

    def _build_compositions(self) -> None:
        """Build one constant, swappable composition relation per configured name."""
        compositions = self.config.compositions
        if not compositions:
            return
        validate_port_names(tuple(compositions), "compositions")
        tb = self._find_time_block()
        for name, value in compositions.items():
            constant = self.declare_process_parameter(
                f"composition_{name}",
                value,
                pyunits.dimensionless,
                f"Constant outlet fraction of {name}.",
                bounds=(0.0, 1.0),
            )
            self.add_component(
                f"outlet_composition_{name}",
                pyo.Var(
                    tb.time_index,
                    initialize=pyo.value(constant),
                    bounds=(0.0, 1.0),
                    units=pyunits.dimensionless,
                    doc=f"Outlet fraction of {name}.",
                ),
            )
            target = self.find_component(f"outlet_composition_{name}")
            self.add_component(
                f"outlet_composition_{name}_relation",
                pyo.Constraint(
                    tb.time_index,
                    rule=lambda b, t, _target=target, _constant=constant: _target[t]
                    == _constant,
                    doc=f"Outlet fraction of {name} equals composition_{name}. "
                    "Swappable.",
                ),
            )
            self.register_relation(
                self.find_component(f"outlet_composition_{name}_relation"),
                target=target,
            )
            self.register_io_variable(target, role="output")
