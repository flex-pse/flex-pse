"""Tank(SISOBlock): holdup dynamics with logic disabled.

A tank has no on/off status, so it forces ``unit_commitment.status`` to
``False`` regardless of what a caller passes -- the canonical example of a
physical subclass turning off a base capability.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexops.unit_models.base.siso import SISOBlockData


@declare_process_block_class("Tank")
class TankData(SISOBlockData):
    r"""A storage tank: holdup difference equation, no on/off status.

    Inherits inlet/outlet ports from
    :class:`~flexops.unit_models.base.siso.SISOBlockData` but *replaces* the
    SISO pass-through mass balance with a holdup difference equation (a
    pump's inlet equals its outlet; a tank stores the difference). Per the
    project's backward-differencing convention for rate/difference equations
    (``plan/00_conventions.md`` §2), the volume *ending* period ``t`` is
    written in terms of the flows sampled *at* ``t``:

    .. math::

        \text{volume}[t] = \text{volume}[t-1]
            + \Delta t \cdot (\dot{V}_{in}[t] - \dot{V}_{out}[t]),
        \quad t = 1, \dots, N-1

    Both flows are dispatch inputs (a tank has no natural "output" flow), so
    the outlet ``flow_vol_phase`` is re-registered as ``role="input"``;
    ``volume`` is the registered output. No ``declare_power``/``register_power``
    call -- the tank draws nothing.

    **``max_volume`` vs. ``capacity``.** ``max_volume`` is the maximum
    *possible* tank volume -- fixed by prior investment in an existing tank,
    or by space constraints on a potential build. It is a static config
    constant and the upper bound on ``capacity``. ``capacity`` is the
    *chosen* tank volume (a design ``Var``), which may be ``<= max_volume``;
    it is fixed at ``max_volume`` by default (operations mode) and unfixed,
    subject to that same upper bound, in the design mode.

    **``level``: bounded fractional fill.** ``level[t] = volume[t] /
    capacity`` is the tank's fill fraction relative to its *chosen* size,
    bounded by ``(level_min, level_max)`` so the tank neither drains all the
    way down nor overfills. Because ``capacity`` is itself a ``Var``, the
    defining equation ``volume[t] == level[t] * capacity`` is a product of
    two variables: linear (and LP) when ``capacity`` is fixed (operations
    mode), but bilinear (NLP) when ``capacity`` is free (design mode / M16
    multi-period sizing) -- a deliberate, documented tradeoff. Design-mode
    solves of a model containing a tank therefore need IPOPT or an explicit
    ``flexschedule.SolveSequence``, not HiGHS.

    ``level_definition`` is registered as a swappable relation (see
    :meth:`~flexops.core.ops_block.OpsBlockData.register_relation`) — a
    non-prismatic tank geometry can replace it in place via
    :meth:`~flexops.core.ops_block.OpsBlockData.swap_relation`. The holdup
    difference equation is deliberately **not** registered and so can never be
    swapped.

    Config:
        Inherits the SISO/OpsBlock config; adds ``min_volume`` (default
        0 m^3), ``max_volume`` (default 1000 m^3), ``initial_volume``
        (default 500 m^3, half of the default ``max_volume``),
        ``level_min`` (default 0.0), and ``level_max`` (default 1.0).
        Both volumes have defaults only so an unsized ``Tank(...)`` builds;
        a real tank is always given its own.

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Tank
        >>> from pyomo.environ import units as pyunits
        >>> m = dummy_time_block(4)
        >>> m.tank = Tank(  # doctest: +SKIP
        ...     property_package=m.properties,
        ...     max_volume=1000 * pyunits.m**3,
        ...     initial_volume=200 * pyunits.m**3,
        ... )
    """

    CONFIG = SISOBlockData.CONFIG()
    CONFIG.declare(
        "min_volume",
        ConfigValue(
            default=0 * pyunits.m**3,
            description="Minimum tank volume, a lower bound on volume[t] "
            "(and on capacity).",
        ),
    )
    CONFIG.declare(
        "max_volume",
        ConfigValue(
            default=1000 * pyunits.m**3,
            description="Maximum POSSIBLE tank volume -- fixed by prior "
            "investment in an existing tank or by space constraints on a "
            "potential build. The upper bound on capacity (the chosen "
            "volume) and the default value of the fixable capacity design "
            "variable.",
        ),
    )
    CONFIG.declare(
        "initial_volume",
        ConfigValue(
            default=500 * pyunits.m**3,
            description="Initial tank volume, volume[0]: a mutable Param "
            "and a rolling-horizon initial-state hook. Defaults to half of "
            "the default max_volume, so a tank built with no sizing at all "
            "starts half full; give it explicitly whenever max_volume is.",
        ),
    )
    CONFIG.declare(
        "level_min",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Minimum fractional fill (level = volume/capacity), "
            "in [0, 1]. Prevents the tank from draining all the way down.",
        ),
    )
    CONFIG.declare(
        "level_max",
        ConfigValue(
            default=1.0,
            domain=float,
            description="Maximum fractional fill (level = volume/capacity), "
            "in [0, 1]. Prevents the tank from overfilling.",
        ),
    )

    def build(self) -> None:
        """Build the SISO ports/holdup, then capacity, level, and registration
        fixups."""
        super().build()
        tb = self._find_time_block()

        # A tank has no on/off status; force it off even if a caller asked.
        self.config.unit_commitment.status = False

        min_volume = pyo.value(pyunits.convert(self.config.min_volume, pyunits.m**3))
        max_volume = pyo.value(pyunits.convert(self.config.max_volume, pyunits.m**3))
        self.capacity = pyo.Var(
            initialize=max_volume,
            bounds=(min_volume, max_volume),
            units=pyunits.m**3,
            doc="Chosen tank volume (<= max_volume). Fixed at max_volume by "
            "default; the design mode unfixes it, subject to this bound.",
        )
        self.capacity.fix(max_volume)

        @self.Constraint(tb.time_index, doc="Tank volume never exceeds capacity.")
        def capacity_limit(b, t):
            return b.volume[t] <= b.capacity

        self.level = pyo.Var(
            tb.time_index,
            bounds=(self.config.level_min, self.config.level_max),
            initialize=pyo.value(self.initial_volume) / max_volume,
            units=pyunits.dimensionless,
            doc="Fractional fill relative to the chosen capacity: "
            "volume[t] / capacity, bounded by (level_min, level_max) so the "
            "tank neither drains all the way down nor overfills.",
        )

        @self.Constraint(
            tb.time_index,
            doc="Defines level as volume relative to the chosen capacity: "
            "volume[t] == level[t] * capacity. Linear (LP) when capacity is "
            "fixed (operations mode); bilinear (NLP) when capacity is free "
            "(design mode / M16 sizing) -- a deliberate tradeoff.",
        )
        def level_definition(b, t):
            return b.volume[t] == b.level[t] * b.capacity

        self.register_relation(self.level_definition, target=self.level)

        # Both flows are dispatch inputs for a tank; the inherited
        # add_stream_ports() registered the outlet as an output, so correct it.
        self._io_registry.io_variables = [
            rec
            for rec in self._io_registry.io_variables
            if rec.var is not self.outlet_state.flow_vol_phase
        ]
        self.register_io_variable(self.outlet_state.flow_vol_phase, role="input")
        self.register_io_variable(self.volume, role="output")

    def _build_mass_balance(self) -> None:
        """Replace the SISO pass-through balance with the holdup difference equation."""
        tb = self._find_time_block()
        pkg = self.config.property_package

        min_volume = pyo.value(pyunits.convert(self.config.min_volume, pyunits.m**3))
        max_volume = pyo.value(pyunits.convert(self.config.max_volume, pyunits.m**3))
        initial_volume = pyo.value(
            pyunits.convert(self.config.initial_volume, pyunits.m**3)
        )

        self.volume = pyo.Var(
            tb.time_index,
            bounds=(min_volume, max_volume),
            initialize=initial_volume,
            units=pyunits.m**3,
            doc="Tank holdup volume.",
        )

        self.flow_in = pyo.Reference(self.inlet_state.flow_vol_phase[:, "Liq"])
        self.flow_out = pyo.Reference(self.outlet_state.flow_vol_phase[:, "Liq"])

        self.initial_volume = pyo.Param(
            initialize=initial_volume,
            mutable=True,
            units=pyunits.m**3,
            doc="Initial tank volume, volume[0] (rolling-horizon initial state).",
        )
        tb.register_initial_state(self.initial_volume)
        self.register_process_parameter(self.initial_volume, regressable=False)

        @self.Constraint(doc="Initial condition: volume[0] equals initial_volume.")
        def initial_volume_eq(b):
            return b.volume[0] == b.initial_volume

        @self.Constraint(
            list(tb.time_index)[1:],
            doc="Holdup difference equation (backward): volume[t] = "
            "volume[t-1] + dt*(in[t] - out[t]). Uses pyunits.convert on the "
            "whole right-hand side rather than assuming the flow basis's "
            "units, so it stays correct regardless of the property "
            "package's flow units (m^3/hr, m^3/s, ...).",
        )
        def holdup(b, t):
            delta_volume = pyunits.convert(
                tb.dt * (b.flow_in[t] - b.flow_out[t]), to_units=pyunits.m**3
            )
            return b.volume[t] == b.volume[t - 1] + delta_volume

        # A tank governs flow itself (via the holdup equation above), so
        # pass every OTHER state variable straight through inlet->outlet
        # (e.g. pressure/temperature, when a richer property package enables
        # them); flow is excluded here and re-registered as a dispatch input
        # below (in build()).
        #
        # TODO: if a future property package exposes a variable
        # flow_mass_phase_comp (mass/TDS basis) alongside flow_vol_phase,
        # composition mixing at the tank is not modeled -- either assume TDS
        # is constant or add real mixing constraints then. Detect that case
        # by checking for flow_mass_phase_comp on the port.
        self.add_pass_through_constraints(
            self.inlet, self.outlet, exclude_vars=[pkg.get_flow_basis_var_name()]
        )
