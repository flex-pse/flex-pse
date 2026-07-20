"""StorageTank(SISOBlock): holdup dynamics with logic disabled (architecture §3.4, R6).

A tank has no on/off status, so it forces ``unit_commitment.status`` to
``False`` regardless of what a caller passes -- the canonical example of a
physical subclass turning off a base capability.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexops.unit_models.base.siso import SISOBlockData


@declare_process_block_class("StorageTank")
class StorageTankData(SISOBlockData):
    r"""A storage tank: holdup difference equation, no on/off status (R6).

    Inherits inlet/outlet ports from
    :class:`~flexops.unit_models.base.siso.SISOBlockData` but *replaces* the
    SISO pass-through mass balance with a holdup difference equation (a
    pump's inlet equals its outlet; a tank stores the difference). Per the
    project's backward-differencing convention for rate/difference equations
    (``plan/00_conventions.md`` §2), the volume *ending* period ``t`` is
    written in terms of the flows sampled *at* ``t``:

    .. math::

        V[t] = V[t-1] + dt \cdot (\dot{V}_{in}[t] - \dot{V}_{out}[t]),
        \quad t = 1, \dots, N-1

    Both flows are dispatch inputs (a tank has no natural "output" flow), so
    the outlet ``flow_vol_phase`` is re-registered as ``role="input"``; ``V``
    is the registered output. No ``declare_power``/``register_power`` call --
    the tank draws nothing.

    Config:
        Inherits the SISO/OpsBlock config; adds ``min_volume`` (default
        0 m^3), ``max_volume`` (required), and ``initial_volume`` (required).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import StorageTank
        >>> from pyomo.environ import units as pyunits
        >>> m = dummy_time_block(4)
        >>> m.tank = StorageTank(  # doctest: +SKIP
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
            description="Minimum tank volume, a lower bound on V.",
        ),
    )
    CONFIG.declare(
        "max_volume",
        ConfigValue(
            description="Maximum tank volume: an upper bound on V and the "
            "default value of the fixable capacity design variable. Required."
        ),
    )
    CONFIG.declare(
        "initial_volume",
        ConfigValue(
            description="Initial tank volume V[0]: a mutable Param and a "
            "rolling-horizon initial-state hook. Required."
        ),
    )

    def build(self) -> None:
        """Build the SISO ports/holdup, then capacity, registration fixups, R6."""
        super().build()
        tb = self._find_time_block()

        # R6: a tank has no on/off status; force it off even if a caller asked.
        self.config.unit_commitment.status = False

        max_volume = pyo.value(pyunits.convert(self.config.max_volume, pyunits.m**3))
        self.capacity = pyo.Var(
            initialize=max_volume,
            units=pyunits.m**3,
            doc="Fixable design capacity; fixed at max_volume by default "
            "(unfixed in M07's design mode).",
        )
        self.capacity.fix(max_volume)

        @self.Constraint(tb.time_index, doc="Tank volume never exceeds capacity.")
        def capacity_limit(b, t):
            return b.V[t] <= b.capacity

        # Both flows are dispatch inputs for a tank; the inherited
        # add_stream_ports() registered the outlet as an output, so correct it.
        self._io_registry.io_variables = [
            rec
            for rec in self._io_registry.io_variables
            if rec.var is not self.outlet_state.flow_vol_phase
        ]
        self.register_io_variable(self.outlet_state.flow_vol_phase, role="input")
        self.register_io_variable(self.V, role="output")

    def _build_mass_balance(self) -> None:
        """Replace the SISO pass-through balance with the holdup difference equation."""
        tb = self._find_time_block()

        min_volume = pyo.value(pyunits.convert(self.config.min_volume, pyunits.m**3))
        max_volume = pyo.value(pyunits.convert(self.config.max_volume, pyunits.m**3))
        initial_volume = pyo.value(
            pyunits.convert(self.config.initial_volume, pyunits.m**3)
        )

        self.V = pyo.Var(
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
            doc="Initial tank volume, V[0] (rolling-horizon initial state).",
        )
        tb.register_initial_state(self.initial_volume)
        self.register_process_parameter(self.initial_volume, regressable=False)

        @self.Constraint(doc="Initial condition: V[0] equals initial_volume.")
        def initial_volume_eq(b):
            return b.V[0] == b.initial_volume

        @self.Constraint(
            list(tb.time_index)[1:],
            doc="Holdup difference equation (backward): "
            "V[t] = V[t-1] + dt*(in[t] - out[t]).",
        )
        def holdup(b, t):
            dt_hr = pyunits.convert(tb.dt, to_units=pyunits.hr)
            return b.V[t] == b.V[t - 1] + dt_hr * (b.flow_in[t] - b.flow_out[t])
