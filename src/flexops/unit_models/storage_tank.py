"""``StorageTank``: holdup-difference-equation SISO unit (architecture §3.4, LP)."""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore.config.schema import UnitCommitmentConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.time_block import TimeBlockData
from flexops.unit_models.base.siso import SISOBlockData


def _unit_commitment_off(value) -> UnitCommitmentConfig:
    """ConfigValue domain: coerce any caller-passed UC config to fully off.

    A tank has no on/off status (R6), so whatever a caller passes is replaced
    at config-processing time — before any build step can read it.
    """
    return UnitCommitmentConfig(status=False)


@declare_process_block_class("StorageTank")
class StorageTankData(SISOBlockData):
    """A storage tank whose holdup difference equation replaces the SISO balance.

    Inherits the inlet/outlet ports and the ``flow_vol`` handle from
    :class:`~flexops.unit_models.base.siso.SISOBlockData`; inlet and outlet
    flows differ — the tank stores the difference:

    .. math::

        V[t] = V[t-1] + \\Delta t \\,(\\text{flow\\_in}[t] -
        \\text{flow\\_out}[t]), \\qquad t = 1 \\dots N-1

    a backwards difference, with :math:`\\Delta t` converted to hours so
    m³/hr × hr = m³, and :math:`V[0] = V_0` (the mutable ``initial_volume``
    Param, registered as rolling-horizon state). There is no holdup constraint
    at ``t = 0`` (it would reference ``V[-1]``); the flows at the first time
    point drive no holdup change.

    **A tank has no on/off status** (architecture §3.4/§3.5, decision R6): this
    class forces the inherited ``unit_commitment`` config off regardless of
    what a caller passes, so no ``status[t]`` Binary or unit-commitment
    constraint is ever built for it — the canonical example of a physical
    subclass turning off a base capability.

    Config: the SISO/OpsBlock config plus ``min_volume`` (default 0 m³),
    ``max_volume`` (required, m³), and ``initial_volume`` (required, m³).

    Example:
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import StorageTank
        >>> m = dummy_time_block(4)
        >>> m.tank = StorageTank(
        ...     property_package=m.properties, max_volume=1000.0,
        ...     initial_volume=200.0,
        ... )
    """

    CONFIG = SISOBlockData.CONFIG()
    # A tank cannot be shut off: replace the inherited unit_commitment entry's
    # domain and default so the stored config is off no matter what a caller
    # passes, before any build step (including future M08 logic) reads it (R6).
    CONFIG.get("unit_commitment").set_domain(_unit_commitment_off)
    CONFIG.get("unit_commitment").set_default_value(UnitCommitmentConfig(status=False))
    CONFIG.declare(
        "min_volume",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Lower bound on the stored volume, in m^3 (default 0).",
        ),
    )
    CONFIG.declare(
        "max_volume",
        ConfigValue(
            default=None,
            description="Required upper bound on the stored volume, in m^3; "
            "also the default value of the fixable `capacity` design Var.",
        ),
    )
    CONFIG.declare(
        "initial_volume",
        ConfigValue(
            default=None,
            description="Required stored volume at the first time point, in "
            "m^3; becomes a mutable Param registered as rolling-horizon state.",
        ),
    )

    def build(self) -> None:
        """Build the SISO base (minus its balance), holdup dynamics, and bounds.

        Raises:
            FlexConfigError: If ``max_volume`` or ``initial_volume`` is missing.
        """
        super().build()
        for required in ("max_volume", "initial_volume"):
            if self.config[required] is None:
                raise FlexConfigError(
                    f"StorageTank requires {required} (in m^3); e.g. "
                    "StorageTank(property_package=..., max_volume=1000.0, "
                    "initial_volume=200.0).",
                    field=required,
                )
        time_block = self._find_time_block()

        self.flow_in = pyo.Reference(self.properties_in[:].flow_vol)
        self.flow_in.doc = "Inlet volumetric flow"
        self.flow_out = pyo.Reference(self.properties_out[:].flow_vol)
        self.flow_out.doc = "Outlet volumetric flow"

        self.V = pyo.Var(
            time_block.time_index,
            initialize=self.config.initial_volume,
            bounds=(self.config.min_volume, self.config.max_volume),
            units=pyunits.m**3,
            doc="Stored liquid volume",
        )

        self.initial_volume = pyo.Param(
            initialize=self.config.initial_volume,
            mutable=True,
            units=pyunits.m**3,
            doc="Stored volume at the first time point (rolling-horizon state)",
        )
        time_block.register_initial_state(self.initial_volume)
        self.register_process_parameter(self.initial_volume, regressable=False)

        @self.Constraint(doc="Initial condition: V[0] equals initial_volume")
        def initial_condition(blk):
            return blk.V[0] == blk.initial_volume

        dt_hours = pyunits.convert(time_block.dt, to_units=pyunits.hr)

        @self.Constraint(
            list(time_block.time_index)[1:],
            doc="Backwards-difference holdup equation: V[t] = V[t-1] + dt * "
            "(flow_in[t] - flow_out[t]), dt in hours so m^3/hr x hr = m^3",
        )
        def holdup_balance(blk, t):
            return blk.V[t] == blk.V[t - 1] + dt_hours * (
                blk.flow_in[t] - blk.flow_out[t]
            )

        self.capacity = pyo.Var(
            initialize=self.config.max_volume,
            units=pyunits.m**3,
            doc="Fixable design capacity, fixed to max_volume by default "
            "(M07's design mode unfixes it)",
        )
        self.capacity.fix()

        @self.Constraint(
            time_block.time_index,
            doc="Stored volume cannot exceed the design capacity",
        )
        def capacity_limit(blk, t):
            return blk.V[t] <= blk.capacity

        self.register_io_variable(self.flow_in, role="input")
        self.register_io_variable(self.flow_out, role="input")
        self.register_io_variable(self.V, role="output")
        # No register_power call: the tank draws nothing.

    def _build_mass_balance(self, time_block: TimeBlockData) -> None:
        """Build no pass-through balance; the holdup equation replaces it.

        In a pump inlet == outlet; in a tank they differ by the stored volume,
        so the SISO per-stream balance is never written for this unit.

        Args:
            time_block: The model's TimeBlock (unused).
        """
