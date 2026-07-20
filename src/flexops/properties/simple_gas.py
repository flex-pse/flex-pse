"""SimpleGasFlow: the minimal gas-stream property package (§3.7).

The gas-phase counterpart of
:mod:`flexops.properties.simple_aqueous`. Where the aqueous package is
flow-only with opt-in extras, a gas stream's density varies with pressure and
temperature, so every ``SimpleGasFlow`` state block **always** carries four
state variables: ``flow_vol_phase``, ``dens_mass``, ``pressure``, and
``temperature``. No equation of state links them — units add whatever relation
they need as their own constraints.

Volumetric flow is *extensive* (conserved across an arc); density, pressure,
and temperature are *intensive* (equal across an arc / at a node). The topology
base classes build ports honoring that distinction (``Port.Extensive`` for
flow, ``Port.Equality`` for the intensive states) in M09.
"""

from idaes.core import (
    Component,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    VaporPhase,
    declare_process_block_class,
)
from idaes.core.util.initialization import fix_state_vars, revert_state_vars
from pyomo.common.config import ConfigValue
from pyomo.environ import NonNegativeReals, PositiveReals, Var
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError


@declare_process_block_class("SimpleGasFlow")
class SimpleGasFlowData(PhysicalParameterBlock):
    """Parameter block for a simple gas stream.

    State blocks built from this package always carry ``flow_vol_phase``,
    ``dens_mass``, ``pressure``, and ``temperature``; there are no config
    options beyond the ``PhysicalParameterBlock`` base.

    Example:
        >>> import pyomo.environ as pyo
        >>> import flexops as fo
        >>> m = pyo.ConcreteModel()
        >>> m.props = fo.SimpleGasFlow()
    """

    CONFIG = PhysicalParameterBlock.CONFIG()

    def build(self) -> None:
        """Set the state-block class and the single vapor phase/component."""
        super().build()
        self._state_block_class = SimpleGasStateBlock

        self.Vap = VaporPhase()
        self.gas = Component()

    def get_flow_basis_var_name(self) -> str:
        """Return the name of this package's extensive flow state variable.

        Returns:
            The state-variable name carrying extensive flow, ``"flow_vol_phase"``.
        """
        return "flow_vol_phase"

    @classmethod
    def define_metadata(cls, obj) -> None:
        """Declare supported properties and the five required default units."""
        obj.add_properties(
            {
                "flow_vol_phase": {"method": None, "units": "m^3/hr"},
                "dens_mass": {"method": None, "units": "kg/m^3"},
                "pressure": {"method": None, "units": "Pa"},
                "temperature": {"method": None, "units": "K"},
            }
        )
        obj.add_default_units(
            {
                "time": pyunits.hr,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


class _SimpleGasStateBlock(StateBlock):
    """Whole-set methods for SimpleGas state blocks (fix/unfix hooks)."""

    def fix_initialization_states(self) -> None:
        """Fix all state variables for initialization."""
        fix_state_vars(self)

    def initialize(self, *args, hold_state: bool = False, **kwargs):
        """Fix state vars for initialization; optionally hold them fixed.

        Args:
            hold_state: If True, leave the state vars fixed and return the flags
                needed to release them later.

        Returns:
            The fix flags if ``hold_state`` is True, else None.
        """
        flags = fix_state_vars(self)
        if hold_state:
            return flags
        self.release_state(flags)
        return None

    def release_state(self, flags, **kwargs) -> None:
        """Unfix state vars fixed during initialization.

        Args:
            flags: The fix flags returned by :meth:`initialize`.
        """
        if flags is not None:
            revert_state_vars(self, flags)


@declare_process_block_class("SimpleGasStateBlock", block_class=_SimpleGasStateBlock)
class SimpleGasStateBlockData(StateBlockData):
    """State block carrying flow, density, pressure, and temperature.

    State variables are indexed over time directly: the owning unit passes the
    ``time_index`` Set via ``build_state_block(time_index=...)`` and gets a
    single scalar state block whose variables span the horizon. Extensive,
    per-phase quantities lead with time then phase (``flow_vol_phase[t, phase]``);
    intensive stream properties drop the phase index (``dens_mass[t]``,
    ``pressure[t]``, ``temperature[t]``, assumed equal across phases).
    """

    CONFIG = StateBlockData.CONFIG()
    CONFIG.declare(
        "time_index",
        ConfigValue(
            default=None,
            description="Ordered Pyomo time Set the state variables are indexed "
            "over (the owning unit passes TimeBlock.time_index).",
        ),
    )

    def build(self) -> None:
        """Create the four time-indexed gas state variables."""
        super().build()
        time = self.config.time_index
        if time is None:
            raise FlexConfigError(
                "SimpleGasStateBlock requires a time_index; build it with "
                "build_state_block(time_index=tb.time_index).",
                field="time_index",
                value=None,
            )
        self.flow_vol_phase = Var(
            time,
            self.params.phase_list,
            initialize=1.0,
            domain=NonNegativeReals,
            units=pyunits.m**3 / pyunits.hr,
            doc="Volumetric flowrate by time and phase",
        )
        self.dens_mass = Var(
            time,
            initialize=1.2,
            domain=PositiveReals,
            units=pyunits.kg / pyunits.m**3,
            doc="Mass density",
        )
        self.pressure = Var(
            time,
            initialize=101325.0,
            domain=PositiveReals,
            units=pyunits.Pa,
            doc="Pressure",
        )
        self.temperature = Var(
            time,
            initialize=298.15,
            domain=PositiveReals,
            units=pyunits.K,
            doc="Temperature",
        )

    def define_state_vars(self) -> dict:
        """Return the state-variable dict (all four gas states)."""
        return {
            "flow_vol_phase": self.flow_vol_phase,
            "dens_mass": self.dens_mass,
            "pressure": self.pressure,
            "temperature": self.temperature,
        }

    def define_display_vars(self) -> dict:
        """Return the display-variable dict for reporting."""
        return {name: var for name, var in self.define_state_vars().items()}


# ``declare_process_block_class`` injects the constructible ``SimpleGasStateBlock``
# wrapper into this module's namespace at runtime; bind the name explicitly so
# static tools resolve the forward reference in ``SimpleGasFlowData.build``.
SimpleGasStateBlock = globals()["SimpleGasStateBlock"]
