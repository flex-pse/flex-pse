"""SimpleAqueousFlow: the minimal flow-carrying property package (§3.7).

A minimal IDAES ``PhysicalParameterBlock``/``StateBlock`` pair whose one required
state variable is a volumetric flow, structurally modeled on WaterTAP's zero-order
package (``prop_ZO``). Ports built from these state blocks carry flow between
flex-pse units via standard IDAES/Pyomo ``Arc``s.

Volumetric flow is *extensive* (conserved across an arc); when enabled, pressure
and temperature are *intensive* (equal across an arc / at a node). The topology
base classes build ports honoring that distinction (``Port.Extensive`` for flow,
``Port.Equality`` for pressure/temperature) in M09. Pressure and temperature are
**opt-in** (default off) so the v0 default stays flow-only.
"""

from idaes.core import (
    Component,
    LiquidPhase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
)
from idaes.core.util.initialization import fix_state_vars, revert_state_vars
from pyomo.common.config import ConfigValue
from pyomo.environ import NonNegativeReals, Param, PositiveReals, Var
from pyomo.environ import units as pyunits


@declare_process_block_class("SimpleAqueousFlow")
class SimpleAqueousFlowData(PhysicalParameterBlock):
    """Parameter block for a flow-only aqueous stream.

    Config options (see the CONFIG entries below):

    * ``fixed_density`` (default True) carries a fixed ``dens_mass`` Param whose
      value is ``density`` (default ``1000 kg/m^3``).
    * ``has_pressure`` / ``has_temperature`` (default False) add the intensive
      ``pressure`` / ``temperature`` state variables.

    Example:
        >>> import pyomo.environ as pyo
        >>> import flexops as fo
        >>> m = pyo.ConcreteModel()
        >>> m.props = fo.SimpleAqueousFlow(fixed_density=True)
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare(
        "fixed_density",
        ConfigValue(
            default=True,
            domain=bool,
            description="Whether to carry a fixed mass-density parameter.",
        ),
    )
    CONFIG.declare(
        "density",
        ConfigValue(
            default=1000 * pyunits.kg / pyunits.m**3,
            description="Fixed mass density used when fixed_density is True.",
        ),
    )
    CONFIG.declare(
        "has_pressure",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether state blocks carry an intensive pressure state "
            "variable (equal across arcs).",
        ),
    )
    CONFIG.declare(
        "has_temperature",
        ConfigValue(
            default=False,
            domain=bool,
            description="Whether state blocks carry an intensive temperature "
            "state variable (equal across arcs).",
        ),
    )

    def build(self) -> None:
        """Set the state-block class, one phase/component, and density."""
        super().build()
        self._state_block_class = SimpleAqueousStateBlock

        self.Liq = LiquidPhase()
        self.H2O = Component()

        if self.config.fixed_density:
            self.dens_mass = Param(
                initialize=self.config.density,
                units=pyunits.kg / pyunits.m**3,
                mutable=True,
                doc="Fixed mass density of the aqueous stream",
            )

    @classmethod
    def define_metadata(cls, obj) -> None:
        """Declare supported properties and default units (all five required)."""
        obj.add_properties(
            {
                "flow_vol": {"method": None, "units": "m^3/hr"},
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


class _SimpleAqueousStateBlock(StateBlock):
    """Whole-set methods for SimpleAqueous state blocks (fix/unfix hooks)."""

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


@declare_process_block_class(
    "SimpleAqueousStateBlock", block_class=_SimpleAqueousStateBlock
)
class SimpleAqueousStateBlockData(StateBlockData):
    """State block carrying a single volumetric-flow state variable."""

    def build(self) -> None:
        """Create ``flow_vol`` and any enabled intensive state variables."""
        super().build()
        self.flow_vol = Var(
            initialize=1.0,
            domain=NonNegativeReals,
            units=pyunits.m**3 / pyunits.hr,
            doc="Volumetric flowrate",
        )
        if self.params.config.has_pressure:
            self.pressure = Var(
                initialize=101325.0,
                domain=PositiveReals,
                units=pyunits.Pa,
                doc="Pressure",
            )
        if self.params.config.has_temperature:
            self.temperature = Var(
                initialize=298.15,
                domain=PositiveReals,
                units=pyunits.K,
                doc="Temperature",
            )

    def define_state_vars(self) -> dict:
        """Return the state-variable dict (``flow_vol`` plus any enabled ones)."""
        state_vars = {"flow_vol": self.flow_vol}
        if hasattr(self, "pressure"):
            state_vars["pressure"] = self.pressure
        if hasattr(self, "temperature"):
            state_vars["temperature"] = self.temperature
        return state_vars

    def define_display_vars(self) -> dict:
        """Return the display-variable dict for reporting."""
        return {name: var for name, var in self.define_state_vars().items()}


# ``declare_process_block_class`` injects the constructible ``SimpleAqueousStateBlock``
# wrapper into this module's namespace at runtime; bind the name explicitly so
# static tools resolve the forward reference in ``SimpleAqueousFlowData.build``.
SimpleAqueousStateBlock = globals()["SimpleAqueousStateBlock"]
