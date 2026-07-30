"""SimpleAqueousFlow: the minimal flow-carrying property package (§3.7).

A minimal IDAES ``PhysicalParameterBlock``/``StateBlock`` pair carrying a
volumetric flow and a mass density, structurally modeled on WaterTAP's
zero-order package (``prop_ZO``). Ports built from these state blocks carry
flow between flex-pse units via standard IDAES/Pyomo ``Arc``s.

Volumetric flow is *extensive* (conserved across an arc); density and, when
enabled, pressure and temperature are *intensive* (equal across an arc / at a
node). The topology base classes build ports honoring that distinction
(``Port.Extensive`` for flow, ``Port.Equality`` for the intensive states).
Pressure and temperature are **opt-in** (default off); density is fixed at
the configured value by default (``fixed_density=True``) so the v0 default
stays flow-only in its degrees of freedom.
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
from pyomo.environ import NonNegativeReals, PositiveReals, Var, value
from pyomo.environ import units as pyunits

from flexcore.exceptions import FlexConfigError


@declare_process_block_class("SimpleAqueousFlow")
class SimpleAqueousFlowData(PhysicalParameterBlock):
    """Parameter block for a simple aqueous stream.

    Config options (see the CONFIG entries below):

    * ``fixed_density`` (default True) fixes each state block's ``dens_mass``
      state variable at ``density`` (default ``1000 kg/m^3``).
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
            description="Whether state blocks fix dens_mass at 'density'.",
        ),
    )
    CONFIG.declare(
        "density",
        ConfigValue(
            default=1000 * pyunits.kg / pyunits.m**3,
            description="Units-carrying mass density used to initialize "
            "dens_mass (and fix it when fixed_density is True).",
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
        """Set the state-block class and the single liquid phase/component."""
        super().build()
        self._state_block_class = SimpleAqueousStateBlock

        self.Liq = LiquidPhase()
        self.H2O = Component()

    def get_flow_basis_var_name(self) -> str:
        """Return the name of this package's extensive flow state variable.

        Lets callers (e.g. ``Tank``'s pass-through wiring) exclude "the
        flow" from a generic pass-through without hardcoding a variable name
        that varies by property package (a future mass/TDS package would
        return ``"flow_mass_phase_comp"`` instead).

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
    """State block carrying volumetric flow, density, and optional extras.

    State variables are indexed over time directly: the owning unit passes the
    ``time_index`` Set via ``build_state_block(time_index=...)`` and gets a
    single scalar state block whose variables span the horizon. Extensive,
    per-phase quantities lead with time then phase (``flow_vol_phase[t, phase]``);
    intensive stream properties drop the phase index (``dens_mass[t]``, and the
    opt-in ``pressure[t]``/``temperature[t]``, assumed equal across phases).
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
        """Create time-indexed ``flow_vol_phase``, ``dens_mass``, and extras."""
        super().build()
        time = self.config.time_index
        if time is None:
            raise FlexConfigError(
                "SimpleAqueousStateBlock requires a time_index; build it with "
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
            initialize=value(
                pyunits.convert(self.params.config.density, pyunits.kg / pyunits.m**3)
            ),
            domain=PositiveReals,
            units=pyunits.kg / pyunits.m**3,
            doc="Mass density",
        )
        if self.params.config.fixed_density:
            self.dens_mass.fix()
        if self.params.config.has_pressure:
            self.pressure = Var(
                time,
                initialize=101325.0,
                domain=PositiveReals,
                units=pyunits.Pa,
                doc="Pressure",
            )
        if self.params.config.has_temperature:
            self.temperature = Var(
                time,
                initialize=298.15,
                domain=PositiveReals,
                units=pyunits.K,
                doc="Temperature",
            )

    def define_state_vars(self) -> dict:
        """Return the state-variable dict (flow and density plus enabled ones)."""
        state_vars = {
            "flow_vol_phase": self.flow_vol_phase,
            "dens_mass": self.dens_mass,
        }
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
