"""Pump(SISOBlock): electrical pump with two selectable power laws (architecture §3.4).

Inherits its inlet/outlet ports and pass-through mass balance from
:class:`~flexops.unit_models.base.siso.SISOBlockData`; adds only the
flow-to-power relationship, chosen via ``config.power_relation``.
"""

import enum

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops.unit_models.base.siso import SISOBlockData


class PumpPowerRelation(enum.StrEnum):
    """Which flow-to-power law a :class:`Pump` enforces."""

    CONSTANT_INTENSITY = "constant_intensity"
    HYDRAULIC = "hydraulic"


def _power_relation_domain(value):
    """ConfigValue domain: coerce to a :class:`PumpPowerRelation`."""
    try:
        return PumpPowerRelation(value)
    except ValueError as exc:
        allowed = ", ".join(repr(p.value) for p in PumpPowerRelation)
        raise FlexConfigError(
            f"power_relation must be one of {allowed}, got {value!r}.",
            field="power_relation",
            value=value,
        ) from exc


def _efficiency_domain(value):
    """ConfigValue domain: efficiency must be a fraction in (0, 1]."""
    if isinstance(value, (int, float)) and 0 < value <= 1:
        return float(value)
    raise FlexConfigError(
        f"efficiency must be a float in (0, 1], got {value!r}.",
        field="efficiency",
        value=value,
    )


@declare_process_block_class("Pump")
class PumpData(SISOBlockData):
    r"""A pump with a constant energy intensity, or a hydraulic power law.

    ``config.power_relation`` selects one of:

    * ``"constant_intensity"`` (default):

      .. math::

          P_{elec}[t] = \text{energy\_intensity} \cdot \dot{V}_{in}[t]

      ``energy_intensity`` is in kWh/m^3 and the inlet ``flow_vol_phase`` in
      m^3/hr, so kWh/m^3 * m^3/hr = kWh/hr = kW -- dimensionally exact with no
      fudge factor.

    * ``"hydraulic"``:

      .. math::

          P_{elec}[t] = \frac{\Delta P[t] \cdot \dot{V}_{in}[t]}{\eta}

      :math:`\Delta P[t]` is ``delta_pressure[t]``, a ``Var`` tied to the
      inlet/outlet pressure states by the ``pressure_change`` equality
      constraint (``delta_pressure[t] == outlet_state.pressure[t] -
      inlet_state.pressure[t]``) -- so this relation requires a
      ``property_package`` built with ``has_pressure=True``, and excludes
      ``pressure`` from the inherited
      inlet-to-outlet bypass (a pump raises pressure between its ports; it
      does not pass it through unchanged). Both ``inlet_state.pressure`` and
      ``outlet_state.pressure`` are registered as IO inputs -- boundary
      conditions the caller fixes, like a fixed dispatch flow. ``efficiency``
      (:math:`\eta`) is a dimensionless fraction in (0, 1].

    Config:
        Inherits the SISO/OpsBlock config; adds ``power_relation`` (default
        ``"constant_intensity"``), ``energy_intensity`` (default 0.5
        kWh/m^3, used only when ``power_relation="constant_intensity"``),
        and ``efficiency`` (default 0.7, used only when
        ``power_relation="hydraulic"``).

    Example:
        >>> from flexops import SimpleAqueousFlow
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import Pump
        >>> m = dummy_time_block(3)
        >>> m.pump = Pump(property_package=m.properties)  # doctest: +SKIP
        >>> m.hydraulic_properties = SimpleAqueousFlow(has_pressure=True)
        >>> m.hydraulic_pump = Pump(
        ...     property_package=m.hydraulic_properties,
        ...     power_relation="hydraulic",
        ...     efficiency=0.75,
        ... )  # doctest: +SKIP
    """

    CONFIG = SISOBlockData.CONFIG()
    CONFIG.declare(
        "power_relation",
        ConfigValue(
            default=PumpPowerRelation.CONSTANT_INTENSITY,
            domain=_power_relation_domain,
            description="Flow-to-power law: 'constant_intensity' (power = "
            "energy_intensity * flow) or 'hydraulic' (power = delta_pressure "
            "* flow / efficiency, delta_pressure tied to inlet/outlet "
            "pressure by the pressure_change constraint).",
        ),
    )
    CONFIG.declare(
        "energy_intensity",
        ConfigValue(
            default=0.5 * pyunits.kWh / pyunits.m**3,
            description="Electrical energy per unit volume pumped (a mutable "
            "Param once built), kWh/m^3. Used only when "
            "power_relation='constant_intensity'.",
        ),
    )
    CONFIG.declare(
        "efficiency",
        ConfigValue(
            default=0.7,
            domain=_efficiency_domain,
            description="Pump hydraulic efficiency, a dimensionless fraction "
            "in (0, 1] (a Var fixed at this value once built). Used only "
            "when power_relation='hydraulic'.",
        ),
    )

    def build(self) -> None:
        """Build the SISO base, then the configured relation's Param(s)/power draw."""
        super().build()
        tb = self._find_time_block()

        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.register_io_variable(power, role="output")

        if self.config.power_relation == PumpPowerRelation.HYDRAULIC:
            self._build_hydraulic_relation(tb, power)
        else:
            self._build_constant_intensity_relation(tb, power)

    def _build_mass_balance(self) -> None:
        """Bypass every state var, except pressure under the hydraulic relation.

        A pump raises pressure between its ports rather than passing it
        through unchanged, so ``pressure`` is excluded from the generic
        bypass and instead governed by :meth:`_build_hydraulic_relation`.
        """
        if self.config.power_relation == PumpPowerRelation.HYDRAULIC:
            self.add_bypass_constraints(
                self.inlet, self.outlet, exclude_vars=("pressure",)
            )
        else:
            super()._build_mass_balance()

    def _build_constant_intensity_relation(self, tb, power) -> None:
        """Build ``energy_intensity`` and the constant-intensity ``power_eq``."""
        self.energy_intensity = pyo.Param(
            initialize=pyo.value(
                pyunits.convert(
                    self.config.energy_intensity, pyunits.kWh / pyunits.m**3
                )
            ),
            mutable=True,
            units=pyunits.kWh / pyunits.m**3,
            doc="Electrical energy per unit volume pumped.",
        )
        self.register_process_parameter(self.energy_intensity, regressable=True)

        @self.Constraint(
            tb.time_index,
            doc="power_electrical = energy_intensity * inlet flow; "
            "kWh/m^3 * m^3/hr = kW exactly, no fudge factor.",
        )
        def power_eq(b, t):
            return power[t] == pyunits.convert(
                b.energy_intensity * b.inlet_state.flow_vol_phase[t, "Liq"],
                pyunits.kW,
            )

    def _build_hydraulic_relation(self, tb, power) -> None:
        """Register inlet/outlet pressure as IO, then build efficiency/power_eq.

        Raises:
            FlexConfigError: If the configured ``property_package`` was not
                built with ``has_pressure=True`` (no ``pressure`` state to
                compute ``delta_pressure`` from).
        """
        if not hasattr(self.inlet_state, "pressure"):
            raise FlexConfigError(
                "power_relation='hydraulic' requires a property_package "
                "built with has_pressure=True (delta_pressure is computed "
                "from the inlet/outlet pressure states).",
                field="property_package",
                value=self.config.property_package,
            )
        self.register_io_variable(self.inlet_state.pressure, role="input")
        self.register_io_variable(self.outlet_state.pressure, role="input")

        self.delta_pressure = pyo.Var(
            tb.time_index,
            initialize=0.0,
            units=pyunits.Pa,
            doc="Pump pressure rise: outlet pressure - inlet pressure.",
        )

        @self.Constraint(
            tb.time_index,
            doc="delta_pressure = outlet pressure - inlet pressure.",
        )
        def pressure_change(b, t):
            return (
                b.delta_pressure[t]
                == b.outlet_state.pressure[t] - b.inlet_state.pressure[t]
            )

        self.efficiency = pyo.Var(
            initialize=self.config.efficiency,
            bounds=(0.0, 1.0),
            units=pyunits.dimensionless,
            doc="Pump hydraulic efficiency. Fixed at the configured value "
            "by default; a future design mode may unfix it, subject to "
            "this (0, 1] bound.",
        )
        self.efficiency.fix(self.config.efficiency)
        self.register_process_parameter(self.efficiency, regressable=True)

        @self.Constraint(
            tb.time_index,
            doc="power_electrical = delta_pressure * inlet flow / efficiency.",
        )
        def power_eq(b, t):
            return power[t] == pyunits.convert(
                b.delta_pressure[t]
                * b.inlet_state.flow_vol_phase[t, "Liq"]
                / b.efficiency,
                pyunits.kW,
            )
