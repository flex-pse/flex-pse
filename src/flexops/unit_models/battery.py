r"""BatteryModel: SOC dynamics + first-class DERMS dispatch (M08, §3.4/§3.6, R4/R9).

No fluid ports (no ``property_package`` needed): a battery is an energy-only
unit with two dispatch-input actuators, ``power_charge[t]``/
``power_discharge[t]`` (kW), and a state of charge tracked in absolute energy
terms, ``charge[t]`` (kWh). ``capacity`` is the fixable sizing ``Var`` (R4):
fixed at the constructor value by default (operations mode); a
``costing_package=`` associates it with :meth:`FlexCostingData.set_design_mode`/
:meth:`~flexops.costing.flex_costing.FlexCostingData.set_operations_mode`.

* ``UnitCommitmentConfig.status`` defaults ``True`` project-wide; here, when
  ``True``, :func:`~flexops.logic.status.add_status` is attached to
  ``power_charge`` and a symmetric ``discharge_exclusivity`` constraint bounds
  ``power_discharge`` by ``(1 - status)`` -- one binary gives
  mutually-exclusive charge/discharge. This requires both
  ``power_charge_max``/``power_discharge_max`` (the semicontinuous link needs a
  finite bound); a caller who wants an unbounded, non-UC battery passes
  ``unit_commitment=UnitCommitmentConfig(status=False)``.
* ``soc[t]`` is a Var, bounded by ``soc_min``/``soc_max``, tied to
  ``charge[t]``/``capacity`` by the equality Constraint ``soc_capacity_link``
  (``charge[t] == soc[t] * capacity``), matching the milestone spec literally.
  This is a genuine product of two *free* Vars whenever ``capacity`` is
  unfixed (design mode) -- an unconditional bilinear equality that forces
  that solve to NLP. Accepted at explicit user request: a single bilinear
  equality per timestep is not a significant modeling burden.
* ``charge_balance`` covers **every** ``t``, including ``t=0`` (referencing
  ``charge_init`` in place of ``charge[t-1]`` at the boundary), rather than
  the spec's separate ``t=1..N-1`` difference equation plus an unrelated
  ``charge[0] == charge_init``/``soc[0] == soc_init`` initial condition.
  Leaving ``t=0`` governed only by a hard pin on ``charge[0]`` -- as
  ``Tank.initial_volume_eq`` does -- would leave
  ``power_charge[0]``/``power_discharge[0]`` completely free of any
  energy-conservation tie: the MIP arbitrage test caught this directly, with
  the solver "discharging" a free 50 kW at ``t=0`` for a cost rebate with no
  physical backing. Folding ``t=0`` into ``charge_balance`` closes that hole
  and drops the need for a separate, differently-named initial-condition
  constraint.
* ``soh`` (state of health) and ``soh_capacity_limit`` are **not** part of the
  M08 spec; added at explicit user request as a v0 placeholder ahead of any
  future degradation-modeling milestone. ``soh`` is fixed at the constructor
  value (default 0.85, mid-life) like ``capacity``.
* ``eta_charge``/``eta_discharge`` are fixed **Vars** (registered
  ``regressable=True``), not plain floats read from config -- also added at
  explicit user request, matching :class:`~flexops.unit_models.pump.Pump`'s
  ``efficiency`` pattern. ``soh`` is registered the same way. FlexParameterize
  fits all three from SCADA data: ``power_charge``/``power_discharge`` in,
  ``charge`` (the SOC state, registered as the process output) out, ``capacity``
  known.
* ``charge_leakage_rate`` (self-discharge) is **not** part of the M08 spec;
  added at explicit user request as a v0 self-discharge proxy. Like
  ``eta_charge``/``eta_discharge``/``soh``, it is a fixed **Var** (fraction of
  stored charge lost per day, default ``0.0005`` i.e. 0.05%/day) registered
  ``regressable=True``, so a future FlexParameterize regression may fit it
  from an observed ``charge`` decay trajectory alongside the efficiencies.

.. math::

    \text{charge}[t] = \text{charge}[t-1] \left(1 - \lambda \, \Delta t \right)
        + \Delta t \left(
        \eta_{charge} \, P_{charge}[t] - \frac{P_{discharge}[t]}{\eta_{discharge}}
    \right), \quad t = 0, \dots, N-1 \;\; (\text{charge}[-1] := \text{charge\_init})

    \lambda = \text{charge\_leakage\_rate (fraction lost per day)}

Usage::

    >>> from flexops.testing import dummy_time_block
    >>> from flexops.unit_models import BatteryModel
    >>> from pyomo.environ import units as pyunits
    >>> m = dummy_time_block(4)
    >>> m.battery = BatteryModel(capacity=10 * pyunits.kWh)  # doctest: +SKIP

Config: see ``capacity``, ``power_charge_max``/``power_discharge_max``,
``eta_charge``/``eta_discharge``, ``soc_min``/``soc_max``, ``initial_soc``, ``soh``,
``charge_leakage_rate`` below, plus the inherited
``relaxation``/``unit_commitment``/``external_dispatch``/``costing_package``
(architecture §3.2).

**Behind-the-meter assumption (v0).** ``power_electrical[t]`` may go negative
(discharge exports power); any facility-level "net draw >= 0" constraint
belongs at the plant/costing level, not here.
"""

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.logic.status import add_status


def _fraction_domain(value):
    """ConfigValue domain: a fraction in [0, 1]."""
    if isinstance(value, (int, float)) and 0.0 <= value <= 1.0:
        return float(value)
    raise FlexConfigError(f"Expected a fraction in [0, 1], got {value!r}.", value=value)


def _efficiency_domain(value):
    """ConfigValue domain: an efficiency fraction in (0, 1]."""
    if isinstance(value, (int, float)) and 0.0 < value <= 1.0:
        return float(value)
    raise FlexConfigError(
        f"Expected an efficiency fraction in (0, 1], got {value!r}.", value=value
    )


@declare_process_block_class("BatteryModel")
class BatteryModelData(OpsBlockData):
    """A battery: SOC dynamics, fixable capacity, DERMS dispatch (module docstring)."""

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.declare(
        "capacity",
        ConfigValue(
            description="Initial battery energy capacity, kWh -- the fixed "
            "design Var value at construction. Required."
        ),
    )
    CONFIG.declare(
        "power_charge_max",
        ConfigValue(
            default=None,
            description="Maximum charging power, kW. None (default) leaves "
            "power_charge unbounded above; required (along with "
            "power_discharge_max) when unit_commitment.status is enabled, "
            "since the semicontinuous link needs a finite bound.",
        ),
    )
    CONFIG.declare(
        "power_discharge_max",
        ConfigValue(
            default=None,
            description="Maximum discharging power, kW. Same requirement as "
            "power_charge_max.",
        ),
    )
    CONFIG.declare(
        "eta_charge",
        ConfigValue(
            default=1.0,
            domain=_efficiency_domain,
            description="Charging efficiency, a fraction in (0, 1].",
        ),
    )
    CONFIG.declare(
        "eta_discharge",
        ConfigValue(
            default=1.0,
            domain=_efficiency_domain,
            description="Discharging efficiency, a fraction in (0, 1].",
        ),
    )
    CONFIG.declare(
        "soc_min",
        ConfigValue(
            default=0.0,
            domain=_fraction_domain,
            description="Minimum state of charge, a fraction of capacity in [0, 1].",
        ),
    )
    CONFIG.declare(
        "soc_max",
        ConfigValue(
            default=1.0,
            domain=_fraction_domain,
            description="Maximum state of charge, a fraction of capacity in [0, 1].",
        ),
    )
    CONFIG.declare(
        "initial_soc",
        ConfigValue(
            default=0.5,
            domain=_fraction_domain,
            description="Initial state of charge, a fraction of capacity in "
            "[0, 1]; fixes charge[0] via charge_init (rolling-horizon state).",
        ),
    )
    CONFIG.declare(
        "soh",
        ConfigValue(
            default=0.85,
            domain=_fraction_domain,
            description="State of health, a fraction of nameplate capacity "
            "still available due to degradation; fixes soh at construction. "
            "Default 0.85 represents a mid-life battery.",
        ),
    )
    CONFIG.declare(
        "charge_leakage_rate",
        ConfigValue(
            default=0.0005 / pyunits.day,
            description="Self-discharge (charge leakage) rate: fraction of "
            "stored charge lost per day, fixing charge_leakage_rate at "
            "construction. Default 0.0005/day (0.05%/day).",
        ),
    )

    def build(self) -> None:
        """Build capacity, power/charge dynamics, SOC bounds, and UC status."""
        super().build()
        tb = self._find_time_block()

        if self.config.unit_commitment.status and (
            self.config.power_charge_max is None
            or self.config.power_discharge_max is None
        ):
            raise FlexConfigError(
                "BatteryModel requires both power_charge_max and "
                "power_discharge_max when unit_commitment.status is enabled "
                "(the default): the mutually-exclusive charge/discharge link "
                "needs a finite bound. Pass both, or disable status via "
                "unit_commitment=UnitCommitmentConfig(status=False).",
                field="unit_commitment.status",
                value=True,
            )

        capacity_val = pyo.value(pyunits.convert(self.config.capacity, pyunits.kWh))
        self.capacity = pyo.Var(
            initialize=capacity_val,
            bounds=(0.0, None),
            units=pyunits.kWh,
            doc="Chosen battery energy capacity (design Var, R4); fixed at "
            "the constructor value by default (operations mode); "
            "costing.set_design_mode() unfixes it.",
        )
        self.capacity.fix(capacity_val)
        self.register_process_parameter(self.capacity, regressable=False)
        costing_package = self.config.costing_package
        if costing_package is not None:
            costing_package.register_sizing_variable(self.capacity)

        soh_val = self.config.soh
        self.soh = pyo.Var(
            initialize=soh_val,
            bounds=(0.0, 1.0),
            units=pyunits.dimensionless,
            doc="State of health: fraction of nameplate capacity still "
            "available due to degradation. Fixed at the constructor value "
            "(v0 has no degradation dynamics); default 0.85 is mid-life.",
        )
        self.soh.fix(soh_val)
        self.register_process_parameter(self.soh, regressable=True)

        charge_max_val = (
            pyo.value(pyunits.convert(abs(self.config.power_charge_max), pyunits.kW))
            if self.config.power_charge_max is not None
            else None
        )
        discharge_max_val = (
            pyo.value(pyunits.convert(abs(self.config.power_discharge_max), pyunits.kW))
            if self.config.power_discharge_max is not None
            else None
        )

        self.power_charge = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0.0, charge_max_val),
            units=pyunits.kW,
            doc="Charging power draw (dispatch input).",
        )
        self.power_discharge = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0.0, discharge_max_val),
            units=pyunits.kW,
            doc="Discharging power output (dispatch input).",
        )
        self.register_io_variable(self.power_charge, role="input")
        self.register_io_variable(self.power_discharge, role="input")

        # declare_power builds power_electrical with no domain kwarg, so it
        # defaults to Reals (Pitfall 3) -- discharge must export power as a
        # negative draw.
        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.register_io_variable(power, role="output")

        @self.Constraint(
            tb.time_index,
            doc="power_electrical == power_charge - power_discharge "
            "(discharge is a negative/export draw; v0 is behind-the-meter).",
        )
        def net_electrical(b, t):
            return power[t] == b.power_charge[t] - b.power_discharge[t]

        self.charge = pyo.Var(
            tb.time_index,
            bounds=(0.0, None),
            units=pyunits.kWh,
            doc="Stored energy content.",
        )

        initial_charge_val = self.config.initial_soc * capacity_val
        self.charge_init = pyo.Param(
            initialize=initial_charge_val,
            mutable=True,
            units=pyunits.kWh,
            doc="Initial stored energy, charge[0] (rolling-horizon initial state).",
        )
        tb.register_initial_state(self.charge_init)
        self.register_process_parameter(self.charge_init, regressable=False)

        self.eta_charge = pyo.Var(
            initialize=self.config.eta_charge,
            bounds=(0.0, 1.0),
            units=pyunits.dimensionless,
            doc="Charging efficiency. Fixed at the configured value by "
            "default; a future FlexParameterize regression may estimate it "
            "from SCADA data (power_charge/power_discharge in, charge/soc "
            "out, capacity known).",
        )
        self.eta_charge.fix(self.config.eta_charge)
        self.register_process_parameter(self.eta_charge, regressable=True)

        self.eta_discharge = pyo.Var(
            initialize=self.config.eta_discharge,
            bounds=(0.0, 1.0),
            units=pyunits.dimensionless,
            doc="Discharging efficiency. Fixed at the configured value by "
            "default; a future FlexParameterize regression may estimate it "
            "the same way as eta_charge.",
        )
        self.eta_discharge.fix(self.config.eta_discharge)
        self.register_process_parameter(self.eta_discharge, regressable=True)

        leakage_rate_val = pyo.value(
            pyunits.convert(self.config.charge_leakage_rate, pyunits.day**-1)
        )
        self.charge_leakage_rate = pyo.Var(
            initialize=leakage_rate_val,
            bounds=(0.0, 0.001),  # 0.1%/day is an upper bound for a Li-ion battery
            units=pyunits.day**-1,
            doc="Self-discharge rate: fraction of stored charge lost per day "
            "(applied to charge[t-1] each step in charge_balance). Fixed at "
            "the configured value by default; a future FlexParameterize "
            "regression may estimate it from an observed charge decay "
            "trajectory, the same way as eta_charge/eta_discharge.",
        )
        self.charge_leakage_rate.fix(leakage_rate_val)
        self.register_process_parameter(self.charge_leakage_rate, regressable=True)

        @self.Constraint(
            tb.time_index,
            doc="Charge holdup (backward difference, conventions §2): "
            "charge[t] == charge[t-1]*(1 - charge_leakage_rate*dt) + "
            "dt*(eta_charge*power_charge[t] - "
            "power_discharge[t]/eta_discharge); t=0 references charge_init in "
            "place of charge[-1], so power_charge[0]/power_discharge[0] are "
            "energy-conserving too (a separate, unconstrained-at-t=0 initial "
            "condition would let power_charge[0]/power_discharge[0] move for "
            "free, manufacturing energy from nothing).",
        )
        def charge_balance(b, t):
            previous = b.charge[t - 1] if t > 0 else b.charge_init
            delta_charge = pyunits.convert(
                tb.dt
                * (
                    b.eta_charge * b.power_charge[t]
                    - b.power_discharge[t] / b.eta_discharge
                ),
                to_units=pyunits.kWh,
            )
            leakage = pyunits.convert(
                previous * b.charge_leakage_rate * tb.dt, to_units=pyunits.kWh
            )
            return b.charge[t] == previous + delta_charge - leakage

        soc_min = self.config.soc_min
        soc_max = self.config.soc_max

        self.soc = pyo.Var(
            tb.time_index,
            bounds=(soc_min, soc_max),
            units=pyunits.dimensionless,
            doc="State of charge, a fraction of capacity in [soc_min, "
            "soc_max]; tied to charge/capacity by soc_capacity_link.",
        )

        @self.Constraint(
            tb.time_index,
            doc="Ties soc to stored energy: charge[t] == soc[t] * capacity "
            "(a bilinear equality when capacity is unfixed -- see the module "
            "docstring's 'Deviations from the milestone spec').",
        )
        def soc_capacity_link(b, t):
            return b.charge[t] == b.soc[t] * b.capacity

        @self.Constraint(
            tb.time_index,
            doc="State-of-health capacity limit: charge[t] <= soh * "
            "capacity. Independent of soc_max -- soh models capacity fade "
            "from degradation, while soc_max is an operational headroom "
            "fraction of nameplate capacity.",
        )
        def soh_capacity_limit(b, t):
            return b.charge[t] <= b.soh * b.capacity

        if self.config.unit_commitment.status:
            status = add_status(
                self,
                self.power_charge,
                0.0 * pyunits.kW,
                charge_max_val * pyunits.kW,
            )

            @self.Constraint(
                tb.time_index,
                doc="Mutually-exclusive discharge link: power_discharge[t] <= "
                "power_discharge_max * (1 - status[t]) (a battery may not "
                "charge and discharge in the same step).",
            )
            def discharge_exclusivity(b, t):
                return b.power_discharge[t] <= discharge_max_val * (1 - status[t])

        # charge (SOC state) is the regression target for eta_charge/
        # eta_discharge/soh: FlexParameterize fits them from SCADA power
        # in/out against the observed charge trajectory, using capacity.
        self.register_io_variable(self.charge, role="output")

    def set_dispatch(self, series) -> None:
        """Fix net battery dispatch from an external (DERMS) command series (R9).

        Splits ``series[t]`` (signed net kW, positive = charging, negative =
        discharging) into the ``power_charge``/``power_discharge`` actuators
        and fixes both via
        :meth:`~flexops.core.ops_block.OpsBlockData.set_external_dispatch`,
        removing the dispatch degree of freedom while leaving ``capacity``
        free (Pitfall 9). Fixing only the net ``power_electrical`` would
        leave the charge/discharge split underdetermined -- their
        efficiencies differ, so the split affects the SOC trajectory -- so
        both actuators are pinned directly instead.

        Args:
            series: A mapping or pandas Series of signed net dispatch power
                (kW), aligned to the time set (see
                :meth:`~flexops.core.ops_block.OpsBlockData.set_external_dispatch`).
        """
        tb = self._find_time_block()
        resolved = self._resolve_dispatch_series(series, tb)
        charge_series = {t: max(v, 0.0) for t, v in resolved.items()}
        discharge_series = {t: max(-v, 0.0) for t, v in resolved.items()}
        self.set_external_dispatch(self.power_charge, charge_series)
        self.set_external_dispatch(self.power_discharge, discharge_series)
