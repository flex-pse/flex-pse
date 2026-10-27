"""Server(OpsBlockData): one server's IT power, PDU loss and heat release."""

from collections.abc import Sized

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.config.schema import UnitCommitmentConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData
from flexops.logic._records import suppress_logic_records
from flexops.logic.status import add_status


@declare_process_block_class("Server")
class ServerData(OpsBlockData):
    """A server whose step-mean draw follows its utilization.

    Config:
        Inherits the OpsBlock config; each added option is documented on its
        ``CONFIG`` entry.

    Raises:
        FlexConfigError: If an option is missing, out of range, or does not
            match the horizon.
    """

    allows_status_element = False
    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("unit_commitment").set_default_value(UnitCommitmentConfig(status=False))
    CONFIG.declare(
        "utilization",
        ConfigValue(
            description="Step-mean utilization in [0, 1]: a length-T sequence, or "
            "a live time-indexed Pyomo Param, Var or Expression. Required.",
        ),
    )
    CONFIG.declare(
        "idle_power",
        ConfigValue(description="IT power at zero utilization, kW. Required."),
    )
    CONFIG.declare(
        "peak_power",
        ConfigValue(description="IT power at full utilization, kW. Required."),
    )
    CONFIG.declare(
        "utilization_exponent",
        ConfigValue(
            default=1.0,
            domain=float,
            description="Exponent of the server power curve in utilization; 1 is "
            "linear. Values other than 1 need data utilization.",
        ),
    )
    CONFIG.declare(
        "fan_rated_power",
        ConfigValue(
            default=0.0 * pyunits.kW,
            description="Server fan power at rated_airflow, kW.",
        ),
    )
    CONFIG.declare(
        "rated_airflow",
        ConfigValue(
            default=None,
            description="Airflow at fan_rated_power, kg/s; required when "
            "fan_rated_power is non-zero.",
        ),
    )
    CONFIG.declare(
        "airflow",
        ConfigValue(
            default=None,
            description="Step-mean air mass flow through the server, a length-T "
            "sequence in kg/s, each positive; required when fan_rated_power is "
            "non-zero.",
        ),
    )
    CONFIG.declare(
        "liquid_fraction",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Fraction of server heat removed by the liquid loop rather "
            "than room air, in [0, 1].",
        ),
    )
    CONFIG.declare(
        "pdu_loss_fraction",
        ConfigValue(
            default=0.02,
            domain=float,
            description="Power distribution unit loss per kW of IT power.",
        ),
    )
    CONFIG.declare(
        "rated_power",
        ConfigValue(
            default=None,
            description="Server nameplate draw, kW; required with "
            "unit_commitment.status.",
        ),
    )

    def build(self) -> None:
        """Build the utilization input, IT power, heat split and draw."""
        super().build()
        tb = self._find_time_block()
        self._build_parameters()
        self._build_utilization(tb)
        self._build_fans(tb)
        self._build_server(tb)
        self._build_draw(tb)

    def _build_parameters(self) -> None:
        """Declare the server's fixed, registered coefficients.

        Raises:
            FlexConfigError: If a required power is missing or out of range.
        """
        cfg = self.config
        for field in ("idle_power", "peak_power"):
            if cfg[field] is None:
                raise FlexConfigError(f"{field} is required.", field=field, value=None)
        idle = self.declare_process_parameter(
            "idle_power",
            cfg.idle_power,
            pyunits.kW,
            "IT power at zero utilization.",
            bounds=(0, None),
        )
        peak = self.declare_process_parameter(
            "peak_power",
            cfg.peak_power,
            pyunits.kW,
            "IT power at full utilization.",
            bounds=(0, None),
        )
        if not 0.0 <= idle.value <= peak.value:
            raise FlexConfigError(
                f"peak_power ({peak.value} kW) must be at least idle_power "
                f"({idle.value} kW) and both non-negative.",
                field="peak_power",
                value=peak.value,
            )
        if not 0.0 <= cfg.liquid_fraction <= 1.0:
            raise FlexConfigError(
                f"liquid_fraction must lie in [0, 1], got {cfg.liquid_fraction}.",
                field="liquid_fraction",
                value=cfg.liquid_fraction,
            )
        self.declare_process_parameter(
            "utilization_exponent",
            cfg.utilization_exponent,
            pyunits.dimensionless,
            "Exponent of the server power curve in utilization.",
            bounds=(0.1, 3),
            regressable=False,
        )
        self.declare_process_parameter(
            "liquid_fraction",
            cfg.liquid_fraction,
            pyunits.dimensionless,
            "Fraction of server heat removed by the liquid loop.",
            bounds=(0, 1),
        )
        self.declare_process_parameter(
            "pdu_loss_fraction",
            cfg.pdu_loss_fraction,
            pyunits.dimensionless,
            "PDU loss per kW of IT power.",
            bounds=(0, 1),
        )

    def _build_utilization(self, tb) -> None:
        """Build ``utilization[t]``: fixed to data, or tied to a live component.

        Args:
            tb: The model's TimeBlock.

        Raises:
            FlexConfigError: If the source has the wrong length or range, or a
                live source is combined with a nonlinear power curve.
        """
        source = self.config.utilization
        live = callable(getattr(source, "is_indexed", None))
        if live:
            if not source.is_indexed() or len(source.index_set()) != tb.n_points:
                raise FlexConfigError(
                    "A live utilization source must be indexed by one entry per "
                    f"time step ({tb.n_points}).",
                    field="utilization",
                    value=source,
                )
            terms = [source[i] for i in source.index_set()]
        else:
            if not isinstance(source, Sized) or len(source) != tb.n_points:
                raise FlexConfigError(
                    f"utilization must be a sequence of {tb.n_points} values.",
                    field="utilization",
                    value=source,
                )
            terms = [float(value) for value in source]
        for t, term in enumerate(terms):
            try:
                value = pyo.value(term)
            except (ValueError, TypeError):
                continue
            if value is not None and not 0.0 <= value <= 1.0:
                raise FlexConfigError(
                    f"utilization[{t}] is {value}, outside [0, 1].",
                    field="utilization",
                    value=value,
                )
        if live and self.config.utilization_exponent != 1.0:
            raise FlexConfigError(
                "A live utilization source needs utilization_exponent == 1; the "
                "nonlinear power curve is precomputed from data.",
                field="utilization_exponent",
                value=self.config.utilization_exponent,
            )

        self.utilization = pyo.Var(
            tb.time_index,
            initialize=0.5 if live else dict(enumerate(terms)),
            bounds=(0, 1),
            units=pyunits.dimensionless,
            doc="Step-mean server utilization.",
        )
        if live:
            self.utilization_definition = pyo.Constraint(
                tb.time_index,
                rule=lambda b, t: b.utilization[t] == terms[t],
                doc="Utilization follows the configured live source.",
            )
        else:
            self.utilization.fix()
        self.register_io_variable(self.utilization, role="input")
        if self.config.utilization_exponent != 1.0:
            self.utilization_curve = pyo.Param(
                tb.time_index,
                mutable=True,
                initialize=0.0,
                units=pyunits.dimensionless,
                doc="utilization ** utilization_exponent.",
            )
            self._refresh_curve()

    def _refresh_curve(self) -> None:
        """Recompute ``utilization_curve`` from utilization and the exponent."""
        gamma = pyo.value(self.utilization_exponent)
        for t in self.utilization_curve:
            self.utilization_curve[t] = pyo.value(self.utilization[t]) ** gamma

    def update_parameters(self, values: dict) -> None:
        """Update registered parameters in place, then refresh the power curve.

        Args:
            values: Registered parameter name mapped to its new value.

        Raises:
            FlexConfigError: If a name is not a registered process parameter.
        """
        super().update_parameters(values)
        if self.find_component("utilization_curve") is not None:
            self._refresh_curve()

    def _build_fans(self, tb) -> None:
        """Build ``fan_power[t]`` from the airflow by the fan cube law.

        Args:
            tb: The model's TimeBlock.

        Raises:
            FlexConfigError: If the fans draw power without a rated airflow or a
                positive airflow at every step.
        """
        cfg = self.config
        fan_rated = self.declare_process_parameter(
            "fan_rated_power",
            cfg.fan_rated_power,
            pyunits.kW,
            "Server fan power at the rated airflow.",
            bounds=(0, None),
        )
        if not fan_rated.value:
            self.fan_power = pyo.Expression(
                tb.time_index, rule=0.0 * pyunits.kW, doc="Server fan power."
            )
            return
        if cfg.rated_airflow is None:
            raise FlexConfigError(
                "rated_airflow is required when fan_rated_power is non-zero.",
                field="rated_airflow",
                value=None,
            )
        airflow = cfg.airflow
        if (
            not isinstance(airflow, Sized)
            or len(airflow) != tb.n_points
            or min(airflow) <= 0
        ):
            raise FlexConfigError(
                f"airflow must be {tb.n_points} positive values when "
                "fan_rated_power is non-zero.",
                field="airflow",
                value=airflow,
            )
        self.declare_process_parameter(
            "rated_airflow",
            cfg.rated_airflow,
            pyunits.kg / pyunits.s,
            "Airflow at the rated fan power.",
            bounds=(1e-6, None),
            regressable=False,
        )
        self.airflow = pyo.Param(
            tb.time_index,
            initialize={t: float(v) for t, v in enumerate(airflow)},
            mutable=True,
            units=pyunits.kg / pyunits.s,
            doc="Step-mean air mass flow through the server.",
        )
        self.fan_power = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: b.fan_rated_power * (b.airflow[t] / b.rated_airflow) ** 3,
            doc="Server fan power: fan_rated_power * (airflow / rated_airflow)^3.",
        )

    def _build_server(self, tb) -> None:
        """Build IT power and the heat released to air and the liquid loop.

        Args:
            tb: The model's TimeBlock.
        """
        self.it_power = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Step-mean server IT power while running.",
        )
        self.register_io_variable(self.it_power, role="output")
        curve = self.find_component("utilization_curve")
        curve = self.utilization if curve is None else curve
        self.it_power_relation = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: b.it_power[t]
            == b.idle_power + (b.peak_power - b.idle_power) * curve[t] + b.fan_power[t],
            doc="IT power: idle + (peak - idle) * power curve + fans.",
        )
        self.register_relation(self.it_power_relation, target=self.it_power)
        self.heat_output = pyo.Var(
            tb.time_index,
            initialize=0.0,
            bounds=(0, None),
            units=pyunits.kW,
            doc="Step-mean heat released to room air and the liquid loop.",
        )
        self.register_io_variable(self.heat_output, role="output")
        self.air_heat = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: (1 - b.liquid_fraction) * b.heat_output[t],
            doc="Step-mean heat released to room air.",
        )
        self.liquid_heat = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: b.liquid_fraction * b.heat_output[t],
            doc="Step-mean heat removed by the liquid loop.",
        )

    def _build_draw(self, tb) -> None:
        """Build the draw and heat, gated by status when it is enabled.

        Args:
            tb: The model's TimeBlock.

        Raises:
            FlexConfigError: If status is enabled without rated_power.
        """
        self.running_power = pyo.Expression(
            tb.time_index,
            rule=lambda b, t: (1 + b.pdu_loss_fraction) * b.it_power[t],
            doc="Step-mean server draw while running: IT power plus the PDU loss.",
        )
        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.register_io_variable(power, role="output")

        if not self.config.unit_commitment.status:
            self.power_electrical_definition = pyo.Constraint(
                tb.time_index,
                rule=lambda b, t: power[t] == b.running_power[t],
                doc="The server draws its running power.",
            )
            self.heat_output_definition = pyo.Constraint(
                tb.time_index,
                rule=lambda b, t: b.heat_output[t] == b.it_power[t],
                doc="All IT power leaves the server as heat.",
            )
            return

        if self.config.rated_power is None:
            raise FlexConfigError(
                "unit_commitment.status needs rated_power.",
                field="rated_power",
                value=None,
            )
        rated = pyo.value(pyunits.convert(self.config.rated_power, pyunits.kW))
        rated = rated * pyunits.kW
        # unit_commitment.status rebuilds this status; it is not a spec element.
        with suppress_logic_records():
            status = add_status(self, power, 0.0 * pyunits.kW, rated)
        self.register_io_variable(status, role="input")
        self.power_gate_upper = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: power[t] <= b.running_power[t],
            doc="The draw never exceeds the running power.",
        )
        self.power_gate_lower = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: power[t] >= b.running_power[t] - rated * (1 - status[t]),
            doc="The draw equals the running power while on.",
        )
        self.heat_gate_upper = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: b.heat_output[t] <= b.it_power[t],
            doc="Heat released never exceeds IT power.",
        )
        self.heat_gate_lower = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: b.heat_output[t]
            >= b.it_power[t] - rated * (1 - status[t]),
            doc="Heat released equals IT power while on.",
        )
        self.heat_gate_off = pyo.Constraint(
            tb.time_index,
            rule=lambda b, t: b.heat_output[t] <= rated * status[t],
            doc="No heat is released while off.",
        )
