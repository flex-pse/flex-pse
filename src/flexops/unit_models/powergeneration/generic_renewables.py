r"""GenericRenewables(OpsBlockData): capacity_factor-driven electrical export (§3.4).

Like :class:`~flexops.unit_models.powergeneration.combustor.Combustor` and
:class:`~flexops.unit_models.storage.battery.BatteryModel`, this unit has no
fluid ports (no ``property_package``): it turns an exogenous production
profile -- solar irradiance, a wind-speed-derived output curve, or any other
precomputed capacity-factor series -- into an electrical export, scaled by a
fixable nameplate ``capacity``:

.. math::

    P_{elec}[t] = -\text{capacity\_factor}[t] \cdot \text{capacity}

Like a discharging :class:`BatteryModel` or an exporting :class:`Combustor`,
this is **negative** (export): ``power_electrical[t]`` is upper-bounded at 0,
so plant aggregation (a plain sum) nets generation against load with no
per-unit sign flipping.

``capacity_factor`` accepts exactly two forms, checked once the model's
:class:`~flexops.core.time_block.TimeBlockData` (and hence its horizon length)
is known, in ``build()``:

* An **array-like** of length ``T`` (list/tuple/``numpy.ndarray``/
  ``pandas.Series``) -- copied into a plain per-timestep list at construction.
* A **time-indexed Pyomo Var/Param** of length ``T``, living anywhere on the
  model (e.g. an upstream irradiance/wind-profile signal) -- kept as a live
  symbolic reference, so a later change to that component's value changes
  ``power_electrical`` with no rebuild.

A scalar value (a bare number, or an unindexed Pyomo component) is rejected:
the whole point of this unit is a timeseries, not a broadcast constant.
``capacity_factor`` is deliberately **not** IO-registered: like
``FlexCosting``'s ``energy_prices``, it is exogenous data feeding an
expression, not a decision Var this unit owns.

``technology`` (``"solar"``/``"wind"``/``None``) is a **placeholder**: it is
stored on the config and nowhere else consumed by this unit's own build
logic. It is reserved for a future costing-function selection (different
technology-specific capex/opex parameter sets), not yet implemented.
"""

import enum
from collections.abc import Sized

import pyomo.environ as pyo
from idaes.core import declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.environ import units as pyunits

from flexcore import nomenclature as nm
from flexcore.config.schema import UnitCommitmentConfig
from flexcore.exceptions import FlexConfigError
from flexops.core.ops_block import OpsBlockData


class RenewableTechnology(enum.StrEnum):
    """Which renewable technology a :class:`GenericRenewables` represents.

    A placeholder (module docstring): not yet consumed by this unit's build
    logic, reserved for a future costing-function selection.
    """

    SOLAR = "solar"
    WIND = "wind"


def _technology_domain(value):
    """ConfigValue domain: coerce to a :class:`RenewableTechnology`, or ``None``."""
    if value is None:
        return None
    try:
        return RenewableTechnology(value)
    except ValueError as exc:
        allowed = ", ".join(repr(t.value) for t in RenewableTechnology)
        raise FlexConfigError(
            f"technology must be one of {allowed}, or None, got {value!r}.",
            field="technology",
            value=value,
        ) from exc


def _capacity_factor_terms(value, n_points: int) -> list:
    """Validate and normalize ``capacity_factor`` into ``n_points`` per-step terms.

    Run from :meth:`GenericRenewablesData.build`, not a ``ConfigValue``
    domain: a ``ConfigValue`` domain that raises ``FlexConfigError`` has it
    wrapped into a generic ``ValueError`` by Pyomo's own ``ConfigValue._cast``,
    losing the ``field``/``value`` the caller needs -- so both the shape check
    (scalar vs. timeseries) and the length-vs-horizon check happen here, once
    the TimeBlock is known.

    Mirrors :func:`flexops.costing.flex_costing._price_terms`: a time-indexed
    Pyomo component's members are kept as live symbolic references (so a
    later change to the component's value is reflected with no rebuild); an
    array-like is copied into a plain list. The indexed-component branch is
    checked first, since a scalar Pyomo component is itself ``Sized`` (length
    1).

    Args:
        value: The configured ``capacity_factor``.
        n_points: The number of time points the horizon carries.

    Returns:
        A list of ``n_points`` terms, in time-index order.

    Raises:
        FlexConfigError: If ``value`` is a scalar (a bare number or an
            unindexed Pyomo component) rather than a timeseries, or does not
            carry exactly ``n_points`` values.
    """
    is_indexed = getattr(value, "is_indexed", None)
    if callable(is_indexed):
        if not is_indexed():
            raise FlexConfigError(
                "capacity_factor must be time-indexed; got a scalar Pyomo "
                f"component ({value!r}). GenericRenewables takes a "
                "timeseries, not a broadcast constant.",
                field="capacity_factor",
                value=value,
            )
        index = value.index_set()
        if len(index) != n_points:
            raise FlexConfigError(
                f"capacity_factor is indexed by {index.name!r}, which has "
                f"{len(index)} members, but the horizon has {n_points} time "
                "points. capacity_factor needs exactly one value per time "
                "point.",
                field="capacity_factor",
                value=value,
            )
        return [value[i] for i in index]
    if not isinstance(value, Sized) or isinstance(value, (str, bytes)):
        raise FlexConfigError(
            "capacity_factor must be an array-like of length T (one value "
            "per time point) or a time-indexed Pyomo Var/Param of length T; "
            f"got {type(value).__name__}.",
            field="capacity_factor",
            value=value,
        )
    if len(value) != n_points:
        raise FlexConfigError(
            f"capacity_factor has {len(value)} values, but the horizon has "
            f"{n_points} time points. capacity_factor needs exactly one "
            "value per time point.",
            field="capacity_factor",
            value=value,
        )
    return list(value)


@declare_process_block_class("GenericRenewables")
class GenericRenewablesData(OpsBlockData):
    """A capacity_factor-driven generator, exporting electrical power.

    See the module docstring for the governing relation and the two accepted
    ``capacity_factor`` forms.

    Config:
        ``capacity`` (required, kW): the nameplate design capacity -- a fixed,
        regressable=False sizing ``Var``, matching
        :attr:`~flexops.unit_models.storage.battery.BatteryModel.capacity`'s
        pattern (fixed at the constructor value; a ``costing_package``
        associates it with design/operations mode toggling).
        ``capacity_factor`` (required): the timeseries production profile, an
        array-like or time-indexed Pyomo component of length equal to the
        horizon (see module docstring).
        ``technology`` (default ``None``): ``"solar"`` or ``"wind"`` -- a
        placeholder, not yet consumed by ``build()`` (see module docstring).

    Example:
        >>> from pyomo.environ import units as pyunits
        >>> from flexops.testing import dummy_time_block
        >>> from flexops.unit_models import GenericRenewables
        >>> m = dummy_time_block(3)
        >>> m.solar = GenericRenewables(  # doctest: +SKIP
        ...     capacity=100 * pyunits.kW,
        ...     capacity_factor=[0.0, 0.6, 0.9],
        ... )
    """

    CONFIG = OpsBlockData.CONFIG()
    CONFIG.get("unit_commitment").set_default_value(UnitCommitmentConfig(status=False))
    CONFIG.declare(
        "capacity",
        ConfigValue(
            description="Nameplate design capacity, kW -- the fixed sizing "
            "Var value at construction. Required."
        ),
    )
    CONFIG.declare(
        "capacity_factor",
        ConfigValue(
            description="Timeseries production profile: an array-like of "
            "length T, or a time-indexed Pyomo Var/Param of length T "
            "(validated against the horizon in build()). Required.",
        ),
    )
    CONFIG.declare(
        "technology",
        ConfigValue(
            default=None,
            domain=_technology_domain,
            description="Which renewable technology this unit represents, "
            "'solar' or 'wind' (or None). A placeholder: not yet consumed by "
            "this unit's build logic; reserved for a future costing-function "
            "selection (technology-specific capex/opex parameters).",
        ),
    )

    def build(self) -> None:
        """Build the capacity Var, resolve capacity_factor, and the power relation."""
        super().build()
        tb = self._find_time_block()

        if self.config.capacity is None:
            raise FlexConfigError(
                "GenericRenewables requires capacity (the nameplate design "
                "capacity, kW).",
                field="capacity",
                value=None,
            )
        if self.config.capacity_factor is None:
            raise FlexConfigError(
                "GenericRenewables requires capacity_factor (an array-like "
                "or time-indexed Pyomo component of length equal to the "
                "horizon).",
                field="capacity_factor",
                value=None,
            )

        capacity_val = pyo.value(pyunits.convert(self.config.capacity, pyunits.kW))
        self.capacity = pyo.Var(
            initialize=capacity_val,
            bounds=(0.0, None),
            units=pyunits.kW,
            doc="Nameplate design capacity (fixable sizing Var); fixed at "
            "the constructor value by default (operations mode); "
            "costing.set_design_mode() unfixes it.",
        )
        self.capacity.fix(capacity_val)
        self.register_process_parameter(self.capacity, regressable=False)
        costing_package = self.config.costing_package
        if costing_package is not None:
            costing_package.register_sizing_variable(self.capacity)

        terms = _capacity_factor_terms(self.config.capacity_factor, tb.n_points)
        self._capacity_factor = dict(zip(tb.time_index, terms, strict=True))

        power = self.declare_power(nm.PowerKind.ELECTRICAL)
        self.register_io_variable(power, role="output")
        for t in tb.time_index:
            power[t].setub(0.0)

        capacity_factor = self._capacity_factor
        capacity = self.capacity

        @self.Constraint(
            tb.time_index,
            doc="power_electrical == -capacity_factor[t] * capacity; an "
            "export (kW).",
        )
        def power_electrical_relation(b, t):
            return -power[t] == pyunits.convert(
                capacity_factor[t] * capacity, pyunits.kW
            )

        self.register_relation(self.power_electrical_relation, target=power)
        surrogate = getattr(self.config.flexops_config, "surrogate", None)
        if surrogate is not None and surrogate.functional_form != "constant_intensity":
            self.swap_relation("power_electrical_relation", surrogate)
