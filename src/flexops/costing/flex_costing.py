"""FlexCosting: the costing block that wraps EECO (architecture §3.6).

``FlexCosting`` subclasses IDAES :class:`FlowsheetCostingBlockData` for its
registration/CapEx machinery and organizes every cost into two sub-blocks it
owns:

* **``opex``** — all operating cost: **electricity** and **fuel** cost (both
  delegated to the external EECO package via the :mod:`flexops.costing.opex`
  bridge), a user-defined **fixed operating cost** (maintenance/labor/chemicals),
  and any **scalar operating cost** (non-energy flows/supplies/products priced
  natively in flex-pse, never through EECO). ``opex.total_operating_cost`` is
  their sum and is re-exposed as :attr:`aggregate_operating_cost`.
* **``capex``** — capital cost. In v0 an **empty placeholder**
  (``total_capital_cost == 0``, re-exposed as :attr:`aggregate_capital_cost`);
  later milestones aggregate per-unit capital costs into it. Capital cost enters
  the objective **only in design mode** (:meth:`set_design_mode`); the
  operations-mode objective is :attr:`aggregate_operating_cost` alone.

Every quantity FlexCosting exposes is a decision-visible ``Var`` defined by an
equality ``Constraint`` (not a bare ``Expression``), so aggregate power, the
per-line-item costs, the annualized cost, and the totals are all first-class
model variables.

**Energy carriers.** FlexCosting aggregates every registered power draw into an
indexed kW series :attr:`aggregate_power` ``[t, carrier]``: ``"electrical"`` and
one carrier per distinct **thermal** temperature (``"thermal@<T>K"`` — heat duties
at different temperatures are never summed together). Electricity is billed
through EECO via ``add_electricity_cost``, unless ``energy_prices`` gives its
carrier a price — a carrier priced there is billed natively as
``Σ_t price[t] × quantity[t] × dt`` and never reaches EECO. That price may be one
value for the whole horizon, one per time point, or a Pyomo component indexed
over the horizon (so it can be a model Param or Var).

**Fuel is a volume, not a power.** A combustible fuel is metered and billed on
volume, so every fuel-usage flow a unit registered
(:meth:`~flexops.core.ops_block.OpsBlockData.register_fuel_usage`) is aggregated
separately into :attr:`aggregate_fuel_usage` ``[t, fuel]`` in EECO's m³/hr and
billed via ``add_fuel_cost`` against the same tariff. Fuels are discovered from
the model, so there is nothing to declare on this block. flex-pse assumes **no**
heating value: if a tariff prices gas on an energy basis, converting it is EECO's
job. FlexCosting writes **no** tariff cost math of its own (that is EECO's).

Construction-order invariant: FlexCosting may be constructed before any units
exist, because all aggregation and the EECO call are deferred to
:meth:`cost_process`, which **pulls** every unit's registered power and fuel usage
from the model (via :func:`~flexops.core.registration.iter_io_registry`).
"""

import dataclasses
import logging
from collections.abc import Mapping, Sequence, Sized
from typing import Any

import numpy as np
import pyomo.environ as pyo
from idaes.core import FlowsheetCostingBlockData, declare_process_block_class
from pyomo.common.config import ConfigValue
from pyomo.core.base.units_container import UnitsError
from pyomo.environ import units as pyunits
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent

from flexcore import nomenclature as nm
from flexcore.exceptions import FlexConfigError
from flexops.core.registration import iter_io_registry
from flexops.costing.opex import (
    EECO_GAS_USAGE_UNITS,
    EECO_POWER_UNITS,
    DRConfig,
    add_electricity_cost,
    add_fuel_cost,
    currency_units,
    evaluate_cost,
    evaluate_fuel_cost,
    load_dr_program,
    load_tariff,
    merge_tariffs,
    tariff_currency_units,
)

_log = logging.getLogger(__name__)


def _is_multi_tariff_source(source) -> bool:
    """Whether a configured tariff source holds several tariffs to merge.

    Distinguishes the multi-tariff forms from the single-source forms
    :func:`~flexops.costing.opex.load_tariff` already accepts, which include a
    records ``list[dict]`` and a ``{"tariff_data": [...]}`` payload:

    * a mapping is multi-tariff unless it carries the ``"tariff_data"`` key;
    * a sequence is multi-tariff unless its entries are row records (dicts).

    Args:
        source: The configured ``tariff_file``/``tariff`` value.

    Returns:
        ``True`` if ``source`` should go to
        :func:`~flexops.costing.opex.merge_tariffs`.
    """
    if isinstance(source, Mapping):
        return "tariff_data" not in source
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        return False
    return len(source) > 0 and not all(isinstance(item, Mapping) for item in source)


def _energy_prices_domain(value):
    """Validate the ``energy_prices`` mapping: carrier name -> price.

    Checks only that the input is a mapping of string keys to a supported
    price form. The length of a per-period price is checked against the
    horizon in :meth:`FlexCostingData.build`, which is where the
    ``time_block`` is available.

    Args:
        value: The configured mapping, or ``None``.

    Returns:
        The mapping unchanged (``{}`` for ``None``).

    Raises:
        FlexConfigError: If it is not a mapping, a key is not a string, or a price
            is itself a mapping (which would be costed over its keys).
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise FlexConfigError(
            "energy_prices must be a mapping of carrier or fuel name to a price; "
            f"got {type(value).__name__}.",
            field="energy_prices",
            value=value,
        )
    for name, price in value.items():
        if not isinstance(name, str):
            raise FlexConfigError(
                f"energy_prices keys must be carrier or fuel names (strings); got "
                f"{name!r}.",
                field="energy_prices",
                value=name,
            )
        if isinstance(price, Mapping):
            raise FlexConfigError(
                f"energy_prices[{name!r}] is a mapping, which would be costed over "
                "its keys. A price is a single value, an array-like with one value "
                "per time point, or a Pyomo component indexed over the horizon.",
                field="energy_prices",
                value=price,
            )
    return dict(value)


def _price_terms(name: str, value, n_points: int):
    """Normalize one configured price to a scalar, or one value per time point.

    Recognizes the three price forms, in the order they must be tested: a Pyomo
    component (indexed over the horizon, or a scalar Param/expression), an
    array-like of per-period values, or a plain number. A Pyomo scalar component
    is ``Sized`` with length 1, so the component check has to come first.

    Args:
        name: The carrier or fuel name, for error messages.
        value: The configured price.
        n_points: The number of time points a per-period price must cover.

    Returns:
        The price itself when scalar, or a ``list`` of ``n_points`` per-period
        prices (in index-set order for an indexed component).

    Raises:
        FlexConfigError: If a per-period price does not have exactly ``n_points``
            values.
    """
    is_indexed = getattr(value, "is_indexed", None)
    if callable(is_indexed):
        if not is_indexed():
            return value
        index = value.index_set()
        if len(index) != n_points:
            raise FlexConfigError(
                f"energy_prices[{name!r}] is indexed by {index.name!r}, which has "
                f"{len(index)} members, but the horizon has {n_points} time points. "
                "An indexed price needs exactly one value per time point.",
                field="energy_prices",
                value=name,
            )
        return [value[i] for i in index]
    if isinstance(value, Sized):
        if len(value) != n_points:
            raise FlexConfigError(
                f"energy_prices[{name!r}] has {len(value)} values, but the horizon "
                f"has {n_points} time points. A price is either a single value or "
                "an array-like with exactly one value per time point.",
                field="energy_prices",
                value=name,
            )
        return list(value)
    return value


def _price_in_units(price, per_quantity_units):
    """Give a dimensionless price the units of the quantity it is billed on.

    A price that already carries units is returned untouched, so a price whose
    units do not reconcile with its carrier still fails the unit-consistency
    check rather than being silently reinterpreted.

    Args:
        price: A price value, with or without Pyomo units.
        per_quantity_units: The units to assume when it has none, as currency over
            the metered quantity (e.g. ``USD/kWh`` for a power carrier).

    Returns:
        The units-carrying price.
    """
    try:
        assert_units_equivalent(price, pyunits.dimensionless)
    except UnitsError:
        return price
    return price * per_quantity_units


def _optional_rate_domain(value):
    """Validate an optional annual rate (a fraction, or ``None`` for unset).

    Args:
        value: A float-like rate, or ``None``.

    Returns:
        The rate as a float, or ``None``.

    Raises:
        FlexConfigError: If ``value`` is not None and not float-convertible.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise FlexConfigError(
            f"interest_rate must be a fraction (e.g. 0.06) or None; got {value!r}.",
            field="interest_rate",
            value=value,
        ) from exc


@dataclasses.dataclass
class OperatingCostBreakdown:
    """Categorized operating cost from :meth:`FlexCostingData.report_cost`.

    Every value is a magnitude in the report's currency (:attr:`CostReport.currency`).

    Attributes:
        electricity: EECO post-hoc electricity bill on the realized aggregate
            power.
        fuel: EECO post-hoc fuel bill on the realized volumetric usage, summed
            over every fuel found in the model; ``0`` when none burns fuel.
        fixed: The configured fixed operating cost (a constant).
        scalar: Non-energy scalar operating cost summed over registered
            scalar-cost entries; ``0`` when none registered.
        dr_revenue: Demand-response incentive credit (subtracted); ``0`` in v0
            (DR is containers-only).
        total: ``electricity + fuel + fixed + scalar - dr_revenue``.
    """

    electricity: float
    fuel: float
    fixed: float
    scalar: float
    dr_revenue: float
    total: float


@dataclasses.dataclass
class CapitalCostBreakdown:
    """Categorized capital cost from :meth:`FlexCostingData.report_cost`.

    Every value is a magnitude in the report's currency (:attr:`CostReport.currency`).

    Attributes:
        by_component: Per-unit capital cost keyed by unit block name; ``{}`` in
            v0 (the capex block is an empty placeholder).
        total: Sum over ``by_component``; ``0`` in v0.
    """

    by_component: dict[str, float]
    total: float


@dataclasses.dataclass
class CostReport:
    """The reported, categorized cost from :meth:`FlexCostingData.report_cost`.

    Attributes:
        operating: The :class:`OperatingCostBreakdown`.
        capital: The :class:`CapitalCostBreakdown`.
        total: ``operating.total + capital.total``.
        currency: The currency every value in this report is a magnitude in —
            the costing block's ``base_currency`` as a string (e.g. ``"USD"``),
            which is the tariff sheet's basis when a tariff is given and the
            configured ``currency`` otherwise.
    """

    operating: OperatingCostBreakdown
    capital: CapitalCostBreakdown
    total: float
    currency: str


@dataclasses.dataclass
class ScalarCostSpec:
    """A non-energy scalar cost registered on FlexCosting (never billed via EECO).

    Attributes:
        name: The cost's name (e.g. ``"water"``, ``"chemicals"``).
        quantity: The time-indexed ``Var``/``Expression`` being costed (a rate).
        price: The signed price per unit quantity (positive = cost, negative =
            revenue/credit).
        quantity_units: The Pyomo units ``quantity`` is converted to before
            costing (a rate, e.g. ``m**3/hr``); a quantity that does not convert
            raises, forcing unit consistency.
        unit: The unit block this cost is attributed to, or ``None`` for a
            facility-level cost; recorded for per-unit cost attribution.
    """

    name: str
    quantity: Any
    price: float
    quantity_units: Any
    unit: Any = None


@dataclasses.dataclass
class _SizingEntry:
    """A registered sizing Var and its (optional) capex-defining constraint.

    Attributes:
        var: The sizing ``Var`` (e.g. a battery/tank capacity) modes fix/unfix.
        capex_constraint: The constraint modes (de)activate, or ``None``.
    """

    var: Any
    capex_constraint: Any | None


@declare_process_block_class("FlexCosting")
class FlexCostingData(FlowsheetCostingBlockData):
    """EECO-backed costing block with ``opex``/``capex`` sub-blocks (module docstring).

    Example:
        >>> import pyomo.environ as pyo
        >>> from pyomo.environ import units as pyunits
        >>> import flexops as fo
        >>> m = pyo.ConcreteModel()
        >>> m.time_block = fo.TimeBlock(
        ...     start_date="2025-07-08", end_date="2025-07-09",
        ...     time_step=1 * pyunits.hr,
        ... )
        >>> m.costing = fo.FlexCosting(  # doctest: +SKIP
        ...     time_block=m.time_block, tariff_file="tariff.json",
        ... )
    """

    CONFIG = FlowsheetCostingBlockData.CONFIG()
    CONFIG.declare(
        "time_block",
        ConfigValue(
            default=None,
            description="The fo.TimeBlock instance whose time_index/dt/"
            "datetime_index this costing aggregates and bills against. Required.",
        ),
    )
    CONFIG.declare(
        "tariff_file",
        ConfigValue(
            default=None,
            description="Path to an EECO tariff file, or several to merge: a list "
            "of paths, or a mapping of EECO utility ('electric'/'gas') to path to "
            "also assign each file to a utility. At most one of tariff_file or "
            "tariff; omit both to price every carrier from energy_prices.",
        ),
    )
    CONFIG.declare(
        "tariff",
        ConfigValue(
            default=None,
            description="An already-loaded EECO tariff object (or a list/mapping "
            "of them to merge, as for tariff_file). At most one of tariff or "
            "tariff_file.",
        ),
    )
    CONFIG.declare(
        "energy_prices",
        ConfigValue(
            default=None,
            domain=_energy_prices_domain,
            description="Optional native prices, as a mapping of carrier or fuel "
            "name to its price. Keys are 'electrical' or a registered fuel name. A "
            "price is a single value (flat over the horizon), an array-like with "
            "one value per time point, or a Pyomo component indexed over the "
            "horizon (so the price may itself be a model Param or Var); anything "
            "with the wrong number of values raises. A price may carry Pyomo units "
            "(0.12 * currency_units('USD') / pyunits.kWh) or be a bare number, "
            "which is read in the currency over the carrier's metered quantity — "
            "kWh for a power carrier, m**3 for a fuel. A carrier priced here is "
            "billed natively and is NOT sent to EECO, so it needs no tariff.",
        ),
    )
    CONFIG.declare(
        "currency",
        ConfigValue(
            default="USD",
            domain=str,
            description="Currency basis to use when no tariff is given (a "
            "tariff's own currency basis always wins). Registered as a Pyomo "
            "currency unit, so flat prices must be written in it.",
        ),
    )
    CONFIG.declare(
        "prorate_monthly_charges",
        ConfigValue(
            default=True,
            domain=bool,
            description="Prorate a tariff's monthly-assessed demand charge and "
            "fixed (customer) charge to the horizon length when the horizon is "
            "shorter than the calendar month it starts in. Set False to bill the "
            "full monthly charges regardless of horizon length.",
        ),
    )
    CONFIG.declare(
        "dr_event_file",
        ConfigValue(
            default=None,
            description="Optional path to an EECO demand-response program file. "
            "v0 loads it into a container only (no DR constraints built).",
        ),
    )
    CONFIG.declare(
        "fixed_operating_cost",
        ConfigValue(
            default=0.0,
            domain=float,
            description="Fixed operating cost in dollars over the horizon "
            "(non-tariff: maintenance, labor, chemicals). Distinct from the "
            "tariff's own fixed charge, which EECO folds into electricity cost.",
        ),
    )
    CONFIG.declare(
        "lifetime_years",
        ConfigValue(
            default=20.0,
            domain=float,
            description="Plant lifetime in years (> 0), used with the effective "
            "rate to form the capital recovery factor that annualizes capital "
            "cost.",
        ),
    )
    CONFIG.declare(
        "discount_rate",
        ConfigValue(
            default=0.08,
            domain=float,
            description="Annual discount rate (fraction, e.g. 0.08 = 8%). Used "
            "alone as the effective rate when interest_rate is unset; otherwise "
            "it deflates interest_rate into a real effective rate. An effective "
            "rate of 0 falls back to straight-line 1/lifetime.",
        ),
    )
    CONFIG.declare(
        "interest_rate",
        ConfigValue(
            default=None,
            domain=_optional_rate_domain,
            description="Optional annual cost of capital (fraction, e.g. 0.06). "
            "When given, the capital recovery factor uses the effective rate "
            "(1 + interest_rate) / (1 + discount_rate) - 1. Leave unset to use "
            "discount_rate alone.",
        ),
    )

    # -- required FlowsheetCostingBlockData overloads ---------------------

    def build_global_params(self) -> None:
        """Resolve the tariff (if any) and set the base currency.

        Called during :meth:`build`. When a tariff is given the base currency is
        the tariff sheet's currency basis (EECO tariffs are dollar-based → ``USD``,
        from the charge ``units`` column); otherwise it is the configured
        ``currency``. Every operating-cost expression FlexCosting builds is
        labeled with it. EECO's own cost expressions are dimensionless dollars,
        so FlexCosting casts them to this currency (:meth:`cost_process`).

        Also records which utilities the tariff prices, so :meth:`cost_process`
        can tell a carrier the tariff covers from one it does not.
        """
        self._tariff = self._resolve_tariff()
        if self._tariff is None:
            self._currency = currency_units(self.config.currency)
            self._tariff_utilities: set[str] = set()
        else:
            self._currency = tariff_currency_units(self._tariff)
            self._tariff_utilities = set(self._tariff["utility"].dropna().unique())
        self.base_currency = self._currency
        self.base_period = pyunits.year

    def _resolve_tariff(self):
        """Load the tariff from config, or ``None`` when pricing is all native.

        Accepts a single source, a sequence of sources to merge, or a mapping of
        EECO utility to source to merge *and* assign per utility — several files
        always collapse into one ``rate_data`` frame, whose ``utility`` column is
        what selects rows per leg.

        Returns:
            The loaded EECO tariff object, or ``None`` if no tariff was given.

        Raises:
            FlexConfigError: If both ``tariff_file`` and ``tariff`` are given, or
                if neither is and ``energy_prices`` is empty (nothing would price
                the model).
        """
        tariff_file = self.config.tariff_file
        tariff = self.config.tariff
        if tariff_file is not None and tariff is not None:
            raise FlexConfigError(
                "Provide at most one of tariff_file (a path, or several to merge) "
                "or tariff (an already-loaded EECO tariff object); got both.",
                field="tariff_file",
                value=tariff_file,
            )
        source = tariff_file if tariff_file is not None else tariff
        if source is None:
            if not self.config.energy_prices:
                raise FlexConfigError(
                    "Nothing prices this costing block: give a tariff "
                    "(tariff_file= or tariff=) or a native price per carrier "
                    "(energy_prices={'electrical': 0.12 * currency_units('USD') / "
                    "pyunits.kWh}).",
                    field="tariff_file",
                    value=None,
                )
            return None
        if _is_multi_tariff_source(source):
            return merge_tariffs(source)
        return load_tariff(source)

    def build_process_costs(self) -> None:
        """No-op: flex-native process costs are built in :meth:`cost_process`.
        Required override from IDAES `FlowsheetCostingBlockData`."""

    def initialize_build(self) -> None:
        """No-op: FlexCosting builds only Vars/Constraints/Params, nothing to init.
        Required override from IDAES `FlowsheetCostingBlockData`."""

    # -- build (construction time; no aggregation, no EECO call) ----------

    def build(self) -> None:
        """Validate config, load the tariff + DR container, init empty registries.

        Loads the tariff and the DR program (into the ``dr`` container) but
        builds **no** aggregation, **no** ``opex``/``capex`` sub-blocks, and
        **no** DR constraints here — everything is deferred to
        :meth:`cost_process` so the block may be constructed before any units
        exist (the construction-order invariant).

        Raises:
            FlexConfigError: If not exactly one of ``tariff_file``/``tariff`` is
                given, ``time_block`` is missing, or a per-period price in
                ``energy_prices`` does not have one value per time point.
        """
        # super().build() runs build_global_params, which resolves the tariff
        # (exclusivity check) and sets self._tariff, self._currency, base_currency.
        super().build()

        if self.config.time_block is None:
            raise FlexConfigError(
                "FlexCosting requires a time_block=fo.TimeBlock instance.",
                field="time_block",
                value=None,
            )

        # Normalize every configured price to a scalar or one value per time point.
        # This is the earliest point the horizon length is known, so it is where a
        # misaligned price array fails.
        # Pyomo stores a ConfigValue default verbatim without running its domain,
        # so an unset energy_prices is None rather than the domain's {}.
        n_points = self.config.time_block.n_points
        self._prices: dict[str, Any] = {
            name: _price_terms(name, value, n_points)
            for name, value in (self.config.energy_prices or {}).items()
        }

        self.dr = DRConfig(program=load_dr_program(self.config.dr_event_file))

        # _registered_power: units that opted into this costing package. Power
        # AGGREGATION does not read this -- it pulls from the model in
        # cost_process so it is construction-order independent.
        self._registered_power: list[tuple[Any, Any, nm.PowerKind]] = []
        # _registered_sizing: sizing Vars + capex constraints the modes toggle.
        self._registered_sizing: list[_SizingEntry] = []
        # Non-energy scalar costs registered on this block. (Fuels need no
        # registration here -- cost_process discovers them from the model.)
        self._registered_scalar_costs: dict[str, ScalarCostSpec] = {}

    # -- registration ---------------

    def register_unit_power(self, unit, var, kind: nm.PowerKind) -> None:
        """Record that ``unit`` associated a power draw with this costing package.

        Called by :meth:`~flexops.core.ops_block.OpsBlockData.register_power` when
        a unit is built with ``costing_package=`` set. The record is the explicit
        unit↔costing association used for capex attribution; power aggregation in
        :meth:`cost_process` pulls from the model instead, so it
        does not depend on this being called.

        Args:
            unit: The unit block registering the draw.
            var: The unit's power-draw ``Var`` (kW).
            kind: The :class:`~flexcore.nomenclature.PowerKind` of the draw.
        """
        self._registered_power.append((unit, var, kind))

    def register_sizing_variable(self, var, capex_constraint=None) -> None:
        """Register a sizing Var (and its capex constraint) for the mode toggles.

        Args:
            var: A sizing ``Var`` (e.g. battery/tank capacity) that
                :meth:`set_operations_mode`/:meth:`set_design_mode` fix/unfix.
            capex_constraint: The capex-defining constraint the modes
                (de)activate, or ``None``.
        """
        self._registered_sizing.append(_SizingEntry(var, capex_constraint))

    def register_scalar_cost(
        self, name: str, quantity, price: float, quantity_units, *, unit=None
    ) -> ScalarCostSpec:
        """Register a non-energy scalar operating cost (never billed via EECO).

        Costs an arbitrary time-indexed rate as ``price × Σ_t quantity[t] × dt`` —
        e.g. water withdrawal per m³, chemical dosing per kg, or a product-revenue
        credit (a negative ``price``). Built entirely in flex-pse; EECO is not
        involved.

        Args:
            name: The cost's name.
            quantity: A time-indexed ``Var``/``Expression`` (a rate).
            price: The signed price per unit quantity, in the base currency
                (positive = cost, negative = revenue/credit).
            quantity_units: The Pyomo units ``quantity`` is converted to before
                costing (a rate, e.g. ``m**3/hr``).
            unit: The unit block this cost is attributed to, or ``None`` for a
                facility-level cost; recorded for per-unit cost attribution.

        Returns:
            The stored :class:`ScalarCostSpec`.

        Raises:
            FlexConfigError: If ``name`` was already registered.
        """
        if name in self._registered_scalar_costs:
            raise FlexConfigError(
                f"Scalar cost {name!r} is already registered.",
                field="name",
                value=name,
            )
        spec = ScalarCostSpec(
            name=name,
            quantity=quantity,
            price=price,
            quantity_units=quantity_units,
            unit=unit,
        )
        self._registered_scalar_costs[name] = spec
        return spec

    # -- DR hook (no-op in v0; containers-only) ---------------------------

    def _build_dr(self) -> None:
        """No-op demand-response hook (v0 is containers-only, architecture §2.4).

        Exists so later DR work is additive; it builds no DR event, curtailment,
        incentive, or capacity constraints.
        """
        if self.dr is not None and self.dr.program is not None:
            _log.debug(
                "DR program present but v0 is containers-only; building no DR "
                "constraints on %s.",
                self.name,
            )

    # -- carrier helpers --------------------------------------------------

    @staticmethod
    def _carrier_key(record) -> str:
        """Return the aggregation carrier key for a power record.

        Electrical draws share one ``"electrical"`` carrier; thermal draws use a
        per-temperature label (``"thermal@<T>K"``) so duties at different
        temperatures never mix.

        Args:
            record: A :class:`~flexops.core.registration.PowerRecord`.

        Returns:
            The carrier key string.
        """
        if record.kind is nm.PowerKind.THERMAL:
            temp_k = pyo.value(pyunits.convert(record.temperature, pyunits.K))
            return f"thermal@{temp_k:.6g}K"
        return "electrical"

    # -- cost_process (all aggregation + the EECO call, deferred here) ----

    def cost_process(self) -> None:
        """Aggregate energy, build the ``opex``/``capex`` blocks, enter operations mode.

        Overrides the parent ``FlowsheetCostingBlockData.cost_process`` (whose
        ``aggregate_capital_cost`` Var would collide with the flex-native
        names): FlexCosting builds its own flex-native Vars/Constraints and does
        not invoke the parent aggregation machinery. Every derived quantity is a
        ``Var`` defined by an ``eq_<name>`` equality ``Constraint``.
        """
        tb = self.config.time_block
        cur = self._currency
        dt_hours = pyo.value(pyunits.convert(tb.dt, pyunits.hr))

        self._build_composition_aggregates()
        self._build_power_aggregation(tb)
        self._build_fuel_aggregation(tb)
        self._build_opex(tb, cur, dt_hours)
        self._build_capex(cur)
        self._build_totals_and_annualization(tb, cur)

        self.set_operations_mode()  # default final state (scheduling first)

    def _build_composition_aggregates(self) -> None:
        """Build every PlantBlock's/NetworkBlock's own aggregation Expressions.

        Those totals are deferred exactly as this block's are: a plant may be
        built before its units (the frozen api-freeze script does), so it cannot
        aggregate at its own construction. Costing is the natural trigger — it
        is the point at which the model is declared complete — so it drives
        them here. Duck-typed on ``_build_aggregates`` so this module needs no
        import of the composition layer; the call is idempotent.
        """
        for block in self.model().component_data_objects(pyo.Block, descend_into=True):
            build_aggregates = getattr(block, "_build_aggregates", None)
            if callable(build_aggregates):
                build_aggregates()

    def _build_power_aggregation(self, tb) -> None:
        """Build the indexed per-carrier kW aggregation (Var + Constraint).

        Pulls every registered power draw from the model, buckets it by carrier
        (``"electrical"`` / ``"thermal@<T>K"``), and defines
        ``aggregate_power[t, carrier]`` in kW. ``pyunits.convert(v[t], kW)``
        loudly rejects any draw that is not a power. Also exposes the API-freeze
        ``aggregate_electrical_power`` (a Reference) and a temperature-blind
        ``aggregate_thermal_power`` total.
        """
        vars_by_carrier: dict[str, list] = {}
        for _block, registry in iter_io_registry(self.model()):
            for rec in registry.power:
                vars_by_carrier.setdefault(self._carrier_key(rec), []).append(rec.var)

        carriers = sorted(set(vars_by_carrier) | {"electrical"})
        thermal_carriers = [c for c in carriers if c.startswith("thermal@")]

        self.aggregate_power = pyo.Var(
            tb.time_index,
            carriers,
            initialize=0.0,
            units=pyunits.kW,
            doc="Aggregate power by carrier (kW).",
        )

        def _agg_rule(_b, t, carrier):
            terms = vars_by_carrier.get(carrier, [])
            return self.aggregate_power[t, carrier] == (
                sum(pyunits.convert(v[t], pyunits.kW) for v in terms) + 0 * pyunits.kW
            )

        self.eq_aggregate_power = pyo.Constraint(
            tb.time_index, carriers, rule=_agg_rule
        )

        # API-freeze accessors: electrical (a Reference) + a temperature-blind
        # thermal total (its own Var + Constraint, 0 when no thermal draws).
        self.aggregate_electrical_power = pyo.Reference(
            self.aggregate_power[:, "electrical"]
        )
        self.aggregate_thermal_power = pyo.Var(
            tb.time_index,
            initialize=0.0,
            units=pyunits.kW,
            doc="Aggregate thermal duty across all temperatures (kW).",
        )

        def _therm_rule(_b, t):
            return self.aggregate_thermal_power[t] == (
                sum(self.aggregate_power[t, c] for c in thermal_carriers)
                + 0 * pyunits.kW
            )

        self.eq_aggregate_thermal_power = pyo.Constraint(
            tb.time_index, rule=_therm_rule
        )

    def _build_fuel_aggregation(self, tb) -> None:
        """Build the indexed per-fuel volumetric aggregation (Var + Constraint).

        Pulls every registered fuel-usage flow from the model, buckets it by fuel
        name, and defines ``aggregate_fuel_usage[t, fuel]`` directly in EECO's
        m³/hr — so this Var *is* the series EECO bills, with no heating value
        applied anywhere. ``pyunits.convert(v[t], m³/hr)`` loudly rejects a flow
        that is not a volumetric rate. Fuels are discovered here rather than
        declared, so a model with no fuel gets an empty Var and no gas leg.
        """
        vars_by_fuel: dict[str, list] = {}
        for _block, registry in iter_io_registry(self.model()):
            for rec in registry.fuel:
                vars_by_fuel.setdefault(rec.fuel_name, []).append(rec.var)

        self._fuel_names = sorted(vars_by_fuel)
        self.aggregate_fuel_usage = pyo.Var(
            tb.time_index,
            self._fuel_names,
            initialize=0.0,
            units=EECO_GAS_USAGE_UNITS,
            doc=f"Aggregate fuel usage by fuel ({EECO_GAS_USAGE_UNITS}).",
        )

        def _agg_rule(_b, t, fuel):
            return self.aggregate_fuel_usage[t, fuel] == sum(
                pyunits.convert(v[t], EECO_GAS_USAGE_UNITS) for v in vars_by_fuel[fuel]
            )

        self.eq_aggregate_fuel_usage = pyo.Constraint(
            tb.time_index, self._fuel_names, rule=_agg_rule
        )

    def _price_for(self, carrier: str, per_quantity_units):
        """Return the configured native price for ``carrier``, or ``None``.

        A carrier priced here is billed natively and never sent to EECO. The
        normalized price is read from the registry ``build`` validated, and any
        dimensionless value is given ``per_quantity_units`` — inference happens
        here, at the leg being billed, because only the caller knows what quantity
        the carrier is metered on.

        Args:
            carrier: A power carrier key (``"electrical"``) or a fuel name.
            per_quantity_units: Units to assume for a bare price, as currency over
                the metered quantity (e.g. ``USD/kWh``).

        Returns:
            ``None`` if not priced, a units-carrying price for a flat price, or a
            ``{time point: units-carrying price}`` dict for a per-period price.
        """
        terms = self._prices.get(carrier)
        if terms is None:
            return None
        if isinstance(terms, list):
            return {
                t: _price_in_units(price, per_quantity_units)
                for t, price in zip(
                    self.config.time_block.time_index, terms, strict=True
                )
            }
        return _price_in_units(terms, per_quantity_units)

    def _require_priced(self, carrier: str, utility: str) -> None:
        """Fail if nothing prices ``carrier`` — neither a flat price nor a tariff.

        Without this an unpriced carrier would silently contribute ``0`` to the
        bill, which looks like a working model rather than a missing tariff.

        Args:
            carrier: The carrier/fuel name, named in the error.
            utility: The EECO utility that would have to price it.

        Raises:
            FlexConfigError: If no native price covers ``carrier`` and the tariff
                does not cover ``utility``.
        """
        if carrier in self._prices:
            return
        if utility in self._tariff_utilities:
            return
        priced = sorted(self._tariff_utilities) or "nothing"
        raise FlexConfigError(
            f"Nothing prices the {carrier!r} carrier: the tariff has no "
            f"{utility!r} rows (it prices {priced}) and energy_prices has no "
            f"{carrier!r} entry. Add {utility!r} charges to the tariff, or give it "
            f"a flat price: energy_prices={{{carrier!r}: ...}}.",
            field="energy_prices",
            value=carrier,
        )

    def _build_opex(self, tb, cur, dt_hours) -> None:
        """Build the ``opex`` block: electricity + fuel + fixed + scalar (Vars).

        Each energy carrier is billed one of two ways: a price from
        ``energy_prices`` is applied natively here (flat, or per time point), and
        anything else goes to the
        ``opex.py`` bridge so EECO owns the tariff cost math. Fuel legs are built
        on per-fuel sub-blocks so EECO's ``gas_*`` components never collide.
        Non-energy scalar costs are always native. A final unit-consistency check
        forces every operating-cost line item onto the base currency or errors
        loudly.
        """
        self.opex = pyo.Block(
            doc="All operating cost: electricity + fuel + fixed + scalar."
        )
        opex = self.opex

        # --- electricity: a native price, or EECO against the tariff --------
        self._require_priced("electrical", "electric")
        opex.electricity_cost = pyo.Var(
            initialize=0.0, units=cur, doc="Electricity cost (base currency)."
        )
        price = self._price_for("electrical", cur / pyunits.kWh)
        if price is not None:
            opex.eq_electricity_cost = pyo.Constraint(
                expr=opex.electricity_cost
                == self._priced_integral(
                    pyo.Reference(self.aggregate_power[:, "electrical"]),
                    price,
                    tb,
                    dt_hours,
                )
            )
        else:
            # EECO bills bare numbers, so hand it the magnitude in EECO's units
            # (convert, then divide the units out) rather than a units-carrying
            # expression, which would make EECO's own conversion constraints
            # dimensionally inconsistent.
            opex.eeco_aggregate_electrical_power = pyo.Var(
                tb.time_index,
                initialize=0.0,
                doc=f"Aggregate electrical power as a bare {EECO_POWER_UNITS} number.",
            )

            def _norm_elec(_b, t):
                return (
                    opex.eeco_aggregate_electrical_power[t]
                    == pyunits.convert(
                        self.aggregate_power[t, "electrical"], EECO_POWER_UNITS
                    )
                    / EECO_POWER_UNITS
                )

            opex.eq_eeco_aggregate_electrical_power = pyo.Constraint(
                tb.time_index, rule=_norm_elec
            )
            elec = add_electricity_cost(
                block=opex,
                electrical_power=opex.eeco_aggregate_electrical_power,
                time_index=tb.datetime_index,
                dt_hours=dt_hours,
                tariff=self._tariff,
                dr_config=self.dr,
                prorate=self.config.prorate_monthly_charges,
            )
            opex.eq_electricity_cost = pyo.Constraint(
                expr=opex.electricity_cost == elec.total_operating_cost * cur
            )

        # --- fuels: bill each discovered fuel on its own sub-block ---------
        fuel_names = self._fuel_names
        for name in fuel_names:
            self._build_fuel_leg(tb, cur, dt_hours, name)

        opex.fuel_cost = pyo.Var(
            initialize=0.0, units=cur, doc="Total EECO fuel cost (base currency)."
        )
        opex.eq_fuel_cost = pyo.Constraint(
            expr=opex.fuel_cost
            == sum(getattr(opex, f"fuel_cost_{n}") for n in fuel_names) + 0 * cur
        )

        # --- fixed operating cost (a config constant) ---------------------
        opex.fixed_operating_cost = pyo.Param(
            initialize=self.config.fixed_operating_cost,
            mutable=True,
            units=cur,
            doc="Non-tariff fixed operating cost over the horizon (base currency).",
        )

        # --- non-energy scalar costs (native; never via EECO) -------------
        scalar_names = sorted(self._registered_scalar_costs)
        for name in scalar_names:
            self._build_scalar_leg(tb, cur, dt_hours, name)

        opex.scalar_cost = pyo.Var(
            initialize=0.0,
            units=cur,
            doc="Total non-energy scalar cost (base currency).",
        )
        opex.eq_scalar_cost = pyo.Constraint(
            expr=opex.scalar_cost
            == sum(getattr(opex, f"scalar_cost_{n}") for n in scalar_names) + 0 * cur
        )

        # --- total operating cost -----------------------------------------
        # The grand total sums the category sub-totals. Add a new cost category by
        # building its sub-total Var and appending its local name here; nothing
        # else changes. (0 * cur seeds the currency and guards an empty list.)
        line_items = [
            "electricity_cost",
            "fuel_cost",
            "fixed_operating_cost",
            "scalar_cost",
        ]
        opex.total_operating_cost = pyo.Var(
            initialize=0.0,
            units=cur,
            doc="electricity + fuel + fixed + scalar (base currency).",
        )
        opex.eq_total_operating_cost = pyo.Constraint(
            expr=opex.total_operating_cost
            == sum((getattr(opex, n) for n in line_items), 0 * cur)
        )

        self._build_dr()  # no-op in v0
        self._assert_cost_units_consistent(fuel_names, scalar_names)

    def _priced_integral(self, series, price, tb, dt_hours, *, convert_to=None):
        """Return ``Σ_t price[t] × series[t] × dt`` — the one native cost formula.

        Shared by every natively priced line item: a natively priced energy carrier
        or fuel, and any registered scalar cost. ``price`` carries Pyomo units, so
        the resulting constraint is dimensionally checked and a price whose units
        do not reconcile with the metered quantity raises rather than producing a
        silently wrong number. A flat price is factored out of the sum; a
        per-period price (a ``{t: price}`` mapping) multiplies term by term.

        The integration domain is the *series'* own index set rather than an
        assumed one, and the timestep is passed in — so this stays correct if a
        series is ever defined over a subset of time. It is still guarded to the
        model's single time index, because everything upstream (aggregation, the
        EECO ``block.t``, ``TimeBlock`` discovery) assumes one uniform grid;
        costing a batch or cyclic unit on its own time index needs those changed
        first, and until then a foreign index is an error, not a wrong bill.

        Args:
            series: A time-indexed rate (Var/Expression/Reference).
            price: A units-carrying price per unit of the integrated quantity, or a
                ``{time point: price}`` mapping of them.
            tb: The TimeBlock, whose ``time_index`` the series must be on.
            dt_hours: Timestep length in hours.
            convert_to: Optional units to convert each ``series[t]`` into first.

        Returns:
            The cost expression.

        Raises:
            FlexConfigError: If ``series`` is not indexed by the TimeBlock's
                ``time_index``.
        """
        index = series.index_set()
        if index is not tb.time_index:
            raise FlexConfigError(
                f"Cannot cost a series indexed by {index.name!r}: native pricing "
                f"integrates over the model's time index ({tb.time_index.name!r}). "
                "Costing a series on its own time index (a batch or cyclic unit) "
                "is not supported yet — the power/fuel aggregation and the EECO "
                "bridge are built on a single uniform grid.",
                field="energy_prices",
                value=index.name,
            )

        def quantity(t):
            if convert_to is None:
                return series[t]
            return pyunits.convert(series[t], convert_to)

        if isinstance(price, Mapping):
            total = sum(price[t] * quantity(t) for t in index)
        else:
            total = price * sum(quantity(t) for t in index)
        return total * dt_hours * pyunits.hr

    def _build_fuel_leg(self, tb, cur, dt_hours, name: str) -> None:
        """Bill one fuel: natively on its own price, or EECO's gas leg on the tariff.

        ``aggregate_fuel_usage`` is already in EECO's units, so the native path
        bills it directly through a ``Reference``. The tariff path normalizes it to
        a bare number first, as EECO bills magnitudes, not units-carrying
        expressions. No heating value is applied either way.
        """
        opex = self.opex
        self._require_priced(name, "gas")
        usage = pyo.Reference(self.aggregate_fuel_usage[:, name])
        cost = pyo.Var(
            initialize=0.0, units=cur, doc=f"Cost of fuel {name} (base currency)."
        )
        opex.add_component(f"fuel_cost_{name}", cost)

        price = self._price_for(name, cur / pyunits.m**3)
        if price is not None:
            opex.add_component(
                f"eq_fuel_cost_{name}",
                pyo.Constraint(
                    expr=cost == self._priced_integral(usage, price, tb, dt_hours)
                ),
            )
            return

        # EECO namespaces its gas_* components by utility, not by fuel; give each
        # fuel its own sub-block so multiple fuels never collide.
        leg = pyo.Block()
        opex.add_component(f"fuel_{name}", leg)
        leg.eeco_aggregate_fuel_usage = pyo.Var(
            tb.time_index,
            initialize=0.0,
            doc=f"Aggregate {name} usage as a bare {EECO_GAS_USAGE_UNITS} number.",
        )

        def _norm_usage(_b, t):
            return (
                leg.eeco_aggregate_fuel_usage[t]
                == pyunits.convert(
                    self.aggregate_fuel_usage[t, name], EECO_GAS_USAGE_UNITS
                )
                / EECO_GAS_USAGE_UNITS
            )

        leg.eq_eeco_aggregate_fuel_usage = pyo.Constraint(
            tb.time_index, rule=_norm_usage
        )
        fuel = add_fuel_cost(
            block=leg,
            fuel_power=leg.eeco_aggregate_fuel_usage,
            time_index=tb.datetime_index,
            dt_hours=dt_hours,
            tariff=self._tariff,
            dr_config=self.dr,
            prorate=self.config.prorate_monthly_charges,
        )
        opex.add_component(
            f"eq_fuel_cost_{name}",
            pyo.Constraint(expr=cost == fuel.total_operating_cost * cur),
        )

    def _build_scalar_leg(self, tb, cur, dt_hours, name: str) -> None:
        """Build one native scalar-cost line item (price × Σ quantity × dt)."""
        opex = self.opex
        spec = self._registered_scalar_costs[name]
        # spec.price is a bare currency/quantity number; attach
        # cur/(quantity_units*hr) to
        # make it a units-carrying price, so a mis-unit quantity raises.
        price = spec.price * cur / (spec.quantity_units * pyunits.hr)
        cost = pyo.Var(
            initialize=0.0, units=cur, doc=f"Scalar cost {name} (base currency)."
        )
        opex.add_component(f"scalar_cost_{name}", cost)
        opex.add_component(
            f"eq_scalar_cost_{name}",
            pyo.Constraint(
                expr=cost
                == self._priced_integral(
                    spec.quantity,
                    price,
                    tb,
                    dt_hours,
                    convert_to=spec.quantity_units,
                )
            ),
        )

    def _assert_cost_units_consistent(self, fuel_names, scalar_names) -> None:
        """Force every operating-cost line item onto the tariff currency, or error.

        Runs Pyomo's unit-consistency check over the operating-cost equality
        constraints and re-raises any inconsistency as a :class:`FlexConfigError`
        (the "force consistency or loudly error" rule): the fixed operating cost
        and any scalar cost must reconcile with the tariff's currency.
        """
        opex = self.opex
        constraints = [
            opex.eq_total_operating_cost,
            opex.eq_electricity_cost,
            opex.eq_fuel_cost,
            opex.eq_scalar_cost,
        ]
        constraints += [getattr(opex, f"eq_fuel_cost_{n}") for n in fuel_names]
        constraints += [getattr(opex, f"eq_scalar_cost_{n}") for n in scalar_names]
        try:
            for con in constraints:
                assert_units_consistent(con)
        except UnitsError as exc:
            raise FlexConfigError(
                "Operating-cost line items are not dimensionally consistent with "
                f"the tariff currency ({self._currency}); reconcile the units: "
                f"{exc}",
                field="fixed_operating_cost",
            ) from exc

    def _build_capex(self, cur) -> None:
        """Build the empty ``capex`` placeholder block (total_capital_cost == 0)."""
        self.capex = pyo.Block(doc="Capital cost (empty placeholder in v0).")
        self.capex.total_capital_cost = pyo.Var(
            initialize=0.0,
            units=cur,
            doc="Sum of registered units' capital cost; 0 in v0.",
        )
        self.capex.eq_total_capital_cost = pyo.Constraint(
            expr=self.capex.total_capital_cost == 0 * cur
        )

    def _effective_rate(self) -> float:
        """Return the single annual rate the capital recovery factor is built on.

        Two rates can be configured and they play different roles: ``interest_rate``
        is the nominal cost of capital, and ``discount_rate`` deflates it to a real
        rate. Combining them gives one effective rate,
        ``(1 + interest) / (1 + discount) - 1``. With ``interest_rate`` unset the
        effective rate is just ``discount_rate``, so a config that predates the
        second rate annualizes exactly as before.

        Returns:
            The effective annual rate (a fraction; may be negative).

        Raises:
            FlexConfigError: If the rates imply an effective rate ``<= -1``, where
                the capital recovery factor is undefined.
        """
        discount = self.config.discount_rate
        interest = self.config.interest_rate
        rate = discount if interest is None else (1 + interest) / (1 + discount) - 1
        if rate <= -1.0:
            raise FlexConfigError(
                f"interest_rate={interest!r} and discount_rate={discount!r} imply "
                f"an effective rate of {rate:.4g}, but the capital recovery factor "
                "is only defined for an effective rate > -1. Check both rates are "
                "fractions (0.08, not 8).",
                field="interest_rate",
                value=interest,
            )
        if rate < 0:
            _log.warning(
                "Annualizing %s at a negative effective rate (%.4g) from "
                "interest_rate=%r and discount_rate=%r.",
                self.name,
                rate,
                interest,
                discount,
            )
        return rate

    def _build_totals_and_annualization(self, tb, cur) -> None:
        """Build the aggregate cost Vars, the design-mode total, and annualized cost."""
        self.aggregate_operating_cost = pyo.Var(
            initialize=0.0,
            units=cur,
            doc="IDAES-aggregate name for the opex total (operations objective).",
        )
        self.eq_aggregate_operating_cost = pyo.Constraint(
            expr=self.aggregate_operating_cost == self.opex.total_operating_cost
        )

        self.aggregate_capital_cost = pyo.Var(
            initialize=0.0,
            units=cur,
            doc="IDAES-aggregate name for the capex total (0 in v0).",
        )
        self.eq_aggregate_capital_cost = pyo.Constraint(
            expr=self.aggregate_capital_cost == self.capex.total_capital_cost
        )

        self.total_cost = pyo.Var(
            initialize=0.0,
            units=cur,
            doc="Design-mode objective: operating + capital cost (base currency).",
        )
        self.eq_total_cost = pyo.Constraint(
            expr=self.total_cost
            == self.aggregate_operating_cost + self.aggregate_capital_cost
        )

        # Annualization: capital recovery factor + opex scaled horizon -> year.
        rate = self._effective_rate()
        n = self.config.lifetime_years
        if n <= 0:
            raise FlexConfigError(
                "lifetime_years must be > 0 (the capital recovery factor is "
                f"undefined otherwise); got {n}.",
                field="lifetime_years",
                value=n,
            )
        crf = (1.0 / n) if rate == 0 else rate * (1 + rate) ** n / ((1 + rate) ** n - 1)
        self.lifetime = pyo.Param(
            initialize=n,
            mutable=True,
            units=pyunits.year,
            doc="Plant lifetime (years).",
        )
        self.effective_rate = pyo.Param(
            initialize=rate,
            mutable=True,
            units=pyunits.dimensionless,
            doc="Effective annual rate behind the capital recovery factor.",
        )
        self.capital_recovery_factor = pyo.Param(
            initialize=crf,
            mutable=True,
            units=1 / pyunits.year,
            doc="Capital recovery factor (1/year).",
        )
        horizon_years = pyo.value(pyunits.convert(tb.horizon, pyunits.year))
        self.annualized_cost = pyo.Var(
            initialize=0.0,
            units=cur / pyunits.year,
            doc="Total cost on an annual basis (base currency per year).",
        )
        self.eq_annualized_cost = pyo.Constraint(
            expr=self.annualized_cost
            == self.aggregate_operating_cost / (horizon_years * pyunits.year)
            + self.aggregate_capital_cost * self.capital_recovery_factor
        )

    # -- design / operations modes --

    def set_operations_mode(self) -> None:
        """Fix every registered sizing Var and deactivate its capex constraint.

        The operations objective is ``aggregate_operating_cost`` alone. In the
        sizing registry is empty, so this is a no-op; it is the documented
        single-model toggle later milestones populate. Idempotent.
        """
        for entry in self._registered_sizing:
            entry.var.fix()
            if entry.capex_constraint is not None:
                entry.capex_constraint.deactivate()

    def set_design_mode(self) -> None:
        """Unfix every registered sizing Var and activate its capex constraint.

        The design objective is ``total_cost`` (operating + capital).
        """
        for entry in self._registered_sizing:
            entry.var.unfix()
            if entry.capex_constraint is not None:
                entry.capex_constraint.activate()

    # -- reported cost (post-solve; never the objective, §6) --------

    def _natively_priced_cost(
        self,
        realized,
        price,
        dt_hours: float,
        time_index,
        *,
        quantity_units=EECO_POWER_UNITS,
    ) -> float:
        """Recompute a natively priced carrier's realized cost, off the model.

        The reporting counterpart of :meth:`_priced_integral`: same
        ``Σ price × quantity × dt``, evaluated on the realized array rather than
        built as a constraint, so the reported cost is a genuine recomputation and
        never a read of the objective. Needs no EECO.

        Args:
            realized: The realized per-timestep rates, in ``time_index`` order.
            price: The units-carrying price, or a ``{time point: price}`` mapping.
            dt_hours: Timestep length in hours.
            time_index: The time points ``realized`` was sampled over, used to look
                up a per-period price.
            quantity_units: The units ``realized`` is in.

        Returns:
            The horizon-total cost in the base currency, as a float.
        """
        if isinstance(price, Mapping):
            priced = sum(
                price[t] * float(value)
                for t, value in zip(time_index, realized, strict=True)
            )
        else:
            priced = price * float(realized.sum())
        total = priced * quantity_units * dt_hours * pyunits.hr
        return float(pyo.value(pyunits.convert(total, self._currency)))

    def report_cost(self, model) -> CostReport:
        """Return the reported, categorized cost, evaluated **post-solve**.

        The user-facing cost (§6 reporting rule; M13 surfaces it). Operating
        electricity/fuel are EECO **post-hoc** evaluations on the realized
        dispatch; fixed is the config constant; scalar costs are recomputed
        natively; DR revenue is ``0`` in v0 (containers-only); capital is read off
        the (empty in v0) capex block. This is an independent recomputation —
        never ``value(model.objective)``, which is a relaxed/scalarized proxy.

        Args:
            model: The solved model (accepted for the documented API; the costing
                block reads its own components).

        Returns:
            The :class:`CostReport` breakdown, whose ``currency`` names the basis
            every value in it is a magnitude in.
        """
        tb = self.config.time_block
        dt_hours = pyo.value(pyunits.convert(tb.dt, pyunits.hr))

        realized_power = np.array(
            [pyo.value(self.aggregate_electrical_power[t]) for t in tb.time_index]
        )
        price = self._price_for("electrical", self._currency / pyunits.kWh)
        if price is not None:
            electricity = self._natively_priced_cost(
                realized_power, price, dt_hours, tb.time_index
            )
        else:
            electricity = evaluate_cost(
                realized_power,
                self._tariff,
                dt_hours,
                dr_config=self.dr,
                time_index=tb.datetime_index,
                prorate=self.config.prorate_monthly_charges,
            )

        fuel = 0.0
        for name in self._fuel_names:
            realized_usage = np.array(
                [pyo.value(self.aggregate_fuel_usage[t, name]) for t in tb.time_index]
            )
            price = self._price_for(name, self._currency / pyunits.m**3)
            if price is not None:
                fuel += self._natively_priced_cost(
                    realized_usage,
                    price,
                    dt_hours,
                    tb.time_index,
                    quantity_units=EECO_GAS_USAGE_UNITS,
                )
            else:
                fuel += evaluate_fuel_cost(
                    realized_usage,
                    self._tariff,
                    dt_hours,
                    dr_config=self.dr,
                    time_index=tb.datetime_index,
                    prorate=self.config.prorate_monthly_charges,
                )

        fixed = float(pyo.value(self.opex.fixed_operating_cost))

        scalar = 0.0
        for spec in self._registered_scalar_costs.values():
            scalar += (
                spec.price
                * dt_hours
                * sum(
                    pyo.value(pyunits.convert(spec.quantity[t], spec.quantity_units))
                    for t in tb.time_index
                )
            )

        dr_revenue = 0.0  # DR containers-only (do not fabricate a credit)
        operating = OperatingCostBreakdown(
            electricity=electricity,
            fuel=fuel,
            fixed=fixed,
            scalar=scalar,
            dr_revenue=dr_revenue,
            total=electricity + fuel + fixed + scalar - dr_revenue,
        )
        # Capex is an empty placeholder in v0 -> no per-component capital costs.
        by_component: dict[str, float] = {}
        capital = CapitalCostBreakdown(
            by_component=by_component,
            total=float(pyo.value(self.aggregate_capital_cost)),
        )
        return CostReport(
            operating=operating,
            capital=capital,
            total=operating.total + capital.total,
            currency=str(self.base_currency),
        )
