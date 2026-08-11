flexops.costing
===============

.. currentmodule:: flexops.costing

EECO integration
----------------

flex-pse does **not** build its own tariff/cost engine. Tariffs, demand
charges, tiered/fixed charges, and both the optimization-time and
post-optimization cost computations come from the external **EECO** package
(``eeco`` on PyPI), a core runtime dependency (architecture §2.4, decisions
R4/R9). ``flexops.costing.opex`` is the thin flex-pse interface around it —
and, by convention (decision R12), the **only** module in the codebase that
imports ``eeco``, so there is one file to fix when EECO's API moves.

EECO owns all cost math; these wrappers are glue: they marshal inputs, rename
EECO's outputs to stable flex-pse names, and translate EECO/pandas errors into
the flex-pse exception hierarchy. A flex-pse tariff object is simply an EECO
``rate_data`` ``DataFrame`` (EECO 0.3.0 has no tariff loader of its own; its
cost functions consume that frame directly).

.. note:: **EECO is only needed for tariffs.** The ``eeco`` import is soft, so
   every carrier given a native :ref:`energy price <native-energy-prices>`
   builds, solves, and reports without EECO installed — including
   ``report_cost``. Reaching a tariff path without ``eeco`` raises
   :class:`~flexcore.exceptions.FlexConfigError` naming both remedies (install
   it, or price the carrier natively).

Loaders and CSV conversion
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_tariff
   merge_tariffs
   load_dr_program
   tariff_csv_to_dict
   tariff_currency_units
   currency_units
   monthly_scale_factor

**Several tariff files** merge into one frame with :func:`merge_tariffs`, which
is what ``FlexCosting(tariff_file=...)`` uses when given a list or a mapping.
Passing a list merges every source as-is; passing a mapping of EECO utility to
source *assigns* each file to a utility and keeps only that utility's rows, so a
sheet carrying stray rows for another utility cannot leak into the bill::

    m.costing = fo.FlexCosting(
        time_block=m.time_block,
        tariff_file={"electric": "pge_b20.json", "gas": "socal_gn10.json"},
    )

Both forms end at a single ``rate_data`` frame — there is no separate tariff
object per utility, because the frame's ``utility`` column is what selects rows
for each cost leg. Two sources defining the same charge raise rather than
silently doubling that line item (EECO sums colliding charge keys).

A tariff may be authored as a JSON records structure or imported from an EECO
``rate_data`` CSV. The JSON form is a ``tariff_data`` records list, e.g. (an
excerpt of the demo ``flexdemo-b20`` time-of-use tariff)::

    {
      "tariff_data": [
        {"utility": "electric", "type": "energy", "name": "peak",
         "month_start": 6, "month_end": 9, "weekday_start": 0, "weekday_end": 4,
         "hour_start": 16, "hour_end": 21, "basic_charge_limit (metric)": 0,
         "charge (metric)": 0.18, "units": "$/kWh"},
        {"utility": "electric", "type": "demand", "name": "peak-demand",
         "month_start": 6, "month_end": 9, "weekday_start": 0, "weekday_end": 4,
         "hour_start": 16, "hour_end": 21, "basic_charge_limit (metric)": 0,
         "charge (metric)": 21.5, "units": "$/kW"}
      ]
    }

Charge windows are half-open on the hour: ``hour_start=16, hour_end=21`` bills
hours 16:00–20:59 inclusive (21:00 is off-peak).

An optional ``assessed`` column controls the billing period of a ``demand``
charge row: EECO defaults to ``"monthly"`` when the column is absent or the
row omits it, applying one demand-charge epigraph over the whole
``month_start``–``month_end`` window. Set ``"assessed": "daily"`` on a demand
row to instead apply a separate epigraph per calendar day within that
window — the pattern behind daily demand charges on tariffs such as those
increasingly common in California.

Tariff signal helpers
----------------------

Plain-pandas signals over a tariff, for writing logic/heuristic constraints.
Each is a flex-pse helper built on EECO's ``get_charge_dict`` charge arrays
(the source of the price data); EECO 0.2.1 exposes no per-stamp price accessor.

.. autosummary::
   :toctree: generated
   :nosignatures:

   price_series
   is_peak
   peak_windows
   price_gradient

In-objective cost bridge
------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   add_operating_cost
   add_electricity_cost
   add_fuel_cost
   OperatingCostHandles

:func:`add_operating_cost` is the facility-level umbrella: it builds **both**
the electricity and fuel costs onto one opex block (EECO namespaces its Pyomo
components by utility, so ``electric_*`` and ``gas_*`` never collide) and returns
a single :class:`OperatingCostHandles` whose ``total_operating_cost`` is the sum
across utilities. The facility consumption defaults to the standard series on the
block — ``block.power_electrical`` and ``block.fuel_usage`` — so a caller need not
re-declare them each use; pass ``electrical_power``/``fuel_power`` to override.
Every series handed to these builders must carry **no units**: a bare magnitude in
kW (electric) or m³/hr (fuel volumetric flow), because EECO applies its unit
conversion as a plain float factor and constrains its own dimensionless Vars to
the series it is given. The
single-utility builders :func:`add_electricity_cost` and :func:`add_fuel_cost`
remain available for building one leg alone. :func:`add_fuel_cost` takes a
``fuel_type`` (default ``"gas"``, the only value EECO 0.2.1 supports); a
hydrogen utility is expected upstream and will add a second value.

Post-optimization evaluators
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate_cost
   evaluate_fuel_cost

Demand response (containers-only in v0)
---------------------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   DRConfig

.. note::

   Demand response is **containers-only** in v0 (architecture §2.4). A
   :class:`DRConfig` holds a loaded DR program so the wiring exists, and the
   internal DR hook is a no-op: supplying a DR file never changes the
   objective. Building DR event/curtailment/incentive/capacity constraints is
   post-v0. EECO 0.2.1 exposes no DR API, so the DR file format is a flex-pse
   placeholder loaded into the container only.

In-objective vs. reported cost
------------------------------

EECO is used two ways. :func:`add_operating_cost` asks EECO to build the
**convex-relaxed** operating-cost ``Expression`` on a Pyomo block — the
tractable proxy the scheduler minimizes. :func:`evaluate_cost` evaluates EECO on
a **fixed, realized** aggregate-power numpy array to compute the TRUE
(de-relaxed) cost — the user-facing bill (§6 reporting rule, R4/R9).

Because the relaxation drops the tiered energy surcharge when no consumption
estimate is supplied, the in-objective total is a proxy that is **≤ or ≈** the
post-hoc true bill. The raw solver objective is never reported as the
user-facing cost.

.. admonition:: Timezones / DST

   EECO reasons in naive **local wall-clock time**: its charge windows are keyed
   on ``datetime.month``/``weekday``/``hour`` with no timezone conversion.
   flex-pse v0 is consistently naive-local (matching
   :class:`~flexops.core.time_block.TimeBlock`). Timezone-aware datetime indices
   are rejected at the wrapper boundary with
   :class:`~flexcore.exceptions.FlexDataError`; strip the timezone
   (``index.tz_localize(None)``) before passing an index in.

FlexCosting block
-----------------

.. currentmodule:: flexops.costing.flex_costing

.. autosummary::
   :toctree: generated
   :nosignatures:

   FlexCosting
   FlexCostingData
   CostReport
   OperatingCostBreakdown
   CapitalCostBreakdown
   ScalarCostSpec

``FlexCosting`` subclasses IDAES ``FlowsheetCostingBlockData`` and **delegates
all tariff/energy operating cost to EECO** (decision R4), in two ways: it hands
EECO the aggregate electrical power (as a bare kW magnitude, via a dimensionless
``opex.eeco_aggregate_electrical_power`` normalization Var) + tariff to build the
convex-relaxed in-objective cost
(:func:`~flexops.costing.add_electricity_cost`), and post-solve
calls EECO's evaluator (:func:`~flexops.costing.evaluate_cost`) for the reported
bill. Its own jobs are aggregation, the ``opex``/``capex`` block structure and
naming, CapEx + modes, and ``report_cost``; it writes no tariff cost math.

Every quantity FlexCosting exposes is a **decision-visible** ``Var`` defined by an
``eq_<name>`` equality ``Constraint`` — aggregate power, each cost line item, the
annualized cost, and the totals are first-class model variables (not bare
Expressions).

Every cost lives in one of two sub-blocks built by
:meth:`~FlexCostingData.cost_process`:

* **``opex``** holds all operating cost — ``electricity_cost`` and ``fuel_cost``
  (both from EECO), ``fixed_operating_cost`` (a non-tariff facility cost:
  maintenance/labor/chemicals, from ``CostingConfig.fixed_operating_cost``), and
  ``scalar_cost`` (non-energy flows/supplies/products, below). Their sum,
  ``total_operating_cost``, is re-exposed as ``aggregate_operating_cost`` — the
  operations-mode objective. The fixed operating cost is **distinct** from the
  tariff's own ``fixed_charge``, which EECO already folds into ``electricity_cost``.
* **``capex``** is an **empty placeholder** in v0 (``total_capital_cost == 0``,
  re-exposed as ``aggregate_capital_cost``); later milestones aggregate per-unit
  capital costs into it.

.. note:: **Indexed per-carrier power aggregation.**

   :meth:`~FlexCostingData.cost_process` pulls every registered power draw from
   the model and defines ``aggregate_power[t, carrier]`` in kW, where ``carrier``
   is ``"electrical"`` or a per-temperature thermal label ``"thermal@<T>K"``.
   Every draw is normalized to kW with ``pyunits.convert`` at aggregation, so a
   non-power (or non-kW-convertible) draw fails **loudly**. Thermal duties at
   **different temperatures are never summed** together — each temperature is its
   own carrier; ``aggregate_thermal_power`` is a temperature-blind total kept for
   backward compatibility.

.. note:: **Fuel is a volume, not a power.**

   A combustible fuel is metered and billed on **volume**, so it is aggregated
   separately from power: ``cost_process`` pulls every fuel-usage flow a unit
   registered with
   :meth:`~flexops.core.ops_block.OpsBlockData.register_fuel_usage` and defines
   ``aggregate_fuel_usage[t, fuel]`` directly in EECO's **m³/hr** (a flow that is
   not a volumetric rate fails **loudly** there). Each fuel is billed through
   :func:`~flexops.costing.add_fuel_cost` against the **same tariff** loaded at
   construction, on its own ``opex.fuel_<name>`` sub-block so EECO's ``gas_*``
   components never collide — via that sub-block's dimensionless
   ``eeco_aggregate_fuel_usage`` Var, the m³/hr aggregate as a bare magnitude.

   Fuels are **discovered from the model**, so there is nothing to declare on the
   costing block; a model with no fuel flow builds no gas leg and
   ``opex.fuel_cost`` is ``0``. flex-pse applies **no heating value** and does
   **no** fuel-type recognition: a fuel priced in the tariff sheet's
   ``gas``-utility rows just works, a tariff missing those rows surfaces EECO's
   own validation error, and if a tariff prices gas on an energy basis, converting
   it is EECO's job (it applies its own fuel heating-value assumption).

.. note:: **Non-energy scalar costs — native, never via EECO.**

   :meth:`~FlexCostingData.register_scalar_cost` costs an arbitrary time-indexed
   rate (a flow/supply/product) as ``price × Σ_t quantity[t] × dt`` — e.g. water
   withdrawal per m³, chemical dosing per kg, or a product-revenue credit (a
   negative ``price``). Built entirely in flex-pse; EECO is not involved. A
   ``quantity`` that does not convert to the declared ``quantity_units`` raises,
   forcing unit consistency.

.. _native-energy-prices:

.. note:: **Native energy prices — no tariff, no EECO.**

   A carrier does not need a tariff at all: give ``energy_prices`` a price per
   carrier or fuel and that carrier is billed natively as
   ``Σ_t price[t] × quantity[t] × dt``, exactly like a scalar cost::

       m.costing = fo.FlexCosting(
           time_block=m.time_block,
           energy_prices={
               "electrical":  0.12 * currency_units("USD") / pyunits.kWh,
               "natural_gas": 0.50 * currency_units("USD") / pyunits.m**3,
           },
       )

   Keys are ``"electrical"`` or a registered fuel name. A native price **wins
   over a tariff** that also covers that carrier, so the two can be mixed freely
   — tariff-priced electricity alongside a natively priced fuel, say.

   **A price may vary over the horizon.** Each entry is one of three things: a
   single value (flat over the horizon), an array-like with one value per time
   point, or a Pyomo component indexed over the horizon — so the price can be a
   ``Param`` you update between solves, or even a ``Var``::

       m.day_ahead = pyo.Param(
           m.time_block.time_index, initialize=hourly_prices, mutable=True,
           units=currency_units("USD") / pyunits.kWh,
       )
       m.costing = fo.FlexCosting(
           time_block=m.time_block,
           energy_prices={"electrical": m.day_ahead},
       )

   An array-like is read in order against the time index; an indexed component is
   read in its own index-set order, so it does not have to live on the
   TimeBlock's ``time_index`` — only to have exactly one value per time point.
   Anything with the wrong number of values raises
   :class:`~flexcore.exceptions.FlexConfigError` at construction, naming both
   counts. A mapping is rejected outright, because iterating it would cost its
   keys.

   **Units are explicit or inferred.** A price may carry Pyomo units
   (``0.12 * currency_units("USD") / pyunits.kWh``), which must reconcile with the
   base currency or the unit-consistency check raises. A bare number is read in
   the currency over the carrier's metered quantity: ``kWh`` for a power carrier,
   ``m³`` for a fuel — so ``{"electrical": 0.12}`` means 0.12 per kWh and
   ``{"natural_gas": 0.50}`` means 0.50 per m³, both in the base currency.

   A carrier that has registered draws but is priced by *neither* a native price
   nor a tariff covering its utility raises
   :class:`~flexcore.exceptions.FlexConfigError` naming the carrier, rather than
   contributing a silent ``0`` to the bill.

.. note:: **Sub-monthly horizons prorate the monthly-assessed charges.**

   A demand charge and a tariff's fixed (customer) charge are billed per calendar
   month, so a horizon shorter than a month owes only its share. With
   ``prorate_monthly_charges`` (the default), both are scaled by
   :func:`~flexops.costing.monthly_scale_factor` — the horizon's length over the
   real length of the calendar month it starts in. Energy charges are never
   scaled, and a demand charge the tariff marks ``"assessed": "daily"`` is
   already at horizon granularity so it is left alone. Set the option to
   ``False`` to bill full monthly charges regardless of horizon length.

   Prorating is applied to the charge *rates* before EECO sees them, which leaves
   EECO's tiered-surcharge arithmetic untouched. flex-pse does this itself because
   EECO's ``get_charge_dict`` path exposes no equivalent option: its
   ``calculate_cost`` adds the customer charge whole, and its
   ``demand_scale_factor`` is suppressed for charges spanning ``<= 1`` day —
   which, because EECO clips charge-key dates to the horizon, would silently
   include *every* monthly charge on a one-day horizon.

.. note:: **Annualization.**

   ``cost_process`` builds a ``capital_recovery_factor`` and an
   ``annualized_cost`` Var (base currency per year): operating cost scaled from
   the horizon to a year plus capital cost times the CRF. With the empty v0 capex
   block, the annualized cost is just the operating cost on an annual basis.

   The CRF is built on a single ``effective_rate``. Supplying only
   ``discount_rate`` uses it directly; supplying ``interest_rate`` as well (the
   nominal cost of capital) deflates it into a real rate,
   ``(1 + interest_rate) / (1 + discount_rate) - 1``. An effective rate of ``0``
   falls back to straight-line ``1/lifetime_years``, and an effective rate
   ``<= -1`` or a non-positive lifetime raises.

.. note:: **Currency basis.**

   The costing block's ``base_currency`` — and the units on every operating- and
   capital-cost expression it builds — is the tariff sheet's currency basis, read
   from the charge ``units`` column by
   :func:`~flexops.costing.tariff_currency_units` (EECO tariffs
   are dollar-based: ``"$"`` → ``USD``). With no tariff it is the configured
   ``currency`` (default ``USD``), available as
   :func:`~flexops.costing.currency_units` for writing flat prices. EECO's own
   cost expressions are dimensionless dollars, so FlexCosting casts them to this
   currency; the ``report_cost`` numbers are magnitudes in that currency, and
   :attr:`CostReport.currency` names it (``str(base_currency)``, e.g. ``"USD"``)
   so a report is self-describing away from the block that produced it.

.. note:: **Capital cost enters the objective only in design mode.**

   The operations-mode objective is ``aggregate_operating_cost`` alone; the
   design-mode objective is ``total_cost`` (operating **+** capital).
   :meth:`~FlexCostingData.set_operations_mode` fixes the registered sizing Vars
   and deactivates their capex constraints;
   :meth:`~FlexCostingData.set_design_mode` unfixes and activates them. Both are
   idempotent single-model toggles; in v0 the sizing registry is empty so they are
   no-ops (wiring for M08/M16). Merging representative months and linking sizing
   Vars across them is the M16 design wrapper, not this mode.

.. note:: **Reporting rule (R9, §6).**

   :meth:`~FlexCostingData.report_cost` returns a categorized :class:`CostReport`
   — capital vs. operating, each itemized — recomputed **post-solve**, never read
   off the solver objective (a relaxed/scalarized proxy). Operating electricity
   and fuel are EECO post-hoc evaluations on the realized dispatch (fuel on the
   realized ``aggregate_fuel_usage``, ``0`` when the model burns none); fixed is
   the config constant. In v0 ``dr_revenue`` and the whole ``capital`` breakdown
   are zero placeholders, so the structure is stable as those features land.

Construction-order invariant
----------------------------

``FlexCosting`` may be constructed **before any units exist** — the API-freeze
script builds ``m.costing`` before ``m.waterfacility.tank`` — because all aggregation and
the EECO call are deferred to ``cost_process()``, which pulls every unit's
registered power and fuel usage from the model. Building costing first, last, or
between units gives the identical result.
