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
``rate_data`` ``DataFrame`` (EECO 0.2.1 has no tariff loader of its own; its
cost functions consume that frame directly).

Loaders and CSV conversion
---------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_tariff
   load_dr_program
   tariff_csv_to_dict

A tariff may be authored as a JSON records structure or imported from an EECO
``rate_data`` CSV. The JSON form is a ``rate_data`` records list, e.g. (an
excerpt of the demo ``flexdemo-b20`` time-of-use tariff)::

    {
      "rate_data": [
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
   add_gas_cost
   OperatingCostHandles

:func:`add_operating_cost` is the facility-level umbrella: it builds **both**
the electricity and gas costs onto one opex block (EECO namespaces its Pyomo
components by utility, so ``electric_*`` and ``gas_*`` never collide) and returns
a single :class:`OperatingCostHandles` whose ``total_operating_cost`` is the sum
across utilities. The facility consumption defaults to the standard series on the
block — ``block.power_electrical`` and ``block.gas_usage`` — so a caller need not
re-declare them each use; pass ``electrical_power``/``gas_power`` to override. The
single-utility builders :func:`add_electricity_cost` and :func:`add_gas_cost`
remain available for building one leg alone.

Post-optimization evaluators
----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   evaluate_cost
   evaluate_gas_cost

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
