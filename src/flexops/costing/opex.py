"""EECO integration: the sole ``eeco`` import point for flex-pse.

flex-pse does **not** build its own tariff/cost engine. Tariffs, demand charges,
tiered/fixed charges, and both the optimization-time and post-optimization cost
computations come from the external **EECO** package (``eeco`` on PyPI). This
module is the thin flex-pse interface around it — loaders, a CSV→dict tariff
converter, pandas signal helpers, the in-objective Pyomo bridge, and the
post-optimization evaluator — and, by convention (``plan/00_conventions.md``
§6), the **only** file in the codebase that imports
``eeco``. Localizing the import means one file to fix when EECO's API moves.

**EECO owns the math; this file is glue.** No cost arithmetic lives here — no
price-lookup loops, no demand-charge epigraphs, no kWh conversion. Every dollar
figure is produced by ``eeco.costs``; the wrappers only marshal inputs, rename
outputs to stable flex-pse names, and translate errors into the flex-pse
exception hierarchy.

**Two ways EECO is used** (architecture §2.4):

1. *In-objective* — :func:`add_operating_cost` (the facility umbrella over the
   single-utility :func:`add_electricity_cost` / :func:`add_fuel_cost`) asks EECO
   to build the **convex-relaxed** operating-cost ``Expression`` on a Pyomo block.
   This is the tractable proxy the scheduler minimizes, not the reported bill.
2. *Post-optimization* — :func:`evaluate_cost` / :func:`evaluate_fuel_cost`
   evaluate EECO on a **fixed, realized** aggregate-power numpy array to compute
   the TRUE (de-relaxed) cost — the user-facing number (§6 reporting rule).

Because EECO convex-relaxes a non-convex pricing structure (notably the tiered
energy surcharge, which the relaxation drops when no consumption estimate is
supplied), the in-objective total is a proxy that is **≤ or ≈** the post-hoc
true bill. The raw solver objective is never the user-facing cost.

**Units.** Electrical power is a **kW** series and fuel usage is a **volumetric
m³/hr** series — always, since fuel is metered and billed on volume. Both are
passed to ``eeco`` with their units stated explicitly. EECO converts to energy
(kWh / m³) internally from the timestep, so ``dt_hours`` is passed exactly once —
never multiply by it here. flex-pse applies **no** heating value: if a tariff
prices gas on an energy basis, EECO converts it with its own fuel heating-value
assumption.

**Timezones / DST.** EECO reasons in naive local wall-clock time (its charge
windows are keyed on ``datetime.hour``/``weekday``/``month`` with no tz
conversion). flex-pse v0 is consistently naive-local (matching ``TimeBlock``);
tz-aware indices are rejected at the wrapper boundary with :class:`FlexDataError`.

**Demand response.** v0 is **containers-only** (architecture §2.4): :class:`DRConfig`
holds a loaded DR program, and the internal :func:`_build_dr` hook is a no-op.
Supplying a DR file never changes the objective. EECO 0.2.1 exposes no DR API,
so the DR file format is a flex-pse placeholder loaded into the container only.
"""

import calendar
import dataclasses
import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

# THE sole eeco import point in the whole codebase (§6). The import is soft:
# EECO is only needed to price a *tariff*. A model whose every carrier carries a
# flat price (FlexCosting's ``energy_prices``) builds, solves, and reports without
# it, so ``import flexops`` must not hard-require it. Every function that reaches
# into eeco calls _require_eeco() first.
try:
    from eeco import costs as _eeco_costs
    from eeco.units import u as _eeco_units
except ImportError:  # pragma: no cover - exercised by monkeypatching _HAS_EECO
    _eeco_costs = None
    _eeco_units = None

import flexcore.nomenclature as nm
from flexcore.exceptions import FlexConfigError, FlexDataError

_log = logging.getLogger(__name__)

_HAS_EECO = _eeco_costs is not None

# EECO rate_data columns flex-pse requires present before handing the frame to
# eeco.costs.get_charge_dict. The *choice* of which columns to require is
# flex-pse's; the column *names* are sourced from eeco's own constants so an
# upstream rename tracks automatically instead of silently breaking validation.
# Only reachable from tariff paths, so they are empty when eeco is absent.
if _HAS_EECO:
    _REQUIRED_COLUMNS = (
        _eeco_costs.UTILITY,
        _eeco_costs.TYPE,
        _eeco_costs.NAME,
        _eeco_costs.MONTH_START,
        _eeco_costs.MONTH_END,
        _eeco_costs.WEEKDAY_START,
        _eeco_costs.WEEKDAY_END,
        _eeco_costs.HOUR_START,
        _eeco_costs.HOUR_END,
    )
    # One of these charge columns must be present (eeco accepts
    # metric/imperial/bare).
    _CHARGE_COLUMNS = (
        _eeco_costs.CHARGE_METRIC,
        _eeco_costs.CHARGE_IMPERIAL,
        _eeco_costs.CHARGE,
    )
    _ELECTRIC = _eeco_costs.ELECTRIC
    _GAS = _eeco_costs.GAS
    _CUSTOMER = _eeco_costs.CUSTOMER
    _DEMAND = _eeco_costs.DEMAND
    _ASSESSED = _eeco_costs.ASSESSED
else:  # pragma: no cover - see above
    _REQUIRED_COLUMNS = ()
    _CHARGE_COLUMNS = ()
    _ELECTRIC = "electric"
    _GAS = "gas"
    _CUSTOMER = "customer"
    _DEMAND = "demand"
    _ASSESSED = "assessed"

# Fuel type -> underlying EECO utility. "gas" is the only fuel utility EECO
# 0.2.1 exposes; every registered fuel (natural gas, biogas, ...) bills through
# it today. Add an entry here once EECO exposes a hydrogen utility.
_FUEL_UTILITY = {"gas": _GAS}

# The units EECO measures in across the cost boundary: electrical power in kW and
# gas/fuel usage as a volumetric rate in m^3/hr. What crosses the boundary is a
# bare number in these units, never a units-carrying expression: eeco applies its
# unit conversion as a plain float factor and constrains its own dimensionless
# Vars to the series it is given, so a units-carrying input makes eeco's internal
# constraints dimensionally inconsistent. Callers convert to these units and
# divide them out (see FlexCosting's normalization Vars). The pint counterparts
# are passed to eeco explicitly at every call site (they match its defaults, so an
# upstream default change surfaces as a loud conversion rather than a silently
# mis-scaled bill).
EECO_POWER_UNITS = pyo.units.kW
EECO_GAS_USAGE_UNITS = pyo.units.m**3 / pyo.units.hr


def _eeco_consumption_units() -> dict:
    """Return the pint consumption units to pass EECO at every call site.

    Built on demand rather than at import so the module loads without ``eeco``
    installed (only tariff paths need it).

    Returns:
        The ``electric_consumption_units``/``gas_consumption_units`` kwargs.
    """
    return {
        "electric_consumption_units": _eeco_units.kW,
        "gas_consumption_units": _eeco_units.meter**3 / _eeco_units.hour,
    }


# Currency basis: map a tariff charge-``units`` symbol (the numerator before the
# ``/``) to a Pyomo currency unit name. EECO tariffs are dollar-based.
_CURRENCY_SYMBOLS = {"$": "USD"}

# Standard block attribute names the combined :func:`add_operating_cost` reads
# when a consumption series is not passed explicitly, both canonical nomenclature
# names: ``power_electrical`` (a bare kW magnitude) and ``fuel_usage`` — a
# volumetric gas-usage series in m³/hr, not a kW ``power_thermal`` duty.


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _require_eeco(feature: str) -> None:
    """Fail with both remedies if ``feature`` needs ``eeco`` and it is missing.

    Args:
        feature: What the caller was trying to do, named in the message.

    Raises:
        FlexConfigError: If the optional ``eeco`` package is unavailable.
    """
    if not _HAS_EECO:
        raise FlexConfigError(
            f"{feature} needs the optional 'eeco' package, which is not "
            "installed. Either install it (pip install eeco) to price against a "
            "tariff, or give this carrier a flat price instead — e.g. "
            "FlexCosting(energy_prices={'electrical': 0.12 * pyunits.USD / "
            "pyunits.kWh}), which needs no tariff and no eeco.",
            field="tariff_source",
        )


def monthly_scale_factor(time_index: pd.DatetimeIndex, dt_hours: float) -> float:
    """Fraction of its starting calendar month that a horizon covers (``<= 1``).

    Monthly-assessed charges — the demand charge and the tariff's fixed
    (customer) charge — are billed per calendar month, so a horizon shorter than
    a month must only pay its share of them. This is that share, computed from
    the real length of the month the horizon starts in (February and 30-day
    months differ), and clamped to ``1.0`` so a full-month horizon pays the full
    charge and nothing is ever scaled up.

    Args:
        time_index: The horizon's naive datetime index.
        dt_hours: Timestep length in hours.

    Returns:
        The prorating factor in ``(0, 1]``.

    Raises:
        FlexDataError: If ``time_index`` is tz-aware.
    """
    _reject_tz_aware(time_index)
    start = time_index[0]
    horizon_hours = len(time_index) * dt_hours
    month_hours = calendar.monthrange(start.year, start.month)[1] * 24.0
    return min(1.0, horizon_hours / month_hours)


def _resolution_str(dt_hours: float) -> str:
    """Return EECO's resolution string for a timestep length in hours.

    Args:
        dt_hours: Timestep length in hours (e.g. ``1.0`` hourly, ``0.25`` for
            15-minute steps).

    Returns:
        An EECO ``"<binsize>m"`` resolution string (minutes), e.g. ``"60m"``.

    Raises:
        FlexConfigError: If ``dt_hours`` is not a positive whole number of
            minutes.
    """
    minutes = dt_hours * 60.0
    if dt_hours <= 0 or abs(minutes - round(minutes)) > 1e-9:
        raise FlexConfigError(
            f"dt_hours={dt_hours!r} must be a positive whole number of minutes "
            "(e.g. 1.0 for hourly, 0.25 for 15-minute steps).",
            field="dt_hours",
            value=dt_hours,
        )
    return f"{round(minutes)}m"


def _reject_tz_aware(index: pd.DatetimeIndex) -> None:
    """Reject a timezone-aware index (EECO/flex-pse v0 are naive-local only).

    Args:
        index: The datetime index to check.

    Raises:
        FlexDataError: If ``index`` carries timezone information.
    """
    if getattr(index, "tz", None) is not None:
        raise FlexDataError(
            "Timezone-aware datetime indices are not supported: EECO reasons in "
            "naive local wall-clock time and flex-pse v0 follows that convention. "
            f"Strip the timezone (got tz={index.tz}) before passing the index "
            "in, e.g. index.tz_localize(None)."
        )


def _validate_rate_data(frame: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Check a rate_data frame has EECO's required columns.

    Args:
        frame: A candidate EECO ``rate_data`` DataFrame.
        source: Human-readable origin (file path or ``"<DataFrame>"``) for errors.

    Returns:
        ``frame`` unchanged, once validated.

    Raises:
        FlexDataError: If a required column or a charge column is missing.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise FlexDataError(
            f"Tariff {source} is missing required rate_data column(s): "
            f"{', '.join(missing)}. Provide EECO rate_data columns "
            f"({', '.join(_REQUIRED_COLUMNS)}, and a charge column).",
            field=missing[0],
        )
    if not any(c in frame.columns for c in _CHARGE_COLUMNS):
        raise FlexDataError(
            f"Tariff {source} has no charge column; expected one of "
            f"{', '.join(_CHARGE_COLUMNS)}.",
            field="charge",
        )
    return frame


def _tariff_data_from_dict(payload: Any, *, source: str) -> pd.DataFrame:
    """Build a rate_data DataFrame from a dict/list records structure.

    Args:
        payload: Either a mapping with a ``"tariff_data"`` records list, or a
            bare list of row records.
        source: Origin label for error messages.

    Returns:
        The validated rate_data DataFrame.

    Raises:
        FlexDataError: If the structure is not a records list / tariff_data
            mapping.
    """
    if isinstance(payload, dict) and "tariff_data" in payload:
        records = payload["tariff_data"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise FlexDataError(
            f"Tariff {source} must be a list of row records or a mapping with a "
            "'tariff_data' key; got "
            f"{type(payload).__name__}.",
            field="tariff_data",
        )
    return _validate_rate_data(pd.DataFrame.from_records(records), source=source)


# --------------------------------------------------------------------------- #
# Loaders + CSV conversion
# --------------------------------------------------------------------------- #
def load_tariff(source: str | Path | dict | list | pd.DataFrame) -> pd.DataFrame:
    """Return an EECO tariff object (a ``rate_data`` DataFrame) from ``source``.

    ``source`` is what a ``CostingConfig.tariff_source`` string resolves to: a
    JSON or CSV file path, an in-memory dict/records structure, or an already
    built rate_data ``DataFrame`` (passed through after validation). A ``.csv``
    path is routed through :func:`tariff_csv_to_dict` first. EECO 0.2.1 has no
    tariff loader of its own — its cost functions consume a ``rate_data``
    DataFrame directly — so that DataFrame *is* the EECO tariff object here.

    Args:
        source: Tariff source: a JSON/CSV path, a dict/records structure, or a
            rate_data DataFrame.

    Returns:
        The validated EECO ``rate_data`` DataFrame.

    Raises:
        FlexDataError: If the file/structure is malformed or missing required
            columns; the message names the file and the offending field.
    """
    _require_eeco("Loading a tariff")
    if isinstance(source, pd.DataFrame):
        return _validate_rate_data(source.copy(), source="<DataFrame>")
    if isinstance(source, (dict, list)):
        return _tariff_data_from_dict(source, source="<dict>")

    path = Path(source)
    if path.suffix.lower() == ".csv":
        return load_tariff(tariff_csv_to_dict(path))
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FlexDataError(
            f"Could not load tariff file {path}: {exc}. Provide a valid EECO "
            "tariff JSON (a 'tariff_data' records list) or a rate_data CSV.",
            field="tariff_source",
        ) from exc
    return _tariff_data_from_dict(payload, source=str(path))


# The row fields that identify one charge. Two rows agreeing on all of these are
# the same charge: EECO keys its charge arrays on exactly this tuple, and
# add_to_charge_array *sums* colliding keys, so a duplicate silently doubles a
# line item rather than erroring. merge_tariffs rejects them instead.
_CHARGE_IDENTITY_COLUMNS = (
    "utility",
    "type",
    "name",
    "month_start",
    "month_end",
    "weekday_start",
    "weekday_end",
    "hour_start",
    "hour_end",
)


def merge_tariffs(sources: Any) -> pd.DataFrame:
    """Load several tariff sources and merge them into one rate_data frame.

    Use this when a facility's charges live in more than one file — typically an
    electric sheet and a gas sheet. Both forms end at the same place: a single
    validated ``rate_data`` frame whose ``utility`` column is what selects rows
    per leg downstream, so there is no separate tariff object per utility.

    * **Sequence** — ``["electric.json", "gas.json"]`` merges every source as-is.
    * **Mapping** — ``{"electric": "e.json", "gas": "g.json"}`` additionally
      *assigns* each source to a utility and keeps only that utility's rows, so a
      sheet carrying stray rows for another utility cannot leak into the bill.

    Args:
        sources: A sequence of tariff sources, or a mapping of EECO utility name
            (``"electric"``/``"gas"``) to tariff source. Each source is anything
            :func:`load_tariff` accepts.

    Returns:
        The merged, validated EECO ``rate_data`` DataFrame.

    Raises:
        FlexConfigError: If ``sources`` is empty or not a mapping/sequence, if a
            mapping key is not a known EECO utility, if a mapping-assigned source
            contributes no rows for its utility, or if two sources define the
            same charge.
    """
    _require_eeco("Merging tariffs")
    if isinstance(sources, Mapping):
        known = (_ELECTRIC, _GAS)
        frames = []
        for utility, source in sources.items():
            if utility not in known:
                raise FlexConfigError(
                    f"Unknown tariff utility {utility!r}; expected one of "
                    f"{sorted(known)}.",
                    field="tariff_source",
                    value=utility,
                )
            frame = load_tariff(source)
            rows = frame[frame["utility"] == utility]
            if rows.empty:
                raise FlexConfigError(
                    f"The tariff assigned to utility {utility!r} contains no "
                    f"{utility!r} rows (it has "
                    f"{sorted(set(frame['utility']))}). Assign it to the utility "
                    "it actually prices, or pass the sources as a list to merge "
                    "them without per-utility assignment.",
                    field="tariff_source",
                    value=utility,
                )
            frames.append(rows)
    elif isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)):
        frames = [load_tariff(s) for s in sources]
    else:
        raise FlexConfigError(
            "merge_tariffs needs a sequence of tariff sources or a mapping of "
            f"utility -> source; got {type(sources).__name__}.",
            field="tariff_source",
            value=sources,
        )

    if not frames:
        raise FlexConfigError(
            "No tariff sources to merge.", field="tariff_source", value=sources
        )

    merged = pd.concat(frames, ignore_index=True)
    identity = [c for c in _CHARGE_IDENTITY_COLUMNS if c in merged.columns]
    duplicated = merged.duplicated(subset=identity, keep=False)
    if duplicated.any():
        dupes = merged.loc[duplicated, identity].drop_duplicates()
        raise FlexConfigError(
            "Merged tariffs define the same charge more than once, which would "
            "silently double that line item. Remove the duplicate row(s) or "
            "assign each source to one utility. Duplicated charge(s):\n"
            f"{dupes.to_string(index=False)}",
            field="tariff_source",
        )
    return _validate_rate_data(merged, source="<merged tariffs>")


def load_dr_program(source: str | Path | dict | None) -> dict | None:
    """Load a demand-response program (v0: containers-only; None-safe).

    EECO 0.2.1 exposes no DR API, so a DR program is a plain records structure
    (a flex-pse placeholder) loaded into a :class:`DRConfig` container. No DR
    constraints are built from it in v0.

    Args:
        source: A JSON DR-program file path, an in-memory dict, or ``None``.

    Returns:
        The loaded DR program object, or ``None`` when ``source`` is ``None``.

    Raises:
        FlexDataError: If the file cannot be read or parsed; the message names
            the file.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        return source
    path = Path(source)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FlexDataError(
            f"Could not load demand-response file {path}: {exc}. Provide a valid "
            "DR-program JSON file.",
            field="dr.events_source",
        ) from exc


def tariff_csv_to_dict(
    source: str | Path | pd.DataFrame,
    *,
    write_to: str | Path | None = None,
) -> dict:
    """Convert an EECO rate_data CSV (or DataFrame) into the dict load_tariff takes.

    Reads a tariff CSV in EECO's ``rate_data`` column schema (``utility``,
    ``type``, ``name``, ``month_start``/``end``, ``weekday_start``/``end``,
    ``hour_start``/``end``, a charge column, ``basic_charge_limit``…) and returns
    the equivalent ``{"tariff_data": [records]}`` structure that :func:`load_tariff`
    accepts. Does **no** charge math — a pure schema conversion, for authoring or
    importing tariffs in the common CSV exchange format instead of hand-typed
    JSON.

    Args:
        source: A path to a rate_data CSV, or an already-loaded DataFrame with
            the same columns (e.g. read and pre-filtered by the caller). Both go
            through the same column validation.
        write_to: If given, also persist the converted tariff as a JSON tariff
            file at this path (loadable thereafter via :func:`load_tariff`). The
            dict is returned either way.

    Returns:
        A ``{"tariff_data": [records]}`` dict equivalent to the CSV.

    Raises:
        FlexDataError: If the CSV cannot be read or is missing required columns;
            the message names the file and column.
    """
    if isinstance(source, pd.DataFrame):
        frame = source
        label = "<DataFrame>"
    else:
        path = Path(source)
        label = str(path)
        try:
            frame = pd.read_csv(path)
        except (OSError, ValueError) as exc:
            raise FlexDataError(
                f"Could not read tariff CSV {path}: {exc}. Provide a valid EECO "
                "rate_data CSV.",
                field="tariff_source",
            ) from exc

    _validate_rate_data(frame, source=label)
    payload = {"tariff_data": frame.to_dict(orient="records")}

    if write_to is not None:
        Path(write_to).write_text(json.dumps(payload, indent=2) + "\n")

    return payload


# --------------------------------------------------------------------------- #
# Currency basis (from the tariff sheet)
# --------------------------------------------------------------------------- #
def _register_currency(unit_name: str):
    """Return the Pyomo currency unit ``unit_name``, registering it on first use.

    Args:
        unit_name: The currency unit name (e.g. ``"USD"``).

    Returns:
        The Pyomo units object for ``unit_name``.
    """
    if not hasattr(pyo.units, unit_name):
        pyo.units.load_definitions_from_strings([f"{unit_name} = [currency]"])
    return getattr(pyo.units, unit_name)


def currency_units(symbol: str = "USD"):
    """Return the Pyomo currency unit named ``symbol``, registering it on first use.

    Pyomo has no built-in currency dimension, so a currency unit has to be
    declared before it can be used. This is the public way to get one, for
    writing flat prices (``0.12 * currency_units("USD") / pyunits.kWh``) and for
    FlexCosting's currency basis when no tariff supplies one. Tariff-priced
    models get their basis from the sheet instead — see
    :func:`tariff_currency_units`.

    Args:
        symbol: The currency unit name, e.g. ``"USD"``.

    Returns:
        The Pyomo units object for ``symbol``.
    """
    return _register_currency(symbol)


def tariff_currency_units(tariff: pd.DataFrame):
    """Return the Pyomo currency unit for a tariff's currency basis.

    Reads the tariff's charge ``units`` column (e.g. ``"$/kWh"``), takes the
    currency symbol (the numerator before the ``/``), and returns the matching
    Pyomo currency unit — registering it on first use. EECO tariffs are
    dollar-based (``"$"`` → ``USD``). FlexCosting labels every operating-cost
    expression with this unit, so the currency basis is data-driven from the
    tariff sheet rather than hardcoded.

    Args:
        tariff: An EECO ``rate_data`` DataFrame (from :func:`load_tariff`).

    Returns:
        The Pyomo currency units object (e.g. ``USD``).

    Raises:
        FlexDataError: If the tariff has no ``units`` column, or its charges do
            not share exactly one recognized currency symbol.
    """
    if "units" not in tariff.columns:
        raise FlexDataError(
            "Tariff has no 'units' column, so its currency basis cannot be "
            "determined. Add a charge 'units' column (e.g. '$/kWh').",
            field="units",
        )
    symbols = {str(u).split("/")[0].strip() for u in tariff["units"].dropna()}
    if len(symbols) != 1:
        raise FlexDataError(
            "Tariff charges must share exactly one currency basis; found "
            f"{sorted(symbols)} in the 'units' column.",
            field="units",
        )
    symbol = symbols.pop()
    if symbol not in _CURRENCY_SYMBOLS:
        raise FlexDataError(
            f"Unrecognized currency symbol {symbol!r} in the tariff 'units'; "
            f"known symbols: {sorted(_CURRENCY_SYMBOLS)}.",
            field="units",
        )
    return _register_currency(_CURRENCY_SYMBOLS[symbol])


# --------------------------------------------------------------------------- #
# Tariff signal helpers (pandas out, for logic/heuristic constraints)
# --------------------------------------------------------------------------- #
def _energy_price_series(tariff: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Marginal base energy price ($/kWh) at each stamp of ``index``.

    Delegates the per-stamp charge arrays to EECO's ``get_charge_dict`` (the
    source of the price data), then sums the base-tier electric energy charges
    (``basic_charge_limit == 0``) element-wise. Tiered surcharges (limit > 0)
    are excluded so this reflects the price the next kWh is billed at, not the
    high-consumption surcharge.

    Args:
        tariff: An EECO rate_data DataFrame (from :func:`load_tariff`).
        index: A naive ``pd.DatetimeIndex`` of stamps to price.

    Returns:
        A ``pd.Series`` of $/kWh indexed by ``index``.
    """
    _reject_tz_aware(index)
    dt_hours = _index_dt_hours(index)
    start = index[0].to_pydatetime()
    end = (index[0] + len(index) * pd.Timedelta(hours=dt_hours)).to_pydatetime()
    charge_dict = _eeco_costs.get_charge_dict(
        start, end, tariff, resolution=_resolution_str(dt_hours)
    )
    prices = np.zeros(len(index))
    for key, array in charge_dict.items():
        # key form: utility_type_name_start_end_limit
        parts = key.split("_")
        utility, charge_type, limit = parts[0], parts[1], parts[-1]
        if (
            utility == _ELECTRIC
            and charge_type == _eeco_costs.ENERGY
            and int(limit) == 0
        ):
            prices = prices + np.asarray(array)
    return pd.Series(prices, index=index)


def _index_dt_hours(index: pd.DatetimeIndex) -> float:
    """Infer the (uniform) timestep of ``index`` in hours.

    Args:
        index: A ``pd.DatetimeIndex`` with at least two evenly spaced stamps.

    Returns:
        The timestep length in hours.

    Raises:
        FlexDataError: If ``index`` has fewer than two stamps or is not evenly
            spaced.
    """
    if len(index) < 2:
        raise FlexDataError(
            "Tariff signal helpers need an index of at least two stamps to infer "
            "the timestep."
        )
    deltas = index[1:] - index[:-1]
    if deltas.nunique() != 1:
        raise FlexDataError(
            "Tariff signal helpers require an evenly spaced datetime index."
        )
    return deltas[0].total_seconds() / 3600.0


def price_series(tariff: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Base energy price ($/kWh) at each stamp (flex-pse helper over EECO).

    A flex-pse helper: EECO 0.2.1 has no per-stamp price accessor, so this sums
    EECO's own ``get_charge_dict`` electric-energy charge arrays (base tier).

    Args:
        tariff: An EECO rate_data DataFrame.
        index: A naive ``pd.DatetimeIndex`` to price.

    Returns:
        A ``pd.Series`` of $/kWh indexed by ``index``.
    """
    return _energy_price_series(tariff, index)


def is_peak(tariff: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Boolean mask: True at the highest-price stamps (flex-pse helper over EECO).

    A flex-pse helper derived from :func:`price_series`: True where the stamp's
    base energy price equals the horizon maximum.

    Args:
        tariff: An EECO rate_data DataFrame.
        index: A naive ``pd.DatetimeIndex`` to classify.

    Returns:
        A boolean ``pd.Series`` indexed by ``index``.
    """
    prices = _energy_price_series(tariff, index)
    return prices >= prices.max() - 1e-12


def peak_windows(tariff: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Sub-index of the peak (highest-price) stamps (flex-pse helper over EECO).

    Args:
        tariff: An EECO rate_data DataFrame.
        index: A naive ``pd.DatetimeIndex`` to filter.

    Returns:
        The ``pd.DatetimeIndex`` of stamps where :func:`is_peak` is True.
    """
    mask = is_peak(tariff, index)
    return index[mask.to_numpy()]


def price_gradient(tariff: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Δ base energy price between successive stamps (flex-pse helper over EECO).

    A flex-pse helper: the first difference of :func:`price_series`, with the
    leading value set to 0. Nonzero at off-peak↔peak transitions, 0 within a
    flat run.

    Args:
        tariff: An EECO rate_data DataFrame.
        index: A naive ``pd.DatetimeIndex``.

    Returns:
        A ``pd.Series`` of price differences indexed by ``index``.
    """
    prices = _energy_price_series(tariff, index)
    return prices.diff().fillna(0.0)


# --------------------------------------------------------------------------- #
# Demand-response container (v0: containers-only, no constraints)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class DRConfig:
    """A container/config slot for a demand-response program (v0 no-op).

    Holds the loaded DR program object (or ``None``) so the wiring exists;
    building actual DR constraints is post-v0 (architecture §2.4, PLAN §4).

    Attributes:
        program: The loaded DR program object, or ``None``.
    """

    program: object | None = None


def _build_dr(block: pyo.Block, dr_config: "DRConfig | None") -> None:
    """No-op demand-response hook (v0 is containers-only).

    Exists so FlexCosting wiring and later DR work are additive: it only
    stores/verifies the container and builds **no** DR event, curtailment,
    incentive, or capacity constraints.

    Args:
        block: The Pyomo block the operating cost is being built on.
        dr_config: The DR container, or ``None``.
    """
    if dr_config is not None and dr_config.program is not None:
        _log.debug(
            "DR program present but v0 is containers-only; building no DR "
            "constraints on %s.",
            block.name,
        )


# --------------------------------------------------------------------------- #
# In-objective Pyomo bridge
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class OperatingCostHandles:
    """Stable flex-pse handles for EECO's in-objective cost expressions.

    Attributes:
        energy_cost: Aggregate electric (or gas) **energy** charge Expression.
        demand_charge: Aggregate **demand** charge Expression, already prorated.
        customer_charge: The tariff's fixed (customer) charge for the horizon, as
            billed — i.e. already prorated. A constant, not a decision.
        total_operating_cost: Scalar horizon-total cost — the CONVEX-RELAXED
            proxy the scheduler minimizes, **not** the reported bill (use
            :func:`evaluate_cost` post-solve for that).
        scale_factor: The prorating factor applied to the monthly-assessed
            charges (``1.0`` when not prorating or on a full-month horizon).
        eeco_block: The raw EECO itemized-cost structure, for debugging only.
    """

    energy_cost: Any
    demand_charge: Any
    customer_charge: Any
    total_operating_cost: Any
    scale_factor: float
    eeco_block: Any


def _daily_assessed_demand_names(tariff: pd.DataFrame) -> set[str]:
    """Names of demand charges the tariff assesses per day rather than per month.

    A daily-assessed demand charge is already billed at the horizon's own
    granularity, so it must never be prorated. EECO treats a missing/absent
    ``assessed`` value as ``"monthly"``.

    Args:
        tariff: An EECO rate_data DataFrame.

    Returns:
        The charge names, in the dashed form EECO builds its keys from.
    """
    if _ASSESSED not in tariff.columns or "type" not in tariff.columns:
        return set()
    rows = tariff[
        (tariff["type"] == _DEMAND)
        & (tariff[_ASSESSED].astype(str).str.lower() == "daily")
    ]
    return {str(name).replace("_", "-") for name in rows["name"]}


def _prorate_charge_dict(charge_dict: dict, tariff: pd.DataFrame, scale: float) -> dict:
    """Scale monthly-assessed demand and fixed charge *rates* to the horizon.

    Prorating is applied to the charge rates rather than to the computed cost.
    Both are billed linearly in their rate (``$/kW × kW``, and a flat ``$/month``),
    so scaling the rate scales that line item exactly — and doing it here leaves
    the energy charges, and therefore EECO's tiered-surcharge arithmetic,
    completely untouched. This is the same technique EECO applies internally in
    ``get_charge_df``; flex-pse needs its own copy because ``get_charge_dict`` (the
    Pyomo-compatible path) exposes no such option.

    Two EECO behaviors make this the right level to intervene at:
    ``calculate_cost`` adds the customer charge whole, with no scale factor
    available at all; and its ``demand_scale_factor`` is suppressed for charges
    spanning ``<= 1`` day, which — because EECO clips charge-key dates to the
    horizon — silently includes *every* monthly charge on a one-day horizon, the
    very case prorating exists for. Deciding from the tariff's ``assessed``
    column instead is correct at any horizon length.

    Args:
        charge_dict: EECO's charge-array dictionary, modified in place.
        tariff: The rate_data frame the dictionary came from.
        scale: The prorating factor from :func:`monthly_scale_factor`.

    Returns:
        ``charge_dict``, with monthly-assessed arrays scaled.
    """
    if scale >= 1.0:
        return charge_dict
    daily = _daily_assessed_demand_names(tariff)
    for key, array in charge_dict.items():
        # Keys are "<utility>_<type>_<name>_<start>_<end>_<limit>"; the name never
        # contains an underscore (EECO dashes them) so this split is unambiguous.
        _utility, charge_type, name = key.split("_")[:3]
        if charge_type == _CUSTOMER or (charge_type == _DEMAND and name not in daily):
            charge_dict[key] = array * scale
    return charge_dict


def _charge_dict(
    tariff: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    dt_hours: float,
    *,
    prorate: bool = False,
) -> dict:
    """Build EECO's charge dictionary for a horizon (delegates to EECO).

    Args:
        tariff: An EECO rate_data DataFrame.
        time_index: The horizon's naive datetime index.
        dt_hours: Timestep length in hours.
        prorate: Scale monthly-assessed demand and fixed charges to the horizon
            length (see :func:`_prorate_charge_dict`).

    Returns:
        EECO's charge-array dictionary.
    """
    _require_eeco("Pricing against a tariff")
    _reject_tz_aware(time_index)
    start = time_index[0].to_pydatetime()
    end = (
        time_index[0] + len(time_index) * pd.Timedelta(hours=dt_hours)
    ).to_pydatetime()
    charge_dict = _eeco_costs.get_charge_dict(
        start, end, tariff, resolution=_resolution_str(dt_hours)
    )
    if prorate:
        scale = monthly_scale_factor(time_index, dt_hours)
        charge_dict = _prorate_charge_dict(charge_dict, tariff, scale)
    return charge_dict


def _add_utility_cost(
    *,
    block: pyo.Block,
    power,
    time_index: pd.DatetimeIndex,
    dt_hours: float,
    tariff: pd.DataFrame,
    utility: str,
    dr_config: "DRConfig | None",
    prorate: bool = True,
) -> OperatingCostHandles:
    """Ask EECO to build the convex-relaxed in-objective cost for one utility.

    Shared implementation of :func:`add_operating_cost` (electric) and
    :func:`add_fuel_cost` (fuel). EECO owns all cost math; this only ensures the
    ``block.t`` index set EECO's Pyomo helpers require, calls
    ``eeco.costs.calculate_itemized_cost`` on ``block``, and renames the outputs.

    Args:
        block: The Pyomo block to build cost expressions on.
        power: Time-indexed Var or Expression carrying **no units** — a bare kW
            (electric) / m³/hr (gas) magnitude, indexed ``0..N-1`` to align with
            ``time_index`` order. EECO constrains its own dimensionless Vars to
            this series, so a units-carrying input makes those constraints
            inconsistent.
        time_index: Naive datetime index aligning to ``power``'s order.
        dt_hours: Timestep length in hours (EECO does the kW→kWh conversion).
        tariff: An EECO rate_data DataFrame, covering ``utility``.
        utility: ``"electric"`` or ``"gas"``.
        dr_config: DR container (v0: stored via the no-op hook only).
        prorate: Prorate monthly-assessed demand and fixed charges to the horizon
            (see :func:`monthly_scale_factor`).

    Returns:
        The renamed :class:`OperatingCostHandles`.

    Raises:
        FlexConfigError: If EECO produced a nonlinear (``max()``) demand term,
            breaking the LP/relaxable character.
    """
    n = len(time_index)
    charge_dict = _charge_dict(tariff, time_index, dt_hours, prorate=prorate)
    scale = monthly_scale_factor(time_index, dt_hours) if prorate else 1.0

    # EECO's Pyomo helpers (eeco.utils) build indexed vars/constraints against
    # `model.t`; provide it as the block's 0..N-1 step set if absent.
    if not hasattr(block, "t"):
        block.t = pyo.RangeSet(0, n - 1)

    _build_dr(block, dr_config)

    itemized, _ = _eeco_costs.calculate_itemized_cost(
        charge_dict,
        {utility: power},
        resolution=_resolution_str(dt_hours),
        desired_utility=utility,
        model=block,
        **_eeco_consumption_units(),
    )
    util_costs = itemized[utility]
    handles = OperatingCostHandles(
        energy_cost=util_costs["energy"],
        demand_charge=util_costs["demand"],
        customer_charge=util_costs.get("customer", 0.0),
        total_operating_cost=util_costs["total"],
        scale_factor=scale,
        eeco_block=itemized,
    )
    _assert_linear_demand(handles.demand_charge)
    return handles


def _assert_linear_demand(demand_charge) -> None:
    """Fail loud if EECO emitted a nonlinear demand term (architecture §3.6).

    The demand charge must stay an epigraph (linear), not a ``max()``: a
    nonlinear objective silently breaks the LP/relaxable character the scheduler
    depends on. EECO's Pyomo path builds ``>=`` epigraph vars, so a
    higher-than-quadratic degree here signals an incompatibility to fail.

    Args:
        demand_charge: The demand-charge Expression/Var from EECO.

    Raises:
        FlexConfigError: If the demand term is non-polynomial.
    """
    if hasattr(demand_charge, "polynomial_degree"):
        degree = demand_charge.polynomial_degree()
        if degree is None:
            raise FlexConfigError(
                "EECO produced a nonlinear demand-charge term (a max() rather "
                "than an epigraph), which breaks the LP/relaxable character the "
                "scheduler needs. Choose EECO options that express demand charges "
                "as epigraphs, or file an EECO issue.",
                field="demand_charge",
            )


def add_electricity_cost(
    *,
    block: pyo.Block,
    electrical_power,
    time_index: pd.DatetimeIndex,
    dt_hours: float,
    tariff: pd.DataFrame,
    dr_config: "DRConfig | None" = None,
    prorate: bool = True,
) -> OperatingCostHandles:
    """Build EECO's convex-relaxed in-objective **electricity** cost on ``block``.

    The single-utility electric builder. :func:`add_operating_cost` is the
    facility-level umbrella that wraps this and :func:`add_fuel_cost` onto one
    opex block; call this directly only when you want the electric leg alone.

    Hands EECO the kW magnitudes, tariff, and timestep; EECO owns the math (energy
    cost, demand-charge epigraphs, kWh conversion). The returned
    ``total_operating_cost`` is a RELAXED proxy for the objective, **not** the
    reported bill — use :func:`evaluate_cost` post-solve for that. DR is
    containers-only in v0: ``dr_config`` is accepted and stored but builds NO DR
    constraints.

    Args:
        block: The Pyomo block to build cost expressions on.
        electrical_power: Time-indexed aggregate-load Var/Expression carrying **no
            units** — a bare kW magnitude — indexed ``0..N-1`` to align with
            ``time_index`` order.
        time_index: Naive ``pd.DatetimeIndex`` aligning to ``electrical_power``.
        dt_hours: Timestep length in hours; passed to EECO once for kW→kWh.
        tariff: An EECO rate_data DataFrame (electric utility).
        dr_config: Optional DR container (v0: no constraints built).
        prorate: Prorate monthly demand and fixed charges to the horizon length
            (see :func:`monthly_scale_factor`).

    Returns:
        The :class:`OperatingCostHandles` for the electric utility.

    Raises:
        FlexConfigError: If EECO produced a nonlinear demand term.
        FlexDataError: If ``time_index`` is timezone-aware.
    """
    return _add_utility_cost(
        block=block,
        power=electrical_power,
        time_index=time_index,
        dt_hours=dt_hours,
        tariff=tariff,
        utility=_ELECTRIC,
        dr_config=dr_config,
        prorate=prorate,
    )


def add_fuel_cost(
    *,
    block: pyo.Block,
    fuel_power,
    time_index: pd.DatetimeIndex,
    dt_hours: float,
    tariff: pd.DataFrame,
    fuel_type: str = "gas",
    dr_config: "DRConfig | None" = None,
    prorate: bool = True,
) -> OperatingCostHandles:
    """Build EECO's convex-relaxed in-objective fuel cost on ``block``.

    Mirrors :func:`add_electricity_cost` for a fuel utility: same rules (EECO
    owns the math, no cost math here, DR is a no-op container in v0), same
    :class:`OperatingCostHandles` shape (fuel-flavored). ``tariff`` is the same
    rate_data object; EECO selects the underlying utility's rows.

    Args:
        block: The Pyomo block to build cost expressions on.
        fuel_power: Time-indexed fuel-usage Var/Expression carrying **no units** —
            a bare volumetric-flow magnitude in m³/hr — indexed ``0..N-1``.
        time_index: Naive ``pd.DatetimeIndex`` aligning to ``fuel_power``.
        dt_hours: Timestep length in hours; passed to EECO once.
        tariff: An EECO rate_data DataFrame (must carry the utility's rows).
        fuel_type: The fuel's EECO utility. ``"gas"`` (the default) is the only
            value EECO 0.2.1 supports; every registered fuel (natural gas,
            biogas, ...) bills through it today.
        dr_config: Optional DR container (v0: no constraints built).
        prorate: Prorate monthly demand and fixed charges to the horizon length
            (see :func:`monthly_scale_factor`).

    Returns:
        The :class:`OperatingCostHandles` for the fuel utility.

    Raises:
        FlexConfigError: If ``fuel_type`` is not a supported EECO utility, or
            EECO produced a nonlinear demand term.
        FlexDataError: If ``time_index`` is timezone-aware.
    """
    if fuel_type not in _FUEL_UTILITY:
        raise FlexConfigError(
            f"Unsupported fuel_type={fuel_type!r}; EECO 0.2.1 supports "
            f"{sorted(_FUEL_UTILITY)}.",
            field="fuel_type",
            value=fuel_type,
        )
    return _add_utility_cost(
        block=block,
        power=fuel_power,
        time_index=time_index,
        dt_hours=dt_hours,
        tariff=tariff,
        utility=_FUEL_UTILITY[fuel_type],
        dr_config=dr_config,
        prorate=prorate,
    )


def add_operating_cost(
    *,
    block: pyo.Block,
    time_index: pd.DatetimeIndex,
    dt_hours: float,
    tariff: pd.DataFrame,
    electrical_power=None,
    fuel_power=None,
    dr_config: "DRConfig | None" = None,
    prorate: bool = True,
) -> OperatingCostHandles:
    """Build the facility's whole in-objective operating cost — electric **and** fuel.

    The umbrella over :func:`add_electricity_cost` and :func:`add_fuel_cost`: it
    builds each present utility's convex-relaxed cost on the **same** opex
    ``block`` (EECO namespaces its components by utility, so the two never
    collide) and returns one :class:`OperatingCostHandles` whose fields are the
    per-utility sums — the single ``total_operating_cost`` the scheduler
    minimizes. Still a RELAXED proxy, not the reported bill (use
    :func:`evaluate_cost`/:func:`evaluate_fuel_cost` post-solve).

    The facility-level consumption defaults to the standard series registered on
    ``block`` — ``block.power_electrical`` (the canonical
    :data:`~flexcore.nomenclature.POWER_ELECTRICAL` name) and ``block.fuel_usage``
    — so a caller need not re-declare them each use; pass
    ``electrical_power``/``fuel_power`` to override (e.g. a toy model or a
    pre-aggregated series). A utility whose series is neither passed nor present
    on the block is simply omitted; at least one must resolve. The fuel leg
    bills through EECO's ``"gas"`` utility (:func:`add_fuel_cost`'s only
    supported ``fuel_type`` today).

    Args:
        block: The Pyomo opex block to build both utilities' cost on.
        time_index: Naive ``pd.DatetimeIndex`` aligning to the power series.
        dt_hours: Timestep length in hours; passed to EECO once for kW→kWh.
        tariff: An EECO rate_data DataFrame (its electric and/or gas rows).
        electrical_power: Time-indexed series of bare kW magnitudes (no units);
            defaults to ``block.power_electrical`` if present, or whatever pyo
            object is provided, else the electric leg is skipped.
        fuel_power: Time-indexed fuel-usage series of bare m³/hr magnitudes (no
            units); defaults to ``block.fuel_usage`` if present, else the fuel leg
            is skipped.
        dr_config: Optional DR container (v0: no constraints built).

    Returns:
        A combined :class:`OperatingCostHandles`: ``energy_cost``,
        ``demand_charge`` and ``total_operating_cost`` are the sums across the
        built utilities, and ``eeco_block`` maps each built utility name
        (``"electric"``/``"gas"``) to its raw EECO itemized structure.

    Raises:
        FlexConfigError: If neither an electric nor a fuel consumption series is
            passed or found on ``block``.
        FlexDataError: If ``time_index`` is timezone-aware.
    """
    if electrical_power is None:
        electrical_power = getattr(block, nm.POWER_ELECTRICAL, None)
    if fuel_power is None:
        fuel_power = getattr(block, nm.FUEL_USAGE, None)
    if electrical_power is None and fuel_power is None:
        raise FlexConfigError(
            "add_operating_cost found no utility consumption to cost: pass "
            "electrical_power and/or fuel_power, or register them on the block "
            f"as '{nm.POWER_ELECTRICAL}' / '{nm.FUEL_USAGE}'.",
            field="electrical_power",
        )

    per_utility: dict[str, OperatingCostHandles] = {}
    if electrical_power is not None:
        per_utility[_ELECTRIC] = add_electricity_cost(
            block=block,
            electrical_power=electrical_power,
            time_index=time_index,
            dt_hours=dt_hours,
            tariff=tariff,
            dr_config=dr_config,
            prorate=prorate,
        )
    if fuel_power is not None:
        per_utility[_GAS] = add_fuel_cost(
            block=block,
            fuel_power=fuel_power,
            time_index=time_index,
            dt_hours=dt_hours,
            tariff=tariff,
            dr_config=dr_config,
            prorate=prorate,
        )

    legs = list(per_utility.values())
    return OperatingCostHandles(
        energy_cost=sum(leg.energy_cost for leg in legs),
        demand_charge=sum(leg.demand_charge for leg in legs),
        customer_charge=sum(leg.customer_charge for leg in legs),
        total_operating_cost=sum(leg.total_operating_cost for leg in legs),
        scale_factor=legs[0].scale_factor,
        eeco_block={util: leg.eeco_block for util, leg in per_utility.items()},
    )


# --------------------------------------------------------------------------- #
# Post-optimization evaluator (the reported bill; §2.4)
# --------------------------------------------------------------------------- #
def _evaluation_index(
    n: int, dt_hours: float, time_index: "pd.DatetimeIndex | None"
) -> pd.DatetimeIndex:
    """Return the calendar index to align a realized array against EECO windows.

    EECO's charge windows are keyed on ``month``/``weekday``/``hour``, so a bare
    realized array must be paired with a calendar to bill it correctly. When
    ``time_index`` is given it is used; otherwise a naive epoch-based index is
    fabricated — valid only for a calendar-independent (flat) tariff.

    Args:
        n: Number of timesteps in the realized array.
        dt_hours: Timestep length in hours.
        time_index: The realized array's datetime index, or ``None``.

    Returns:
        A naive ``pd.DatetimeIndex`` of length ``n``.

    Raises:
        FlexDataError: If ``time_index`` is tz-aware or its length != ``n``.
    """
    if time_index is None:
        return pd.date_range("1970-01-01", periods=n, freq=pd.Timedelta(hours=dt_hours))
    _reject_tz_aware(time_index)
    if len(time_index) != n:
        raise FlexDataError(
            f"time_index length ({len(time_index)}) must match the realized "
            f"array length ({n}).",
            field="time_index",
        )
    return time_index


def _itemized_cost(
    usage: np.ndarray,
    tariff: pd.DataFrame,
    dt_hours: float,
    *,
    utility: str,
    by_charge_key: bool = False,
    time_index: "pd.DatetimeIndex | None" = None,
    prorate: bool = True,
) -> dict:
    """Evaluate EECO's itemized cost on a fixed, realized usage array.

    Args:
        usage: Realized usage per timestep (kW electric / m³/hr gas).
        tariff: An EECO rate_data DataFrame.
        dt_hours: Timestep length in hours.
        utility: ``"electric"`` or ``"gas"``.
        by_charge_key: Pass through to EECO to itemize by individual charge key.
        time_index: Calendar index aligning ``usage`` to the tariff's windows.
        prorate: Prorate monthly demand and fixed charges to the horizon. Must
            match what the in-objective leg used, or the reported bill will not
            reconcile with the objective.

    Returns:
        EECO's per-utility itemized-cost dict (``itemized[utility]``), plus a
        ``"scale_factor"`` entry recording the prorating factor that was applied.
    """
    array = np.asarray(usage, dtype=float)
    index = _evaluation_index(len(array), dt_hours, time_index)
    charge_dict = _charge_dict(tariff, index, dt_hours, prorate=prorate)
    itemized, _ = _eeco_costs.calculate_itemized_cost(
        charge_dict,
        {utility: array},
        resolution=_resolution_str(dt_hours),
        desired_utility=utility,
        by_charge_key=by_charge_key,
        **_eeco_consumption_units(),
    )
    util_costs = itemized[utility]
    util_costs["scale_factor"] = (
        monthly_scale_factor(index, dt_hours) if prorate else 1.0
    )
    return util_costs


def _itemized_electricity_cost(
    aggregate_power_kw: np.ndarray,
    tariff: pd.DataFrame,
    dt_hours: float,
    time_index: "pd.DatetimeIndex | None" = None,
) -> dict:
    """By-charge-key itemized electricity cost on a realized load (tests/reporting).

    Args:
        aggregate_power_kw: Realized aggregate power per timestep, kW.
        tariff: An EECO rate_data DataFrame.
        dt_hours: Timestep length in hours.
        time_index: Calendar index aligning the load to the tariff's windows.

    Returns:
        EECO's ``itemized["electric"]`` dict, itemized by individual charge key.
    """
    return _itemized_cost(
        aggregate_power_kw,
        tariff,
        dt_hours,
        utility=_ELECTRIC,
        by_charge_key=True,
        time_index=time_index,
    )


def evaluate_cost(
    aggregate_power_kw: np.ndarray,
    tariff: pd.DataFrame,
    dt_hours: float,
    *,
    dr_config: "DRConfig | None" = None,
    time_index: "pd.DatetimeIndex | None" = None,
    prorate: bool = True,
) -> float:
    """Compute the TRUE (de-relaxed) electricity cost on a fixed realized load.

    This is the user-facing bill (§6 reporting rule): once the dispatch is
    fixed the pricing non-convexity is harmless, so it is an exact post-hoc EECO
    evaluation, not a relaxation. All ``eeco.*`` calls route through this module
    (the sole import point).

    Args:
        aggregate_power_kw: Realized aggregate power per timestep, kW.
        tariff: An EECO rate_data DataFrame.
        dt_hours: Timestep length in hours; passed to EECO once for kW→kWh.
        dr_config: Ignored in v0 (DR is containers-only).
        time_index: The load's naive datetime index, aligning it to the tariff's
            month/weekday/hour windows. Omit only for a calendar-independent
            (flat) tariff. EECO's charge windows are calendar-dependent, so a
            bare array is insufficient to reproduce a real (e.g. July) bill.
        prorate: Prorate monthly demand and fixed charges to the horizon length.
            Leave at the default so the reported bill matches the objective; pass
            ``False`` only to reproduce a full un-prorated monthly bill.

    Returns:
        The horizon-total electricity cost in dollars.
    """
    return float(
        _itemized_cost(
            aggregate_power_kw,
            tariff,
            dt_hours,
            utility=_ELECTRIC,
            time_index=time_index,
            prorate=prorate,
        )["total"]
    )


def evaluate_fuel_cost(
    aggregate_fuel_usage: np.ndarray,
    tariff: pd.DataFrame,
    dt_hours: float,
    *,
    fuel_type: str = "gas",
    dr_config: "DRConfig | None" = None,
    time_index: "pd.DatetimeIndex | None" = None,
    prorate: bool = True,
) -> float:
    """Compute the TRUE (de-relaxed) fuel cost on a fixed realized usage array.

    Mirrors :func:`evaluate_cost` for a fuel utility.

    Args:
        aggregate_fuel_usage: Realized fuel usage per timestep, a volumetric flow
            in m³/hr.
        tariff: An EECO rate_data DataFrame (the utility's rows).
        dt_hours: Timestep length in hours; passed to EECO once.
        fuel_type: The fuel's EECO utility; see :func:`add_fuel_cost`.
        dr_config: Ignored in v0 (DR is containers-only).
        time_index: The usage array's naive datetime index (see
            :func:`evaluate_cost`); omit only for a flat tariff.
        prorate: Prorate monthly demand and fixed charges to the horizon length
            (see :func:`evaluate_cost`).

    Returns:
        The horizon-total fuel cost in dollars.

    Raises:
        FlexConfigError: If ``fuel_type`` is not a supported EECO utility.
    """
    if fuel_type not in _FUEL_UTILITY:
        raise FlexConfigError(
            f"Unsupported fuel_type={fuel_type!r}; EECO 0.2.1 supports "
            f"{sorted(_FUEL_UTILITY)}.",
            field="fuel_type",
            value=fuel_type,
        )
    return float(
        _itemized_cost(
            aggregate_fuel_usage,
            tariff,
            dt_hours,
            utility=_FUEL_UTILITY[fuel_type],
            time_index=time_index,
            prorate=prorate,
        )["total"]
    )
