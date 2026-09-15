"""ArimaRegressor: time-series ARIMA regressor with exogenous inputs.

Fits a univariate ARIMA (or SARIMAX with seasonal terms and exogenous
regressors) by directly minimizing the mean-equation residual using
``scipy.optimize.least_squares``, and reduces the result to the standard
:class:`~flexparameterize.regression.base.FitResult` / ``SurrogateSpec``
shape, so every downstream consumer (provenance logging,
``emit_model_config``, ``apply_to_model``) works without change.

Only the ``[parameterize]`` extra is required: ``scipy`` is a core
runtime dependency and ``statsforecast`` is used only when ``auto=True``
for order selection; the final parameter fit uses the direct
OLS / Levenberg-Marquardt backend that minimizes the exact residual the
Pyomo surrogate implements.

**Restriction**: ``d`` may be ``0`` or ``1``; seasonal differencing
(``D``) must be ``0``, and seasonal AR/MA terms (``P>0`` or ``Q>0``) are not
supported by this backend at all.  The Pyomo surrogate implements the mean
ARIMA equation in closed form, which supports at most a single order of
differencing.  Models with ``d>1``, ``D>0``, or a non-trivial seasonal AR/MA
order will raise ``FlexConfigError``.

Typical usage::

    import pandas as pd
    from flexparameterize.regression.arima import ArimaRegressor

    df = pd.read_csv(
        "imputed_bio_gas_generation.csv", parse_dates=["timestamp"]
    ).set_index("timestamp")
    X = df[["feed_volume_kg", "TS_pct"]]
    y = df[["biogas_m3_hour"]]

    regressor = ArimaRegressor(order=(1, 0, 1)).fit(
        X, y,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )
    result = regressor.to_fit_result()
    spec  = regressor.to_surrogate_spec()

    # Or let statsforecast pick the best order (no differencing), then
    # refit the winning order with scipy for Pyomo compatibility:
    auto_regressor = ArimaRegressor(auto=True, max_p=3, max_q=3).fit(
        X, y,
        input_units={"feed_volume_kg": "kg", "TS_pct": "%"},
        output_units="m^3/hr",
    )

Attributes:
    model: The fitted :class:`_DirectResults` (or ``None`` before
        :meth:`fit`).
    n_samples: Number of rows the fit used, after dropping nulls.
    metrics: ``{"aic": ..., "rmse": ...}`` of the fitted model against ``y``.
    data_window: ``(first, last)`` index value of the rows used.
    exogenous_variables: Column names of the fitted exogenous inputs.
    output_variable: Column name of the fitted output.
    input_units: Units of every fitted exogenous column, keyed by column
        name.  Set by :meth:`fit`.
    output_units: Units of the fitted output column.  Set by :meth:`fit`.
    order: ``(p, d, q)`` order used (or ``None`` when ``auto`` was used).
    seasonal_order: ``(P, D, Q, m)`` seasonal order, or ``None``.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexparameterize.regression.base import FitResult


class ArimaRegressor:
    """Fit a univariate ARIMA / SARIMAX model using direct OLS/NLS and expose
    the shared fit protocol.

    Either supply an explicit ``order`` (and optional ``seasonal_order``), or
    set ``auto=True`` to delegate order selection to statsforecast
    ``AutoARIMA``, then refit the winning order with the direct scipy
    backend.

    **Restriction**: ``d`` may be ``0`` or ``1`` (the Pyomo surrogate's mean
    ARIMA equation supports at most a single order of differencing).
    Seasonal differencing (``D``) must be ``0``, and seasonal AR/MA terms
    (``P>0`` or ``Q>0``) are not supported by the direct-fit backend at all —
    only a plain, non-seasonal ARIMA (``seasonal_order=None`` or the trivial
    ``(0, 0, 0, m)``) can be fit.

    **Fitting backend**: All final parameter estimation uses
    ``scipy.optimize.least_squares`` (Levenberg-Marquardt) to minimise the
    mean-equation residual directly.  This is the exact objective the Pyomo
    surrogate implements, so fitted parameters reproduce the surrogate
    one-to-one.  Pure AR(p) models use closed-form OLS.

    **AR persistence bound**: AR coefficients are constrained to
    ``[-max_ar_persistence, max_ar_persistence]`` (default 0.85) during the
    fit itself, not merely checked afterward: the Pyomo surrogate implements
    the mean ARIMA equation as a difference equation, and coefficients near
    the unit root (\\|ar\\| >= 1) make the resulting optimisation problem
    numerically unstable regardless of the backend used to estimate them.
    Set ``max_ar_persistence=None`` to fit unconstrained.

    Args:
        order: ``(p, d, q)`` ARIMA order; required when ``auto`` is ``False``.
            Must satisfy ``d in (0, 1)``.
        seasonal_order: ``(P, D, Q, m)`` seasonal order; pass ``None`` (the
            default) for a plain (non-seasonal) ARIMA.  If provided, must
            satisfy ``D == 0`` and ``P == Q == 0`` — the direct-fit backend
            does not support seasonal AR/MA terms, so only the trivial
            ``(0, 0, 0, m)`` (equivalent to ``None``) is accepted.
        include_mean: Whether to include the model's deterministic term.
            This is a level intercept for ``d=0`` and constant drift for
            ``d=1``. Default ``True``.
        include_drift: Whether to include a drift term (constant in the
            differenced series). Default ``False``. For ``d=1`` this is an
            explicit alias for the default ``include_mean`` deterministic
            term; setting it with ``d=0`` raises ``FlexConfigError``.
        auto: If ``True``, run ``statsforecast.models.AutoARIMA`` to
            discover the best ``(p, d, q)`` and ``(P, D, Q, m)``, then
            refit that order with the direct scipy backend for Pyomo
            compatibility.  AutoARIMA may select ``d=0`` or ``d=1``;
            ``D`` is always forced to 0.  Use ``auto_kwargs`` to restrict
            the search (e.g. ``max_d=1``).
        auto_kwargs: Extra keyword arguments forwarded to ``AutoARIMA``
            (e.g. ``max_p``, ``max_q``, ``max_P``, ``season_length``).
            ``D`` defaults to 0; pass ``max_d=1`` to allow differencing.
        max_ar_persistence: Bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]`` during the fit
            itself (via bounded least squares). Default ``0.85``. Set to
            ``None`` to fit unconstrained.
        stationary: If ``True``, force ``stationary=True`` in
            ``statsforecast.models.AutoARIMA``, which restricts the
            search to models with stationary AR coefficients.  Default
            ``False``.  This is an additional safety net for the Pyomo
            surrogate; it does not replace the ``max_ar_persistence`` check.

    Raises:
        FlexConfigError: If ``d > 1``, ``D > 0``, or a seasonal AR/MA order
            (``P > 0`` or ``Q > 0``) is requested, if ``include_drift=True``
            is requested, or if any fitted AR
            coefficient exceeds ``max_ar_persistence``.
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        include_mean: bool = True,
        include_drift: bool = False,
        auto: bool = False,
        max_ar_persistence: float | None = 0.85,
        stationary: bool = False,
        **auto_kwargs: object,
    ) -> None:
        if include_drift and (order is None or order[1] != 1):
            raise FlexConfigError(
                "ArimaRegressor only supports include_drift=True when d=1. "
                f"Got include_drift=True with order={order}. "
                "Set include_drift=False or use order=(p, 1, q).",
                field="include_drift",
                value=True,
            )
        if order is not None and order[1] != 0:
            if order[1] != 1:
                raise FlexConfigError(
                    f"ArimaRegressor only supports d=0 or d=1. "
                    f"Got order={order} with d={order[1]}. "
                    f"Use order=(p, 0, q) or order=(p, 1, q).",
                    field="order",
                    value=order,
                )
        if seasonal_order is not None and seasonal_order[1] != 0:
            raise FlexConfigError(
                f"ArimaRegressor only supports non-differenced seasonal models "
                f"(D=0). Got seasonal_order={seasonal_order} with "
                f"D={seasonal_order[1]}. Use seasonal_order=(P, 0, Q, m) instead.",
                field="seasonal_order",
                value=seasonal_order,
            )
        if seasonal_order is not None and (
            seasonal_order[0] != 0 or seasonal_order[2] != 0
        ):
            raise FlexConfigError(
                f"ArimaRegressor's direct-fit backend does not support seasonal "
                f"AR/MA terms (P>0 or Q>0). Got seasonal_order={seasonal_order} "
                f"with P={seasonal_order[0]}, Q={seasonal_order[2]}. Use "
                f"seasonal_order=(0, 0, 0, m) or None instead.",
                field="seasonal_order",
                value=seasonal_order,
            )
        if max_ar_persistence is not None and (
            not isinstance(max_ar_persistence, (int, float))
            or max_ar_persistence <= 0
            or max_ar_persistence > 1
        ):
            raise FlexConfigError(
                f"ArimaRegressor max_ar_persistence must be a number in "
                f"(0, 1] or None, got {max_ar_persistence!r}.",
                field="max_ar_persistence",
                value=max_ar_persistence,
            )

        self._order = order
        self._seasonal_order = seasonal_order
        self._include_mean = include_mean
        self._include_drift = include_drift
        self._auto = auto
        self._auto_kwargs: dict[str, object] = auto_kwargs
        self._max_ar_persistence = max_ar_persistence
        self._stationary = stationary

        self.model = None
        self.n_samples: int = 0
        self.metrics: dict[str, float] = {}
        self.data_window: tuple = ()
        self.exogenous_variables: list[str] = []
        self.output_variable: str = ""
        self.input_units: dict[str, str] = {}
        self.output_units: str = ""
        self._fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        input_units: dict[str, str] | None = None,
        output_units: str | None = None,
    ) -> ArimaRegressor:
        """Fit an ARIMA model to ``y`` with optional exogenous regressors ``X``.

        Uses ``scipy.optimize.least_squares`` to minimize the mean-equation
        residual directly, so the fitted parameters exactly match the Pyomo
        surrogate.  When ``auto=True``, statsforecast ``AutoARIMA`` is used
        for order selection only; the winning order is then refitted with
        scipy.  Pure AR(p) models use closed-form OLS.

        Rows with any null value across ``X``/``y`` are dropped before fitting.

        **Restriction**: ``d`` may be ``0`` or ``1``; ``D`` must be ``0``, and
        seasonal AR/MA terms (``P>0`` or ``Q>0``) are not supported.

        Args:
            X: Zero or more exogenous-input columns. Pass an empty
                ``DataFrame`` for a pure ARIMA fit.
            y: One output column (a one-column ``DataFrame`` or a
                ``Series``).
            input_units: Units of every fitted exogenous column, keyed by its
                column name.  Defaults to ``{}``.  Recorded for
                :meth:`to_surrogate_spec`.
            output_units: Units of the fitted output column.  Defaults to
                ``""``.  Recorded for :meth:`to_surrogate_spec`.

        Returns:
            ``self``, fitted.

        Raises:
            FlexConfigError: If scipy is not installed, ``auto``
                is ``False`` and no ``order`` was supplied, or
                ``input_units`` is missing an entry for one of ``X``'s
                columns.
            FlexDataError: If ``y`` does not hold exactly one column, or
                fewer than ``order[2] + 1`` rows survive dropping nulls.
        """
        if not self._auto and self._order is None:
            raise FlexConfigError(
                "ArimaRegressor.fit requires either `order=(p,d,q)` or "
                "`auto=True`. Pass both or set `auto=True`."
            )

        try:
            import scipy.optimize  # noqa: F401
        except ImportError as exc:
            raise FlexConfigError(
                "ArimaRegressor requires scipy. Install it with "
                "`pip install 'flex-pse[parameterize]'`."
            ) from exc

        if input_units is None:
            input_units = {}
        if output_units is None:
            output_units = ""

        missing = [name for name in X.columns if name not in input_units]
        if missing:
            raise FlexConfigError(
                f"fit is missing input_units for {missing}; every input "
                f"column ({list(X.columns)}) needs an entry.",
                field="input_units",
                value=missing,
            )

        self.input_units = dict(input_units)
        self.output_units = output_units

        output = _single_column(y, "y")
        exogenous = _exog_columns(X)
        paired = pd.concat([output.rename("__y__"), exogenous], axis=1).dropna()

        if paired.empty:
            raise FlexDataError(
                "ArimaRegressor has no usable rows after dropping nulls "
                f"(of {len(output)} rows). Supply non-null data.",
                field="y",
            )

        if self._auto:
            min_rows = 1
            detail = ""
        else:
            p, d, q = self._order or (0, 0, 0)
            n_exog = exogenous.shape[1]
            k_params = (1 if self._include_mean else 0) + p + q + n_exog
            # Rows consumed by lags/differencing (max(p, q) + d), plus
            # enough remaining equations to identify every fitted parameter
            # with at least one degree of freedom left over. Without this,
            # e.g. order=(4, 0, 0) on 5 rows "fits" via a rank-deficient,
            # minimum-norm `lstsq` solve that silently returns meaningless
            # coefficients instead of failing loudly.
            min_rows = max(p, q) + d + k_params + 1
            detail = f" to fit order={self._order} ({k_params} parameter(s))"
        if len(paired) < min_rows:
            raise FlexDataError(
                f"ArimaRegressor needs at least {min_rows} row(s){detail}; "
                f"only {len(paired)} survived dropping nulls.",
                field="y",
            )

        self.n_samples = len(paired)
        self.data_window = (paired.index.min(), paired.index.max())
        self.exogenous_variables = list(exogenous.columns)
        self.output_variable = str(output.name)

        y_values = paired["__y__"].values
        x_df = exogenous if not exogenous.empty else None
        self._y_values = y_values
        self._training_index = paired["__y__"].index

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self._auto:
                order, seasonal_order = _auto_select_order(
                    y_values,
                    x_df.values if x_df is not None else None,
                    seasonal=self._seasonal_order is not None,
                    stationary=self._stationary,
                    **self._auto_kwargs,
                )
                P, D, Q, m = seasonal_order or (0, 0, 0, 0)
                if P != 0 or Q != 0:
                    raise FlexConfigError(
                        f"ArimaRegressor auto=True selected a seasonal order "
                        f"with P>0 or Q>0 ({seasonal_order}), which the "
                        f"direct-fit backend does not support. Retry with "
                        f"seasonal_order=None (the default, which also skips "
                        f"the seasonal search) or restrict the search via "
                        f"auto_kwargs (e.g. max_P=0, max_Q=0).",
                        field="seasonal_order",
                        value=seasonal_order,
                    )
                self._order = order
                self._seasonal_order = seasonal_order
                p, d, q = order
                if d > 1:
                    raise FlexConfigError(
                        f"ArimaRegressor auto=True selected d={d}, but the "
                        f"direct-fit backend and Pyomo surrogate only support "
                        f"d=0 or d=1. Retry with auto_kwargs (e.g. max_d=1).",
                        field="order",
                        value=order,
                    )
                fitted_model = _fit_direct(
                    y_values,
                    x_df,
                    order=order,
                    seasonal_order=(P, D, Q, m),
                    include_mean=self._include_mean,
                    include_drift=self._include_drift,
                    max_ar_persistence=self._max_ar_persistence,
                )
            else:
                p, d, q = self._order
                P, D, Q, m = self._seasonal_order or (0, 0, 0, 0)
                fitted_model = _fit_direct(
                    y_values,
                    x_df,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, m),
                    include_mean=self._include_mean,
                    include_drift=self._include_drift,
                    max_ar_persistence=self._max_ar_persistence,
                )

        self.model = fitted_model
        self._fitted = True
        self._order = _extract_order(fitted_model)
        self._seasonal_order = _extract_seasonal_order(fitted_model)

        if self._max_ar_persistence is not None:
            coef = self.model_.get("coef", {})
            ar_keys = [k for k in coef.keys() if str(k).startswith("ar")]
            for key in ar_keys:
                val = float(coef[key])
                # The fit is bounded to +/-max_ar_persistence (see
                # _fit_ar_ols/_fit_arma_nls), so this can only fire from
                # floating-point slop right at the boundary.
                if abs(val) > self._max_ar_persistence * (1 + 1e-9):
                    raise FlexConfigError(
                        f"ArimaRegressor rejected fitted AR coefficient "
                        f"{key}={val:.4f} because it exceeds "
                        f"max_ar_persistence={self._max_ar_persistence}. "
                        f"High-persistence AR models are unstable in the "
                        f"Pyomo mean-equation surrogate. Use a lower AR order "
                        f"or an MA-only model.",
                        field="ar_coefs",
                        value=val,
                    )

        fitted_values = fitted_model.fittedvalues
        if isinstance(fitted_values, pd.Series):
            fitted_values = fitted_values.values
        self._fitted_values = fitted_values

        fitted_level = fitted_model.fittedvalues_level
        valid = ~np.isnan(fitted_level)
        residual_ss = float(np.nansum((y_values[valid] - fitted_level[valid]) ** 2))
        rmse = math.sqrt(residual_ss / max(valid.sum(), 1))
        aic = float(fitted_model.aic)

        self.metrics = {"aic": aic, "rmse": rmse}
        return self

    @property
    def model_(self) -> dict[str, object]:
        """A dict-like view of the fitted model's key attributes.

        Provides ``"coef"``, ``"residuals"``, ``"aic"``, ``"bic"``,
        ``"aicc"``, ``"loglik"``, and ``"sigma2"`` keys so that any
        downstream code that accessed ``self.model.model_["..."]``
        continues to work without change.

        The direct backend names parameters ``ar1``, ``ma1``, ``const``,
        and exogenous column names directly, so no name normalisation is
        needed.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if not self._fitted or self.model is None:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "accessing model attributes."
            )
        m = self.model
        coef = {
            str(k): float(v) for k, v in zip(m.model.param_names, m.params, strict=True)
        }
        sigma2 = coef.pop("sigma2", float(m.sigma2))

        return {
            "coef": coef,
            "residuals": np.asarray(m.resid),
            "aic": float(m.aic),
            "bic": float(m.bic),
            "aicc": _aicc_from_direct(m),
            "loglik": float(m.llf),
            "sigma2": float(sigma2),
        }

    @property
    def coefficients(self) -> dict[str, float] | None:
        """The fitted model's parameters as a name -> value mapping.

        Returns:
            The parameter map, or ``None`` before :meth:`fit` is called.
        """
        if not self._fitted or self.model is None:
            return None
        return {
            str(k): float(v)
            for k, v in zip(
                self.model.model.param_names, self.model.params, strict=True
            )
        }

    @property
    def order(self) -> tuple[int, int, int] | None:
        """The fitted ``(p, d, q)`` order, or ``None`` before :meth:`fit`."""
        return self._order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int] | None:
        """The fitted seasonal order, or ``None`` before :meth:`fit`."""
        return self._seasonal_order

    @property
    def fitted(self) -> bool:
        """``True`` once :meth:`fit` has succeeded."""
        return self._fitted

    def _params(self) -> dict[str, float]:
        """Return the fitted model's named parameters as a flat dict.

        Returns:
            Mapping of parameter name to its fitted value.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if not self._fitted or self.model is None:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "accessing fitted parameters."
            )
        return dict(self.coefficients)

    def to_fit_result(self) -> FitResult:
        """Return this fit as the shared :class:`~.base.FitResult` shape.

        The coefficient map contains:
        - ``"const"`` when the model has a constant term,
        - ``"ar.L{j}"`` for each AR lag ``j``,
        - ``"ma.L{j}"`` for each MA lag ``j``,
        - exogenous coefficients keyed by their column name.

        Returns:
            A :class:`~flexparameterize.regression.base.FitResult` carrying
            the model's coefficients, ``aic``, ``rmse``, sample count, and
            data window.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        params = self._params()
        p, d, q = self._order  # type: ignore[misc]

        ar_coefs = _collect_lags(params, "ar", p)
        ma_coefs = _collect_lags(params, "ma", q)
        exog_coefs = [params.get(col, 0.0) for col in self.exogenous_variables]
        const = params.get("const", 0.0)

        coefficients = {
            **{f"ar.L{j}": v for j, v in enumerate(ar_coefs, 1)},
            **{f"ma.L{j}": v for j, v in enumerate(ma_coefs, 1)},
            **dict(zip(self.exogenous_variables, exog_coefs, strict=True)),
        }
        drift = params.get("drift", 0.0)
        deterministic = const if d == 0 else drift
        if deterministic != 0.0:
            coefficients["const" if d == 0 else "drift"] = deterministic

        return FitResult(
            coefficients=coefficients,
            metrics=dict(self.metrics),
            n_samples=self.n_samples,
            data_window=self.data_window,
        )

    def to_surrogate_spec(self) -> SurrogateSpec:
        """Return the fit as a persistable ``arima`` ``SurrogateSpec``.

        Uses the ``input_units``/``output_units`` recorded by :meth:`fit`.

        The ``data`` field of the returned spec matches the contract expected
        by :class:`~flexops.surrogates.arima.ArimaSurrogate`:

        - ``input_variables``: all fitted exogenous variable names and units.
        - ``output_variables``: the output variable name and its units.
                - ``order``: the fitted ``(p, d, q)`` tuple.
                - ``intercept``: the fitted level intercept when ``d=0``.
                - ``drift``: the fitted differenced-equation constant when ``d=1``.
        - ``ar_coefs``: list of AR coefficients in lag order.
        - ``ma_coefs``: list of MA coefficients in lag order.
        - ``exog_coefs``: list of exogenous coefficients, one per column in
          fitted order.
                - ``initial_state``: final levels and innovations for the next local
                    horizon.

        Returns:
            A :class:`~flexcore.config.schema.SurrogateSpec` of type
            ``SurrogateType.ARIMA``.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
            FlexConfigError: If ``input_units`` is missing an entry for a
                fitted exogenous column.
        """
        if not self._fitted:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "to_surrogate_spec()."
            )

        missing = [
            name for name in self.exogenous_variables if name not in self.input_units
        ]
        if missing:
            raise FlexConfigError(
                f"to_surrogate_spec is missing input_units for {missing}; "
                f"every fitted exogenous column ({self.exogenous_variables}) "
                f"needs an entry.",
                field="input_units",
                value=missing,
            )

        p, d, q = self._order  # type: ignore[misc]
        params = self._params()

        ar_coefs = _collect_lags(params, "ar", p)
        ma_coefs = _collect_lags(params, "ma", q)
        exog_coefs = [params.get(col, 0.0) for col in self.exogenous_variables]
        const = params.get("const", 0.0)
        drift = params.get("drift", None)

        coefficients: dict[str, object] = {"order": [int(p), int(d), int(q)]}
        if d == 0:
            coefficients["intercept"] = float(const)
        elif drift is not None:
            coefficients["drift"] = float(drift)
        if p > 0:
            coefficients["ar_coefs"] = [float(v) for v in ar_coefs]
        if q > 0:
            coefficients["ma_coefs"] = [float(v) for v in ma_coefs]
        if len(self.exogenous_variables) > 0:
            coefficients["exog_coefs"] = [float(v) for v in exog_coefs]
        # NOT IMPLEMENTED PLACE HOLDER
        # seasonal_order = None
        # if self._seasonal_order is not None:
        #     P, D, Q, m = self._seasonal_order
        #     seasonal_order = [int(P), int(D), int(Q), int(m)]

        residuals = np.asarray(self.model_["residuals"])
        history_prefix = max(p + d, q)
        training_index = self._training_index
        if getattr(training_index, "freq", None) is not None:
            time_step_seconds = float(pd.Timedelta(training_index.freq).total_seconds())
        elif len(training_index) > 1:
            time_step_seconds = float(
                (training_index[1] - training_index[0]).total_seconds()
            )
        else:
            time_step_seconds = 3600.0
        history = {
            "start_date": training_index[0].isoformat(),
            "time_step_seconds": time_step_seconds,
            "y_values": (
                np.asarray(
                    [self._y_values[0]] * (history_prefix - (p + d))
                    + self._y_values[: p + d].tolist()
                ).tolist()
                + np.asarray(self._y_values).tolist()
            ),
            "eps_values": [0.0] * (history_prefix + d) + residuals.tolist(),
        }

        return SurrogateSpec(
            surrogate_type=SurrogateType.ARIMA,
            data={
                "input_variables": {
                    name: self.input_units[name] for name in self.exogenous_variables
                },
                "output_variables": {self.output_variable: self.output_units},
                "coefficients": coefficients,
                "history": history,
            },
        )

    def fit_diagnostics(self) -> dict[str, float]:
        """Return extended fit statistics from the underlying direct fit result.

        Includes AIC, BIC, AICc, log-likelihood, and residual standard
        deviation alongside the standard ``aic`` and ``rmse``.

        Returns:
            Mapping of diagnostic name to value.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if not self._fitted or self.model is None:
            raise FlexDataError(
                "ArimaRegressor has no fit yet; call fit(X, y) before "
                "fit_diagnostics()."
            )
        m_ = self.model_
        residuals = m_["residuals"]
        n = len(residuals)
        params = self._params()
        k = len(params)
        llf = float(m_["loglik"])
        aic = float(m_["aic"])
        bic = float(m_["bic"])
        aicc = float(m_["aicc"])
        sigma2 = float(m_["sigma2"])
        fitted_level = self.model.fittedvalues_level
        valid = ~np.isnan(fitted_level)
        rmse = math.sqrt(
            float(np.nansum((self._y_values[valid] - fitted_level[valid]) ** 2))
            / max(valid.sum(), 1)
        )

        return {
            "aic": aic,
            "bic": bic,
            "aicc": aicc,
            "log_likelihood": llf,
            "n_parameters": float(k),
            "n_samples": float(n),
            "sigma2": sigma2,
            "rmse": rmse,
        }


# -- helpers -------------------------------------------------------------------


def _single_column(frame: pd.DataFrame | pd.Series, role: str) -> pd.Series:
    """Return the one column of ``frame`` as a named Series.

    Args:
        frame: A one-column ``DataFrame`` or a ``Series``.
        role: ``"X"`` or ``"y"``, for the error message.

    Returns:
        The column as a named ``Series``.

    Raises:
        FlexDataError: If ``frame`` does not hold exactly one column.
    """
    if isinstance(frame, pd.Series):
        return frame
    if frame.shape[1] != 1:
        raise FlexDataError(
            f"ArimaRegressor fits one output column; {role} has "
            f"{frame.shape[1]} ({list(frame.columns)}). Select the "
            "single output column.",
            field=role,
        )
    return frame.iloc[:, 0]


def _exog_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Return exogenous columns as a DataFrame (empty when no columns).

    Args:
        X: Zero or more exogenous input columns.

    Returns:
        A ``DataFrame`` (possibly empty) with the same index as ``X``.
    """
    if isinstance(X, pd.Series):
        return X.to_frame()
    return X if X.shape[1] > 0 else pd.DataFrame(index=X.index)


def _collect_lags(params: dict[str, float], prefix: str, n: int) -> list[float]:
    """Collect ``params[prefix{j}]`` for ``j = 1 … n``.

    Missing lags default to 0.0 so the resulting list always has exactly
    ``n`` entries, matching the AR/MA order the caller declared.

    Args:
        params: Fitted parameter mapping (name -> value).
        prefix: Parameter name prefix, either ``"ar"`` or ``"ma"``.
        n: Number of lags to collect.

    Returns:
        A list of ``n`` floats.
    """
    return [params.get(f"{prefix}{j}", 0.0) for j in range(1, n + 1)]


def _extract_order(fitted_model) -> tuple[int, int, int]:
    """Return ``(p, d, q)`` from a fitted :class:`_DirectResults`.

    Args:
        fitted_model: A fitted result object with ``model.k_ar``,
            ``model.k_diff``, and ``model.k_ma`` attributes.

    Returns:
        ``(p, d, q)`` as a tuple of ints.
    """
    return (
        int(fitted_model.model.k_ar),
        int(fitted_model.model.k_diff),
        int(fitted_model.model.k_ma),
    )


def _extract_seasonal_order(fitted_model) -> tuple[int, int, int, int] | None:
    """Return ``(P, D, Q, m)`` from a fitted result, or ``None``.

    Returns ``None`` when all seasonal terms are zero (i.e. a plain
    non-seasonal ARIMA).

    Args:
        fitted_model: A fitted result object with a ``model.seasonal_order``
            attribute.

    Returns:
        ``(P, D, Q, m)`` as a tuple of ints, or ``None`` when all seasonal
        terms are zero.
    """
    m_spec = fitted_model.model.seasonal_order
    P, D, Q, m = int(m_spec[0]), int(m_spec[1]), int(m_spec[2]), int(m_spec[3])
    if P == 0 and D == 0 and Q == 0:
        return None
    return (P, D, Q, m)


# -- Direct OLS/NLS fitting backend -------------------------------------------


class _DirectModel:
    """Structural descriptor for a direct-fit ARIMA model.

    This lightweight container mirrors the structural attributes of a
    statsmodels ``SARIMAX`` model so that the rest of
    ``ArimaRegressor`` (``_extract_order``, ``_extract_seasonal_order``,
    ``to_fit_result``, ``to_surrogate_spec``) can read fitted-model
    metadata without caring which backend produced it.

    Notes:
        Final parameter estimation always uses the direct scipy fit in
        this module; ``statsforecast`` is only ever used for
        ``auto=True`` order *selection*, never for final estimation.

    Attributes:
        param_names: Ordered parameter names, e.g.
            ``["const", "ar1", "ma1", "feed_volume_kg"]``.
        k_ar: Number of AR terms (``p``).
        k_ma: Number of MA terms (``q``).
        k_diff: Differencing order (``d``).
        seasonal_order: Seasonal ``(P, D, Q, m)`` tuple, or ``(0, 0, 0, 0)``
            when no seasonal terms are present.
        has_const: Whether the model includes a constant/intercept term.
        has_drift: Whether the model includes a drift term (only possible
            when ``d == 1``).
    """

    def __init__(
        self,
        param_names: list[str],
        k_ar: int,
        k_ma: int,
        k_diff: int,
        seasonal_order: tuple[int, int, int, int],
        has_const: bool,
        has_drift: bool = False,
    ) -> None:
        self._param_names = list(param_names)
        self.k_ar = k_ar
        self.k_ma = k_ma
        self.k_diff = k_diff
        self.seasonal_order = seasonal_order
        self.has_const = has_const
        self.has_drift = has_drift

    @property
    def param_names(self) -> list[str]:
        """Ordered names of all fitted parameters."""
        return self._param_names


class _DirectResults:
    """Fitted ARIMA model result, mirroring the statsmodels
    ``SARIMAXResults`` interface.

    Returned by :func:`_fit_direct` so the rest of ``ArimaRegressor`` can
    consume fitted models without knowing which backend produced them.

    Attributes:
        params: Fitted parameter vector in the same order as
            ``model.param_names``.
        resid: In-sample residuals, aligned to the differenced series
            ``y`` (length ``nobs``; leading lags are zero-padded).
        fittedvalues: In-sample fitted values on the *differenced* scale
            when ``d > 0``; on the level scale when ``d == 0``.
        nobs: Number of effective observations (length of the differenced
            series when ``d > 0``).
        df_model: Number of fitted parameters, including the constant if
            present.
        llf: Log-likelihood computed from the residual sum of squares.
        aic: Akaike information criterion.
        bic: Bayesian information criterion.
        sigma2: Residual variance estimate.
    """

    def __init__(
        self,
        params: np.ndarray,
        residuals: np.ndarray,
        fitted_values: np.ndarray,
        y: np.ndarray,
        k_params: int,
        param_names: list[str],
        k_ar: int,
        k_ma: int,
        k_diff: int,
        seasonal_order: tuple[int, int, int, int],
        has_const: bool,
        has_drift: bool = False,
        y_original: np.ndarray | None = None,
    ) -> None:
        self._params = np.asarray(params, dtype=float)
        self._resid = np.asarray(residuals, dtype=float)
        self._fittedvalues = np.asarray(fitted_values, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._y_original = (
            np.asarray(y_original, dtype=float) if y_original is not None else None
        )
        self._k_params = k_params
        self._model = _DirectModel(
            param_names, k_ar, k_ma, k_diff, seasonal_order, has_const, has_drift
        )

    @property
    def params(self) -> np.ndarray:
        """Fitted parameter vector, ordered to match :attr:`model.param_names`."""
        return self._params

    @property
    def model(self) -> _DirectModel:
        """Structural model descriptor (order, seasonal order, etc.)."""
        return self._model

    @property
    def resid(self) -> np.ndarray:
        """In-sample residuals from the mean equation.

        Length equals ``nobs``; leading lags up to ``max(k_ar, k_ma)``
        are zero-padded because no fitted values are defined there.
        """
        return self._resid

    @property
    def fittedvalues(self) -> np.ndarray:
        """In-sample fitted values.

        On the differenced scale when ``k_diff > 0``; on the level scale
        when ``k_diff == 0``.  Use :attr:`fittedvalues_level` to always
        obtain level-scale fitted values.
        """
        return self._fittedvalues

    @property
    def fittedvalues_level(self) -> np.ndarray:
        """In-sample fitted values integrated back to the level scale.

        When ``k_diff == 0`` or the original level series is unavailable,
        this is identical to :attr:`fittedvalues`.  When ``k_diff > 0``
        and ``y_original`` is present, the differenced fitted values are
        cumulatively summed and added to the initial level so the result
        is directly comparable to the original observed ``y``.
        """
        if self.model.k_diff == 0 or self._y_original is None:
            return self._fittedvalues
        fitted = np.full(len(self._y_original), np.nan)
        d = int(self.model.k_diff)
        fitted[:d] = self._y_original[:d]
        if d == 1:
            fitted[d:] = self._y_original[d - 1 : -1] + self._fittedvalues[d - 1 :]
        else:
            fitted[d:] = self._y_original[d - 1 : -1] + np.cumsum(
                self._fittedvalues[d - 1 :]
            )
        return fitted

    @property
    def nobs(self) -> int:
        """Number of effective observations in the fitted series."""
        return int(len(self._y))

    @property
    def df_model(self) -> int:
        """Number of fitted parameters, counting the constant if present."""
        return int(self._k_params)

    def _llf(self) -> float:
        n_eff = len(self._resid)
        rss = float(np.sum(self._resid**2))
        if rss <= 0 or n_eff == 0:
            return -np.inf
        return (
            -n_eff / 2.0 * np.log(2.0 * np.pi)
            - n_eff / 2.0 * np.log(rss / n_eff)
            - n_eff / 2.0
        )

    @property
    def llf(self) -> float:
        return self._llf()

    @property
    def aic(self) -> float:
        k = self.df_model + 1
        return -2.0 * self.llf + 2.0 * k

    @property
    def bic(self) -> float:
        k = self.df_model + 1
        n = self.nobs
        return -2.0 * self.llf + k * np.log(n)

    @property
    def sigma2(self) -> float:
        rss = float(np.sum(self._resid**2))
        n_eff = len(self._resid)
        return rss / n_eff if n_eff > 0 else 0.0

    def predict(
        self,
        steps: int = 1,
        exog: np.ndarray | None = None,
        start: int | None = None,
        dynamic: bool = True,
    ) -> np.ndarray:
        """Recursive multi-step forecast using the mean equation.

        For ``steps=0`` (or when ``exog`` matches the training length),
        returns the in-sample fitted values.  For ``steps>0``, forecasts
        are generated recursively: each predicted value feeds back as the
        AR lag for the next step, and MA terms are zeroed after the first
        step (matching the Pyomo surrogate's forecast behaviour).

        Args:
            steps: Number of steps to forecast ahead.
            exog: Future exogenous values of shape ``(steps, n_exog)``.
                Pass ``None`` when the model has no exogenous regressors.
            start: Optional start index within the training data for
                in-sample dynamic prediction.  When ``start`` is provided,
                the method uses training data up to ``start`` as initial
                history and recursively predicts from ``start`` onward.
                This matches the Pyomo surrogate's 0DOF solve behaviour.
            dynamic: Must be ``True`` (the default). Recursive prediction
                (each predicted value feeds back as an AR lag) is the only
                mode this direct-fit backend implements; one-step-ahead
                prediction using actual observed lags is not implemented.

        Returns:
            Array of forecast values of length ``steps``.

        Raises:
            FlexConfigError: If ``dynamic=False`` is passed.
        """
        if not dynamic:
            raise FlexConfigError(
                "ArimaRegressor's direct-fit predict() only implements "
                "dynamic=True (recursive) prediction; one-step-ahead "
                "dynamic=False prediction is not implemented. Use "
                "`fittedvalues` for in-sample one-step-ahead values, or "
                "omit `dynamic` (it defaults to True)."
            )
        p = self.model.k_ar
        q = self.model.k_ma
        _d = self.model.k_diff
        has_const = self.model.has_const
        has_drift = getattr(self.model, "has_drift", False)

        idx = 0
        c = float(self._params[idx]) if has_const else 0.0
        idx += has_const
        drift = float(self._params[idx]) if has_drift else 0.0
        idx += has_drift
        ar = self._params[idx : idx + p]
        idx += p
        ma = self._params[idx : idx + q]
        idx += q
        n_exog = self._params.shape[0] - idx
        beta = self._params[idx : idx + n_exog] if n_exog > 0 else np.zeros(n_exog)

        if start is not None:
            # In-sample dynamic prediction: use training data BEFORE `start`
            # as initial history, then recursively predict forward from `start`.
            y_source = self._y_original if self._y_original is not None else self._y
            if _d == 0:
                y_hist = list(y_source[max(0, start - p) : start])
                eps_hist = list(self._resid[max(0, start - q) : start]) if q > 0 else []
            else:
                # For d=1, work with original (level) series.
                # Need at least p+1 history values: y[t-1], y[t-2], ..., y[t-p-1]
                y_hist = list(y_source[max(0, start - p - 1) : start])
                # self._resid is aligned to the differenced series (length
                # nobs = n - d), so a level-index `start` must be shifted by
                # `_d` before indexing into it.
                eps_hist = (
                    list(self._resid[max(0, start - q - _d) : start - _d])
                    if q > 0
                    else []
                )
                if len(y_hist) < p + 1:
                    y_hist = list(y_source[: p + 1])
            forecasts = []

            for step in range(steps):
                if _d == 0:
                    ar_part = sum(ar[j] * y_hist[-(j + 1)] for j in range(p))
                    ma_part = (
                        sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                        if q > 0 and len(eps_hist) >= q
                        else 0.0
                    )
                    exog_part = (
                        sum(beta[k] * exog[step, k] for k in range(n_exog))
                        if exog is not None
                        else 0.0
                    )
                    drift_part = 0.0
                    y_hat = c + drift_part + ar_part + ma_part + exog_part
                else:
                    # d=1: y[t] = y[t-1] + drift +
                    # sum(ar[i]*(y[t-i]-y[t-i-1])) + ma + exog
                    ar_diff_part = sum(
                        ar[j] * (y_hist[-(j + 1)] - y_hist[-(j + 2)]) for j in range(p)
                    )
                    ma_part = (
                        sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                        if q > 0 and len(eps_hist) >= q
                        else 0.0
                    )
                    exog_part = (
                        sum(beta[k] * exog[step, k] for k in range(n_exog))
                        if exog is not None
                        else 0.0
                    )
                    drift_part = drift
                    y_diff_hat = c + drift_part + ar_diff_part + ma_part + exog_part
                    y_hat = y_hist[-1] + y_diff_hat

                forecasts.append(y_hat)
                y_hist.append(y_hat)
                if q > 0:
                    eps_hist.append(0.0)

            return np.array(forecasts)

        # Out-of-sample forecast from end of training data
        if _d == 0:
            y_hist = list(self._y_original[-p:] if p > 0 else [])
        else:
            # For d=1, need actual level values as lags
            y_hist = list(
                self._y_original[-(p + 1) :] if p > 0 else self._y_original[-2:]
            )
        eps_hist = list(self._resid[-q:] if q > 0 else [])
        forecasts = []

        for step in range(steps):
            if _d == 0:
                ar_part = sum(ar[j] * y_hist[-(j + 1)] for j in range(p))
                ma_part = sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                exog_part = (
                    sum(beta[k] * exog[step, k] for k in range(n_exog))
                    if exog is not None
                    else 0.0
                )
                drift_part = 0.0
                y_hat = c + drift_part + ar_part + ma_part + exog_part
            else:
                # d=1: y[t] = y[t-1] + drift +
                # sum(ar[i] * (y[t-i] - y[t-i-1])) + ma + exog
                ar_diff_part = sum(
                    ar[j] * (y_hist[-(j + 1)] - y_hist[-(j + 2)]) for j in range(p)
                )
                ma_part = sum(ma[j] * eps_hist[-(j + 1)] for j in range(q))
                exog_part = (
                    sum(beta[k] * exog[step, k] for k in range(n_exog))
                    if exog is not None
                    else 0.0
                )
                drift_part = drift
                y_diff_hat = c + drift_part + ar_diff_part + ma_part + exog_part
                y_hat = y_hist[-1] + y_diff_hat

            forecasts.append(y_hat)
            y_hist.append(y_hat)
            eps_hist.append(0.0)

        return np.array(forecasts)


def _fit_pure_regression(
    y: np.ndarray,
    x_values: np.ndarray | None,
    exog_names: list[str],
    has_const: bool,
    has_drift: bool,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Fit y = c + drift + Xβ + ε via OLS."""
    n = len(y)
    cols: list[np.ndarray] = []
    if has_const:
        cols.append(np.ones(n))
    if has_drift:
        cols.append(np.ones(n))
    if x_values is not None and x_values.shape[1] > 0:
        for k in range(x_values.shape[1]):
            cols.append(x_values[:, k])

    param_names: list[str] = []
    if has_const:
        param_names.append("const")
    if has_drift:
        param_names.append("drift")
    param_names.extend(exog_names)

    if not cols:
        c = float(np.mean(y))
        residuals = y - c
        fitted = np.full(n, c)
        return np.array([]), param_names, residuals, fitted

    X = np.column_stack(cols)
    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ theta
    return theta, param_names, y - y_hat, y_hat


def _fit_ar_ols(
    y: np.ndarray,
    p: int,
    x_values: np.ndarray | None,
    exog_names: list[str],
    has_const: bool,
    has_drift: bool,
    max_ar_persistence: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Fit pure AR(p) via OLS, bounded to +/-max_ar_persistence when set."""
    n = len(y)
    max_lag = p
    T = n - max_lag

    cols: list[np.ndarray] = []
    if has_const:
        cols.append(np.ones(T))
    if has_drift:
        cols.append(np.ones(T))
    for j in range(1, p + 1):
        cols.append(y[max_lag - j : n - j])
    if x_values is not None and x_values.shape[1] > 0:
        for k in range(x_values.shape[1]):
            cols.append(x_values[max_lag:, k])

    param_names: list[str] = []
    if has_const:
        param_names.append("const")
    if has_drift:
        param_names.append("drift")
    for j in range(1, p + 1):
        param_names.append(f"ar{j}")
    param_names.extend(exog_names)

    X = np.column_stack(cols)
    y_vec = y[max_lag:]
    if max_ar_persistence is None:
        theta, _, _, _ = np.linalg.lstsq(X, y_vec, rcond=None)
    else:
        from scipy.optimize import lsq_linear

        ar_offset = has_const + has_drift
        lb = np.full(X.shape[1], -np.inf)
        ub = np.full(X.shape[1], np.inf)
        lb[ar_offset : ar_offset + p] = -max_ar_persistence
        ub[ar_offset : ar_offset + p] = max_ar_persistence
        theta = lsq_linear(X, y_vec, bounds=(lb, ub)).x

    residuals = np.zeros(n)
    fitted = np.full(n, np.nan)
    if T > 0:
        y_hat = X @ theta
        residuals[max_lag:] = y_vec - y_hat
        fitted[max_lag:] = y_hat

    return theta, param_names, residuals, fitted


def _arma_residuals(
    theta: np.ndarray,
    y: np.ndarray,
    x: np.ndarray | None,
    p: int,
    q: int,
    has_const: bool,
    has_drift: bool,
    n_exog: int,
) -> np.ndarray:
    """Residual function for scipy.optimize.least_squares.

    Computes the mean-equation residual for each observation t:
        r[t] = y[t] - (c + drift + Σ ar_j·y[t-j-1]
        + Σ ma_j·ε[t-j-1] + Σ β_k·x[t,k])

    Returns the effective residuals (t >= max(p, q)) as a flat vector for
    least-squares optimisation.
    """
    n = len(y)
    idx = 0
    c = float(theta[idx]) if has_const else 0.0
    idx += has_const
    drift = float(theta[idx]) if has_drift else 0.0
    idx += has_drift

    ar = theta[idx : idx + p]
    idx += p
    ma = theta[idx : idx + q]
    idx += q
    beta = theta[idx : idx + n_exog] if n_exog > 0 else np.zeros(n_exog)

    max_lag = max(p, q)
    eps = np.zeros(n)

    for t in range(max_lag, n):
        ar_part = 0.0
        for j in range(p):
            ar_part += ar[j] * y[t - j - 1]

        ma_part = 0.0
        for j in range(q):
            ma_part += ma[j] * eps[t - j - 1]

        exog_part = 0.0
        for k in range(n_exog):
            exog_part += beta[k] * x[t, k]

        drift_part = drift
        eps[t] = y[t] - (c + drift_part + ar_part + ma_part + exog_part)

    return eps[max_lag:]


def _fit_arma_nls(
    y: np.ndarray,
    p: int,
    q: int,
    x_values: np.ndarray | None,
    exog_names: list[str],
    has_const: bool,
    has_drift: bool,
    max_ar_persistence: float | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Fit ARMA(p,q) via nonlinear least squares (Levenberg-Marquardt).

    The initial guess is obtained by OLS on the AR lags + exog (ignoring
    MA terms), with MA coefficients starting at zero. AR coefficients are
    bounded to +/-max_ar_persistence when set (this forces the 'trf'
    solver, since Levenberg-Marquardt does not support bounds).
    """
    from scipy.optimize import least_squares

    n = len(y)
    n_exog = x_values.shape[1] if x_values is not None else 0

    # Build initial guess via OLS (no MA terms)
    max_lag = p
    cols: list[np.ndarray] = []
    if has_const:
        cols.append(np.ones(n - max_lag) if max_lag > 0 else np.ones(n))
    if has_drift:
        cols.append(np.ones(n - max_lag) if max_lag > 0 else np.ones(n))
    for j in range(1, p + 1):
        start = max_lag - j
        end = n - j
        if end > start:
            cols.append(y[start:end])
    if x_values is not None and n_exog > 0:
        for k in range(n_exog):
            cols.append(x_values[max_lag:, k])

    theta0 = np.zeros(has_const + has_drift + p + q + n_exog)

    if cols:
        y_init = y[max_lag:] if max_lag > 0 else y
        X_init = np.column_stack(cols)
        try:
            ols_theta, _, _, _ = np.linalg.lstsq(X_init, y_init, rcond=None)
            idx_ols = 0
            idx_nls = 0
            if has_const:
                theta0[idx_nls] = ols_theta[idx_ols]
                idx_ols += 1
                idx_nls += 1
            if has_drift:
                theta0[idx_nls] = ols_theta[idx_ols]
                idx_ols += 1
                idx_nls += 1
            if p > 0:
                theta0[idx_nls : idx_nls + p] = ols_theta[idx_ols : idx_ols + p]
                idx_ols += p
                idx_nls += p
            idx_nls += q  # skip MA slots
            if n_exog > 0:
                theta0[idx_nls : idx_nls + n_exog] = ols_theta[
                    idx_ols : idx_ols + n_exog
                ]
        except Exception:
            if has_const:
                theta0[0] = float(np.mean(y))

    if max_ar_persistence is None:
        bounds = (-np.inf, np.inf)
        method = "lm"
    else:
        ar_offset = has_const + has_drift
        lb = np.full(theta0.shape, -np.inf)
        ub = np.full(theta0.shape, np.inf)
        lb[ar_offset : ar_offset + p] = -max_ar_persistence
        ub[ar_offset : ar_offset + p] = max_ar_persistence
        theta0[ar_offset : ar_offset + p] = np.clip(
            theta0[ar_offset : ar_offset + p],
            lb[ar_offset : ar_offset + p],
            ub[ar_offset : ar_offset + p],
        )
        bounds = (lb, ub)
        method = "trf"  # 'lm' does not support bounds

    result = least_squares(
        _arma_residuals,
        theta0,
        args=(y, x_values, p, q, has_const, has_drift, n_exog),
        method=method,
        bounds=bounds,
        verbose=0,
        max_nfev=5000,
        ftol=1e-8,
        xtol=1e-8,
    )
    theta_opt = result.x

    # Reconstruct fitted values and residuals from optimal parameters
    max_lag_final = max(p, q)
    eps = np.zeros(n)
    fitted = np.full(n, np.nan)

    idx = 0
    c = float(theta_opt[idx]) if has_const else 0.0
    idx += has_const
    drift_val = float(theta_opt[idx]) if has_drift else 0.0
    idx += has_drift
    ar = theta_opt[idx : idx + p]
    idx += p
    ma = theta_opt[idx : idx + q]
    idx += q
    beta = theta_opt[idx : idx + n_exog] if n_exog > 0 else np.zeros(n_exog)

    for t in range(max_lag_final, n):
        ar_part = sum(ar[j] * y[t - j - 1] for j in range(p))
        ma_part = sum(ma[j] * eps[t - j - 1] for j in range(q))
        exog_part = (
            sum(beta[k] * x_values[t, k] for k in range(n_exog)) if n_exog > 0 else 0.0
        )
        drift_part = drift_val
        y_hat = c + drift_part + ar_part + ma_part + exog_part
        fitted[t] = y_hat
        eps[t] = y[t] - y_hat

    residuals = np.zeros(n)
    residuals[max_lag_final:] = eps[max_lag_final:]

    param_names: list[str] = []
    if has_const:
        param_names.append("const")
    if has_drift:
        param_names.append("drift")
    for j in range(1, p + 1):
        param_names.append(f"ar{j}")
    for j in range(1, q + 1):
        param_names.append(f"ma{j}")
    param_names.extend(exog_names)

    return theta_opt, param_names, residuals, fitted


def _fit_direct(
    y_values: np.ndarray,
    x_df: pd.DataFrame | None,
    *,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    include_mean: bool,
    include_drift: bool = False,
    max_ar_persistence: float | None = None,
) -> _DirectResults:
    """Fit ARIMA directly via OLS/NLS, matching the Pyomo surrogate's objective.

    Uses ordinary least squares for pure AR(p) models (closed-form) and
    ``scipy.optimize.least_squares`` (Levenberg-Marquardt) for models with
    MA terms.  The residual function is exactly the mean ARIMA equation
    that the Pyomo surrogate implements, so fitted parameters reproduce the
    surrogate one-to-one.

    Differencing (``d>0``) is applied to ``y_values`` before fitting.  The
    Pyomo surrogate supports ``d=0`` and ``d=1``; ``d>1`` will raise
    ``FlexConfigError`` when ``to_surrogate_spec`` is called.

    Args:
        y_values: Endogenous time-series values.
        x_df: Exogenous regressor DataFrame, or ``None``.
        order: ``(p, d, q)`` ARIMA order.
        seasonal_order: ``(P, D, Q, m)`` seasonal order.  Only
            non-seasonal ``(0, 0, 0, 0)`` is currently supported.
        include_mean: Whether to include a constant term.
        include_drift: Whether to include a drift term.  Only meaningful
            when ``d=1``.
        max_ar_persistence: Bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]`` during the fit.
            ``None`` fits unconstrained.

    Returns:
        Fitted :class:`_DirectResults`.
    """
    p, d, q = order
    P, D, Q, m = seasonal_order

    if P > 0 or D > 0 or Q > 0:
        # Defense in depth: __init__ and the auto=True path both already
        # reject P>0/D>0/Q>0 before this is ever called; this should be
        # unreachable via the public API.
        raise FlexConfigError(
            f"Seasonal ARIMA terms are not supported by the direct fit "
            f"backend. Got seasonal_order={seasonal_order}.",
            field="seasonal_order",
            value=seasonal_order,
        )

    exog_names = list(x_df.columns) if x_df is not None else []
    x_values = x_df.values if x_df is not None else None
    n_exog = x_values.shape[1] if x_values is not None else 0

    if d > 0:
        y_original = y_values.copy()
        y_values = np.diff(y_values, n=d)
        if x_values is not None:
            x_values = x_values[d:]
    else:
        y_original = y_values.copy()

    has_const = include_mean and d == 0
    has_drift = d == 1 and (include_mean or include_drift)
    k_params = (1 if has_const else 0) + (1 if has_drift else 0) + p + q + n_exog

    if p == 0 and q == 0:
        theta, param_names, residuals, fitted = _fit_pure_regression(
            y_values, x_values, exog_names, has_const, has_drift
        )
    elif q == 0:
        theta, param_names, residuals, fitted = _fit_ar_ols(
            y_values,
            p,
            x_values,
            exog_names,
            has_const,
            has_drift,
            max_ar_persistence,
        )
    else:
        theta, param_names, residuals, fitted = _fit_arma_nls(
            y_values,
            p,
            q,
            x_values,
            exog_names,
            has_const,
            has_drift,
            max_ar_persistence,
        )

    return _DirectResults(
        params=theta,
        residuals=residuals,
        fitted_values=fitted,
        y=y_values,
        k_params=k_params,
        param_names=param_names,
        k_ar=p,
        k_ma=q,
        k_diff=d,
        seasonal_order=(P, D, Q, m),
        has_const=has_const,
        has_drift=has_drift,
        y_original=y_original,
    )


def _aicc_from_direct(results: _DirectResults) -> float:
    """Compute AICc from a :class:`_DirectResults` object.

    Uses the effective sample size (where residuals are defined) for the
    likelihood and the full sample size for the penalty term.

    Args:
        results: Fitted :class:`_DirectResults`.

    Returns:
        The corrected AIC value.
    """
    k = results.df_model + 1
    n = results.nobs
    aic = float(results.aic)
    return aic + (2.0 * k * (k + 1)) / (n - k - 1)


def _auto_select_order(
    y_values: np.ndarray,
    x_values: np.ndarray | None,
    *,
    seasonal: bool,
    stationary: bool,
    **auto_kwargs: object,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int] | None]:
    """Use statsforecast AutoARIMA to select the best order.

    ``d`` and ``D`` are forced to 0 regardless of what is passed in
    ``auto_kwargs``.  The caller is responsible for refitting the returned
    order with the direct scipy backend for Pyomo-compatible parameters.

    Args:
        y_values: Endogenous time-series values.
        x_values: Exogenous regressor matrix, or ``None``.
        seasonal: Whether to include seasonal terms in the search.
        stationary: If ``True``, restrict the search to stationary models.
        **auto_kwargs: Extra keyword arguments forwarded to AutoARIMA
            (e.g. ``max_p``, ``max_q``, ``season_length``).

    Returns:
        ``(order, seasonal_order)`` where ``order`` is ``(p, d, q)`` with
        ``d`` in ``{0, 1}``, and ``seasonal_order`` is ``(P, 0, Q, m)`` or
        ``None``.

    Raises:
        FlexConfigError: If statsforecast is not installed.
    """
    try:
        from statsforecast.models import AutoARIMA
    except ImportError as exc:
        raise FlexConfigError(
            "ArimaRegressor auto=True requires statsforecast for order "
            "selection. Install it with `pip install 'flex-pse[parameterize]'`."
        ) from exc

    kwargs = dict(auto_kwargs)
    kwargs.setdefault("D", 0)
    if stationary:
        kwargs.setdefault("stationary", True)

    auto_model = AutoARIMA(seasonal=seasonal, ic="aic", **kwargs)
    fitted = auto_model.fit(y_values, X=x_values)

    arma = fitted.model_["arma"]
    p, q, P, Q, m_sf, d, D = arma
    d = int(d)
    if d not in (0, 1):
        raise FlexConfigError(
            f"ArimaRegressor auto=True selected d={d}, but the direct-fit "
            f"backend and Pyomo surrogate only support d=0 or d=1. "
            f"Restrict the search via auto_kwargs (e.g. max_d=1).",
            field="order",
            value=(int(p), d, int(q)),
        )
    order = (int(p), d, int(q))
    seasonal_order: tuple[int, int, int, int] | None
    if P == 0 and Q == 0:
        seasonal_order = None
    else:
        seasonal_order = (int(P), 0, int(Q), int(m_sf))
    return order, seasonal_order
