"""ArimaRegressor: time-series ARIMA regressor with exogenous inputs.

Fits regression with ARIMA errors by directly minimizing the mean-equation
residual with ``scipy.optimize.least_squares``, and reduces the result to the
standard :class:`~flexparameterize.regression.base.FitResult` /
``SurrogateSpec`` shape, so every downstream consumer (provenance logging,
``emit_model_config``, ``apply_to_model``) works without change. The fitting
and forecasting machinery lives in
:mod:`flexparameterize.regression.utils.arima_utils`.

``statsforecast`` ships in the ``[parameterize]`` extra and is used only when
``auto=True``, for order selection; the final parameter fit always uses the
direct least-squares backend, which minimizes the exact residual the Pyomo
surrogate implements.

**Restriction**: ``d`` may be ``0`` or ``1``; seasonal differencing (``D``)
must be ``0``, and seasonal AR/MA terms (``P>0`` or ``Q>0``) are not
supported. The Pyomo surrogate implements the mean ARIMA equation in closed
form, which supports at most a single order of differencing.

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

:class:`ArimaRegressor` attributes:
    model: The fitted
        :class:`~flexparameterize.regression.utils.arima_utils.DirectResults`
        (or ``None`` before :meth:`fit`).
    n_samples: Number of rows the fit used, after dropping nulls.
    metrics: ``{"aic", "rmse", "free_run_rmse"}`` of the fitted model.
    data_window: ``(first, last)`` index value of the rows used.
    exogenous_variables: Column names of the fitted exogenous inputs.
    output_variable: Column name of the fitted output.
    input_units: Units of every fitted exogenous column, keyed by column
        name.  Set by :meth:`fit`.
    output_units: Units of the fitted output column.  Set by :meth:`fit`.
"""

from __future__ import annotations

import warnings
from functools import partial

import pandas as pd

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexcore.logger import get_logger
from flexparameterize.regression.base import FitResult
from flexparameterize.regression.utils.arima_utils import (
    ArimaTerms,
    ar_max_root,
    auto_forecast_horizon,
    auto_select_order,
    fit_direct,
    fit_output_error_ipopt,
    history_payload,
    ma_max_root,
    step_seconds,
    surrogate_coefficients,
)

_log = get_logger(__name__)

_FIT_OBJECTIVES = ("equation_error", "output_error")
_FIT_SOLVERS = ("scipy", "ipopt")


class ArimaRegressor:
    """Fit regression with ARIMA errors and expose the shared fit protocol.

    Either supply an explicit ``order``, or set ``auto=True`` to delegate
    order selection to statsforecast ``AutoARIMA`` and refit the winning
    order with the direct least-squares backend.

    **Restriction**: ``d`` may be ``0`` or ``1`` (the Pyomo surrogate's mean
    ARIMA equation supports at most a single order of differencing).
    Seasonal differencing (``D``) must be ``0``, and seasonal AR/MA terms
    (``P>0`` or ``Q>0``) are not supported at all — only a plain,
    non-seasonal ARIMA (``seasonal_order=None`` or the trivial
    ``(0, 0, 0, m)``) can be fit.

    **Fitting backend**: ``scipy.optimize.least_squares`` minimizes the
    mean-equation residual directly. This is the exact objective the Pyomo
    surrogate implements, so fitted parameters reproduce the surrogate
    one-to-one.

    **Stability**: AR coefficients are bounded to
    ``[-max_ar_persistence, max_ar_persistence]`` (default 0.85) during the
    fit itself, and :meth:`fit` rejects a fitted AR block that is not
    stationary, which a per-coefficient bound alone does not prevent. The
    surrogate implements the mean equation as a difference equation, so a
    non-stationary AR block makes every forecast and optimization diverge.

    Args:
        order: ``(p, d, q)`` ARIMA order; required when ``auto`` is ``False``.
            Must satisfy ``d in (0, 1)``.
        seasonal_order: ``(P, D, Q, m)`` seasonal order; pass ``None`` (the
            default) for a plain (non-seasonal) ARIMA.  If provided, must
            satisfy ``D == 0`` and ``P == Q == 0``, so only the trivial
            ``(0, 0, 0, m)`` (equivalent to ``None``) is accepted.
        include_mean: Whether to include the model's deterministic term.
            This is a level intercept for ``d=0`` and constant drift for
            ``d=1``. Default ``True``.
        include_drift: Whether to include a drift term (constant in the
            differenced series). Default ``False``. For ``d=1`` this is an
            explicit alias for the default ``include_mean`` deterministic
            term; setting it with ``d=0`` raises ``FlexConfigError``.
        auto: If ``True``, run ``statsforecast.models.AutoARIMA`` to
            discover the best order, then refit that order with the direct
            backend for Pyomo compatibility.  AutoARIMA may select ``d=0``
            or ``d=1``; ``D`` is always forced to 0.  Use ``auto_kwargs`` to
            restrict the search (e.g. ``max_d=1``).
        max_ar_persistence: Bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]`` during the fit
            itself (via bounded least squares). Default ``0.85``. Set to
            ``None`` to leave the coefficients unbounded; the stationarity
            check still applies.
        stationary: If ``True``, force ``stationary=True`` in
            ``statsforecast.models.AutoARIMA``, which restricts the
            search to models with stationary AR coefficients.  Default
            ``False``.
        fit_objective: ``"equation_error"`` (default) minimizes one-step
            residuals using actual lagged values. ``"output_error"`` then
            refines that fit against windowed free-run error, the error the
            surrogate commits when it simulates forward with innovations at
            zero; its MA coefficients are bounded to ``[-0.99, 0.99]`` and a
            non-invertible MA block is rejected.
        forecast_horizon: Free-run window length in steps for an
            ``output_error`` fit, or ``"auto"`` (default) to span the data
            up to 192 steps. Set it to the horizon you will forecast over.
        fit_solver: ``"scipy"`` (default) or ``"ipopt"``, the optimizer
            behind an ``output_error`` fit. ``"ipopt"`` minimizes the free
            run through the real ``ArimaSurrogate`` over the whole series.
        **auto_kwargs: Extra keyword arguments forwarded to ``AutoARIMA``
            (e.g. ``max_p``, ``max_q``, ``season_length``). ``D`` defaults
            to 0; pass ``max_d=1`` to allow differencing.

    Raises:
        FlexConfigError: If ``d > 1``, ``D > 0``, or a seasonal AR/MA order
            (``P > 0`` or ``Q > 0``) is requested; if ``include_drift=True``
            is combined with anything other than an explicit ``d=1`` order
            (including ``auto=True``, where no order is known yet); if
            ``max_ar_persistence`` is not a number in ``(0, 1]`` or ``None``;
            or if ``fit_objective``, ``forecast_horizon``, or ``fit_solver``
            is invalid or combined with an option it does not apply to.
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
        fit_objective: str = "equation_error",
        forecast_horizon: int | str = "auto",
        fit_solver: str = "scipy",
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
        if fit_objective not in _FIT_OBJECTIVES:
            raise FlexConfigError(
                f"ArimaRegressor fit_objective must be one of "
                f"{list(_FIT_OBJECTIVES)}, got {fit_objective!r}.",
                field="fit_objective",
                value=fit_objective,
            )
        if forecast_horizon != "auto" and (
            isinstance(forecast_horizon, bool)
            or not isinstance(forecast_horizon, int)
            or forecast_horizon < 1
        ):
            raise FlexConfigError(
                f"ArimaRegressor forecast_horizon must be a positive int or "
                f'"auto", got {forecast_horizon!r}.',
                field="forecast_horizon",
                value=forecast_horizon,
            )
        if fit_objective == "equation_error" and forecast_horizon != "auto":
            raise FlexConfigError(
                "ArimaRegressor forecast_horizon only affects an "
                'output_error fit; with fit_objective="equation_error" it '
                "would be silently ignored. Drop forecast_horizon or set "
                'fit_objective="output_error".',
                field="forecast_horizon",
                value=forecast_horizon,
            )
        if fit_solver not in _FIT_SOLVERS:
            raise FlexConfigError(
                f"ArimaRegressor fit_solver must be one of "
                f"{list(_FIT_SOLVERS)}, got {fit_solver!r}.",
                field="fit_solver",
                value=fit_solver,
            )
        if fit_solver == "ipopt" and fit_objective != "output_error":
            raise FlexConfigError(
                'ArimaRegressor fit_solver="ipopt" only applies to an '
                "output_error fit; the equation-error fit is a direct "
                "OLS/least-squares solve with no NLP to hand to a solver. "
                'Set fit_objective="output_error" or fit_solver="scipy".',
                field="fit_solver",
                value=fit_solver,
            )
        if fit_solver == "ipopt" and forecast_horizon != "auto":
            raise FlexConfigError(
                'ArimaRegressor fit_solver="ipopt" builds one Pyomo model '
                "over the whole training series, so it cannot window the "
                "free run and forecast_horizon does not apply. Drop "
                'forecast_horizon or use fit_solver="scipy".',
                field="forecast_horizon",
                value=forecast_horizon,
            )

        self._fit_solver = fit_solver
        self._fit_objective = fit_objective
        # What the caller asked for (None for "auto"), and what the latest
        # fit used; "auto" is re-resolved on every fit.
        self._requested_horizon = (
            None if forecast_horizon == "auto" else int(forecast_horizon)
        )
        self._forecast_horizon = self._requested_horizon
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

    def _require_fit(self, action: str) -> None:
        """Raise unless :meth:`fit` has succeeded.

        Args:
            action: What the caller tried, for the message.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if not self._fitted:
            raise FlexDataError(
                f"ArimaRegressor has no fit yet; call fit(X, y) before {action}."
            )

    def _resolve_horizon(self, n_rows: int, terms: ArimaTerms) -> int:
        """Resolve and remember the free-run window length for this fit.

        An explicit ``forecast_horizon`` is used verbatim; ``"auto"`` is
        resolved from this fit's rows and order, so a refit on different data
        or an auto-selected order gets a matching horizon.

        Args:
            n_rows: Number of rows the fit is using.
            terms: The model structure.

        Returns:
            The resolved window length in steps.
        """
        self._forecast_horizon = self._requested_horizon or auto_forecast_horizon(
            n_rows, terms
        )
        return self._forecast_horizon

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
        for order selection only; the winning order is then refitted.

        Rows with any null value across ``X``/``y`` are dropped before fitting.

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
            FlexConfigError: If scipy is not installed; if ``auto`` is
                ``False`` and no ``order`` was supplied; if ``input_units``
                is missing an entry for one of ``X``'s columns; if ``auto``
                selects an unsupported order; if the fitted AR block is not
                stationary; or if an ``output_error`` fit's MA block is not
                invertible.
            FlexDataError: If ``y`` does not hold exactly one column, if no
                rows survive dropping nulls, or if fewer than
                ``max(p, q) + d + k + 1`` rows survive (where ``k`` is the
                number of fitted parameters), which would leave the fit
                rank-deficient. With ``auto=True`` only one row is required,
                because the order is not yet known.
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

        input_units = {} if input_units is None else input_units
        missing = [name for name in X.columns if name not in input_units]
        if missing:
            raise FlexConfigError(
                f"fit is missing input_units for {missing}; every input "
                f"column ({list(X.columns)}) needs an entry.",
                field="input_units",
                value=missing,
            )
        self.input_units = dict(input_units)
        self.output_units = "" if output_units is None else output_units

        output = _single_column(y, "y")
        exogenous = _exog_columns(X)
        paired = pd.concat([output.rename("__y__"), exogenous], axis=1).dropna()
        if paired.empty:
            raise FlexDataError(
                "ArimaRegressor has no usable rows after dropping nulls "
                f"(of {len(output)} rows). Supply non-null data.",
                field="y",
            )

        n_exog = exogenous.shape[1]
        if not self._auto:
            p, d, q = self._order
            k_params = (1 if self._include_mean else 0) + p + q + n_exog
            # Rows consumed by lags/differencing, plus enough equations to
            # identify every parameter with a degree of freedom to spare.
            # Without this, e.g. order=(4, 0, 0) on 5 rows "fits" a
            # rank-deficient problem and returns meaningless coefficients.
            min_rows = max(p, q) + d + k_params + 1
            if len(paired) < min_rows:
                raise FlexDataError(
                    f"ArimaRegressor needs at least {min_rows} row(s) to fit "
                    f"order={self._order} ({k_params} parameter(s)); only "
                    f"{len(paired)} survived dropping nulls.",
                    field="y",
                )

        self.n_samples = len(paired)
        self.data_window = (paired.index.min(), paired.index.max())
        self.exogenous_variables = list(exogenous.columns)
        self.output_variable = str(output.name)
        self._training_index = paired.index
        y_values = paired["__y__"].to_numpy(dtype=float)
        x_values = (
            paired[self.exogenous_variables].to_numpy(dtype=float) if n_exog else None
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self._auto:
                self._order = self._auto_order(y_values, x_values)
            p, d, q = self._order
            terms = ArimaTerms(
                p=p,
                d=d,
                q=q,
                n_exog=n_exog,
                has_const=self._include_mean and d == 0,
                has_drift=d == 1 and (self._include_mean or self._include_drift),
            )
            refine = None
            if self._fit_solver == "ipopt":
                refine = partial(
                    fit_output_error_ipopt,
                    terms=terms,
                    y=y_values,
                    x=x_values,
                    exog_names=self.exogenous_variables,
                    input_units=self.input_units,
                    output_name=self.output_variable,
                    output_units=self.output_units,
                    training_index=self._training_index,
                    max_ar_persistence=self._max_ar_persistence,
                )
            self.model = fit_direct(
                y_values,
                x_values,
                terms,
                self.exogenous_variables,
                fit_objective=self._fit_objective,
                forecast_horizon=self._resolve_horizon(len(paired), terms),
                max_ar_persistence=self._max_ar_persistence,
                refine=refine,
            )
        self._fitted = True
        # Only the trivial seasonal order can be fit, so none survives it.
        self._seasonal_order = None
        self._check_stability()

        self.metrics = {
            "aic": float(self.model.aic),
            "rmse": self.model.rmse,
            "free_run_rmse": self.model.free_run_rmse,
        }
        return self

    def _auto_order(self, y_values, x_values) -> tuple[int, int, int]:
        """Select an order with AutoARIMA and check the backend can fit it.

        Args:
            y_values: Output series.
            x_values: Exogenous matrix, or ``None``.

        Returns:
            The selected ``(p, d, q)``.

        Raises:
            FlexConfigError: If AutoARIMA selects seasonal AR/MA terms or
                ``d > 1``.
        """
        order, seasonal_order = auto_select_order(
            y_values,
            x_values,
            seasonal=self._seasonal_order is not None,
            stationary=self._stationary,
            auto_kwargs=self._auto_kwargs,
        )
        if seasonal_order is not None:
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
        if order[1] > 1:
            raise FlexConfigError(
                f"ArimaRegressor auto=True selected d={order[1]}, but the "
                f"direct-fit backend and Pyomo surrogate only support "
                f"d=0 or d=1. Retry with auto_kwargs (e.g. max_d=1).",
                field="order",
                value=order,
            )
        return order

    def _check_stability(self) -> None:
        """Reject a fit whose forecasts would diverge; warn on weaker cases.

        Raises:
            FlexConfigError: If the AR block is not stationary, or an
                ``output_error`` fit's MA block is not invertible.
        """
        _c, ar, ma, _beta = self.model.terms.unpack(self.model.params)
        ar_root = ar_max_root(ar)
        if ar_root >= 1.0:
            raise FlexConfigError(
                f"ArimaRegressor fitted an AR block that is not stationary for "
                f"order={self._order}: its largest root magnitude is "
                f"{ar_root:.4f}. Forecasts from it diverge, both here and in "
                f"the Pyomo surrogate. Bounding each coefficient by "
                f"max_ar_persistence does not prevent this; use a lower AR "
                f"order or a smaller max_ar_persistence.",
                field="ar_coefs",
                value=ar_root,
            )
        ma_root = ma_max_root(ma)
        if ma_root >= 1.0 and self._fit_objective == "output_error":
            raise FlexConfigError(
                f"ArimaRegressor fitted a non-invertible MA block for "
                f"order={self._order} under fit_objective='output_error': its "
                f"largest root magnitude is {ma_root:.4f}. The forecast "
                f"seeds its MA lags from one-step residuals, which such a block "
                f"makes explode. Use a lower q or "
                f'fit_objective="equation_error".',
                field="ma_coefs",
                value=ma_root,
            )
        if ma_root >= 1.0:
            _log.warning(
                "ArimaRegressor fitted a non-invertible MA block for "
                "order=%s: largest MA root magnitude is %.4f. Forecasts are "
                "unaffected, since they seed from the residuals this fit "
                "minimized, but the reported one-step statistics (aic, bic, "
                "sigma2) and an in-Pyomo regression with free innovations are "
                "unreliable.",
                self._order,
                ma_root,
            )

    @property
    def model_(self) -> dict[str, object]:
        """A dict view of the fitted model's key attributes.

        Provides ``"coef"``, ``"residuals"``, ``"aic"``, ``"bic"``,
        ``"aicc"``, ``"loglik"``, and ``"sigma2"`` keys. ``"residuals"`` is
        the full-length array, whose leading ``max(p, q)`` entries are zero
        padding; every derived statistic excludes that padding. ``"aicc"``
        is ``None`` when too few effective observations remain to define it.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        self._require_fit("accessing model attributes")
        m = self.model
        return {
            "coef": self.coefficients,
            "residuals": m.resid,
            "aic": float(m.aic),
            "bic": float(m.bic),
            "aicc": m.aicc,
            "loglik": float(m.llf),
            "sigma2": m.sigma2,
        }

    @property
    def coefficients(self) -> dict[str, float] | None:
        """The fitted parameters as a name -> value mapping.

        Names are ``const`` or ``drift``, ``ar{j}``, ``ma{j}``, then the
        exogenous column names.

        Returns:
            The parameter map, or ``None`` before :meth:`fit` is called.
        """
        if not self._fitted:
            return None
        return {
            name: float(value)
            for name, value in zip(
                self.model.param_names, self.model.params, strict=True
            )
        }

    @property
    def order(self) -> tuple[int, int, int] | None:
        """The fitted ``(p, d, q)`` order, or ``None`` before :meth:`fit`."""
        return self._order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int] | None:
        """The requested seasonal order; ``None`` once fitted."""
        return self._seasonal_order

    @property
    def fit_objective(self) -> str:
        """The residual this fit minimizes.

        ``"equation_error"`` (the default) minimizes one-step-ahead residuals
        using actual lagged values. ``"output_error"`` minimizes windowed
        free-run error, the criterion the Pyomo surrogate exercises when it
        simulates forward with its innovations fixed at zero.
        """
        return self._fit_objective

    @property
    def fit_solver(self) -> str:
        """The optimizer behind an ``output_error`` fit.

        ``"scipy"`` (the default) uses ``scipy.optimize.least_squares`` on
        the windowed free-run residual. ``"ipopt"`` instead builds the real
        :class:`~flexops.surrogates.arima.ArimaSurrogate` over the training
        series and minimizes the squared output error through it.
        Irrelevant to an ``equation_error`` fit.
        """
        return self._fit_solver

    @property
    def forecast_horizon(self) -> int | None:
        """The free-run window length in steps.

        An explicit horizon is readable immediately; ``"auto"`` resolves
        during :meth:`fit`, so this is ``None`` until then.
        """
        return self._forecast_horizon

    @property
    def fitted(self) -> bool:
        """``True`` once :meth:`fit` has succeeded."""
        return self._fitted

    def to_fit_result(self) -> FitResult:
        """Return this fit as the shared :class:`~.base.FitResult` shape.

        The coefficient map contains:

        - ``"const"`` (``d=0``) or ``"drift"`` (``d=1``) when the model has
          a deterministic term, and neither when it does not,
        - ``"ar.L{j}"`` for each AR lag ``j``,
        - ``"ma.L{j}"`` for each MA lag ``j``,
        - exogenous coefficients keyed by their column name.

        Returns:
            A :class:`~flexparameterize.regression.base.FitResult` carrying
            the model's coefficients, metrics, sample count, and data window.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        self._require_fit("to_fit_result()")
        terms = self.model.terms
        c, ar, ma, beta = terms.unpack(self.model.params)
        coefficients = {
            **{f"ar.L{j}": float(v) for j, v in enumerate(ar, 1)},
            **{f"ma.L{j}": float(v) for j, v in enumerate(ma, 1)},
            **{
                name: float(v)
                for name, v in zip(self.exogenous_variables, beta, strict=True)
            },
        }
        if terms.n_deterministic:
            coefficients["const" if terms.has_const else "drift"] = c
        return FitResult(
            coefficients=coefficients,
            metrics=dict(self.metrics),
            n_samples=self.n_samples,
            data_window=self.data_window,
        )

    def to_surrogate_spec(self) -> SurrogateSpec:
        """Return the fit as a persistable ``arima`` ``SurrogateSpec``.

        Uses the ``input_units``/``output_units`` recorded by :meth:`fit`.
        The ``data`` field matches the contract expected by
        :class:`~flexops.surrogates.arima.ArimaSurrogate`:

        - ``input_variables``: all fitted exogenous variable names and units.
        - ``output_variables``: the output variable name and its units.
        - ``coefficients``: ``order``; ``intercept`` (``d=0``) or ``drift``
          (``d=1``) when the fit has a deterministic term, omitted entirely
          when it does not; and ``ar_coefs``, ``ma_coefs``, ``exog_coefs``
          in lag/column order for each non-empty block.
        - ``history``: ``start_date``, ``time_step_seconds``, and the
          disturbance/innovation series the surrogate replays to seed its
          pre-horizon lags. No true pre-sample data exists, so the first
          ``max(p + d, q)`` entries repeat the start of the series.

        Returns:
            A :class:`~flexcore.config.schema.SurrogateSpec` of type
            ``SurrogateType.ARIMA``.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
            FlexConfigError: If ``input_units`` is missing an entry for a
                fitted exogenous column.
        """
        self._require_fit("to_surrogate_spec()")
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
        return SurrogateSpec(
            surrogate_type=SurrogateType.ARIMA,
            data={
                "input_variables": {
                    name: self.input_units[name] for name in self.exogenous_variables
                },
                "output_variables": {self.output_variable: self.output_units},
                "coefficients": surrogate_coefficients(
                    self.model.params, self.model.terms
                ),
                "history": history_payload(
                    self.model.eta,
                    self.model.resid,
                    self.model.terms,
                    start_date=self._training_index[0].isoformat(),
                    time_step_seconds=step_seconds(self._training_index),
                ),
            },
        )

    def fit_diagnostics(self) -> dict[str, float | None]:
        """Return extended fit statistics.

        All of them exclude the zero-padded leading lags, so ``n_samples``
        here is the number of observations the mean equation defines, which
        is ``max(p, q) + d`` fewer than the regressor's ``n_samples``.
        ``ma_max_root`` is the largest MA root magnitude: below 1.0 the
        fitted MA block is invertible.

        Returns:
            Mapping of diagnostic name to value. ``aicc`` is ``None`` when
            too few effective observations remain to define it.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        self._require_fit("fit_diagnostics()")
        m = self.model
        return {
            "aic": float(m.aic),
            "bic": float(m.bic),
            "aicc": m.aicc,
            "log_likelihood": float(m.llf),
            "n_parameters": float(m.terms.n_params),
            "n_samples": float(m.nobs_effective),
            "sigma2": m.sigma2,
            "rmse": m.rmse,
            "free_run_rmse": m.free_run_rmse,
            "forecast_horizon": float(m.forecast_horizon),
            "ma_max_root": ma_max_root(m.terms.unpack(m.params)[2]),
        }


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


def _exog_columns(X: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """Return exogenous columns as a DataFrame (empty when no columns).

    Args:
        X: Zero or more exogenous input columns.

    Returns:
        A ``DataFrame`` (possibly empty) with the same index as ``X``.
    """
    if isinstance(X, pd.Series):
        return X.to_frame()
    return X if X.shape[1] > 0 else pd.DataFrame(index=X.index)
