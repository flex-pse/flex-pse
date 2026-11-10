"""ArimaRegressor: time-series ARIMA regressor with exogenous inputs.

Fits regression with ARIMA errors by directly minimizing the mean-equation
residual with ``scipy.optimize.least_squares``, and reduces the result to the
standard :class:`~flexparameterize.regression.base.FitResult` /
``SurrogateSpec`` shape, so every downstream consumer (provenance logging,
``emit_model_config``, ``apply_to_model``) works without change. The fitting
and forecasting machinery lives in
:mod:`flexparameterize.regression.utils.arima_utils`.

With ``auto=True`` the order is chosen by an in-house search (KPSS and ADF
tests for ``d``, then an AIC grid over ``(p, q)``); the final parameter fit
always uses the direct least-squares backend, which minimizes the exact
residual the Pyomo surrogate implements.

**Restriction**: ``d`` may be ``0`` or ``1``, and the model is
non-seasonal. The Pyomo surrogate implements the mean ARIMA equation in closed
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
    spec  = regressor.to_surrogate_spec()  # last week of history
    forecast_spec = regressor.to_surrogate_spec(history_length=0)

:class:`ArimaRegressor` attributes:
    model: The fitted
        :class:`~flexparameterize.regression.utils.arima_utils.DirectResults`
        (or ``None`` before :meth:`fit`).
    n_samples: Number of rows the fit used.
    metrics: ``{"aic", "rmse", "free_run_rmse"}`` of the fitted model.
    data_window: ``(first, last)`` index value of the rows used.
    exogenous_variables: Column names of the fitted exogenous inputs.
    output_variable: Column name of the fitted output.
    input_units: Units of every fitted exogenous column, keyed by column
        name.  Set by :meth:`fit`.
    output_units: Units of the fitted output column.  Set by :meth:`fit`.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexcore.logger import get_logger
from flexparameterize.regression.base import (
    FitResult,
    require_frame,
    single_column,
)
from flexparameterize.regression.utils.arima_utils import (
    ArimaTerms,
    ar_max_root,
    auto_search_options,
    auto_select_order,
    fit_direct,
    history_payload,
    history_steps,
    horizon_steps,
    ma_max_root,
    parse_forecast_horizon,
    step_seconds,
    surrogate_coefficients,
)

_log = get_logger(__name__)


class ArimaRegressor:
    """Fit regression with ARIMA errors and expose the shared fit protocol.

    Either supply an explicit ``order``, or set ``auto=True`` to search for
    one and refit the winning order with the direct least-squares backend.

    **Restriction**: ``d`` may be ``0`` or ``1`` (the Pyomo surrogate's mean
    ARIMA equation supports at most a single order of differencing), and
    the model is non-seasonal.

    **Fitting backend**: ``scipy.optimize.least_squares`` minimizes the
    mean-equation residual directly. This is the exact objective the Pyomo
    surrogate implements, so fitted parameters reproduce the surrogate
    one-to-one.

    **Which beta is kept**: the exogenous coefficients are estimated several
    times: by OLS for the ``auto`` differencing test and inside every
    ``auto`` candidate fit, by OLS as the warm start, jointly with the ARIMA
    terms in the one-step fit, and, when ``forecast_horizon > 1``, again by
    the free-run refinement. Only the last of these is kept. It is the beta
    in :attr:`coefficients` and in the spec's ``exog_coefs``, and the one
    the spec's history ``eta = y - X @ beta`` is computed with.

    **Stability**: AR coefficients are bounded to
    ``[-max_ar_persistence, max_ar_persistence]`` (default 0.85) during the
    fit itself, and :meth:`fit` rejects a fitted AR block that is not
    stationary, which a per-coefficient bound alone does not prevent. The
    surrogate implements the mean equation as a difference equation, so a
    non-stationary AR block makes every forecast and optimization diverge.

    Args:
        order: ``(p, d, q)`` ARIMA order; required when ``auto`` is ``False``.
            Must satisfy ``d in (0, 1)``.
        include_mean: Whether to include the model's deterministic term.
            This is a level intercept for ``d=0`` and constant drift for
            ``d=1``. Default ``True``.
        auto: If ``True``, search for the order, then refit it at the
            configured ``forecast_horizon``. ``d`` comes from KPSS (null:
            stationary) and ADF (null: unit root) tests at 5% on the
            disturbance ``y - X @ beta``: 1 when only KPSS rejects, 0 when
            only ADF rejects, and an error when both or neither do, which
            passing ``d`` in ``auto_kwargs`` avoids. Every ``(p, q)`` within
            the limits is then fitted by one-step least squares and the
            lowest-AIC stationary, invertible candidate wins. Use
            ``auto_kwargs`` to restrict it.
        max_ar_persistence: Bounds every AR coefficient to
            ``[-max_ar_persistence, max_ar_persistence]`` during the fit
            itself (via bounded least squares). Default ``0.85``. Set to
            ``None`` to leave the coefficients unbounded; the stationarity
            check still applies.
        forecast_horizon: Length of the free-run windows whose error the
            fit minimizes: each window is seeded from actual data, then
            simulated forward with innovations at zero, as the surrogate
            forecasts. Either an int number of steps, or a duration (``"2D"``,
            ``"48h"``, ``pd.Timedelta(days=2)``) that :meth:`fit` rounds up
            to whole steps of the training index. ``1`` (default) is the
            one-step (equation-error) fit using actual lagged values, as is
            any duration of one step or less. A longer horizon refines that
            fit against multi-step error, bounds the MA coefficients to
            ``[-0.99, 0.99]``, and rejects a non-invertible MA block. Set it
            to the horizon you will forecast over; very long windows make
            the objective flat and slow to solve.
        **auto_kwargs: Order-search limits, only with ``auto=True``:
            ``max_p`` and ``max_q`` (default 5), ``max_order`` (largest
            ``p + q``, default 5), ``max_d`` (0 or 1, default 1), ``d``
            (fix the differencing order instead of testing for it), and
            ``stationary`` (``True`` forces ``d=0``).

    Raises:
        FlexConfigError: If neither ``order`` nor ``auto=True`` is given; if
            ``d > 1``; if ``auto_kwargs`` is given without
            ``auto=True``, or holds an unknown or out-of-range option; if
            ``max_ar_persistence`` is not a number in ``(0, 1]`` or ``None``;
            or if ``forecast_horizon`` is neither a positive int nor a
            positive duration.
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        include_mean: bool = True,
        auto: bool = False,
        max_ar_persistence: float | None = 0.85,
        forecast_horizon: int | str | timedelta = 1,
        **auto_kwargs: object,
    ) -> None:
        if order is not None and order[1] != 0:
            if order[1] != 1:
                raise FlexConfigError(
                    f"ArimaRegressor only supports d=0 or d=1. "
                    f"Got order={order} with d={order[1]}. "
                    f"Use order=(p, 0, q) or order=(p, 1, q).",
                    field="order",
                    value=order,
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
        forecast_horizon = parse_forecast_horizon(forecast_horizon)
        if auto_kwargs and not auto:
            raise FlexConfigError(
                f"ArimaRegressor got unexpected keyword argument(s) "
                f"{sorted(auto_kwargs)}. Extra keywords configure the order "
                "search, so they only apply with auto=True.",
                field="auto_kwargs",
                value=sorted(auto_kwargs),
            )
        search_options = auto_search_options(auto_kwargs) if auto else {}
        if not auto and order is None:
            raise FlexConfigError(
                "ArimaRegressor.fit requires either `order=(p,d,q)` or "
                "`auto=True`. Pass both or set `auto=True`."
            )
        self._forecast_horizon = forecast_horizon
        self._order = order
        self._include_mean = include_mean
        self._auto = auto
        self._search_options = search_options
        self._max_ar_persistence = max_ar_persistence

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
        surrogate.  When ``auto=True``, the order is searched for first (see
        the ``auto`` argument), then refitted.  Of the several ``beta``
        estimates made along the way, only the last is kept (see the class
        docstring).

        ``X`` and ``y`` must be clean: no null values, and a shared
        ``DatetimeIndex`` that strictly increases at one constant step. The
        fitted lags and the surrogate's dated history both assume that
        grid, so gaps must be imputed (or the data split) before fitting.

        Args:
            X: Zero or more exogenous-input columns. Pass an empty
                ``DataFrame`` for a pure ARIMA fit.
            y: A one-column ``DataFrame`` holding the output.
            input_units: Units of every fitted exogenous column, keyed by its
                column name.  Defaults to ``{}``.  Recorded for
                :meth:`to_surrogate_spec`.
            output_units: Units of the fitted output column.  Defaults to
                ``""``.  Recorded for :meth:`to_surrogate_spec`.

        Returns:
            ``self``, fitted.

        Raises:
            FlexConfigError: If ``auto`` must test for ``d`` and statsmodels
                (the ``[parameterize]`` extra) is not installed; if
                ``input_units`` is missing an entry for one of ``X``'s
                columns; if ``auto`` finds no stationary, invertible
                candidate; if the fitted AR
                block is not stationary; or if a ``forecast_horizon > 1``
                fit's MA block is not invertible.
            FlexDataError: If ``X`` or ``y`` is not a ``DataFrame``; if
                ``y`` does not hold exactly one column; if
                ``X`` and ``y`` lack a ``DatetimeIndex``, hold a null value,
                or are not evenly spaced in time; if there are no rows; or if
                there are fewer than ``max(p, q) + d + k + 1`` rows (where
                ``k`` is the number of fitted parameters), which would leave
                the fit rank-deficient. With ``auto=True``, candidate orders
                needing more rows are skipped, and the error is raised only
                if none fits. Also if ``auto`` must choose ``d`` and the KPSS
                and ADF tests both reject or both fail to reject, or the
                series is too short to run ADF.
        """
        require_frame(X, regressor="ArimaRegressor", role="X")
        output = single_column(y, regressor="ArimaRegressor", role="y", what="output")
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

        paired = pd.concat([output.rename("__y__"), X], axis=1)
        _require_clean_series(paired)

        n_exog = X.shape[1]
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
                    f"order={self._order} ({k_params} parameter(s)); got "
                    f"{len(paired)}.",
                    field="y",
                )

        self.n_samples = len(paired)
        self.data_window = (paired.index.min(), paired.index.max())
        self.exogenous_variables = list(X.columns)
        self.output_variable = str(output.name)
        self._training_index = paired.index
        y_values = paired["__y__"].to_numpy(dtype=float)
        x_values = (
            paired[self.exogenous_variables].to_numpy(dtype=float) if n_exog else None
        )

        if self._auto:
            self._order = auto_select_order(
                y_values,
                x_values,
                include_mean=self._include_mean,
                max_ar_persistence=self._max_ar_persistence,
                **self._search_options,
            )
        p, d, q = self._order
        terms = ArimaTerms(
            p=p,
            d=d,
            q=q,
            n_exog=n_exog,
            has_const=self._include_mean and d == 0,
            has_drift=d == 1 and self._include_mean,
        )
        # fit_direct fits once: a one-step warm start, then (for a horizon
        # above 1) a refinement, then DirectResults, which only evaluates
        # the final coefficients.
        self.model = fit_direct(
            y_values,
            x_values,
            terms,
            self.exogenous_variables,
            forecast_horizon=horizon_steps(
                self._forecast_horizon, step_seconds(self._training_index)
            ),
            max_ar_persistence=self._max_ar_persistence,
        )
        self._fitted = True
        self._check_stability()

        self.metrics = {
            "aic": float(self.model.aic),
            "rmse": self.model.rmse,
            "free_run_rmse": self.model.free_run_rmse,
        }
        return self

    def _check_stability(self) -> None:
        """Reject a fit whose forecasts would diverge; warn on weaker cases.

        Raises:
            FlexConfigError: If the AR block is not stationary, or a
                ``forecast_horizon > 1`` fit's MA block is not invertible.
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
        if ma_root >= 1.0 and self.model.forecast_horizon > 1:
            raise FlexConfigError(
                f"ArimaRegressor fitted a non-invertible MA block for "
                f"order={self._order} at forecast_horizon="
                f"{self.model.forecast_horizon} steps: its largest root magnitude is "
                f"{ma_root:.4f}. The forecast seeds its MA lags from one-step "
                f"residuals, which such a block makes explode. Use a lower q "
                f"or forecast_horizon=1.",
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
    def coefficients(self) -> dict[str, float] | None:
        """The fitted parameters as a name -> value mapping.

        Names match the coefficient registry of the
        :class:`~flexops.surrogates.arima.ArimaSurrogate` built from
        :meth:`to_surrogate_spec`: ``intercept`` (``d=0``) or ``drift``
        (``d=1``) when the model has a deterministic term, ``ar.L{j}``,
        ``ma.L{j}``, then ``exog.<column>`` for each exogenous input.

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
    def forecast_horizon(self) -> int | pd.Timedelta:
        """The requested free-run window, as an int number of steps or a
        ``pd.Timedelta``.

        A duration is converted to steps on each :meth:`fit`, from that fit's
        time step; the steps used are ``fit_diagnostics()["forecast_horizon"]``.
        ``1`` is the one-step (equation-error) fit.
        """
        return self._forecast_horizon

    @property
    def fitted(self) -> bool:
        """``True`` once :meth:`fit` has succeeded."""
        return self._fitted

    def to_fit_result(self) -> FitResult:
        """Return this fit as the shared :class:`~.base.FitResult` shape.

        The coefficient map is :attr:`coefficients`, holding the last
        ``beta`` estimated (see the class docstring).

        Returns:
            A :class:`~flexparameterize.regression.base.FitResult` carrying
            the model's coefficients, metrics, sample count, and data window.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        self._require_fit("to_fit_result()")
        return FitResult(
            coefficients=self.coefficients,
            metrics=dict(self.metrics),
            n_samples=self.n_samples,
            data_window=self.data_window,
        )

    def to_surrogate_spec(self, history_length="7D") -> SurrogateSpec:
        """Return the fit as a persistable ``arima`` ``SurrogateSpec``.

        Uses the ``input_units``/``output_units`` recorded by :meth:`fit`.
        The ``data`` field matches the contract expected by
        :class:`~flexops.surrogates.arima.ArimaSurrogate`:

        - ``input_variables``: all fitted exogenous variable names and units.
        - ``output_variables``: the output variable name and its units.
        - ``coefficients``: ``order``; ``intercept`` (``d=0``) or ``drift``
          (``d=1``) when the fit has a deterministic term, omitted entirely
          when it does not; and ``ar_coefs``, ``ma_coefs``, ``exog_coefs``
          in lag/column order for each non-empty block. ``exog_coefs`` is
          the last ``beta`` the fit estimated (see the class docstring).
        - ``history``: ``start_date``, ``time_step_seconds``, and the
          disturbance/innovation series the surrogate replays to seed its
          pre-horizon lags, covering the last ``history_length`` of the
          training data plus the ``max(p + d, q)`` lags ahead of it. When the
          window reaches the start of the series, where no true pre-sample
          data exists, those lags repeat the start of the series.

        A surrogate built from the spec can start its horizon anywhere from
        the first kept point onward; a start after the training data is
        forecast from the terminal state with zero innovations.

        Args:
            history_length: How much of the training history to keep: a
                duration (``"7D"``, ``pd.Timedelta(days=7)``), ``0`` to keep
                only the terminal state for forecast-only use, or ``None``
                for the full training series. Rounded up to whole steps and
                capped at the training length.

        Returns:
            A :class:`~flexcore.config.schema.SurrogateSpec` of type
            ``SurrogateType.ARIMA``.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
            FlexConfigError: If ``input_units`` is missing an entry for a
                fitted exogenous column, or ``history_length`` is invalid.
        """
        self._require_fit("to_surrogate_spec()")
        time_step_seconds = step_seconds(self._training_index)
        steps = history_steps(history_length, time_step_seconds)
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
                    time_step_seconds=time_step_seconds,
                    history_steps=steps,
                ),
            },
        )

    def fit_diagnostics(self) -> dict[str, float]:
        """Return extended fit statistics.

        All of them exclude the zero-padded leading lags, so ``n_samples``
        here is the number of observations the mean equation defines, which
        is ``max(p, q) + d`` fewer than the regressor's ``n_samples``.
        ``ma_max_root`` is the largest MA root magnitude: below 1.0 the
        fitted MA block is invertible.

        Returns:
            Mapping of diagnostic name to value.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        self._require_fit("fit_diagnostics()")
        m = self.model
        return {
            "aic": float(m.aic),
            "bic": float(m.bic),
            "log_likelihood": float(m.llf),
            "n_parameters": float(m.terms.n_params),
            "n_samples": float(m.nobs_effective),
            "sigma2": m.sigma2,
            "rmse": m.rmse,
            "free_run_rmse": m.free_run_rmse,
            "forecast_horizon": float(m.forecast_horizon),
            "ma_max_root": ma_max_root(m.terms.unpack(m.params)[2]),
        }


def _require_clean_series(paired: pd.DataFrame) -> None:
    """Require a non-empty, null-free frame on an evenly spaced DatetimeIndex.

    Args:
        paired: ``y`` joined column-wise with the exogenous inputs. A row
            present in only one of them shows up here as a null.

    Raises:
        FlexDataError: If the index is not a ``DatetimeIndex``, the frame is
            empty or holds a null value, or consecutive timestamps are not
            separated by one constant, positive step.
    """
    index = paired.index
    if not isinstance(index, pd.DatetimeIndex):
        raise FlexDataError(
            "ArimaRegressor.fit needs X and y indexed by a DatetimeIndex, "
            "so the fitted surrogate's history can be dated against a "
            f"TimeBlock; got {type(index).__name__}.",
            field="y",
        )
    if paired.empty:
        raise FlexDataError("ArimaRegressor.fit got no rows.", field="y")
    null_rows = index[paired.isna().any(axis=1)]
    if len(null_rows):
        columns = [
            "y" if name == "__y__" else name
            for name in paired.columns[paired.isna().any()]
        ]
        raise FlexDataError(
            f"ArimaRegressor.fit needs clean data, but {len(null_rows)} row(s) "
            f"hold a null value (first at {null_rows[0]}, in column(s) "
            f"{columns}). Impute or trim the gaps, and make X and y share "
            "the same index, before fitting.",
            field="y",
        )
    if len(index) > 1:
        steps = index[1:] - index[:-1]
        irregular = steps != steps[0]
        if steps[0] <= pd.Timedelta(0) or irregular.any():
            at = index[1:][irregular][0] if irregular.any() else index[1]
            raise FlexDataError(
                "ArimaRegressor.fit needs evenly spaced, increasing timestamps; "
                f"the first step is {steps[0]}, but the spacing breaks at {at}. "
                "Resample onto a regular grid and impute the gaps before "
                "fitting.",
                field="y",
            )
