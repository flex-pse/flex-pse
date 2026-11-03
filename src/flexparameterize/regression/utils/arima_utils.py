"""Fitting and forecasting machinery behind :class:`ArimaRegressor`.

The model is regression with ARIMA errors, the same equation
:class:`~flexops.surrogates.arima.ArimaSurrogate` builds in Pyomo::

    eta[t] = y[t] - X[t] @ beta                        (disturbance)
    z[t]   = eta[t]  (d=0)   or   eta[t] - eta[t-1]  (d=1)
    z[t]   = c + sum(ar_j * z[t-j]) + sum(ma_j * eps[t-j]) + eps[t]

Every function here takes one flat parameter vector ``theta`` laid out as
``[c?, ar_1..ar_p, ma_1..ma_q, beta_1..beta_k]``, with the layout described
by an :class:`ArimaTerms`. The fit minimizes one criterion, set by the
forecast horizon ``H``: the error of free runs over consecutive ``H``-step
windows, each seeded from actual data, that feed their own predictions back
as lags with innovations held at zero, which is what the surrogate does when
it forecasts (:func:`output_error_residuals`). At ``H=1`` that is exactly
the one-step innovations ``eps`` (equation error), computed by the
vectorized :func:`equation_error_residuals`.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfiltic

from flexcore.exceptions import FlexConfigError, FlexDataError

# Bound applied to every MA coefficient in a fit with forecast_horizon > 1.
# The free-run objective barely sees the MA block (it only reaches
# the first q steps of each window), so unbounded it trades MA against the
# bounded AR block and drives the MA roots outside the unit circle. The
# forecast seeds its MA lags from the one-step residuals, which such a block
# makes explode: measured first-step forecasts of ~1e6 on biogas data.
MA_BOUND = 0.99


@dataclass(frozen=True)
class ArimaTerms:
    """The structure of a model: its order and which terms it carries.

    Attributes:
        p: Autoregressive order.
        d: Differencing order, 0 or 1.
        q: Moving-average order.
        n_exog: Number of exogenous regressors.
        has_const: Whether ``theta`` opens with a level intercept (``d=0``).
        has_drift: Whether ``theta`` opens with a differenced-equation
            constant (``d=1``). At most one of the two is set.
    """

    p: int
    d: int
    q: int
    n_exog: int
    has_const: bool
    has_drift: bool

    @property
    def n_deterministic(self) -> int:
        """1 when ``theta`` opens with a constant (``const`` or ``drift``)."""
        return int(self.has_const or self.has_drift)

    @property
    def n_params(self) -> int:
        """Length of ``theta``."""
        return self.n_deterministic + self.p + self.q + self.n_exog

    @property
    def seed(self) -> int:
        """Rows a free run consumes as history before its first step."""
        return max(self.p + self.d, self.q)

    def unpack(
        self, theta: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Split ``theta`` into ``(c, ar, ma, beta)``.

        Args:
            theta: Flat parameter vector.

        Returns:
            The blocks; ``c`` is 0.0 when there is no deterministic term.
        """
        k = self.n_deterministic
        c = float(theta[0]) if k else 0.0
        ar = theta[k : k + self.p]
        ma = theta[k + self.p : k + self.p + self.q]
        return c, ar, ma, theta[k + self.p + self.q :]

    def param_names(self, exog_names: list[str]) -> list[str]:
        """Return the names of the entries of ``theta``, in order.

        Args:
            exog_names: Exogenous column names, in fitted order.

        Returns:
            ``intercept``/``drift`` (when present), ``ar.L1..``, ``ma.L1..``,
            then ``exog.<column>``: the names
            :class:`~flexops.surrogates.arima.ArimaSurrogate` registers its
            coefficient Vars under.
        """
        names = ["intercept"] if self.has_const else ["drift"] if self.has_drift else []
        names += [f"ar.L{j}" for j in range(1, self.p + 1)]
        names += [f"ma.L{j}" for j in range(1, self.q + 1)]
        return names + [f"exog.{name}" for name in exog_names]


def ma_max_root(ma_coefs) -> float:
    """Return the largest MA root magnitude, 0.0 when there is no MA block.

    The MA polynomial ``1 + t1*B + ... + tq*B**q`` is invertible exactly when
    this value is below 1.0.

    Args:
        ma_coefs: MA coefficients in lag order.

    Returns:
        The largest root magnitude.
    """
    if len(ma_coefs) == 0:
        return 0.0
    return float(np.abs(np.roots([1.0, *ma_coefs])).max())


def ar_max_root(ar_coefs) -> float:
    """Return the largest AR root magnitude, 0.0 when there is no AR block.

    The AR polynomial ``1 - a1*B - ... - ap*B**p`` is stationary exactly when
    this value is below 1.0. Bounding each coefficient does not imply it:
    ``[0.85, 0.85]`` has a root of ~1.44.

    Args:
        ar_coefs: AR coefficients in lag order.

    Returns:
        The largest root magnitude.
    """
    return ma_max_root([-value for value in ar_coefs])


def one_step_innovations(
    theta: np.ndarray, terms: ArimaTerms, y: np.ndarray, x: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Run the mean equation over the series using actual lagged values.

    Args:
        theta: Flat parameter vector.
        terms: Its layout.
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None`` when ``terms.n_exog`` is 0.

    Returns:
        ``(eta, eps)``: the disturbance on the level scale, and the one-step
        innovations on the differenced series, zero for the first
        ``max(p, q)`` entries where the equation defines none.
    """
    c, ar, ma, beta = terms.unpack(theta)
    eta = y - x @ beta if terms.n_exog else np.asarray(y, dtype=float)
    z = np.diff(eta) if terms.d else eta
    lag = max(terms.p, terms.q)
    ar_part = sum(ar[j] * z[lag - j - 1 : len(z) - j - 1] for j in range(terms.p))
    # eps[t] = w[t] - sum(ma_j * eps[t-j]) is an IIR filter of w, started at zero.
    eps = np.zeros(len(z))
    eps[lag:] = lfilter([1.0], [1.0, *ma], z[lag:] - c - ar_part)
    return eta, eps


def equation_error_residuals(
    theta: np.ndarray, terms: ArimaTerms, y: np.ndarray, x: np.ndarray | None
) -> np.ndarray:
    """The one-step innovations the equation defines, for ``least_squares``.

    Args:
        theta: Flat parameter vector.
        terms: Its layout.
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.

    Returns:
        The innovations from lag ``max(p, q)`` onward.
    """
    return one_step_innovations(theta, terms, y, x)[1][max(terms.p, terms.q) :]


def seed_history(
    terms: ArimaTerms, eta: np.ndarray, eps: np.ndarray, start: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the lags a free run starting at level row ``start`` needs.

    Args:
        terms: The model structure.
        eta: Disturbance series on the level scale.
        eps: One-step innovations on the differenced series.
        start: Level row of the first simulated step, at least ``terms.seed``.

    Returns:
        ``(eta_history, eps_history)``: the last ``p + d`` disturbances and
        the last ``q`` innovations before ``start``, oldest first. Lags that
        precede the first innovation are zero, matching the surrogate.
    """
    p, d, q = terms.p, terms.d, terms.q
    # eps lives on the differenced series: the lag behind `start` is start - d - 1.
    lags = eps[max(start - d - q, 0) : max(start - d, 0)]
    return eta[start - p - d : start], np.concatenate([np.zeros(q - len(lags)), lags])


def free_run(
    theta: np.ndarray,
    terms: ArimaTerms,
    eta_history: np.ndarray,
    eps_history: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Simulate the disturbance forward with future innovations at zero.

    Each prediction feeds back as the next AR lag, exactly as the Pyomo
    surrogate solves forward with ``eps`` fixed at zero.

    Args:
        theta: Flat parameter vector.
        terms: Its layout.
        eta_history: The last ``p + d`` disturbances, oldest first.
        eps_history: The last ``q`` innovations, oldest first.
        steps: Number of steps to simulate.

    Returns:
        The simulated disturbance, on the level scale.
    """
    c, ar, ma, _beta = terms.unpack(theta)
    q = terms.q
    # The constant, plus the MA terms still reached by seeded innovations.
    drive = np.full(steps, c)
    for s in range(min(q, steps)):
        drive[s] += sum(ma[j] * eps_history[q + s - j - 1] for j in range(s, q))
    z = drive
    if terms.p:
        # z[s] = drive[s] + sum(ar_j * z[s-j]), started from the seeded lags.
        z_history = np.diff(eta_history) if terms.d else eta_history
        a = np.concatenate([[1.0], -ar])
        z = lfilter([1.0], a, drive, zi=lfiltic([1.0], a, z_history[::-1]))[0]
    return eta_history[-1] + np.cumsum(z) if terms.d else z


def output_error_residuals(
    theta: np.ndarray,
    terms: ArimaTerms,
    y: np.ndarray,
    x: np.ndarray | None,
    horizon: int,
) -> np.ndarray:
    """Output-error residual: windowed free-run error, for ``least_squares``.

    The series is cut into consecutive windows of ``horizon`` steps. Each
    window is seeded from actual data and then free-runs (:func:`free_run`).
    An error that accumulates over a forecast -- above all the ``d=1`` drift
    -- is charged its accumulated cost here rather than its per-step cost.
    Restarting from data keeps each run grounded and the objective well
    conditioned: on 5757 rows, one window over the whole series left the
    ``d=0`` objective so flat that ``least_squares`` ground to its evaluation
    limit (120-140 s), while 192-step windows took under 8 s, no worse.

    The first window starts where :func:`equation_error_residuals` does, so
    at ``horizon=1`` the two return the same residuals.

    Args:
        theta: Flat parameter vector.
        terms: Its layout.
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.
        horizon: Window length in steps, at least 1.

    Returns:
        Level-scale residuals from row ``max(p, q) + d`` onward. The
        exogenous contribution cancels, so each is ``eta[t]`` less its
        simulation.
    """
    eta, eps = one_step_innovations(theta, terms, y, x)
    residuals = [np.zeros(0)]
    first = max(terms.p, terms.q) + terms.d
    for start in range(first, len(eta), horizon):
        steps = min(horizon, len(eta) - start)
        history = seed_history(terms, eta, eps, start)
        residuals.append(
            eta[start : start + steps] - free_run(theta, terms, *history, steps)
        )
    return np.concatenate(residuals)


def coefficient_bounds(
    terms: ArimaTerms, max_ar_persistence: float | None, ma_bound: float | None = None
) -> np.ndarray:
    """Return the symmetric bound on each entry of ``theta``.

    Args:
        terms: The model structure.
        max_ar_persistence: Bound on every AR coefficient, or ``None``.
        ma_bound: Bound on every MA coefficient, or ``None``.

    Returns:
        ``upper`` such that ``-upper <= theta <= upper``; ``inf`` where free.
    """
    upper = np.full(terms.n_params, np.inf)
    ar = terms.n_deterministic
    if max_ar_persistence is not None:
        upper[ar : ar + terms.p] = max_ar_persistence
    if ma_bound is not None:
        upper[ar + terms.p : ar + terms.p + terms.q] = ma_bound
    return upper


def initial_theta(terms: ArimaTerms, y: np.ndarray, x: np.ndarray | None) -> np.ndarray:
    """Return the OLS starting point for the nonlinear fit.

    Regresses ``y`` on ``X`` (plus the intercept when ``d=0``) for ``beta``,
    then fits the AR block by OLS on the implied disturbance. The MA block
    starts at zero. For a pure AR model this is already the optimum.

    Args:
        terms: The model structure.
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.

    Returns:
        The starting parameter vector.
    """
    p, k = terms.p, terms.n_deterministic
    theta = np.zeros(terms.n_params)
    eta = y
    if terms.n_exog:
        design = np.column_stack([np.ones(len(y)), x]) if terms.has_const else x
        solution = np.linalg.lstsq(design, y, rcond=None)[0]
        theta[k + p + terms.q :] = solution[-terms.n_exog :]
        theta[:k] = solution[:k] if terms.has_const else 0.0
        eta = y - x @ theta[k + p + terms.q :]
    z = np.diff(eta) if terms.d else eta
    if p and len(z) > p:
        # The constant joins the AR regression only when beta did not take it.
        const = [np.ones(len(z) - p)] if k and not terms.n_exog else []
        lags = [z[p - j : len(z) - j] for j in range(1, p + 1)]
        solution = np.linalg.lstsq(np.column_stack(const + lags), z[p:], rcond=None)[0]
        theta[k : k + p] = solution[-p:]
        if const:
            theta[0] = solution[0]
    return theta


def _least_squares(residuals, theta0: np.ndarray, args: tuple, upper: np.ndarray):
    """Minimize ``residuals(theta, *args)`` within ``-upper <= theta <= upper``.

    Uses Levenberg-Marquardt when nothing is bounded, else trust-region
    reflective, starting from ``theta0`` clipped into the bounds.
    """
    bounded = bool(np.isfinite(upper).any())
    return least_squares(
        residuals,
        np.clip(theta0, -upper, upper),
        args=args,
        method="trf" if bounded else "lm",
        bounds=(-upper, upper) if bounded else (-np.inf, np.inf),
        max_nfev=5000,
        ftol=1e-8,
        xtol=1e-8,
    ).x


class DirectResults:
    """A fitted model: its parameters and everything derived from them.

    All statistics are computed from ``params`` and the training data at
    construction, so they always describe exactly the coefficients held.

    Attributes:
        params: Fitted parameter vector, laid out as ``terms`` describes.
        terms: The model structure.
        param_names: Names matching ``params``.
        forecast_horizon: Free-run window length in steps.
        eta: Training disturbance ``y - X @ beta`` on the level scale.
        resid: One-step innovations on the differenced series; the leading
            ``max(p, q)`` entries are zero padding.
        fittedvalues_level: One-step fitted values on the level scale of
            ``y``, ``NaN`` where the equation defines none.
        free_run_resid: Windowed free-run residuals, the error the surrogate
            commits when it simulates forward.

    Note:
        ``llf``, ``aic``, ``bic``, ``sigma2``, and ``rmse`` exclude
        the zero padding, so they are not comparable to statsmodels.
    """

    def __init__(
        self,
        params: np.ndarray,
        terms: ArimaTerms,
        exog_names: list[str],
        y: np.ndarray,
        x: np.ndarray | None,
        forecast_horizon: int,
    ) -> None:
        y = np.asarray(y, dtype=float)
        self.params = np.asarray(params, dtype=float)
        self.terms = terms
        self.param_names = terms.param_names(exog_names)
        self.forecast_horizon = int(forecast_horizon)
        self.eta, self.resid = one_step_innovations(self.params, terms, y, x)
        self.fittedvalues_level = y - np.concatenate([np.zeros(terms.d), self.resid])
        self.fittedvalues_level[: max(terms.p, terms.q) + terms.d] = np.nan
        self.free_run_resid = output_error_residuals(
            self.params, terms, y, x, self.forecast_horizon
        )

    @property
    def effective_resid(self) -> np.ndarray:
        """Innovations with the zero-padded leading lags dropped."""
        return self.resid[max(self.terms.p, self.terms.q) :]

    @property
    def nobs_effective(self) -> int:
        """Number of observations the mean equation defines."""
        return len(self.effective_resid)

    @property
    def sigma2(self) -> float:
        """Innovation variance over :attr:`effective_resid`."""
        n = self.nobs_effective
        return float(np.sum(self.effective_resid**2)) / n if n else 0.0

    @property
    def rmse(self) -> float:
        """One-step root-mean-square error on the level scale."""
        return math.sqrt(self.sigma2)

    @property
    def free_run_rmse(self) -> float:
        """Root-mean-square of :attr:`free_run_resid`."""
        if self.free_run_resid.size == 0:
            return float("nan")
        return float(np.sqrt(np.mean(self.free_run_resid**2)))

    @property
    def llf(self) -> float:
        """Gaussian log-likelihood over :attr:`effective_resid`."""
        n, sigma2 = self.nobs_effective, self.sigma2
        if n == 0 or sigma2 <= 0:
            return -np.inf
        return -n / 2.0 * (np.log(2.0 * np.pi) + np.log(sigma2) + 1.0)

    @property
    def aic(self) -> float:
        """Akaike information criterion, counting the innovation variance."""
        return -2.0 * self.llf + 2.0 * (self.terms.n_params + 1)

    @property
    def bic(self) -> float:
        """Bayesian information criterion, penalized by ``nobs_effective``."""
        if self.nobs_effective == 0:
            return np.inf
        return -2.0 * self.llf + (self.terms.n_params + 1) * np.log(self.nobs_effective)

    def predict(
        self,
        steps: int = 1,
        exog: np.ndarray | None = None,
        start: int | None = None,
    ) -> np.ndarray:
        """Recursive multi-step forecast with future innovations at zero.

        Each predicted value feeds back as the next AR lag, matching the
        Pyomo surrogate's forecast. ``steps=0`` returns an empty array; use
        :attr:`fittedvalues_level` for in-sample one-step values.

        Args:
            steps: Number of steps to forecast.
            exog: Exogenous values over the forecast, shape
                ``(steps, n_exog)``; ``None`` omits their contribution.
            start: Level row of the training data to start an in-sample
                free run from, using the data before it as history. ``None``
                forecasts on from the end of the training data.

        Returns:
            ``steps`` forecasts on the level scale of ``y``.

        Raises:
            FlexDataError: If ``start`` leaves fewer than ``max(p + d, q)``
                rows of history.
        """
        start = len(self.eta) if start is None else start
        if start < self.terms.seed:
            raise FlexDataError(
                f"predict(start={start}) needs at least {self.terms.seed} "
                "rows of history before the first predicted row.",
                field="start",
            )
        history = seed_history(self.terms, self.eta, self.resid, start)
        forecast = free_run(self.params, self.terms, *history, steps)
        if exog is not None and self.terms.n_exog:
            beta = self.terms.unpack(self.params)[3]
            forecast = forecast + np.asarray(exog, dtype=float)[:steps] @ beta
        return forecast


def fit_direct(
    y: np.ndarray,
    x: np.ndarray | None,
    terms: ArimaTerms,
    exog_names: list[str],
    *,
    forecast_horizon: int,
    max_ar_persistence: float | None,
) -> DirectResults:
    """Fit ``theta`` by least squares and wrap it in :class:`DirectResults`.

    Always minimizes the one-step (``horizon=1``) error first, from
    :func:`initial_theta`. A longer ``forecast_horizon`` then refines that
    solution against :func:`output_error_residuals`, which is cheaper and
    better behaved than a cold start on the nonconvex free-run objective.
    Only the refinement bounds MA: a free run reaches the MA block only in
    the first ``q`` steps of each window, so unbounded it drifts outside the
    unit circle (see :data:`MA_BOUND`).

    ``beta`` is estimated at each stage (OLS warm start, joint one-step fit,
    refinement), and only the last stage that runs is kept.

    Args:
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.
        terms: The model structure.
        exog_names: Exogenous column names, in fitted order.
        forecast_horizon: Free-run window length in steps; 1 stops after
            the one-step fit.
        max_ar_persistence: Bound on every AR coefficient, or ``None``.

    Returns:
        The fitted results.
    """
    y = np.asarray(y, dtype=float)
    theta = initial_theta(terms, y, x)
    if theta.size:
        theta = _least_squares(
            equation_error_residuals,
            theta,
            (terms, y, x),
            coefficient_bounds(terms, max_ar_persistence),
        )
    if theta.size and forecast_horizon > 1:
        theta = _least_squares(
            output_error_residuals,
            theta,
            (terms, y, x, forecast_horizon),
            coefficient_bounds(terms, max_ar_persistence, MA_BOUND),
        )
    return DirectResults(theta, terms, exog_names, y, x, forecast_horizon)


def surrogate_coefficients(theta: np.ndarray, terms: ArimaTerms) -> dict[str, object]:
    """Return ``theta`` in the surrogate's ``coefficients`` contract.

    Args:
        theta: Flat parameter vector.
        terms: Its layout.

    Returns:
        ``order``, then ``intercept`` (``d=0``) or ``drift`` (``d=1``) when
        the model has a deterministic term, then ``ar_coefs``, ``ma_coefs``
        and ``exog_coefs`` for each block that is non-empty.
    """
    c, ar, ma, beta = terms.unpack(theta)
    coefficients: dict[str, object] = {"order": [terms.p, terms.d, terms.q]}
    if terms.has_const:
        coefficients["intercept"] = c
    if terms.has_drift:
        coefficients["drift"] = c
    for key, block in (("ar_coefs", ar), ("ma_coefs", ma), ("exog_coefs", beta)):
        if len(block):
            coefficients[key] = [float(value) for value in block]
    return coefficients


def history_payload(
    eta: np.ndarray,
    eps: np.ndarray,
    terms: ArimaTerms,
    start_date: str,
    time_step_seconds: float,
    history_steps: int | None = None,
) -> dict[str, object]:
    """Build the surrogate ``history`` block from a fitted disturbance series.

    The surrogate replays ``max(p + d, q)`` values ahead of its first modeled
    point. No true pre-sample data exists, so the prefix repeats the start of
    the series; both lists share one prefix length so one offset indexes
    either.

    ARIMA state is Markov in those ``max(p + d, q)`` lags, so trimming to the
    last ``history_steps`` modeled points loses nothing for horizons that start
    inside the kept window or after it: the prefix of a trimmed history holds
    real fitted values, and ``start_date`` moves to the first kept point. With
    ``history_steps=0`` only the terminal state is kept, dated one step after
    the last fitted row, so the surrogate can only forecast from there.

    Args:
        eta: Disturbance series ``y - X @ beta`` on the level scale.
        eps: One-step innovations on the differenced series.
        terms: The model structure.
        start_date: ISO-8601 timestamp of the first fitted row.
        time_step_seconds: Sampling step in seconds.
        history_steps: Modeled points to keep at the end of the series;
            ``None`` keeps all of them. Values beyond the series length are
            capped.

    Returns:
        The ``history`` mapping in the surrogate's persisted contract.
    """
    seed, lagged = terms.seed, terms.p + terms.d
    prefix = [float(eta[0])] * (seed - lagged) + [float(v) for v in eta[:lagged]]
    eta_values = prefix + [float(v) for v in eta]
    eps_values = [0.0] * (seed + terms.d) + [float(v) for v in eps]
    n = len(eta)
    dropped = 0 if history_steps is None else n - min(history_steps, n)
    start = pd.Timestamp(start_date) + pd.Timedelta(
        seconds=dropped * float(time_step_seconds)
    )
    return {
        "start_date": start.isoformat(),
        "time_step_seconds": float(time_step_seconds),
        "eta_values": eta_values[dropped:],
        "eps_values": eps_values[dropped:],
    }


def history_steps(history_length, time_step_seconds: float) -> int | None:
    """Convert a ``history_length`` request into a count of modeled steps.

    Args:
        history_length: ``None`` for the full history, ``0`` for the
            terminal state only, or a duration (``pd.Timedelta``,
            ``datetime.timedelta``, or a string such as ``"7D"``).
        time_step_seconds: Sampling step in seconds.

    Returns:
        ``None``, or the number of steps covering ``history_length``,
        rounded up to whole steps.

    Raises:
        FlexConfigError: If ``history_length`` is not one of the accepted
            forms or is negative.
    """
    if history_length is None:
        return None
    if isinstance(history_length, (int, float)) and not isinstance(
        history_length, bool
    ):
        if history_length == 0:
            return 0
        duration = None
    else:
        try:
            duration = pd.Timedelta(history_length)
        except (TypeError, ValueError):
            duration = None
    if duration is None or pd.isna(duration) or duration < pd.Timedelta(0):
        raise FlexConfigError(
            "ArimaRegressor history_length must be None (full history), 0 "
            "(terminal state only), or a non-negative duration such as "
            f'"7D" or pd.Timedelta(days=7); got {history_length!r}.',
            field="history_length",
            value=history_length,
        )
    # Tolerate float error so an exact multiple of the step does not round up.
    return math.ceil(duration.total_seconds() / time_step_seconds - 1e-9)


def parse_forecast_horizon(forecast_horizon) -> int | pd.Timedelta:
    """Validate a ``forecast_horizon`` request without converting it to steps.

    Args:
        forecast_horizon: A positive int number of steps, or a positive
            duration (a string such as ``"2D"`` or ``"48h"``, a
            ``pd.Timedelta``, or a ``datetime.timedelta``).

    Returns:
        The int unchanged, or the duration as a ``pd.Timedelta``.

    Raises:
        FlexConfigError: If it is neither.
    """
    if isinstance(forecast_horizon, int) and not isinstance(forecast_horizon, bool):
        if forecast_horizon >= 1:
            return forecast_horizon
    elif isinstance(forecast_horizon, (str, datetime.timedelta)):
        try:
            duration = pd.Timedelta(forecast_horizon)
        except ValueError:
            duration = None
        if (
            duration is not None
            and not pd.isna(duration)
            and duration > pd.Timedelta(0)
        ):
            return duration
    raise FlexConfigError(
        "ArimaRegressor forecast_horizon must be a positive int number of "
        'steps or a positive duration such as "2D", "48h" or '
        f"pd.Timedelta(days=2); got {forecast_horizon!r}.",
        field="forecast_horizon",
        value=forecast_horizon,
    )


def horizon_steps(
    forecast_horizon: int | pd.Timedelta, time_step_seconds: float
) -> int:
    """Convert a parsed ``forecast_horizon`` into a window length in steps.

    Args:
        forecast_horizon: The output of :func:`parse_forecast_horizon`.
        time_step_seconds: Sampling step of the fitted rows, in seconds.

    Returns:
        An int horizon unchanged; a duration rounded up to whole steps, and
        at least one.
    """
    if isinstance(forecast_horizon, int):
        return forecast_horizon
    # Tolerate float error so an exact multiple of the step does not round up.
    steps = math.ceil(forecast_horizon.total_seconds() / time_step_seconds - 1e-9)
    return max(steps, 1)


def step_seconds(index) -> float:
    """Return the sampling step of ``index`` in seconds.

    Args:
        index: The index of the fitted rows.

    Returns:
        The declared frequency when the index carries one, otherwise the
        gap between the first two rows, falling back to one hour for a
        single-row index.
    """
    if getattr(index, "freq", None) is not None:
        return float(pd.Timedelta(index.freq).total_seconds())
    if len(index) > 1:
        return float((index[1] - index[0]).total_seconds())
    return 3600.0


# Defaults for the ``auto=True`` order search, as in R ``forecast::auto.arima``
# with ``stepwise=False``, except that ``d`` is capped at 1 because the Pyomo
# surrogate supports at most one order of differencing.
AUTO_SEARCH_DEFAULTS: dict[str, object] = {
    "max_p": 5,
    "max_q": 5,
    "max_order": 5,
    "max_d": 1,
    "d": None,
    "stationary": False,
}

# 5% critical value of the KPSS level-stationarity statistic (Kwiatkowski et
# al. 1992, Table 1), the unit-root test auto.arima uses to choose ``d``.
KPSS_CRITICAL_5PCT = 0.463

# Significance level of the ADF unit-root test run alongside KPSS.
ADF_ALPHA = 0.05

# Rows the order search drops after the largest candidate lag before scoring.
# The innovations start from zero, and a candidate with more MA terms can fit
# that start-up transient rather than the process. Scored over it, the search
# missed the true order of 16 of 30 simulated ARMA series (n=400, max_p=max_q=3)
# and 8 of 30 with this burn-in; an exact-likelihood AIC search (statsmodels)
# missed 15.
AUTO_SCORE_BURN_IN = 10


def auto_search_options(auto_kwargs: dict[str, object]) -> dict[str, object]:
    """Validate the ``auto=True`` search options and fill in the defaults.

    Args:
        auto_kwargs: The caller's options, a subset of
            :data:`AUTO_SEARCH_DEFAULTS`.

    Returns:
        Every option, as keyword arguments for :func:`auto_select_order`
        (``stationary`` is folded into ``max_d``).

    Raises:
        FlexConfigError: If an option is unknown or out of range, or if
            ``stationary=True`` is combined with ``d=1``.
    """
    unknown = sorted(set(auto_kwargs) - set(AUTO_SEARCH_DEFAULTS))
    if unknown:
        raise FlexConfigError(
            f"ArimaRegressor auto=True does not accept {unknown}; the order "
            f"search takes only {sorted(AUTO_SEARCH_DEFAULTS)}. It is always "
            "non-seasonal and ranks candidates by AIC.",
            field="auto_kwargs",
            value=unknown,
        )
    options = {**AUTO_SEARCH_DEFAULTS, **auto_kwargs}
    for key in ("max_p", "max_q", "max_order"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FlexConfigError(
                f"ArimaRegressor {key} must be a non-negative int, got {value!r}.",
                field=key,
                value=value,
            )
    for key, allowed in (("max_d", (0, 1)), ("d", (None, 0, 1))):
        value = options[key]
        if isinstance(value, bool) or value not in allowed:
            raise FlexConfigError(
                f"ArimaRegressor {key} must be one of {list(allowed)}, got "
                f"{value!r}; the Pyomo surrogate supports at most d=1.",
                field=key,
                value=value,
            )
    stationary = options.pop("stationary")
    if not isinstance(stationary, bool):
        raise FlexConfigError(
            f"ArimaRegressor stationary must be a bool, got {stationary!r}.",
            field="stationary",
            value=stationary,
        )
    if stationary:
        if options["d"] == 1:
            raise FlexConfigError(
                "ArimaRegressor stationary=True restricts the search to d=0, "
                "so it cannot be combined with d=1.",
                field="stationary",
                value=True,
            )
        options["max_d"] = 0
    return options


def kpss_statistic(z: np.ndarray) -> float:
    """Return the KPSS statistic for level stationarity of ``z``.

    The long-run variance uses a Bartlett kernel with the "short" lag
    truncation ``int(4 * (n / 100) ** 0.25)`` of Kwiatkowski et al. (1992).

    Args:
        z: The series to test.

    Returns:
        The statistic; large values reject stationarity. 0.0 for a constant
        series.
    """
    z = np.asarray(z, dtype=float)
    n = len(z)
    e = z - z.mean()
    lags = min(int(4 * (n / 100) ** 0.25), n - 1)
    long_run_var = e @ e / n
    for j in range(1, lags + 1):
        long_run_var += 2.0 * (1.0 - j / (lags + 1)) * (e[j:] @ e[:-j]) / n
    if long_run_var <= 0:
        return 0.0
    partial_sums = np.cumsum(e)
    return float(partial_sums @ partial_sums / (n * n * long_run_var))


def select_differencing(y: np.ndarray, x: np.ndarray | None, max_d: int) -> int:
    """Choose ``d`` by KPSS and ADF tests on the regression disturbance.

    As in auto.arima, ``y`` is first regressed on ``X`` (plus an intercept),
    so a trend the inputs explain does not count as a unit root. The two
    tests take opposite nulls: KPSS that the disturbance is level
    stationary, ADF that it has a unit root. ``d`` is settled only when
    exactly one of them rejects at 5%.

    Both use their standard critical values, which assume an observed
    series. On a residual from a regression on ``X`` the exact ones
    (Engle-Granger for ADF, Shin 1994 for KPSS) are more extreme, so with
    exogenous inputs ADF rejects a unit root somewhat too readily.

    ADF comes from statsmodels, imported here rather than at module import
    so ``import flexparameterize.regression`` works without the
    ``[parameterize]`` extra.

    Args:
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.
        max_d: Largest ``d`` allowed, 0 or 1.

    Returns:
        0 if ``max_d`` is 0; otherwise 1 when KPSS rejects stationarity and
        ADF does not reject a unit root, and 0 when ADF rejects and KPSS
        does not.

    Raises:
        FlexConfigError: If statsmodels is not installed.
        FlexDataError: If both tests reject (contradictory), neither does
            (inconclusive), or the series is too short to run ADF.
    """
    if max_d == 0:
        return 0
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise FlexConfigError(
            "ArimaRegressor auto=True needs statsmodels to test for a unit "
            "root. Install it with `pip install 'flex-pse[parameterize]'`, "
            "or pass d=0 or d=1 explicitly."
        ) from exc
    eta = np.asarray(y, dtype=float)
    if x is not None and x.shape[1]:
        design = np.column_stack([np.ones(len(eta)), x])
        eta = eta - design @ np.linalg.lstsq(design, eta, rcond=None)[0]
    if np.ptp(eta) == 0.0:
        # A constant disturbance is stationary; neither test can run on it.
        return 0
    try:
        adf_p = float(adfuller(eta, regression="c", autolag="AIC")[1])
    except ValueError as exc:
        raise FlexDataError(
            f"ArimaRegressor auto=True cannot choose d: {len(eta)} row(s) are "
            "too few for the ADF unit-root test. Pass d=0 or d=1 explicitly.",
            field="d",
        ) from exc
    kpss = kpss_statistic(eta)
    kpss_rejects = kpss > KPSS_CRITICAL_5PCT
    adf_rejects = adf_p < ADF_ALPHA
    if kpss_rejects != adf_rejects:
        return int(kpss_rejects)
    verdict = (
        "contradictory: KPSS rejects stationarity and ADF rejects a unit root"
        if kpss_rejects
        else "inconclusive: neither KPSS rejects stationarity nor ADF a unit root"
    )
    raise FlexDataError(
        f"ArimaRegressor auto=True cannot choose d; the unit-root tests are "
        f"{verdict} (KPSS statistic {kpss:.3f} vs 5% critical value "
        f"{KPSS_CRITICAL_5PCT}, ADF p-value {adf_p:.3f}). Consider adding "
        "more exogenous covariates X (e.g. lagged inputs) for missing "
        "structure, consider removing trends or seasonality and retest, or "
        "pass d=0 or d=1 explicitly.",
        field="d",
    )


def auto_select_order(
    y: np.ndarray,
    x: np.ndarray | None,
    *,
    include_mean: bool,
    max_ar_persistence: float | None,
    max_p: int,
    max_q: int,
    max_order: int,
    max_d: int,
    d: int | None,
) -> tuple[int, int, int]:
    """Select a non-seasonal ``(p, d, q)`` by an exhaustive AIC search.

    ``d`` comes from :func:`select_differencing` unless given. Every
    ``(p, q)`` with ``p <= max_p``, ``q <= max_q`` and ``p + q <= max_order``
    is then fitted by equation-error least squares, under the same AR bound
    as the final fit, and ranked by AIC over one common sample: the rows
    after the largest lag any candidate uses plus
    :data:`AUTO_SCORE_BURN_IN`, so candidates of different order are
    compared on the same innovations, past the zero-initialized start-up.
    Candidates the data cannot identify, or whose AR block is not
    stationary or MA block not invertible, are skipped. Ties go to the
    smaller order.

    Args:
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.
        include_mean: Whether candidates carry the deterministic term.
        max_ar_persistence: Bound on every AR coefficient, or ``None``.
        max_p: Largest AR order searched.
        max_q: Largest MA order searched.
        max_order: Largest ``p + q`` searched.
        max_d: Largest ``d`` the differencing test may choose, 0 or 1.
        d: Fixed differencing order, or ``None`` to test for it.

    Returns:
        The selected ``(p, d, q)``.

    Raises:
        FlexDataError: If ``y`` has too few rows to fit any candidate, or
            ``d`` is not given and the unit-root tests cannot settle it (see
            :func:`select_differencing`).
        FlexConfigError: If every candidate is non-stationary or
            non-invertible, or ``d`` must be tested and statsmodels is not
            installed.
    """
    y = np.asarray(y, dtype=float)
    n_exog = 0 if x is None else x.shape[1]
    if d is None:
        d = select_differencing(y, x, max_d)
    candidates = []
    for p in range(max_p + 1):
        for q in range(min(max_q, max_order - p) + 1):
            terms = ArimaTerms(
                p=p,
                d=d,
                q=q,
                n_exog=n_exog,
                has_const=include_mean and d == 0,
                has_drift=include_mean and d == 1,
            )
            # The row requirement ArimaRegressor.fit applies to a fixed order.
            if len(y) >= max(p, q) + d + terms.n_params + 1:
                candidates.append(terms)
    if not candidates:
        raise FlexDataError(
            f"ArimaRegressor auto=True has {len(y)} row(s), too few to fit "
            f"even the smallest candidate order with d={d}.",
            field="y",
        )

    largest_lag = max(max(t.p, t.q) for t in candidates)
    # Keep at least half of the shortest candidate's innovations scored.
    burn_in = min(AUTO_SCORE_BURN_IN, (len(y) - d - largest_lag) // 2)
    score_start = largest_lag + burn_in
    best_order, best_aic = None, np.inf
    for terms in candidates:
        theta = initial_theta(terms, y, x)
        if theta.size:
            theta = _least_squares(
                equation_error_residuals,
                theta,
                (terms, y, x),
                coefficient_bounds(terms, max_ar_persistence),
            )
        _c, ar, ma, _beta = terms.unpack(theta)
        if ar_max_root(ar) >= 1.0 or ma_max_root(ma) >= 1.0:
            continue
        resid = one_step_innovations(theta, terms, y, x)[1][score_start:]
        sigma2 = max(float(resid @ resid) / len(resid), np.finfo(float).tiny)
        # Gaussian AIC, up to a constant every candidate shares.
        aic = len(resid) * np.log(sigma2) + 2.0 * (terms.n_params + 1)
        if aic < best_aic:
            best_order, best_aic = (terms.p, terms.d, terms.q), aic
    if best_order is None:
        raise FlexConfigError(
            f"ArimaRegressor auto=True found no stationary, invertible "
            f"candidate among {len(candidates)} order(s) with d={d}. Lower "
            f"max_p/max_q, set d explicitly, or pass an explicit order.",
            field="order",
        )
    return best_order
