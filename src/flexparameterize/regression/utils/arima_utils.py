"""Fitting and forecasting machinery behind :class:`ArimaRegressor`.

The model is regression with ARIMA errors, the same equation
:class:`~flexops.surrogates.arima.ArimaSurrogate` builds in Pyomo::

    eta[t] = y[t] - X[t] @ beta                        (disturbance)
    z[t]   = eta[t]  (d=0)   or   eta[t] - eta[t-1]  (d=1)
    z[t]   = c + sum(ar_j * z[t-j]) + sum(ma_j * eps[t-j]) + eps[t]

Every function here takes one flat parameter vector ``theta`` laid out as
``[c?, ar_1..ar_p, ma_1..ma_q, beta_1..beta_k]``, with the layout described
by an :class:`ArimaTerms`. Two fitting criteria are supported:

- *equation error*: the one-step innovations ``eps``, computed from actual
  lagged values (:func:`equation_error_residuals`);
- *output error*: the error of a free run that feeds its own predictions
  back as lags with innovations held at zero, which is what the surrogate
  does when it forecasts (:func:`output_error_residuals`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MethodType

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from dateutil.relativedelta import relativedelta
from pyomo.environ import units as pyunits
from scipy.optimize import least_squares
from scipy.signal import lfilter, lfiltic

from flexcore.exceptions import FlexConfigError, FlexDataError
from flexcore.logger import get_logger
from flexcore.solvers import ProblemClass, get_solver
from flexops.core.time_block import TimeBlock
from flexops.core.units import parse_units
from flexops.surrogates.arima import ArimaSurrogate

_log = get_logger(__name__)

# Bound applied to every MA coefficient in an output-error fit, under both
# solvers. The free-run objective barely sees the MA block (it only reaches
# the first q steps of each window), so unbounded it trades MA against the
# bounded AR block and drives the MA roots outside the unit circle. The
# forecast seeds its MA lags from the one-step residuals, which such a block
# makes explode: measured first-step forecasts of ~1e6 on biogas data.
MA_BOUND = 0.99

# Upper bound on the automatically chosen free-run window. The best horizon
# is the one the caller actually forecasts over, which cannot be inferred, so
# "auto" spans the data up to this cap. Measured on 5757 rows, an uncapped
# window left the d=0 objective so flat that least_squares ground to its
# evaluation limit (120-140 s) while 192 steps took under 8 s, no worse.
AUTO_MAX_HORIZON = 192


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
            ``const``/``drift`` (when present), ``ar1..``, ``ma1..``, then
            the exogenous column names.
        """
        names = ["const"] if self.has_const else ["drift"] if self.has_drift else []
        names += [f"ar{j}" for j in range(1, self.p + 1)]
        names += [f"ma{j}" for j in range(1, self.q + 1)]
        return names + list(exog_names)


def auto_forecast_horizon(n_rows: int, terms: ArimaTerms) -> int:
    """Return the default free-run window length for ``n_rows`` of data.

    Prefer an explicit ``forecast_horizon``: the best value is the horizon
    you intend to forecast over, and no rule here can infer it.

    Args:
        n_rows: Number of rows the fit will use.
        terms: The model structure.

    Returns:
        The usable row count capped at :data:`AUTO_MAX_HORIZON`, raised if
        needed so a window outlasts the rows seeding it.
    """
    usable = max(n_rows - terms.seed, 1)
    return max(min(usable, AUTO_MAX_HORIZON), terms.seed + 1)


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

    Args:
        theta: Flat parameter vector.
        terms: Its layout.
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.
        horizon: Window length in steps, at least 1.

    Returns:
        Level-scale residuals from row ``terms.seed`` onward. The exogenous
        contribution cancels, so each is ``eta[t]`` less its simulation.
    """
    eta, eps = one_step_innovations(theta, terms, y, x)
    residuals = [np.zeros(0)]
    for start in range(terms.seed, len(eta), horizon):
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
        ``llf``, ``aic``, ``bic``, ``aicc``, ``sigma2``, and ``rmse`` exclude
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

    @property
    def aicc(self) -> float | None:
        """Small-sample corrected AIC, or ``None`` when it is undefined.

        The correction ``2k(k+1) / (n - k - 1)`` is undefined once ``n``
        drops to ``k + 1``. AICc is only reported, never used by the fit, so
        this logs a warning instead of raising.
        """
        k = self.terms.n_params + 1
        denominator = self.nobs_effective - k - 1
        if denominator <= 0:
            _log.warning(
                "AICc is undefined for this fit: %d effective observation(s) "
                "against %d parameter(s) leaves a correction denominator of "
                "%d. Reporting AICc as None; use AIC or BIC, or fit a lower "
                "order on more data.",
                self.nobs_effective,
                k,
                denominator,
            )
            return None
        return float(self.aic) + (2.0 * k * (k + 1)) / denominator

    def predict(
        self,
        steps: int = 1,
        exog: np.ndarray | None = None,
        start: int | None = None,
        dynamic: bool = True,
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
            dynamic: Must be ``True``; one-step prediction from actual lags
                is not implemented.

        Returns:
            ``steps`` forecasts on the level scale of ``y``.

        Raises:
            FlexConfigError: If ``dynamic=False`` is passed.
            FlexDataError: If ``start`` leaves fewer than ``max(p + d, q)``
                rows of history.
        """
        if not dynamic:
            raise FlexConfigError(
                "ArimaRegressor's direct-fit predict() only implements "
                "dynamic=True (recursive) prediction; one-step-ahead "
                "dynamic=False prediction is not implemented. Use "
                "`fittedvalues_level` for in-sample one-step-ahead values, or "
                "omit `dynamic` (it defaults to True)."
            )
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
    fit_objective: str,
    forecast_horizon: int,
    max_ar_persistence: float | None,
    refine=None,
) -> DirectResults:
    """Fit ``theta`` by least squares and wrap it in :class:`DirectResults`.

    Always minimizes the equation error first, from :func:`initial_theta`.
    An ``"output_error"`` fit then refines that solution against
    :func:`output_error_residuals`, which is cheaper and better behaved than
    a cold start on the nonconvex free-run objective.

    Args:
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.
        terms: The model structure.
        exog_names: Exogenous column names, in fitted order.
        fit_objective: ``"equation_error"`` or ``"output_error"``.
        forecast_horizon: Free-run window length in steps.
        max_ar_persistence: Bound on every AR coefficient, or ``None``.
        refine: Optional ``theta -> theta`` callable used instead of the
            scipy output-error refinement; this is how the ipopt backend is
            injected.

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
    if theta.size and fit_objective == "output_error":
        if refine is not None:
            theta = refine(theta)
        else:
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
) -> dict[str, object]:
    """Build the surrogate ``history`` block from a fitted disturbance series.

    The surrogate replays ``max(p + d, q)`` values ahead of its first modeled
    point. No true pre-sample data exists, so the prefix repeats the start of
    the series; both lists share one prefix length so one offset indexes
    either.

    Args:
        eta: Disturbance series ``y - X @ beta`` on the level scale.
        eps: One-step innovations on the differenced series.
        terms: The model structure.
        start_date: ISO-8601 timestamp of the first fitted row.
        time_step_seconds: Sampling step in seconds.

    Returns:
        The ``history`` mapping in the surrogate's persisted contract.
    """
    seed, lagged = terms.seed, terms.p + terms.d
    prefix = [float(eta[0])] * (seed - lagged) + [float(v) for v in eta[:lagged]]
    return {
        "start_date": start_date,
        "time_step_seconds": float(time_step_seconds),
        "y_values": prefix + [float(v) for v in eta],
        "eps_values": [0.0] * (seed + terms.d) + [float(v) for v in eps],
    }


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


def fit_output_error_ipopt(
    theta0: np.ndarray,
    terms: ArimaTerms,
    y: np.ndarray,
    x: np.ndarray | None,
    *,
    exog_names: list[str],
    input_units: dict[str, str],
    output_name: str,
    output_units: str,
    training_index,
    max_ar_persistence: float | None,
) -> np.ndarray:
    """Minimize output error through the real Pyomo surrogate, with ipopt.

    Builds an :class:`~flexops.surrogates.arima.ArimaSurrogate` over a
    :class:`~flexops.core.time_block.TimeBlock` spanning the training series,
    frees the coefficients, holds the innovations at zero so the block
    free-runs, and minimizes ``sum((y[t] - data[t])**2)``. The coefficients
    are therefore optimal for the exact equation the surrogate solves.

    One model spans the whole series, so the free run cannot be windowed.
    The pre-horizon state is seeded from data, ``y - X @ beta`` over the
    first ``p + d`` rows, and constrained to follow ``beta`` as the solver
    moves it; a seed fixed at the warm start's ``beta`` would leave a
    constant level offset across a ``d=1`` free run. Leaving the state free
    instead lets the solver fit coefficients that suit its own estimated
    state and transfer poorly (measured 0.087 free-run rmse against 0.005
    seeded). ``theta0`` is
    clipped into the same bounds as the scipy backend first. That matters
    for MA in particular: the single window reaches it only through the
    zero-padded seed innovations, so MA drops out of the problem ipopt
    receives and keeps its warm-start value.

    Args:
        theta0: Equation-error solution, used as the warm start.
        terms: The model structure.
        y: Output series on the level scale.
        x: Exogenous matrix, or ``None``.
        exog_names: Exogenous column names, in fitted order.
        input_units: Units of every exogenous column, keyed by name.
        output_name: Name of the fitted output column.
        output_units: Units of the fitted output column.
        training_index: The fitted rows' index; must be a regular
            ``DatetimeIndex`` so a ``TimeBlock`` grid can be derived.
        max_ar_persistence: Bound on every AR coefficient, or ``None``.

    Returns:
        The refined parameter vector.

    Raises:
        FlexConfigError: If the output or an input carries no units, or
            ``training_index`` is not a regular ``DatetimeIndex``.
    """
    if not output_units:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" builds a real Pyomo '
            "surrogate, which needs declared units. Pass output_units to "
            "fit().",
            field="output_units",
            value=output_units,
        )
    missing_units = [name for name in exog_names if not input_units.get(name)]
    if missing_units:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" needs units for every input; '
            f"{missing_units} have none. Pass input_units to fit().",
            field="input_units",
            value=missing_units,
        )
    if output_name in exog_names:
        raise FlexConfigError(
            f"ArimaRegressor cannot fit output {output_name!r} against an "
            "input of the same name.",
            field="output_variables",
            value=output_name,
        )
    if not isinstance(training_index, pd.DatetimeIndex) or len(training_index) < 2:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" needs a regular DatetimeIndex '
            "of at least two rows to build a TimeBlock grid; got "
            f"{type(training_index).__name__} of length "
            f'{len(training_index)}. Use fit_solver="scipy".',
            field="fit_solver",
            value="ipopt",
        )
    step = step_seconds(training_index)
    if step <= 0:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" could not derive a positive '
            f"time step from the training index (got {step} s).",
            field="fit_solver",
            value="ipopt",
        )

    n = len(y)
    start = training_index[0].to_pydatetime()
    model = pyo.ConcreteModel()
    model.time_block = TimeBlock(
        start_date=start.isoformat(),
        end_date=(start + pd.Timedelta(seconds=step * n).to_pytimedelta()).isoformat(),
        time_step=step * pyunits.s,
        max_length=relativedelta(seconds=int(step * (n + 1))),
    )
    time_index = list(model.time_block.time_index)
    if len(time_index) != n:
        raise FlexConfigError(
            'ArimaRegressor fit_solver="ipopt" derived a TimeBlock of '
            f"{len(time_index)} steps for {n} training rows, so the index is "
            'not on a regular grid. Use fit_solver="scipy".',
            field="fit_solver",
            value="ipopt",
        )

    unit = pyo.Block(concrete=True)
    model.unit = unit
    unit.add_component(
        output_name,
        pyo.Var(time_index, initialize=0.0, units=parse_units(output_units)),
    )
    target = unit.find_component(output_name)
    for name in exog_names:
        unit.add_component(
            name,
            pyo.Var(time_index, initialize=0.0, units=parse_units(input_units[name])),
        )

    def _resolve_variable(self, name, field=None):
        """Stand in for OpsBlockData.resolve_variable for the surrogate."""
        component = self.find_component(name)
        if component is None:
            raise FlexConfigError(
                f"ARIMA input {name!r} is not on the fitting block.",
                field=field,
                value=name,
            )
        return component

    unit.resolve_variable = MethodType(_resolve_variable, unit)

    upper = coefficient_bounds(terms, max_ar_persistence, MA_BOUND)
    theta0 = np.clip(theta0, -upper, upper)
    eta, eps = one_step_innovations(theta0, terms, y, x)
    surrogate = ArimaSurrogate(
        {
            "input_variables": {name: input_units[name] for name in exog_names},
            "output_variables": {output_name: output_units},
            "coefficients": surrogate_coefficients(theta0, terms),
            "history": history_payload(
                eta, eps, terms, training_index[0].isoformat(), step
            ),
        },
        max_ar_coeff=max_ar_persistence,
    )
    block, body = surrogate.build(unit, target)
    unit.arima = block
    unit.fitted = pyo.Constraint(time_index, rule=lambda _b, t: target[t] == body(t))

    for position, t in enumerate(time_index):
        for column, name in enumerate(exog_names):
            unit.find_component(name)[t].fix(float(x[position, column]))
        target[t].set_value(float(y[position]))

    block.coefficients.unfix()
    if terms.n_exog:
        # The history holds eta for rows 0..p+d-1; keep it equal to
        # y - X @ beta at the solver's beta, not the warm start's.
        block.initial_y_history.unfix()
        unit.history_seed = pyo.Constraint(
            block.y_history_index,
            rule=lambda _b, i: block.initial_y_history[i]
            == (
                float(y[i])
                - sum(
                    block.exog_coefs[k + 1] * float(x[i, k])
                    for k in range(terms.n_exog)
                )
            )
            * parse_units(output_units),
        )
    for index in range(1, terms.q + 1):
        block.ma_coefs[index].setlb(-MA_BOUND)
        block.ma_coefs[index].setub(MA_BOUND)

    model.objective = pyo.Objective(
        expr=sum(
            (target[t] - float(y[position])) ** 2
            for position, t in enumerate(time_index)
        )
    )
    results = get_solver(problem_class=ProblemClass.NLP, prefer="ipopt").solve(model)
    if not pyo.check_optimal_termination(results):
        _log.warning(
            'ArimaRegressor fit_solver="ipopt" terminated %s rather than '
            "optimally for order=%s. The returned coefficients are the "
            "solver's last iterate; check free_run_rmse before using them, "
            'or refit with fit_solver="scipy".',
            results.solver.termination_condition,
            (terms.p, terms.d, terms.q),
        )

    fitted = surrogate.get_surrogate_spec(block, target)["coefficients"]
    deterministic = [fitted.get("intercept", fitted.get("drift", 0.0))]
    return np.array(
        deterministic[: terms.n_deterministic]
        + list(fitted.get("ar_coefs", []))
        + list(fitted.get("ma_coefs", []))
        + list(fitted.get("exog_coefs", [])),
        dtype=float,
    )


def auto_select_order(
    y: np.ndarray,
    x: np.ndarray | None,
    *,
    seasonal: bool,
    stationary: bool,
    auto_kwargs: dict[str, object],
) -> tuple[tuple[int, int, int], tuple[int, int, int, int] | None]:
    """Select an order with statsforecast ``AutoARIMA``.

    Seasonal differencing ``D`` defaults to 0, since the backend cannot fit
    it. The caller validates ``d`` and the seasonal terms, and refits the
    returned order with :func:`fit_direct`.

    Args:
        y: Output series.
        x: Exogenous matrix, or ``None``.
        seasonal: Whether to include seasonal terms in the search.
        stationary: If ``True``, restrict the search to stationary models.
        auto_kwargs: Extra keyword arguments forwarded to ``AutoARIMA``
            (e.g. ``max_p``, ``max_q``, ``season_length``).

    Returns:
        ``(order, seasonal_order)``, with ``seasonal_order`` ``None`` when
        the selected model has no seasonal AR/MA terms.

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

    kwargs = {"D": 0, **({"stationary": True} if stationary else {}), **auto_kwargs}
    fitted = AutoARIMA(seasonal=seasonal, ic="aic", **kwargs).fit(y, X=x)
    p, q, P, Q, m, d, _D = (int(value) for value in fitted.model_["arma"])
    seasonal_order = None if P == 0 and Q == 0 else (P, 0, Q, m)
    return (p, d, q), seasonal_order
