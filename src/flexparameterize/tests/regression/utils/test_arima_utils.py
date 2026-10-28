"""Tests for the ARIMA fitting and forecasting machinery in ``arima_utils``.

The vectorized recursions are pinned against plain-loop reference
implementations of the mean equation, so any indexing slip shows up here
rather than as a drifted fit downstream.
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from flexcore.exceptions import FlexConfigError, FlexDataError
from flexparameterize.regression.utils.arima_utils import (
    AUTO_MAX_HORIZON,
    KPSS_CRITICAL_5PCT,
    ArimaTerms,
    DirectResults,
    ar_max_root,
    auto_forecast_horizon,
    auto_search_options,
    auto_select_order,
    coefficient_bounds,
    equation_error_residuals,
    free_run,
    history_payload,
    history_steps,
    kpss_statistic,
    ma_max_root,
    one_step_innovations,
    output_error_residuals,
    seed_history,
    select_differencing,
    surrogate_coefficients,
)

_TERMS = [
    ArimaTerms(p=2, d=0, q=1, n_exog=1, has_const=True, has_drift=False),
    ArimaTerms(p=1, d=1, q=2, n_exog=1, has_const=False, has_drift=True),
    ArimaTerms(p=0, d=1, q=1, n_exog=0, has_const=False, has_drift=True),
    ArimaTerms(p=3, d=0, q=0, n_exog=2, has_const=False, has_drift=False),
]


def _theta(terms: ArimaTerms) -> np.ndarray:
    """A fixed, stationary and invertible parameter vector for ``terms``."""
    values = [0.02] * int(terms.has_const or terms.has_drift)
    values += [0.5, -0.2, 0.1][: terms.p]
    values += [0.3, 0.1][: terms.q]
    values += [1.5, -0.7][: terms.n_exog]
    return np.array(values)


def _series(terms: ArimaTerms, n: int = 80):
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, size=(n, terms.n_exog)) if terms.n_exog else None
    return np.cumsum(rng.normal(0, 0.1, size=n)), x


def _reference_innovations(theta, terms, y, x):
    """The mean equation, one step at a time, with actual lagged values."""
    c, ar, ma, beta = terms.unpack(theta)
    eta = y - x @ beta if terms.n_exog else y
    z = np.diff(eta) if terms.d else eta
    eps = np.zeros(len(z))
    for t in range(max(terms.p, terms.q), len(z)):
        eps[t] = (
            z[t]
            - c
            - sum(ar[j] * z[t - j - 1] for j in range(terms.p))
            - sum(ma[j] * eps[t - j - 1] for j in range(terms.q))
        )
    return eta, eps


def _reference_free_run(theta, terms, eta_history, eps_history, steps):
    """Forward simulation with future innovations at zero, one step at a time."""
    c, ar, ma, _beta = terms.unpack(theta)
    eta, eps = list(eta_history), list(eps_history)
    out = []
    for _ in range(steps):
        z = [eta[i] - eta[i - 1] for i in range(1, len(eta))] if terms.d else eta
        mean = c + sum(ar[j] * z[-j - 1] for j in range(terms.p))
        mean += sum(ma[j] * eps[-j - 1] for j in range(terms.q))
        value = eta[-1] + mean if terms.d else mean
        out.append(value)
        eta.append(value)
        eps.append(0.0)
    return np.array(out)


@pytest.mark.unit
def test_terms_unpack_and_name_the_parameter_vector():
    """The flat layout is ``[c?, ar.., ma.., beta..]``, named to match."""
    terms = ArimaTerms(p=2, d=1, q=1, n_exog=1, has_const=False, has_drift=True)
    c, ar, ma, beta = terms.unpack(np.array([0.1, 0.5, -0.2, 0.3, 2.0]))

    assert (c, list(ar), list(ma), list(beta)) == (0.1, [0.5, -0.2], [0.3], [2.0])
    assert terms.n_params == 5
    assert terms.seed == 3
    assert terms.param_names(["feed"]) == [
        "drift",
        "ar.L1",
        "ar.L2",
        "ma.L1",
        "exog.feed",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("terms", _TERMS)
def test_one_step_innovations_match_the_recursion(terms):
    """The filtered innovations equal the step-by-step mean equation."""
    y, x = _series(terms)
    eta, eps = one_step_innovations(_theta(terms), terms, y, x)
    ref_eta, ref_eps = _reference_innovations(_theta(terms), terms, y, x)

    np.testing.assert_allclose(eta, ref_eta, rtol=0, atol=1e-12)
    np.testing.assert_allclose(eps, ref_eps, rtol=0, atol=1e-12)
    lag = max(terms.p, terms.q)
    np.testing.assert_array_equal(
        equation_error_residuals(_theta(terms), terms, y, x), eps[lag:]
    )


@pytest.mark.unit
@pytest.mark.parametrize("terms", _TERMS)
def test_free_run_matches_the_recursion(terms):
    """The filtered free run equals the step-by-step forward simulation."""
    y, x = _series(terms)
    eta, eps = one_step_innovations(_theta(terms), terms, y, x)
    history = seed_history(terms, eta, eps, start=40)

    np.testing.assert_allclose(
        free_run(_theta(terms), terms, *history, steps=25),
        _reference_free_run(_theta(terms), terms, *history, steps=25),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.unit
def test_seed_history_zero_pads_innovations_that_do_not_exist():
    """Before the first defined innovation the MA lags are zero, as in the
    surrogate's own history prefix."""
    terms = ArimaTerms(p=0, d=1, q=3, n_exog=0, has_const=False, has_drift=True)
    eta = np.arange(10.0)
    eps = np.arange(1.0, 10.0)

    eta_history, eps_history = seed_history(terms, eta, eps, start=2)

    assert list(eta_history) == [1.0]
    assert list(eps_history) == [0.0, 0.0, 1.0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "terms",
    [
        ArimaTerms(p=2, d=0, q=1, n_exog=1, has_const=True, has_drift=False),
        ArimaTerms(p=1, d=1, q=1, n_exog=0, has_const=False, has_drift=True),
    ],
)
def test_output_error_at_horizon_one_is_the_one_step_residual(terms):
    """A one-step window seeded from actual data *is* the one-step residual.

    Holds whenever both residuals start at the same row, i.e. ``q <= p + 1``.
    """
    y, x = _series(terms)
    theta = _theta(terms)

    np.testing.assert_allclose(
        output_error_residuals(theta, terms, y, x, horizon=1),
        equation_error_residuals(theta, terms, y, x),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ma_coefs", "expected_invertible"),
    [
        ([], True),
        ([0.5], True),
        ([0.1943], True),
        ([-0.0605, 0.8568], True),
        ([0.9601, 0.952, 0.9145], True),
        ([-1.0142], False),
        ([-0.1243, 1.0064], False),
        ([-0.99, -0.99], False),
        ([54214.0, 53453.0, 43003.0], False),
    ],
)
def test_ma_max_root_detects_noninvertibility(ma_coefs, expected_invertible):
    """The MA guard tracks the unit circle, including boundary cases produced
    by intermediate horizons, by ipopt regression, and inside the MA bound."""
    assert (ma_max_root(ma_coefs) < 1.0) is expected_invertible


@pytest.mark.unit
def test_ar_max_root_sees_through_the_per_coefficient_bound():
    """``[0.85, 0.85]`` passes a per-coefficient bound but is explosive."""
    assert ar_max_root([]) == 0.0
    assert ar_max_root([0.5]) == pytest.approx(0.5)
    assert ar_max_root([0.85, 0.85]) == pytest.approx(1.4402, abs=1e-4)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("n_rows", "p", "d", "q", "expected"),
    [
        (60, 1, 1, 0, 58),  # 60 rows less the 2 consumed seeding
        (100, 1, 1, 0, 98),  # below the cap
        (400, 1, 1, 0, AUTO_MAX_HORIZON),
        (5000, 1, 1, 0, AUTO_MAX_HORIZON),
        (30, 3, 1, 3, 26),  # seed = max(p + d, q) = 4
        (20, 0, 0, 14, 15),  # seed + 1 floor: a window must outlast its seed
    ],
)
def test_auto_forecast_horizon_spans_the_usable_rows(n_rows, p, d, q, expected):
    """The auto horizon spans the usable rows, capped, and stays seedable.

    Uncapped, a full-series window on several thousand rows left the d=0
    objective so flat that least_squares ground to its evaluation limit.
    """
    terms = ArimaTerms(p=p, d=d, q=q, n_exog=0, has_const=False, has_drift=False)

    horizon = auto_forecast_horizon(n_rows, terms)

    assert horizon == expected
    assert horizon > terms.seed


@pytest.mark.unit
def test_coefficient_bounds_cover_only_the_ar_and_ma_slots():
    """The deterministic term and exogenous coefficients stay unbounded."""
    terms = ArimaTerms(p=2, d=0, q=1, n_exog=1, has_const=True, has_drift=False)

    upper = coefficient_bounds(terms, max_ar_persistence=0.85, ma_bound=0.99)

    np.testing.assert_array_equal(upper, [np.inf, 0.85, 0.85, 0.99, np.inf])
    np.testing.assert_array_equal(
        coefficient_bounds(terms, max_ar_persistence=None), [np.inf] * 5
    )


@pytest.mark.unit
@pytest.mark.parametrize("terms", _TERMS)
def test_predict_continues_the_free_run_from_the_end_of_the_data(terms):
    """Out-of-sample prediction is the free run seeded at the last row, plus
    the exogenous contribution of the future inputs."""
    y, x = _series(terms)
    theta = _theta(terms)
    results = DirectResults(theta, terms, ["a", "b"][: terms.n_exog], y, x, 10)
    future_x = np.full((5, terms.n_exog), 0.5) if terms.n_exog else None

    eta, eps = one_step_innovations(theta, terms, y, x)
    expected = free_run(theta, terms, *seed_history(terms, eta, eps, len(y)), 5)
    if terms.n_exog:
        expected = expected + future_x @ terms.unpack(theta)[3]

    np.testing.assert_allclose(results.predict(5, exog=future_x), expected)


@pytest.mark.unit
def test_history_payload_pads_both_series_to_a_shared_prefix():
    """The surrogate indexes y and eps history with one offset."""
    terms = ArimaTerms(p=1, d=1, q=3, n_exog=0, has_const=False, has_drift=True)
    eta = np.array([1.0, 2.0, 3.0, 4.0])
    eps = np.array([0.0, 0.1, 0.2])

    payload = history_payload(eta, eps, terms, "2024-01-01T00:00:00", 900.0)

    # seed = max(p + d, q) = 3; the y prefix repeats eta[0] ahead of eta[:p + d]
    assert payload["eta_values"] == [1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0]
    assert payload["eps_values"] == [0.0] * 4 + [0.0, 0.1, 0.2]
    assert payload["time_step_seconds"] == 900.0


@pytest.mark.unit
def test_history_payload_trims_to_the_trailing_window():
    """A trimmed history is the tail of the full one, re-dated to its start."""
    terms = ArimaTerms(p=1, d=1, q=3, n_exog=0, has_const=False, has_drift=True)
    eta = np.arange(1.0, 9.0)
    eps = np.linspace(0.1, 0.7, 7)
    full = history_payload(eta, eps, terms, "2024-01-01T00:00:00", 900.0)

    trimmed = history_payload(
        eta, eps, terms, "2024-01-01T00:00:00", 900.0, history_steps=2
    )

    # seed = 3 lags ahead of the 2 kept points, all real fitted values.
    assert trimmed["eta_values"] == full["eta_values"][-5:]
    assert trimmed["eps_values"] == full["eps_values"][-5:]
    assert trimmed["eta_values"][:3] == [4.0, 5.0, 6.0]
    assert trimmed["start_date"] == "2024-01-01T01:30:00"


@pytest.mark.unit
def test_history_payload_zero_steps_keeps_only_the_terminal_state():
    """history_steps=0 keeps max(p+d, q) lags, dated one step after the end."""
    terms = ArimaTerms(p=2, d=0, q=1, n_exog=0, has_const=True, has_drift=False)
    eta = np.array([1.0, 2.0, 3.0, 4.0])
    eps = np.array([0.1, 0.2, 0.3, 0.4])

    payload = history_payload(
        eta, eps, terms, "2024-01-01T00:00:00", 3600.0, history_steps=0
    )

    assert payload["eta_values"] == [3.0, 4.0]
    assert payload["eps_values"] == [0.3, 0.4]
    assert payload["start_date"] == "2024-01-01T04:00:00"


@pytest.mark.unit
def test_history_payload_caps_steps_at_the_series_length():
    terms = ArimaTerms(p=1, d=0, q=0, n_exog=0, has_const=True, has_drift=False)
    eta = np.array([1.0, 2.0, 3.0])
    eps = np.array([0.0, 0.1, 0.2])

    full = history_payload(eta, eps, terms, "2024-01-01T00:00:00", 900.0)
    capped = history_payload(
        eta, eps, terms, "2024-01-01T00:00:00", 900.0, history_steps=50
    )

    assert capped == full


@pytest.mark.unit
@pytest.mark.parametrize(
    "history_length, step, expected",
    [
        (None, 900.0, None),
        (0, 900.0, 0),
        ("7D", 900.0, 672),
        (pd.Timedelta(hours=3), 3600.0, 3),
        (datetime.timedelta(minutes=90), 3600.0, 2),
        ("0s", 900.0, 0),
    ],
)
def test_history_steps_converts_durations_to_whole_steps(
    history_length, step, expected
):
    assert history_steps(history_length, step) == expected


@pytest.mark.unit
@pytest.mark.parametrize("history_length", [5, 1.5, True, "-1D", "not a duration"])
def test_history_steps_rejects_invalid_lengths(history_length):
    with pytest.raises(FlexConfigError, match="history_length"):
        history_steps(history_length, 900.0)


@pytest.mark.unit
def test_surrogate_coefficients_name_the_deterministic_term_by_order():
    """``intercept`` for d=0, ``drift`` for d=1, and nothing without one."""
    with_const = ArimaTerms(p=1, d=0, q=0, n_exog=1, has_const=True, has_drift=False)
    with_drift = ArimaTerms(p=0, d=1, q=1, n_exog=0, has_const=False, has_drift=True)
    bare = ArimaTerms(p=1, d=0, q=0, n_exog=0, has_const=False, has_drift=False)

    assert surrogate_coefficients(np.array([0.1, 0.5, 2.0]), with_const) == {
        "order": [1, 0, 0],
        "intercept": 0.1,
        "ar_coefs": [0.5],
        "exog_coefs": [2.0],
    }
    assert surrogate_coefficients(np.array([0.1, 0.3]), with_drift) == {
        "order": [0, 1, 1],
        "drift": 0.1,
        "ma_coefs": [0.3],
    }
    assert surrogate_coefficients(np.array([0.5]), bare) == {
        "order": [1, 0, 0],
        "ar_coefs": [0.5],
    }


# -- auto order search -------------------------------------------------------


def _simulate_arma(ar, ma, n, seed, burn=200):
    """Return ``n`` rows of a zero-mean ARMA process with unit innovations."""
    eps = np.random.default_rng(seed).normal(size=n + burn)
    z = np.zeros(n + burn)
    for t in range(n + burn):
        z[t] = eps[t]
        z[t] += sum(a * z[t - j - 1] for j, a in enumerate(ar) if t > j)
        z[t] += sum(m * eps[t - j - 1] for j, m in enumerate(ma) if t > j)
    return z[burn:]


def _search(y, x=None, **overrides):
    options = auto_search_options(overrides)
    return auto_select_order(
        y, x, include_mean=True, max_ar_persistence=None, **options
    )


@pytest.mark.unit
def test_kpss_separates_white_noise_from_a_random_walk():
    """KPSS stays below the 5% critical value for noise and exceeds it for a walk."""
    noise = np.random.default_rng(0).normal(size=400)
    assert kpss_statistic(noise) < KPSS_CRITICAL_5PCT
    assert kpss_statistic(np.cumsum(noise)) > KPSS_CRITICAL_5PCT
    assert kpss_statistic(np.full(50, 3.0)) == 0.0


@pytest.mark.unit
def test_select_differencing_tests_the_disturbance_not_y():
    """A trend the exogenous input explains is not mistaken for a unit root."""
    rng = np.random.default_rng(1)
    trend = np.linspace(0.0, 50.0, 300)
    y = 2.0 * trend + rng.normal(size=300)
    assert select_differencing(y, None, max_d=1) == 1
    assert select_differencing(y, trend[:, None], max_d=1) == 0
    assert select_differencing(y, None, max_d=0) == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ar", "ma", "expected"),
    [([0.6, -0.3], [], (2, 0, 0)), ([], [0.7], (0, 0, 1)), ([0.5], [0.4], (1, 0, 1))],
)
def test_auto_select_order_recovers_the_generating_order(ar, ma, expected):
    """On a long simulated series the AIC search finds the true order."""
    y = 5.0 + _simulate_arma(ar, ma, n=1500, seed=3)
    assert _search(y, max_p=3, max_q=3) == expected


@pytest.mark.unit
def test_auto_select_order_differences_a_random_walk():
    """A random walk with AR(1) increments is fitted as d=1."""
    y = np.cumsum(_simulate_arma([0.5], [], n=800, seed=4))
    assert _search(y, max_p=2, max_q=2) == (1, 1, 0)
    assert _search(y, max_p=2, max_q=2, stationary=True)[1] == 0
    assert _search(y, max_p=2, max_q=2, d=0)[1] == 0


@pytest.mark.unit
def test_auto_select_order_respects_max_order():
    """No candidate with p + q above max_order is considered."""
    y = _simulate_arma([0.5], [0.4], n=1000, seed=5)
    p, _d, q = _search(y, max_p=3, max_q=3, max_order=1)
    assert p + q <= 1


@pytest.mark.unit
def test_auto_select_order_with_exogenous_input():
    """An exogenous effect is fitted out: the order matches the bare disturbance."""
    rng = np.random.default_rng(6)
    x = rng.uniform(0.0, 2.0, size=(1000, 1))
    disturbance = _simulate_arma([0.7], [], n=1000, seed=6)
    y = 3.0 * x[:, 0] + disturbance
    assert _search(y, x, max_p=2, max_q=2) == _search(disturbance, max_p=2, max_q=2)


@pytest.mark.unit
def test_auto_select_order_skips_orders_the_data_cannot_identify():
    """Short series drop large candidates; with none left it is a data error."""
    y = _simulate_arma([0.5], [], n=8, seed=7)
    p, _d, q = _search(y, d=0)
    assert max(p, q) + 0 + 1 + p + q + 1 <= len(y)
    with pytest.raises(FlexDataError, match="too few to fit"):
        _search(y[:1], d=0)


@pytest.mark.unit
def test_auto_search_options_defaults_and_stationary():
    """Defaults fill in, and stationary=True is folded into max_d=0."""
    assert auto_search_options({}) == {
        "max_p": 5,
        "max_q": 5,
        "max_order": 5,
        "max_d": 1,
        "d": None,
    }
    assert auto_search_options({"stationary": True})["max_d"] == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"approximation": True}, "does not accept"),
        ({"max_p": -1}, "non-negative int"),
        ({"max_q": 1.5}, "non-negative int"),
        ({"max_order": True}, "non-negative int"),
        ({"max_d": 2}, "at most d=1"),
        ({"d": True}, "at most d=1"),
        ({"stationary": "yes"}, "must be a bool"),
        ({"stationary": True, "d": 1}, "cannot be combined"),
    ],
)
def test_auto_search_options_reject_invalid(kwargs, match):
    with pytest.raises(FlexConfigError, match=match):
        auto_search_options(kwargs)
