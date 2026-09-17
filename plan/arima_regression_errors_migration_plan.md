# Plan: Migration to Regression with ARIMA Errors Formulation

## 1. Objective & Motivation

Migrate `ArimaRegressor` (fitting/predicting backend) and `ArimaSurrogate` (Pyomo surrogate representation) from an ARMAX difference equation on undifferenced $X$ to the standard **Regression with ARIMA Errors** formulation (used in StatsModels, Hyndman's `forecast`, and standard time-series literature).

### Key Benefits:
1. **Physical Exogenous Coupling**: $\beta_k$ represents the direct level contribution of input $X_k$ to output $y$ ($y_t = \sum \beta_k X_{t,k} + \eta_t$).
2. **Eliminates $d=1$ Infinite Integration Bug**: With $d=1$, constant exogenous inputs no longer accumulate linearly into infinity or crash during multi-step optimization/forecasting.
3. **High Fidelity with StatsModels/StatsForecast**: Forecast discrepancy on real biogas data decreases from $>20-150\%$ down to $<0.5-4\%$.
4. **Preserves In-Pyomo IPOPT Regression**: Keeps all standalone Pyomo regression capabilities (`tmp_arima_validation.py`) fully functional.

---

## 2. Mathematical Definition

### 2.1 Regression with ARIMA Errors
Output $y_t$ is decomposed into a deterministic regression on exogenous inputs $X_t$ and an ARIMA disturbance $\eta_t$:

$$y_t = \sum_{k=1}^{K} \beta_k X_{t,k} + \eta_t$$

Equivalently, the disturbance is:
$$\eta_t = y_t - \sum_{k=1}^{K} \beta_k X_{t,k}$$

### 2.2 Disturbance Dynamics $\eta_t$
Let $z_t$ be the working series for the ARIMA process:
- **For $d = 0$**: $z_t = \eta_t$
- **For $d = 1$**: $z_t = \Delta \eta_t = \eta_t - \eta_{t-1} = (y_t - y_{t-1}) - \sum_{k=1}^K \beta_k (X_{t,k} - X_{t-1,k})$

The mean ARIMA equation on $z_t$ is:
$$z_t = c + \sum_{i=1}^p \phi_i z_{t-i} + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$

where $c$ is:
- Level intercept when $d=0$
- Constant drift in differenced disturbance when $d=1$

### 2.3 Expressed in Terms of $y_t$
- **For $d = 0$**:
  $$y_t = \sum_{k=1}^K \beta_k X_{t,k} + c + \sum_{i=1}^p \phi_i (y_{t-i} - \sum_k \beta_k X_{t-i,k}) + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$

- **For $d = 1$**:
  $$y_t = \sum_{k=1}^K \beta_k X_{t,k} + \eta_{t-1} + c + \sum_{i=1}^p \phi_i (\eta_{t-i} - \eta_{t-i-1}) + \sum_{j=1}^q \theta_j \epsilon_{t-j} + \epsilon_t$$
  where $\eta_{t-1} = y_{t-1} - \sum_k \beta_k X_{t-1,k}$.

---

## 3. Required Codebase Changes

### 3.1 `src/flexparameterize/regression/arima.py` (Fitting & Prediction)
1. **`_arma_residuals` (Residual Minimization)**:
   - Compute disturbance $\eta_t = y_t - \sum_k \beta_k X_{t,k}$.
   - Difference $\eta_t$ when $d=1$: $z_t = \eta_t$ ($d=0$) or $z_t = \Delta \eta_t$ ($d=1$).
   - Form residual $\epsilon_t = z_t - (c + \sum \phi_i z_{t-i} + \sum \theta_j \epsilon_{t-j})$.
   - Initial OLS guess for $\beta_0$: Standard OLS of $y$ on $X$.
2. **`_DirectResults.predict` (Dynamic & Out-of-Sample Forecasting)**:
   - Maintain historical $\eta_{\text{hist}}$ queue:
     - For $d=0$: $\eta_{\text{hist}} = [y_{t-p} - X_{t-p}\beta, \dots, y_{t-1} - X_{t-1}\beta]$.
     - For $d=1$: $\eta_{\text{hist}} = [y_{t-p-1} - X_{t-p-1}\beta, \dots, y_{t-1} - X_{t-1}\beta]$.
   - At each forecast step $h$:
     - Compute $\hat{\eta}_{t+h} = c + \sum \phi_i \eta_{\text{hist}} + \sum \theta_j \epsilon_{\text{hist}}$ (or with first difference for $d=1$).
     - Compute $\hat{y}_{t+h} = \sum \beta_k X_{t+h,k} + \hat{\eta}_{t+h}$.
     - Append $\hat{\eta}_{t+h}$ to $\eta_{\text{hist}}$ and $0.0$ to $\epsilon_{\text{hist}}$.
3. **`to_surrogate_spec` (History Serialization)**:
   - Persist $y_{\text{values}}$, $X_{\text{values}}$, and $\epsilon_{\text{values}}$ in `history`, OR serialize precomputed $\eta_{\text{values}} = y - X\beta$ directly.

### 3.2 `src/flexops/surrogates/arima.py` (Pyomo Surrogate)
1. **Historical Disturbance Seeding**:
   - In `build()`, compute initial pre-horizon disturbance history $\eta_{\text{history}} = y_{\text{history}} - \sum \beta_k X_{\text{history},k}$.
   - Stored in `initial_eta_history` Var (or derived from `initial_y_history` and `initial_x_history`).
2. **`body(t)` Pyomo Expression**:
   - Let $\eta(position)$ helper return:
     - For $position \ge 0$ (in-horizon): $y[t] - \sum_{k} \beta_k X_k[t]$
     - For $position < 0$ (pre-horizon): `block.initial_eta_history[history_index]`
   - Form $\text{body}(t)$ as:
     $$\text{body}(t) = \sum_{k} \beta_k X_k[t] + \text{ARIMA\_disturbance\_step}(\eta, \epsilon)$$
3. **In-Pyomo IPOPT Parameter Estimation (`get_surrogate_objective`)**:
   - `eps[t]` in Pyomo will directly match the NLS residual $\epsilon_t = \eta_t - \hat{\eta}_t$.
   - Unfixing $\beta_k, c, \phi_i, \theta_j$ allows IPOPT to minimize $\sum \epsilon_t^2$, exactly reproducing `ArimaRegressor` NLS fitting.

---

## 4. Preservation of In-Pyomo Regression (`tmp_arima_validation.py`)

`tmp_arima_validation.py` tests fitting ARIMA surrogates inside Pyomo using IPOPT standalone.

To guarantee zero regression in functionality:
1. **Fixed Endogenous Mode**: When $y[t]$ and $X[t]$ are fixed to observed data, $\eta[t] = y[t] - \sum \beta_k X_k[t]$ is a linear function of free $\beta_k$.
2. **Residual Identification**: The surrogate relation $y[t] == \text{body}(t)$ enforces $\epsilon[t] == \eta[t] - \hat{\eta}_t(\phi, \theta, c, \beta)$.
3. **Objective Optimization**: Minimizing $\sum \epsilon[t]^2$ via `get_surrogate_objective()` inside Pyomo solves the exact nonlinear least squares problem for $(c, \beta, \phi, \theta)$.
4. **Roundtrip Rebuild**: The extracted solved spec from IPOPT will plug into a new prediction model and match SciPy NLS forecasts 1-to-1.

---

## 5. Testing & Validation Plan

1. **Unit Tests (`test_arima.py`)**:
   - Validate direct NLS fits across all 14 orders against `statsmodels.SARIMAX` on real biogas data (`imputed_bio_gas_generation.csv`).
   - Expected forecast max error $< 5\%$ across all $d=0$ and $d=1$ orders (down from $>150\%$).
2. **Component Tests (`test_arima_surrogate.py`)**:
   - Validate 1-to-1 fidelity between `ArimaRegressor.predict()` and Pyomo surrogate IPOPT 0-DOF solve for all 14 orders.
   - Validate optimization roundtrip where IPOPT optimizes feed controls $X_t$ to hit in-sample biogas targets.
3. **Standalone Script (`tmp_arima_validation.py`)**:
   - Run both in-sample prediction and in-Pyomo parameter estimation mode.
   - Verify IPOPT recovers the same parameters as SciPy `least_squares`.
4. **Formatting, Linting & Docs**:
   - Ensure `ruff check`, `black --check`, and Sphinx docs build cleanly.
