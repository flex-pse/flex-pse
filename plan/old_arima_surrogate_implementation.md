# ARIMA Surrogate Implementation Plan

## 1. Current State

### Surrogate (`src/flexops/surrogates/arima.py`)
- `build()` returns `(block, body(t))`
- Block carries `coefficient_vars` indexed Var + `CoefficientRegistry`
- All coefficients are individual fixed Vars: `const`, `ar1`, `ma1`, exog names
- `_residuals` and `init_values` are inlined Python floats on the block
- `body(t)` is a closure using baked `_residuals` for MA terms

### Regressor (`src/flexparameterize/regression/arima.py`)
- `to_surrogate_spec()` emits flat `coefficients` dict: `order`, `const`, `drift`, `ar1..`, `ma1..`, exog names
- Also emits `_residuals`, `init_values`, `training_*` metadata

---

## 2. Design Goals

1. Enable ipopt regression of the full ARIMA model (all coefficients, including MA)
2. Match multilinear surrogate structure exactly (one flat `coefficients` dict)
3. Support mixed horizons: in-sample window + forecast window in the same model
4. Clean separation of structural data, trainable coefficients, and initialization data

---

## 3. Data Contract (Final)

```python
data = {
    # Structural (always present)
    "input_variables": {"feed_volume_kg": "kg", "TS_pct": "dimensionless"},
    "output_variables": {"biogas_m3_hour": "m^3/hr"},
    "coefficients": {
        "order": [p, d, q],              # [int, int, int]
        "const": <float>,                # always present, may be 0.0
        "drift": <float> | None,         # only when d=1; omit when absent
        "ar_coefs": [phi_1, ..., phi_p], # list of p floats
        "ma_coefs": [theta_1, ..., theta_q], # list of q floats
        "exog_coefs": [beta_1, ..., beta_k], # list of k floats (1 per input)
    },
    # Initialization (needed for d=1 differencing, pre-offset AR lags)
    "init_values": [y0, ...],  # d values for d=1; p values for d=0
    # Training metadata (for alignment, warm-start, provenance)
    "training_start_date": "2025-01-01T00:00:00",
    "training_time_step_seconds": 3600.0,
    "training_y_values": [<float>, ...],  # full series (needed for pre-offset lags)
    # OPTIONAL
    "seasonal_order": [P, D, Q, m],
    # NOTE: _residuals is NOT part of the built model. May be kept in data
    # for provenance only, or dropped entirely.
}
```

**Key changes from current:**
- `coefficients` is the ONLY place for trainable parameters (flat dict, no nesting)
- `ar_coefs`, `ma_coefs`, `exog_coefs` are **lists** inside `coefficients`
- `exogenous_variables` top-level key is **removed** — `input_variables` IS the exogenous list
- `const`, `drift`, `order` move from top-level into `coefficients`
- `_residuals` removed from built model (computed via eps[t] Vars)

---

## 4. Coefficient Block Layout

```
block.coefficient_vars = {
    "const":      pyo.Var (scalar),
    "drift":      pyo.Var (scalar, optional),
    "ar_coefs":   pyo.Var(indexed over [1..p]),
    "ma_coefs":   pyo.Var(indexed over [1..q]),
    "exog_coefs": pyo.Var(indexed over [1..k]),
}
block.coefficients = CoefficientRegistry()  # registers all above
block.eps = pyo.Var(time_index)  # residual Vars, one per time step
```

**Access pattern:**
```python
block.coefficient_vars["const"]          # scalar
block.coefficient_vars["ar_coefs"][1]    # phi_1
block.coefficient_vars["ma_coefs"][2]    # theta_2
block.coefficient_vars["exog_coefs"][1]  # beta_1
block.coefficients["ar_coefs"][1]        # same, via registry
block.eps[t]                             # residual at time t
```

---

## 5. Residual Strategy (Critical Decision)

### Current approach (WRONG for regression)
- `_residuals` baked in as Python floats from training fit
- MA terms: `sum(theta_j * residuals[t-j])`
- Problem: residuals are frozen; MA coefficients can't respond to coefficient changes during regression

### Proposed approach: eps[t] Vars with residual constraint

**For in-sample (t < offset):**
- `y[t]` is fixed to observed values
- `eps[t]` is a **free Var**
- Constraint: `eps[t] == y[t] − (c + AR(y[t-j]) + MA(eps[t-j]) + exog[t])`
- This computes residuals consistently with current coefficients

**For forecast (t ≥ offset):**
- `y[t]` is free
- `eps[t]` is **fixed to 0** (mean forecast assumption)
- Constraint: `y[t] == c + AR(y[t-j]) + exog[t]` (MA terms are 0)
- Exception: at t = offset (first forecast step), MA uses eps[offset−1] which is the last in-sample residual (free Var)

**Residual constraint handles boundaries:**
- `eps[t-j]` for `t-j < 0` → 0 (no prior residual)
- `eps[t-j]` for `t-j >= offset` → 0 (forecast, mean assumption)
- `y[t-j]` for `t-j < 0` → `init_values` (d=0) or `training_y_values[offset + t - j]` (d=1)

**Regression objective:**
```
min  Σ (y[t] − y_obs[t])²   for t in in-sample window
```
with y[t] free but constrained by the ARIMA equation. Solver adjusts coefficients; eps[t] adjust consistently via the residual constraint.

---

## 6. Two-Mode Data Contract

### Mode 1: Deployment (reproduce fitted path)
```python
data = {
    "input_variables": {"feed_volume_kg": "kg"},
    "output_variables": {"biogas_m3_hour": "m^3/hr"},
    "coefficients": {
        "order": [p, d, q],
        "const": <float>,
        "drift": <float> | None,
        "ar_coefs": [phi_1..phi_p],
        "ma_coefs": [theta_1..theta_q],
        "exog_coefs": [beta_1..beta_k],
    },
    "init_values": [y0..],  # d or p values
    "training_start_date": "2025-01-01T00:00:00",  # REQUIRED
    "training_time_step_seconds": 3600.0,  # REQUIRED
    "training_y_values": [..],  # REQUIRED
    "seasonal_order": [P,D,Q,m],  # optional
}
```
- `offset` = (model_start - training_start) / dt
- In-sample (`t < offset`): eps[t] free, y[t] fixed to observed
- Forecast (`t >= offset`): eps[t] = 0, y[t] free

### Mode 2: Regression (build with coefficient guesses)
```python
data = {
    "input_variables": {"feed_volume_kg": "kg"},
    "output_variables": {"biogas_m3_hour": "m^3/hr"},
    "coefficients": {
        "order": [p, d, q],
        # All of the following are OPTIONAL in regression mode:
        "const": <float>,        # omit to exclude intercept
        "drift": <float>,        # omit to exclude drift (only meaningful d=1)
        "ar_coefs": [phi_1..],   # omit for pure MA/no-AR; defaults to [1.0]*p
        "ma_coefs": [theta_1..], # omit for pure AR/no-MA; defaults to [1.0]*q
        "exog_coefs": [beta_1..],# omit for no exog; defaults to [1.0]*k
    },
    "init_values": [y0..],  # optional, defaults to zeros
    # NO training metadata needed
}
```
- No `training_start_date` → `offset = 0`
- All `eps[t]` are free Vars
- Residual constraint: `eps[t] == y[t] - model(t)` for all t
- User fixes y[t] for observed steps, unfixes for forecast
- User unfixes coefficients, runs ipopt
- Omitted coefficient groups are **not added to the block at all** (not zeroed out)

### Default values for omitted coefficients (regression mode)

| Coefficient | Default if omitted | Block behavior |
|-------------|-------------------|----------------|
| `const` | not included | No intercept term in model |
| `drift` | not included | No drift term in model |
| `ar_coefs` | `[1.0] * p` | Indexed Var `ar_coefs[1..p]` added, initialized to 1.0 |
| `ma_coefs` | `[1.0] * q` | Indexed Var `ma_coefs[1..q]` added, initialized to 1.0 |
| `exog_coefs` | `[1.0] * k` | Indexed Var `exog_coefs[1..k]` added, initialized to 1.0 |

### Optional Keys
| Key | Mode 1 | Mode 2 | Default if absent |
|-----|--------|--------|-------------------|
| `training_start_date` | required | omitted | N/A |
| `training_time_step_seconds` | required | omitted | N/A |
| `training_y_values` | required | omitted | N/A |
| `init_values` | required | optional | zeros |
| `seasonal_order` | optional | optional | None |
| `_residuals` | **removed** | **removed** | N/A |
| `coefficients.const` | required (deploy) | optional | not included |
| `coefficients.drift` | only when d=1 | optional | not included |
| `coefficients.ar_coefs` | required (p>0) | optional | [1.0] * p |
| `coefficients.ma_coefs` | required (q>0) | optional | [1.0] * q |
| `coefficients.exog_coefs` | required (k>0) | optional | [1.0] * k |

---

## 7. `get_surrogate_spec()()` Helper

After solving, users need to extract the solved state to:
1. Refit the ARIMA model with new observed data
2. Build a new surrogate spec with the solved coefficients

```python
def get_surrogate_spec()(self, block, target, time_index) -> dict:
    """Extract solved coefficients, init_values, and training_y_values.
    
    Reads solved values from the coefficient Vars and target[t] after
    ipopt (or any solver) has converged.  Returns a dict suitable for
    constructing a new SurrogateSpec or passing to ArimaRegressor.fit().
    
    Args:
        block: The surrogate block returned by build().
        target: The time-indexed Var the surrogate constrains.
        time_index: The Pyomo time index set (e.g. m.time_block.time_index).
    
    Returns:
        Dict with keys:
            "coefficients": flat dict with order, const, drift, ar_coefs, ma_coefs, exog_coefs
            "init_values": list of floats (length p or d)
            "training_y_values": list of floats (solved y[t] for t < offset)
    """
```

The method centralizes:
- Reading `pyo.value()` from coefficient Vars (with proper indexing for ar_coefs, ma_coefs, exog_coefs)
- Reading `pyo.value(target[t])` for all t
- Slicing solved y[t] into init_values and training_y_values based on order and offset

---

## 8. Open Questions (resolved)

1. **Residuals**: Drop `_residuals` from data entirely. Use `eps[t]` Vars with residual constraint.
2. **MA regression**: Yes, MA coefficients become regressable with eps[t] Vars. Desired behavior.
3. **Index base**: 1-based (`ar_coefs[1]` = phi_1).
4. **Drift**: Omit from `coefficients` when d != 1.
5. **get_surrogate_spec()**: Method on `ArimaSurrogate` instance.
6. **Regression-only mode**: Supported — omit `training_start_date` and `training_y_values`, all eps[t] are free Vars.
7. **Optional coefficients in regression mode**: `ar_coefs`, `ma_coefs`, `exog_coefs` default to `[1.0]*n` if omitted; `const` and `drift` are omitted entirely if not provided.
