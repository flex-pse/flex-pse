# ARIMA Surrogate and Latent-State Estimation Plan

## 1. Objective

Implement one mathematically explicit Pyomo ARIMA/ARIMAX equation that supports:

1. Forecasting with fixed fitted coefficients.
2. Direct regression when the ARIMA output `y[t]` is observed.
3. Indirect joint estimation when `y[t]` is latent and only downstream variables are observed.

The third case is the primary design constraint. For example:

```text
ARIMA:       y[t] = dynamic_mean[t] + eps[t]
Correlation: b[t] = f(y[t]; gamma)
Observation: b_obs[t]
```

Both ARIMA coefficients and `gamma` may be estimated in one larger Pyomo model.

This plan supports only non-seasonal `d in {0, 1}`. Seasonal terms and `d > 1` are out of scope.

---

## 2. Fundamental Modeling Decision

`y[t]` is an endogenous state. `eps[t]` is its innovation or process disturbance.

`eps[t]` is not a fitted coefficient. Given `y`, model coefficients, exogenous inputs, and initial state, the ARIMA equations determine `eps[t]` recursively.

When `y[t]` is latent, downstream measurement error alone is not sufficient. Free innovations can reproduce any `y[t]` path. Joint estimation therefore needs an innovation penalty in the global objective.

There are only two valid choices:

1. Free `eps[t]` and penalize it. This retains ARIMA/MA estimation.
2. Fix `eps[t] = 0`. This gives a deterministic AR/ARX trajectory; MA coefficients are then not identifiable and must remain fixed or be omitted.

The implementation will use choice 1 during estimation and choice 2 during mean forecasting.

---

## 3. Governing Equations

### 3.1 Transformed output

Define `z[t]` as:

```text
z[t] = y[t]                    when d = 0
z[t] = y[t] - y[t-1]           when d = 1
```

### 3.2 ARIMAX equation

For each local model time `t >= 0`:

```text
z[t] = a[d]
       + sum(phi[i] * z[t-i], i=1..p)
       + sum(theta[j] * eps[t-j], j=1..q)
       + sum(beta[k] * x[t,k], k=1..K)
       + eps[t]
```

Deterministic term convention:

- For `d=0`, `a[0]` is named `intercept`. It is an intercept in the level equation.
- For `d=1`, `a[1]` is named `drift`. It is a constant in the differenced equation and creates linear drift in levels.
- A `drift * t` term in the differenced equation is not supported. It would create quadratic trend in levels.

Exogenous term convention:

- `beta[k] * x[t,k]` enters the transformed `z[t]` equation directly.
- This is the project ARIMAX parameterization and must match `_fit_direct()` exactly.
- This is not claimed to match every library's “regression with ARIMA errors” parameterization.

### 3.3 Level form used by `Surrogate.build()`

`OpsBlock.swap_relation()` creates `target[t] == body(t)`. Therefore `body(t)` must include the current innovation.

```text
d = 0:
    body(t) = mean_z[t] + eps[t]

d = 1:
    body(t) = y[t-1] + mean_z[t] + eps[t]
```

where:

```text
mean_z[t] = deterministic
            + AR terms
            + lagged MA terms
            + exogenous terms
```

There must be no second residual constraint. Adding both `target[t] == body(t)` and `eps[t] == target[t] - mean_z[t]` would duplicate the equation and can force `eps[t]` to zero.

---

## 4. Variable Roles

### 4.1 Regressable coefficients

These are fixed by default and registered in `CoefficientRegistry`:

| Name | Symbol | Units | Present when |
|---|---|---|---|
| `intercept` | `a[0]` | output units | `d=0` and enabled |
| `drift` | `a[1]` | output units per sample | `d=1` and enabled |
| `ar_coefs[i]` | `phi[i]` | dimensionless | `p>0` |
| `ma_coefs[j]` | `theta[j]` | dimensionless | `q>0` |
| `exog_coefs[k]` | `beta[k]` | output/input units | exogenous inputs exist |

An estimation workflow explicitly unfixes selected coefficients.

### 4.2 Latent dynamic variables

- `target[t]` is `y[t]`. It may be fixed, measured with error, or fully latent.
- `eps[t]` is the innovation. It is fixed to zero by default for forecasting and unfixed for estimation.
- `initial_y_history[h]` contains pre-horizon output levels.
- `initial_eps_history[h]` contains pre-horizon innovations.

Initial-state variables are not model coefficients and are not registered in `CoefficientRegistry`.

### 4.3 Observations

Observed values such as `b_obs[t]` are immutable `Param` data. They do not belong in the ARIMA persisted surrogate specification.

---

## 5. Minimal Initial State

The local Pyomo horizon always starts at `t=0`. It does not index into a full training series.

Required pre-horizon state:

```text
y_history length   = p       when d=0
y_history length   = p + 1   when d=1
eps_history length = q
```

Equivalently, `y_history` has length `p + d`.

History is ordered from oldest to newest. The final entry is the value immediately before local `t=0`.

For `d=1`, one extra level is required because each lagged `z` value is a difference of two levels. When `p=0`, one previous level is still required to integrate the first forecast or estimated difference.

`eps_history` is normally fixed to residuals from a previous fitted window. If unavailable, conditional estimation fixes it to zero. It must never be left free without a penalty or prior.

Full `training_y_values`, `_residuals`, `training_start_date`, and offset arithmetic are not needed by the equation block.

---

## 6. Runtime Modes

Mode is runtime estimation state, not persisted model data. The same built equations support every mode through variable fixation.

### 6.1 Mean forecast

```text
coefficients          fixed
initial_y_history     fixed
initial_eps_history   fixed
eps[t]                fixed to zero
y[t]                  free
```

This computes the conditional mean forecast. The first `q` steps can use nonzero `initial_eps_history`; later MA innovations are zero.

### 6.2 Direct ARIMA regression

```text
observed y[t]         fixed
selected coefficients free
initial_y_history     fixed, or estimated with a prior
initial_eps_history   fixed, normally zero
eps[t]                free
```

Objective contribution:

```text
J_arima = sum((eps[t] / sigma_eps[t])**2, t in fit_index)
```

This is conditional sum-of-squares estimation.

### 6.3 Indirect joint estimation

```text
y[t]                     free latent state
selected ARIMA coefficients free
selected f coefficients    free
initial_y_history          fixed or penalized
initial_eps_history        fixed, normally zero
eps[t]                     free
exogenous x[t,k]           fixed to data
b[t]                       constrained by b[t] = f(y[t]; gamma)
```

The top-level model owns one objective containing all measurement and process residuals.

---

## 7. Composite Estimation Objective

For one observed downstream variable:

```text
J_measurement = sum(((b[t] - b_obs[t]) / sigma_b[t])**2, t in observed_index)
J_innovation  = sum((eps[t] / sigma_eps[t])**2, t in fit_index)
J_initial     = optional initial-state prior penalty
J_parameter   = optional coefficient prior penalty

J_total = J_measurement + J_innovation + J_initial + J_parameter
```

For multiple units and correlations:

```text
J_total = sum(all normalized measurement residuals)
          + sum(all normalized ARIMA innovation residuals)
          + sum(all selected priors)
```

Every residual must be dimensionless. Raw squared errors from variables with different units or scales must not be added directly.

`sigma_b` represents measurement uncertainty. `sigma_eps` represents expected ARIMA process variation. Their ratio determines whether the optimizer favors fitting downstream measurements or following ARIMA dynamics.

If `sigma_eps` is estimated, the Gaussian log-variance term must also be present. Otherwise the optimizer can increase `sigma_eps` without bound. Initial implementation will treat scales as fixed estimation inputs.

The surrogate block must expose raw `eps[t]` and a helper that returns its normalized innovation objective expression. It must not create an active `Objective`, because a larger model can contain many surrogate blocks but should have one aggregate objective.

---

## 8. Identifiability Requirements

Indirect estimation can have enough equations and still be structurally non-identifiable.

Example:

```text
b[t] = gamma * y[t]
```

If `gamma`, `y[t]`, ARIMA scale terms, and initial level are all free, scaling `y` and inversely scaling `gamma` can leave every `b[t]` unchanged.

Each joint fit must provide enough anchors. At least one of these is normally required:

1. Some direct `y` observations.
2. A fixed or strongly regularized coefficient in `f(y; gamma)`.
3. A known initial `y` level.
4. Informative parameter priors.
5. Additional independently observed variables linked to `y`.

Physical bounds help numerics but do not prove identifiability.

Other requirements:

- Exogenous inputs must contain enough excitation to identify `beta` and dynamic coefficients.
- Effective observations must exceed the number of free coefficients and free initial-state quantities.
- AR and MA orders must remain small enough for the available horizon.
- AR stationarity and MA invertibility policy must be explicit. Root-based constraints or coefficient transformations are preferable to independent coefficient bounds for orders above one.
- Joint ARIMA/MA estimation is a nonconvex NLP because `theta[j] * eps[t-j]` is bilinear when both are free.

The estimation API must report free-variable/equality counts and warn that zero degrees of freedom does not prove structural identifiability.

---

## 9. Pyomo Block Contract

`ArimaSurrogate.build(unit, target)` will construct:

```text
block.coefficient_vars
block.coefficients                 CoefficientRegistry
block.eps[t]
block.initial_y_history[h]
block.initial_eps_history[h]
block.innovation_square[t]         Expression
block.fitted[t]                    added by swap_relation()
```

Default fixation after build:

- Coefficients are fixed to persisted values.
- Initial-state variables are fixed to persisted values.
- Future `eps[t]` values are fixed to zero.

Estimation setup explicitly changes fixation. Avoid hidden behavior based on the presence or absence of timestamps.

Lag accessors have one responsibility:

- `level_at(t)` returns `target[t]` for `t>=0`, otherwise the matching `initial_y_history` value.
- `difference_at(t)` returns `level_at(t) - level_at(t-1)`.
- `innovation_at(t)` returns `block.eps[t]` for `t>=0`, otherwise the matching `initial_eps_history` value.

No accessor reads a full training array or converts between local and global indices.

### 9.1 Regression helper API

The block must provide a pure helper:

```text
block.get_regression_objective(
    innovation_scale=None,
    time_index=None,
) -> Pyomo numeric expression
```

The returned expression is:

```text
sum((block.eps[t] / innovation_scale[t])**2 for t in selected_time_index)
```

API rules:

- The method returns a Pyomo expression, not an `Objective` component.
- The method has no model-state side effects. It does not fix or unfix variables.
- `time_index=None` selects the complete local horizon.
- `innovation_scale=None` defaults to `1.0` in the declared output units and emits a user-visible warning.
- The warning states that the default gives unit numerical weighting, not an estimated innovation standard deviation.
- Supplying `innovation_scale=1.0` explicitly accepts this weighting and suppresses the warning.
- `innovation_scale` accepts one positive scalar or a positive value indexed by time.
- A bare numeric scale is interpreted in the declared output units.
- A units-bearing scale must be compatible with the output units.
- Unknown time indices, nonpositive scales, and incompatible units raise `FlexConfigError`.
- The expression includes only the required ARIMA innovation penalty. It does not include downstream measurement errors or optional anchor penalties.

Explicit separation keeps regression setup easy to inspect:

```text
block.coefficients.unfix() # should include eps

arima_error = block.get_regression_objective(
    innovation_scale=sigma_eps,
    time_index=fit_index,
)

model.objective = pyo.Objective(
    expr=measurement_error + arima_error + optional_anchor_penalties
)
```

Forecast setup remains equally explicit:

```text
block.coefficients.fix()
block.eps.fix(0.0)
```

Do not combine variable-fixation changes with `get_regression_objective()`. A method named `get_*` must be safe to call repeatedly while assembling a global objective.

Recommended warning text:

```text
ARIMA regression objective is defaulting innovation_scale to 1.0 <output units>.
This is a numerical weighting, not an estimated innovation standard deviation.
Pass innovation_scale explicitly to control its weight relative to other objective terms.
```

When two residual families should have equal standardized weight, callers may explicitly use `1.0` in each residual's own units. A numeric `1:1` choice does not mean the physical units are directly comparable.

---

## 10. Persisted Data Contract

```text
data = {
    "input_variables":  {name: units, ...},
    "output_variables": {name: units},
    "coefficients": {
        "order": [p, d, q],
        "intercept": value,          # optional; d=0 only
        "drift": value,              # optional; d=1 only
        "ar_coefs": [p values],
        "ma_coefs": [q values],
        "exog_coefs": [K values],
    },
    "initial_state": {
        "y_history": [p+d values],
        "eps_history": [q values],
    },
}
```

Validation rules:

- Coefficient and history lengths must exactly match `(p,d,q,K)`.
- `intercept` is invalid for `d=1`.
- `drift` is invalid for `d=0`.
- Unsupported seasonal keys are rejected.
- Missing state is an error for forecasting.
- Estimation code may construct an initial-state guess explicitly, then unfix selected state variables.

Runtime-only estimation inputs remain outside persisted data:

- Observed values and masks.
- Which coefficients or initial states are free.
- Measurement and innovation scales.
- Prior definitions.
- Solver options.

---

## 11. Example Joint Model

Conceptual model:

```text
ARIMA block:
    y[t] = ARIMAX_mean(t, y_history, eps_history, x, coefficients) + eps[t]

Second surrogate:
    b[t] = f(y[t], gamma)

Fixed data:
    x[t]
    b_obs[t]

Free variables:
    y[t]
    eps[t]
    selected ARIMA coefficients
    selected gamma coefficients

Global objective:
    minimize measurement_error(b, b_obs)
             + arima_block.get_regression_objective(sigma_eps)
             + selected priors
```

No direct `y_true[t]` term is required. The downstream observations infer `y[t]`, while the innovation penalty makes the inferred trajectory obey ARIMA dynamics as closely as its assumed process variance requires.

---

## 12. Implementation Sequence

### Phase 1: Freeze mathematical behavior with tests

Write deterministic failing tests before implementation:

1. Verify `body(t)` includes current `eps[t]` and each lag exactly once.
2. Verify `d=0` and `d=1` equations against hand calculations.
3. Verify pre-horizon `y_history` and `eps_history` indexing.
4. Verify `swap_relation()` creates only one governing equation per active `t`.
5. Verify fixed `eps[t]=0` gives recursive mean forecasts.
6. Verify nonzero `eps_history` affects only the first `q` forecast steps.
7. Verify `get_regression_objective()` returns the expected symbolic normalized SSE.
8. Verify calling `get_regression_objective()` does not change any variable fixation.
9. Verify scalar, indexed, and units-bearing innovation scales.
10. Verify omitted scale defaults to one output unit and emits the documented warning.
11. Verify explicit `innovation_scale=1.0` suppresses the default warning.

### Phase 2: Direct regression equivalence

1. Fix `y[t]` to deterministic synthetic observations.
2. Unfix selected coefficients and `eps[t]`.
3. Minimize innovation SSE.
4. Match Pyomo innovations, objective, coefficients, and fitted path against `_arma_residuals()` using identical initialization.
5. Cover `(0,0,0)`, AR, MA, ARMA, `d=1`, and exogenous cases.

### Phase 3: Latent-state estimation

Use staged tests to isolate identifiability:

1. Fix downstream correlation coefficients; recover latent `y[t]` from `b_obs[t]`.
2. Fix ARIMA coefficients; recover downstream coefficients.
3. Jointly estimate both groups with one explicit anchor.
4. Demonstrate that removing the anchor produces degeneracy or non-unique solutions.
5. Infer missing `y` intervals from downstream observations and ARIMA dynamics.

### Phase 4: Multiple-block objective composition

1. Build two or more ARIMA/correlation chains in one model.
2. Aggregate their normalized measurement and innovation expressions into one objective.
3. Verify changing one residual scale changes the intended weighting without unit errors.
4. Verify each block exposes its own residual expressions without adding an active objective.
5. Verify expressions returned by multiple `get_regression_objective()` calls can be summed directly.

### Phase 5: Persist and forecast

1. Extract fitted coefficients.
2. Extract only the final `p+d` output levels and final `q` innovations as `initial_state`.
3. Build a fresh forecast model from the resulting specification.
4. Verify forecast continuity and direct-backend equivalence.

---

## 13. Files Expected to Change During Implementation

- `src/flexops/surrogates/arima.py`
- `src/flexops/tests/surrogates/test_arima.py`
- `src/flexparameterize/regression/arima.py`
- `src/flexparameterize/tests/regression/test_arima.py`
- `src/flexparameterize/tests/regression/test_arima_surrogate.py`

Joint-objective helpers belong in the estimation layer if reusable support is added. The ARIMA surrogate itself must remain objective-agnostic.

---

## 14. Acceptance Conditions

Implementation is complete only when:

1. One equation per time step defines `y[t]`; no duplicate residual equation exists.
2. Direct and indirect regression use `eps[t]` as innovations and penalize them in the top-level objective.
3. Full MA coefficients affect regression and are not multiplied only by fixed zeros.
4. Forecasts need only `p+d` prior output levels and `q` prior innovations.
5. No full training output series or timestamp offset is required to build a forecast horizon.
6. Direct Pyomo regression matches `_arma_residuals()` under the same initialization convention.
7. A synthetic anchored joint model recovers latent `y`, ARIMA coefficients, and downstream coefficients within declared tolerances.
8. An unanchored joint model is documented and tested as non-identifiable.
9. Every combined objective term is dimensionless and explicitly weighted.
10. `get_regression_objective()` returns a composable expression and never creates an objective or changes fixation.
11. Seasonal orders and `d>1` fail validation with actionable errors.

---

## 15. Deferred Work

- Exact maximum-likelihood or state-space estimation.
- Estimated process and measurement variances.
- Seasonal ARIMA terms.
- Differencing orders above one.
- Automatic structural-identifiability analysis.
- Global optimization for bilinear MA estimation.
