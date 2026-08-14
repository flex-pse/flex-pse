"""The constant energy-intensity regressor: one coefficient, no free parameters."""

import math

import pandas as pd

from flexcore import nomenclature as nm
from flexcore.config.schema import SurrogateSpec
from flexcore.exceptions import FlexConfigError, FlexDataError

COEFFICIENT_NAME = nm.INTENSITY_VARS[nm.PowerKind.ELECTRICAL]
"""str: key the fitted coefficient is stored under in a ``SurrogateSpec``; it
is also the name of the process parameter the fit determines on a unit."""


def constant_intensity_coefficient(surrogate: SurrogateSpec) -> float:
    """Read the intensity out of a ``constant_intensity`` relationship.

    The one place either FlexParameterize direction gets the coefficient from,
    so a spec that a fit did not produce is rejected the same way in both.

    Args:
        surrogate: A ``constant_intensity``
            :class:`~flexcore.config.schema.SurrogateSpec`.

    Returns:
        The intensity, in kWh/m^3.

    Raises:
        FlexConfigError: If the spec carries no coefficient under
            :data:`COEFFICIENT_NAME`.
    """
    if COEFFICIENT_NAME not in surrogate.coefficients:
        raise FlexConfigError(
            f"A 'constant_intensity' relationship must carry its coefficient "
            f"under {COEFFICIENT_NAME!r}; got {sorted(surrogate.coefficients)}.",
            field="coefficients",
            value=sorted(surrogate.coefficients),
        )
    return surrogate.coefficients[COEFFICIENT_NAME]


def _single_column(frame, role: str) -> pd.Series:
    """Return the one column of ``frame`` as a Series.

    Args:
        frame: A one-column ``DataFrame`` (or a ``Series``).
        role: ``"X"`` or ``"y"``, for the error message.

    Returns:
        The column as a named Series.

    Raises:
        FlexDataError: If ``frame`` does not hold exactly one column.
    """
    if isinstance(frame, pd.Series):
        return frame
    if frame.shape[1] != 1:
        raise FlexDataError(
            f"ConstantIntensityRegressor fits one column against one column; "
            f"{role} has {frame.shape[1]} ({list(frame.columns)}). Select the "
            "single flow (X) and power (y) column, or use a multi-input "
            "regressor.",
            field=role,
        )
    return frame.iloc[:, 0]


class ConstantIntensityRegressor:
    """Fits the mean energy intensity of a unit from paired flow and power data.

    The simplest zero-degree-of-freedom fit: one input, one output, one
    coefficient. Rows where either side is null, or where the flow is zero, are
    dropped, and the coefficient is the **mean of the per-row ratios**,
    ``mean(power / flow)`` — not a least-squares slope. The mean ratio weights
    every operating point equally, which is the behaviour an operator expects
    from an "average kWh per m^3" number, and it recovers a noise-free data set
    exactly (which is what the round-trip invariant asserts). A least-squares
    slope through the origin would weight high-flow points more heavily.

    Fit results are stored on the instance.

    Attributes:
        coefficient: The fitted intensity, in the ratio of y's units to X's
            (kW over m^3/hr, i.e. kWh/m^3).
        n_samples: Number of rows the fit used, after dropping.
        metrics: ``{"r2": ..., "rmse": ...}`` of ``coefficient * flow`` against
            the observed power.
        data_window: ``(first, last)`` index value of the rows used.
        input_variable: Column name of the fitted input.
        output_variable: Column name of the fitted output.

    Example:
        >>> import pandas as pd
        >>> flow = pd.DataFrame({"flow_in": [1.0, 2.0, 4.0]})
        >>> power = pd.DataFrame({"power_electrical": [0.5, 1.0, 2.0]})
        >>> ConstantIntensityRegressor().fit(flow, power).coefficient
        0.5
    """

    def __init__(self) -> None:
        self.coefficient: float | None = None
        self.n_samples: int = 0
        self.metrics: dict[str, float] = {}
        self.data_window: tuple = ()
        self.input_variable: str = ""
        self.output_variable: str = ""

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "ConstantIntensityRegressor":
        """Fit the mean energy intensity of ``y`` against ``X``.

        Args:
            X: One flow column (a one-column ``DataFrame`` or a ``Series``).
            y: One power column, sharing ``X``'s index.

        Returns:
            ``self``, fitted.

        Raises:
            FlexDataError: If either side does not hold exactly one column, or
                no row survives dropping nulls and zero-flow rows.
        """
        flow = _single_column(X, "X")
        power = _single_column(y, "y")
        paired = pd.concat([flow.rename("flow"), power.rename("power")], axis=1)
        usable = paired.dropna()
        usable = usable[usable["flow"] != 0]
        if usable.empty:
            raise FlexDataError(
                f"No usable rows to fit {power.name!r} against {flow.name!r}: "
                f"every row is null or has zero flow (of {len(paired)} rows). "
                "Supply data covering periods when the unit was running.",
                field=str(flow.name),
            )

        coefficient = float((usable["power"] / usable["flow"]).mean())
        residual_ss = float(
            ((usable["power"] - coefficient * usable["flow"]) ** 2).sum()
        )
        total_ss = float(((usable["power"] - usable["power"].mean()) ** 2).sum())

        self.coefficient = coefficient
        self.n_samples = len(usable)
        self.metrics = {
            # A constant power series has no variance to explain: R^2 is then 1
            # if the fit reproduces it exactly and 0 if it does not.
            "r2": (
                1.0 - residual_ss / total_ss
                if total_ss > 0
                else float(residual_ss == 0.0)
            ),
            "rmse": math.sqrt(residual_ss / len(usable)),
        }
        self.data_window = (usable.index.min(), usable.index.max())
        self.input_variable = str(flow.name)
        self.output_variable = str(power.name)
        return self

    def to_surrogate_spec(self) -> SurrogateSpec:
        """Return the fit as a persistable ``SurrogateSpec``.

        The one place a constant-intensity fit becomes a spec: both
        FlexParameterize directions
        (:func:`~flexparameterize.apply.apply_to_model` and
        :func:`~flexparameterize.emit.emit_model_config`) consume this, so the
        two can never disagree about what was fitted.

        Returns:
            A ``constant_intensity`` :class:`~flexcore.config.schema.SurrogateSpec`
            carrying the coefficient under
            :data:`COEFFICIENT_NAME` and the fitted column names.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if self.coefficient is None:
            raise FlexDataError(
                "ConstantIntensityRegressor has no fit yet; call fit(X, y) "
                "before to_surrogate_spec()."
            )
        return SurrogateSpec(
            functional_form="constant_intensity",
            coefficients={COEFFICIENT_NAME: self.coefficient},
            input_variables=[self.input_variable],
            output_variables=[self.output_variable],
        )
