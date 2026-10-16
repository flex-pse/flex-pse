"""The constant energy-intensity regressor: one coefficient, no free parameters."""

import math

import pandas as pd

from flexcore import nomenclature as nm
from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexDataError
from flexparameterize.regression.base import FitResult

COEFFICIENT_NAME = nm.INTENSITY_VARS[nm.PowerKind.ELECTRICAL]
"""str: key the fitted coefficient is stored under in a ``SurrogateSpec``'s
``data['coefficients']``; it is also the name of the process parameter the fit
determines on a unit."""


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

    def to_fit_result(self) -> FitResult:
        """Return this fit as the shared :class:`~.base.FitResult` shape.

        Returns:
            The coefficient under :data:`COEFFICIENT_NAME`, alongside the
            fit's metrics, sample count and data window.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if self.coefficient is None:
            raise FlexDataError(
                "ConstantIntensityRegressor has no fit yet; call fit(X, y) "
                "before to_fit_result()."
            )
        return FitResult(
            coefficients={COEFFICIENT_NAME: self.coefficient},
            metrics=self.metrics,
            n_samples=self.n_samples,
            data_window=self.data_window,
        )

    def to_surrogate_spec(
        self, *, input_units: str = "", output_units: str = ""
    ) -> SurrogateSpec:
        """Return the fit as a persistable ``SurrogateSpec``.

        The one place a constant-intensity fit becomes a spec, which
        :func:`~flexparameterize.apply.apply_to_model` consumes. ``constant_intensity``
        has no surrogate class (it fixes a process parameter rather than
        swapping a Constraint), so unlike a real surrogate's declared units,
        these are recorded for documentation, not enforced by a build.

        Args:
            input_units: Units of the fitted input column, as a string (the
                caller reads them off the model's registered variable).
            output_units: Units of the fitted output column.

        Returns:
            A ``SurrogateType.CONSTANT_INTENSITY``
            :class:`~flexcore.config.schema.SurrogateSpec` carrying the
            coefficient under :data:`COEFFICIENT_NAME` and the fitted column
            names.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if self.coefficient is None:
            raise FlexDataError(
                "ConstantIntensityRegressor has no fit yet; call fit(X, y) "
                "before to_surrogate_spec()."
            )
        return SurrogateSpec(
            surrogate_type=SurrogateType.CONSTANT_INTENSITY,
            data={
                "coefficients": {COEFFICIENT_NAME: self.coefficient},
                "input_variables": {self.input_variable: input_units},
                "output_variables": {self.output_variable: output_units},
            },
        )
