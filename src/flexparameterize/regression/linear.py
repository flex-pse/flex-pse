"""LinearRegressor: ordinary least squares against one or more input columns.

Available only with the ``[parameterize]`` extra -- scikit-learn is imported
lazily in :meth:`LinearRegressor.fit`, never at module import time, so
``import flexparameterize`` and ``import flexparameterize.regression`` always
succeed with no scikit-learn installed.
"""

import math

import pandas as pd

from flexcore.config.schema import SurrogateSpec, SurrogateType
from flexcore.exceptions import FlexConfigError, FlexDataError
from flexparameterize.regression.base import (
    FitResult,
    require_frame,
    single_column,
)


class LinearRegressor:
    """Fits a unit's output as an ordinary-least-squares function of its inputs.

    One output column, any number of input columns, degree 1 in each (no
    automatic cross terms -- engineer one into an ``X`` column named
    ``"a*b"`` before fitting if the relationship needs one). Rows with any
    null across ``X``/``y`` are dropped before fitting.

    Attributes:
        coefficients: Fitted column name -> coefficient, plus ``"intercept"``.
        n_samples: Number of rows the fit used, after dropping nulls.
        metrics: ``{"r2": ..., "rmse": ...}`` of the fitted line against ``y``.
        data_window: ``(first, last)`` index value of the rows used.
        input_variables: Column names of the fitted inputs, in order.
        output_variable: Column name of the fitted output.
        input_units: Units of every fitted input column, keyed by its column
            name.
        output_units: Units of the fitted output column.
    """

    def __init__(self) -> None:
        self.coefficients: dict[str, float] | None = None
        self.n_samples: int = 0
        self.metrics: dict[str, float] = {}
        self.data_window: tuple = ()
        self.input_variables: list[str] = []
        self.output_variable: str = ""
        self.input_units: dict[str, str] = {}
        self.output_units: str = ""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        *,
        input_units: dict[str, str],
        output_units: str,
    ) -> "LinearRegressor":
        """Fit an OLS line of ``y`` against ``X``'s columns.

        Args:
            X: One or more input columns.
            y: A one-column ``DataFrame`` holding the output.
            input_units: Units of every fitted input column, keyed by its
                column name (every column in ``X`` must have an entry).
                Recorded for :meth:`to_surrogate_spec`.
            output_units: Units of the fitted output column. Recorded for
                :meth:`to_surrogate_spec`.

        Returns:
            ``self``, fitted.

        Raises:
            FlexConfigError: If scikit-learn is not installed, or if
                ``input_units`` is missing an entry for one of ``X``'s
                columns.
            FlexDataError: If ``X`` or ``y`` is not a ``DataFrame``, if
                ``y`` does not hold exactly one column, or if fewer rows
                survive dropping nulls than
                ``X.shape[1] + 1`` (the minimum an OLS fit with an intercept
                needs to be determined).
        """
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError as exc:
            raise FlexConfigError(
                "LinearRegressor requires scikit-learn. Install it with "
                "`pip install 'flex-pse[parameterize]'`."
            ) from exc

        require_frame(X, regressor=type(self).__name__, role="X")
        missing = [name for name in X.columns if name not in input_units]
        if missing:
            raise FlexConfigError(
                f"fit is missing input_units for {missing}; every input "
                f"column ({list(X.columns)}) needs an entry.",
                field="input_units",
                value=missing,
            )

        output = single_column(
            y, regressor=type(self).__name__, role="y", what="output"
        )
        paired = pd.concat([X, output.rename("__y__")], axis=1).dropna()
        minimum = X.shape[1] + 1
        if len(paired) < minimum:
            raise FlexDataError(
                f"LinearRegressor needs at least {minimum} row(s) to fit "
                f"{X.shape[1]} input column(s) plus an intercept; only "
                f"{len(paired)} survived dropping null rows.",
                field="X",
            )

        fitted = LinearRegression(fit_intercept=True).fit(
            paired[list(X.columns)], paired["__y__"]
        )
        predicted = fitted.predict(paired[list(X.columns)])
        residual_ss = float(((paired["__y__"] - predicted) ** 2).sum())

        self.coefficients = dict(zip(X.columns, fitted.coef_, strict=True)) | {
            "intercept": float(fitted.intercept_)
        }
        self.n_samples = len(paired)
        self.metrics = {
            "r2": float(fitted.score(paired[list(X.columns)], paired["__y__"])),
            "rmse": math.sqrt(residual_ss / len(paired)),
        }
        self.data_window = (paired.index.min(), paired.index.max())
        self.input_variables = list(X.columns)
        self.output_variable = str(output.name)
        self.input_units = dict(input_units)
        self.output_units = output_units
        return self

    def to_fit_result(self) -> FitResult:
        """Return this fit as the shared :class:`~.base.FitResult` shape.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if self.coefficients is None:
            raise FlexDataError(
                "LinearRegressor has no fit yet; call fit(X, y) before "
                "to_fit_result()."
            )
        return FitResult(
            coefficients=self.coefficients,
            metrics=self.metrics,
            n_samples=self.n_samples,
            data_window=self.data_window,
        )

    def to_surrogate_spec(self) -> SurrogateSpec:
        """Return the fit as a ``multilinear`` ``SurrogateSpec``.

        Uses the ``input_units``/``output_units`` recorded by :meth:`fit`.

        Returns:
            A ``SurrogateType.MULTILINEAR`` spec whose ``data`` is exactly
            :class:`~flexops.surrogates.multilinear.MultilinearSurrogate`'s
            contract: the fitted column names, verbatim, are both the
            ``input_variables`` keys and the (single-factor) coefficient keys.

        Raises:
            FlexDataError: If :meth:`fit` has not been called.
        """
        if self.coefficients is None:
            raise FlexDataError(
                "LinearRegressor has no fit yet; call fit(X, y) before "
                "to_surrogate_spec()."
            )
        return SurrogateSpec(
            surrogate_type=SurrogateType.MULTILINEAR,
            data={
                "input_variables": {
                    name: self.input_units[name] for name in self.input_variables
                },
                "output_variables": {self.output_variable: self.output_units},
                "coefficients": self.coefficients,
            },
        )
