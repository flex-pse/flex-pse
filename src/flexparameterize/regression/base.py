"""The pluggable-regressor seam: ``Regressor`` Protocol and ``FitResult``.

Every regressor fits on instance attributes and reduces its own fit to a
:class:`FitResult` for provenance, so a new regressor is "implement the
protocol, register the name" -- no pipeline changes (see
:func:`flexparameterize.regression.get_regressor`).
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from flexcore.config.schema import SurrogateSpec
from flexcore.exceptions import FlexDataError


def require_frame(frame: object, *, regressor: str, role: str) -> pd.DataFrame:
    """Return ``frame`` if it is a ``DataFrame``, as every ``fit`` requires.

    Args:
        frame: The ``X`` or ``y`` argument passed to ``fit``.
        regressor: The regressor's class name, for the error message.
        role: ``"X"`` or ``"y"``.

    Raises:
        FlexDataError: If ``frame`` is not a ``DataFrame``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise FlexDataError(
            f"{regressor}.fit takes {role} as a DataFrame, got "
            f"{type(frame).__name__}. Wrap a Series with .to_frame().",
            field=role,
        )
    return frame


def single_column(
    frame: object, *, regressor: str, role: str, what: str, hint: str = ""
) -> pd.Series:
    """Return the one column of a ``fit`` argument as a named Series.

    Args:
        frame: The ``X`` or ``y`` argument passed to ``fit``.
        regressor: The regressor's class name, for the error message.
        role: ``"X"`` or ``"y"``.
        what: What the column holds (e.g. ``"output"``), for the message.
        hint: Optional extra advice appended to the message.

    Raises:
        FlexDataError: If ``frame`` is not a ``DataFrame`` or does not hold
            exactly one column.
    """
    frame = require_frame(frame, regressor=regressor, role=role)
    if frame.shape[1] != 1:
        raise FlexDataError(
            f"{regressor} fits one {what} column; {role} has {frame.shape[1]} "
            f"({list(frame.columns)}). Select the single {what} column"
            f"{hint}.",
            field=role,
        )
    return frame.iloc[:, 0]


@dataclass(frozen=True)
class FitResult:
    """The common shape every regressor's fit reduces to, for provenance.

    Attributes:
        coefficients: Coefficient name -> value (``"intercept"`` when present).
        metrics: At minimum ``{"r2": ..., "rmse": ...}``.
        n_samples: Number of rows the fit used.
        data_window: ``(first, last)`` index value of the fitted rows.
    """

    coefficients: dict[str, float]
    metrics: dict[str, float]
    n_samples: int
    data_window: tuple


@runtime_checkable
class Regressor(Protocol):
    """Structural contract every pluggable regressor conforms to.

    Conformance is structural (``isinstance`` only checks the methods exist —
    call them to verify behavior, not just presence). ``to_surrogate_spec``
    takes ``**kwargs`` because different regressors need different unit
    metadata at that boundary; each regressor documents its own keywords.
    """

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Regressor":
        """Fit this regressor on ``X``/``y`` and return ``self``."""
        ...

    def to_fit_result(self) -> FitResult:
        """Return this fit's :class:`FitResult`, once fitted."""
        ...

    def to_surrogate_spec(self, **kwargs) -> SurrogateSpec:
        """Return this fit as a persistable ``SurrogateSpec``, once fitted."""
        ...
