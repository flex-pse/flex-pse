"""Project-wide exception hierarchy for flex-pse.

Every error raised by flex-pse code subclasses :class:`FlexError`. Callers can
catch :class:`FlexError` to handle any flex-pse-specific failure, or catch a
concrete subclass to handle one category. Raised messages must state both what
was wrong and what the user should do about it (see ``plan/00_conventions.md``
§3).

There is no compatibility/isolation layer here (``plan/01_architecture.md``
§2.1): ``idaes.*`` and ``pyomo.*`` are imported
directly at point of use throughout the codebase, and their own exceptions are
allowed to propagate unwrapped.
"""


class FlexError(Exception):
    """Base class for all flex-pse errors.

    Catch this to handle any flex-pse-specific failure regardless of category.
    Prefer raising one of the concrete subclasses below over this base class
    directly, so callers can distinguish failure categories.
    """


class FlexConfigError(FlexError):
    """Raised when persisted or runtime configuration is invalid.

    Raise this for malformed or missing configuration: a pydantic config file
    that fails validation, a Pyomo ``ConfigDict`` entry given an invalid value,
    or a config referencing a field/unit/tag that does not exist. The message
    must name the exact field that was wrong and what value would fix it.

    Attributes:
        field: Dotted name of the config field, unit, or tag that was invalid,
            if known.
        value: The invalid value that was supplied, if known.
    """

    def __init__(
        self, message: str, *, field: str | None = None, value: object = None
    ) -> None:
        super().__init__(message)
        self.field = field
        self.value = value


class FlexSolverError(FlexError):
    """Raised when a solver cannot be found, is unsupported, or fails to run.

    Raise this when the solver facade cannot satisfy a request: the named
    solver is not installed, no available solver supports the model's
    classification (e.g. NLP vs. MILP), or the solver process itself errors
    out before returning a status. The message must tell the user which
    solver was requested and how to install or substitute one.

    Attributes:
        solver: Name of the solver that was requested or attempted, if known.
        problem_class: Name of the problem classification (e.g. ``"MINLP"``)
            the solver needed to support, if known.
    """

    def __init__(
        self,
        message: str,
        *,
        solver: str | None = None,
        problem_class: str | None = None,
    ) -> None:
        super().__init__(message)
        self.solver = solver
        self.problem_class = problem_class


class FlexDataError(FlexError):
    """Raised when input plant/historian data is missing or insufficient.

    Raise this for problems with data supplied to FlexParameterize or FlexOps:
    missing historian tags, insufficient data to regress a parameter, or data
    that fails a sufficiency/validity check. The message must name the missing
    or invalid data and what additional data or action would resolve it.

    Attributes:
        field: Name of the missing or invalid historian tag or field, if known.
    """

    def __init__(self, message: str, *, field: str | None = None) -> None:
        super().__init__(message)
        self.field = field
