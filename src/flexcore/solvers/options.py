"""Solver-neutral options and named option settings, translated per solver.

``OPTION_NAMES`` maps each generic :class:`SolverOption` to a solver's native
spelling, and ``SETTINGS_OPTIONS`` holds each :class:`SolverSettings`'s native
options per solver. Both are plain module constants so a user can extend them
for a new solver before calling ``get_solver``.
"""

import enum
from typing import Any

from flexcore.logger import get_logger

_log = get_logger(__name__)


class SolverOption(enum.StrEnum):
    """Solver-neutral options accepted as keywords by ``SolverFacade.solve``."""

    MIP_GAP = "mip_gap"
    TIME_LIMIT = "time_limit"


class SolverSettings(enum.StrEnum):
    """Named sets of solver tuning options for a kind of model."""

    DEFAULT = "default"
    UNIT_COMMITMENT = "unit_commitment"


OPTION_NAMES: dict[str, dict[SolverOption, str]] = {
    "gurobi": {SolverOption.MIP_GAP: "MIPGap", SolverOption.TIME_LIMIT: "TimeLimit"},
    "highs": {
        SolverOption.MIP_GAP: "mip_rel_gap",
        SolverOption.TIME_LIMIT: "time_limit",
    },
    "appsi_highs": {
        SolverOption.MIP_GAP: "mip_rel_gap",
        SolverOption.TIME_LIMIT: "time_limit",
    },
    "scip": {
        SolverOption.MIP_GAP: "limits/gap",
        SolverOption.TIME_LIMIT: "limits/time",
    },
    "cbc": {SolverOption.MIP_GAP: "ratioGap", SolverOption.TIME_LIMIT: "seconds"},
    "ipopt": {SolverOption.TIME_LIMIT: "max_cpu_time"},
}

SETTINGS_OPTIONS: dict[SolverSettings, dict[str, dict[str, Any]]] = {
    SolverSettings.DEFAULT: {},
    SolverSettings.UNIT_COMMITMENT: {
        "gurobi": {"MIPFocus": 3, "Symmetry": 2, "Cuts": 2},
    },
}


def solver_options(
    name: str,
    settings: SolverSettings,
    mip_gap: float | None,
    time_limit: float | None,
    options: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the native option dict for one solver from settings and generic options.

    Args:
        name: The Pyomo solver name the options are for.
        settings: The tuning settings to start from.
        mip_gap: Relative MIP gap at which to stop, or ``None`` for the solver default.
        time_limit: Time limit in seconds, or ``None`` for the solver default.
        options: Native solver options, passed through unchanged.

    Returns:
        The combined native options. A generic option the solver has no spelling
        for is dropped with a warning.

    Raises:
        ValueError: If two sources set the same native option.
    """
    entries = [
        (key, value, "settings")
        for key, value in SETTINGS_OPTIONS[settings].get(name, {}).items()
    ]
    spellings = OPTION_NAMES.get(name, {})
    for option, value in (
        (SolverOption.MIP_GAP, mip_gap),
        (SolverOption.TIME_LIMIT, time_limit),
    ):
        if value is None:
            continue
        if option in spellings:
            entries.append((spellings[option], value, option.value))
        else:
            _log.warning("Solver %r has no %s option; ignoring it.", name, option.value)
    entries += [(key, value, "options") for key, value in (options or {}).items()]

    native: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for key, value, source in entries:
        if key in sources:
            raise ValueError(
                f"Solver option {key!r} is set by both {sources[key]} and {source}; "
                "set it in one place."
            )
        native[key] = value
        sources[key] = source
    return native
