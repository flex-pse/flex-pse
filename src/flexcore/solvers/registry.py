"""Solver capability matrix and cached availability probing.

``CAPABILITIES`` maps a solver name to the set of :class:`ProblemClass` values
it can handle. :func:`available_solvers` returns the subset whose solver is
actually installed, probing Pyomo once per solver and caching the result (a
probe can take ~1 s). This module is the single source of truth for the pytest
``needs_*`` availability markers (see the root ``conftest.py``).

``CAPABILITIES`` is a plain module constant on purpose: a user can register a
new solver entry (e.g. a MINLP-capable solver) before calling ``get_solver``.
"""

import contextlib
import functools
import os

import pyomo.environ as pyo

from flexcore.solvers.classify import ProblemClass

CAPABILITIES: dict[str, set[ProblemClass]] = {
    "highs": {ProblemClass.LP, ProblemClass.MILP},
    "cbc": {ProblemClass.LP, ProblemClass.MILP},
    "ipopt": {ProblemClass.NLP, ProblemClass.QP},
    "scip": {ProblemClass.MILP, ProblemClass.MINLP},
    "gurobi": {ProblemClass.LP, ProblemClass.QP, ProblemClass.MILP},
}


@contextlib.contextmanager
def _clean_idaes_from_libpath():
    """Strip idaes's bundled-binary dir from the dynamic-loader search path.
    ``import idaes`` prepends ``~/.idaes/bin`` (which ships idaes's own
    ``libipopt``) to ``DYLD_LIBRARY_PATH``/``LD_LIBRARY_PATH``. A separately
    installed SCIP that links a *different* ``libipopt`` then loads idaes's copy
    and crashes on a missing symbol. Removing idaes entries for the duration of
    a SCIP subprocess makes SCIP load its own libraries; the idaes IPOPT binary
    is unaffected because it runs at other times with the path intact. The
    ``os.environ`` edit is restored on exit (not thread-safe, but solver probes
    and solves run on the main thread).
    """
    saved: dict[str, str] = {}
    for var in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        current = os.environ.get(var)
        if current is None:
            continue
        cleaned = ":".join(
            part for part in current.split(":") if part and "idaes" not in part.lower()
        )
        if cleaned != current:
            saved[var] = current
            os.environ[var] = cleaned
    try:
        yield
    finally:
        for var, value in saved.items():
            os.environ[var] = value


def _solver_runtime_env(name: str):
    """Context manager preparing the process environment for a solver run.

    Only SCIP needs special handling (see :func:`_clean_idaes_from_libpath`);
    every other solver runs under the unmodified environment.

    Args:
        name: The Pyomo solver name about to be invoked.

    Returns:
        A context manager active for the duration of the solver call.
    """
    if name == "scip":
        return _clean_idaes_from_libpath()
    return contextlib.nullcontext()


@contextlib.contextmanager
def _stdin_from_devnull():
    """Temporarily point the OS-level stdin (fd 0) at ``/dev/null``.

    Some solver version checks shell out to the solver binary (e.g.
    ``scip --version``); binaries that inherit an open interactive stdin can
    drop into a shell and hang until the probe times out. Feeding the probe a
    closed stdin makes such a binary see EOF and exit immediately, so
    availability detection is reliable in an interactive terminal, not only in
    CI where stdin is already closed.
    """
    with open(os.devnull) as devnull:
        saved_fd = os.dup(0)
        try:
            os.dup2(devnull.fileno(), 0)
            yield
        finally:
            os.dup2(saved_fd, 0)
            os.close(saved_fd)


def _probe(name: str) -> bool:
    """Report whether a solver is installed and usable.

    Wraps ``SolverFactory(name).available(...)`` in ``try/except`` because some
    Pyomo solver plugins raise (e.g. ``ApplicationError``) instead of returning
    ``False``; any exception is treated as "not available". The probe runs with
    stdin redirected to ``/dev/null`` (see :func:`_stdin_from_devnull`).

    Args:
        name: The Pyomo solver name to probe.

    Returns:
        ``True`` if the solver is available, ``False`` otherwise.
    """
    try:
        with _stdin_from_devnull(), _solver_runtime_env(name):
            return bool(pyo.SolverFactory(name).available(exception_flag=False))
    except Exception:
        return False


@functools.cache
def _cached_probe(name: str) -> bool:
    """Cache :func:`_probe` per solver name for the session."""
    return _probe(name)


def available_solvers() -> dict[str, set[ProblemClass]]:
    """Return the capability matrix restricted to installed solvers.

    Returns:
        A dict mapping each installed solver name to its capability set — the
        subset of ``CAPABILITIES`` whose solver probes as available.
    """
    return {name: caps for name, caps in CAPABILITIES.items() if _cached_probe(name)}


def _reset_availability_cache() -> None:
    """Clear the cached availability probes (for tests)."""
    _cached_probe.cache_clear()
