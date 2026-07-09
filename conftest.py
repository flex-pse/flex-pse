"""Repo-wide pytest configuration: tier-marker enforcement and unit-tier guards."""

import pytest

pytest_plugins = ["pytester"]

TIER_MARKERS = {"unit", "component", "integration"}


def _solver_availability():
    """Probe installed solvers once per session via the M05 registry.

    Imported inside the hook (not at module top) to keep the conftest ↔
    ``flexcore.solvers`` coupling lazy and avoid any test-time import cycle
    (M05 pitfall 5). The registry caches its own probes, so this is cheap on
    repeated calls.
    """
    from flexcore.solvers.registry import available_solvers

    return available_solvers()


def pytest_collection_modifyitems(config, items):
    """Enforce tier markers and skip tests whose required solver is absent.

    Fails collection if any test carries zero or more than one tier marker.
    For each ``needs_<name>`` marker, consults the solver registry once per
    session and attaches a ``skip`` (never a failure) when that solver is not
    installed.
    """
    errors = []
    for item in items:
        tiers = TIER_MARKERS & {m.name for m in item.iter_markers()}
        if len(tiers) != 1:
            errors.append(
                f"{item.nodeid}: needs exactly one tier marker "
                f"(unit/component/integration), got {sorted(tiers) or 'none'}"
            )
    if errors:
        raise pytest.UsageError("\n".join(errors))

    needs = {
        marker.name[len("needs_") :]
        for item in items
        for marker in item.iter_markers()
        if marker.name.startswith("needs_")
    }
    if not needs:
        return
    available = _solver_availability()
    for item in items:
        for marker in item.iter_markers():
            if marker.name.startswith("needs_"):
                name = marker.name[len("needs_") :]
                if name not in available:
                    item.add_marker(
                        pytest.mark.skip(reason=f"solver {name} not installed")
                    )


@pytest.fixture(autouse=True)
def _no_solver_in_unit_tier(request, monkeypatch):
    """Block solver invocation and network access during unit-tier tests.

    Under the ``unit`` tier, any attempt to invoke a solver — through
    ``flexcore.solvers.facade.get_solver`` or ``SolverFacade.solve`` — raises
    immediately, and outbound socket connections are blocked, so a unit test
    that secretly builds and solves a model fails fast and loudly at the
    offending call instead of silently turning the sub-second loop slow.
    ``component``/``integration`` runs are meant to solve, so the guard is off.
    """
    if "unit" not in {m.name for m in request.node.iter_markers()}:
        return

    import socket

    from flexcore.solvers import facade

    def _forbidden(*args, **kwargs):
        raise RuntimeError("solver invocation is forbidden in unit-tier tests")

    monkeypatch.setattr(facade.SolverFacade, "solve", _forbidden)
    monkeypatch.setattr(facade, "get_solver", _forbidden)

    def _no_network(*args, **kwargs):
        raise RuntimeError("network access is forbidden in unit-tier tests")

    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
