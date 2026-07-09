"""Unit tests for :mod:`flexcore.solvers.registry`."""

import pytest

from flexcore.solvers import ProblemClass, available_solvers, registry


@pytest.mark.unit
def test_scip_registered_for_minlp():
    """SCIP is the registered open-source MINLP solver."""
    assert ProblemClass.MINLP in registry.CAPABILITIES["scip"]


@pytest.mark.unit
def test_available_solvers_keys_subset():
    registry._reset_availability_cache()
    result = available_solvers()
    assert isinstance(result, dict)
    assert set(result) <= set(registry.CAPABILITIES)


@pytest.mark.unit
def test_probing_is_cached(monkeypatch):
    """Each solver is probed at most once across repeated calls."""
    calls: list[str] = []

    def fake_probe(name: str) -> bool:
        calls.append(name)
        return True

    monkeypatch.setattr(registry, "_probe", fake_probe)
    registry._reset_availability_cache()
    try:
        registry.available_solvers()
        registry.available_solvers()

        assert set(calls) == set(registry.CAPABILITIES)
        # Second call is served from the cache: no solver is probed twice.
        assert len(calls) == len(registry.CAPABILITIES)
    finally:
        # Drop the fake-probe results so real availability is re-probed by
        # later tests (the cache is module-level and would otherwise leak).
        registry._reset_availability_cache()
