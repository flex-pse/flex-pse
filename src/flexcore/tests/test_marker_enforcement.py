"""Meta-tests: the root conftest's tier-marker enforcement hook works.

Uses the pytest ``pytester`` fixture to run an inner pytest session with the
real root ``conftest.py`` installed, so these tests exercise the actual
collection hook rather than a reimplementation of it.
"""

import pytest

pytest_plugins = ["pytester"]


def _install_root_conftest(
    pytester: pytest.Pytester, request: pytest.FixtureRequest
) -> None:
    conftest_source = (request.config.rootpath / "conftest.py").read_text()
    lines = [
        line for line in conftest_source.splitlines() if "pytest_plugins" not in line
    ]
    pytester.makeconftest("\n".join(lines))


@pytest.mark.unit
def test_unmarked_test_fails_collection(
    pytester: pytest.Pytester, request: pytest.FixtureRequest
):
    """A test with no tier marker fails collection with an actionable message."""
    _install_root_conftest(pytester, request)
    pytester.makepyfile("""
        def test_nothing():
            pass
        """)
    result = pytester.runpytest()
    assert "exactly one tier marker" in "\n".join(result.outlines + result.errlines)


@pytest.mark.unit
def test_double_marked_test_fails_collection(
    pytester: pytest.Pytester, request: pytest.FixtureRequest
):
    """A test with two tier markers fails collection with an actionable message."""
    _install_root_conftest(pytester, request)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.unit
        @pytest.mark.component
        def test_both():
            pass
        """)
    result = pytester.runpytest()
    assert "exactly one tier marker" in "\n".join(result.outlines + result.errlines)
