"""Every released schema version keeps a fixture that must load forever."""

from pathlib import Path

import pytest

from flexcore.config.io import load_model_config, load_spec
from flexcore.config.schema import CURRENT_SCHEMA_VERSION
from flexcore.config.spec import KINDS, SCHEMA_VERSION

_FIXTURES = sorted(
    (Path(__file__).parent.parent / "fixtures" / "configs").glob("*.json")
)


@pytest.mark.unit
@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: p.stem)
def test_every_migration_fixture_loads(path):
    """Each stored old-version config loads and ends at the current schema version."""
    cfg = load_model_config(path)

    assert cfg.schema_version == CURRENT_SCHEMA_VERSION


_SPEC_FIXTURES = sorted(
    (Path(__file__).parent.parent / "fixtures" / "specs").glob("*.json")
)


@pytest.mark.unit
@pytest.mark.parametrize("path", _SPEC_FIXTURES, ids=lambda p: p.stem)
def test_every_spec_fixture_loads(path):
    """Each stored flat-spec version loads and ends at the current spec version."""
    spec = load_spec(path)

    assert spec.schema_version == SCHEMA_VERSION
    assert {el.kind for el in spec.elements} == set(KINDS)
