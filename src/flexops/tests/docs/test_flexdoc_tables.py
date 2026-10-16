"""Unit tests for the flexdoc Sphinx extension's table-generation functions.

Loads ``docs/_ext/flexdoc.py`` directly from disk (it lives outside the
installed package) so ``collect_unit_tables``/``collect_config_table`` are
exercised with no Sphinx application involved -- the same split the extension
itself is built around.
"""

import importlib.util
from pathlib import Path

import pytest

from flexcore.config.spec import FlowsheetSpec
from flexops.unit_models.pump import Pump

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FLEXDOC_PATH = _REPO_ROOT / "docs" / "_ext" / "flexdoc.py"

if not _FLEXDOC_PATH.exists():
    pytest.skip("docs/ is absent from this checkout", allow_module_level=True)

_spec = importlib.util.spec_from_file_location("flexdoc", _FLEXDOC_PATH)
flexdoc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flexdoc)


@pytest.mark.unit
def test_unit_tables_pump():
    """Pump's generated tables carry its registered variables, docs, and DoF."""
    tables = flexdoc.collect_unit_tables(Pump)

    variable_names = {row[0] for row in tables["variables"][1:]}
    assert "flow_vol_phase" in variable_names
    assert "power_electrical" in variable_names

    for row in tables["variables"][1:]:
        _name, _index, units, _role, description = row
        assert units
        assert description

    assert len(tables["dof"]) > 1


@pytest.mark.unit
def test_config_table_flowsheet_spec():
    """FlowsheetSpec's field rows include schema_version and its description."""
    rows = flexdoc.collect_config_table(FlowsheetSpec)

    fields = {row[0]: row for row in rows[1:]}
    assert "schema_version" in fields
    assert fields["schema_version"][3]
