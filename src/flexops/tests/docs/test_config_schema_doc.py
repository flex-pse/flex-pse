"""The kinds table in docs/explanation/config_schema.md matches KINDS."""

import re
from pathlib import Path

import pytest

from flexcore.config.spec import KINDS

DOC = Path(__file__).resolve().parents[4] / "docs" / "explanation" / "config_schema.md"

if not DOC.parent.exists():
    pytest.skip("docs/ is absent from this checkout", allow_module_level=True)


@pytest.mark.unit
def test_kinds_table_matches_the_registry():
    rows = re.findall(r"^\| `(\w+)` \| (\w+) \| (yes|no) \|$", DOC.read_text(), re.M)

    table = {kind: (stage, mutable == "yes") for kind, stage, mutable in rows}

    assert table == KINDS
