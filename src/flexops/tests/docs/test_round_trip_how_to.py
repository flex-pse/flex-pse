"""Run every Python snippet of docs/how_to/round_trip_a_model.md, in order.

A snippet whose first line is ``# file: <name>.py`` is written to that file
(so the snippets after it can import it) instead of being run.
"""

import re
from pathlib import Path

import pytest

HOW_TO = (
    Path(__file__).resolve().parents[4] / "docs" / "how_to" / "round_trip_a_model.md"
)

if not HOW_TO.parent.exists():
    pytest.skip("docs/ is absent from this checkout", allow_module_level=True)


@pytest.mark.unit
def test_round_trip_how_to_snippets_run(tmp_path, monkeypatch):
    snippets = re.findall(r"```python\n(.*?)```", HOW_TO.read_text(), re.DOTALL)
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    namespace = {"__name__": "how_to"}

    for snippet in snippets:
        file_name = re.match(r"# file: (\S+)\n", snippet)
        if file_name:
            (tmp_path / file_name.group(1)).write_text(snippet)
        else:
            exec(compile(snippet, str(HOW_TO), "exec"), namespace)

    assert len(snippets) >= 3
