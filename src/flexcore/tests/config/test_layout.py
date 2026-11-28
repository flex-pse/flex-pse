"""Tests for flexcore.config.layout: the config folder and its data subfolders."""

import pytest

from flexcore.config import layout
from flexcore.config.io import dump_spec, dumps_spec, load_spec
from flexcore.config.layout import ConfigLayout, register_data_category, spec_file
from flexcore.config.spec import SCHEMA_VERSION, FlowsheetSpec
from flexcore.exceptions import FlexConfigError

SPEC = {
    "schema_version": SCHEMA_VERSION,
    "name": "site",
    "elements": [
        {
            "kind": "time",
            "start_date": "2025-01-01",
            "end_date": "2025-01-01T03:00",
            "time_step": "1 hr",
        }
    ],
}


@pytest.fixture
def registry(monkeypatch):
    """Let a test register categories without leaking them into other tests."""
    monkeypatch.setattr(layout, "DATA_CATEGORIES", dict(layout.DATA_CATEGORIES))
    monkeypatch.setattr(layout, "EMIT_CATEGORIES", dict(layout.EMIT_CATEGORIES))


@pytest.mark.unit
def test_layout_paths(tmp_path):
    folder = ConfigLayout(tmp_path / "config")

    assert folder.spec_path == tmp_path / "config" / "config.json"
    assert folder.dir("price") == tmp_path / "config" / "price"
    assert folder.emit_dir("costing") == tmp_path / "config" / "price"
    assert folder.emit_dir("surrogate") == tmp_path / "config" / "sweep"


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["fix", "unfix", "dispatch"])
def test_state_kinds_stay_in_the_spec(tmp_path, kind):
    assert ConfigLayout(tmp_path).emit_dir(kind) is None


@pytest.mark.unit
def test_unknown_category_names_the_registered_ones(tmp_path):
    with pytest.raises(FlexConfigError, match="opsdata, price, sweep"):
        ConfigLayout(tmp_path).dir("weather")


@pytest.mark.unit
def test_a_module_registers_its_own_category(tmp_path, registry):
    register_data_category("dr", "Demand-response programs.", kinds=("extension",))
    register_data_category("dr", "Demand-response programs.")

    assert ConfigLayout(tmp_path).emit_dir("extension") == tmp_path / "dr"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "description", "kinds", "match"),
    [
        ("Prices", "x", (), "valid folder name"),
        ("price", "Something else.", (), "already registered"),
        ("dr", "x", ("no_such_kind",), "not in KINDS"),
        ("dr", "x", ("costing",), "already writes"),
    ],
)
def test_register_refuses(registry, name, description, kinds, match):
    with pytest.raises(FlexConfigError, match=match):
        register_data_category(name, description, kinds=kinds)


@pytest.mark.unit
def test_spec_file_for_a_folder_or_a_file(tmp_path):
    assert spec_file(tmp_path / "config") == tmp_path / "config" / "config.json"
    assert spec_file(tmp_path / "site.json") == tmp_path / "site.json"


@pytest.mark.unit
def test_dump_and_load_a_config_folder(tmp_path):
    spec = FlowsheetSpec.model_validate(SPEC)

    dump_spec(spec, tmp_path / "config")
    loaded = load_spec(tmp_path / "config")

    assert (tmp_path / "config" / "config.json").is_file()
    assert dumps_spec(loaded) == dumps_spec(spec)
    assert loaded._base_dir == tmp_path / "config"
