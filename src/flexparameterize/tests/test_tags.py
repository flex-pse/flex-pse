"""Tests for TagMap: source-column to model-alias aliasing and its report."""

import json

import pandas as pd
import pytest

from flexcore.exceptions import FlexConfigError
from flexparameterize.tags import TagMap

FLOW_ALIAS = "facility.plant.inlet_state.flow_vol_phase"
POWER_ALIAS = "facility.plant.power_electrical"


@pytest.mark.unit
def test_apply_renames_mapped_columns():
    """Mapped columns are renamed, unmapped ones kept, the input never mutated."""
    df = pd.DataFrame(
        {"FT_0231.PV": [1.0, 2.0], "MTR_KW_04": [3.0, 4.0], "AMB_TEMP": [5.0, 6.0]}
    )
    tagmap = TagMap({"FT_0231.PV": FLOW_ALIAS, "MTR_KW_04": POWER_ALIAS})

    renamed = tagmap.apply(df)

    assert list(renamed.columns) == [FLOW_ALIAS, POWER_ALIAS, "AMB_TEMP"]
    assert list(df.columns) == ["FT_0231.PV", "MTR_KW_04", "AMB_TEMP"]
    assert renamed[FLOW_ALIAS].tolist() == [1.0, 2.0]


@pytest.mark.unit
def test_report_unmapped_with_fuzzy_suggestions():
    """A near-miss column is reported unmapped with a difflib suggestion."""
    tagmap = TagMap({"FT_0231.PV": FLOW_ALIAS, "MTR_KW_04": POWER_ALIAS})
    df = pd.DataFrame({"FT_0231.PV ": [1.0], "MTR_KW_04": [2.0]})

    report = tagmap.report_unmapped(df)

    assert report.unmapped_columns == ["FT_0231.PV "]
    assert "FT_0231.PV" in report.suggestions["FT_0231.PV "]
    assert "FT_0231.PV" in str(report)


@pytest.mark.unit
def test_from_file_json_and_bad_shape(tmp_path):
    """JSON loads; a non-flat mapping, a bad suffix, and duplicate targets fail."""
    good = tmp_path / "tags.json"
    good.write_text(json.dumps({"FT_0231.PV": FLOW_ALIAS}))
    assert TagMap.from_file(good).mapping == {"FT_0231.PV": FLOW_ALIAS}

    nested = tmp_path / "nested.json"
    nested.write_text(json.dumps({"FT_0231.PV": {"alias": FLOW_ALIAS}}))
    with pytest.raises(FlexConfigError, match="FT_0231.PV"):
        TagMap.from_file(nested)

    with pytest.raises(FlexConfigError, match=".yaml"):
        TagMap.from_file(tmp_path / "tags.yaml")

    with pytest.raises(FlexConfigError, match=FLOW_ALIAS):
        TagMap({"FT_0231.PV": FLOW_ALIAS, "FT_0232.PV": FLOW_ALIAS})
