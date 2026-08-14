"""Tests for check_sufficiency: per-IO-pair data coverage of a built model."""

import pytest

from flexops.core.registration import iter_io_registry
from flexparameterize.tags import model_alias
from flexparameterize.tests.helpers import build_plant, evaluate_data
from flexparameterize.validate import check_sufficiency


@pytest.mark.unit
def test_sufficient_happy_path():
    """Full data is sufficient, whether walked from a model or an IORegistry."""
    m, unit = build_plant()
    data = evaluate_data(unit)

    assert check_sufficiency(m, data, m.time_block).sufficient is True

    _, registry = next(iter_io_registry(m))
    assert check_sufficiency(registry, data, m.time_block).sufficient is True


@pytest.mark.unit
def test_missing_column():
    """An absent IO column makes the report insufficient and names the column."""
    m, unit = build_plant()
    power = model_alias(unit.power_electrical)
    data = evaluate_data(unit).drop(columns=[power])

    report = check_sufficiency(m, data, m.time_block)

    assert report.sufficient is False
    assert power in str(report)


@pytest.mark.unit
def test_all_nan_column():
    """A present but all-NaN column is insufficient with a zero non-null count."""
    m, unit = build_plant()
    power = model_alias(unit.power_electrical)
    data = evaluate_data(unit)
    data[power] = float("nan")

    report = check_sufficiency(m, data, m.time_block)

    assert report.sufficient is False
    counts = {
        column: count
        for pair in report.pairs
        for column, count in pair.non_null_counts.items()
    }
    assert counts[power] == 0


@pytest.mark.unit
def test_misaligned_index():
    """A plain integer index is not a DatetimeIndex, so the data is insufficient."""
    m, unit = build_plant()
    data = evaluate_data(unit).reset_index(drop=True)

    report = check_sufficiency(m, data, m.time_block)

    assert report.sufficient is False
    assert "DatetimeIndex" in str(report)


@pytest.mark.unit
def test_extra_columns_ignored():
    """Columns no registered variable claims do not affect sufficiency."""
    m, unit = build_plant()
    data = evaluate_data(unit)
    data["AMB_TEMP"] = 20.0

    assert check_sufficiency(m, data, m.time_block).sufficient is True


@pytest.mark.unit
def test_multiple_io_pairs_per_unit():
    """The unit's two IO pairs are reported separately: one satisfied, one not."""
    m, unit = build_plant()
    flow_in = model_alias(unit.inlet_state.flow_vol_phase)
    flow_out = model_alias(unit.outlet_state.flow_vol_phase)
    power = model_alias(unit.power_electrical)
    data = evaluate_data(unit).drop(columns=[flow_out])

    report = check_sufficiency(m, data, m.time_block)

    statuses = {pair.variables: pair.sufficient for pair in report.pairs}
    assert statuses == {(flow_in, flow_out): False, (flow_in, power): True}
    assert report.sufficient is False
