"""Tests for the tariff loaders and pandas signal helpers.

Most tests here are ``unit`` tier: the loaders and signal helpers touch only
pandas + EECO's charge-array builder, never a solver.
"""

from pathlib import Path

import pandas as pd
import pytest

from flexcore.exceptions import FlexDataError
from flexops.costing import (
    is_peak,
    load_dr_program,
    load_tariff,
    peak_windows,
    price_gradient,
    price_series,
    tariff_csv_to_dict,
)

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_TARIFF_JSON = _FIXTURES / "tariff_tou_demo.json"
_TARIFF_CSV = _FIXTURES / "tariff_tou_demo.csv"
_DR_JSON = _FIXTURES / "dr_events_demo.json"


def _july_index() -> pd.DatetimeIndex:
    """Hourly July-2025 index (744 stamps) used across the signal tests."""
    return pd.date_range("2025-07-01", periods=744, freq="h")


@pytest.mark.unit
def test_price_series_values():
    """Energy price is $0.18 on summer-weekday 16:00-20:59, $0.09 elsewhere."""
    tariff = load_tariff(_TARIFF_JSON)
    index = _july_index()
    prices = price_series(tariff, index)

    peak_mask = (index.weekday < 5) & (index.hour >= 16) & (index.hour < 21)
    assert prices[peak_mask].to_numpy() == pytest.approx(0.18)
    assert prices[~peak_mask].to_numpy() == pytest.approx(0.09)


@pytest.mark.unit
def test_is_peak_and_peak_windows():
    """`is_peak` is True exactly on the 115 July peak stamps; windows match."""
    tariff = load_tariff(_TARIFF_JSON)
    index = _july_index()

    peak = is_peak(tariff, index)
    expected = (index.weekday < 5) & (index.hour >= 16) & (index.hour < 21)
    assert list(peak.values) == list(expected)
    assert int(peak.sum()) == 115  # 23 weekdays x 5 peak hours

    windows = peak_windows(tariff, index)
    assert isinstance(windows, pd.DatetimeIndex)
    assert list(windows) == list(index[expected])


@pytest.mark.unit
def test_price_gradient():
    """Gradient is nonzero at peak transitions and zero within a flat run."""
    tariff = load_tariff(_TARIFF_JSON)
    index = _july_index()
    grad = price_gradient(tariff, index)

    # 2025-07-01 is a Tuesday: 15:00->16:00 enters peak (+0.09),
    # 20:00->21:00 leaves peak (-0.09); 02:00->03:00 is flat off-peak (0).
    assert grad.loc[pd.Timestamp("2025-07-01 16:00")] == pytest.approx(0.09)
    assert grad.loc[pd.Timestamp("2025-07-01 21:00")] == pytest.approx(-0.09)
    assert grad.loc[pd.Timestamp("2025-07-01 03:00")] == pytest.approx(0.0)


@pytest.mark.unit
def test_tz_or_load_errors():
    """A tz-aware index is rejected with a FlexDataError naming the fix."""
    tariff = load_tariff(_TARIFF_JSON)
    tz_index = pd.date_range(
        "2025-07-01", periods=24, freq="h", tz="America/Los_Angeles"
    )
    with pytest.raises(FlexDataError) as exc:
        price_series(tariff, tz_index)
    assert "tz" in str(exc.value).lower() or "timezone" in str(exc.value).lower()


@pytest.mark.unit
def test_loaders_wrap_errors(tmp_path):
    """A malformed tariff file raises FlexDataError naming the file + field."""
    bad = tmp_path / "bad_tariff.csv"
    # missing the required 'charge' column entirely
    bad.write_text("utility,type,name\nelectric,energy,peak\n")
    with pytest.raises(FlexDataError) as exc:
        load_tariff(bad)
    assert str(bad) in str(exc.value)


@pytest.mark.unit
def test_dr_load_errors(tmp_path):
    """A malformed DR file raises FlexDataError naming the file."""
    bad = tmp_path / "bad_dr.json"
    bad.write_text("{ this is not valid json ")
    with pytest.raises(FlexDataError) as exc:
        load_dr_program(bad)
    assert str(bad) in str(exc.value)


@pytest.mark.unit
def test_tariff_csv_to_dict_accepts_path_or_dataframe():
    """CSV path and pre-loaded DataFrame convert to the same dict, and that
    dict round-trips through load_tariff to the JSON tariff object."""
    from_path = tariff_csv_to_dict(_TARIFF_CSV)
    from_frame = tariff_csv_to_dict(pd.read_csv(_TARIFF_CSV))
    assert from_path == from_frame

    round_tripped = load_tariff(from_path)
    direct = load_tariff(_TARIFF_JSON)
    assert round_tripped.equals(direct)


@pytest.mark.unit
def test_tariff_csv_to_dict_write_to(tmp_path):
    """`write_to` persists a loadable JSON file and returns the same dict."""
    out = tmp_path / "tariff.json"
    with_write = tariff_csv_to_dict(_TARIFF_CSV, write_to=out)
    without_write = tariff_csv_to_dict(_TARIFF_CSV)
    assert with_write == without_write
    assert out.exists()

    # the written file loads to the same tariff object as the CSV conversion
    assert load_tariff(out).equals(load_tariff(with_write))


@pytest.mark.unit
def test_dr_container_loads_noop():
    """`load_dr_program` returns the loaded program; None-safe on None input."""
    program = load_dr_program(_DR_JSON)
    assert program is not None
    assert program["events"][0]["incentive"] == pytest.approx(0.5)
    assert load_dr_program(None) is None
