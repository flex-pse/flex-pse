import os

import pandas as pd
import pytest

from flexparameterize.tests.test_time_series_data.impute_timeseries import (
    impute_timeseries,
)


@pytest.fixture
def sample_csv(tmp_path):
    data = {
        "timestamp": [
            "2025-01-01 00:00:00+00:00",
            "2025-01-01 01:00:00+00:00",
            "2025-01-01 02:00:00+00:00",
            "2025-01-01 03:00:00+00:00",
        ],
        "value": [0.0, 4.0, 2.0, 6.0],
        "flag": [1, 2, 3, 4],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.unit
def test_impute_creates_output_file(sample_csv, tmp_path):
    out = impute_timeseries(str(sample_csv), "30m")
    assert os.path.exists(out)
    assert out.endswith("imputed_sample.csv")


@pytest.mark.unit
def test_impute_row_count(sample_csv):
    out = impute_timeseries(str(sample_csv), "15m")
    df = pd.read_csv(out)
    assert len(df) == 13


@pytest.mark.unit
def test_impute_timestamp_format(sample_csv):
    out = impute_timeseries(str(sample_csv), "15m")
    df = pd.read_csv(out)
    assert df["timestamp"].iloc[0] == "2025-01-01 00:00:00"
    assert df["timestamp"].iloc[-1] == "2025-01-01 03:00:00"


@pytest.mark.unit
def test_impute_interpolation(sample_csv):
    out = impute_timeseries(str(sample_csv), "15m")
    df = pd.read_csv(out)
    mid = df[df["timestamp"] == "2025-01-01 00:30:00"].iloc[0]
    assert mid["value"] == pytest.approx(2.0)


@pytest.mark.unit
def test_impute_start_date_shift(sample_csv, tmp_path):
    out = impute_timeseries(str(sample_csv), "1h", start_date_str="2025-02-01 00:00:00")
    df = pd.read_csv(out)
    assert df["timestamp"].iloc[0] == "2025-02-01 00:00:00"
    assert len(df) == 4
