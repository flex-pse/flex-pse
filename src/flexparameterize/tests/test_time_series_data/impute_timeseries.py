import argparse
import os
import sys
from datetime import timedelta

import pandas as pd


def parse_interval(interval_str):
    interval_map = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "1d": timedelta(days=1),
    }
    if interval_str not in interval_map:
        supported = list(interval_map.keys())
        raise ValueError(
            f"Unsupported interval: {interval_str}. Supported: {supported}"
        )
    return interval_map[interval_str]


def impute_timeseries(input_path, interval_str, start_date_str=None):
    df = pd.read_csv(input_path)

    time_col = "timestamp"
    if time_col not in df.columns:
        print(f"Error: Expected time column '{time_col}' not found in CSV.")
        sys.exit(1)

    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).reset_index(drop=True)

    if start_date_str:
        start_dt = pd.to_datetime(start_date_str)
        start_dt = (
            start_dt.tz_localize("UTC")
            if start_dt.tzinfo is None
            else start_dt.tz_convert("UTC")
        )
        time_offset = start_dt - df[time_col].iloc[0]
        df[time_col] = df[time_col] + time_offset
    else:
        start_dt = df[time_col].iloc[0]

    end_dt = df[time_col].iloc[-1]
    interval = parse_interval(interval_str)

    new_times = pd.date_range(start=start_dt, end=end_dt, freq=interval, tz="UTC")

    value_cols = [c for c in df.columns if c != time_col]

    imputed = pd.DataFrame({time_col: new_times})
    imputed = imputed.set_index(time_col)
    df_indexed = df.set_index(time_col)

    imputed[value_cols] = (
        df_indexed[value_cols].reindex(imputed.index).interpolate(method="index")
    )

    imputed = imputed.reset_index()
    imputed[time_col] = imputed[time_col].dt.strftime("%Y-%m-%d %H:%M:%S")

    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    output_name = f"imputed_{basename}"
    output_path = os.path.join(dirname, output_name)

    imputed.to_csv(output_path, index=False)
    print(f"Saved imputed time series to: {output_path}")
    print(
        "Original rows:"
        f" {len(df)}, Imputed rows: {len(imputed)}, Interval: {interval_str}"
    )
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time series imputation script")
    parser.add_argument("input_file", help="Input CSV file path")
    parser.add_argument("interval", help="Time interval (e.g., 1m, 15m, 1h)")
    parser.add_argument(
        "--start-date",
        help="Start date in YYYY-MM-DD HH:MM:SS format (optional)",
        default=None,
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
        sys.exit(1)

    impute_timeseries(args.input_file, args.interval, args.start_date)
