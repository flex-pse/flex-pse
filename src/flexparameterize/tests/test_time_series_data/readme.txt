This folder contains data from public spaces for testing surrogate modeling methods in flex-pse. 

The data files come from the following sources:
biogas_generation.csv
    Adopted from https://github.com/MuneebX00/Biogas-Time-Series-ML-Engine
    This extracted time amount of feed, and biogas generated volume from biogas_hourly.csv retrieved on 9/3/2026
imputed_bio_gas_generation.csv
    This is generated using impute_timeseries.py from biogas_generation.csv to match typical time series use in flex-pse