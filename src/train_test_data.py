import pandas as pd


def training_threshold(col, date_threshold: pd.DatetimeIndex):
    return col <= date_threshold


def testing_treshold(
    col, date_lower_threshold: pd.DatetimeIndex, date_upper_threshold: pd.DatetimeIndex
):
    return ~training_threshold(col, date_lower_threshold) & (
        col <= date_upper_threshold
    )


def split_datasets(
    df: pd.DataFrame,
    date_lower_threshold: pd.DatetimeIndex,
    date_upper_threshold: pd.DatetimeIndex,
):
    return (
        df[training_threshold(df.index, date_upper_threshold)].dropna(),
        df[
            testing_treshold(df.index, date_lower_threshold, date_upper_threshold)
        ].dropna(),
    )
