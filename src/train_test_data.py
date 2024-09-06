import pandas as pd


def training_threshold(col, date_threshold, mask):
    return col <= pd.to_datetime(date_threshold, format=mask)


def testing_treshold(col, date_lower_threshold, date_upper_threshold, mask):
    return ~training_threshold(col, date_lower_threshold, mask) & (
        col <= pd.to_datetime(date_upper_threshold, format=mask)
    )


def split_datasets(df, date_lower_threshold, date_upper_threshold, mask):
    return (
        df[training_threshold(df.index, date_upper_threshold, mask)].dropna(),
        df[
            testing_treshold(df.index, date_lower_threshold, date_upper_threshold, mask)
        ].dropna(),
    )
