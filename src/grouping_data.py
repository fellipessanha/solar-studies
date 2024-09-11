import pandas as pd
from src.utils import DEFAULT_INDEX_COLUMN, DEFAULT_TARGET_COLUMN


def group_df_by_day(
    df: pd.DataFrame,
    idx: str = DEFAULT_INDEX_COLUMN,
    target: str = DEFAULT_TARGET_COLUMN,
):
    df[idx] = df.index

    by_day = df.groupby([df.index.year, df.index.day_of_year]).mean().set_index(idx)

    by_day = by_day.loc[by_day[target].notna()]
    by_day[idx] = by_day.index
    by_day[idx] = by_day[idx].apply(lambda d: d.replace(hour=0, minute=0, second=0))

    return by_day.set_index(idx)


def make_rolling_window(df: pd.DataFrame, window_size=28):
    return df.rolling(window=window_size).mean()
