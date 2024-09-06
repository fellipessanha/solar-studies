import pandas as pd
from src.utils import DEAFULT_INDEX_COLUMN, DEAFULT_TARGET_COLUMN


def group_df_by_day(
    df: pd.DataFrame,
    idx: str = DEAFULT_INDEX_COLUMN,
    target: str = DEAFULT_TARGET_COLUMN,
):
    df[idx] = df.index

    by_day = df.groupby([df.index.year, df.index.day_of_year]).mean().set_index(idx)

    by_day = by_day.loc[by_day[target].notna()]
    by_day[idx] = by_day.index
    by_day[idx] = by_day[idx].apply(lambda d: d.replace(hour=0, minute=0, second=0))

    return by_day.set_index(idx)
