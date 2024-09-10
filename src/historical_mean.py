import pandas as pd


def group_data_by_day_of_year(df: pd.DataFrame, idx) -> pd.DataFrame:
    historical_data = df.copy()
    historical_data["doy"] = historical_data.index.day_of_year
    historical_data[idx] = historical_data.index
    historical_mean = historical_data.groupby("doy").mean()
    historical_mean["doy"] = historical_mean.index
    historical_mean[idx] = historical_mean[idx].apply(
        lambda d: d.replace(hour=0, minute=0, second=0, microsecond=0, nanosecond=0)
    )
    return historical_mean.set_index("doy")
