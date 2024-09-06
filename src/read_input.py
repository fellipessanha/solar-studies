import os
from typing import List
import pandas as pd
import numpy as np


def group_datasets_by_location(location: str, years: List[str]):
    filenames = []
    for year in years:
        filenames.extend(
            [
                f"{year}/{file.name}"
                for file in os.scandir(f"inmet-data/{year}")
                if location in file.name
            ]
        )

    filenames.sort()

    dfs = []
    for f in filenames:
        dfs.append(read_dataset(f))

    return pd.concat(dfs).reset_index()


def read_dataset(filename: str):
    return pd.read_csv(
        f"inmet-data/{filename}",
        delimiter=";",
        header=8,
        encoding="iso-8859-1",
    )


def format_dataset(df: pd.DataFrame):
    df = df.copy()
    cols = df.columns.copy()

    # treating hour data bc it's not super consistent
    df[cols[2]] = df[cols[2]].apply(
        lambda s: ":".join([s[:2], s[2:4]]) if isinstance(s, str) and "UTC" in s else s
    )
    # same for dates
    df[cols[1]] = df[cols[1]].apply(
        lambda s: s.replace("/", "-") if isinstance(s, str) else s
    )

    df["datetime"] = pd.to_datetime(
        df[cols[1]] + " " + df[cols[2]], format="%Y-%m-%d %H:%M"
    )

    def parseToFloat(col):
        return (
            df[col]
            .apply(
                lambda x: np.float64(x.replace(",", ".")) if (isinstance(x, str)) else x
            )
            .apply(lambda x: np.nan if x == -9999.0 else x)
        )

    # global-radiation
    df["global-radiation"] = parseToFloat(cols[7])

    # precipitation
    df["total-precipitation"] = parseToFloat(cols[3])

    df["relative-humidity"] = parseToFloat("UMIDADE RELATIVA DO AR, HORARIA (%)")

    # temperature
    df["temperature-last-hour-max"] = parseToFloat(
        "TEMPERATURA MÁXIMA NA HORA ANT. (AUT) (°C)"
    )
    df["temperature-last-hour-min"] = parseToFloat(
        "TEMPERATURA MÍNIMA NA HORA ANT. (AUT) (°C)"
    )

    # wind
    df["wind-speed"] = parseToFloat("VENTO, VELOCIDADE HORARIA (m/s)")
    df["wind-speed-max-gust"] = parseToFloat("VENTO, RAJADA MAXIMA (m/s)")

    # removing unused columns and renaming the remainder to more friendly names
    return df.set_index("datetime").drop(columns=cols[: len(cols)])
