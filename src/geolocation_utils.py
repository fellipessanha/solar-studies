import numpy as np
import pandas as pd


def harversine_from_geolocation(lat_a, lon_a, lat_b, lon_b):
    R = 6371  # Radius of the Earth in kilometers

    lat_a = np.radians(lat_a)
    lon_a = np.radians(lon_a)
    lat_b = np.radians(lat_b)
    lon_b = np.radians(lon_b)

    dlat = lat_b - lat_a
    dlon = lon_b - lon_a

    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c

    return distance


def in_radius_from_station_code(df: pd.DataFrame, code: str, radius) -> pd.DataFrame:
    lat, lon = df.loc[df.code == code, ["lat", "lon"]].iloc[0]
    return harversine_from_geolocation(lat, lon, df.lat, df.lon) < radius
