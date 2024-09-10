import pandas as pd

def make_rolling_window(df: pd.DataFrame, window_size=28):
    return df.rolling(window=window_size).mean()
