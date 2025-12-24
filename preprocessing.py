import pandas as pd

def compute_rolling_average(df, window):
    df = df.sort_values(["station", "hour"])
    
    df["rolling_avg"] = (
        df.groupby("station")["temperature"]
        .rolling(window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return df
