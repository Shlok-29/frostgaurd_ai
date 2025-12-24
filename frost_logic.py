def detect_frost_windows(df, threshold_temp, min_stations):
    frost_hours = []

    for hour, group in df.groupby("hour"):
        count = (group["rolling_avg"] < threshold_temp).sum()
        
        if count >= min_stations:
            frost_hours.append((hour, count))

    return frost_hours
