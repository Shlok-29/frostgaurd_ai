import pandas as pd
import numpy as np

np.random.seed(42)

stations = ['S1', 'S2', 'S3', 'S4', 'S5']
hours = range(24)

data = []

for hour in hours:
    for station in stations:
        # Simulating night cooling
        base_temp = 8 - (hour * 0.25)
        noise = np.random.normal(0, 0.8)
        temp = base_temp + noise

        data.append([hour, station, round(temp, 2)])

df = pd.DataFrame(data, columns=["hour", "station", "temperature"])
df.to_csv("data/temperature_data.csv", index=False)
