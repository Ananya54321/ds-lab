# import statsmodels
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

# Sample time series data
data = {"Time": range(1, 11), "Value": [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]}

df = pd.DataFrame(data)
print("Original Data:")
print(df)


# Single Exponential Smoothing
alpha = 0.5  # Smoothing parameter
model_ses = SimpleExpSmoothing(df["Value"]).fit(smoothing_level=alpha)
df["SES"] = model_ses.fittedvalues

print("\nData with Single Exponential Smoothing (SES):")
print(df)

# Double Exponential Smoothing
alpha = 0.5  # Smoothing parameter for level
beta = 0.5  # Smoothing parameter for trend
model_des = ExponentialSmoothing(df["Value"], trend="add", seasonal=None).fit(
    smoothing_level=alpha, smoothing_trend=beta
)
df["DES"] = model_des.fittedvalues

print("\nData with Double Exponential Smoothing (DES):")
print(df)

# Plot the original data, single exponential smoothing, and double exponential smoothing
plt.figure(figsize=(10, 6))
plt.plot(df["Time"], df["SES"], label="Single Exponential Smoothing (SES)", marker="o")
plt.plot(df["Time"], df["DES"], label="Double Exponential Smoothing (DES)", marker="o")

plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Single and Double Exponential Smoothing")
plt.legend()
plt.grid(True)
plt.show()
