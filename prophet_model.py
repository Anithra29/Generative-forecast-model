import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import random

# LOAD CLEANED DATA

data_path = "data/prophet_ready.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Please run eda.py first.")

df = pd.read_csv(data_path, parse_dates=['ds'])

# Split into train/test
N = 30
train = df[:-N]
test = df[-N:]

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")

# TRAINING PROPHET MODEL
m = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)
m.fit(train)

future = m.make_future_dataframe(periods=N, freq='D')
forecast = m.predict(future)


# EVALUATION

y_true = test['y'].values
y_pred = forecast.iloc[-N:]['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")


# PLOTS

m.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

m.plot_components(forecast)
plt.show()

# GENERATIVE HUMAN-LIKE EXPLANATION ENGINE
# Ensure required columns

available_cols = forecast.columns.tolist()
components = ['ds', 'yhat', 'trend', 'weekly', 'yearly']
if 'holidays' in available_cols:
    components.append('holidays')

components_df = forecast[components].tail(N)
explain_df = test.merge(components_df, on='ds', how='left')

def generate_human_explanation(row):
    reasons = []

    # Compare forecast to trend to gauge direction
    if row['yhat'] > row['trend'] * 1.05:
        direction = "increase"
    elif row['yhat'] < row['trend'] * 0.95:
        direction = "decrease"
    else:
        direction = "stable"

    # Trend-based reasoning
    if direction == "increase":
        reasons.append(random.choice([
            "Sales are expected to rise due to a continuing upward trend.",
            "A positive growth pattern is driving sales higher.",
            "Momentum from previous days is pushing demand upward."
        ]))
    elif direction == "decrease":
        reasons.append(random.choice([
            "Sales are projected to dip slightly following a downward trend.",
            "A slowdown in recent demand is contributing to lower sales.",
            "Decline is expected as the trend indicates reduced activity."
        ]))
    else:
        reasons.append(random.choice([
            "Sales are expected to remain steady, following a stable trend.",
            "No major change is anticipated; the overall trend looks consistent."
        ]))

    # Weekly effect
    if 'weekly' in row and abs(row['weekly']) > 0.05 * abs(row['yhat']):
        if row['weekly'] > 0:
            reasons.append(random.choice([
                "Strong weekend or midweek demand is contributing to higher sales.",
                "Weekly patterns, especially near weekends, are boosting performance."
            ]))
        else:
            reasons.append(random.choice([
                "Weekday slowdown is likely causing lower sales volume.",
                "Midweek dip in activity is slightly pulling sales down."
            ]))

    # Yearly seasonality
    if 'yearly' in row and abs(row['yearly']) > 0.05 * abs(row['yhat']):
        if row['yearly'] > 0:
            reasons.append("Seasonal factors for this time of year are favoring higher sales.")
        else:
            reasons.append("This time of year typically sees a dip in customer activity.")

    # Holiday effects
    if 'holidays' in row and not pd.isna(row['holidays']) and row['holidays'] != 0:
        if row['holidays'] > 0:
            reasons.append("Holiday promotions and celebrations are boosting demand.")
        else:
            reasons.append("Post-holiday fatigue is causing a temporary slowdown.")

    # Compose a natural sentence
    summary = " ".join(reasons)
    return summary.strip() if summary else "Sales appear stable with minor seasonal variation."

explain_df['explanation'] = explain_df.apply(generate_human_explanation, axis=1)


# OUTPUT RESULTS

print("\nForecast Explanations:\n")
for i, row in explain_df.tail(10).iterrows():
    print(f"Date: {row['ds'].date()} | Forecast: {row['yhat']:.2f}")
    print(f"Explanation: {row['explanation']}\n")

output_path = "data/forecast_explanations.csv"
explain_df.to_csv(output_path, index=False)
print(f"\n Explanations saved to: {output_path}")
