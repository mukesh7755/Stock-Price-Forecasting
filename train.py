import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ------------------------------------------------
# Load Stock Data
# ------------------------------------------------
stock = "AAPL"
df = yf.download(stock, start="2016-01-01", end="2024-01-01")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()


# ------------------------------------------------
# Ensure Close is 1D
# ------------------------------------------------
close_series = df["Close"]
if isinstance(close_series, pd.DataFrame):
    close_series = close_series.squeeze()


# ------------------------------------------------
# Technical Indicators
# ------------------------------------------------
df["RSI"] = ta.momentum.RSIIndicator(close=close_series).rsi()
df["EMA"] = ta.trend.EMAIndicator(close=close_series).ema_indicator()
df["MACD"] = ta.trend.MACD(close=close_series).macd()

df.dropna(inplace=True)


# ------------------------------------------------
# Feature Scaling
# ------------------------------------------------
features = df[["Close", "RSI", "EMA", "MACD"]].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)


# ------------------------------------------------
# Create Sequences
# ------------------------------------------------
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step][0])
    return np.array(X), np.array(y)


X, y = create_dataset(scaled_data)


# ------------------------------------------------
# Train-Test Split
# ------------------------------------------------
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


# ------------------------------------------------
# Build LSTM Model
# ------------------------------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(60, 4)),
    Dropout(0.2),
    LSTM(64),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

print("Training started...")
model.fit(X_train, y_train, epochs=10, batch_size=32)
print("Training completed")


# ------------------------------------------------
# Save Model (MODERN FORMAT)
# ------------------------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/lstm_model.keras")
print("Model saved as lstm_model.keras")


# ------------------------------------------------
# Predictions
# ------------------------------------------------
y_pred = model.predict(X_test)

dummy_test = np.zeros((len(y_test), 3))
dummy_pred = np.zeros((len(y_pred), 3))

y_test_actual = scaler.inverse_transform(
    np.concatenate((y_test.reshape(-1, 1), dummy_test), axis=1)
)[:, 0]

y_pred_actual = scaler.inverse_transform(
    np.concatenate((y_pred, dummy_pred), axis=1)
)[:, 0]


# ------------------------------------------------
# Evaluation
# ------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\nModel Performance:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R2 Score:", r2)


# ------------------------------------------------
# Save Plot
# ------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual")
plt.plot(y_pred_actual, label="Predicted")
plt.title("Stock Price Prediction (LSTM)")
plt.legend()
plt.savefig("prediction_plot.png")
print("Plot saved as prediction_plot.png")
