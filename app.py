import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sentiment import get_sentiment_score


st.set_page_config(page_title="Stock Forecasting AI", layout="centered")

st.title("📈 Stock Price Forecasting with LSTM + Sentiment")

stock = st.text_input("Enter Stock Symbol", "AAPL")

# ------------------------------------------------
# Load Model (Modern Keras Format)
# ------------------------------------------------
model = load_model("models/lstm_model.keras")


# ------------------------------------------------
# Download Latest Data
# ------------------------------------------------
df = yf.download(stock, period="1y")

# Fix MultiIndex if exists
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# ------------------------------------------------
# FIX CLOSE COLUMN (VERY IMPORTANT)
# ------------------------------------------------
close_series = df["Close"]

# Force 1D
if isinstance(close_series, pd.DataFrame):
    close_series = close_series.squeeze()

close_series = pd.Series(close_series.values.flatten(), index=df.index)


# ------------------------------------------------
# Add Technical Indicators
# ------------------------------------------------
df["RSI"] = ta.momentum.RSIIndicator(close=close_series).rsi()
df["EMA"] = ta.trend.EMAIndicator(close=close_series).ema_indicator()
df["MACD"] = ta.trend.MACD(close=close_series).macd()

df.dropna(inplace=True)


# ------------------------------------------------
# Scaling
# ------------------------------------------------
features = df[["Close", "RSI", "EMA", "MACD"]].values

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# ------------------------------------------------
# Prepare Input for Prediction
# ------------------------------------------------
last_60 = scaled_data[-60:]
X_input = np.expand_dims(last_60, axis=0)

prediction = model.predict(X_input)

predicted_price = scaler.inverse_transform(
    np.concatenate((prediction, np.zeros((1, 3))), axis=1)
)[0][0]


# ------------------------------------------------
# Display Results
# ------------------------------------------------
st.subheader("📊 Current Price")
st.write(float(df["Close"].iloc[-1]))

st.subheader("🔮 Predicted Next Price")
st.write(float(predicted_price))


# ------------------------------------------------
# Sentiment Analysis
# ------------------------------------------------
news = [
    f"{stock} stock shows strong growth",
    f"{stock} faces market volatility"
]

sentiment_score = get_sentiment_score(news)

st.subheader("📰 News Sentiment Score")
st.write(sentiment_score)


# ------------------------------------------------
# Chart
# ------------------------------------------------
st.line_chart(df["Close"])
