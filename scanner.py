import requests
import pandas as pd
import numpy as np
import ta
import openai
from datetime import datetime, timedelta
from polygon import RESTClient
from config import POLYGON_API_KEY, OPENAI_API_KEY

# Fetch stock data from Polygon.io
def fetch_stock_data(ticker, days=100):
    """Fetches historical stock data from Polygon.io."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url).json()
    if 'results' in response:
        df = pd.DataFrame(response['results'])
        df['date'] = pd.to_datetime(df['t'])
        df.set_index('date', inplace=True)
        return df
    return None

# AI-powered news sentiment analysis (Fixed for OpenAI v1)
def analyze_sentiment(ticker):
    """Uses OpenAI GPT-4 to analyze stock news sentiment."""
    openai.api_key = OPENAI_API_KEY  # Ensure API key is set properly

    news = f"Latest earnings & headlines for {ticker}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that analyzes stock news sentiment."},
                {"role": "user", "content": f"Analyze the sentiment of this news: {news}"}
            ],
            temperature=0.5
        )

        sentiment = response["choices"][0]["message"]["content"].strip().lower()
        return "bullish" in sentiment or "positive" in sentiment  # Return True if sentiment is bullish

    except openai.error.OpenAIError as e:
        print(f"OpenAI API Error: {e}")
        return False  # Default to False if API call fails

# Detect AI-based breakout patterns
def detect_patterns(ticker):
    """Detects bullish chart patterns using AI & technical analysis."""
    df = fetch_stock_data(ticker, days=200)
    if df is None:
        return None

    df["20SMA"] = ta.trend.SMAIndicator(df["c"], window=20).sma_indicator()
    df["50SMA"] = ta.trend.SMAIndicator(df["c"], window=50).sma_indicator()

    patterns = []
    
    if df["c"].iloc[-1] > df["20SMA"].iloc[-1] and df["20SMA"].iloc[-1] > df["50SMA"].iloc[-1]:
        patterns.append("Bullish Trend Detected")

    return patterns if patterns else None

# AI-driven support & resistance
def dynamic_support_resistance(ticker):
    """Detects AI-driven support & resistance levels."""
    df = fetch_stock_data(ticker, days=100)
    if df is None:
        return None

    df['Support'] = df['c'].rolling(window=20).min()
    df['Resistance'] = df['c'].rolling(window=20).max()

    return df['c'].iloc[-1] >= (0.98 * df['Resistance'].iloc[-1])

# Confirm breakout with momentum indicators
def momentum_confirmation(ticker):
    """Checks RSI, MACD, and ATR expansion for confirmation."""
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return None

    df["RSI"] = ta.momentum.RSIIndicator(df["c"]).rsi()
    df["MACD"] = ta.trend.MACD(df["c"]).macd()
    df["ATR"] = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range()

    return df["RSI"].iloc[-1] > 60 and df["MACD"].iloc[-1] > df["MACD"].iloc[-2]

# AI-based stop-loss calculation
def stop_loss_exit(ticker):
    """Calculates AI-based stop-loss using ATR and swing lows."""
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return None

    df["ATR"] = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range()
    return df["c"].iloc[-1] - (2 * df["ATR"].iloc[-1])


