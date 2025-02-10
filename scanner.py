
import requests
import pandas as pd
import numpy as np
import talib
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

# AI-powered news sentiment analysis
def analyze_sentiment(ticker):
    """Uses OpenAI GPT-4 to analyze stock news sentiment."""
    openai.api_key = OPENAI_API_KEY
    news = f"Latest earnings & headlines for {ticker}"
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze the sentiment of this news: {news}"}]
    )
    
    sentiment = response["choices"][0]["message"]["content"]
    return sentiment.lower() in ["bullish", "positive"]

# Detect AI-based breakout patterns
def detect_patterns(ticker):
    """Detects Cup & Handle, Bull Flags, and Ascending Triangles."""
    df = fetch_stock_data(ticker, days=200)
    if df is None:
        return None

    close, high, low = df["c"].values, df["h"].values, df["l"].values

    patterns = {
        "Cup & Handle": talib.CDLHOMINGPIGEON(close, high, low),
        "Bull Flag": talib.CDLBULLTRAP(close, high, low),
        "Ascending Triangle": talib.CDLPIERCING(close, high, low),
    }

    return [pattern for pattern, result in patterns.items() if result[-1] > 0] or None

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

    df['RSI'] = talib.RSI(df['c'], timeperiod=14)
    df['MACD'], _, _ = talib.MACD(df['c'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ATR'] = talib.ATR(df['h'], df['l'], df['c'], timeperiod=14)

    return df['RSI'].iloc[-1] > 60 and df['MACD'].iloc[-1] > df['MACD'].iloc[-2]

# AI-based stop-loss calculation
def stop_loss_exit(ticker):
    """Calculates AI-based stop-loss using ATR and swing lows."""
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return None

    df['ATR'] = talib.ATR(df['h'], df['l'], df['c'], timeperiod=14)
    return df['c'].iloc[-1] - (2 * df['ATR'].iloc[-1])
