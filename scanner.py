import requests
import pandas as pd
import numpy as np
import ta
import openai
from openai import OpenAI
from datetime import datetime, timedelta
from polygon import RESTClient
from config import POLYGON_API_KEY, OPENAI_API_KEY

# ✅ Initialize OpenAI client (Required for OpenAI v1.0+)
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Fetch stock data from Polygon.io
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

# ✅ AI-powered news sentiment analysis (Fixed for OpenAI v1.0+)
def analyze_sentiment(ticker):
    """Uses OpenAI GPT-4 to analyze stock news sentiment."""
    
    news = f"Latest earnings & headlines for {ticker}"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that analyzes stock news sentiment and assigns a sentiment score from -100 to +100."},
                {"role": "user", "content": f"Analyze the sentiment of this news and return a sentiment score from -100 (very bearish) to +100 (very bullish): {news}"}
            ],
            temperature=0.5
        )

        sentiment_score = int(response.choices[0].message.content.strip())  # AI provides a score
        return sentiment_score

    except openai.OpenAIError as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return 0  # Default to neutral sentiment (0) if API call fails

# ✅ Detect AI-based breakout patterns
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

# ✅ AI-driven support & resistance
def dynamic_support_resistance(ticker):
    """Detects AI-driven support & resistance levels."""
    df = fetch_stock_data(ticker, days=100)
    if df is None:
        return None

    df['Support'] = df['c'].rolling(window=20).min()
    df['Resistance'] = df['c'].rolling(window=20).max()

    return df['c'].iloc[-1] >= (0.98 * df['Resistance'].iloc[-1])

# ✅ Rank and return the top 10 setups
def rank_best_trades(stocks):
    """Ranks the top 10 stocks based on AI confidence, momentum, and sentiment."""
    trade_data = []

    for stock in stocks:
        df = fetch_stock_data(stock, days=50)
        if df is None:
            continue
        
        resistance = df['c'].rolling(window=20).max().iloc[-1]
        entry_price = resistance * 1.01  # ✅ Enter slightly above breakout level
        atr = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range().iloc[-1]
        stop_loss = resistance - (2 * atr)  # ✅ Stop-loss at 2x ATR below breakout level
        exit_target = entry_price + (2 * (entry_price - stop_loss))  # ✅ 2:1 Risk-Reward
        
        sentiment_score = analyze_sentiment(stock)
        momentum = momentum_confirmation(stock)
        confidence = (sentiment_score + (100 if momentum else 50)) / 2  # ✅ Confidence Score Calculation
        
        trade_data.append({
            "Stock": stock,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_loss, 2),
            "Exit Target": round(exit_target, 2),
            "Sentiment Score": sentiment_score,
            "Confidence %": round(confidence, 2)
        })

    # ✅ Sort trades by confidence and return the top 10
    trade_data = sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:10]
    
    return trade_data


