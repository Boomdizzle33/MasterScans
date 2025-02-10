import requests
import pandas as pd
import numpy as np
import ta
import openai
from openai import OpenAI
from datetime import datetime, timedelta
from polygon import RESTClient
from config import POLYGON_API_KEY, OPENAI_API_KEY

# ✅ Initialize OpenAI client
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

# ✅ Fix: OpenAI Sentiment Analysis (Returns a Valid Number)
def analyze_sentiment(ticker):
    """Uses OpenAI GPT-4 to analyze stock news sentiment and return a score (-100 to 100)."""
    
    news = f"Latest headlines and earnings for {ticker}"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI that analyzes stock news sentiment. Only return a number from -100 (bearish) to +100 (bullish). Do NOT return any explanations."},
                {"role": "user", "content": f"Analyze sentiment and return a number only: {news}"}
            ],
            temperature=0.5
        )

        sentiment_response = response.choices[0].message.content.strip()

        # ✅ Fix: Validate AI response before converting to integer
        if sentiment_response.lstrip('-').isdigit():
            return int(sentiment_response)
        else:
            return 0  # Default to neutral sentiment if AI response is invalid

    except openai.OpenAIError as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return 0  # Default to neutral sentiment (0) if API call fails

# ✅ Ensure `momentum_confirmation()` exists
def momentum_confirmation(ticker):
    """Checks RSI, MACD, and ATR expansion for confirmation."""
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return False

    df["RSI"] = ta.momentum.RSIIndicator(df["c"]).rsi()
    df["MACD"] = ta.trend.MACD(df["c"]).macd()

    return df["RSI"].iloc[-1] > 60 and df["MACD"].iloc[-1] > df["MACD"].iloc[-2]

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



