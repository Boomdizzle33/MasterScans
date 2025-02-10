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

# ✅ Fetch Stock Data from Polygon.io
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

# ✅ Fetch News from Polygon.io
def fetch_polygon_news(ticker):
    """Fetches latest news articles for a stock from Polygon.io."""
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url).json()
    if 'results' in response:
        return response['results']  # Returns news articles
    return None

# ✅ AI-Powered Sentiment Analysis (Using GPT-4)
def analyze_news_with_ai(ticker):
    """Uses OpenAI GPT-4 to analyze news sentiment from Polygon.io."""
    
    news_articles = fetch_polygon_news(ticker)
    if not news_articles:
        return 0  # No news available
    
    # ✅ Extract only headlines and summaries
    headlines = [f"Title: {article['title']}. Summary: {article['description']}" for article in news_articles if article.get('title') and article.get('description')]

    if not headlines:
        return 0  # No valid news

    # ✅ Send news to OpenAI for sentiment analysis
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial analyst that analyzes stock news sentiment. Return a score between -100 (bearish) to +100 (bullish). Do NOT return any explanations."},
                {"role": "user", "content": f"Analyze the sentiment of these news articles:\n{headlines}"}
            ],
            temperature=0.5
        )

        sentiment_response = response.choices[0].message.content.strip()

        # ✅ Validate AI response before converting to an integer
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
    """Ranks the top 10 stocks based on AI sentiment and momentum."""
    trade_data = []

    for stock in stocks:
        df = fetch_stock_data(stock, days=50)
        if df is None:
            continue
        
        resistance = df['c'].rolling(window=20).max().iloc[-1]
        entry_price = resistance * 1.01
        atr = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range().iloc[-1]
        stop_loss = resistance - (2 * atr)
        exit_target = entry_price + (2 * (entry_price - stop_loss))
        
        sentiment_score = analyze_news_with_ai(stock)
        momentum = momentum_confirmation(stock)
        confidence = (sentiment_score + (100 if momentum else 50)) / 2
        
        trade_data.append({
            "Stock": stock,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_loss, 2),
            "Exit Target": round(exit_target, 2),
            "Sentiment Score": sentiment_score,
            "Confidence %": round(confidence, 2)
        })

    return sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:10]



