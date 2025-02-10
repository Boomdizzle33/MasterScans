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
    
    headlines = [f"Title: {article['title']}. Summary: {article['description']}" for article in news_articles if article.get('title') and article.get('description')]

    if not headlines:
        return 0  # No valid news

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

        if sentiment_response.lstrip('-').isdigit():
            return int(sentiment_response)
        else:
            return 0  

    except openai.OpenAIError as e:
        print(f"⚠️ OpenAI API Error: {e}")
        return 0  

# ✅ Detect Volume Contraction Before Breakout
def volume_contraction(ticker):
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return False  

    df["Volume_Slope"] = df["v"].rolling(5).mean().diff()
    price_consolidation = df["c"].std() < df["c"].rolling(20).std().mean() * 0.5  
    
    return (df["Volume_Slope"].iloc[-1] < 0) and price_consolidation

# ✅ Dynamic Stop-Loss Based on Volatility
def dynamic_stop_loss(ticker):
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return None

    atr = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range().iloc[-1]
    return df["c"].iloc[-1] - (1.5 * atr) if volume_contraction(ticker) else df["c"].iloc[-1] - (2.5 * atr)

# ✅ Rank & Return Top 10 Pre-Breakout Setups
def rank_best_trades(stocks):
    trade_data = []

    for stock in stocks:
        df = fetch_stock_data(stock, days=50)
        if df is None:
            continue
        
        resistance = df['c'].rolling(20).max().iloc[-1]
        entry_price = resistance * 0.99  
        stop_loss = dynamic_stop_loss(stock)
        exit_target = entry_price + (3 * (entry_price - stop_loss))  

        sentiment_score = analyze_news_with_ai(stock)
        volume_contraction_flag = volume_contraction(stock)

        confidence = (sentiment_score * 0.3) + (50 if volume_contraction_flag else 30) + 20  

        trade_data.append({
            "Stock": stock,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_loss, 2),
            "Exit Target": round(exit_target, 2),
            "Sentiment Score": sentiment_score,
            "Confidence %": round(confidence, 2)
        })

    return sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:10]




