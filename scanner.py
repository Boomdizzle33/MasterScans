import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from config import POLYGON_API_KEY
from duckduckgo_search import DDGS  
from transformers import pipeline

# ✅ Fix: Load FinBERT in Safe Mode (Prevents PyTorch Errors)
try:
    finbert = pipeline("text-classification", model="ProsusAI/finbert", device=-1)  
except Exception as e:
    print(f"⚠️ FinBERT failed to load: {e}")
    finbert = None  

# ✅ Fix: Fetch Stock Data from Polygon.io
def fetch_stock_data(ticker, days=100):
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

# ✅ Fix: Fetch News from DuckDuckGo
def fetch_duckduckgo_news(ticker):
    """Fetches top financial news headlines for a stock using DuckDuckGo."""
    search_query = f"{ticker} stock news"
    ddgs = DDGS()
    results = list(ddgs.news(search_query, max_results=5)) 
    
    if results:
        return [result['title'] + ". " + result.get('body', '') for result in results]  
    return []

# ✅ Fix: Analyze Sentiment Using FinBERT
def analyze_sentiment_finbert(ticker):
    """Fetches news and analyzes sentiment using FinBERT."""
    news_articles = fetch_duckduckgo_news(ticker)
    if not news_articles or finbert is None:
        return 0  

    total_sentiment = 0
    for article in news_articles:
        result = finbert(article[:512])[0]  
        sentiment_score = 1 if result['label'] == "positive" else -1 if result['label'] == "negative" else 0
        total_sentiment += sentiment_score

    avg_sentiment = total_sentiment / len(news_articles)
    
    return round(avg_sentiment * 100, 2)  

# ✅ Fix: Wyckoff-Style Accumulation/Distribution Zones
def accumulation_distribution_zone(ticker):
    df = fetch_stock_data(ticker, days=100)
    if df is None:
        return "Neutral"

    df["ADL"] = (2 * df["c"] - df["l"] - df["h"]) / (df["h"] - df["l"]) * df["v"]
    df["ADL"] = df["ADL"].cumsum()
    adl_trend = df["ADL"].diff().rolling(10).mean()
    
    if adl_trend.iloc[-1] > 0 and df["c"].std() < df["c"].rolling(20).std().mean() * 0.5:
        return "Accumulation"
    elif adl_trend.iloc[-1] < 0 and df["c"].std() > df["c"].rolling(20).std().mean():
        return "Distribution"
    return "Neutral"

# ✅ Fix: Rank & Return Top 10 Pre-Breakout Setups
def rank_best_trades(stocks):
    trade_data = []

    for stock in stocks:
        df = fetch_stock_data(stock, days=50)
        if df is None:
            continue
        
        resistance = df['c'].rolling(20).max().iloc[-1]
        entry_price = resistance * 0.99  
        stop_loss = entry_price - (df['c'].std() * 2)
        exit_target = entry_price + (3 * (entry_price - stop_loss))  

        sentiment_score = analyze_sentiment_finbert(stock)  
        ad_zone = accumulation_distribution_zone(stock)

        confidence = (
            (sentiment_score * 0.2) +
            (15 if ad_zone == "Accumulation" else -10 if ad_zone == "Distribution" else 0)
        )

        trade_data.append({
            "Stock": stock,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_loss, 2),
            "Exit Target": round(exit_target, 2),
            "Wyckoff Zone": ad_zone,
            "Sentiment Score": sentiment_score,
            "Confidence %": round(confidence, 2)
        })

    return sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:10]


