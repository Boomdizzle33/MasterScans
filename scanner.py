import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from config import POLYGON_API_KEY
from duckduckgo_search import DDGS  
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  

# ✅ Initialize VADER Sentiment Analyzer (No PyTorch Needed)
analyzer = SentimentIntensityAnalyzer()

# ✅ Fetch Stock Data from Polygon.io
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

# ✅ Fetch News from DuckDuckGo
def fetch_duckduckgo_news(ticker):
    search_query = f"{ticker} stock news"
    ddgs = DDGS()
    results = list(ddgs.news(search_query, max_results=5)) 
    
    if results:
        return [result['title'] + ". " + result.get('body', '') for result in results]  
    return []

# ✅ Sentiment Score (ONLY News-Based, No Technicals)
def analyze_sentiment_vader(ticker):
    news_articles = fetch_duckduckgo_news(ticker)
    if not news_articles:
        return 0  

    total_sentiment = 0
    for article in news_articles:
        score = analyzer.polarity_scores(article)['compound']  
        total_sentiment += score

    avg_sentiment = (total_sentiment / len(news_articles)) * 100  
    return round(avg_sentiment, 2)  

# ✅ Wyckoff Accumulation/Distribution Zones
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

# ✅ Relative Volume (RVOL) for Breakout Strength
def relative_volume(ticker):
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return 1  

    avg_volume = df["v"].rolling(20).mean().iloc[-1]  
    rvol = df["v"].iloc[-1] / avg_volume  
    
    return round(rvol, 2)  

# ✅ Technical Indicators (RSI, SMA, MACD)
def technical_confirmation(ticker):
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return 0  

    df["RSI"] = ta.momentum.RSIIndicator(df["c"]).rsi()
    df["50_SMA"] = df["c"].rolling(50).mean()
    df["200_SMA"] = df["c"].rolling(200).mean()
    df["MACD"] = ta.trend.MACD(df["c"]).macd()

    return (
        (10 if df["RSI"].iloc[-1] > 50 else -10) +
        (10 if df["c"].iloc[-1] > df["50_SMA"].iloc[-1] > df["200_SMA"].iloc[-1] else -10) +
        (10 if df["MACD"].iloc[-1] > 0 else -10)
    )

# ✅ Rank & Return Top 20 Pre-Breakout Setups
def rank_best_trades(stocks, top_n, progress_bar, status_text):
    """Ranks and returns the top N pre-breakout setups."""
    trade_data = []

    for i, stock in enumerate(stocks):
        if progress_bar and status_text:
            progress_bar.progress((i + 1) / len(stocks))
            status_text.text(f"🔍 Scanning {stock} ({i+1}/{len(stocks)})...")

        df = fetch_stock_data(stock, days=50)
        if df is None:
            continue

        confidence = (analyze_sentiment_vader(stock) * 0.2) + (relative_volume(stock) * 10)
        trade_data.append({"Stock": stock, "Confidence %": round(confidence, 2)})

    return sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:top_n]



