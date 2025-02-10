import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from config import POLYGON_API_KEY
from duckduckgo_search import DDGS  
from transformers import pipeline

# ✅ Load FinBERT in Safe Mode (Prevents PyTorch Errors)
try:
    finbert = pipeline("text-classification", model="ProsusAI/finbert", device=-1)  
except Exception as e:
    print(f"⚠️ FinBERT failed to load: {e}")
    finbert = None  

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

# ✅ Wyckoff-Style Accumulation/Distribution Zones
def accumulation_distribution_zone(ticker):
    df = fetch_stock_data(ticker, days=100)
    if df is None:
        return "Neutral"

    # ✅ Calculate A/D Line (Institutional Buying/Selling)
    df["ADL"] = (2 * df["c"] - df["l"] - df["h"]) / (df["h"] - df["l"]) * df["v"]
    df["ADL"] = df["ADL"].cumsum()

    # ✅ Identify Accumulation/Distribution
    adl_trend = df["ADL"].diff().rolling(10).mean()
    
    if adl_trend.iloc[-1] > 0 and df["c"].std() < df["c"].rolling(20).std().mean() * 0.5:
        return "Accumulation"  # ✅ Smart money buying
    elif adl_trend.iloc[-1] < 0 and df["c"].std() > df["c"].rolling(20).std().mean():
        return "Distribution"  # ❌ Smart money selling
    return "Neutral"

# ✅ Relative Volume (RVOL) for Breakout Strength
def relative_volume(ticker):
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return 1  

    avg_volume = df["v"].rolling(20).mean().iloc[-1]  
    rvol = df["v"].iloc[-1] / avg_volume  
    
    return round(rvol, 2)  

# ✅ Sentiment Analysis Using FinBERT
def analyze_sentiment_finbert(news_articles):
    if not news_articles or finbert is None:
        return 0  

    total_sentiment = 0
    for article in news_articles:
        result = finbert(article[:512])[0]  
        sentiment_score = 1 if result['label'] == "positive" else -1 if result['label'] == "negative" else 0
        total_sentiment += sentiment_score

    avg_sentiment = total_sentiment / len(news_articles)
    
    return round(avg_sentiment * 100, 2)  

# ✅ Market Condition Filter
def is_market_favorable():
    df_vix = fetch_stock_data("VIX", days=50)  
    if df_vix is None:
        return True  
    return df_vix["c"].iloc[-1] < 20  

# ✅ Rank & Return Top 10 Pre-Breakout Setups
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

        sentiment_score = analyze_sentiment_finbert(fetch_duckduckgo_news(stock))  
        relative_vol = relative_volume(stock)  
        market_favorable = is_market_favorable()  
        ad_zone = accumulation_distribution_zone(stock)  # ✅ Wyckoff Zones

        # ✅ New Weighted Confidence Formula (Added A/D Zone)
        confidence = (
            (sentiment_score * 0.2) +
            (relative_vol * 10) +  
            (20 if market_favorable else 0) +
            (15 if ad_zone == "Accumulation" else -10 if ad_zone == "Distribution" else 0)  # ✅ Gives priority to accumulation setups
        )

        trade_data.append({
            "Stock": stock,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_loss, 2),
            "Exit Target": round(exit_target, 2),
            "Relative Volume": relative_vol,
            "Market Favorable": market_favorable,
            "Wyckoff Zone": ad_zone,
            "Sentiment Score": sentiment_score,
            "Confidence %": round(confidence, 2)
        })

    return sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:10]



