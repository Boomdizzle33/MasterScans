import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from polygon import RESTClient
from config import POLYGON_API_KEY
from duckduckgo_search import DDGS  # ✅ Fixed DuckDuckGo import
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ✅ Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

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

# ✅ Fetch News from DuckDuckGo (FIXED)
def fetch_duckduckgo_news(ticker):
    """Fetches top financial news headlines for a stock using DuckDuckGo."""
    search_query = f"{ticker} stock news"
    ddgs = DDGS()  # ✅ Initialize the DuckDuckGo search object
    results = list(ddgs.news(search_query, max_results=5))  # ✅ Fixed method to get news
    
    if results:
        return [result['title'] + ". " + result.get('body', '') for result in results]  
    return []

# ✅ Analyze Sentiment with VADER NLP
def analyze_sentiment_duckduckgo(ticker):
    """Analyzes sentiment of stock news headlines using DuckDuckGo & VADER NLP."""
    
    news_headlines = fetch_duckduckgo_news(ticker)
    if not news_headlines:
        return 0  

    total_sentiment = 0
    for headline in news_headlines:
        sentiment_score = analyzer.polarity_scores(headline)["compound"]  
        total_sentiment += sentiment_score

    avg_sentiment = total_sentiment / len(news_headlines)
    
    return round(avg_sentiment * 100, 2)  

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

        sentiment_score = analyze_sentiment_duckduckgo(stock)  
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



