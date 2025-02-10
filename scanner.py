import requests
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from polygon import RESTClient
from config import POLYGON_API_KEY
from duckduckgo_search import DDGS  
from transformers import pipeline
from bs4 import BeautifulSoup

# ✅ Load FinBERT Model (Preventing Crashes)
try:
    finbert = pipeline("text-classification", model="ProsusAI/finbert", device=-1)  # Run on CPU (safe mode)
except Exception as e:
    print(f"⚠️ FinBERT loading failed: {e}")
    finbert = None  

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

# ✅ Fetch News from DuckDuckGo
def fetch_duckduckgo_news(ticker):
    """Fetches top financial news headlines for a stock using DuckDuckGo."""
    search_query = f"{ticker} stock news"
    ddgs = DDGS()
    results = list(ddgs.news(search_query, max_results=5)) 
    
    if results:
        return [result['title'] + ". " + result.get('body', '') for result in results]  
    return []

# ✅ Fetch News from Yahoo Finance
def fetch_yahoo_finance_news(ticker):
    """Fetches news from Yahoo Finance."""
    url = f"https://finance.yahoo.com/quote/{ticker}/news?p={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}  
    
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("h3")
        return [article.text for article in articles[:5]]  
    return []

# ✅ Analyze Sentiment Using FinBERT
def analyze_sentiment_finbert(news_articles):
    """Analyzes sentiment using FinBERT (better for financial news)."""
    if not news_articles:
        return 0  

    total_sentiment = 0
    for article in news_articles:
        result = finbert(article[:512])[0]  
        sentiment_score = 1 if result['label'] == "positive" else -1 if result['label'] == "negative" else 0
        total_sentiment += sentiment_score

    avg_sentiment = total_sentiment / len(news_articles)
    
    return round(avg_sentiment * 100, 2)  

# ✅ Weighted Sentiment Scoring
def weighted_sentiment_score(ticker):
    """Calculates a weighted sentiment score based on news type."""
    
    general_news = fetch_duckduckgo_news(ticker)  
    yahoo_news = fetch_yahoo_finance_news(ticker)  
    
    sentiment_general = analyze_sentiment_finbert(general_news)
    sentiment_yahoo = analyze_sentiment_finbert(yahoo_news)

    weighted_score = (
        (sentiment_general * 0.2) +  
        (sentiment_yahoo * 0.4)  
    )
    
    return round(weighted_score, 2)

# ✅ Dynamic Stop-Loss Calculation
def dynamic_stop_loss(ticker):
    """Calculates a dynamic stop-loss based on volatility (ATR) and volume contraction."""
    df = fetch_stock_data(ticker, days=50)
    if df is None:
        return None

    atr = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range().iloc[-1]
    return df["c"].iloc[-1] - (1.5 * atr) if True else df["c"].iloc[-1] - (2.5 * atr)

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

        sentiment_score = weighted_sentiment_score(stock)  
        confidence = (sentiment_score * 0.25) + 50  

        trade_data.append({
            "Stock": stock,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_loss, 2),
            "Exit Target": round(exit_target, 2),
            "Sentiment Score": sentiment_score,
            "Confidence %": round(confidence, 2)
        })

    return sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:10]



