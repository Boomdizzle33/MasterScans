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

# ✅ Fetch Raw News Data from Polygon.io
def fetch_polygon_news(ticker):
    """Fetches raw news articles for a stock from Polygon.io."""
    url = f"https://api.polygon.io/v2/reference/news?ticker={ticker}&limit=5&apiKey={POLYGON_API_KEY}"
    
    response = requests.get(url).json()
    if 'results' in response:
        return response['results']  # Returns news articles
    return None

# ✅ AI-Powered Sentiment Analysis on News Data
def analyze_news_with_ai(ticker):
    """Uses OpenAI GPT-4 to analyze news sentiment."""
    
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


