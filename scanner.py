import streamlit as st
import pandas as pd
from scanner import detect_patterns, dynamic_support_resistance, analyze_sentiment, momentum_confirmation, stop_loss_exit

# Streamlit Title
st.title("🚀 AI-Powered Swing Trading Scanner")

# File Uploader for CSV
uploaded_file = st.file_uploader("Upload TradingView Stock List (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Ensure the CSV file contains either "Tickers" or "Symbol" columns
    if "Tickers" in df.columns:
        stocks = df["Tickers"].tolist()
    elif "Symbol" in df.columns:
        stocks = df["Symbol"].tolist()
    else:
        st.error("CSV file must contain either a 'Tickers' or 'Symbol' column.")
        st.stop()  # Stop execution if columns are missing

    # Process stocks
    for stock in stocks:
        st.subheader(f"📌 {stock}")
        
        patterns = detect_patterns(stock)
        if patterns:
            st.info(f"📊 Patterns Detected: {patterns}")

        if dynamic_support_resistance(stock):
            st.success("🔥 Stock Near Breakout Zone")

        if analyze_sentiment(stock):
            st.success("📈 AI Confirms Bullish Sentiment")

        if momentum_confirmation(stock):
            st.success("⚡ Momentum Confirmation (RSI, MACD)")

        stop_loss = stop_loss_exit(stock)
        if stop_loss:
            st.warning(f"🚨 AI Stop-Loss: ${stop_loss:.2f}")

    st.write("📊 AI-Confirmed Pre-Breakout Stocks Ready for Trade.")

