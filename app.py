import streamlit as st
import pandas as pd
from scanner import detect_patterns, dynamic_support_resistance, analyze_sentiment, momentum_confirmation, stop_loss_exit

# Streamlit Title
st.title("🚀 AI-Powered Swing Trading Scanner")

# File Uploader for CSV
uploaded_file = st.file_uploader("Upload TradingView Stock List (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "Symbol" not in df.columns:
        st.error("CSV file must contain a 'Symbol' column.")
    else:
        stocks = df["Symbol"].tolist()

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
