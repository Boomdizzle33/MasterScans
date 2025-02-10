import streamlit as st
import pandas as pd
from scanner import rank_best_trades

# Streamlit Title
st.title("🚀 AI-Powered Swing Trading Scanner - Top 10 Pre-Breakout Setups")

# File Uploader for CSV
uploaded_file = st.file_uploader("Upload TradingView Stock List (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if "Ticker" in df.columns:
        stocks = df["Ticker"].tolist()
    else:
        st.error("CSV file must contain a 'Ticker' column.")
        st.stop()

    ranked_trades = rank_best_trades(stocks)

    st.subheader("🏆 Top 10 Pre-Breakout Setups")
    st.dataframe(pd.DataFrame(ranked_trades))


