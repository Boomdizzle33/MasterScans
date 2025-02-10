import streamlit as st
import pandas as pd
import time
from scanner import rank_best_trades

st.set_page_config(page_title="🚀 AI Breakout Scanner", layout="wide")

st.title("🚀 AI-Powered Swing Trading Scanner")

uploaded_file = st.file_uploader("📂 Upload Stock List (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Ticker" in df.columns:
        stocks = df["Ticker"].tolist()
    else:
        st.error("❌ CSV file must contain a 'Ticker' column.")
        st.stop()

    progress_bar = st.progress(0)

    ranked_trades = rank_best_trades(stocks)  

    progress_bar.progress(1.0)

    st.subheader("🏆 Top 10 Pre-Breakout Setups")

    table_data = pd.DataFrame(ranked_trades, columns=[
        "Stock", "Entry", "Stop Loss", "Exit Target", "Wyckoff Zone",
        "Sentiment Score", "Confidence %"
    ])
    st.dataframe(table_data)

    st.success("🎯 Scan Complete! Ready to Trade!")



