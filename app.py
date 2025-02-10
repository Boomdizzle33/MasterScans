import streamlit as st
import pandas as pd
import time
from scanner import rank_best_trades

# ✅ Configure Streamlit Page
st.set_page_config(page_title="🚀 AI Breakout Scanner", layout="wide")

st.title("🚀 AI-Powered Swing Trading Scanner")

# 📂 Upload CSV File (TradingView Export)
uploaded_file = st.file_uploader("📂 Upload Stock List (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Ticker" in df.columns:
        stocks = df["Ticker"].tolist()
    else:
        st.error("❌ CSV file must contain a 'Ticker' column.")
        st.stop()

    # ✅ Progress Bar & Status
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()  
    ranked_trades = rank_best_trades(stocks)  

    for i in range(len(stocks)):
        progress_bar.progress((i + 1) / len(stocks))
        elapsed_time = time.time() - start_time
        estimated_time_remaining = (elapsed_time / (i + 1)) * (len(stocks) - (i + 1))
        status_text.text(f"🔍 Scanning {stocks[i]}... Estimated time left: {int(estimated_time_remaining)}s")
        time.sleep(0.5)  

    progress_bar.progress(1.0)
    status_text.text("✅ Scan Complete! Showing Best Pre-Breakout Setups")

    # ✅ Display Table of Top 10 Setups
    st.subheader("🏆 Top 10 Pre-Breakout Setups")

    table_data = pd.DataFrame(ranked_trades, columns=[
        "Stock", "Entry", "Stop Loss", "Exit Target",
        "Technical Score", "Relative Volume", "Market Favorable",
        "Wyckoff Zone", "Sentiment Score", "Confidence %"
    ])
    st.dataframe(table_data)

    st.success("🎯 Scan Complete! Ready to Trade!")



