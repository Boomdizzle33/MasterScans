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

    # ✅ Scanner Status Indicator (New)
    status_text = st.empty()
    status_text.info("⏳ Scanner is initializing...")

    # ✅ Progress Bar (Live Updates)
    progress_bar = st.progress(0)

    start_time = time.time()

    with st.spinner("🔍 Scanning stocks, please wait..."):
        ranked_trades = rank_best_trades(stocks)

    progress_bar.progress(1.0)
    status_text.success("✅ Scan Complete! Showing Best Pre-Breakout Setups")

    st.subheader("🏆 Top 20 Pre-Breakout Setups")

    table_data = pd.DataFrame(ranked_trades)
    st.dataframe(table_data)

    st.success("🎯 Scan Complete! Ready to Trade!")

    csv = table_data[["Stock"]].rename(columns={"Stock": "Ticker"}).to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download CSV for TradingView",
        data=csv,
        file_name="tradingview_import.csv",
        mime="text/csv",
    )




