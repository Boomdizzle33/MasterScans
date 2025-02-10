import streamlit as st
import pandas as pd
import time
from scanner import rank_best_trades

st.title("🚀 AI-Powered Swing Trading Scanner - Top 10 Pre-Breakout Setups")

uploaded_file = st.file_uploader("Upload TradingView Stock List (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Ticker" in df.columns:
        stocks = df["Ticker"].tolist()
    else:
        st.error("CSV file must contain a 'Ticker' column.")
        st.stop()

    progress_bar = st.progress(0)  # ✅ Initialize progress bar
    status_text = st.empty()  # ✅ Text area to update progress status

    num_stocks = len(stocks)
    ranked_trades = []

    # ✅ Process each stock & update progress bar dynamically
    for i, stock in enumerate(stocks):
        start_time = time.time()

        # ✅ Analyze stock & add to ranked trades
        ranked_trades.extend(rank_best_trades([stock]))

        # ✅ Update progress
        progress = (i + 1) / num_stocks
        progress_bar.progress(progress)

        # ✅ Update status text with remaining time estimate
        elapsed_time = time.time() - start_time
        remaining_time = (num_stocks - (i + 1)) * elapsed_time

        status_text.text(f"🔍 Analyzing {stock}... Estimated time left: {int(remaining_time)}s")

    # ✅ Show results after processing all stocks
    st.subheader("🏆 Top 10 Pre-Breakout Setups")
    st.dataframe(pd.DataFrame(ranked_trades))

    st.success("✅ Scan complete!")

