import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
from scanner import rank_best_trades, fetch_stock_data, fetch_duckduckgo_news

# ✅ Set Streamlit Page Configuration
st.set_page_config(page_title="🚀 AI Breakout Scanner", layout="wide")

st.title("🚀 AI-Powered Swing Trading Scanner - Top 10 Pre-Breakout Setups")

# 📂 Upload CSV File from TradingView
uploaded_file = st.file_uploader("📂 Upload TradingView Stock List (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "Ticker" in df.columns:
        stocks = df["Ticker"].tolist()
    else:
        st.error("❌ CSV file must contain a 'Ticker' column.")
        st.stop()

    # ✅ Initialize Progress Bar & Timer
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()  # Track start time

    st.subheader("⏳ Scanning for the Best Breakout Setups...")
    
    # ✅ Start Processing
    ranked_trades = rank_best_trades(stocks)  # ✅ Only fetch the Top 10 Best Breakout Trades

    for i, stock in enumerate(stocks):
        progress = (i + 1) / len(stocks)
        progress_bar.progress(progress)

        elapsed_time = time.time() - start_time
        estimated_time_remaining = (elapsed_time / (i + 1)) * (len(stocks) - (i + 1))

        status_text.text(f"🔍 Scanning {stock}... Estimated time left: {int(estimated_time_remaining)}s")
        time.sleep(0.5)  

    # ✅ Scan Complete
    progress_bar.progress(1.0)
    status_text.text("✅ Scan Complete! Showing Best Pre-Breakout Setups")

    # 📊 Display Ranked Setups in Table
    st.subheader("🏆 Top 10 Pre-Breakout Setups")
    st.dataframe(pd.DataFrame(ranked_trades))

    # 📈 Show Interactive Charts & Sentiment for Each Setup
    for trade in ranked_trades:
        stock = trade["Stock"]
        entry_price = trade["Entry"]
        stop_loss = trade["Stop Loss"]
        exit_target = trade["Exit Target"]
        sentiment_score = trade["Sentiment Score"]
        confidence = trade["Confidence %"]

        # 📌 Expandable Section for Each Stock
        with st.expander(f"📊 {stock} - Confidence: {confidence}% | Sentiment: {sentiment_score}%"):
            # ✅ Fetch Stock Data for Chart
            df = fetch_stock_data(stock, days=50)
            if df is not None:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(df.index, df["c"], label="Close Price", color="blue")
                ax.axhline(entry_price, color="green", linestyle="dashed", label="Entry Price")
                ax.axhline(stop_loss, color="red", linestyle="dashed", label="Stop Loss")
                ax.axhline(exit_target, color="purple", linestyle="dashed", label="Target Price")
                ax.set_title(f"{stock} Price Chart")
                ax.set_ylabel("Price ($)")
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning(f"⚠️ No chart data available for {stock}")

            # 📰 Show Latest News Headlines
            st.subheader("📰 Latest News & Sentiment")
            news_headlines = fetch_duckduckgo_news(stock)
            if news_headlines:
                for headline in news_headlines:
                    st.write(f"- {headline}")
            else:
                st.warning("⚠️ No recent news found.")

    st.success("🎯 Ready to Trade? Use This Scanner to Find the Best Setups!")


