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
        stocks = df["Ticker"].tolist()  # ✅ Use "Ticker" from TradingView CSV
    else:
        st.error("CSV file must contain a 'Ticker' column.")
        st.stop()

    # ✅ Rank the best trades
    ranked_trades = rank_best_trades(stocks)

    # ✅ Display the results in a Streamlit DataFrame
    st.subheader("🏆 Top 10 Pre-Breakout Setups")
    st.dataframe(pd.DataFrame(ranked_trades))

    # ✅ Display insights
    for trade in ranked_trades:
        st.markdown(f"""
        **📌 {trade["Stock"]}**  
        - **Entry Price:** ${trade["Entry"]}  
        - **Stop Loss:** ${trade["Stop Loss"]}  
        - **Exit Target:** ${trade["Exit Target"]}  
        - **Sentiment Score:** {trade["Sentiment Score"]}  
        - **Confidence:** {trade["Confidence %"]}%  
        """)



