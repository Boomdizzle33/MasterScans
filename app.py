def rank_best_trades(stocks):
    """Ranks the top 10 stocks based on AI sentiment and momentum."""
    trade_data = []

    for stock in stocks:
        df = fetch_stock_data(stock, days=50)
        if df is None:
            continue
        
        resistance = df['c'].rolling(window=20).max().iloc[-1]
        entry_price = resistance * 1.01  # ✅ Enter slightly above breakout level
        atr = ta.volatility.AverageTrueRange(df["h"], df["l"], df["c"]).average_true_range().iloc[-1]
        stop_loss = resistance - (2 * atr)  # ✅ Stop-loss at 2x ATR below breakout level
        exit_target = entry_price + (2 * (entry_price - stop_loss))  # ✅ 2:1 Risk-Reward
        
        sentiment_score = analyze_news_with_ai(stock)  # ✅ AI-powered sentiment
        momentum = momentum_confirmation(stock)
        confidence = (sentiment_score + (100 if momentum else 50)) / 2  # ✅ Confidence Score Calculation
        
        trade_data.append({
            "Stock": stock,
            "Entry": round(entry_price, 2),
            "Stop Loss": round(stop_loss, 2),
            "Exit Target": round(exit_target, 2),
            "Sentiment Score": sentiment_score,
            "Confidence %": round(confidence, 2)
        })

    # ✅ Sort trades by confidence and return the top 10
    trade_data = sorted(trade_data, key=lambda x: x["Confidence %"], reverse=True)[:10]
    
    return trade_data



