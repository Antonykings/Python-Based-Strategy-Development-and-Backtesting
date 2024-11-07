import os
import pandas as pd
import pandas_ta as ta
import numpy as np


def load_data(file_path, frequency='D'):

    df = pd.read_csv(file_path, delimiter=';')
    
    # Rename columns
    df.rename(columns={'timeOpen': 'Date', 'open': 'Open', 'high': 'High', 
                       'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    
    # Convert 'Date'by chronological order
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Resample if frequency is weekly ('W')
    if frequency == 'W':
        df = df.set_index('Date').resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna().reset_index()
    
    return df

# Weekly Timeframe
def determine_weekly_signal(df_weekly):
    # 30-period SMA
    df_weekly['SMA_30'] = ta.sma(df_weekly['Close'], length=30)
    
    # Determine weekly close and SMA
    if df_weekly['Close'].iloc[-1] > df_weekly['SMA_30'].iloc[-1]:
        return 'buy'
    else:
        return 'sell'


# Daily Timeframe
def apply_daily_strategy(df_daily, weekly_signal):

    df_daily['RSI_14'] = ta.rsi(df_daily['Close'], length=14)
    df_daily['SMA_30'] = ta.sma(df_weekly['Close'], length=30)
    df_daily['ATR_14'] = ta.atr(df_daily['High'], df_daily['Low'], df_daily['Close'], length=14)
    
    # Initialize trade signals
    df_daily['Signal'] = None
    df_daily['Stop_Loss'] = None
    df_daily['Take_Profit'] = None
    
    for i in range(1, len(df_daily)):
        # Buy Triggers
        if weekly_signal == 'buy':

            if df_daily['RSI_14'].iloc[i] < 55:
                df_daily.loc[i, 'Signal'] = 'buy'
                
                # Dynamic stop-loss
                df_daily.loc[i, 'Stop_Loss'] = df_daily['SMA_30'].iloc[i] - 2 * df_daily['ATR_14'].iloc[i]
                
                # Take profit
                df_daily.loc[i, 'Take_Profit'] = df_daily['Close'].iloc[i] * 1.25
        
        # Sell Triggers
        elif weekly_signal == 'sell':

            if df_daily['RSI_14'].iloc[i] > 70:
                df_daily.loc[i, 'Signal'] = 'sell'
                
                # Dynamic stop-loss
                df_daily.loc[i, 'Stop_Loss'] = df_daily['SMA_30'].iloc[i] + 2 * df_daily['ATR_14'].iloc[i]
                
                # Take profit
                df_daily.loc[i, 'Take_Profit'] = df_daily['Close'].iloc[i] * 0.75

    return df_daily

# Backtesting function
def backtest_strategy(df_weekly, df_daily):
    results = [] 

    # Determine weekly buy or sell signal
    weekly_signal = determine_weekly_signal(df_weekly)

    # Apply strategy on daily data and get detailed trading signals
    df_daily = apply_daily_strategy(df_daily, weekly_signal)

    trade_no = 0
    wins = 0
    losses = 0

    for i in range(1, len(df_daily)):
        # Entry condition
        if df_daily['Signal'].iloc[i] in ['buy', 'sell']:
            entry_price = df_daily['Close'].iloc[i]
            entry_date = df_daily['Date'].iloc[i]
            direction = df_daily['Signal'].iloc[i]
            trade_no += 1

            # Track until trade exit condition
            for j in range(i + 1, len(df_daily)):
                exit_triggered = False
                exit_price = df_daily['Close'].iloc[j]
                exit_date = df_daily['Date'].iloc[j]

                # Check stop-loss and take-profit conditions
                if direction == 'buy':
                    if exit_price <= df_daily['Stop_Loss'].iloc[i] or exit_price >= df_daily['Take_Profit'].iloc[i]:
                        exit_triggered = True
                elif direction == 'sell':
                    if exit_price >= df_daily['Stop_Loss'].iloc[i] or exit_price <= df_daily['Take_Profit'].iloc[i]:
                        exit_triggered = True

                # If exit is triggered, record trade results
                if exit_triggered:
                    trade_duration = (exit_date - entry_date).days
                    profit_pct = ((exit_price - entry_price) / entry_price) * 100 if direction == 'buy' else ((entry_price - exit_price) / entry_price) * 100
                    win = profit_pct > 0
                    wins += int(win)
                    losses += int(not win)

                    # Record trade details
                    results.append({
                        "Trade No": trade_no,
                        "Entry Date": entry_date,
                        "Exit Date": exit_date,
                        "Duration (Days)": trade_duration,
                        "Profit %": round(profit_pct, 2),
                        "Win": win
                    })
                    break  

    # Calculate overall performance
    total_trades = wins + losses
    win_loss_ratio = round(wins / losses, 2) if losses != 0 else float('inf')

    # Create DataFram
    results_df = pd.DataFrame(results)
    summary = {
        "Total Trades": total_trades,
        "Wins": wins,
        "Losses": losses,
        "Win/Loss Ratio": win_loss_ratio
    }
    
    return df_daily, results_df, summary

# Performance metrics
def calculate_performance_metrics(df_daily):
    # Daily returns
    df_daily['Daily_Returns'] = df_daily['Close'].pct_change()

    # Sharpe Ratio
    risk_free_rate = 0.1157 / 365  # BTC Funding Rate APY 
    excess_daily_return = df_daily['Daily_Returns'].mean() - risk_free_rate
    sharpe_ratio = excess_daily_return / df_daily['Daily_Returns'].std()

    # Maximum Drawdown
    cumulative_returns = (1 + df_daily['Daily_Returns']).cumprod()
    running_max = cumulative_returns.cummax()
    drawdowns = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdowns.min()

    metrics = {
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown
    }
    
    return metrics

# Load CSV files
weekly_file_path = 'data/BONK_W.csv'  
daily_file_path = 'data/BONK.csv'    

# Extract base name from daily file path
base_name = os.path.splitext(os.path.basename(daily_file_path))[0]

df_weekly = load_data(weekly_file_path, frequency='W')
df_daily = load_data(daily_file_path, frequency='D')

# Run backtest and calculate metrics
df_daily, results_df, summary = backtest_strategy(df_weekly, df_daily)
performance_metrics = calculate_performance_metrics(df_daily)

# Add Sharpe Ratio and Max Drawdown to the summary
summary["Sharpe Ratio"] = performance_metrics["Sharpe Ratio"]
summary["Maximum Drawdown"] = performance_metrics["Maximum Drawdown"]

# Print results
print("Trade Results:\n", results_df)
print("\nSummary:\n", summary)
print("\nPerformance Metrics:\n", performance_metrics)

# Export Trade Results to CSV file with dynamic naming
trade_results_filename = f'Trade_Results_{base_name}.csv'
results_df.to_csv(trade_results_filename, index=False)

# Export summary with Sharpe Ratio and Maximum Drawdown to CSV with dynamic naming
summary_df = pd.DataFrame([summary])  
trade_summary_filename = f'Trade_Results_Summary_{base_name}.csv'
summary_df.to_csv(trade_summary_filename, index=False)

print(f"\nFiles saved as:\n - {trade_results_filename}\n - {trade_summary_filename}")
