import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import datetime as dt
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from matplotlib.backends.backend_agg import RendererAgg


# Set up the app title and description
st.title("Stock Visualization App")

# Create a sidebar title
st.sidebar.title("Enter a stock symbol and select a date range")

# Get the user input for stock symbol and date range
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

options= ['MACD','SMA_EMA','RSI','KDJ','WR','BB','Hammer Strategy']
selected_option = st.sidebar.selectbox("Choose an potential opportunity", options)


# Download stock data using yfinance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Plot the stock data with Plotly
fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'{stock_symbol} Closing Price')
st.write("Stock Price Chart")
st.plotly_chart(fig)

# Display the stock data as a table
st.write("Stock Data")
st.write(stock_data)


# SMA_EMA strategy
def sma_ema_strategy(data, short_period, long_period):
    # Calculate SMA and EMA
    data['SMA'] = data['Close'].rolling(window=short_period).mean()
    data['EMA'] = data['Close'].ewm(span=long_period).mean()

    # Generate signals
    data['Signal'] = 0
    data.loc[data['EMA'] > data['SMA'], 'Signal'] = 1
    data.loc[data['EMA'] < data['SMA'], 'Signal'] = -1

    # Get trades
    win_loss = []
    profit = []
    position = None
    entry_price = None
    latest_position = None

    for i in range(len(data)):
        current_signal = data.iloc[i]['Signal']
        previous_signal = data.iloc[i - 1]['Signal'] if i > 0 else None

        if current_signal == 1 and previous_signal != 1:
            if position == 'Short':  # Close short position
                exit_price = data.iloc[i]['Close']
                pf = (entry_price - exit_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Long'
            entry_price = data.iloc[i]['Close']
        elif current_signal == -1 and previous_signal != -1:
            if position == 'Long':  # Close long position
                exit_price = data.iloc[i]['Close']
                pf = (exit_price - entry_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Short'
            entry_price = data.iloc[i]['Close']
    latest_position = position
    win_loss_ratio = round(sum(win_loss) / len(win_loss), 3)
    profit_ratio = round(sum(profit) / len(profit), 3)
    return data, win_loss_ratio, profit_ratio,latest_position

def plot_sma_ema_strategy(data):
    fig = go.Figure()
    # Plot the Close prices, SMA, and EMA lines
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], name="SMA", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA'], name="EMA", line=dict(color="green")))
    # Add markers for buy signals (Long positions)
    buy_signals = data[data['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy', marker=dict(color='lime', size=8, symbol='circle')))
    # Add markers for sell signals (Short positions)
    sell_signals = data[data['Signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell', marker=dict(color='red', size=8, symbol='circle')))
    # Customize layout
    fig.update_layout(title='SMA-EMA Crossover Strategy', xaxis_title='Date', yaxis_title='Price')
    return fig

# MACD strategy
def macd_strategy(data, short_period, long_period, signal_period):
    # Calculate MACD and Signal
    data['EMA_short'] = data['Close'].ewm(span=short_period).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_period).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['Signal'] = data['MACD'].ewm(span=signal_period).mean()

    # Generate signals
    data['Signal_flag'] = 0
    data.loc[data['MACD'] > data['Signal'], 'Signal_flag'] = 1
    data.loc[data['MACD'] < data['Signal'], 'Signal_flag'] = -1

    # Process trades and calculate win-loss ratio, profit ratio, and latest position
    win_loss = []
    profit = []
    position = None
    entry_price = None
    latest_position = None

    for i in range(len(data)):
        current_signal = data.iloc[i]['Signal_flag']
        previous_signal = data.iloc[i - 1]['Signal_flag'] if i > 0 else None

        if current_signal == 1 and previous_signal != 1:
            if position == 'Short':  # Close short position
                exit_price = data.iloc[i]['Close']
                pf = (entry_price - exit_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Long'
            entry_price = data.iloc[i]['Close']
        elif current_signal == -1 and previous_signal != -1:
            if position == 'Long':  # Close long position
                exit_price = data.iloc[i]['Close']
                pf = (exit_price - entry_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Short'
            entry_price = data.iloc[i]['Close']

    # Save the latest position
    latest_position = position
    win_loss_ratio = round(sum(win_loss) / len(win_loss), 3)
    profit_ratio = round(sum(profit) / len(profit), 3)
    return data, win_loss_ratio, profit_ratio, latest_position

def plot_macd_strategy(data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)

    # Plot the price chart
    ax1.plot(data.index, data['Close'], label='Close', linewidth=1, alpha=0.8)
    ax1.set_title('Close Price')
    ax1.set_ylabel('Price')

    # Plot buy and sell signals
    ax1.plot(data[data['Signal_flag'] == 1].index, data['Close'][data['Signal_flag'] == 1], '^', markersize=10, color='g', label='Buy signal')
    ax1.plot(data[data['Signal_flag'] == -1].index, data['Close'][data['Signal_flag'] == -1], 'v', markersize=10, color='r', label='Sell signal')

    # Plot the MACD chart
    ax2.plot(data.index, data['MACD'], label='MACD', linewidth=1, alpha=0.8)
    ax2.plot(data.index, data['Signal'], label='Signal', linewidth=1, alpha=0.8)
    ax2.set_title('MACD & Signal')
    ax2.set_ylabel('Value')

    # Plot the MACD line chart with buy and sell signals
    ax3.plot(data.index, data['MACD'] - data['Signal'], label='MACD Line', linewidth=1, alpha=0.8)
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.8)
    ax3.plot(data[data['Signal_flag'] == 1].index, (data['MACD'] - data['Signal'])[data['Signal_flag'] == 1], '^', markersize=10, color='g', label='Buy signal')
    ax3.plot(data[data['Signal_flag'] == -1].index, (data['MACD'] - data['Signal'])[data['Signal_flag'] == -1], 'v', markersize=10, color='r', label='Sell signal')
    ax3.set_title('MACD Line')
    ax3.set_ylabel('Value')
    ax3.set_xlabel('Date')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    return fig

#RSI strategy
def rsi_strategy_single(data, rsi_period, rsi_low, rsi_high):
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (delta.where(delta < 0, 0)).fillna(0).abs()

    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()

    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Generate RSI signals
    data['Signal_flag'] = 0
    data.loc[data['RSI'] < rsi_low, 'Signal_flag'] = 1
    data.loc[data['RSI'] > rsi_high, 'Signal_flag'] = -1

    # Get RSI trades
    win_loss = []
    profit = []
    position = None
    entry_price = None
    latest_position = None

    for i in range(len(data)):
        current_signal = data.iloc[i]['Signal_flag']
        previous_signal = data.iloc[i - 1]['Signal_flag'] if i > 0 else None

        if current_signal == 1 and previous_signal != 1:
            if position == 'Short':  # Close short position
                exit_price = data.iloc[i]['Close']
                pf = (entry_price - exit_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Long'
            entry_price = data.iloc[i]['Close']
        elif current_signal == -1 and previous_signal != -1:
            if position == 'Long':  # Close long position
                exit_price = data.iloc[i]['Close']
                pf = (exit_price - entry_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Short'
            entry_price = data.iloc[i]['Close']

        # Save the latest position
        latest_position = position

    win_loss_ratio = round(sum(win_loss) / len(win_loss), 3)
    profit_ratio = round(sum(profit) / len(profit), 3)
    return data, win_loss_ratio, profit_ratio, latest_position

def plot_rsi_strategy(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Plot the price chart
    ax1.plot(data.index, data['Close'], label='Close', linewidth=1, alpha=0.8)
    ax1.set_title('Close Price')
    ax1.set_ylabel('Price')

    # Plot buy and sell signals
    ax1.plot(data[data['Signal_flag'] == 1].index, data['Close'][data['Signal_flag'] == 1], '^', markersize=10, color='g', label='Buy signal')
    ax1.plot(data[data['Signal_flag'] == -1].index, data['Close'][data['Signal_flag'] == -1], 'v', markersize=10, color='r', label='Sell signal')

    # Plot the RSI chart
    ax2.plot(data.index, data['RSI'], label='RSI', linewidth=1, alpha=0.8)
    ax2.axhline(y=30, color='r', linestyle='--', linewidth=1, alpha=0.8)
    ax2.axhline(y=70, color='r', linestyle='--', linewidth=1, alpha=0.8)
    ax2.set_title('RSI')
    ax2.set_ylabel('RSI Value')
    ax2.set_xlabel('Date')

    ax1.legend()
    ax2.legend()

    plt.show()
    
    
#KDJ strategy
def kdj_strategy(data, k_period=14, d_period=3, j_period=3, buy_level=20, sell_level=80):
    # Calculate KDJ
    data['Lowest_Low'] = data['Low'].rolling(window=k_period).min()
    data['Highest_High'] = data['High'].rolling(window=k_period).max()
    data['%K'] = ((data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])) * 100
    data['%D'] = data['%K'].rolling(window=d_period).mean()
    data['%J'] = (3 * data['%D']) - (2 * data['%K'])

    # Generate KDJ signals
    data['Signal_flag'] = 0
    data.loc[data['%K'] < buy_level, 'Signal_flag'] = 1
    data.loc[data['%K'] > sell_level, 'Signal_flag'] = -1

    # Get KDJ trades
    win_loss = []
    profit = []
    position = None
    entry_price = None
    latest_position = None

    for i in range(len(data)):
        current_signal = data.iloc[i]['Signal_flag']
        previous_signal = data.iloc[i - 1]['Signal_flag'] if i > 0 else None

        if current_signal == 1 and previous_signal != 1:
            if position == 'Short':  # Close short position
                exit_price = data.iloc[i]['Close']
                pf = (entry_price - exit_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Long'
            entry_price = data.iloc[i]['Close']
        elif current_signal == -1 and previous_signal != -1:
            if position == 'Long':  # Close long position
                exit_price = data.iloc[i]['Close']
                pf = (exit_price - entry_price) / entry_price
                profit.append(pf)
                win_loss.append(1 if pf > 0 else 0)
                position = None
                entry_price = None

            position = 'Short'
            entry_price = data.iloc[i]['Close']

        # Save the latest position
        latest_position = position

    win_loss_ratio = round(sum(win_loss) / len(win_loss), 3)
    profit_ratio = round(sum(profit) / len(profit), 3)

    return data, win_loss_ratio, profit_ratio, latest_position

def plot_kdj_signals(data):
    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Plot KDJ lines
    fig.add_trace(go.Scatter(x=data.index, y=data['%K'], mode='lines', name='%K', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['%D'], mode='lines', name='%D', line=dict(color='red')), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['%J'], mode='lines', name='%J', line=dict(color='green')), row=1, col=1)

    # Plot buy and sell signals
    buys = data[data['Signal_flag'] == 1]
    sells = data[data['Signal_flag'] == -1]
    fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=8, symbol='triangle-up')), row=2, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=8, symbol='triangle-down')), row=2, col=1)

    # Plot closing price
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close', line=dict(color='black')), row=2, col=1)

    # Update layout
    fig.update_layout(
        title='KDJ and Buy/Sell Signals',
        xaxis_title='Date',
        legend=dict(orientation='h', yanchor='bottom', xanchor='right', y=1.02, x=1),
        template='plotly_white'
    )

    fig.update_yaxes(title_text="KDJ Values", row=1, col=1)
    fig.update_yaxes(title_text="Close Price", row=2, col=1)
    return fig



#WR strategy
def wr_strategy_and_ratios(data, period=14, low_wr=-80, high_wr=-20):
    data['High_max'] = data['High'].rolling(window=period).max()
    data['Low_min'] = data['Low'].rolling(window=period).min()
    data['WR'] = -100 * ((data['High_max'] - data['Close']) / (data['High_max'] - data['Low_min']))
    
    data['Buy'] = ((data['WR'] > low_wr) & (data['WR'].shift(1) <= low_wr))
    data['Sell'] = ((data['WR'] < high_wr) & (data['WR'].shift(1) >= high_wr))

    win_loss = []
    profit = []
    position = None
    entry_price = None
    latest_position = None

    for i in range(len(data)):
        if data.iloc[i]['Buy'] and position is None:
            position = 'Long'
            entry_price = data.iloc[i]['Close']
        elif data.iloc[i]['Sell'] and position == 'Long':
            exit_price = data.iloc[i]['Close']
            pf = (exit_price - entry_price) / entry_price
            profit.append(pf)
            if pf > 0:
                win_loss.append(1)
            elif pf <= 0:
                win_loss.append(0)
            position = None
            entry_price = None
        
        latest_position = position

    win_loss_ratio = round(sum(win_loss) / len(win_loss), 3)
    profit_ratio = round(sum(profit) / len(profit), 3)
    return data, win_loss_ratio, profit_ratio, latest_position

def plot_wr_and_strategy(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot buy and sell signals
    ax1.plot(data['Close'], label='Close Price', alpha=0.4)
    buy_signals = data[data['Buy']]
    ax1.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green')
    sell_signals = data[data['Sell']]
    ax1.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red')
    ax1.set_title('Williams %R Strategy Buy and Sell Signals')
    ax1.set_ylabel('Close Price')
    ax1.legend(loc='upper left')
    
    # Plot Williams %R
    ax2.plot(data['WR'], label='Williams %R', color='blue')
    ax2.axhline(y=-80, color='r', linestyle='--', label='Low WR')
    ax2.axhline(y=-20, color='g', linestyle='--', label='High WR')
    ax2.set_title('Williams %R')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Williams %R')
    ax2.legend(loc='upper left')
    
    return fig

#Bollinger Band
def bollinger_bands_strategy_and_bands(data, period=20):
    # Calculate Bollinger Bands
    data['SMA'] = data['Close'].rolling(window=period).mean()
    data['STD'] = data['Close'].rolling(window=period).std()
    data['Upper_Band'] = data['SMA'] + 2 * data['STD']
    data['Lower_Band'] = data['SMA'] - 2 * data['STD']
    
    # Bollinger Bands Strategy
    position = None
    entry_price = None
    win_loss = []
    profit = []
    
    for i in range(len(data)):
        current_close = data.iloc[i]['Close']
        upper_band = data.iloc[i]['Upper_Band']
        lower_band = data.iloc[i]['Lower_Band']
        
        if current_close < lower_band and position is None:
            position = 'Long'
            entry_price = current_close
        elif current_close > upper_band and position == 'Long':
            exit_price = current_close
            pf = (exit_price - entry_price) / entry_price
            profit.append(pf)
            if pf > 0:
                win_loss.append(1)
            elif pf <= 0:
                win_loss.append(0)
            position = None
            entry_price = None
    
    win_loss_ratio = round(sum(win_loss) / len(win_loss), 3)
    profit_ratio = round(sum(profit) / len(profit), 3)
    latest_position = position
    
    return data, win_loss_ratio, profit_ratio, latest_position

def plot_bollinger_bands_strategy_and_bands(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot Buy and Sell Signals
    ax1.plot(data['Close'], label='Close Price', alpha=0.4)
    buy_signals = data[data['Close'] < data['Lower_Band']]
    ax1.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green')
    sell_signals = data[data['Close'] > data['Upper_Band']]
    ax1.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red')
    ax1.set_title('Bollinger Bands Strategy Buy and Sell Signals')
    ax1.set_ylabel('Close Price')
    ax1.legend(loc='upper left')
    
    # Plot Bollinger Bands
    ax2.plot(data['Close'], label='Close Price', alpha=0.4)
    ax2.plot(data['Upper_Band'], label='Upper Band', linestyle='--', color='red')
    ax2.plot(data['Lower_Band'], label='Lower Band', linestyle='--', color='red')
    ax2.plot(data['SMA'], label='SMA', linestyle='-', color='blue')
    ax2.set_title('Bollinger Bands')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Close Price')
    ax2.legend(loc='upper left')
    return fig

#Hammer Strategy
def hammer_strategy(data):
    data['bullish_hammer'] = ((data['Open'] - data['High']).abs() <= 0.01 * (data['High'] - data['Low'])) & \
                           ((data['Close'] - data['Low']).abs() <= 0.01 * (data['High'] - data['Low'])) & \
                           ((data['Close'] > data['Open']) & ((data['Close'] - data['Open']) > 0.6 * (data['High'] - data['Low'])))
    data['bearish_hammer'] = ((data['High'] - data['Open']).abs() <= 0.01 * (data['High'] - data['Low'])) & \
                           ((data['Close'] - data['Low']).abs() <= 0.01 * (data['High'] - data['Low'])) & \
                           ((data['Open'] > data['Close']) & ((data['Open'] - data['Close']) > 0.6 * (data['High'] - data['Low'])))

    position = None
    entry_price = None
    win_loss = []
    profit = []

    for i in range(1, len(data)):
        bullish_hammer = data.iloc[i]['bullish_hammer']
        bearish_hammer = data.iloc[i]['bearish_hammer']

        if bullish_hammer and position is None:
            position = 'Long'
            entry_price = data.iloc[i]['Close']
        elif bearish_hammer and position == 'Long':
            exit_price = data.iloc[i]['Close']
            pf = (exit_price - entry_price) / entry_price
            profit.append(pf)
            if pf > 0:
                win_loss.append(1)
            elif pf <= 0:
                win_loss.append(0)
            position = None
            entry_price = None

        if bearish_hammer and position is None:
            position = 'Short'
            entry_price = data.iloc[i]['Close']
        elif bullish_hammer and position == 'Short':
            exit_price = data.iloc[i]['Close']
            pf = (entry_price - exit_price) / entry_price
            profit.append(pf)
            if pf > 0:
                win_loss.append(1)
            elif pf <= 0:
                win_loss.append(0)
            position = None
            entry_price = None

    win_loss_ratio = round(np.sum(win_loss) / len(win_loss), 3) if len(win_loss) > 0 else float('NaN')
    profit_ratio = round(np.sum(profit) / len(profit), 3) if len(profit) > 0 else float('NaN')
    current_bullish_hammer = data.iloc[-1]['bullish_hammer']
    current_bearish_hammer = data.iloc[-1]['bearish_hammer']

    return data,win_loss_ratio, profit_ratio, position, current_bullish_hammer, current_bearish_hammer


def plot_hammer_strategy_and_patterns(data):
    fig, ax = plt.subplots(figsize=(12, 6))

    buy_signals = data[data['bullish_hammer']]
    sell_signals = data[data['bearish_hammer']]

    ax.plot(data.index, data['Close'], label='Close Price', alpha=0.5)
    ax.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='g', label='Buy Signal / Bullish Hammer', alpha=1)
    ax.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='r', label='Sell Signal / Bearish Hammer', alpha=1)

    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Hammer Strategy and Hammer Patterns')
    ax.legend(loc='best')

    plt.show()







if selected_option == 'SMA_EMA':
    short_period = st.sidebar.slider("Short Period", min_value=5, max_value=50, value=10, step=1)
    long_period = st.sidebar.slider("Long Period", min_value=50, max_value=200, value=50, step=1)
    data, win_loss_ratio, profit_ratio,position= sma_ema_strategy(stock_data, short_period, long_period)
    fig = plot_sma_ema_strategy(data)
    st.plotly_chart(fig)
    st.write("Win Loss Ratio: ", win_loss_ratio)
    st.write("Profit Ratio: ", profit_ratio)
    st.write("Current Recommended Position: ", position)
    
elif selected_option == "MACD":
    short_period = st.sidebar.slider("Short Period", min_value=5, max_value=50, value=12, step=1)
    long_period = st.sidebar.slider("Long Period", min_value=20, max_value=100, value=26, step=1)
    signal_period = st.sidebar.slider("Signal Period", min_value=5, max_value=50, value=9, step=1)
    data, win_loss_ratio, profit_ratio, latest_position = macd_strategy(stock_data, short_period, long_period, signal_period)
    fig = plot_macd_strategy(data)
    st.pyplot(fig)
    st.write("Win Loss Ratio: ", win_loss_ratio)
    st.write("Profit Ratio: ", profit_ratio)
    st.write("Latest Position: ", latest_position)
    
elif selected_option == "RSI":
    rsi_period = st.sidebar.slider("RSI Period", 1, 100, 14)
    rsi_low = st.sidebar.slider("RSI Low", 1, 100, 30)
    rsi_high = st.sidebar.slider("RSI High", 1, 100, 70)
    data, win_loss_ratio, profit_ratio, latest_position = rsi_strategy_single(stock_data, rsi_period, rsi_low, rsi_high)
    st.write("RSI Strategy")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig = plot_rsi_strategy(data)
    st.pyplot(fig)
    st.write(f"Win Loss Ratio: {win_loss_ratio}")
    st.write(f"Profit Ratio: {profit_ratio}")
    st.write(f"Latest Position: {latest_position}")

elif selected_option == "KDJ":
    k_period = st.sidebar.slider('K Period', min_value=1, max_value=100, value=14)
    d_period = st.sidebar.slider('D Period', min_value=1, max_value=100, value=3)
    j_period = st.sidebar.slider('J Period', min_value=1, max_value=100, value=3)
    buy_level = st.sidebar.slider('Buy Level', min_value=0, max_value=100, value=20)
    sell_level = st.sidebar.slider('Sell Level', min_value=0, max_value=100, value=80)
    data, win_loss_ratio, profit_ratio, latest_position = kdj_strategy(stock_data, k_period, d_period, j_period, buy_level, sell_level)
    fig = plot_kdj_signals(data)
    st.plotly_chart(fig)
    st.write(f"Win Loss Ratio: {win_loss_ratio}")
    st.write(f"Profit Ratio: {profit_ratio}")
    st.write(f"Latest Position: {latest_position}")
    
elif selected_option == "WR":
    st.title('Williams %R Strategy')
    # Create sliders
    period = st.sidebar.slider('Period', min_value=1, max_value=50, value=14, step=1)
    low_wr = st.sidebar.slider('Low WR', min_value=-100, max_value=0, value=-80, step=1)
    high_wr = st.sidebar.slider('High WR', min_value=-100, max_value=0, value=-20, step=1)
    # Calculate the strategy and ratios
    data, win_loss_ratio, profit_ratio, latest_position = wr_strategy_and_ratios(stock_data, period, low_wr, high_wr)
    # Plot the strategy
    fig = plot_wr_and_strategy(data)
    st.pyplot(fig)
    # Display the results
    st.write(f"Win Loss Ratio: {win_loss_ratio}")
    st.write(f"Profit Ratio: {profit_ratio}")
    st.write(f"Latest Position: {latest_position}")
    
elif selected_option == "BB":
    st.title('Bollinger Bands Strategy')

    # Create a slider for the period in the sidebar
    period = st.sidebar.slider('Period', min_value=1, max_value=50, value=20, step=1)

    # Calculate the strategy and bands with the selected period
    data, win_loss_ratio, profit_ratio, latest_position = bollinger_bands_strategy_and_bands(stock_data, period)
    
    # Plot the strategy and Bollinger Bands
    fig = plot_bollinger_bands_strategy_and_bands(data)
    st.pyplot(fig)
    
    # Display the results
    st.write(f"Win Loss Ratio: {win_loss_ratio}")
    st.write(f"Profit Ratio: {profit_ratio}")
    st.write(f"Latest Position: {latest_position}")
    
elif selected_option == "Hammer Strategy":
   
    _lock = RendererAgg.lock
    data,win_loss_ratio, profit_ratio, position, current_bullish_hammer, current_bearish_hammer = hammer_strategy(stock_data)
    with _lock:
        st.pyplot(plot_hammer_strategy_and_patterns(data))
    # Display strategy results
    st.write(f"Win Loss Ratio: {win_loss_ratio}")
    st.write(f"Profit Ratio: {profit_ratio}")
    st.write(f"Latest Position: {position}")
    st.write(f"Current Bullish Hammer: {current_bullish_hammer}")
    st.write(f"Current Bearish Hammer: {current_bearish_hammer}")
    st.write(data)
   

    

    
    
    
    
   








