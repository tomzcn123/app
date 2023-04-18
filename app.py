import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import datetime as dt
import plotly.graph_objects as go

# Set up the app title and description
st.title("Stock Visualization App")

# Create a sidebar title
st.sidebar.title("Enter a stock symbol and select a date range")

# Get the user input for stock symbol and date range
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

options= ['SMA_EMA', 'MACD', 'Option 3']
selected_option = st.sidebar.selectbox("Choose an potential opportunity", options)


# Download stock data using yfinance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Plot the stock data with Plotly
fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'{stock_symbol} Closing Price')
st.write("Stock Price Chart")
st.plotly_chart(fig)

# Display the stock data as a table
#st.write("Stock Data")
#st.write(stock_data)


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
    fig = go.Figure()

    # Plot Close prices
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close", line=dict(color="blue")))

    # Plot MACD and Signal lines
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], name="MACD", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal'], name="Signal", line=dict(color="green")))

    # Customize layout
    fig.update_layout(title='MACD Strategy', xaxis_title='Date', yaxis_title='Price')

    return fig


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
    st.plotly_chart(fig) 
    st.write("Latest Position: ", latest_position)
    st.write("Win Loss Ratio: ", win_loss_ratio)
    st.write("Profit Ratio: ", profit_ratio)
    st.write(data)

       
    
   








