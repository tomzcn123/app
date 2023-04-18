import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np
import datetime as dt

# Set up the app title and description
st.title("Stock Visualization App")

# Create a sidebar title
st.sidebar.title("Enter a stock symbol and select a date range")

# Get the user input for stock symbol and date range
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("today"))

options= ['SMA_EMA', 'Option 2', 'Option 3']
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
    
if selected_option == 'Option 2':
    st.write("You have selected Option 1. Here is the information related to Option 121221.")







def calculate_sma_ema(data, short_period, long_period):
    data['SMA'] = data['Close'].rolling(window=short_period).mean()
    data['EMA'] = data['Close'].ewm(span=long_period).mean()
    return data

def generate_signals(data):
    data['Signal'] = 0
    data.loc[data['EMA'] > data['SMA'], 'Signal'] = 1
    data.loc[data['EMA'] < data['SMA'], 'Signal'] = -1
    return data

def get_trades(data):
    win_loss = []
    profit =[]
    position = None
    entry_price = None
    
    for i in range(len(data)):
        current_signal = data.iloc[i]['Signal']
        previous_signal = data.iloc[i - 1]['Signal'] if i > 0 else None
        
        if current_signal == 1 and previous_signal != 1:
            position = 'Long'
            entry_price = data.iloc[i]['Close']
        elif current_signal == -1 and previous_signal != -1 and position == 'Long':
            exit_price = data.iloc[i]['Close']
            pf = (exit_price -entry_price)/entry_price
            profit.append(pf)
            if pf > 0:
                win_loss.append(1)
            elif pf <= 0:
                win_loss.append(0)
            position = None
            entry_price = None
    print("The win loss ratio for EMA and SMA strategy is " + str(round(sum(win_loss)/len(win_loss),3)) + " The profit ratio for EMA and SMA strategy is " + str(round(sum(profit)/len(profit),3)))
    return 

def sma_ema_strategy(data, short_period, long_period):
    data = calculate_sma_ema(data, short_period, long_period)
    data = generate_signals(data)
    trades = get_trades(data)
    return
if selected_option == 'SMA_EMA':
    st.write(sma_ema_strategy(stock_data, 10, 50))




