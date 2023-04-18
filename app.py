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

options= ['SMA_EMA', 'Option 2', 'Option 3']
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
    


if selected_option == 'SMA_EMA':
    short_period = st.sidebar.slider("Short Period", min_value=5, max_value=50, value=10, step=1)
    long_period = st.sidebar.slider("Long Period", min_value=50, max_value=200, value=50, step=1)









