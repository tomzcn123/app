import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# Set up the app title and description
st.title("Stock Visualization App")
st.write("""
Simple app to visualize stock price data. Enter a stock symbol and select a date range.
""")

# Get the user input for stock symbol and date range
stock_symbol = st.text_input("Enter Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

# Download stock data using yfinance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Display the stock data as a table
st.write("Stock Data")
st.write(stock_data)

# Plot the stock data with Plotly
fig = px.line(stock_data, x=stock_data.index, y='Close', title=f'{stock_symbol} Closing Price')
st.write("Stock Price Chart")
st.plotly_chart(fig)
