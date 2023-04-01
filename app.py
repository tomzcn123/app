import pandas as pd
import pandas_datareader.data as web
import streamlit as st
import datetime as dt
import plotly.graph_objs as go

st.title("Stock Visualization App")

# Input stock ticker and date range
ticker = st.text_input("Enter a stock ticker:", "AAPL")
start_date = st.date_input("Start date:", dt.datetime(2020, 1, 1))
end_date = st.date_input("End date:", dt.datetime.now())

# Fetch stock data using pandas-datareader
try:
    data = web.get_data_yahoo(ticker, start=start_date, end=end_date)
except Exception as e:
    st.write(f"Error fetching data: {e}")
    data = pd.DataFrame()

# Check if data is available
if data.empty:
    st.write("No data available for the selected ticker and date range.")
else:
    # Display the fetched data
    st.write("Stock data:")
    st.dataframe(data)

    # Plot the stock data
    fig = go.Figure(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=ticker,
        )
    )

    fig.update_layout(title=f"{ticker} Stock Price", xaxis_title="Date", yaxis_title="Price (USD)")

    st.write("Stock price chart:")
    st.plotly_chart(fig)

