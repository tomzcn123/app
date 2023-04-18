import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

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

# Create a sidebar title
st.sidebar.title("My Sidebar")

# Add a text input widget to the sidebar
user_input = st.sidebar.text_input("Enter some text")

# Add a selectbox widget to the sidebar
options = ['Option 1', 'Option 2', 'Option 3']
selected_option = st.sidebar.selectbox("Choose an option", options)

# Add a slider widget to the sidebar
slider_value = st.sidebar.slider("Select a value", min_value=0, max_value=100, value=50)

# Add a checkbox widget to the sidebar
checkbox_value = st.sidebar.checkbox("Check me")

# Display the values of the widgets in the main content area
st.write("You entered:", user_input)
st.write("You selected:", selected_option)
st.write("Slider value:", slider_value)
st.write("Checkbox value:", checkbox_value)





def calculate_macd(prices, fast_period, slow_period, signal_period):
    # Convert prices to a pandas DataFrame
    df = pd.DataFrame({'price': prices})

    # Calculate the fast and slow EMA
    df['fast_ema'] = df['price'].ewm(span=fast_period, adjust=False).mean()
    df['slow_ema'] = df['price'].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD line
    df['macd'] = df['fast_ema'] - df['slow_ema']

    # Calculate the signal line
    df['signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()

    return df

def identify_cross_points(df):
    # Create a new column to indicate the direction of the MACD line
    df['macd_direction'] = np.where(df['macd'] > df['signal'], 1, -1)

    # Identify the cross points by looking for changes in the direction of the MACD line
    cross_points = np.where(np.diff(df['macd_direction']) != 0)[0]

    return cross_points

def trade_based_on_macd(df):
    position = 0
    profits_1 = []
    profits_15 =[]
    profits_30 = []
    profits_60 =[]
    WL_1 = []
    WL_15 =[]
    WL_30 = []
    WL_60 =[]
    for i in range(len(df)):
        if i < 1:
            continue
        if df['macd'].iloc[i-1] < df['signal'].iloc[i-1] and df['macd'].iloc[i] > df['signal'].iloc[i]:
            buy_price = df['price'].iloc[i]
            if i+1 < len(df):
                sell_price_1 = df['price'].iloc[i+1]
                pf_1 = sell_price_1 - buy_price
                profits_1.append(pf_1/buy_price)
                if pf_1 > 0:
                    WL_1.append(1)
                elif pf_1 < 0:
                    WL_1.append(0)
            if i+15 < len(df):  
                sell_price_15 = df['price'].iloc[i+15]
                pf_15 = sell_price_15 - buy_price
                profits_15.append(pf_15/buy_price)
                if pf_15 > 0:
                    WL_15.append(1)
                elif pf_15 < 0:
                    WL_15.append(0)
            if i+30 < len(df):  
                sell_price_30 = df['price'].iloc[i+30]
                pf_30 = sell_price_30 - buy_price
                profits_30.append(pf_30/buy_price)
                if pf_30 > 0:
                    WL_30.append(1)
                elif pf_30 < 0:
                    WL_30.append(0)
            if i+60 < len(df):  
                sell_price_60 = df['price'].iloc[i+60]
                pf_60 = sell_price_60 - buy_price
                profits_60.append(pf_60/buy_price)
                if pf_60 > 0:
                    WL_60.append(1)
                elif pf_60 < 0:
                    WL_60.append(0)
    if len(WL_1) > 0:
            print("The one day win loss ratio is " + str(round(sum(WL_1)/len(WL_1),3)) + " The one day profit ratio is " + str(round(sum(profits_1)/len(profits_1),3)))
    else:
            print("The one day win loss ratio is 0 " + " The one day profit ratio is " + str(round(sum(profits_1)/len(profits_1),3)))
    if len(WL_15) > 0:
        print("The fifteen day win loss ratio is " + str(round(sum(WL_15)/len(WL_15),3)) + " The fifteen day profit ratio is " + str(round(sum(profits_15)/len(profits_15),3)))
    else:
        print("The fifteen day win loss ratio is 0 " + " The fifteen day profit ratio is " + str(round(sum(profits_15)/len(profits_15),3)))
    if len(WL_30) > 0:    
        print("The thirty day win loss ratio is " + str(round(sum(WL_30)/len(WL_30),3)) + " The thirty day profit ratio is " + str(round(sum(profits_30)/len(profits_30),3)))
    else:
        print("The thirty day win loss ratio is 0 " + " The thirty day profit ratio is " + str(round(sum(profits_30)/len(profits_30),3)))
    if len(WL_60) > 0: 
        print("The sixty day win loss ratio is " + str(round(sum(WL_60)/len(WL_60),3))+ " The sixty day profit ratio is " + str(round(sum(profits_60)/len(profits_60),3)))
    else:        
        print("The sixty day win loss ratio is 0 " + " The sixty day profit ratio is " + str(round(sum(profits_60)/len(profits_60),3)))
    return
