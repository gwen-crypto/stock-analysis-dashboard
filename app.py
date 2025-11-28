import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf

# ---------------------------------------------------------
# Page configuration
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")
st.title("Stock Analysis Dashboard (Light Version)")
st.caption("This version focuses on live data and charts only.")

# ---------------------------------------------------------
# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Enter a stock or ETF ticker", value="ACN").strip().upper()
    st.caption("Example: ACN, VGT, VOO, MGC")

# ---------------------------------------------------------
# Fetch live data
def fetch_data(symbol):
    tk = yf.Ticker(symbol)
    info = tk.info
    price = info.get("currentPrice", "N/A")
    market_cap = info.get("marketCap", "N/A")
    pe_ratio = info.get("trailingPE", "N/A")
    sector = info.get("sector", "N/A")
    name = info.get("longName", symbol)

    # 1-year history
    hist = tk.history(period="1y")
    return name, price, market_cap, pe_ratio, sector, hist

# ---------------------------------------------------------
# Display data
name, price, market_cap, pe_ratio, sector, hist = fetch_data(ticker)

st.subheader(f"Snapshot: {name} ({ticker})")
st.write(f"**Price:** {price}")
st.write(f"**Market Cap:** {market_cap}")
st.write(f"**P/E Ratio:** {pe_ratio}")
st.write(f"**Sector:** {sector}")

# ---------------------------------------------------------
# Chart
if not hist.empty:
    hist = hist.reset_index()
    fig = px.line(hist, x="Date", y="Close", title=f"{ticker} - 1 Year Price History")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No historical data available for this ticker.")

st.info("Lightweight version deployed successfully. PDF export and advanced scoring will be added later.")
