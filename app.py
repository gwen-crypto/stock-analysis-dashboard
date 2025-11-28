import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from io import BytesIO
from datetime import datetime

# ---------------------------------------------------------
# Best-practice categories & default weights
CATEGORIES = [
    'Business Model & Industry Understanding',
    'Financial Health Analysis',
    'Valuation Metrics',
    'Growth Catalysts',
    'Risk Assessment',
    'Competitive Benchmarking',
    'Management & Governance',
    'Technical & Sentiment Analysis',
    'Diversification Context'
]
DEFAULT_WEIGHTS = [20, 20, 15, 15, 10, 7, 6, 5, 2]

# Baseline ratings (you can edit or add more tickers here):
DEFAULT_RATINGS = {
    "ACN": [9, 8, 6, 7, 7, 8, 9, 6, 5],  # Accenture example baseline
    "IBM": [8, 7, 7, 6, 6, 7, 8, 6, 5],
    "INFY": [8, 8, 7, 8, 7, 7, 8, 6, 5],
    "VGT": [8, 8, 6, 8, 6, 8, 8, 7, 6],  # ETF examples if you want to compare
    "VONG": [8, 8, 7, 7, 7, 8, 8, 7, 7],
    "MGC": [8, 9, 8, 6, 8, 8, 9, 7, 6],
    "VOO": [9, 9, 9, 6, 8, 9, 9, 7, 8],
}

# ---------------------------------------------------------
# Streamlit page config
st.set_page_config(page_title="AI Stock Analysis Dashboard", layout="wide")
st.title("AI Stock Analysis Dashboard")
st.caption("Lightweight version (no PDF export). You can add PDF later once the app is live.")

# ---------------------------------------------------------
# Sidebar controls
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Primary Ticker", value="ACN").strip().upper()
    peers = st.text_input("Peer Tickers (comma-separated)", value="IBM, INFY").upper()
    st.caption("Examples: IBM, INFY, VGT, VONG, MGC, VOO")

    st.header("Weights (sum should be ~100%)")
    weights = []
    for i, cat in enumerate(CATEGORIES):
        w = st.number_input(f"{cat}", min_value=0, max_value=100, value=DEFAULT_WEIGHTS[i], step=1)
        weights.append(w)

    total_w = sum(weights)
    if total_w == 0:
        weights = DEFAULT_WEIGHTS
        total_w = 100
    norm_weights = [w * 100 / total_w for w in weights]
    st.caption(f"Total weight: {total_w} → normalized to 100% for scoring")

# ---------------------------------------------------------
# Helper functions
def fetch_live_data(sym: str):
    """Fetch a live snapshot and 1-year price history for the symbol using yfinance."""
    tk = yf.Ticker(sym)
    try:
        info = tk.info  # yfinance will handle cookies; Streamlit Cloud has internet access
    except Exception:
        info = {}

    price = info.get('currentPrice')
    market_cap = info.get('marketCap')
    pe = info.get('trailingPE')
    fpe = info.get('forwardPE')
    beta = info.get('beta')
    sector = info.get('sector')
    long_name = info.get('longName', sym)

    # History (1Y)
    hist = pd.DataFrame()
    try:
        hist = tk.history(period="1y")
    except Exception:
        pass

    return {
        "symbol": sym,
        "name": long_name,
        "currentPrice": price,
        "marketCap": market_cap,
        "pe": pe,
        "forwardPe": fpe,
        "beta": beta,
        "sector": sector,
        "history": hist
    }

def compute_composite(ratings: list, weights_pct: list) -> float:
    """Weighted average on 1–10 scale."""
    return float(np.dot(ratings, weights_pct) / sum(weights_pct))

def radar_chart(ratings: list, title: str):
    """Return a BytesIO PNG of a Plotly radar chart."""
    df = pd.DataFrame({"Category": CATEGORIES, "Rating": ratings})
    fig = px.line_polar(df, r="Rating", theta="Category", line_close=True, title=title, range_r=[0,10])
    fig.update_traces(fill="toself")
    buf = BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    return buf

def bar_comparison(scores_dict: dict):
    """Return a BytesIO PNG of a bar chart comparison."""
    df = pd.DataFrame({"Company": list(scores_dict.keys()), "Composite Score": list(scores_dict.values())})
    fig = px.bar(df, x="Company", y="Composite Score", color="Company", text="Composite Score", range_y=[0,10],
                 title="Composite Score Comparison")
    fig.update_traces(textposition="outside")
    buf = BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    return buf

# ---------------------------------------------------------
# Layout
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Live Data & Charts")
    snapshot = fetch_live_data(ticker)
    st.write(f"**{snapshot['name']} ({ticker})** — Sector: {snapshot.get('sector', 'N/A')}")
    st.write(
        f"Price: {snapshot.get('currentPrice', 'N/A')} • "
        f"Market Cap: {snapshot.get('marketCap', 'N/A')} • "
        f"P/E: {snapshot.get('pe', 'N/A')} • Forward P/E: {snapshot.get('forwardPe', 'N/A')} • "
        f"Beta: {snapshot.get('beta', 'N/A')}"
    )

    # History chart
    hist = snapshot["history"]
    if hist is not None and not hist.empty:
        hist = hist.reset_index()
        fig_hist = px.line(hist, x="Date", y="Close", title=f"{ticker} — 1Y Price History")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No recent price history available for this ticker.")

    # Ratings input (editable sliders)
    st.subheader("Ratings (1–10)")
    base = DEFAULT_RATINGS.get(ticker, [7]*len(CATEGORIES))
    user_ratings = [st.slider(cat, 1, 10, base[i]) for i, cat in enumerate(CATEGORIES)]
    composite_score = compute_composite(user_ratings, norm_weights)
    st.metric("Composite Score", f"{composite_score:.2f} / 10")

with col2:
    st.subheader("Visual Profile")
    radar_png = radar_chart(user_ratings, f"{snapshot['name']} ({ticker}) — Profile")
    st.image(radar_png, caption="Best Practices Radar", use_column_width=True)

    compare_png = None
    if peers:
        st.subheader("Peer Comparison")
        scores = {snapshot['name']: round(composite_score, 2)}
        for sym in [p.strip() for p in peers.split(",") if p.strip()]:
            try:
                peer_snap = fetch_live_data(sym)
                peer_base = DEFAULT_RATINGS.get(sym, [7]*len(CATEGORIES))
                peer_score = compute_composite(peer_base, norm_weights)
                scores[peer_snap.get('name', sym)] = round(peer_score, 2)
            except Exception:
                scores[sym] = np.nan
        compare_png = bar_comparison(scores)
        st.image(compare_png, caption="Composite Score Comparison", use_column_width=True)

    # Lightweight export: CSV of ratings & weights
    st.subheader("Download Ratings/Weights (CSV)")
    export_df = pd.DataFrame({
        "Category": CATEGORIES,
        "Rating": user_ratings,
        "Weight (%)": norm_weights
    })
    csv_buf = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv_buf,
        file_name=f"{ticker}_ratings_weights_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
st.info("This is the lightweight version without PDF export to ensure fast deployment. We can add PDF back later once your app is running smoothly.")
st.caption("Data source: Yahoo Finance via yfinance. Charts: Plotly. No personal or sensitive data stored.")
