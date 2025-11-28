import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import fitz  # PyMuPDF
from io import BytesIO
from datetime import datetime

# -------------------------------
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

st.set_page_config(page_title="AI Stock Analysis Dashboard", layout="wide")
st.title("AI Stock Analysis Dashboard")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Primary Ticker", value="ACN").strip().upper()
    peers = st.text_input("Peer Tickers (comma-separated)", value="IBM, INFY").upper()
    st.caption("Example: IBM, INFY, CTSH")

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

def fetch_live_data(sym: str):
    tk = yf.Ticker(sym)
    info = tk.info
    # Snapshot
    price = info.get('currentPrice')
    market_cap = info.get('marketCap')
    pe = info.get('trailingPE')
    fpe = info.get('forwardPE')
    beta = info.get('beta')
    sector = info.get('sector')
    longName = info.get('longName', sym)
    # History (1Y)
    hist = tk.history(period="1y")
    return {
        "symbol": sym,
        "name": longName,
        "currentPrice": price,
        "marketCap": market_cap,
        "pe": pe,
        "forwardPe": fpe,
        "beta": beta,
        "sector": sector,
        "history": hist
    }

def compute_composite(ratings: list, weights_pct: list) -> float:
    return float(np.dot(ratings, weights_pct) / sum(weights_pct))

def radar_chart(ratings: list, title: str) -> BytesIO:
    df = pd.DataFrame({"Category": CATEGORIES, "Rating": ratings})
    fig = px.line_polar(df, r="Rating", theta="Category", line_close=True, title=title, range_r=[0,10])
    fig.update_traces(fill="toself")
    buf = BytesIO(); fig.write_image(buf, format="png"); buf.seek(0)
    return buf

def bar_comparison(scores_dict: dict) -> BytesIO:
    df = pd.DataFrame({"Company": list(scores_dict.keys()), "Composite Score": list(scores_dict.values())})
    fig = px.bar(df, x="Company", y="Composite Score", color="Company", text="Composite Score", range_y=[0,10],
                 title="Composite Score Comparison")
    fig.update_traces(textposition="outside")
    buf = BytesIO(); fig.write_image(buf, format="png"); buf.seek(0)
    return buf

def pdf_report(primary, snapshot, ratings, weights_pct, composite, radar_png, compare_png) -> BytesIO:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 40), f"Stock Analysis Report: {snapshot['name']} ({primary})", fontsize=18)
    page.insert_text((40, 70), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10)

    y = 100
    page.insert_text((40, y), "Live Snapshot:", fontsize=14); y += 20
    for label, key in [
        ("Current Price", "currentPrice"),
        ("Market Cap", "marketCap"),
        ("Trailing P/E", "pe"),
        ("Forward P/E", "forwardPe"),
        ("Beta", "beta"),
        ("Sector", "sector")
    ]:
        page.insert_text((40, y), f"- {label}: {snapshot.get(key)}", fontsize=10); y += 14

    y += 10
    page.insert_text((40, y), "Category Ratings (1–10):", fontsize=14); y += 20
    for cat, r, w in zip(CATEGORIES, ratings, weights_pct):
        page.insert_text((40, y), f"{cat}: {r}/10  (Weight: {w:.1f}%)", fontsize=9); y += 12

    y += 10
    page.insert_text((40, y), f"Composite Score: {composite:.2f}/10", fontsize=14)

    y += 20
    page.insert_text((40, y), "Profile Radar:", fontsize=12); y += 10
    page.insert_image(fitz.Rect(40, y, 320, y+220), stream=radar_png.read())

    if compare_png:
        y = 360
        page.insert_text((360, 340), "Peer Comparison:", fontsize=12)
        page.insert_image(fitz.Rect(360, 360, 560, 560), stream=compare_png.read())

    buf = BytesIO(); doc.save(buf); buf.seek(0); doc.close()
    return buf

DEFAULT_RATINGS = { "ACN": [9,8,6,7,7,8,9,6,5] }

col1, col2 = st.columns([3,2])

with col1:
    st.subheader("Live Data & Charts")
    try:
        snapshot = fetch_live_data(ticker)
    except Exception as e:
        st.error("Unable to fetch live data. Try again or check ticker symbol.")
        st.exception(e)
        st.stop()

    st.write(f"**{snapshot['name']} ({ticker})** — Sector: {snapshot['sector']}")
    st.write(f"Price: {snapshot['currentPrice']} • Market Cap: {snapshot['marketCap']} • P/E: {snapshot['pe']} • Forward P/E: {snapshot['forwardPe']} • Beta: {snapshot['beta']}")

    if snapshot["history"] is not None and not snapshot["history"].empty:
        hist = snapshot["history"].reset_index()
        fig_hist = px.line(hist, x="Date", y="Close", title=f"{ticker} — 1Y Price History")
        st.plotly_chart(fig_hist, use_container_width=True)

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
        scores = {snapshot['name']: composite_score}
        for sym in [p.strip() for p in peers.split(",") if p.strip()]:
            try:
                peer_snap = fetch_live_data(sym)
                peer_base = DEFAULT_RATINGS.get(sym, [7]*len(CATEGORIES))
                peer_score = compute_composite(peer_base, norm_weights)
                scores[peer_snap['name']] = round(peer_score, 2)
            except Exception:
                scores[sym] = np.nan
        compare_png = bar_comparison(scores)
        st.image(compare_png, caption="Composite Score Comparison", use_column_width=True)

    st.subheader("Download Report")
    pdf_buf = pdf_report(ticker, snapshot, user_ratings, norm_weights, composite_score, radar_png, compare_png)
    st.download_button(
        label="Download PDF Report",
        data=pdf_buf,
        file_name=f"{ticker}_stock_analysis_report.pdf",
        mime="application/pdf"
    )

st.info("Tip: Adjust weights and ratings to reflect your own judgment. The composite score updates instantly.")
st.caption("Data source: Yahoo Finance via yfinance. Charts: Plotly. PDF: PyMuPDF.")
