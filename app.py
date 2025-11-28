import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import fitz  # PyMuPDF
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

# Baseline ratings (editable / extendable)
DEFAULT_RATINGS = {
    "ACN": [9, 8, 6, 7, 7, 8, 9, 6, 5],
    "IBM": [8, 7, 7, 6, 6, 7, 8, 6, 5],
    "INFY": [8, 8, 7, 8, 7, 7, 8, 6, 5],
    "VGT": [8, 8, 6, 8, 6, 8, 8, 7, 6],
    "VONG": [8, 8, 7, 7, 7, 8, 8, 7, 7],
    "MGC": [8, 9, 8, 6, 8, 8, 9, 7, 6],
    "VOO": [9, 9, 9, 6, 8, 9, 9, 7, 8],
}

# ---------------------------------------------------------
# Streamlit page config
st.set_page_config(page_title="AI Stock Analysis Dashboard", layout="wide")
st.title("AI Stock Analysis Dashboard")
st.caption("Upgraded version: PDF export, weighted scoring, peer comparison, and portfolio view.")

# ---------------------------------------------------------
# Sidebar controls
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Primary Ticker", value="ACN").strip().upper()
    peers = st.text_input("Peer Tickers (comma-separated)", value="IBM, INFY, VGT").upper()
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
# Helpers
def fetch_live_data(sym: str):
    tk = yf.Ticker(sym)
    # Safeguard: yfinance can raise occasionally; handle gracefully
    try:
        info = tk.info
    except Exception:
        info = {}
    price = info.get('currentPrice')
    market_cap = info.get('marketCap')
    pe = info.get('trailingPE')
    fpe = info.get('forwardPE')
    beta = info.get('beta')
    sector = info.get('sector')
    long_name = info.get('longName', sym)
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

def make_radar_figure(ratings: list, title: str):
    df = pd.DataFrame({"Category": CATEGORIES, "Rating": ratings})
    fig = px.line_polar(df, r="Rating", theta="Category", line_close=True, title=title, range_r=[0,10])
    fig.update_traces(fill="toself")
    return fig

def make_bar_figure(scores_dict: dict, title="Composite Score Comparison"):
    df = pd.DataFrame({"Company": list(scores_dict.keys()), "Composite Score": list(scores_dict.values())})
    fig = px.bar(df, x="Company", y="Composite Score", color="Company", text="Composite Score", range_y=[0,10], title=title)
    fig.update_traces(textposition="outside")
    return fig

def fig_to_png_bytes(fig) -> bytes:
    """Render Plotly figure to PNG (requires kaleido)."""
    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        # Fallback: empty bytes if kaleido fails
        return b""

def build_pdf_report(primary, snapshot, ratings, weights_pct, composite, radar_fig, compare_fig=None, portfolio_df=None, pie_fig=None):
    """Generate PDF using PyMuPDF; embed charts via PNG bytes."""
    doc = fitz.open()
    page = doc.new_page()

    # Title & header
    page.insert_text((40, 40), f"Stock Analysis Report: {snapshot['name']} ({primary})", fontsize=18)
    page.insert_text((40, 70), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=10)

    # Live snapshot
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

    # Ratings table
    y += 10
    page.insert_text((40, y), "Category Ratings (1–10):", fontsize=14); y += 20
    for cat, r, w in zip(CATEGORIES, ratings, weights_pct):
        page.insert_text((40, y), f"{cat}: {r}/10  (Weight: {w:.1f}%)", fontsize=9); y += 12

    # Composite score
    y += 10
    page.insert_text((40, y), f"Composite Score: {composite:.2f}/10", fontsize=14)

    # Insert radar chart
    y += 20
    radar_png = fig_to_png_bytes(radar_fig)
    if radar_png:
        page.insert_text((40, y), "Profile Radar:", fontsize=12); y += 10
        page.insert_image(fitz.Rect(40, y, 320, y+220), stream=BytesIO(radar_png).read())
    else:
        page.insert_text((40, y), "Radar chart (PNG) unavailable.", fontsize=10); y += 10

    # Insert comparison chart
    if compare_fig is not None:
        compare_png = fig_to_png_bytes(compare_fig)
        if compare_png:
            page.insert_text((360, y-10), "Peer Comparison:", fontsize=12)
            page.insert_image(fitz.Rect(360, y, 560, y+220), stream=BytesIO(compare_png).read())

    # Portfolio view (optional second page)
    if portfolio_df is not None:
        p2 = doc.new_page()
        p2.insert_text((40, 40), "Portfolio View", fontsize=18)
        yy = 70
        # Summary composite
        try:
            portfolio_df["Allocation"] = portfolio_df["Allocation"].astype(float)
            total_alloc = portfolio_df["Allocation"].sum()
            # Guard if not 100
            p2.insert_text((40, yy), f"Total Allocation: {total_alloc:.2f}%", fontsize=12); yy += 18
        except Exception:
            p2.insert_text((40, yy), "Total Allocation: (unavailable)", fontsize=12); yy += 18

        # Table header
        p2.insert_text((40, yy), "Holdings:", fontsize=14); yy += 18
        # Render table rows
        for _, row in portfolio_df.iterrows():
            line = f"- {row.get('Ticker')} | Name: {row.get('Name')} | Alloc: {row.get('Allocation')}% | Sector: {row.get('Sector')} | Score: {row.get('Score')}"
            p2.insert_text((40, yy), line, fontsize=10); yy += 12

        # Sector pie
        if pie_fig is not None:
            pie_png = fig_to_png_bytes(pie_fig)
            if pie_png:
                yy += 10
                p2.insert_text((40, yy), "Portfolio Sector Allocation:", fontsize=12); yy += 10
                p2.insert_image(fitz.Rect(40, yy, 320, yy+220), stream=BytesIO(pie_png).read())

    # Output buffer
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    doc.close()
    return buf

# ---------------------------------------------------------
# Main layout
col1, col2 = st.columns([3, 2])

# --- Left column: Live data & scoring
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

    hist = snapshot["history"]
    if hist is not None and not hist.empty:
        hist = hist.reset_index()
        fig_hist = px.line(hist, x="Date", y="Close", title=f"{ticker} — 1Y Price History")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No recent price history available for this ticker.")

    # Ratings input (sliders)
    st.subheader("Ratings (1–10)")
    base = DEFAULT_RATINGS.get(ticker, [7]*len(CATEGORIES))
    user_ratings = [st.slider(cat, 1, 10, base[i]) for i, cat in enumerate(CATEGORIES)]
    composite_score = compute_composite(user_ratings, norm_weights)
    st.metric("Composite Score", f"{composite_score:.2f} / 10")

# --- Right column: Radar + peer comparison + PDF
with col2:
    st.subheader("Visual Profile")
    radar_fig = make_radar_figure(user_ratings, f"{snapshot['name']} ({ticker}) — Profile")
    st.plotly_chart(radar_fig, use_container_width=True)

    compare_fig = None
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
        compare_fig = make_bar_figure(scores)
        st.plotly_chart(compare_fig, use_container_width=True)

    # PDF export button
    st.subheader("Download PDF Report")
    pdf_buf = build_pdf_report(
        ticker, snapshot, user_ratings, norm_weights, composite_score,
        radar_fig, compare_fig, portfolio_df=None, pie_fig=None
    )
    st.download_button(
        label="Download PDF",
        data=pdf_buf,
        file_name=f"{ticker}_stock_analysis_report.pdf",
        mime="application/pdf"
    )

# ---------------------------------------------------------
# Portfolio View
st.markdown("---")
st.header("Portfolio View (Allocations, Composite & Sector Mix)")

st.caption("Add tickers and allocations (%). Tip: allocations should sum to ~100%.")
default_port = pd.DataFrame({
    "Ticker": ["ACN", "VGT", "VOO"],
    "Allocation": [40.0, 30.0, 30.0]
})

editable = st.data_editor(
    default_port,
    num_rows="dynamic",
    key="portfolio_editor"
)

# Build portfolio table
portfolio_rows = []
sector_counts = {}

for _, row in editable.iterrows():
    sym = str(row.get("Ticker", "")).strip().upper()
    try:
        alloc = float(row.get("Allocation", 0))
    except Exception:
        alloc = 0.0

    if not sym:
        continue

    snap = fetch_live_data(sym)
    base = DEFAULT_RATINGS.get(sym, [7]*len(CATEGORIES))
    score = compute_composite(base, norm_weights)
    sec = snap.get("sector") or "N/A"
    portfolio_rows.append({
        "Ticker": sym,
        "Name": snap.get("name", sym),
        "Allocation": alloc,
        "Sector": sec,
        "Score": round(score, 2)
    })
    sector_counts[sec] = sector_counts.get(sec, 0.0) + alloc

portfolio_df = pd.DataFrame(portfolio_rows)

colA, colB = st.columns([2,2])
with colA:
    st.subheader("Holdings & Scores")
    st.dataframe(portfolio_df)

with colB:
    st.subheader("Portfolio Sector Allocation")
    pie_df = pd.DataFrame({"Sector": list(sector_counts.keys()), "Allocation": list(sector_counts.values())})
    if not pie_df.empty:
        pie_fig = px.pie(pie_df, names="Sector", values="Allocation", title="Sector Mix")
        st.plotly_chart(pie_fig, use_container_width=True)
    else:
        pie_fig = None
        st.info("Add tickers to see sector mix.")

# Portfolio composite (allocation-weighted)
try:
    total_alloc = portfolio_df["Allocation"].astype(float).sum()
    if total_alloc > 0:
        weighted_score = (portfolio_df["Allocation"] * portfolio_df["Score"]).sum() / total_alloc
        st.metric("Portfolio Composite Score", f"{weighted_score:.2f} / 10")
    else:
        st.metric("Portfolio Composite Score", "N/A")
except Exception:
    st.metric("Portfolio Composite Score", "N/A")

# Portfolio PDF export
st.subheader("Download Portfolio PDF")
portfolio_pdf_buf = build_pdf_report(
    ticker, snapshot, user_ratings, norm_weights, composite_score,
    radar_fig, compare_fig, portfolio_df=portfolio_df, pie_fig=pie_fig
)
st.download_button(
    label="Download Portfolio PDF",
    data=portfolio_pdf_buf,
    file_name=f"portfolio_analysis_{datetime.now().strftime('%Y%m%d')}.pdf",
    mime="application/pdf"
)

st.caption("Charts rendered via Plotly + Kaleido; PDFs via PyMuPDF. No personal data is stored.")
