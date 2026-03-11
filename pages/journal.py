"""
pages/journal.py
AI Journal page — drop into your existing pages/ folder.
Add to your app.py nav:
    st.sidebar.page_link("pages/journal.py", label="AI Journal")
"""

import sys
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide")

# utils/ lives one level up from pages/
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from data_manager import (
    load_market_thoughts, load_stock_thoughts, load_watchlist,
    load_trade_log, load_corrections, load_daily_protocol,
    build_master_journal, _join_all,
    BASE_DIR, MARKET_THOUGHTS_PATH, STOCK_THOUGHTS_PATH, WATCHLIST_PATH,
    TRADE_LOG_PATH, CORRECTIONS_PATH, MASTER_PATH, PROTOCOL_PATH,
    load_or_generate_demo, build_daily_protocol,
)
from watchlist_curator import render_curator
from voice_analytics import (
    render_kpis, render_market_pulse, render_psychology,
    render_selection_funnel, render_stock_intelligence,
    render_tigers_analysis, render_memo_explorer,
)
from watchlist_annotator import render_watchlist_annotator
from trade_annotator import render_trade_annotator
from regime_analysis import render_regime_analysis
from memo_analysis import render_memo_analysis

# ── Path to indices file (one level up from pages/) ───────────────────────────
INDICES_PATH   = Path(__file__).parent.parent / "data" / "indices_with_breadth.csv"
MASTER_DF_PATH      = Path(__file__).parent.parent / "data" / "master_df.csv"
MEMO_ANALYSIS_PATH  = Path(__file__).parent.parent / "data" / "memo_analysis.json"

# ── STYLES ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, .stApp { font-family: 'DM Sans', sans-serif !important; }
[data-testid="metric-container"] {
    background: #1a1d25 !important;
    border: 1px solid #252830 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Journal Settings")

    use_demo = st.toggle("Use demo data", value=False,
                          help="Toggle on to explore with generated sample data")

    if not use_demo:
        if st.button("🔄 Rebuild Master Journal", use_container_width=True):
            with st.spinner("Joining data sources..."):
                build_master_journal()
            st.success("✅ Rebuilt")
            st.rerun()

    st.markdown("---")

    # Date range filter — only shown when real data is loaded
    if not use_demo:
        mdf_check = load_market_thoughts()
        if not mdf_check.empty:
            min_d = mdf_check["date"].min().date()
            max_d = mdf_check["date"].max().date()
            dr = st.date_input("Date range", value=(min_d, max_d),
                                min_value=min_d, max_value=max_d)
        else:
            dr = None
    else:
        dr = None

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
if use_demo:
    if "journal_demo" not in st.session_state:
        with st.spinner("Generating demo data..."):
            st.session_state["journal_demo"] = load_or_generate_demo()
    mdf, sdf, wdf, tdf, master, protocol = st.session_state["journal_demo"]
    corrections = {}

else:
    corrections = load_corrections()
    mdf         = load_market_thoughts()
    sdf         = load_stock_thoughts(corrections=corrections)
    wdf         = load_watchlist()
    tdf         = load_trade_log()
    protocol    = load_daily_protocol()

    # Auto-build master if it doesn't exist yet or is stale
    if Path(MASTER_PATH).exists():
        master = pd.read_csv(MASTER_PATH, parse_dates=["date"])
    elif not sdf.empty or not wdf.empty:
        with st.spinner("Building master journal for the first time..."):
            master = _join_all(sdf, mdf, wdf, tdf)
            protocol = build_daily_protocol(master, mdf, save_path=PROTOCOL_PATH)
    else:
        master = pd.DataFrame()

    # Show what was found
    found = []
    if not wdf.empty:    found.append(f"✅ Watchlist ({len(wdf)} rows)")
    if not tdf.empty:    found.append(f"✅ Trades ({len(tdf)} rows)")
    if not mdf.empty:    found.append(f"✅ Market memos ({len(mdf)} rows)")
    if not sdf.empty:    found.append(f"✅ Stock thoughts ({len(sdf)} rows)")
    missing = []
    if wdf.empty:        missing.append("watchlist_curation.csv")
    if tdf.empty:        missing.append("raw_trades.csv")
    if mdf.empty:        missing.append("market_thoughts.csv")
    if sdf.empty:        missing.append("stock_thoughts.csv")

    if found:
        with st.sidebar:
            st.markdown("**Data loaded:**")
            for f in found: st.markdown(f)
            if missing:
                st.markdown("**Not found:**")
                for m in missing: st.markdown(f"⬜ {m}")

    # Apply date range filter
    if dr and len(dr) == 2:
        d0, d1 = pd.Timestamp(dr[0]), pd.Timestamp(dr[1])
        if not mdf.empty:
            mdf = mdf[(mdf["date"] >= d0) & (mdf["date"] <= d1)]
        if not master.empty:
            master = master[(master["date"] >= d0) & (master["date"] <= d1)]
        if not protocol.empty:
            protocol = protocol[(protocol["date"] >= d0) & (protocol["date"] <= d1)]

# ── PAGE HEADER ────────────────────────────────────────────────────────────────
demo_badge = "&nbsp;&nbsp;<span style='font-size:11px;color:#f59e0b;border:1px solid #f59e0b44;border-radius:4px;padding:2px 8px;background:#f59e0b11'>DEMO</span>" if use_demo else ""
st.markdown(
    f'<h2 style="font-family:Space Mono,monospace;font-weight:700;margin-bottom:4px">'
    f'📊 AI Journal{demo_badge}'
    f'</h2>',
    unsafe_allow_html=True
)

# ── DIAGNOSTIC (remove once data loads correctly) ──────────────────────────────
if not use_demo:
    with st.expander("🔍 Data path diagnostic — expand if charts are empty", expanded=master.empty):
        st.code(f"BASE_DIR: {BASE_DIR}")
        st.code(f"BASE_DIR exists: {BASE_DIR.exists()}")
        for label, path, df in [
            ("watchlist_curation.csv", WATCHLIST_PATH, wdf),
            ("raw_trades.csv",         TRADE_LOG_PATH, tdf),
            ("market_thoughts.csv",    MARKET_THOUGHTS_PATH, mdf),
            ("stock_thoughts.csv",     STOCK_THOUGHTS_PATH, sdf),
            ("master_journal.csv",     MASTER_PATH, master),
        ]:
            exists = Path(path).exists()
            rows   = len(df) if not df.empty else 0
            status = f"✅ found — {rows} rows loaded" if exists else "❌ NOT FOUND"
            st.markdown(f"`{label}` → `{path}`  {status}")

st.markdown("---")

# ── PAGE SELECTOR ──────────────────────────────────────────────────────────────
page = st.radio(
    "View",
    ["🎙️ Voice Analytics", "📋 Daily Curation", "✏️ Annotations", "🌍 Regime Analysis", "🧠 Memo Analysis"],
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown("")

# ── RENDER ─────────────────────────────────────────────────────────────────────
if "Voice Analytics" in page:
    render_kpis(master, mdf, protocol)
    st.markdown("---")

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📈 Market Pulse",
        "🧠 Psychology",
        "🔍 Selection Funnel",
        "📋 Stock Intel",
        "🐯 TIGERS",
        "🗂️ Memo Explorer",
    ])
    with t1: render_market_pulse(master, mdf)
    with t2: render_psychology(master, mdf)
    with t3: render_selection_funnel(master, protocol)
    with t4: render_stock_intelligence(master)
    with t5: render_tigers_analysis(master)
    with t6: render_memo_explorer(master, mdf)

elif "Daily Curation" in page:
    render_curator(mdf, sdf, wdf, corrections, protocol)

elif "Annotations" in page:
    if use_demo:
        st.info("Annotation editing is disabled in demo mode — switch off 'Use demo data' to annotate real trades.")
    else:
        ann_tab1, ann_tab2 = st.tabs(["📋 Watchlist", "📝 Trades"])
        with ann_tab1:
            render_watchlist_annotator(wdf, WATCHLIST_PATH, MASTER_DF_PATH)
        with ann_tab2:
            render_trade_annotator(tdf, TRADE_LOG_PATH, MASTER_DF_PATH)

elif "Memo Analysis" in page:
    if use_demo:
        st.info("Memo analysis requires real data — switch off Use demo data.")
    else:
        render_memo_analysis(
            sdf=sdf, mdf=mdf, wdf=wdf, tdf=tdf,
            output_path=MEMO_ANALYSIS_PATH,
            watchlist_path=WATCHLIST_PATH,
        )

elif "Regime Analysis" in page:
    if use_demo:
        st.info("Regime analysis requires real data — switch off 'Use demo data'.")
    else:
        render_regime_analysis(
            tdf_path=TRADE_LOG_PATH,
            watchlist_path=WATCHLIST_PATH,
            indices_path=INDICES_PATH,
        )