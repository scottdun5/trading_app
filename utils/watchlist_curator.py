"""
watchlist_curator.py
Daily curation UI — simplified binary watchlist, corrected funnel.
Three tabs: Daily Entry | Memo Calendar | Ticker Corrections
"""

import streamlit as st
import pandas as pd
import numpy as np
import calendar
from datetime import date, timedelta

import sys
from pathlib import Path

from data_manager import (
    load_watchlist, save_watchlist_entry, delete_watchlist_entry,
    load_stock_thoughts, load_corrections, add_correction,
    apply_corrections_to_csv, save_corrections,
    load_market_thoughts, load_daily_protocol,
    build_master_journal,
    TIGERS, STOCK_THOUGHTS_PATH, WATCHLIST_PATH, BASE_DIR,
    CORRECTIONS_PATH, PROTOCOL_PATH,
)

SETUP_TYPES  = ["VCP","Flat base","Cup with handle","High tight flag","IPO base",
                "3-weeks-tight","Bull flag","Ascending base","Double bottom","Other"]
CONVICTION   = ["high","med","low"]


def render_curator(mdf: pd.DataFrame, sdf: pd.DataFrame,
                   wdf: pd.DataFrame, corrections: dict,
                   protocol: pd.DataFrame = None):
    st.markdown("""
    <div style="border-bottom:1px solid #252830;padding-bottom:12px;margin-bottom:20px;">
      <span style="font-family:'Space Mono',monospace;font-size:20px;
                   color:#e8eaf0;font-weight:700;">📋 Daily Curation</span>
      <span style="font-family:'Space Mono',monospace;font-size:10px;color:#6b7280;
                   letter-spacing:.12em;margin-left:12px;">WATCHLIST · CALENDAR · CORRECTIONS</span>
    </div>
    """, unsafe_allow_html=True)

    t1, t2, t3 = st.tabs([
        "🔄  Data Sync",
        "📅  Memo Calendar",
        "🔧  Ticker Corrections",
    ])
    with t1: _render_data_sync(sdf, wdf)
    with t2: _render_memo_calendar(mdf, wdf, protocol)
    with t3: _render_ticker_corrections(sdf, corrections)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA SYNC
# ─────────────────────────────────────────────────────────────────────────────

def _render_data_sync(sdf: pd.DataFrame, wdf: pd.DataFrame):
    """Button-driven sync: ingest watchlists folder, rebuild master journal."""

    st.markdown("### Data Sync")
    st.caption(
        "Run these after dropping new Focus CSVs into your watchlists folder "
        "or recording new voice memos. The system tracks everything else automatically."
    )

    WATCHLIST_DIR = BASE_DIR / "watchlists"

    # ── Step 1: Ingest watchlists ─────────────────────────────────────────────
    st.markdown("---")
    col_hdr, col_path = st.columns([2, 3])
    with col_hdr:
        st.markdown("#### 1 · Ingest Watchlists")
    with col_path:
        st.caption(f"Reading from: `{WATCHLIST_DIR}`")

    # Show what's in the folder
    focus_files = []
    if WATCHLIST_DIR.exists():
        focus_files = sorted([
            f for f in WATCHLIST_DIR.iterdir()
            if f.suffix.lower() == ".csv" and "focus" in f.name.lower()
        ])

    c_files, c_btn = st.columns([3, 1])
    with c_files:
        if focus_files:
            badges = " ".join([
                '<span style="font-family:Space Mono,monospace;font-size:11px;'
                'background:rgba(124,92,252,0.10);color:#7c5cfc;'
                'border:1px solid #7c5cfc33;border-radius:5px;'
                'padding:2px 8px;margin:2px;display:inline-block;">'
                + f.stem + '</span>'
                for f in focus_files
            ])
            st.markdown(
                '<div style="font-size:12px;color:#6b7280;margin-bottom:6px;">'
                + str(len(focus_files)) + ' Focus file(s) found in watchlists/</div>'
                + badges,
                unsafe_allow_html=True
            )
        elif not WATCHLIST_DIR.exists():
            st.warning(f"Watchlists folder not found: `{WATCHLIST_DIR}`")
        else:
            st.info("No Focus CSV files found. Drop files named like `2_26 Focus.csv` into the watchlists folder.")

    with c_btn:
        ingest_btn = st.button(
            "▶ Run Ingest",
            type="primary",
            use_container_width=True,
            disabled=not focus_files,
            key="ingest_watchlist_btn"
        )

    if ingest_btn:
        try:
            import importlib.util, sys as _sys
            script_path = Path(__file__).parent.parent / "build_watchlist.py"
            if not script_path.exists():
                script_path = Path(__file__).parent / "build_watchlist.py"

            spec = importlib.util.spec_from_file_location("build_watchlist", script_path)
            bw   = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(bw)

            with st.spinner("Ingesting watchlists..."):
                result_df = bw.build_watchlist(
                    watchlist_dir=WATCHLIST_DIR,
                    output_path=WATCHLIST_PATH,
                )

            if result_df is not None and not result_df.empty:
                n_dates  = result_df["date"].nunique()
                n_stocks = len(result_df)
                st.success(
                    f"✅ Ingested {n_stocks} watchlist entries across {n_dates} trading days "
                    f"→ saved to `watchlist_curation.csv`"
                )
                st.rerun()
            else:
                st.error("Ingest returned no data. Check that your Focus files have a 'Symbol' column.")

        except Exception as e:
            st.error(f"Ingest failed: {e}")

    # ── Step 2: Rebuild master journal ───────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 2 · Rebuild Master Journal")
    st.caption(
        "Joins watchlist, voice memos, and trades into `master_journal.csv`. "
        "Run after ingesting watchlists or after new memos have been processed."
    )

    rb_col, rb_info = st.columns([1, 3])
    with rb_col:
        rebuild_btn = st.button(
            "▶ Rebuild Journal",
            use_container_width=True,
            key="rebuild_journal_btn"
        )
    with rb_info:
        from data_manager import MASTER_PATH
        if MASTER_PATH.exists():
            import os
            mtime = pd.Timestamp(os.path.getmtime(MASTER_PATH), unit="s")
            st.caption(f"Last built: {mtime.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.caption("master_journal.csv not yet built.")

    if rebuild_btn:
        try:
            with st.spinner("Building master journal..."):
                mj = build_master_journal()
            st.success(f"✅ master_journal.csv rebuilt — {len(mj)} rows")
            st.rerun()
        except Exception as e:
            st.error(f"Rebuild failed: {e}")

    # ── Coverage summary ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Coverage Summary")

    if wdf.empty or sdf.empty:
        st.info("Need both watchlist data and voice memo data to show coverage.")
        return

    wdf2 = wdf.copy()
    sdf2 = sdf.copy()
    wdf2["_date"] = pd.to_datetime(wdf2["date"]).dt.date
    sdf2["_date"] = pd.to_datetime(sdf2["date"]).dt.date

    memo_pairs  = set(zip(sdf2["_date"], sdf2["stock"].str.upper()))
    wdf2["in_memo"] = wdf2.apply(
        lambda r: (r["_date"], str(r["stock"]).upper()) in memo_pairs, axis=1
    )

    total_wl     = len(wdf2)
    covered      = wdf2["in_memo"].sum()
    uncovered    = total_wl - covered
    cov_rate     = covered / total_wl if total_wl else 0

    wl_pairs     = set(zip(wdf2["_date"], wdf2["stock"].str.upper()))
    sdf2["organic"] = sdf2.apply(
        lambda r: (r["_date"], str(r["stock"]).upper()) not in wl_pairs, axis=1
    )
    organic_count = sdf2["organic"].sum()

    m1, m2, m3, m4 = st.columns(4)

    # Python 3.8-safe: no backslashes inside f-string expressions
    def _metric_card(col, label, value, sub="", color="#e8eaf0"):
        sub_html = (
            '<div style="font-size:10px;color:#9ca3af;margin-top:2px;">' + str(sub) + '</div>'
            if sub else ""
        )
        col.markdown(
            f'<div style="background:#1a1d25;border:1px solid #252830;border-radius:8px;'
            f'padding:14px 16px;">'
            f'<div style="font-family:Space Mono,monospace;font-size:22px;'
            f'font-weight:700;color:{color};">{value}</div>'
            f'<div style="font-family:Space Mono,monospace;font-size:10px;'
            f'color:#6b7280;margin-top:4px;letter-spacing:.08em;">{label}</div>'
            + sub_html +
            '</div>',
            unsafe_allow_html=True
        )

    _metric_card(m1, "WATCHLIST ENTRIES", total_wl, str(wdf2["_date"].nunique()) + " days")
    _metric_card(m2, "MEMO COVERAGE", str(round(cov_rate * 100)) + "%",
                 str(covered) + "/" + str(total_wl) + " stocks covered",
                 color="#4ade80" if cov_rate >= 0.8 else "#fbbf24" if cov_rate >= 0.5 else "#ff6b6b")
    _metric_card(m3, "UNCOVERED", uncovered,
                 "watchlist stocks not in memo",
                 color="#ff6b6b" if uncovered > 0 else "#4ade80")
    _metric_card(m4, "ORGANIC IDEAS", organic_count,
                 "memo stocks not on watchlist",
                 color="#00d4ff")

    uncovered_df = wdf2[~wdf2["in_memo"]].copy()
    if not uncovered_df.empty:
        st.markdown("##### Watchlist Stocks Not Covered in Memo")
        uncovered_df["date_str"] = uncovered_df["_date"].astype(str)
        display = (uncovered_df[["date_str","stock","setup_type","conviction"]]
                   .rename(columns={"date_str":"date"})
                   .sort_values(["date","stock"], ascending=[False, True]))
        st.dataframe(display, use_container_width=True, hide_index=True)

    organic_df = sdf2[sdf2["organic"]].copy()
    if not organic_df.empty:
        with st.expander(f"🔍 Organic memo ideas ({organic_count} stocks — not pre-watchlisted)"):
            organic_df["date_str"] = organic_df["_date"].astype(str)
            display_org = (organic_df[["date_str","stock","summarized_thoughts"]]
                           .rename(columns={"date_str":"date"})
                           .sort_values(["date","stock"], ascending=[False, True]))
            st.dataframe(display_org, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MEMO CALENDAR
# ─────────────────────────────────────────────────────────────────────────────

def _render_memo_calendar(mdf: pd.DataFrame, wdf: pd.DataFrame,
                           protocol: pd.DataFrame = None):
    st.markdown("### Voice Memo + Protocol Calendar")
    st.caption("Tracks recording consistency and daily protocol scores at a glance.")

    today = date.today()

    nav1, nav2, nav3 = st.columns([1, 2, 1])
    with nav1:
        if st.button("← Prev"):
            m = st.session_state.get("cal_month", today.replace(day=1))
            st.session_state["cal_month"] = (m - timedelta(days=1)).replace(day=1)
    with nav3:
        if st.button("Next →"):
            m = st.session_state.get("cal_month", today.replace(day=1))
            st.session_state["cal_month"] = (m.replace(day=28) + timedelta(days=4)).replace(day=1)
    cal_month = st.session_state.get("cal_month", today.replace(day=1))
    with nav2:
        st.markdown(
            f'<div style="text-align:center;font-family:Space Mono,monospace;'
            f'font-size:15px;color:#e8eaf0;padding:5px 0">{cal_month.strftime("%B %Y")}</div>',
            unsafe_allow_html=True
        )

    memo_dates = set()
    wl_dates   = set()
    protocol_scores = {}
    off_wl_days = set()

    if not mdf.empty:
        memo_dates = set(
            mdf[(mdf["date"].dt.year == cal_month.year) &
                (mdf["date"].dt.month == cal_month.month)]["date"].dt.date.tolist()
        )
    if not wdf.empty:
        wl_dates = set(
            wdf[(wdf["date"].dt.year == cal_month.year) &
                (wdf["date"].dt.month == cal_month.month)]["date"].dt.date.tolist()
        )
    if protocol is not None and not protocol.empty:
        month_prot = protocol[
            (protocol["date"].dt.year == cal_month.year) &
            (protocol["date"].dt.month == cal_month.month)
        ]
        protocol_scores = dict(zip(month_prot["date"].dt.date, month_prot["protocol_score"]))
        if "off_watchlist_count" in month_prot.columns:
            off_wl_days = set(
                month_prot[month_prot["off_watchlist_count"] > 0]["date"].dt.date.tolist()
            )

    st.markdown(
        _build_calendar_html(cal_month, memo_dates, wl_dates, protocol_scores, off_wl_days, today),
        unsafe_allow_html=True
    )

    _, days_in_month = calendar.monthrange(cal_month.year, cal_month.month)
    trading_days = sum(
        1 for d in range(1, days_in_month + 1)
        if date(cal_month.year, cal_month.month, d).weekday() < 5
        and date(cal_month.year, cal_month.month, d) <= today
    )
    recorded = len(memo_dates)
    avg_score = (
        sum(protocol_scores.values()) / len(protocol_scores)
        if protocol_scores else None
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Trading Days (so far)", trading_days)
    with m2: st.metric("Memos Recorded", recorded)
    with m3:
        pct = round(recorded / trading_days * 100) if trading_days else 0
        st.metric("Consistency", f"{pct}%")
    with m4:
        st.metric("Avg Protocol Score",
                  f"{avg_score:.1f}/3" if avg_score is not None else "—")


def _build_calendar_html(cal_month, memo_dates, wl_dates,
                          protocol_scores, off_wl_days, today) -> str:
    _, days_in_month = calendar.monthrange(cal_month.year, cal_month.month)
    first_dow = date(cal_month.year, cal_month.month, 1).weekday()

    html = """
    <style>
    .cal-wrap  { margin:8px 0 16px; }
    .cal-grid  { display:grid;grid-template-columns:repeat(7,1fr);gap:5px; }
    .cal-hdr   { text-align:center;font-family:'Space Mono',monospace;font-size:10px;
                 color:#6b7280;padding:5px 0;letter-spacing:.05em; }
    .cal-cell  { border-radius:8px;padding:7px 4px;text-align:center;
                 font-family:'Space Mono',monospace;font-size:12px;
                 min-height:52px;display:flex;flex-direction:column;
                 align-items:center;justify-content:center;gap:3px; }
    .c-empty   { background:transparent; }
    .c-wknd    { background:#0d0f14;color:#374151; }
    .c-future  { background:#111318;color:#374151;border:1px solid #1a1d25; }
    .c-missing { background:#1a1d25;color:#6b7280;border:1px solid #252830; }
    .c-memo    { background:rgba(74,222,128,0.1);color:#4ade80;border:1px solid rgba(74,222,128,0.25); }
    .c-today   { box-shadow:0 0 0 2px #00d4ff !important; }
    .score-pip { display:flex;gap:2px;justify-content:center; }
    .pip       { width:5px;height:5px;border-radius:50%;background:#374151; }
    .pip-on    { background:#00d4ff; }
    .off-badge { font-size:8px;color:#ff6b6b;font-family:'Space Mono',monospace; }
    .legend    { display:flex;gap:14px;margin-top:10px;
                 font-family:'Space Mono',monospace;font-size:10px;color:#6b7280; }
    </style>
    <div class="cal-wrap">
    <div class="cal-grid">
    """
    for h in ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]:
        html += '<div class="cal-hdr">' + h + '</div>'

    for _ in range(first_dow):
        html += '<div class="cal-cell c-empty"></div>'

    for day_num in range(1, days_in_month + 1):
        d         = date(cal_month.year, cal_month.month, day_num)
        is_wknd   = d.weekday() >= 5
        is_today  = d == today
        is_future = d > today
        has_memo  = d in memo_dates
        score     = protocol_scores.get(d)
        has_off   = d in off_wl_days

        cls = "cal-cell "
        if   is_wknd:   cls += "c-wknd"
        elif is_future: cls += "c-future"
        elif has_memo:  cls += "c-memo"
        else:           cls += "c-missing"
        if is_today:    cls += " c-today"

        pips = ""
        if has_memo and score is not None:
            pip_divs = "".join(
                '<div class="pip' + ("" if i >= score else " pip-on") + '"></div>'
                for i in range(3)
            )
            pips = '<div class="score-pip">' + pip_divs + '</div>'

        off_badge = '<div class="off-badge">⚠OWL</div>' if has_off else ""

        html += (
            '<div class="' + cls + '">'
            '<span>' + str(day_num) + '</span>'
            + pips + off_badge +
            '</div>'
        )

    html += """
    </div>
    <div class="legend">
      <span><span style="color:#4ade80">■</span> Memo recorded</span>
      <span><span style="color:#374151">■</span> Missing</span>
      <span><span style="color:#00d4ff">●●●</span> Protocol score (0–3)</span>
      <span><span style="color:#ff6b6b">⚠OWL</span> Off-watchlist trade</span>
    </div>
    </div>
    """
    return html


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TICKER CORRECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _render_ticker_corrections(sdf: pd.DataFrame, corrections: dict):
    st.markdown("### Ticker Correction Editor")
    st.caption(
        "Fixes are saved to `ticker_corrections.json`, applied to `stock_thoughts.csv` "
        "immediately, and injected into all future prompt runs — so every correction "
        "permanently improves transcription accuracy."
    )

    if not sdf.empty and "ticker_confidence" in sdf.columns:
        low_conf = (sdf[sdf["ticker_confidence"]=="low"]
                    [["date","stock","raw_thoughts"]]
                    .drop_duplicates("stock"))
        if not low_conf.empty:
            st.markdown("#### ⚠️ Low-Confidence Tickers (model flagged these)")
            for _, row in low_conf.head(10).iterrows():
                lc1, lc2, lc3 = st.columns([1, 1, 3])
                with lc1:
                    st.markdown(
                        '<span style="font-family:Space Mono,monospace;'
                        'color:#f59e0b;font-size:13px;">' + str(row["stock"]) + '</span>',
                        unsafe_allow_html=True
                    )
                with lc2:
                    fix = st.text_input("Correct to", key=f"lc_{row['stock']}",
                                         placeholder="RKLB").upper().strip()
                with lc3:
                    excerpt = str(row.get("raw_thoughts",""))
                    st.caption(excerpt[:100] + ("..." if len(excerpt)>100 else ""))
                if fix and fix != row["stock"]:
                    if st.button(f"Apply {row['stock']} → {fix}",
                                  key=f"btn_{row['stock']}"):
                        add_correction(row["stock"], fix, CORRECTIONS_PATH)
                        if STOCK_THOUGHTS_PATH.exists():
                            apply_corrections_to_csv(STOCK_THOUGHTS_PATH, {row["stock"]: fix})
                        st.success(f"✅ {row['stock']} → {fix} saved and applied")
                        st.rerun()
            st.markdown("---")

    st.markdown("#### Add Manual Correction")
    m1, m2, m3 = st.columns([1, 1, 1])
    with m1: wrong   = st.text_input("Wrong (as in CSV)").upper().strip()
    with m2: correct = st.text_input("Correct ticker").upper().strip()
    with m3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💾 Save & Apply", type="primary"):
            if wrong and correct:
                add_correction(wrong, correct, CORRECTIONS_PATH)
                if STOCK_THOUGHTS_PATH.exists():
                    apply_corrections_to_csv(STOCK_THOUGHTS_PATH, {wrong: correct})
                st.success(f"✅ {wrong} → {correct}")
                st.rerun()
            else:
                st.error("Fill in both fields.")

    current = load_corrections(CORRECTIONS_PATH)
    if current:
        st.markdown("#### Current Dictionary")
        cdf = pd.DataFrame([{"wrong": k,"correct": v} for k,v in current.items()])
        st.dataframe(cdf, use_container_width=True, hide_index=True)
        del_key = st.selectbox("Remove", [""] + list(current.keys()))
        if del_key and st.button("🗑 Remove"):
            del current[del_key]
            save_corrections(current, CORRECTIONS_PATH)
            st.success(f"Removed {del_key}")
            st.rerun()

    with st.expander("📋 Prompt injection block (copy into PROMPT_STOCKS)"):
        if current:
            block = "  CUSTOM TICKER CORRECTIONS — apply these first:\n"
            for w, c in current.items():
                block += f'    "{w}" → "{c}"\n'
            st.code(block)
        else:
            st.caption("No corrections yet.")