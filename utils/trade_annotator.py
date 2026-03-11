from typing import Union, Optional
"""
trade_annotator.py
Renders an editable trades table with manual annotation checkbox columns.
Also adds a good_day_to_trade column auto-populated from master_df.

Manual annotation columns:
    - stop_5pct_would_have_won       : Would a 5% stop have kept this a winner?
    - stop_at_high_prevented_loss    : Would a stop order at the high have prevented the loss?
    - panic_sold_too_early           : Did you panic-sell before the trade played out?

Auto column:
    - good_day_to_trade              : True if setup_count >= 5 on buy_date, null if recent

Usage in journal.py:
    from trade_annotator import render_trade_annotator, merge_annotations
    render_trade_annotator(tdf, TRADE_LOG_PATH, MASTER_DF_PATH)

To preserve annotations when refreshing trades from source:
    from trade_annotator import merge_annotations
    fresh_df = merge_annotations(fresh_df, TRADE_LOG_PATH)
"""

import ast
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path


ANNOTATION_COLS = {
    "stop_5pct_would_have_won":    "5% Stop Would Have Won",
    "stop_at_high_prevented_loss": "Stop at High Prevented Loss",
    "panic_sold_too_early":        "Panic Sold Too Early",
}

GOOD_DAY_COL   = "good_day_to_trade"
TRADE_KEY_COLS = ["symbol", "buy_date"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_annotation_cols(df):
    for col in ANNOTATION_COLS:
        if col not in df.columns:
            df[col] = False
        else:
            df[col] = df[col].fillna(False).astype(bool)
    return df


def _load_master_df(master_df_path):
    p = Path(master_df_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _apply_good_day(df, master):
    """
    Add/update good_day_to_trade on the trades df from master_df setup_count.
    Null if buy_date is within current calendar month.
    Only fills currently-null cells (preserves manual overrides).
    """
    if master.empty or "date" not in master.columns or "setup_count" not in master.columns:
        if GOOD_DAY_COL not in df.columns:
            df[GOOD_DAY_COL] = None
        return df

    today  = date.today()
    cutoff = date(today.year, today.month, 1)

    master["_date"] = master["date"].dt.date
    setup_count_map = dict(zip(master["_date"], master["setup_count"]))

    if GOOD_DAY_COL not in df.columns:
        df[GOOD_DAY_COL] = None
    df[GOOD_DAY_COL] = df[GOOD_DAY_COL].astype(object)

    for idx, row in df.iterrows():
        if not pd.isnull(df.at[idx, GOOD_DAY_COL]):
            continue  # already set — preserve manual override
        buy_dt = pd.Timestamp(row["buy_date"]).date() if not pd.isnull(row["buy_date"]) else None
        if buy_dt is None:
            continue
        if buy_dt >= cutoff:
            df.at[idx, GOOD_DAY_COL] = None
        elif buy_dt in setup_count_map:
            df.at[idx, GOOD_DAY_COL] = bool(setup_count_map[buy_dt] >= 5)
        else:
            df.at[idx, GOOD_DAY_COL] = False

    return df


def _load_fresh(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "buy_date" in df.columns:
        df["buy_date"] = pd.to_datetime(df["buy_date"])
    df = _ensure_annotation_cols(df)
    return df


def _save(df, path):
    df_save = df.copy()
    if "buy_date" in df_save.columns:
        df_save["buy_date"] = pd.to_datetime(df_save["buy_date"]).dt.strftime("%Y-%m-%d")
    df_save.to_csv(path, index=False)


def merge_annotations(fresh_df, path):
    """
    Merges existing annotation + good_day_to_trade columns from disk onto fresh data.
    Call this inside your trades refresh function BEFORE saving.
    """
    p = Path(path)
    if not p.exists():
        return _ensure_annotation_cols(fresh_df)

    existing = pd.read_csv(p)
    existing = _ensure_annotation_cols(existing)

    if "buy_date" in existing.columns:
        existing["buy_date"] = pd.to_datetime(existing["buy_date"])
    if "buy_date" in fresh_df.columns:
        fresh_df["buy_date"] = pd.to_datetime(fresh_df["buy_date"])

    preserve_cols = list(ANNOTATION_COLS.keys())
    if GOOD_DAY_COL in existing.columns:
        preserve_cols.append(GOOD_DAY_COL)

    keep_cols = [c for c in TRADE_KEY_COLS if c in existing.columns] + preserve_cols
    existing_slim = existing[keep_cols].drop_duplicates(subset=TRADE_KEY_COLS)

    merged = fresh_df.merge(existing_slim, on=TRADE_KEY_COLS, how="left")
    merged = _ensure_annotation_cols(merged)
    return merged


# ── Main render ────────────────────────────────────────────────────────────────

def render_trade_annotator(tdf_ignored, trade_log_path, master_df_path=None):
    """
    Renders the full trade log with manual annotation checkboxes
    and auto-populated good_day_to_trade from master_df.
    """
    st.markdown("### Trade Log — Annotations")
    st.caption(
        "Manual checkboxes are saved automatically. "
        "good_day_to_trade is auto-filled from master_df (setup_count >= 5 on buy date)."
    )

    path = Path(trade_log_path)
    if not path.exists():
        st.warning("Trade log file not found: " + str(path))
        return

    df = _load_fresh(path)
    if df.empty:
        st.info("Trade log is empty.")
        return

    # Apply good_day_to_trade from master_df
    if master_df_path:
        master = _load_master_df(master_df_path)
        if not master.empty:
            df = _apply_good_day(df, master)
            _save(df, path)
        else:
            if GOOD_DAY_COL not in df.columns:
                df[GOOD_DAY_COL] = None

    # ── Reload button ──────────────────────────────────────────────────────────
    if st.button("🔄 Reload trades from CSV", key="reload_trades_btn"):
        df = _load_fresh(path)
        if master_df_path:
            master = _load_master_df(master_df_path)
            if not master.empty:
                df = _apply_good_day(df, master)
        st.toast("Trades reloaded!", icon="🔄")
        st.rerun()

    # ── Filters ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        if "buy_date" in df.columns:
            months = sorted(
                df["buy_date"].dt.to_period("M").astype(str).unique(), reverse=True
            )
            sel_months = st.multiselect(
                "Filter by month", options=months, default=[], placeholder="All months"
            )
        else:
            sel_months = []

    with col2:
        if "symbol" in df.columns:
            syms = sorted(df["symbol"].dropna().unique())
            sel_syms = st.multiselect(
                "Filter by symbol", options=syms, default=[], placeholder="All symbols"
            )
        else:
            sel_syms = []

    with col3:
        ann_filter = st.selectbox(
            "Annotation filter",
            ["All trades", "Unannotated only",
             "5% stop would have won",
             "Stop at high prevented loss",
             "Panic sold too early",
             "Good day to trade",
             "NOT a good day to trade"],
            index=0
        )

    view = df.copy()
    if sel_months:
        view = view[view["buy_date"].dt.to_period("M").astype(str).isin(sel_months)]
    if sel_syms:
        view = view[view["symbol"].isin(sel_syms)]

    ann_map = {
        "5% stop would have won":     "stop_5pct_would_have_won",
        "Stop at high prevented loss": "stop_at_high_prevented_loss",
        "Panic sold too early":        "panic_sold_too_early",
    }
    if ann_filter == "Unannotated only":
        ann_cols_list = list(ANNOTATION_COLS.keys())
        mask = ~(view[ann_cols_list[0]] | view[ann_cols_list[1]] | view[ann_cols_list[2]])
        view = view[mask]
    elif ann_filter in ann_map:
        view = view[view[ann_map[ann_filter]] == True]
    elif ann_filter == "Good day to trade":
        view = view[view[GOOD_DAY_COL] == True]
    elif ann_filter == "NOT a good day to trade":
        view = view[view[GOOD_DAY_COL] == False]

    st.caption("Showing " + str(len(view)) + " of " + str(len(df)) + " trades")

    # ── Editable table ─────────────────────────────────────────────────────────
    ann_col_config = {
        col: st.column_config.CheckboxColumn(label, default=False)
        for col, label in ANNOTATION_COLS.items()
    }
    ann_col_config[GOOD_DAY_COL] = st.column_config.CheckboxColumn(
        "Good Day to Trade",
        help="Auto-filled: True if setup_count >= 5 on buy date. Manually overrideable.",
        default=None,
    )

    editable_cols = list(ANNOTATION_COLS.keys()) + [GOOD_DAY_COL]
    priority_cols = [GOOD_DAY_COL] + list(ANNOTATION_COLS.keys())
    trade_cols    = [c for c in view.columns if c not in priority_cols]
    display_cols  = [c for c in priority_cols + trade_cols if c in view.columns]
    view_display  = view[display_cols].reset_index(drop=True)

    edited = st.data_editor(
        view_display,
        column_config=ann_col_config,
        use_container_width=True,
        hide_index=True,
        key="trade_editor_" + ann_filter + "_" + str(len(view)),
        disabled=[c for c in view_display.columns if c not in editable_cols],
    )

    # ── Auto-save on change ────────────────────────────────────────────────────
    if not edited.equals(view_display):
        orig_idx = view.index.tolist()
        for i, orig_i in enumerate(orig_idx):
            for col in list(ANNOTATION_COLS.keys()):
                df.at[orig_i, col] = bool(edited.at[i, col])
            if GOOD_DAY_COL in edited.columns:
                val = edited.at[i, GOOD_DAY_COL]
                df.at[orig_i, GOOD_DAY_COL] = None if pd.isnull(val) else bool(val)
        _save(df, path)
        st.toast("Saved!", icon="💾")

    # ── Summary stats ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Annotation Summary")

    ann_cols_list = list(ANNOTATION_COLS.keys())
    total      = len(df)
    annotated  = int(df[ann_cols_list].any(axis=1).sum())
    good_days  = int((df[GOOD_DAY_COL] == True).sum()) if GOOD_DAY_COL in df.columns else 0
    recent_ct  = int(df[GOOD_DAY_COL].isnull().sum()) if GOOD_DAY_COL in df.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total trades",      total)
    c2.metric("Annotated",         annotated)
    c3.metric("Good day to trade", good_days)
    c4.metric("Recent (null)",     recent_ct)

    st.markdown("##### Flag counts")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("5% stop would have won",      int(df["stop_5pct_would_have_won"].sum()))
    fc2.metric("Stop at high prevented loss", int(df["stop_at_high_prevented_loss"].sum()))
    fc3.metric("Panic sold too early",        int(df["panic_sold_too_early"].sum()))

    if "pct_return" in df.columns:
        st.markdown("##### P&L context for flagged trades")
        rows = []
        for col, label in ANNOTATION_COLS.items():
            flagged_df = df[df[col] == True]
            if flagged_df.empty:
                continue
            rows.append({
                "Annotation":   label,
                "Count":        len(flagged_df),
                "Avg Return %": round(flagged_df["pct_return"].mean(), 2),
                "Win Rate %":   round(100 * (flagged_df["pct_return"] > 0).mean(), 1),
                "Total P&L":    round(flagged_df["gain"].sum(), 2)
                                if "gain" in flagged_df.columns else "—",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Good day vs bad day P&L
        if GOOD_DAY_COL in df.columns:
            st.markdown("##### Returns: good days vs other days")
            judged = df[df[GOOD_DAY_COL].notnull()]
            good   = judged[judged[GOOD_DAY_COL] == True]
            other  = judged[judged[GOOD_DAY_COL] == False]
            if not good.empty or not other.empty:
                pnl_rows = []
                for label, subset in [("Good day (setup_count>=5)", good), ("Other days", other)]:
                    if subset.empty:
                        continue
                    pnl_rows.append({
                        "Group":        label,
                        "Count":        len(subset),
                        "Avg Return %": round(subset["pct_return"].mean(), 2),
                        "Win Rate %":   round(100 * (subset["pct_return"] > 0).mean(), 1),
                        "Total P&L":    round(subset["gain"].sum(), 2)
                                        if "gain" in subset.columns else "—",
                    })
                if pnl_rows:
                    st.dataframe(pd.DataFrame(pnl_rows), use_container_width=True, hide_index=True)
