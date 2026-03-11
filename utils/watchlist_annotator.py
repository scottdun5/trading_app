from typing import Union, Optional
"""
watchlist_annotator.py
Renders an editable watchlist table with annotation columns.
Auto-saves changes back to watchlist_curation.csv.

Columns managed here:
    - good_day_to_trade        : auto from master_df (setup_count >= 5), null if within current month
    - should_have_traded       : auto from master_df setup_list membership, null if within current month
    - should_have_traded_manual: 1 = user has manually overridden should_have_traded, 0 = auto value ok to overwrite

Usage in journal.py:
    from watchlist_annotator import render_watchlist_annotator
    render_watchlist_annotator(wdf, WATCHLIST_PATH, MASTER_DF_PATH)
"""

import ast
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path


SHOULD_COL    = "should_have_traded"
GOOD_DAY_COL  = "good_day_to_trade"
MANUAL_COL    = "should_have_traded_manual"   # tracks user overrides so auto never clobbers them


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_setup_list(val) -> list:
    """
    Parse setup_list cell robustly.
    Handles: "['ASTS', 'FLNC', 'AEVA']", ['ASTS','FLNC'], "ASTS,FLNC", etc.
    Always returns a list of clean uppercase ticker strings.
    """
    if val is None or (isinstance(val, float) and pd.isnull(val)):
        return []
    if isinstance(val, list):
        return [str(s).upper().strip() for s in val if str(s).strip()]

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "[]"):
        return []

    # Try ast.literal_eval first (handles proper Python list strings)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [str(x).upper().strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    # Strip surrounding brackets if present, then split on comma
    s = s.strip("[]")
    parts = [p.strip().strip("'\"").upper() for p in s.split(",")]
    return [p for p in parts if p]


def _load_master_df(master_df_path) -> pd.DataFrame:
    p = Path(master_df_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _load_fresh(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _save(df, path) -> None:
    df_save = df.copy()
    if "date" in df_save.columns:
        df_save["date"] = df_save["date"].dt.strftime("%Y-%m-%d")
    df_save.to_csv(path, index=False)


def _apply_master_defaults(df: pd.DataFrame, master: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """
    Populate GOOD_DAY_COL and SHOULD_COL from master_df.

    Rules:
    - Dates in current calendar month → always None (too recent to judge)
    - GOOD_DAY_COL: always overwritten from master (no manual override concept)
    - SHOULD_COL: only overwritten if MANUAL_COL != 1 (user hasn't manually set it)
    """
    # Ensure required columns exist
    if GOOD_DAY_COL not in df.columns:
        df[GOOD_DAY_COL] = None
    if SHOULD_COL not in df.columns:
        df[SHOULD_COL] = None
    if MANUAL_COL not in df.columns:
        df[MANUAL_COL] = 0

    df[GOOD_DAY_COL] = df[GOOD_DAY_COL].astype(object)
    df[SHOULD_COL]   = df[SHOULD_COL].astype(object)
    df[MANUAL_COL]   = pd.to_numeric(df[MANUAL_COL], errors="coerce").fillna(0).astype(int)

    if master.empty or "date" not in master.columns:
        return df

    today  = date.today()
    cutoff = date(today.year, today.month, 1)

    # Build lookup tables keyed by date (as date objects)
    master = master.copy()
    master["_date"] = master["date"].dt.date

    setup_count_map = {}
    setup_list_map  = {}

    if "setup_count" in master.columns:
        setup_count_map = dict(zip(master["_date"], master["setup_count"]))

    if "setup_list" in master.columns:
        for _, row in master.iterrows():
            parsed = _parse_setup_list(row["setup_list"])
            setup_list_map[row["_date"]] = set(parsed)  # use set for O(1) lookup

    # Detect stock column
    stock_col = "stock" if "stock" in df.columns else (
        "symbol" if "symbol" in df.columns else None
    )

    for idx, row in df.iterrows():
        try:
            row_date = pd.Timestamp(row["date"]).date()
        except Exception:
            continue

        recent = row_date >= cutoff

        # ── good_day_to_trade: always auto (no manual override) ────────────────
        if recent:
            df.at[idx, GOOD_DAY_COL] = None
        elif row_date in setup_count_map:
            df.at[idx, GOOD_DAY_COL] = bool(int(setup_count_map[row_date]) >= 5)
        else:
            df.at[idx, GOOD_DAY_COL] = False

        # ── should_have_traded: skip if manually overridden ────────────────────
        if not force and int(df.at[idx, MANUAL_COL]) == 1:
            continue  # user has set this manually — never touch it

        if recent:
            df.at[idx, SHOULD_COL] = None
        elif stock_col is not None and row_date in setup_list_map:
            sym = str(row[stock_col]).upper().strip()
            df.at[idx, SHOULD_COL] = sym in setup_list_map[row_date]
        elif row_date in setup_count_map:
            # Date exists in master but no setup_list — default False
            df.at[idx, SHOULD_COL] = False
        else:
            # Date not in master at all
            df.at[idx, SHOULD_COL] = None

    return df


# ── Main render ────────────────────────────────────────────────────────────────

def render_watchlist_annotator(wdf_ignored, watchlist_path, master_df_path=None):
    st.markdown("### Watchlist — Annotations")
    st.caption(
        "**good_day_to_trade**: auto from master_df (setup_count >= 5).  "
        "**should_have_traded**: auto from master_df setup_list — check/uncheck to manually override. "
        "Dates in the current month are blank (too recent to judge)."
    )

    path = Path(watchlist_path)
    if not path.exists():
        st.warning("Watchlist file not found: " + str(path))
        return

    df = _load_fresh(path)
    if df.empty:
        st.info("Watchlist is empty.")
        return

    # Load master and apply
    if master_df_path:
        master = _load_master_df(master_df_path)
        if not master.empty:
            btn_col, info_col = st.columns([1, 3])
            with btn_col:
                force_rerun = st.button(
                    "Re-run auto-population",
                    key="watchlist_force_rerun",
                    help="Clears all non-manually-overridden auto values and recomputes from master_df"
                )
            with info_col:
                st.caption("Use this if the good_day_to_trade or should_have_traded columns look wrong.")
            df = _apply_master_defaults(df, master, force=force_rerun)
            _save(df, path)
            if force_rerun:
                st.toast("Auto-population complete!", icon="✅")
                st.rerun()
        else:
            st.caption("master_df.csv not found — manual entry only.")
            for col in [GOOD_DAY_COL, SHOULD_COL]:
                if col not in df.columns:
                    df[col] = None
            if MANUAL_COL not in df.columns:
                df[MANUAL_COL] = 0
    else:
        for col in [GOOD_DAY_COL, SHOULD_COL]:
            if col not in df.columns:
                df[col] = None
        if MANUAL_COL not in df.columns:
            df[MANUAL_COL] = 0

    stock_col = "stock" if "stock" in df.columns else "symbol"

    # ── Filters ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        dates = sorted(df["date"].dt.date.unique(), reverse=True)
        selected_dates = st.multiselect(
            "Filter by date", options=dates, default=[], placeholder="All dates"
        )
    with col2:
        if stock_col in df.columns:
            syms = sorted(df[stock_col].dropna().unique())
            selected_syms = st.multiselect(
                "Filter by symbol", options=syms, default=[], placeholder="All symbols"
            )
        else:
            selected_syms = []
    with col3:
        flag_filter = st.selectbox(
            "Show",
            ["All", "Should have traded", "Should NOT have traded",
             "Good day to trade", "Recent (null)", "Manually overridden"],
            index=0
        )

    view = df.copy()
    if selected_dates:
        view = view[view["date"].dt.date.isin(selected_dates)]
    if selected_syms:
        view = view[view[stock_col].isin(selected_syms)]

    if flag_filter == "Should have traded":
        view = view[view[SHOULD_COL] == True]
    elif flag_filter == "Should NOT have traded":
        view = view[view[SHOULD_COL] == False]
    elif flag_filter == "Good day to trade":
        view = view[view[GOOD_DAY_COL] == True]
    elif flag_filter == "Recent (null)":
        view = view[view[SHOULD_COL].isnull()]
    elif flag_filter == "Manually overridden":
        view = view[view[MANUAL_COL] == 1]

    st.caption("Showing " + str(len(view)) + " of " + str(len(df)) + " rows")

    # ── Editable table ─────────────────────────────────────────────────────────
    col_config = {
        SHOULD_COL: st.column_config.CheckboxColumn(
            "Should Have Traded",
            help="Auto from master_df. Check/uncheck to manually override.",
            default=None,
        ),
        GOOD_DAY_COL: st.column_config.CheckboxColumn(
            "Good Day to Trade",
            help="Auto: True if setup_count >= 5. Not manually editable.",
            default=None,
        ),
    }

    ann_cols     = [GOOD_DAY_COL, SHOULD_COL]
    hidden_cols  = [MANUAL_COL]  # tracked internally, hidden from display
    other_cols   = [c for c in view.columns if c not in ann_cols + hidden_cols]
    display_cols = [c for c in ann_cols + other_cols if c in view.columns]
    view_display = view[display_cols].reset_index(drop=True)

    # good_day_to_trade is read-only (auto only)
    disabled_cols = [c for c in view_display.columns if c not in [SHOULD_COL]]

    edited = st.data_editor(
        view_display,
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        key="watchlist_editor_" + flag_filter + "_" + str(len(view)),
        disabled=disabled_cols,
    )

    # ── Auto-save on change — mark manual override ─────────────────────────────
    if not edited.equals(view_display):
        orig_idx = view.index.tolist()
        for i, orig_i in enumerate(orig_idx):
            old_val = view_display.at[i, SHOULD_COL]
            new_val = edited.at[i, SHOULD_COL]

            # Detect if user actually changed this cell
            old_bool = None if pd.isnull(old_val) else bool(old_val)
            new_bool = None if pd.isnull(new_val) else bool(new_val)

            if old_bool != new_bool:
                df.at[orig_i, SHOULD_COL] = new_bool
                df.at[orig_i, MANUAL_COL] = 1  # mark as manually overridden

        _save(df, path)
        st.toast("Saved!", icon="💾")

    # ── Summary stats ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Annotation Summary")

    judged      = df[df[SHOULD_COL].notnull()]
    total       = len(df)
    recent_ct   = int(df[SHOULD_COL].isnull().sum())
    flagged     = int((judged[SHOULD_COL] == True).sum())
    not_flagged = int((judged[SHOULD_COL] == False).sum())
    good_days   = int((df[GOOD_DAY_COL] == True).sum()) if GOOD_DAY_COL in df.columns else 0
    manual_ct   = int((df[MANUAL_COL] == 1).sum()) if MANUAL_COL in df.columns else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total rows",         total)
    c2.metric("Should have traded", flagged)
    c3.metric("Should NOT have",    not_flagged)
    c4.metric("Good days",          good_days)
    c5.metric("Recent (null)",      recent_ct)
    c6.metric("Manual overrides",   manual_ct)

    if "traded" in df.columns:
        st.markdown("#### Selection Quality")
        judged2 = df[df[SHOULD_COL].notnull()].copy()
        judged2["_traded_bool"] = judged2["traded"].fillna(0).astype(bool)
        judged2["_should_bool"] = judged2[SHOULD_COL].astype(bool)

        took_and_should = int((judged2["_traded_bool"] &  judged2["_should_bool"]).sum())
        took_not_should = int((judged2["_traded_bool"] & ~judged2["_should_bool"]).sum())
        should_not_took = int((~judged2["_traded_bool"] &  judged2["_should_bool"]).sum())
        neither         = int((~judged2["_traded_bool"] & ~judged2["_should_bool"]).sum())

        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Took & Should Have",   took_and_should)
        sc2.metric("Took, Shouldn't Have", took_not_should)
        sc3.metric("Missed Opportunities", should_not_took)
        sc4.metric("Neither",              neither)