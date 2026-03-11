"""
kpi_page.py  —  pages/kpi_page.py
Trading KPI Dashboard: Consistency · Accuracy · Trading · Psychology
Aggregated monthly from November 2025 onward.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).parent.parent / "data"
WATCHLIST_PATH     = BASE_DIR / "watchlist_curation.csv"
MARKET_PATH        = BASE_DIR / "market_thoughts.csv"
STOCK_PATH         = BASE_DIR / "stock_thoughts.csv"
TRADES_PATH        = BASE_DIR / "raw_trades.csv"
SECTOR_THEMES_PATH = BASE_DIR / "sector_themes.csv"
EXCLUDED_DAYS_PATH = BASE_DIR / "kpi_excluded_days.json"
MASTER_DF_PATH     = BASE_DIR / "master_df.csv"

START_MONTH = "2025-11"   # first full memo month

# ── OpenAI client ──────────────────────────────────────────────────────────────
def _get_client():
    try:
        from utils.config import OPENAI_API_KEY
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent.parent / "utils"))
        from config import OPENAI_API_KEY
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_llm(prompt: str, max_tokens: int = 500) -> str:
    try:
        client = _get_client()
        r = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""
    except Exception as e:
        return "ERROR: " + str(e)


# ── Excluded days persistence ──────────────────────────────────────────────────
def _load_excluded() -> List[str]:
    if EXCLUDED_DAYS_PATH.exists():
        try:
            return json.loads(EXCLUDED_DAYS_PATH.read_text())
        except Exception:
            return []
    return []


def _save_excluded(days: List[str]) -> None:
    EXCLUDED_DAYS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXCLUDED_DAYS_PATH.write_text(json.dumps(sorted(days)))


# ── Data loaders ───────────────────────────────────────────────────────────────
def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        for col in ["date", "buy_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _is_true(val) -> bool:
    if val is None:
        return False
    try:
        if isinstance(val, float) and np.isnan(val):
            return False
    except Exception:
        pass
    return str(val).strip().lower() in ("true", "1", "yes")


# ── Setup list parser (mirrors watchlist_annotator logic) ─────────────────────
def _parse_setup_list(val) -> set:
    """Robustly parse setup_list values like ['ASTS', 'FLNC'] into a set of symbols."""
    if val is None:
        return set()
    try:
        if isinstance(val, float) and __import__("math").isnan(val):
            return set()
    except Exception:
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "[]", ""):
        return set()
    s = s.strip("[]")
    parts = []
    for p in s.split(","):
        clean = p.strip()
        # strip surrounding quotes of any kind
        if len(clean) >= 2 and clean[0] in (chr(39), chr(34)) and clean[-1] in (chr(39), chr(34)):
            clean = clean[1:-1]
        clean = clean.upper().strip()
        if clean:
            parts.append(clean)
    return set(parts)


# ── Sector extraction ──────────────────────────────────────────────────────────
def _extract_sectors_prompt(date_str: str, transcript: str) -> str:
    return (
        "You are analyzing a swing trader's daily market voice memo from " + date_str + ".\n"
        "Extract every sector, industry, or sub-industry mentioned (either bullish or bearish).\n"
        "Return ONLY a JSON object on one line: "
        "{\"bullish\": [\"sector1\", \"sector2\"], \"bearish\": [\"sector1\"]}\n"
        "Use standard names like: Technology, Semiconductors, Biotech, Energy, Defense, "
        "Gold/Miners, Financials, Industrials, Consumer Discretionary, Retail, Healthcare, "
        "Aerospace, Real Estate, Utilities, Materials, Homebuilders, etc.\n"
        "If none are mentioned, return {\"bullish\": [], \"bearish\": []}.\n\n"
        "TRANSCRIPT:\n" + transcript[:1500]
    )


def run_sector_extraction(mdf: pd.DataFrame) -> pd.DataFrame:
    """Run sector extraction for all dates not yet in sector_themes.csv."""
    if mdf.empty or "raw_transcript" not in mdf.columns:
        return pd.DataFrame()

    existing = pd.DataFrame()
    if SECTOR_THEMES_PATH.exists():
        try:
            existing = pd.read_csv(SECTOR_THEMES_PATH)
            existing["date"] = pd.to_datetime(existing["date"], errors="coerce")
        except Exception:
            existing = pd.DataFrame()

    done_dates = set()
    if not existing.empty and "date" in existing.columns:
        done_dates = set(existing["date"].dt.date.tolist())

    mdf2 = mdf.copy()
    mdf2["_date"] = mdf2["date"].dt.date
    todo = mdf2[~mdf2["_date"].isin(done_dates)].copy()

    if todo.empty:
        return existing if not existing.empty else pd.DataFrame()

    rows = []
    bar = st.progress(0, text="Extracting sector themes...")
    total = len(todo)
    for i, (_, row) in enumerate(todo.iterrows()):
        bar.progress((i + 1) / total, text="Processing " + str(row["_date"]))
        transcript = str(row.get("raw_transcript", ""))
        if not transcript.strip():
            continue
        resp = _call_llm(_extract_sectors_prompt(str(row["_date"]), transcript))
        try:
            clean = resp.strip().replace("```json", "").replace("```", "").strip()
            data  = json.loads(clean)
            bullish = ",".join(data.get("bullish", []))
            bearish = ",".join(data.get("bearish", []))
        except Exception:
            bullish = ""
            bearish = ""
        rows.append({
            "date":    row["_date"],
            "bullish": bullish,
            "bearish": bearish,
        })
    bar.empty()

    new_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    if not new_df.empty:
        new_df["date"] = pd.to_datetime(new_df["date"])
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        combined.to_csv(SECTOR_THEMES_PATH, index=False)
        return combined
    return existing


# ── Trading day universe ───────────────────────────────────────────────────────
def _get_nyse_holidays(start: date, end: date) -> set:
    """Return NYSE holiday dates using exchange_calendars (always up to date)."""
    try:
        import exchange_calendars as xcals
        nyse = xcals.get_calendar("XNYS")
        sessions = nyse.sessions_in_range(
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        valid = set(pd.DatetimeIndex(sessions).date)
        all_wkdays = set()
        d = start
        while d <= end:
            if d.weekday() < 5:
                all_wkdays.add(d)
            d += timedelta(days=1)
        return all_wkdays - valid
    except Exception:
        # Fallback: weekdays only (no holiday removal)
        return set()


_NYSE_HOLIDAY_CACHE = {}  # (start, end) -> set


def _trading_days_between(start: date, end: date) -> List[date]:
    """Return NYSE trading days (no weekends, no federal/market holidays)."""
    cache_key = (start, end)
    if cache_key not in _NYSE_HOLIDAY_CACHE:
        _NYSE_HOLIDAY_CACHE[cache_key] = _get_nyse_holidays(start, end)
    holidays = _NYSE_HOLIDAY_CACHE[cache_key]
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            days.append(d)
        d += timedelta(days=1)
    return days


# ── KPI computation ────────────────────────────────────────────────────────────
def _compute_kpis(
    wdf: pd.DataFrame,
    mdf: pd.DataFrame,
    sdf: pd.DataFrame,
    tdf: pd.DataFrame,
    sector_df: pd.DataFrame,
    mdf_setup: pd.DataFrame,
    excluded: List[str],
    day_filter: str,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per YYYY-MM month >= START_MONTH,
    each column being a KPI value.
    day_filter: 'all' | 'traded' | 'good_to_trade' | 'focus_list' | 'memo_exists'
    """
    if excluded:
        excluded_dates = set(pd.to_datetime(excluded, errors="coerce").to_series().dt.date.tolist())
    else:
        excluded_dates = set()

    # Normalise dates
    def _dates(df, col):
        if df.empty or col not in df.columns:
            return df
        df = df.copy()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    wdf = _dates(wdf, "date")
    mdf = _dates(mdf, "date")
    sdf = _dates(sdf, "date")
    tdf = _dates(tdf, "buy_date")
    if not sector_df.empty:
        sector_df = _dates(sector_df, "date")

    # Build date sets per data source
    memo_dates_all = set() if mdf.empty else set(mdf["date"].dt.date.dropna().tolist())
    focus_dates_all = set()

    if not wdf.empty and "focus_list_exists" in wdf.columns:
        focus_dates_all = set(
            wdf[wdf["focus_list_exists"].apply(_is_true)]["date"].dt.date.dropna().tolist()
        )

    # ── good_dates_all from master_df where setup_count >= 5 ──────────────────
    good_dates_all = set()
    if not mdf_setup.empty and "setup_count" in mdf_setup.columns:
        mdf_setup_dated = mdf_setup.copy()
        mdf_setup_dated["date"] = pd.to_datetime(mdf_setup_dated["date"], errors="coerce")
        good_dates_all = set(
            mdf_setup_dated[mdf_setup_dated["setup_count"] >= 5]["date"].dt.date.dropna().tolist()
        )

    traded_dates_all = set() if tdf.empty else set(tdf["buy_date"].dt.date.dropna().tolist())

    # Watchlist: reload from disk for should_have_traded
    wdf_full = pd.DataFrame()
    if WATCHLIST_PATH.exists():
        try:
            wdf_full = pd.read_csv(WATCHLIST_PATH)
            wdf_full["date"] = pd.to_datetime(wdf_full["date"], errors="coerce")
        except Exception:
            pass

    # Monthly loop
    today = date.today()
    month_starts = []
    m = pd.Period(START_MONTH, "M")
    end_period = pd.Period(today.strftime("%Y-%m"), "M")
    while m <= end_period:
        month_starts.append(m)
        m += 1

    rows = []
    for period in month_starts:
        m_start = period.start_time.date()
        m_end   = min(period.end_time.date(), today)
        all_tdays = [d for d in _trading_days_between(m_start, m_end) if d not in excluded_dates]

        if not all_tdays:
            continue

        # Apply day_filter to get denominator days
        if day_filter == "traded":
            denom_days = [d for d in all_tdays if d in traded_dates_all]
        elif day_filter == "good_to_trade":
            denom_days = [d for d in all_tdays if d in good_dates_all]
        elif day_filter == "focus_list":
            denom_days = [d for d in all_tdays if d in focus_dates_all]
        elif day_filter == "memo_exists":
            denom_days = [d for d in all_tdays if d in memo_dates_all]
        else:  # all
            denom_days = all_tdays

        n_denom = len(denom_days)
        if n_denom == 0:
            continue

        denom_set = set(denom_days)

        # ── CONSISTENCY ──────────────────────────────────────────────────────
        memo_days  = len([d for d in denom_days if d in memo_dates_all])
        focus_days = len([d for d in denom_days if d in focus_dates_all])

        pct_memo  = round(100 * memo_days  / n_denom, 1)
        pct_focus = round(100 * focus_days / n_denom, 1)

        # % trades on focus list
        month_trades = tdf[tdf["buy_date"].dt.date.isin(denom_set)] if not tdf.empty else pd.DataFrame()
        n_trades = len(month_trades)
        if not month_trades.empty and not wdf_full.empty and "stock" in wdf_full.columns:
            wdf_full["_date"] = wdf_full["date"].dt.date
            wdf_full["_sym"]  = wdf_full["stock"].str.upper().str.strip()
            focus_pairs = set(
                zip(wdf_full["_date"], wdf_full["_sym"])
            )
            month_trades = month_trades.copy()
            month_trades["_date"] = month_trades["buy_date"].dt.date
            month_trades["_sym"]  = month_trades["symbol"].str.upper().str.strip()
            on_focus = month_trades.apply(
                lambda r: (r["_date"], r["_sym"]) in focus_pairs, axis=1
            ).sum()
            pct_trades_on_focus = round(100 * on_focus / n_trades, 1) if n_trades else None
        else:
            pct_trades_on_focus = None

        # % memos discussing sectors (has non-empty bullish or bearish sector data)
        if not sector_df.empty and "bullish" in sector_df.columns:
            month_sector = sector_df[sector_df["date"].dt.date.isin(denom_set)]
            has_sector = month_sector[
                (month_sector["bullish"].fillna("").str.strip() != "") |
                (month_sector["bearish"].fillna("").str.strip() != "")
            ]
            n_sector_days = len(has_sector)
            n_memo_denom  = len([d for d in denom_days if d in memo_dates_all])
            pct_sector = round(100 * n_sector_days / n_memo_denom, 1) if n_memo_denom else None
        else:
            pct_sector = None

        # ── ACCURACY ─────────────────────────────────────────────────────────
        pct_good_setups_on_focus = None
        pct_should_have_traded   = None
        pct_trades_good_days     = None

        # % of master_df setup_list stocks that appear on the focus list (setup discovery rate)
        if not mdf_setup.empty and "setup_list" in mdf_setup.columns and not wdf_full.empty:
            mdf_setup2 = mdf_setup.copy()
            mdf_setup2["_date"] = pd.to_datetime(mdf_setup2["date"], errors="coerce").dt.date
            month_setups = mdf_setup2[mdf_setup2["_date"].isin(denom_set)]

            if not month_setups.empty and "stock" in wdf_full.columns:
                wdf_full["_date"] = wdf_full["date"].dt.date
                wdf_full["_sym"]  = wdf_full["stock"].str.upper().str.strip()
                focus_pairs = set(zip(wdf_full["_date"], wdf_full["_sym"]))

                total_setups  = 0
                found_on_focus = 0
                for _, row in month_setups.iterrows():
                    syms = _parse_setup_list(row["setup_list"])
                    for sym in syms:
                        total_setups += 1
                        if (row["_date"], sym) in focus_pairs:
                            found_on_focus += 1

                pct_good_setups_on_focus = round(
                    100 * found_on_focus / total_setups, 1
                ) if total_setups else None

        if not wdf_full.empty:
            month_wdf = wdf_full[wdf_full["date"].dt.date.isin(denom_set)].copy()

            # % watchlist tickers that are should_have_traded
            if "should_have_traded" in month_wdf.columns:
                n_should = month_wdf["should_have_traded"].apply(_is_true).sum()
                pct_should_have_traded = round(
                    100 * n_should / len(month_wdf), 1
                ) if len(month_wdf) else None

        # % trades taken on good_to_trade days (setup_count >= 5)
        if not month_trades.empty and good_dates_all:
            month_trades_dated = month_trades.copy()
            month_trades_dated["_date"] = month_trades_dated["buy_date"].dt.date
            on_good = month_trades_dated["_date"].isin(good_dates_all).sum()
            pct_trades_good_days = round(100 * on_good / n_trades, 1) if n_trades else None

        # ── TRADING ──────────────────────────────────────────────────────────
        win_rate = avg_win = avg_loss = None
        if not month_trades.empty and "pct_return" in month_trades.columns:
            rets = month_trades["pct_return"].dropna()
            if len(rets):
                wins   = rets[rets > 0]
                losses = rets[rets <= 0]
                win_rate = round(100 * len(wins) / len(rets), 1)
                avg_win  = round(wins.mean(), 2)  if len(wins)   else None
                avg_loss = round(losses.mean(), 2) if len(losses) else None

        # ── PSYCHOLOGY ───────────────────────────────────────────────────────
        pct_confident_focused        = None
        pct_bullish_on_good          = None
        pct_bearish_on_bad           = None
        pct_trades_confident_focused = None

        if not mdf.empty and "emotional_state" in mdf.columns:
            month_mdf = mdf[mdf["date"].dt.date.isin(denom_set)].copy()
            pos_emotions = {"confident", "focused"}

            if len(month_mdf):
                n_pos = month_mdf["emotional_state"].str.lower().isin(pos_emotions).sum()
                pct_confident_focused = round(100 * n_pos / len(month_mdf), 1)

            # % of total trades taken on days where emotional_state was confident/focused
            if n_trades and not month_trades.empty:
                # get the buy_dates for this month's trades
                trade_buy_dates = month_trades.copy()
                trade_buy_dates["_date"] = trade_buy_dates["buy_date"].dt.date
                # build set of memo days where emotional state was confident/focused
                confident_days = set(
                    month_mdf[month_mdf["emotional_state"].str.lower().isin(pos_emotions)][
                        "date"
                    ].dt.date.dropna().tolist()
                )
                n_trades_confident = trade_buy_dates["_date"].isin(confident_days).sum()
                pct_trades_confident_focused = round(
                    100 * n_trades_confident / n_trades, 1
                )

            # Sentiment alignment: % of good_to_trade days (setup_count >= 5) where sentiment was bullish
            if "market_sentiment" in month_mdf.columns and good_dates_all:
                good_memo_days = month_mdf[month_mdf["date"].dt.date.isin(good_dates_all)]
                if len(good_memo_days):
                    n_bull = (good_memo_days["market_sentiment"].str.lower() == "bullish").sum()
                    pct_bullish_on_good = round(100 * n_bull / len(good_memo_days), 1)

            # % bearish/neutral sentiment on NOT-good-to-trade days
            not_good_days = denom_set - good_dates_all
            not_good_memo = month_mdf[month_mdf["date"].dt.date.isin(not_good_days)]
            if len(not_good_memo):
                n_bear = not_good_memo["market_sentiment"].str.lower().isin(
                    {"bearish", "neutral"}
                ).sum()
                pct_bearish_on_bad = round(100 * n_bear / len(not_good_memo), 1)

        rows.append({
            "month":                         str(period),
            "n_trading_days":                n_denom,
            "n_trades":                      n_trades,
            # Consistency
            "pct_days_with_memo":            pct_memo,
            "pct_days_with_focus":           pct_focus,
            "pct_trades_on_focus":           pct_trades_on_focus,
            "pct_memos_with_sectors":        pct_sector,
            # Accuracy
            "pct_good_setups_on_focus":      pct_good_setups_on_focus,
            "pct_should_have_traded":        pct_should_have_traded,
            "pct_trades_on_good_days":       pct_trades_good_days,
            # Trading
            "win_rate":                      win_rate,
            "avg_win_pct":                   avg_win,
            "avg_loss_pct":                  avg_loss,
            # Psychology
            "pct_confident_focused":         pct_confident_focused,
            "pct_bullish_on_good_days":      pct_bullish_on_good,
            "pct_bearish_on_bad_days":       pct_bearish_on_bad,
            "pct_trades_confident_focused":  pct_trades_confident_focused,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Sparkline ──────────────────────────────────────────────────────────────────
def _sparkline_svg(values: list, color: str = "#00d4ff", height: int = 32, width: int = 80) -> str:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) < 2:
        return "<svg width='" + str(width) + "' height='" + str(height) + "'></svg>"
    mn, mx = min(vals), max(vals)
    rng = mx - mn if mx != mn else 1
    pts = []
    for i, v in enumerate(vals):
        x = int(i / (len(vals) - 1) * width)
        y = int(height - ((v - mn) / rng) * (height - 4) - 2)
        pts.append(str(x) + "," + str(y))
    return (
        "<svg width='" + str(width) + "' height='" + str(height) +
        "' style='overflow:visible'>"
        "<polyline points='" + " ".join(pts) + "' fill='none' stroke='" + color +
        "' stroke-width='1.5' stroke-linejoin='round'/>"
        "</svg>"
    )


# ── Metric card ────────────────────────────────────────────────────────────────
def _metric_card(col, label: str, value, sparkline_svg: str = "",
                 color: str = "#e8eaf0", suffix: str = "", delta=None):
    delta_html = ""
    if delta is not None:
        dc = "#4ade80" if delta >= 0 else "#ff6b6b"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = (
            '<div style="font-size:10px;color:' + dc + ';margin-top:2px;">'
            + arrow + " " + str(abs(round(delta, 1))) + suffix + " vs prev mo"
            + "</div>"
        )
    spark_html = (
        '<div style="margin-top:6px;opacity:0.8;">' + sparkline_svg + '</div>'
        if sparkline_svg else ""
    )
    val_str = ("—" if value is None else str(value) + suffix)
    col.markdown(
        '<div style="background:#111318;border:1px solid #1f2330;border-radius:10px;'
        'padding:14px 16px;height:100%;">'
        '<div style="font-family:Space Mono,monospace;font-size:9px;color:#6b7280;'
        'letter-spacing:.1em;margin-bottom:6px;">' + label.upper() + '</div>'
        '<div style="font-family:Space Mono,monospace;font-size:22px;font-weight:700;'
        'color:' + color + ';">' + val_str + '</div>'
        + delta_html + spark_html +
        '</div>',
        unsafe_allow_html=True,
    )


def _color_for_pct(v, good_above: float = 70) -> str:
    if v is None:
        return "#6b7280"
    return "#4ade80" if v >= good_above else "#fbbf24" if v >= 50 else "#ff6b6b"


# ── Excluded day calendar picker ───────────────────────────────────────────────
def _render_exclusion_calendar(excluded: List[str]) -> List[str]:
    st.markdown("#### Exclude Days from KPI Denominator")
    st.caption(
        "Mark vacation, sick days, or any period you were away from trading. "
        "These days won't count against consistency metrics. "
        "Market holidays and weekends are automatically excluded."
    )

    # ── Date range input ───────────────────────────────────────────────────────
    st.markdown("**Add exclusion range**")
    ra, rb = st.columns(2)
    with ra:
        excl_start = st.date_input("From", value=None, key="kpi_excl_start")
    with rb:
        excl_end   = st.date_input("To",   value=None, key="kpi_excl_end")

    if st.button("Add range", key="kpi_excl_btn"):
        if excl_start and excl_end:
            if excl_end < excl_start:
                st.error("End date must be on or after start date.")
            else:
                # Expand range to individual trading days and add all
                new_days = _trading_days_between(excl_start, excl_end)
                added = 0
                for d in new_days:
                    day_str = str(d)
                    if day_str not in excluded:
                        excluded = excluded + [day_str]
                        added += 1
                if added:
                    _save_excluded(excluded)
                    st.success("Added " + str(added) + " trading day(s).")
                    st.rerun()
                else:
                    st.info("All days in that range were already excluded.")
        elif excl_start:
            # Single day
            day_str = str(excl_start)
            if day_str not in excluded:
                excluded = excluded + [day_str]
                _save_excluded(excluded)
                st.rerun()

    # ── Current exclusions ─────────────────────────────────────────────────────
    if excluded:
        st.markdown("---")
        st.markdown("**Currently excluded (" + str(len(excluded)) + " days):**")

        # Group consecutive days into ranges for readability
        sorted_days = sorted(pd.to_datetime(excluded).tolist())
        groups = []
        if sorted_days:
            grp_start = sorted_days[0]
            grp_end   = sorted_days[0]
            for d in sorted_days[1:]:
                if (d - grp_end).days <= 3:  # allow weekends between
                    grp_end = d
                else:
                    groups.append((grp_start.date(), grp_end.date()))
                    grp_start = d
                    grp_end   = d
            groups.append((grp_start.date(), grp_end.date()))

        group_labels = []
        for gs, ge in groups:
            if gs == ge:
                group_labels.append(str(gs))
            else:
                group_labels.append(str(gs) + " → " + str(ge))

        rem = st.multiselect(
            "Select ranges to remove",
            options=group_labels,
            default=[],
            key="kpi_excl_remove",
        )
        if rem and st.button("Remove selected", key="kpi_excl_rem_btn"):
            # Map labels back to date sets
            to_remove = set()
            for label, (gs, ge) in zip(group_labels, groups):
                if label in rem:
                    d = gs
                    while d <= ge:
                        to_remove.add(str(d))
                        d += timedelta(days=1)
            excluded = [d for d in excluded if d not in to_remove]
            _save_excluded(excluded)
            st.rerun()
    else:
        st.caption("No days excluded yet.")
    return excluded


# ── Section header ─────────────────────────────────────────────────────────────
def _section_header(title: str, icon: str):
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;'
        'border-bottom:1px solid #1f2330;padding-bottom:10px;margin:24px 0 16px;">'
        '<span style="font-size:18px;">' + icon + '</span>'
        '<span style="font-family:Space Mono,monospace;font-size:14px;'
        'font-weight:700;color:#e8eaf0;letter-spacing:.06em;">' + title.upper() + '</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# ── Main render ────────────────────────────────────────────────────────────────
def render_kpi_page():
    st.markdown(
        '<div style="border-bottom:1px solid #1f2330;padding-bottom:14px;margin-bottom:20px;">'
        '<span style="font-family:Space Mono,monospace;font-size:22px;font-weight:700;'
        'color:#e8eaf0;">📊 KPI Dashboard</span>'
        '<span style="font-family:Space Mono,monospace;font-size:10px;color:#6b7280;'
        'letter-spacing:.12em;margin-left:12px;">CONSISTENCY · ACCURACY · TRADING · PSYCHOLOGY</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    wdf        = _load(WATCHLIST_PATH)
    mdf        = _load(MARKET_PATH)
    sdf        = _load(STOCK_PATH)
    tdf        = _load(TRADES_PATH)
    sector_df  = _load(SECTOR_THEMES_PATH)
    mdf_setup  = _load(MASTER_DF_PATH)   # master_df with setup_list + setup_count
    excluded   = _load_excluded()

    # ── Controls row ───────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])
    with ctrl1:
        day_filter = st.selectbox(
            "Count denominator as",
            options=["all", "traded", "good_to_trade", "focus_list", "memo_exists"],
            format_func=lambda x: {
                "all":           "All trading days",
                "traded":        "Days traded on",
                "good_to_trade": "Days good to trade",
                "focus_list":    "Days focus list exists",
                "memo_exists":   "Days with voice memo",
            }[x],
            key="kpi_day_filter",
        )
    with ctrl2:
        view_mode = st.selectbox(
            "View",
            options=["Single month", "All months table", "Trend charts"],
            key="kpi_view_mode",
        )
    with ctrl3:
        month_selector_placeholder = st.empty()
    with ctrl4:
        with st.expander("⚙️ Excluded days (" + str(len(excluded)) + ")"):
            excluded = _render_exclusion_calendar(excluded)

    # ── Sector extraction ──────────────────────────────────────────────────────
    sec_col1, sec_col2 = st.columns([3, 1])
    with sec_col2:
        if st.button("🔄 Extract Sectors", key="run_sector_extract",
                     help="Run OpenAI prompt to extract sector/industry mentions from transcripts"):
            if mdf.empty:
                st.error("No market_thoughts.csv found.")
            else:
                with st.spinner("Extracting sectors from transcripts..."):
                    sector_df = run_sector_extraction(mdf)
                if not sector_df.empty:
                    st.success("Sector extraction complete — " + str(len(sector_df)) + " days processed.")
                    st.rerun()
    with sec_col1:
        if sector_df.empty:
            st.caption("⚠️ No sector data yet — click 'Extract Sectors' to populate the sector KPI.")
        else:
            st.caption("✅ Sector data available for " + str(len(sector_df)) + " days.")

    st.markdown("---")

    # ── Compute KPIs ───────────────────────────────────────────────────────────
    kdf = _compute_kpis(wdf, mdf, sdf, tdf, sector_df, mdf_setup, excluded, day_filter)

    if kdf.empty:
        st.warning("No KPI data available yet. Check that data files exist and November 2025 has data.")
        return

    # ── Month selector (fill placeholder now that kdf is available) ───────────
    available_months = kdf["month"].tolist()
    with month_selector_placeholder:
        selected_month = st.selectbox(
            "Month",
            options=available_months,
            index=len(available_months) - 1,  # default to latest
            key="kpi_month_sel",
        )

    # Row for selected month and previous month (for delta)
    sel_row  = kdf[kdf["month"] == selected_month]
    sel_idx  = available_months.index(selected_month)
    prev_row = kdf[kdf["month"] == available_months[sel_idx - 1]] if sel_idx > 0 else pd.DataFrame()

    # ── Helper to get sparkline for a column ──────────────────────────────────
    def spark(col_name: str, color: str = "#00d4ff") -> str:
        if col_name not in kdf.columns:
            return ""
        return _sparkline_svg(kdf[col_name].tolist(), color=color)

    def latest(col_name: str):
        if col_name not in sel_row.columns or sel_row.empty:
            return None
        vals = sel_row[col_name].dropna()
        return vals.iloc[0] if len(vals) else None

    def delta(col_name: str):
        if col_name not in kdf.columns or prev_row.empty:
            return None
        cur  = sel_row[col_name].dropna()
        prev = prev_row[col_name].dropna()
        if cur.empty or prev.empty:
            return None
        return round(float(cur.iloc[0]) - float(prev.iloc[0]), 1)

    # ── SINGLE MONTH VIEW ─────────────────────────────────────────────────────
    if view_mode == "Single month":
        latest_month = selected_month
        n_days       = int(sel_row["n_trading_days"].iloc[0]) if not sel_row.empty else 0
        n_trades_val = int(sel_row["n_trades"].iloc[0]) if not sel_row.empty else 0
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:11px;color:#6b7280;">'
            + latest_month + " · " + str(n_days) + " trading days · " +
            str(n_trades_val) + " trades"
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        _section_header("Consistency", "📋")
        c1, c2, c3, c4 = st.columns(4)
        v = latest("pct_days_with_memo")
        _metric_card(c1, "Days with Memo", v, spark("pct_days_with_memo"),
                     color=_color_for_pct(v, 80), suffix="%", delta=delta("pct_days_with_memo"))
        v = latest("pct_days_with_focus")
        _metric_card(c2, "Days with Focus List", v, spark("pct_days_with_focus"),
                     color=_color_for_pct(v, 80), suffix="%", delta=delta("pct_days_with_focus"))
        v = latest("pct_trades_on_focus")
        _metric_card(c3, "Trades on Focus List", v, spark("pct_trades_on_focus"),
                     color=_color_for_pct(v, 80), suffix="%", delta=delta("pct_trades_on_focus"))
        v = latest("pct_memos_with_sectors")
        _metric_card(c4, "Memos Discussing Sectors", v, spark("pct_memos_with_sectors", "#7c5cfc"),
                     color=_color_for_pct(v, 60), suffix="%", delta=delta("pct_memos_with_sectors"))

        _section_header("Accuracy", "🎯")
        c1, c2, c3 = st.columns(3)
        v = latest("pct_good_setups_on_focus")
        _metric_card(c1, "Good Setups on Focus List", v, spark("pct_good_setups_on_focus", "#4ade80"),
                     color=_color_for_pct(v, 60), suffix="%", delta=delta("pct_good_setups_on_focus"))
        v = latest("pct_should_have_traded")
        _metric_card(c2, "Watchlist → Should Have Traded", v,
                     spark("pct_should_have_traded", "#f59e0b"),
                     color=_color_for_pct(v, 30), suffix="%", delta=delta("pct_should_have_traded"))
        v = latest("pct_trades_on_good_days")
        _metric_card(c3, "Trades on Good-to-Trade Days", v,
                     spark("pct_trades_on_good_days", "#4ade80"),
                     color=_color_for_pct(v, 70), suffix="%", delta=delta("pct_trades_on_good_days"))

        _section_header("Trading", "📈")
        c1, c2, c3 = st.columns(3)
        v = latest("win_rate")
        _metric_card(c1, "Win Rate", v, spark("win_rate", "#4ade80"),
                     color=_color_for_pct(v, 55), suffix="%", delta=delta("win_rate"))
        v = latest("avg_win_pct")
        _metric_card(c2, "Avg Win", v, spark("avg_win_pct", "#4ade80"),
                     color="#4ade80" if v and v > 0 else "#6b7280", suffix="%",
                     delta=delta("avg_win_pct"))
        v = latest("avg_loss_pct")
        _metric_card(c3, "Avg Loss", v, spark("avg_loss_pct", "#ff6b6b"),
                     color="#ff6b6b" if v and v < 0 else "#6b7280", suffix="%",
                     delta=delta("avg_loss_pct"))

        _section_header("Psychology", "🧠")
        c1, c2, c3, c4 = st.columns(4)
        v = latest("pct_confident_focused")
        _metric_card(c1, "Days Confident / Focused", v,
                     spark("pct_confident_focused", "#22d3ee"),
                     color=_color_for_pct(v, 60), suffix="%",
                     delta=delta("pct_confident_focused"))
        v = latest("pct_trades_confident_focused")
        _metric_card(c2, "Trades on Confident Days", v,
                     spark("pct_trades_confident_focused", "#22d3ee"),
                     color=_color_for_pct(v, 60), suffix="%",
                     delta=delta("pct_trades_confident_focused"))
        v = latest("pct_bullish_on_good_days")
        _metric_card(c3, "Bullish Sentiment on Good Days", v,
                     spark("pct_bullish_on_good_days", "#4ade80"),
                     color=_color_for_pct(v, 60), suffix="%",
                     delta=delta("pct_bullish_on_good_days"))
        v = latest("pct_bearish_on_bad_days")
        _metric_card(c4, "Bearish/Neutral on Not-Good Days", v,
                     spark("pct_bearish_on_bad_days", "#f59e0b"),
                     color=_color_for_pct(v, 50), suffix="%",
                     delta=delta("pct_bearish_on_bad_days"))

    # ── ALL MONTHS TABLE ────────────────────────────────────────────────────────
    elif view_mode == "All months table":
        display_cols = {
            "month":                         "Month",
            "n_trading_days":                "Trading Days",
            "n_trades":                      "Trades",
            "pct_days_with_memo":            "Memo %",
            "pct_days_with_focus":           "Focus List %",
            "pct_trades_on_focus":           "Trades on Focus %",
            "pct_memos_with_sectors":        "Sector Memo %",
            "pct_good_setups_on_focus":      "Good Setups on Focus %",
            "pct_should_have_traded":        "Should Have %",
            "pct_trades_on_good_days":       "Trades on Good Days %",
            "win_rate":                      "Win Rate %",
            "avg_win_pct":                   "Avg Win %",
            "avg_loss_pct":                  "Avg Loss %",
            "pct_confident_focused":         "Confident/Focused %",
            "pct_trades_confident_focused":  "Trades on Confident Days %",
            "pct_bullish_on_good_days":      "Bullish on Good %",
            "pct_bearish_on_bad_days":       "Bearish on Bad %",
        }
        show_cols = [c for c in display_cols if c in kdf.columns]
        display_df = kdf[show_cols].rename(columns=display_cols).sort_values("Month", ascending=False)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ── TREND CHARTS ───────────────────────────────────────────────────────────
    elif view_mode == "Trend charts":
        all_metrics = {
            "pct_days_with_memo":            ("% Days with Memo",              "#00d4ff"),
            "pct_days_with_focus":           ("% Days with Focus List",         "#7c5cfc"),
            "pct_trades_on_focus":           ("% Trades on Focus List",         "#4ade80"),
            "pct_memos_with_sectors":        ("% Memos with Sector Themes",     "#a855f7"),
            "pct_good_setups_on_focus":      ("% Good Setups on Focus",         "#4ade80"),
            "pct_should_have_traded":        ("% Should Have Traded",           "#f59e0b"),
            "pct_trades_on_good_days":       ("% Trades on Good Days",          "#22d3ee"),
            "win_rate":                      ("Win Rate %",                     "#4ade80"),
            "avg_win_pct":                   ("Avg Win %",                      "#4ade80"),
            "avg_loss_pct":                  ("Avg Loss %",                     "#ff6b6b"),
            "pct_confident_focused":         ("% Confident / Focused",          "#22d3ee"),
            "pct_trades_confident_focused":  ("% Trades on Confident Days",     "#22d3ee"),
            "pct_bullish_on_good_days":      ("% Bullish on Good Days",         "#4ade80"),
            "pct_bearish_on_bad_days":       ("% Bearish on Bad Days",          "#f59e0b"),
        }
        available = {k: v for k, v in all_metrics.items() if k in kdf.columns}
        sel_metric = st.selectbox(
            "Select metric to explore",
            options=list(available.keys()),
            format_func=lambda x: available[x][0],
            key="kpi_trend_sel",
        )

        if sel_metric and sel_metric in kdf.columns:
            label, color = available[sel_metric]
            plot_df = kdf[["month", sel_metric, "n_trading_days"]].dropna(subset=[sel_metric])

            try:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=plot_df["month"],
                    y=plot_df[sel_metric],
                    mode="lines+markers+text",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=8),
                    text=[str(v) + "%" for v in plot_df[sel_metric]],
                    textposition="top center",
                    textfont=dict(size=10, color=color),
                    name=label,
                ))
                fig.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    font_color="#e0e0e0", height=360,
                    xaxis=dict(gridcolor="#1f2330"),
                    yaxis=dict(gridcolor="#1f2330", range=[0, 105] if "pct" in sel_metric or "rate" in sel_metric else None),
                    margin=dict(l=16, r=16, t=24, b=16),
                    title=dict(text=label, font=dict(size=13, color="#e8eaf0")),
                )
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.line_chart(plot_df.set_index("month")[sel_metric])

            # Raw data table
            with st.expander("📋 Raw monthly data for " + label):
                show = plot_df.rename(columns={
                    "month": "Month",
                    sel_metric: label,
                    "n_trading_days": "Trading Days",
                })
                st.dataframe(show, use_container_width=True, hide_index=True)

        # Mini sparkline grid for all metrics
        st.markdown("---")
        st.markdown("#### All Metrics Overview")
        cols = st.columns(4)
        for i, (col_key, (col_label, col_color)) in enumerate(available.items()):
            vals = kdf[col_key].tolist()
            last = latest(col_key)
            svg  = _sparkline_svg(vals, color=col_color, height=28, width=70)
            _metric_card(
                cols[i % 4],
                col_label,
                last,
                svg,
                color=col_color,
                suffix="%",
            )