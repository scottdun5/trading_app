from typing import Optional
"""
memo_analysis.py
Analyzes voice memo thoughts comparing trades taken vs trades that should have been taken.
Only analyzes dates where voice memos actually exist.
Uses OpenAI gpt-4o. Results saved to data/memo_analysis.json.
"""

import json
import pandas as pd
import streamlit as st
from datetime import date
from pathlib import Path
from openai import OpenAI


OPENAI_MODEL = "gpt-4o"
TIGERS = ["Tightness", "Ignition", "Group", "Earnings", "RS"]
MAX_TOKENS   = 4000


# ── OpenAI client ──────────────────────────────────────────────────────────────

def _get_client():
    try:
        from utils.config import OPENAI_API_KEY
    except ImportError:
        import sys
        sys.path.append(str(Path(__file__).parent))
        from config import OPENAI_API_KEY
    return OpenAI(api_key=OPENAI_API_KEY)


def _call_llm(prompt: str) -> str:
    if not prompt.strip():
        return ""
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return "ERROR: " + str(e)


# ── IO helpers ─────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def _normalise_dates(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _is_true(val) -> bool:
    """Safely check if a value is truthy, handling None/NaN/mixed types."""
    if val is None:
        return False
    try:
        import math
        if isinstance(val, float) and math.isnan(val):
            return False
    except Exception:
        pass
    return str(val).strip().lower() in ("true", "1", "yes")


# ── Data assembly ──────────────────────────────────────────────────────────────


def _extract_tigers(th) -> dict:
    """Extract TIGERS factor presence and score from a stock_thoughts row."""
    if th is None:
        return {"tigers_score": 0, "Tightness": False, "Ignition": False,
                "Group": False, "Earnings": False, "RS": False}
    result = {}
    score = 0
    for f in TIGERS:
        present = f in th.index and str(th[f]).strip() not in ("", "nan", "None")
        result[f] = present
        if present:
            score += 1
    result["tigers_score"] = score
    return result

def _build_date_bundles(
    sdf: pd.DataFrame,
    mdf: pd.DataFrame,
    wdf: pd.DataFrame,
    tdf: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date:   Optional[date] = None,
) -> dict:
    """
    Returns bundles keyed by date string. CRITICAL: only includes dates
    where at least one voice memo exists (in sdf or mdf). No memo = no analysis.

    Each bundle:
        took:        trades taken where memo exists for that stock+date
        should_have: watchlist stocks flagged should_have_traded=True,
                     not in took list, on a date with at least one memo
        market:      market_thoughts for that date
        memo_dates:  set of dates with any memo activity
    """
    sdf = _normalise_dates(sdf, "date")
    mdf = _normalise_dates(mdf, "date")
    wdf = _normalise_dates(wdf, "date")
    tdf = _normalise_dates(tdf, "buy_date")

    # Date range filter
    for df_, col in [(sdf, "date"), (mdf, "date"), (wdf, "date")]:
        if start_date:
            df_.drop(df_[df_[col].dt.date < start_date].index, inplace=True)
        if end_date:
            df_.drop(df_[df_[col].dt.date > end_date].index, inplace=True)
    if start_date:
        tdf = tdf[tdf["buy_date"].dt.date >= start_date]
    if end_date:
        tdf = tdf[tdf["buy_date"].dt.date <= end_date]

    # ── Dates that have any memo (stock or market) — gate for analysis ─────────
    stock_col = "stock" if "stock" in sdf.columns else "symbol"
    sdf["_sym"]  = sdf[stock_col].str.upper().str.strip()
    sdf["_date"] = sdf["date"].dt.date
    mdf["_date"] = mdf["date"].dt.date

    memo_dates = set(sdf["_date"].tolist()) | set(mdf["_date"].tolist())

    # Stock thoughts lookup: (date, symbol) -> row
    thoughts_map = {}
    for _, row in sdf.iterrows():
        thoughts_map[(row["_date"], row["_sym"])] = row

    # Market thoughts lookup: date -> row
    market_map = {row["_date"]: row for _, row in mdf.iterrows()}

    # Watchlist should_have_traded — robust boolean check
    wdf_stock_col = "stock" if "stock" in wdf.columns else "symbol"
    wdf["_sym"]  = wdf[wdf_stock_col].str.upper().str.strip()
    wdf["_date"] = wdf["date"].dt.date

    if "should_have_traded" in wdf.columns:
        should_mask = wdf["should_have_traded"].apply(_is_true)
        should_df   = wdf[should_mask].copy()
    else:
        should_df = pd.DataFrame()

    # Trades
    tdf["_sym"]  = tdf["symbol"].str.upper().str.strip()
    tdf["_date"] = tdf["buy_date"].dt.date

    # Only analyze dates that intersect with memo dates
    trade_dates  = set(tdf["_date"].tolist())
    should_dates = set(should_df["_date"].tolist()) if not should_df.empty else set()
    candidate_dates = (trade_dates | should_dates) & memo_dates

    bundles = {}
    for d in sorted(candidate_dates):
        took_list   = []
        should_list = []

        # Trades taken on this date
        day_trades = tdf[tdf["_date"] == d]
        for _, tr in day_trades.iterrows():
            sym = tr["_sym"]
            th  = thoughts_map.get((d, sym))
            tigers_data = _extract_tigers(th)
            took_list.append({
                "symbol":             sym,
                "raw_thoughts":       str(th["raw_thoughts"]) if th is not None and "raw_thoughts" in th else "",
                "summarized_thoughts":str(th["summarized_thoughts"]) if th is not None and "summarized_thoughts" in th else "",
                "trade_rationale":    str(th["trade_rationale_explained"]) if th is not None and "trade_rationale_explained" in th else "",
                "pct_return":         float(tr["pct_return"]) if "pct_return" in tr.index and pd.notna(tr["pct_return"]) else None,
                "gain":               float(tr["gain"]) if "gain" in tr.index and pd.notna(tr["gain"]) else None,
                **tigers_data,
            })

        # Should-have trades on this date (exclude already-took symbols)
        if not should_df.empty:
            took_syms  = set(day_trades["_sym"].tolist())
            day_should = should_df[should_df["_date"] == d]
            for _, wr in day_should.iterrows():
                sym = wr["_sym"]
                if sym in took_syms:
                    continue
                th = thoughts_map.get((d, sym))
                tigers_data = _extract_tigers(th)
                should_list.append({
                    "symbol":             sym,
                    "raw_thoughts":       str(th["raw_thoughts"]) if th is not None and "raw_thoughts" in th else "",
                    "summarized_thoughts":str(th["summarized_thoughts"]) if th is not None and "summarized_thoughts" in th else "",
                    **tigers_data,
                })

        # Skip if nothing to analyze on this memo date
        if not took_list and not should_list:
            continue

        # Market context
        mkt = market_map.get(d)
        market_ctx = {}
        for field in ["market_sentiment", "emotional_state", "key_themes",
                      "technical_analysis", "emotional_intensity"]:
            market_ctx[field] = str(mkt[field]) if mkt is not None and field in mkt else ""

        bundles[str(d)] = {
            "took":        took_list,
            "should_have": should_list,
            "market":      market_ctx,
        }

    return bundles


def _build_monthly_bundles(bundles: dict) -> dict:
    monthly = {}
    for date_str, data in bundles.items():
        month_key = date_str[:7]
        if month_key not in monthly:
            monthly[month_key] = {"took": [], "should_have": [], "dates": []}
        monthly[month_key]["took"].extend(data["took"])
        monthly[month_key]["should_have"].extend(data["should_have"])
        monthly[month_key]["dates"].append(date_str)
    return monthly


# ── Prompt builders ────────────────────────────────────────────────────────────

SWING_TRADER_CONTEXT = (
    "IMPORTANT CONTEXT: This trader is a SWING TRADER. "
    "Their goal is to hold positions for days to weeks, not intraday. "
    "They are looking for momentum setups (VCP, flat bases, bull flags, etc.) "
    "and trade based on technical breakouts with defined risk. "
    "Keep this context in mind when evaluating conviction, timing, and hesitation patterns."
)


def _build_daily_prompt(date_str: str, bundle: dict) -> str:
    took        = bundle["took"]
    should_have = bundle["should_have"]
    market      = bundle["market"]

    if not took and not should_have:
        return ""

    lines = [
        "You are analyzing a swing trader's voice memo journal for " + date_str + ".",
        SWING_TRADER_CONTEXT,
        "",
        "MARKET CONTEXT:",
        "  Sentiment: "        + market.get("market_sentiment", "N/A"),
        "  Emotional state: "  + market.get("emotional_state", "N/A"),
        "  Key themes: "       + market.get("key_themes", "N/A"),
        "  Technical: "        + market.get("technical_analysis", "N/A"),
        "  Emotional intensity: " + market.get("emotional_intensity", "N/A"),
        "",
    ]

    if took:
        lines.append("TRADES ACTUALLY TAKEN (swing trade entries):")
        for t in took:
            ret = (" | Result: " + str(round(t["pct_return"], 2)) + "%") if t["pct_return"] is not None else ""
            lines.append("  [" + t["symbol"] + "]" + ret)
            if t["raw_thoughts"]:
                lines.append("    Raw memo: " + t["raw_thoughts"][:600])
            if t["summarized_thoughts"]:
                lines.append("    Summary: " + t["summarized_thoughts"][:300])
            if t["trade_rationale"]:
                lines.append("    Rationale: " + t["trade_rationale"][:300])
        lines.append("")

    if should_have:
        lines.append("TRADES THAT SHOULD HAVE BEEN TAKEN (flagged in watchlist, but skipped):")
        for s in should_have:
            lines.append("  [" + s["symbol"] + "]")
            if s["raw_thoughts"]:
                lines.append("    Raw memo: " + s["raw_thoughts"][:600])
            if s["summarized_thoughts"]:
                lines.append("    Summary: " + s["summarized_thoughts"][:300])
            if not s["raw_thoughts"] and not s["summarized_thoughts"]:
                lines.append("    (Stock was on watchlist but no memo recorded for it)")
        lines.append("")

    lines += [
        "Analyze this trading day. Respond with these exact sections:",
        "",
        "1. CONVICTION COMPARISON",
        "   Compare memo language strength for trades taken vs missed.",
        "   Did the memo show clear signals for the should-have trades that were ignored?",
        "   For a swing trader, look for language about setup quality, base tightness, volume, RS.",
        "",
        "2. HESITATION PATTERNS",
        "   For skipped should-have trades: what language suggests why they were passed?",
        "   Look for doubt, hedging, market-condition excuses, position-size concerns.",
        "",
        "3. SELECTION QUALITY",
        "   Given available results, did the trader pick the best setups from their memo?",
        "   Were the skipped stocks likely better swing opportunities than what was taken?",
        "",
        "4. KEY INSIGHT",
        "   One specific, actionable observation from this date the trader should internalize.",
        "",
        "5. SCORES (rate each 1-10, output as JSON on a single line at the end):",
        '   {"conviction_score": X, "selection_quality_score": X, "missed_opportunity_score": X}',
        "   conviction_score: how clearly the memo identified the best setups",
        "   selection_quality_score: how well final picks matched the best memo ideas",
        "   missed_opportunity_score: how significant were the missed trades (10 = very significant)",
        "",
        "Quote specific memo phrases where relevant. Be direct — this is for self-improvement.",
    ]

    return "\n".join(lines)


def _build_monthly_prompt(month_key: str, bundle: dict) -> str:
    took        = bundle["took"]
    should_have = bundle["should_have"]
    n_dates     = len(bundle["dates"])

    lines = [
        "You are analyzing a swing trader's voice memo journal for " + month_key + ".",
        SWING_TRADER_CONTEXT,
        "This covers " + str(n_dates) + " trading days with memo data.",
        "",
        "TRADES TAKEN THIS MONTH (" + str(len(took)) + " total):",
    ]
    for t in took[:25]:
        ret = (" (" + str(round(t["pct_return"], 2)) + "%)") if t["pct_return"] is not None else ""
        memo = t["summarized_thoughts"] or t["raw_thoughts"]
        lines.append("  [" + t["symbol"] + "]" + ret + ": " + memo[:250])

    lines += [
        "",
        "TRADES THAT SHOULD HAVE BEEN TAKEN (" + str(len(should_have)) + " total):",
    ]
    for s in should_have[:25]:
        memo = s["summarized_thoughts"] or s["raw_thoughts"]
        lines.append("  [" + s["symbol"] + "]: " + (memo[:250] if memo else "(no memo)"))

    lines += [
        "",
        "Provide a monthly analysis with these sections:",
        "",
        "1. RECURRING CONVICTION PATTERNS",
        "   What setup types does this swing trader consistently show high conviction for?",
        "   What do they consistently undervalue or talk themselves out of?",
        "",
        "2. RECURRING HESITATION PATTERNS",
        "   Consistent themes in why good swing setups were skipped.",
        "   (e.g. always second-guessing extended markets, certain sectors, low float stocks)",
        "",
        "3. SELECTION BIAS",
        "   Systematic bias between what was picked vs what should have been picked.",
        "   Include win rate comparison where returns are available.",
        "",
        "4. PSYCHOLOGICAL PATTERNS",
        "   Emotional/cognitive tendencies affecting swing trade decision quality this month.",
        "   (overconfidence, fear of extended names, FOMO, anchoring to purchase price, etc.)",
        "",
        "5. TOP 3 ACTIONABLE RECOMMENDATIONS",
        "   Specific changes to improve swing trade selection next month.",
        "",
        "6. MONTHLY SCORES (output as JSON on a single line at the end):",
        '   {"avg_conviction": X, "avg_selection_quality": X, "avg_missed_opportunity": X, "overall_month_score": X}',
        "   All scores 1-10.",
        "",
        "Reference actual symbols and memo language. Be direct.",
    ]

    return "\n".join(lines)


def _build_rollup_prompt(monthly_analyses: dict) -> str:
    lines = [
        "You are analyzing a swing trader's performance patterns across multiple months.",
        SWING_TRADER_CONTEXT,
        "",
    ]
    for month in sorted(monthly_analyses.keys()):
        data = monthly_analyses[month]
        lines.append("=== " + month + " (" + str(data.get("took_count", 0)) + " trades, " +
                      str(data.get("should_have_count", 0)) + " missed) ===")
        lines.append(str(data.get("summary", ""))[:1000])
        lines.append("")

    lines += [
        "Provide an overall analysis:",
        "",
        "1. STRONGEST RECURRING PATTERNS (across all months)",
        "   3-5 most consistent behavioral patterns — positive and negative.",
        "",
        "2. IMPROVEMENT TRAJECTORY",
        "   Is the trader improving on key swing trading dimensions? Cite specific months.",
        "",
        "3. CORE PSYCHOLOGICAL PROFILE",
        "   This trader's defining strengths and weaknesses as a swing trade decision-maker.",
        "",
        "4. PRIORITY FOCUS AREAS",
        "   The 2 highest-impact things to work on to improve swing trade selection.",
        "",
        "5. OVERALL SCORES (output as JSON on a single line at the end):",
        '   {"overall_conviction": X, "overall_selection": X, "overall_missed_opps": X, "improvement_trend": X}',
        "   improvement_trend: 1=declining, 5=flat, 10=strong improvement",
    ]

    return "\n".join(lines)


def _extract_scores(analysis_text: str) -> dict:
    """Pull the JSON score line out of the LLM response."""
    import re
    matches = re.findall(r'\{[^{}]+\}', analysis_text)
    for m in reversed(matches):
        try:
            data = json.loads(m)
            if any("score" in k or "conviction" in k or "selection" in k for k in data):
                return data
        except Exception:
            pass
    return {}


# ── Main render ────────────────────────────────────────────────────────────────

def render_memo_analysis(
    sdf: pd.DataFrame,
    mdf: pd.DataFrame,
    wdf: pd.DataFrame,
    tdf: pd.DataFrame,
    output_path: Path,
    watchlist_path: Optional[Path] = None,
) -> None:
    st.markdown("### Memo vs Trade Analysis")
    st.caption(
        "Compares your voice memo thoughts on swing trades taken vs trades you should have taken. "
        "Only dates with actual voice memos are analyzed. "
        "Results are saved — re-run manually when you want a fresh pass."
    )

    # Always reload watchlist from disk so should_have_traded column is present
    if watchlist_path and Path(watchlist_path).exists():
        wdf = pd.read_csv(watchlist_path)
        if "date" in wdf.columns:
            wdf["date"] = pd.to_datetime(wdf["date"], errors="coerce")
    # else fall through with whatever wdf was passed in

    tdf_dates = pd.to_datetime(tdf["buy_date"], errors="coerce").dropna()
    sdf_dates = pd.to_datetime(sdf["date"], errors="coerce").dropna()

    if tdf_dates.empty:
        st.warning("No trade data found.")
        return

    # Default date range to memo date range, not trade date range
    memo_min = sdf_dates.min().date() if not sdf_dates.empty else tdf_dates.min().date()
    memo_max = sdf_dates.max().date() if not sdf_dates.empty else tdf_dates.max().date()

    dr_col1, dr_col2 = st.columns(2)
    with dr_col1:
        start_date = st.date_input("From", value=memo_min,
                                   min_value=memo_min, max_value=memo_max,
                                   key="ma_start")
    with dr_col2:
        end_date = st.date_input("To", value=memo_max,
                                 min_value=memo_min, max_value=memo_max,
                                 key="ma_end")


    # ── Debug expander ────────────────────────────────────────────────────────
    with st.expander("🔍 Join diagnostic — expand to debug should_have_traded data", expanded=False):
        st.markdown("**wdf shape:** " + str(wdf.shape))
        st.markdown("**wdf columns:** " + str(list(wdf.columns)))

        if "should_have_traded" in wdf.columns:
            st.markdown("**should_have_traded value counts (raw):**")
            st.write(wdf["should_have_traded"].value_counts(dropna=False))
            st.markdown("**should_have_traded dtype:** " + str(wdf["should_have_traded"].dtype))
            st.markdown("**Sample values (first 10):** " + str(wdf["should_have_traded"].head(10).tolist()))

            # Show what _is_true sees
            flagged = wdf[wdf["should_have_traded"].apply(_is_true)]
            st.markdown("**Rows where _is_true() = True:** " + str(len(flagged)))
            if not flagged.empty:
                stock_col_dbg = "stock" if "stock" in flagged.columns else "symbol"
                st.dataframe(flagged[["date", stock_col_dbg, "should_have_traded"]].head(10),
                             use_container_width=True, hide_index=True)
        else:
            st.error("should_have_traded column NOT FOUND in wdf")

        st.markdown("**tdf buy_date sample:** " + str(pd.to_datetime(tdf["buy_date"], errors="coerce").dt.date.head(5).tolist()))
        st.markdown("**sdf date sample:** " + str(pd.to_datetime(sdf["date"], errors="coerce").dt.date.head(5).tolist()))

    # Load existing results
    existing = _load_json(output_path)

    st.markdown("---")
    run_col, info_col = st.columns([1, 3])
    with run_col:
        run_btn = st.button("Run Analysis", type="primary", key="run_memo_analysis")
    with info_col:
        if existing:
            meta = existing.get("meta", {})
            st.caption(
                "Last run: " + str(meta.get("last_run", "unknown")) +
                " | " + str(len(existing.get("daily", {}))) + " days" +
                " | " + str(len(existing.get("monthly", {}))) + " months"
            )
        else:
            st.caption("No saved results yet. Click Run Analysis to generate.")

    if run_btn:
        with st.spinner("Assembling memo + trade data..."):
            bundles = _build_date_bundles(sdf, mdf, wdf, tdf, start_date, end_date)

        if not bundles:
            st.warning(
                "No dates found with voice memos AND trade/watchlist data in this range. "
                "Check that should_have_traded is populated in your watchlist."
            )
            return

        monthly_bundles = _build_monthly_bundles(bundles)

        st.info(
            "Found " + str(len(bundles)) + " dates and " +
            str(len(monthly_bundles)) + " months to analyze. This may take a few minutes."
        )

        results = {"meta": {}, "daily": {}, "monthly": {}, "rollup": "", "scores": {}}

        # Daily
        daily_bar  = st.progress(0, text="Analyzing daily memos...")
        daily_keys = list(bundles.keys())
        for i, date_str in enumerate(daily_keys):
            daily_bar.progress(
                (i + 1) / len(daily_keys),
                text="Analyzing " + date_str + " (" + str(i+1) + "/" + str(len(daily_keys)) + ")"
            )
            prompt   = _build_daily_prompt(date_str, bundles[date_str])
            analysis = _call_llm(prompt)
            scores   = _extract_scores(analysis)
            # Store TIGERS-relevant fields only (keep JSON lean)
            def _slim_entry(e):
                return {k: e[k] for k in ["symbol", "tigers_score", "Tightness",
                        "Ignition", "Group", "Earnings", "RS", "pct_return"]
                        if k in e}

            results["daily"][date_str] = {
                "analysis":          analysis,
                "took_count":        len(bundles[date_str]["took"]),
                "should_have_count": len(bundles[date_str]["should_have"]),
                "scores":            scores,
                "took_entries":      [_slim_entry(e) for e in bundles[date_str]["took"]],
                "should_entries":    [_slim_entry(e) for e in bundles[date_str]["should_have"]],
            }
        daily_bar.empty()

        # Monthly
        monthly_bar = st.progress(0, text="Analyzing monthly patterns...")
        month_keys  = list(monthly_bundles.keys())
        for i, month_key in enumerate(month_keys):
            monthly_bar.progress(
                (i + 1) / len(month_keys),
                text="Analyzing " + month_key
            )
            prompt   = _build_monthly_prompt(month_key, monthly_bundles[month_key])
            analysis = _call_llm(prompt)
            scores   = _extract_scores(analysis)
            results["monthly"][month_key] = {
                "summary":           analysis,
                "took_count":        len(monthly_bundles[month_key]["took"]),
                "should_have_count": len(monthly_bundles[month_key]["should_have"]),
                "dates":             monthly_bundles[month_key]["dates"],
                "scores":            scores,
            }
        monthly_bar.empty()

        # Rollup
        with st.spinner("Generating rolled-up summary..."):
            rollup_text   = _call_llm(_build_rollup_prompt(results["monthly"]))
            rollup_scores = _extract_scores(rollup_text)
            results["rollup"]        = rollup_text
            results["rollup_scores"] = rollup_scores

        results["meta"] = {
            "last_run":   str(date.today()),
            "start_date": str(start_date),
            "end_date":   str(end_date),
            "n_dates":    len(daily_keys),
            "n_months":   len(month_keys),
        }

        _save_json(results, output_path)
        existing = results
        st.success(
            "Done — " + str(len(daily_keys)) + " days, " +
            str(len(month_keys)) + " months analyzed."
        )

    if not existing or (not existing.get("daily") and not existing.get("monthly")):
        return

    # ── Score trend chart ──────────────────────────────────────────────────────
    daily_data   = existing.get("daily", {})
    monthly_data = existing.get("monthly", {})

    if daily_data:
        score_rows = []
        for d, data in sorted(daily_data.items()):
            s = data.get("scores", {})
            if s:
                row = {"date": d}
                row.update(s)
                score_rows.append(row)

        if score_rows:
            st.markdown("---")
            st.markdown("#### Score Trends Over Time")
            score_df = pd.DataFrame(score_rows)
            score_df["date"] = pd.to_datetime(score_df["date"])

            score_cols = [c for c in score_df.columns if c != "date"]
            sel_scores = st.multiselect(
                "Scores to chart", options=score_cols, default=score_cols,
                key="ma_score_sel"
            )
            if sel_scores:
                try:
                    import plotly.express as px
                    fig = px.line(
                        score_df, x="date", y=sel_scores,
                        markers=True, height=350,
                        labels={"value": "Score (1-10)", "date": "Date"},
                        title="Daily Scores Over Time",
                    )
                    fig.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e0e0e0", legend_title="Metric",
                        yaxis=dict(range=[0, 11]),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.line_chart(score_df.set_index("date")[sel_scores])

        # Monthly score chart
        if monthly_data:
            month_score_rows = []
            for m, data in sorted(monthly_data.items()):
                s = data.get("scores", {})
                if s:
                    row = {"month": m}
                    row.update(s)
                    month_score_rows.append(row)

            if month_score_rows:
                st.markdown("#### Monthly Score Summary")
                mscore_df = pd.DataFrame(month_score_rows)
                st.dataframe(mscore_df, use_container_width=True, hide_index=True)


    # ── TIGERS Analysis ────────────────────────────────────────────────────────
    if daily_data:
        st.markdown("---")
        st.markdown("#### TIGERS Analysis: Took vs Should Have Taken")
        st.caption(
            "Compares how well-developed (per TIGERS framework) your setups were "
            "for trades you took vs trades you should have taken."
        )

        # Flatten all took and should_have entries from saved daily results
        # Re-build from raw data if scores present, otherwise use stored bundle
        took_tigers    = []
        should_tigers  = []

        # Reload bundles to get TIGERS data (already stored in daily results if run after patch)
        for date_str, data in daily_data.items():
            for entry in data.get("took_entries", []):
                took_tigers.append(entry)
            for entry in data.get("should_entries", []):
                should_tigers.append(entry)

        if took_tigers or should_tigers:
            ta_col1, ta_col2 = st.columns(2)

            # ── Avg TIGERS score comparison ────────────────────────────────────
            with ta_col1:
                st.markdown("##### Avg TIGERS Score")
                avg_took   = (sum(e.get("tigers_score", 0) for e in took_tigers) / len(took_tigers)) if took_tigers else 0
                avg_should = (sum(e.get("tigers_score", 0) for e in should_tigers) / len(should_tigers)) if should_tigers else 0

                try:
                    import plotly.graph_objects as go
                    fig_avg = go.Figure(go.Bar(
                        x=["Trades Taken", "Should Have Taken"],
                        y=[round(avg_took, 2), round(avg_should, 2)],
                        marker_color=["#00d4ff", "#f59e0b"],
                        text=[str(round(avg_took, 2)), str(round(avg_should, 2))],
                        textposition="outside",
                    ))
                    fig_avg.update_layout(
                        yaxis=dict(range=[0, 5.5], title="Avg TIGERS Score (0-5)"),
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e0e0e0", height=300,
                        margin=dict(l=16, r=16, t=16, b=16),
                    )
                    st.plotly_chart(fig_avg, use_container_width=True)
                except ImportError:
                    st.metric("Took avg TIGERS",        str(round(avg_took, 2)) + "/5")
                    st.metric("Should-have avg TIGERS", str(round(avg_should, 2)) + "/5")

            # ── % each TIGERS factor present ───────────────────────────────────
            with ta_col2:
                st.markdown("##### TIGERS Factor % Present")
                tigers_factors = ["Tightness", "Ignition", "Group", "Earnings", "RS"]

                def _factor_pct(entries, factor):
                    if not entries:
                        return 0.0
                    return round(100 * sum(1 for e in entries if e.get(factor, False)) / len(entries), 1)

                factor_rows = []
                for f in tigers_factors:
                    factor_rows.append({
                        "Factor":            f,
                        "Took %":            _factor_pct(took_tigers, f),
                        "Should Have %":     _factor_pct(should_tigers, f),
                    })
                factor_df = pd.DataFrame(factor_rows)

                try:
                    import plotly.graph_objects as go
                    fig_fac = go.Figure()
                    fig_fac.add_trace(go.Bar(
                        name="Trades Taken",
                        x=factor_df["Factor"],
                        y=factor_df["Took %"],
                        marker_color="#00d4ff",
                        text=[str(v) + "%" for v in factor_df["Took %"]],
                        textposition="outside",
                    ))
                    fig_fac.add_trace(go.Bar(
                        name="Should Have Taken",
                        x=factor_df["Factor"],
                        y=factor_df["Should Have %"],
                        marker_color="#f59e0b",
                        text=[str(v) + "%" for v in factor_df["Should Have %"]],
                        textposition="outside",
                    ))
                    fig_fac.update_layout(
                        barmode="group",
                        yaxis=dict(range=[0, 115], title="% of trades with factor"),
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e0e0e0", height=300,
                        legend=dict(orientation="h", y=1.15),
                        margin=dict(l=16, r=16, t=16, b=16),
                    )
                    st.plotly_chart(fig_fac, use_container_width=True)
                except ImportError:
                    st.dataframe(factor_df, use_container_width=True, hide_index=True)

            # ── TIGERS score over time (took vs should) ────────────────────────
            st.markdown("##### TIGERS Score Over Time: Took vs Should Have Taken")
            time_rows = []
            for date_str, data in sorted(daily_data.items()):
                took_entries   = data.get("took_entries", [])
                should_entries = data.get("should_entries", [])
                if took_entries:
                    time_rows.append({
                        "date":  date_str,
                        "group": "Trades Taken",
                        "avg_tigers": sum(e.get("tigers_score", 0) for e in took_entries) / len(took_entries),
                    })
                if should_entries:
                    time_rows.append({
                        "date":  date_str,
                        "group": "Should Have Taken",
                        "avg_tigers": sum(e.get("tigers_score", 0) for e in should_entries) / len(should_entries),
                    })

            if time_rows:
                time_df = pd.DataFrame(time_rows)
                time_df["date"] = pd.to_datetime(time_df["date"])
                try:
                    import plotly.express as px
                    fig_time = px.line(
                        time_df, x="date", y="avg_tigers", color="group",
                        color_discrete_map={"Trades Taken": "#00d4ff", "Should Have Taken": "#f59e0b"},
                        markers=True, height=300,
                        labels={"avg_tigers": "Avg TIGERS Score", "date": "Date", "group": ""},
                    )
                    fig_time.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e0e0e0",
                        yaxis=dict(range=[0, 5.5]),
                        legend=dict(orientation="h", y=1.15),
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                except ImportError:
                    st.dataframe(time_df, use_container_width=True, hide_index=True)

            # ── Raw factor table ───────────────────────────────────────────────
            with st.expander("View factor breakdown table"):
                st.dataframe(factor_df, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No TIGERS data in saved results. Re-run the analysis to populate TIGERS charts. "
                "This data is captured automatically during the analysis run."
            )

    # ── Tabs ───────────────────────────────────────────────────────────────────
    st.markdown("---")
    tab_rollup, tab_monthly, tab_daily = st.tabs([
        "Overall Summary",
        "Monthly Breakdown",
        "Daily Detail",
    ])

    with tab_rollup:
        rollup_text = existing.get("rollup", "")
        if rollup_text:
            meta = existing.get("meta", {})
            st.caption(
                str(meta.get("start_date", "")) + " → " +
                str(meta.get("end_date", "")) +
                " | Last run: " + str(meta.get("last_run", ""))
            )
            st.markdown(rollup_text)
        else:
            st.info("No rolled-up summary yet — run the analysis.")

    with tab_monthly:
        if not monthly_data:
            st.info("No monthly analyses yet.")
        else:
            rows = []
            for m, data in sorted(monthly_data.items()):
                s = data.get("scores", {})
                rows.append({
                    "Month":             m,
                    "Trades Taken":      data.get("took_count", 0),
                    "Should Have Taken": data.get("should_have_count", 0),
                    "Days":              len(data.get("dates", [])),
                    "Conviction":        s.get("avg_conviction", ""),
                    "Selection Quality": s.get("avg_selection_quality", ""),
                    "Missed Opps":       s.get("avg_missed_opportunity", ""),
                    "Month Score":       s.get("overall_month_score", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")
            sel_month = st.selectbox(
                "View monthly analysis",
                options=sorted(monthly_data.keys(), reverse=True),
                key="ma_month_sel"
            )
            if sel_month and sel_month in monthly_data:
                d = monthly_data[sel_month]
                st.caption(
                    str(d.get("took_count", 0)) + " trades taken · " +
                    str(d.get("should_have_count", 0)) + " should-have · " +
                    str(len(d.get("dates", []))) + " memo days"
                )
                st.markdown(d.get("summary", "No analysis available."))

    with tab_daily:
        if not daily_data:
            st.info("No daily analyses yet.")
        else:
            rows = []
            for d, data in sorted(daily_data.items(), reverse=True):
                s = data.get("scores", {})
                rows.append({
                    "Date":              d,
                    "Trades Taken":      data.get("took_count", 0),
                    "Should Have Taken": data.get("should_have_count", 0),
                    "Conviction":        s.get("conviction_score", ""),
                    "Selection Quality": s.get("selection_quality_score", ""),
                    "Missed Opps":       s.get("missed_opportunity_score", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.markdown("---")
            sel_date = st.selectbox(
                "View daily analysis",
                options=sorted(daily_data.keys(), reverse=True),
                key="ma_date_sel"
            )
            if sel_date and sel_date in daily_data:
                d = daily_data[sel_date]
                st.caption(
                    str(d.get("took_count", 0)) + " trades taken · " +
                    str(d.get("should_have_count", 0)) + " should-have trades"
                )
                st.markdown(d.get("analysis", "No analysis available."))