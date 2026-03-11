from typing import Union, Optional
"""
regime_analysis.py
Explores manual trade/watchlist annotations sliced by market regime.

Joins raw_trades (buy_date) → indices_with_breadth (date).
Regime dimensions: REGIME_POSITION_COLS and STREAK_ZSCORE_COLS.

Usage in journal.py:
    from regime_analysis import render_regime_analysis
    render_regime_analysis(tdf, indices_path=INDICES_PATH)
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path


# ── Regime column groups ────────────────────────────────────────────────────────
REGIME_POSITION_COLS = [
    "qqq_21EMA ATR Distance",  "qqq_50SMA ATR Distance",
    "qqqe_21EMA ATR Distance", "qqqe_50SMA ATR Distance",
    "spy_21EMA ATR Distance",  "spy_50SMA ATR Distance",
    "rsp_21EMA ATR Distance",  "rsp_50SMA ATR Distance",
    "iwm_21EMA ATR Distance",  "iwm_50SMA ATR Distance",
]

STREAK_ZSCORE_COLS = [
    "qqq_21EMA_Low_streak_zscore",  "qqq_50SMA_Low_streak_zscore",
    "qqqe_21EMA_Low_streak_zscore", "qqqe_50SMA_Low_streak_zscore",
    "spy_21EMA_Low_streak_zscore",  "spy_50SMA_Low_streak_zscore",
    "rsp_21EMA_Low_streak_zscore",  "rsp_50SMA_Low_streak_zscore",
    "iwm_21EMA_Low_streak_zscore",  "iwm_50SMA_Low_streak_zscore",
]

ANNOTATION_COLS = {
    "stop_5pct_would_have_won":    "5% Stop Would Have Won",
    "stop_at_high_prevented_loss": "Stop at High Prevented Loss",
    "panic_sold_too_early":        "Panic Sold Too Early",
}

WATCHLIST_FLAG = "should_have_traded"


def _load_indices(indices_path: Union[str, Path]) -> pd.DataFrame:
    p = Path(indices_path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, parse_dates=["date"])
    # Keep only regime cols + date
    keep = ["date"] + [c for c in REGIME_POSITION_COLS + STREAK_ZSCORE_COLS if c in df.columns]
    return df[keep]


def _bin_atr(series: pd.Series, label: str) -> pd.Series:
    """Bin ATR distance into regimes: Extended / Neutral / Oversold."""
    bins   = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
    labels = ["Deeply OS", "Slightly OS", "Neutral", "Slightly OB", "Deeply OB"]
    return pd.cut(series, bins=bins, labels=labels)


def _bin_zscore(series: pd.Series) -> pd.Series:
    """Bin streak z-score into regimes."""
    bins   = [-np.inf, -2, -1, 0, 1, 2, np.inf]
    labels = ["Very Low", "Low", "Below Avg", "Above Avg", "High", "Very High"]
    return pd.cut(series, bins=bins, labels=labels)


def _join_trades_to_indices(tdf: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    if tdf.empty or indices.empty:
        return tdf
    tdf = tdf.copy()
    if "buy_date" in tdf.columns:
        tdf["buy_date"] = pd.to_datetime(tdf["buy_date"])
    merged = tdf.merge(indices, left_on="buy_date", right_on="date", how="left")
    return merged


def _agg_by_regime(df: pd.DataFrame, regime_col_binned: str,
                   metric_col: str, metric_label: str) -> pd.DataFrame:
    """Group by binned regime and compute win rate / count for a binary metric."""
    grp = df.groupby(regime_col_binned, observed=True)[metric_col].agg(
        Count="count",
        Flagged="sum",
    ).reset_index()
    grp["Flag Rate %"] = (grp["Flagged"] / grp["Count"] * 100).round(1)
    grp.columns = [regime_col_binned, "Count", f"{metric_label} Count", f"{metric_label} Rate %"]
    return grp


def render_regime_analysis(
    tdf_path: Union[str, Path],
    watchlist_path: Optional[Union[str, Path]],
    indices_path: Union[str, Path],
) -> None:
    st.markdown("### 🌍 Regime Analysis")
    st.caption(
        "Explore how market regime at entry correlates with your manual annotations "
        "and watchlist flags."
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    indices = _load_indices(indices_path)
    if indices.empty:
        st.warning(f"Indices file not found or empty: `{indices_path}`")
        return

    # Trades
    tp = Path(tdf_path)
    if not tp.exists():
        st.warning(f"Trade log not found: `{tp}`")
        return
    tdf = pd.read_csv(tp)
    if "buy_date" in tdf.columns:
        tdf["buy_date"] = pd.to_datetime(tdf["buy_date"])
    for col in ANNOTATION_COLS:
        if col not in tdf.columns:
            tdf[col] = False
        else:
            tdf[col] = tdf[col].fillna(False).astype(bool)

    merged = _join_trades_to_indices(tdf, indices)

    # Watchlist (optional)
    wdf = pd.DataFrame()
    if watchlist_path and Path(watchlist_path).exists():
        wdf = pd.read_csv(watchlist_path)
        if "date" in wdf.columns:
            wdf["date"] = pd.to_datetime(wdf["date"])
        if WATCHLIST_FLAG not in wdf.columns:
            wdf[WATCHLIST_FLAG] = False
        else:
            wdf[WATCHLIST_FLAG] = wdf[WATCHLIST_FLAG].fillna(False).astype(bool)
        wdf_merged = wdf.merge(indices, on="date", how="left")
    else:
        wdf_merged = pd.DataFrame()

    # ── Controls ───────────────────────────────────────────────────────────────
    tab_trades, tab_watchlist, tab_combined = st.tabs([
        "📉 Trade Annotations by Regime",
        "📋 Watchlist Flag by Regime",
        "🔀 Combined View",
    ])

    # ── TAB 1: Trade annotations ───────────────────────────────────────────────
    with tab_trades:
        if merged.empty:
            st.info("No merged trade + regime data available.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                regime_type = st.radio(
                    "Regime dimension",
                    ["ATR Distance (position)", "Streak Z-Score (breadth)"],
                    horizontal=True, key="trade_regime_type"
                )
            with c2:
                annotation = st.selectbox(
                    "Annotation to analyse",
                    list(ANNOTATION_COLS.values()),
                    key="trade_annotation_sel"
                )

            ann_col = {v: k for k, v in ANNOTATION_COLS.items()}[annotation]
            regime_pool = REGIME_POSITION_COLS if "ATR" in regime_type else STREAK_ZSCORE_COLS
            available = [c for c in regime_pool if c in merged.columns]

            if not available:
                st.warning("No regime columns found in joined data.")
            else:
                regime_col = st.selectbox("Regime column", available, key="trade_regime_col")

                if "ATR" in regime_type:
                    merged["_regime_bin"] = _bin_atr(merged[regime_col], regime_col)
                else:
                    merged["_regime_bin"] = _bin_zscore(merged[regime_col])

                summary = _agg_by_regime(merged, "_regime_bin", ann_col, annotation)
                summary = summary.dropna(subset=["_regime_bin"])

                st.markdown(f"##### {annotation} rate by `{regime_col}`")
                st.dataframe(summary, use_container_width=True, hide_index=True)

                # Bar chart
                import streamlit as st2
                try:
                    import plotly.express as px
                    fig = px.bar(
                        summary,
                        x="_regime_bin",
                        y=f"{annotation} Rate %",
                        color=f"{annotation} Rate %",
                        color_continuous_scale="RdYlGn_r",
                        text=f"{annotation} Rate %",
                        labels={"_regime_bin": regime_col},
                        height=350,
                    )
                    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e0e0e0", showlegend=False,
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    st.bar_chart(summary.set_index("_regime_bin")[f"{annotation} Rate %"])

                # Return breakdown
                if "pct_return" in merged.columns:
                    st.markdown("##### Avg return by regime bin")
                    ret_grp = (
                        merged.groupby("_regime_bin", observed=True)["pct_return"]
                        .mean().round(2).reset_index()
                    )
                    ret_grp.columns = [regime_col, "Avg Return %"]
                    try:
                        fig2 = px.bar(
                            ret_grp, x=regime_col, y="Avg Return %",
                            color="Avg Return %",
                            color_continuous_scale="RdYlGn",
                            height=300,
                        )
                        fig2.update_layout(
                            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                            font_color="#e0e0e0", coloraxis_showscale=False,
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception:
                        st.bar_chart(ret_grp.set_index(regime_col))

    # ── TAB 2: Watchlist flag ──────────────────────────────────────────────────
    with tab_watchlist:
        if wdf_merged.empty:
            st.info("No watchlist data with regime columns found.")
        else:
            c1w, c2w = st.columns(2)
            with c1w:
                wl_regime_type = st.radio(
                    "Regime dimension",
                    ["ATR Distance (position)", "Streak Z-Score (breadth)"],
                    horizontal=True, key="wl_regime_type"
                )
            regime_pool_w = REGIME_POSITION_COLS if "ATR" in wl_regime_type else STREAK_ZSCORE_COLS
            available_w = [c for c in regime_pool_w if c in wdf_merged.columns]

            if not available_w:
                st.warning("No regime columns in watchlist-joined data.")
            else:
                wl_regime_col = st.selectbox("Regime column", available_w, key="wl_regime_col")

                if "ATR" in wl_regime_type:
                    wdf_merged["_regime_bin"] = _bin_atr(wdf_merged[wl_regime_col], wl_regime_col)
                else:
                    wdf_merged["_regime_bin"] = _bin_zscore(wdf_merged[wl_regime_col])

                summary_w = _agg_by_regime(
                    wdf_merged, "_regime_bin", WATCHLIST_FLAG, "Should Have Traded"
                )
                summary_w = summary_w.dropna(subset=["_regime_bin"])

                st.markdown(f"##### 'Should Have Traded' rate by `{wl_regime_col}`")
                st.dataframe(summary_w, use_container_width=True, hide_index=True)

                try:
                    import plotly.express as px
                    fig_w = px.bar(
                        summary_w,
                        x="_regime_bin",
                        y="Should Have Traded Rate %",
                        color="Should Have Traded Rate %",
                        color_continuous_scale="Blues",
                        text="Should Have Traded Rate %",
                        labels={"_regime_bin": wl_regime_col},
                        height=350,
                    )
                    fig_w.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                    fig_w.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e0e0e0", showlegend=False,
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(fig_w, use_container_width=True)
                except ImportError:
                    st.bar_chart(summary_w.set_index("_regime_bin")["Should Have Traded Rate %"])

    # ── TAB 3: Combined view ───────────────────────────────────────────────────
    with tab_combined:
        st.markdown("##### Regime heatmap — all annotations at once")

        regime_pool_c = REGIME_POSITION_COLS
        available_c = [c for c in regime_pool_c if c in merged.columns]
        if not available_c:
            st.info("No regime columns available for combined view.")
        else:
            combo_col = st.selectbox("Regime column", available_c, key="combo_col")
            merged["_regime_bin"] = _bin_atr(merged[combo_col], combo_col)

            rows = []
            for ann_col, ann_label in ANNOTATION_COLS.items():
                grp = (
                    merged.groupby("_regime_bin", observed=True)[ann_col]
                    .mean().mul(100).round(1)
                )
                grp.name = ann_label
                rows.append(grp)

            if rows:
                heatmap_df = pd.concat(rows, axis=1).reset_index()
                heatmap_df = heatmap_df.dropna(subset=["_regime_bin"])
                heatmap_df = heatmap_df.rename(columns={"_regime_bin": combo_col})

                try:
                    import plotly.express as px
                    fig_h = px.imshow(
                        heatmap_df.set_index(combo_col).T,
                        color_continuous_scale="RdYlGn_r",
                        text_auto=".1f",
                        aspect="auto",
                        height=250,
                        labels={"color": "Flag Rate %"},
                    )
                    fig_h.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        font_color="#e0e0e0",
                    )
                    st.plotly_chart(fig_h, use_container_width=True)
                except ImportError:
                    st.dataframe(heatmap_df, use_container_width=True, hide_index=True)
