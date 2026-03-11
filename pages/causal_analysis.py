"""
pages/causal_analysis.py
========================
Streamlit page for the Causal Setup Analysis module.

Tabs:
  1. A/B Analysis  – setup tag treatment effects on ATR Gain Close
  2. Dose-Response – MA extension effect on ATR Gain Close
  3. Data Explorer – filtered raw data + download
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# ── Make sure utils is importable regardless of working directory ─────────────
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.causal_engine import (
    run_full_analysis,
    load_from_cache,
    cache_exists,
    get_cache_meta,
    VALID_TAGS,
    EXTENSION_COLS,
    ALL_CONFOUNDERS,
    OUTCOME_COL,
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Causal Setup Analysis", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: #1e2530;
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-label { color: #8892a4; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #e2e8f0; font-size: 28px; font-weight: 700; margin-top: 4px; }
    .metric-sub   { color: #8892a4; font-size: 12px; margin-top: 2px; }
    .interpret-box {
        background: #1a2332;
        border-left: 3px solid #3b82f6;
        border-radius: 4px;
        padding: 12px 16px;
        color: #94a3b8;
        font-size: 13px;
        margin-top: 8px;
    }
    .warn-box {
        background: #2a1f14;
        border-left: 3px solid #f59e0b;
        border-radius: 4px;
        padding: 12px 16px;
        color: #d4a76a;
        font-size: 13px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Controls
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ Analysis Controls")

    selected_tags = st.multiselect(
        "Setup Tags to Include",
        options=VALID_TAGS,
        default=VALID_TAGS,
        help="Only setups with at least one of these tags will be included. "
             "Multi-tag setups appear in all matched groups.",
    )

    min_atr_gain = st.slider(
        "Minimum ATR Gain Close",
        min_value=0.0,
        max_value=3.0,
        value=0.0,
        step=0.1,
        help="Filter out setups below this outcome threshold.",
    )

    st.divider()

    # ── Cache status ──────────────────────────────────────────────────────────
    if cache_exists():
        meta = get_cache_meta()
        st.success("✅ Cache found")
        if meta:
            st.caption("Last run: {}".format(meta.get("last_run", "unknown")))
            st.caption("Tags: {}".format(", ".join(meta.get("selected_tags", []))))
            st.caption("Min ATR Gain: {}".format(meta.get("min_atr_gain", 0.0)))
    else:
        st.warning("⚠️ No cache found. Run analysis to generate results.")

    run_btn = st.button(
        "🔄 Re-run Analysis",
        type="primary",
        use_container_width=True,
        help="Re-runs the full analysis with the settings above and overwrites the cache. "
             "This takes a few minutes.",
    )

    st.divider()
    with st.expander("ℹ️ Methodology"):
        st.markdown("""
**Q1 - A/B (Setup Tags)**
Uses **Double Machine Learning (LinearDML)** from Microsoft's EconML library.
Each tag is treated as a binary treatment (this tag vs. all others).
Gradient Boosted models partial out the effect of market confounders before
estimating the causal Average Treatment Effect (ATE) with 95% confidence intervals.

**Q2 - Dose-Response (Extension)**
Uses **Causal Forest DML** to estimate the marginal causal effect of MA extension
at entry on forward ATR Gain, as a continuous dose-response curve.

**Confounders controlled for (21 total):**
ATR distances from 21EMA and 50SMA for QQQ, QQQE, SPY, RSP, IWM (10 features) +
streak z-scores for each (10 features) + 20 Day ADR% as a stock-level volatility
confounder (1 feature). ADR% is included because high-volatility stocks can sustain
greater MA extension without being meaningfully overextended.

**Note:** All setups cleared a ~15% gain threshold at curation. Analysis is
conditional on setup success - estimating *how well* setups work, not *whether* they work.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — file-based cache, only re-runs on button press
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    if not selected_tags:
        st.warning("Please select at least one setup tag.")
        st.stop()
    with st.spinner("Running full causal analysis — this takes a few minutes..."):
        results = run_full_analysis(
            selected_tags=selected_tags,
            min_atr_gain=min_atr_gain,
        )
    st.success("Analysis complete. Results cached to disk.")
    st.session_state["results"] = results

elif "results" not in st.session_state:
    # Load from disk cache on first page load
    if cache_exists():
        with st.spinner("Loading results from cache..."):
            st.session_state["results"] = load_from_cache()
    else:
        st.info(
            "No cached results found. "
            "Configure your settings in the sidebar and click **Re-run Analysis** to generate results."
        )
        st.stop()

results = st.session_state["results"]
df      = results["df"]
summary = results["summary"]
ab      = results["ab"]
dr      = results["dr"]

# ══════════════════════════════════════════════════════════════════════════════
# HEADER METRICS
# ══════════════════════════════════════════════════════════════════════════════

st.title("📐 Causal Setup Analysis")
st.caption(
    f"Setups from {summary['date_min'].strftime('%b %Y')} to "
    f"{summary['date_max'].strftime('%b %Y')} · "
    f"Controlling for 21 features (market conditions + stock volatility)"
)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, "Unique Setups",    f"{summary['n_setups']:,}",              "before tag explosion"),
    (c2, "Analysis Rows",    f"{summary['n_rows']:,}",                "after tag explosion"),
    (c3, "Tags Analyzed",    f"{summary['n_tags']}",                  "setup types"),
    (c4, "Mean ATR Gain",    f"{summary['mean_outcome']:.2f}",        "ATRs"),
    (c5, "Median ATR Gain",  f"{summary['median_outcome']:.2f}",      "ATRs"),
]
for col, label, value, sub in metrics:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INTERPRETATION GUIDE (hideable)
# ══════════════════════════════════════════════════════════════════════════════

with st.expander("📖 How to Read This Page — Interpretation Guide", expanded=False):
    st.markdown("""
### The Core Idea: Causation vs. Correlation

Most trading analysis tells you that two things *move together* — for example,
"21MA pullbacks tend to have higher returns." But that could simply be because
21MA pullbacks happen more often in strong bull markets, and bull markets make
*everything* work. That's correlation driven by a hidden third factor (market conditions).

**Causal inference tries to answer a different question:**
*If I took two identical setups in identical market conditions, but one was a 21MA pullback
and the other was a 50MA pullback — which one would actually perform better?*

To do this, we use the market condition features (how far QQQ/SPY/IWM/RSP/QQQE are from
their MAs, and how long they've been trending) plus stock-level volatility (20 Day ADR%)
to "control for" the environment each setup occurred in. ADR% matters because
high-volatility stocks can sit several ATRs above a key MA and still be in a perfectly
buyable spot — a low-volatility stock at the same extension is genuinely stretched.
Only after neutralizing those differences do we estimate the effect of setup type
or extension. This is why the results here may look different from a simple average.

---

### Tab 1 — A/B Setup Tag Effects

**What it's asking:** Does the *type* of setup (21MA pullback, VCP, EP, etc.) causally affect
how many ATRs the stock gains, after accounting for the market environment it occurred in?

**Average Treatment Effect (ATE):**
Think of this as: "On average, how many additional ATRs does a setup tagged X gain,
compared to a similar setup *not* tagged X, in the same market conditions?"
- A positive ATE (green bar) → this setup type outperforms comparable setups
- A negative ATE (red bar) → this setup type underperforms comparable setups
- An ATE near zero → no meaningful causal effect vs. others

**Confidence Intervals (error bars):**
The error bars show the range where the true effect likely falls (95% of the time).
- Narrow bars = more certainty in the estimate
- Wide bars = less data or high variance — interpret with caution
- If the error bar crosses zero, the effect is **not statistically significant**
  (you cannot rule out that it is just noise)

**Regime Heatmap:**
Shows raw (unadjusted) mean ATR Gain split by whether SPY was above/below its 50SMA.
Brighter green = higher mean gains in that regime. This is *descriptive*, not causal —
use it to understand *when* each setup type tends to show up and perform.

**Violin Chart:**
Shows the full distribution of ATR Gain for each tag — not just the average.
A wide violin means high variance (some huge wins, some small ones).
A tall violin means setups in this group can run very far.
Look for tags with a violin that is both tall *and* has most of its weight in the upper half.

---

### Tab 2 — Dose-Response: MA Extension

**What it's asking:** As a setup becomes more extended from a given MA (meaning price has
moved further away from the MA before you buy), does that *causally* reduce how far it runs?

**The dose-response curve:**
- X-axis = how many ATRs the stock was above the MA at entry (extension)
- Y-axis = the estimated causal effect on ATR Gain *relative to entering right at the MA (zero extension)*
- A downward-sloping curve means: the more extended you buy, the less you gain
- A flat curve means: extension from that MA does not matter much
- An upward curve would mean: more extension actually helps (possible for momentum setups like EP)

**The shaded band:**
This is the 95% confidence interval around the curve. Where the band is wide, there is
less certainty — often because few setups occurred at that extreme extension level.

**The yellow dotted line (P90):**
This marks the 90th percentile of extension in your dataset. Setups to the right of this
line are unusually extended — you have less data there and the estimates are less reliable.
This also visually confirms your prior observation that 90% of good setups are below
a certain extension threshold.

**The scatter points (right axis):**
Raw ATR Gain for individual setups, overlaid for reference. Note these use the right Y-axis
(raw ATR Gain), while the curve uses the left Y-axis (causal effect). They are on different
scales — the scatter is there to show data density, not to align with the curve directly.

**The "All Extensions" comparison bar chart:**
Compares the causal effect at the *median* extension level across all four MAs.
This tells you which MA extension matters most for outcomes. If the 200SMA bar
is much more negative than the 21EMA bar, it means being far from the 200SMA hurts
more than being far from the 21EMA.

---

### What This Analysis Cannot Tell You

- **Whether a setup will work at all.** Every setup in this database already cleared a ~15%
  gain threshold. You are learning about *degree of success* within already-successful setups,
  not about predicting success vs. failure.
- **Future performance.** Market regimes change. An effect that held over your historical
  database may not hold in a structurally different market.
- **Causality with certainty.** We control for the 20 market condition features we have,
  but there may be other factors we have not measured (sector, float, earnings proximity, etc.)
  that still confound the results. The estimates are our best attempt given available data.

---

### Glossary

| Term | Plain English |
|---|---|
| **ATE** | Average Treatment Effect — the average causal impact across all setups |
| **Confounder** | A background variable that affects both the treatment and the outcome (e.g. market regime) |
| **Controlling for** | Mathematically removing the influence of confounders before measuring the effect |
| **Dose-Response** | How the outcome changes as the treatment gets stronger (like increasing a drug dose) |
| **Confidence Interval** | The range the true effect likely falls within — narrower means more certain |
| **Statistically Significant** | The confidence interval does not cross zero — effect is unlikely to be pure noise |
| **DML (Double ML)** | A method that uses ML models to remove confounder effects before causal estimation |
| **Causal Forest** | A method that estimates how the causal effect varies across different conditions |
    """)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📊 A/B — Setup Tag Effects",
    "📈 Dose-Response — MA Extension",
    "🔍 Data Explorer",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: A/B ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    st.subheader("Causal Effect of Setup Tag on ATR Gain Close")
    st.markdown("""
    <div class="interpret-box">
    Each bar shows the estimated <b>Average Treatment Effect (ATE)</b> of being that setup type,
    relative to not being that setup type, after controlling for market conditions.
    A positive ATE means setups with this tag produce higher ATR Gain than comparable setups without it.
    Error bars are 95% confidence intervals. Tags with insufficient data (&lt;30 setups) are shown
    in the table but excluded from the causal estimate.
    </div>
    """, unsafe_allow_html=True)

    ate_table = ab["ate_table"]
    regime_hte = ab["regime_hte"]

    # ── ATE Bar Chart ─────────────────────────────────────────────────────────
    ate_valid = ate_table[ate_table["sufficient_data"] & ate_table["ATE"].notna()].copy()

    if len(ate_valid) > 0:
        fig_ate = go.Figure()

        colors = [
            "#22c55e" if v > 0 else "#ef4444"
            for v in ate_valid["ATE"]
        ]

        fig_ate.add_trace(go.Bar(
            x=ate_valid["tag"],
            y=ate_valid["ATE"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=      ate_valid["CI_upper"] - ate_valid["ATE"],
                arrayminus= ate_valid["ATE"] - ate_valid["CI_lower"],
                color="#64748b",
                thickness=2,
                width=6,
            ),
            marker_color=colors,
            marker_line_color="#0f172a",
            marker_line_width=1,
            text=[f"{v:.2f}" for v in ate_valid["ATE"]],
            textposition="outside",
            textfont=dict(size=11, color="#e2e8f0"),
        ))

        fig_ate.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

        fig_ate.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis=dict(title="Setup Tag", tickfont=dict(size=12)),
            yaxis=dict(title="ATE (ATR units)", gridcolor="#1e2d3d"),
            showlegend=False,
        )
        st.plotly_chart(fig_ate, use_container_width=True)
    else:
        st.info("Not enough data in any tag group (need ≥30 per arm) to compute causal estimates.")

    # ── ATE Summary Table ─────────────────────────────────────────────────────
    st.markdown("#### Full Results Table")
    display_cols = {
        "tag":                    "Tag",
        "n_treated":              "N (this tag)",
        "mean_outcome_treated":   "Mean ATR Gain (this tag)",
        "mean_outcome_control":   "Mean ATR Gain (others)",
        "ATE":                    "Causal ATE",
        "CI_lower":               "CI Lower (95%)",
        "CI_upper":               "CI Upper (95%)",
        "sufficient_data":        "Sufficient Data",
    }
    ate_display = ate_table[list(display_cols.keys())].rename(columns=display_cols)

    st.dataframe(
        ate_display.style
            .format({
                "Mean ATR Gain (this tag)": "{:.2f}",
                "Mean ATR Gain (others)":   "{:.2f}",
                "Causal ATE":               lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                "CI Lower (95%)":           lambda x: f"{x:.2f}" if pd.notna(x) else "—",
                "CI Upper (95%)":           lambda x: f"{x:.2f}" if pd.notna(x) else "—",
            })
            .applymap(
                lambda v: "color: #22c55e" if isinstance(v, float) and v > 0
                          else ("color: #ef4444" if isinstance(v, float) and v < 0 else ""),
                subset=["Causal ATE"]
            ),
        use_container_width=True,
        hide_index=True,
    )

    # ── Regime Breakdown ──────────────────────────────────────────────────────
    st.markdown("#### Mean ATR Gain by Tag × SPY Regime")
    st.markdown("""
    <div class="interpret-box">
    Raw (unadjusted) mean outcomes split by whether SPY was above or below its 50SMA on the setup date.
    This is descriptive — use it alongside the causal estimates above to understand regime interaction.
    </div>
    """, unsafe_allow_html=True)

    if not regime_hte.empty:
        fig_heatmap = px.density_heatmap(
            regime_hte,
            x="spy_regime",
            y="tag",
            z="mean_outcome",
            color_continuous_scale="RdYlGn",
            text_auto=".2f",
        )
        fig_heatmap.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_colorbar=dict(title="Mean ATR Gain"),
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Raw Distribution Violin ───────────────────────────────────────────────
    st.markdown("#### ATR Gain Distribution by Tag (Raw)")
    raw_dist = ab["raw_dist"]
    if not raw_dist.empty:
        fig_violin = go.Figure()
        for tag in sorted(raw_dist["tag"].unique()):
            vals = raw_dist[raw_dist["tag"] == tag][OUTCOME_COL]
            fig_violin.add_trace(go.Violin(
                x=[tag] * len(vals),
                y=vals,
                name=tag,
                box_visible=True,
                meanline_visible=True,
                points=False,
            ))
        fig_violin.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Setup Tag",
            yaxis_title="ATR Gain Close",
            showlegend=False,
            yaxis=dict(gridcolor="#1e2d3d"),
        )
        st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown("""
    <div class="warn-box">
    <b>Selection bias note:</b> All setups in this database cleared a ~15% gain threshold at curation.
    ATEs reflect heterogeneity in <i>how well</i> setups work given they already worked —
    not the probability of a setup working in the first place.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: DOSE-RESPONSE
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("Causal Dose-Response: MA Extension → ATR Gain Close")
    st.markdown("""
    <div class="interpret-box">
    Each chart shows the estimated <b>marginal causal effect</b> on ATR Gain of increasing
    extension from that MA at entry, after controlling for setup type and market conditions.
    The effect is relative to a baseline of zero extension (sitting right on the MA).
    The shaded band is a 95% confidence interval. Raw data points are overlaid as a scatter.
    </div>
    """, unsafe_allow_html=True)

    curves  = dr["curves"]
    scatter = dr["scatter"]
    summ_dr = dr["summary"]

    if not curves:
        st.warning("Dose-response models could not be fit. Check that extension columns exist and have sufficient data.")
    else:
        # ── Summary table ──────────────────────────────────────────────────────
        if not summ_dr.empty:
            st.markdown("#### Extension Feature Summary")
            summ_display = summ_dr.rename(columns={
                "extension_col":  "MA Extension",
                "n":              "N",
                "mean_extension": "Mean (ATR units)",
                "p10":            "P10",
                "p50":            "P50",
                "p90":            "P90",
                "corr_raw":       "Raw Correlation with Outcome",
            })
            st.dataframe(
                summ_display.style.format({
                    "Mean (ATR units)":              "{:.2f}",
                    "P10":                           "{:.2f}",
                    "P50":                           "{:.2f}",
                    "P90":                           "{:.2f}",
                    "Raw Correlation with Outcome":  "{:.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Selector for which extension to display ────────────────────────────
        ext_options = list(curves.keys())
        selected_ext = st.selectbox(
            "Select MA Extension to View",
            options=ext_options,
            index=0,
        )

        curve_df  = curves[selected_ext]
        scatter_df = scatter.get(selected_ext, pd.DataFrame())

        # ── Dose-response chart ────────────────────────────────────────────────
        fig_dr = go.Figure()

        # CI band
        fig_dr.add_trace(go.Scatter(
            x=pd.concat([curve_df["extension_val"], curve_df["extension_val"][::-1]]),
            y=pd.concat([curve_df["ci_upper"], curve_df["ci_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(59, 130, 246, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI",
            showlegend=True,
        ))

        # Effect line
        fig_dr.add_trace(go.Scatter(
            x=curve_df["extension_val"],
            y=curve_df["effect"],
            mode="lines",
            line=dict(color="#3b82f6", width=2.5),
            name="Causal Effect",
        ))

        # Zero reference
        fig_dr.add_hline(y=0, line_dash="dash", line_color="#475569", line_width=1)

        # Raw scatter overlay (sample up to 500 points for readability)
        if not scatter_df.empty:
            sample = scatter_df.sample(min(500, len(scatter_df)), random_state=42)
            fig_dr.add_trace(go.Scatter(
                x=sample["extension_val"],
                y=sample["outcome"],
                mode="markers",
                marker=dict(
                    color="#94a3b8",
                    size=4,
                    opacity=0.35,
                ),
                name="Raw Setups",
                yaxis="y2",
            ))

        # P90 vertical reference line
        if not summ_dr.empty:
            row = summ_dr[summ_dr["extension_col"] == selected_ext]
            if not row.empty:
                p90 = float(row["p90"].iloc[0])
                fig_dr.add_vline(
                    x=p90,
                    line_dash="dot",
                    line_color="#f59e0b",
                    annotation_text=f"P90 ({p90:.1f})",
                    annotation_font_color="#f59e0b",
                    annotation_position="top right",
                )

        fig_dr.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=460,
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(title=f"{selected_ext} (ATR units from MA)", gridcolor="#1e2d3d"),
            yaxis=dict(title="Marginal Causal Effect on ATR Gain", gridcolor="#1e2d3d"),
            yaxis2=dict(
                title="Raw ATR Gain (scatter)",
                overlaying="y",
                side="right",
                showgrid=False,
                color="#64748b",
            ),
            legend=dict(
                bgcolor="rgba(15,23,42,0.8)",
                bordercolor="#2d3748",
                borderwidth=1,
            ),
        )
        st.plotly_chart(fig_dr, use_container_width=True)

        # ── All-extensions comparison + slope table ───────────────────────────
        st.markdown("#### All Extensions — Slope Summary")
        comparison_rows = []
        for ext_col, cdf in curves.items():
            if summ_dr.empty:
                continue
            row = summ_dr[summ_dr["extension_col"] == ext_col]
            if row.empty:
                continue
            p50  = float(row["p50"].iloc[0])
            p10  = float(row["p10"].iloc[0])
            p90  = float(row["p90"].iloc[0])
            # Find closest curve point to median
            idx  = (cdf["extension_val"] - p50).abs().idxmin()
            effect_at_median = float(cdf.loc[idx, "effect"])
            ci_lower_med     = float(cdf.loc[idx, "ci_lower"])
            ci_upper_med     = float(cdf.loc[idx, "ci_upper"])

            # Slope = effect at median / median extension (since curve passes near 0 at 0)
            # Interpreted as: ATR gain lost per 1 ATR of additional extension
            slope = (effect_at_median / p50) if p50 != 0 else np.nan

            comparison_rows.append({
                "MA Extension":            ext_col,
                "Median Extension (ATRs)": p50,
                "Effect at Median":        effect_at_median,
                "CI Lower":                ci_lower_med,
                "CI Upper":                ci_upper_med,
                "Slope (ATR gain / ATR ext.)": slope,
            })

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)

            # ── Slope bar chart ───────────────────────────────────────────────
            fig_comp = go.Figure()
            for _, r in comp_df.iterrows():
                color = "#22c55e" if r["Effect at Median"] > 0 else "#ef4444"
                fig_comp.add_trace(go.Bar(
                    name=r["MA Extension"],
                    x=[r["MA Extension"]],
                    y=[r["Effect at Median"]],
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=[r["CI Upper"] - r["Effect at Median"]],
                        arrayminus=[r["Effect at Median"] - r["CI Lower"]],
                    ),
                    marker_color=color,
                    showlegend=False,
                ))
            fig_comp.add_hline(y=0, line_dash="dash", line_color="#475569")
            fig_comp.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0f172a",
                plot_bgcolor="#0f172a",
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis=dict(title="Causal Effect at Median Extension (ATRs)", gridcolor="#1e2d3d"),
                xaxis=dict(title="MA Extension Feature"),
                barmode="group",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # ── Slope summary table ───────────────────────────────────────────
            st.markdown("##### Slope Table — ATR Gain Lost per ATR of Extension")
            slope_display = comp_df[[
                "MA Extension",
                "Median Extension (ATRs)",
                "Effect at Median",
                "CI Lower",
                "CI Upper",
                "Slope (ATR gain / ATR ext.)",
            ]].copy()

            def color_slope(val):
                if pd.isna(val):
                    return ""
                return "color: #ef4444" if val < 0 else "color: #22c55e"

            st.dataframe(
                slope_display.style
                    .format({
                        "Median Extension (ATRs)":         "{:.2f}",
                        "Effect at Median":                "{:.3f}",
                        "CI Lower":                        "{:.3f}",
                        "CI Upper":                        "{:.3f}",
                        "Slope (ATR gain / ATR ext.)":     "{:.3f}",
                    })
                    .applymap(color_slope, subset=["Slope (ATR gain / ATR ext.)"]),
                use_container_width=True,
                hide_index=True,
            )

            # ── Plain-English interpretation ──────────────────────────────────
            st.markdown("##### How to Read the Slope Table")
            interp_lines = []
            for _, r in comp_df.iterrows():
                ma   = r["MA Extension"]
                med  = r["Median Extension (ATRs)"]
                eff  = r["Effect at Median"]
                slp  = r["Slope (ATR gain / ATR ext.)"]
                if pd.isna(slp):
                    continue
                direction = "costs" if slp < 0 else "adds"
                interp_lines.append(
                    "- **{}:** median setup is {:.2f} ATRs above this MA at entry, "
                    "causing an estimated **{:.2f} ATR** change in forward gain at that distance. "
                    "The slope is **{:.3f}** — meaning each additional ATR of extension {} "
                    "approximately **{:.2f} ATRs** of forward gain, controlling for market conditions.".format(
                        ma, med, eff, slp, direction, abs(slp)
                    )
                )
            st.markdown("\n".join(interp_lines))

            st.markdown("""
            <div class="interpret-box">
            <b>How to use this practically:</b> The slope is the single most actionable number
            on this page. It tells you the causal cost (or benefit) of buying one ATR more extended
            from each MA. A slope of -0.22 on the 200SMA means that if two otherwise identical
            setups exist and one is 2 ATRs more extended above the 200SMA, the more extended one
            is expected to run ~0.44 ATRs less. Use the slopes to compare which MA's extension
            matters most — the steeper the negative slope, the more that MA's proximity at entry
            affects your outcome.
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
    <b>Interpretation guide:</b> A negative causal effect at high extension values means that —
    <i>controlling for market conditions and setup type</i> — more extended entries causally
    reduce forward ATR Gain. This is distinct from raw correlation, which could be driven by
    extended setups appearing more often in bull markets.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: DATA EXPLORER
# ─────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("Setup Data Explorer")

    # ── Filters ───────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        tag_filter = st.multiselect(
            "Filter by Tag",
            options=sorted(df["tag"].unique()),
            default=[],
            placeholder="All tags",
        )
    with col_f2:
        date_range = st.date_input(
            "Date Range",
            value=(df["Buy Date"].min().date(), df["Buy Date"].max().date()),
        )
    with col_f3:
        min_gain_filter = st.number_input(
            "Min ATR Gain Close",
            min_value=0.0,
            value=0.0,
            step=0.5,
        )

    filtered = df.copy()
    if tag_filter:
        filtered = filtered[filtered["tag"].isin(tag_filter)]
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["Buy Date"].dt.date >= date_range[0]) &
            (filtered["Buy Date"].dt.date <= date_range[1])
        ]
    filtered = filtered[filtered[OUTCOME_COL] >= min_gain_filter]

    # ── Per-tag summary ───────────────────────────────────────────────────────
    st.markdown(f"**{len(filtered):,} rows** after filters")

    tag_summary = (
        filtered.groupby("tag")[OUTCOME_COL]
        .agg(
            N="count",
            Mean="mean",
            Median="median",
            Std="std",
            P25=lambda x: x.quantile(0.25),
            P75=lambda x: x.quantile(0.75),
            P90=lambda x: x.quantile(0.90),
        )
        .reset_index()
        .rename(columns={"tag": "Tag"})
    )
    st.dataframe(
        tag_summary.style.format({c: "{:.2f}" for c in ["Mean", "Median", "Std", "P25", "P75", "P90"]}),
        use_container_width=True,
        hide_index=True,
    )

    # ── Raw data table ────────────────────────────────────────────────────────
    with st.expander("Show Raw Rows"):
        display_cols_raw = (
            ["Stock", "Buy Date", "tag", OUTCOME_COL]
            + [c for c in EXTENSION_COLS if c in filtered.columns]
        )
        available = [c for c in display_cols_raw if c in filtered.columns]
        st.dataframe(
            filtered[available].sort_values("Buy Date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # ── Download ──────────────────────────────────────────────────────────────
    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        label="⬇ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="causal_analysis_filtered.csv",
        mime="text/csv",
    )
