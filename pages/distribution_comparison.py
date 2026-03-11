"""
pages/distribution_comparison.py
=================================
Distribution Comparison Tool — two modes:

  Mode 1: CSV Field Comparison
    Compare any numeric field from any two CSV files.
    Supports multiple filters per dataset (categorical or numeric range).
    Statistical output: histograms, descriptive stats, KS / Mann-Whitney / t-test, ECDF.

  Mode 2: Regime Bucket Comparison (Heatmap)
    Define two market regime buckets using typed numeric range filters.
    Output: combined heatmap of mean OR median ATR Gain by setup tag,
    with N count overlay — Bucket A | Bucket B | Difference.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.distributions.empirical_distribution import ECDF

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.causal_engine import (
    load_from_cache,
    cache_exists,
    REGIME_POSITION_COLS,
    STREAK_ZSCORE_COLS,
    OUTCOME_COL,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Distribution Comparison", layout="wide")
st.title("📊 Distribution Comparison Tool")

DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")

# ── Shared helpers ─────────────────────────────────────────────────────────────

@st.cache_data
def load_csv(path):
    return pd.read_csv(path, low_memory=False)

def list_csv_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

def get_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def get_categorical_columns(df):
    return df.select_dtypes(exclude=[np.number]).columns.tolist()

@st.cache_data
def compute_ecdf(series_values, max_points=5000):
    s = pd.Series(series_values)
    if len(s) > max_points:
        s = s.sample(max_points, random_state=1)
    ecdf_fn = ECDF(s)
    x_vals = np.linspace(s.min(), s.max(), 500)
    y_vals = ecdf_fn(x_vals)
    return x_vals, y_vals


def num_input_range(label, col_min, col_max, key_lo, key_hi):
    """
    Two side-by-side number inputs replacing a slider.
    Returns (lo, hi) as floats.
    """
    c1, c2 = st.columns(2)
    with c1:
        lo = st.number_input(
            "{} — min".format(label),
            value=float(col_min),
            min_value=float(col_min),
            max_value=float(col_max),
            step=0.01,
            format="%.3f",
            key=key_lo,
        )
    with c2:
        hi = st.number_input(
            "{} — max".format(label),
            value=float(col_max),
            min_value=float(col_min),
            max_value=float(col_max),
            step=0.01,
            format="%.3f",
            key=key_hi,
        )
    return lo, hi


# =============================================================================
# SHARED STATISTICAL COMPARISON PIPELINE (CSV mode)
# =============================================================================

def run_comparison(seriesA, seriesB, label_a, label_b):
    seriesA = seriesA.dropna().reset_index(drop=True)
    seriesB = seriesB.dropna().reset_index(drop=True)

    if len(seriesA) < 5 or len(seriesB) < 5:
        st.warning("Not enough data in one or both series (need ≥5 values each).")
        return

    plt.style.use("dark_background")

    # ── Histograms ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(seriesA, bins=40, color="#3b82f6", edgecolor="#0f172a", alpha=0.85)
        ax.set_xlabel(label_a, color="#94a3b8")
        ax.set_ylabel("Count", color="#94a3b8")
        ax.set_title("Distribution A: {}".format(label_a), color="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#1e2530")
        fig.patch.set_facecolor("#0f172a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(seriesB, bins=40, color="#f59e0b", edgecolor="#0f172a", alpha=0.85)
        ax.set_xlabel(label_b, color="#94a3b8")
        ax.set_ylabel("Count", color="#94a3b8")
        ax.set_title("Distribution B: {}".format(label_b), color="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#1e2530")
        fig.patch.set_facecolor("#0f172a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")
        st.pyplot(fig)
        plt.close(fig)

    # ── Descriptive stats ──────────────────────────────────────────────────────
    st.markdown("#### Descriptive Statistics")

    def fmt_val(v):
        if isinstance(v, (int, np.integer)):
            return "{:,}".format(int(v))
        try:
            return "{:.3f}".format(float(v))
        except Exception:
            return str(v)

    stats_rows = [
        ("N",      len(seriesA),              len(seriesB)),
        ("Mean",   seriesA.mean(),             seriesB.mean()),
        ("Median", seriesA.median(),           seriesB.median()),
        ("Std Dev",seriesA.std(),              seriesB.std()),
        ("P10",    seriesA.quantile(0.10),     seriesB.quantile(0.10)),
        ("P25",    seriesA.quantile(0.25),     seriesB.quantile(0.25)),
        ("P75",    seriesA.quantile(0.75),     seriesB.quantile(0.75)),
        ("P90",    seriesA.quantile(0.90),     seriesB.quantile(0.90)),
        ("Min",    seriesA.min(),              seriesB.min()),
        ("Max",    seriesA.max(),              seriesB.max()),
    ]
    desc_df = pd.DataFrame(stats_rows, columns=["Stat", label_a, label_b])
    desc_df["B − A"] = desc_df[label_b] - desc_df[label_a]
    desc_df[label_a]  = desc_df[label_a].apply(fmt_val)
    desc_df[label_b]  = desc_df[label_b].apply(fmt_val)
    desc_df["B − A"]  = desc_df["B − A"].apply(fmt_val)
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

    # ── Statistical tests ──────────────────────────────────────────────────────
    st.markdown("#### Statistical Tests")
    col_tbl, col_exp = st.columns([1, 1])

    ks = stats.ks_2samp(seriesA, seriesB)
    mw = stats.mannwhitneyu(seriesA, seriesB, alternative="two-sided")
    tt = stats.ttest_ind(seriesA, seriesB, equal_var=False)

    results_df = pd.DataFrame({
        "Test":                   ["KS Test", "Mann-Whitney U", "Welch t-test"],
        "Statistic":              [round(ks.statistic, 4), round(float(mw.statistic), 1), round(tt.statistic, 4)],
        "p-value":                [round(ks.pvalue, 4),   round(mw.pvalue, 4),    round(tt.pvalue, 4)],
        "Significant (p<0.05)":   [ks.pvalue < 0.05,      mw.pvalue < 0.05,       tt.pvalue < 0.05],
    })

    with col_tbl:
        st.dataframe(results_df, use_container_width=True, hide_index=True)

    with col_exp:
        with st.expander("📖 How to interpret these tests", expanded=False):
            st.markdown("""
**KS Test** — Compares the *entire shape* of both distributions.
Large statistic + small p-value = distributions differ somewhere (mean, spread, tails).
Does not tell you *where* they differ.

**Mann-Whitney U** — Non-parametric. Asks: does one distribution tend to produce *larger values*?
Best choice for skewed data like ATR Gain.

**Welch t-test** — Tests specifically for a difference in *means*. Sensitive to outliers.

**Reading together:**
- All three significant → strong difference in both shape and central tendency
- Only KS → shapes differ but means may be similar
- Only t-test → means differ but overall shape is similar
- None → no strong evidence of difference
            """)

    # ── ECDF ──────────────────────────────────────────────────────────────────
    st.markdown("#### ECDF Comparison")
    col_e1, col_e2 = st.columns(2)

    xA, yA = compute_ecdf(seriesA.values)
    xB, yB = compute_ecdf(seriesB.values)

    with col_e1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xA, yA, label="A: {}".format(label_a), color="#3b82f6", linewidth=2)
        ax.plot(xB, yB, label="B: {}".format(label_b), color="#f59e0b", linewidth=2, linestyle="--")
        ax.axvline(x=0, color="#475569", linestyle=":", linewidth=1)
        ax.set_xlabel("Value", color="#94a3b8")
        ax.set_ylabel("Cumulative Probability", color="#94a3b8")
        ax.set_title("ECDF: A vs B", color="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#1e2530")
        fig.patch.set_facecolor("#0f172a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")
        ax.legend(facecolor="#1e2530", labelcolor="#e2e8f0", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📖 How to read the ECDF", expanded=False):
            st.markdown("""
- Shows what fraction of values fall *at or below* each x-axis value
- Curve shifted **rightward** = larger values overall
- Curve shifted **leftward** = smaller values overall
- Large vertical gaps = strong differences at that value
- Crossing lines = one distribution is better at some values, worse at others
            """)

    with col_e2:
        min_len = min(len(seriesA), len(seriesB))
        delta = pd.Series(seriesA.iloc[:min_len].values - seriesB.iloc[:min_len].values)
        x_delta, y_delta = compute_ecdf(delta.values)
        frac_below = float((delta < 0).mean())

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_delta, y_delta, color="#a78bfa", linewidth=2, label="ECDF of A − B")
        ax.axvline(x=0, color="#f59e0b", linestyle="--", linewidth=1.5, label="Zero (A = B)")
        ax.text(0.05, 0.85,
                "A < B: {:.1f}%\nA > B: {:.1f}%".format(frac_below * 100, (1 - frac_below) * 100),
                transform=ax.transAxes, color="#e2e8f0", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="#1e2530", alpha=0.8))
        ax.set_xlabel("A − B", color="#94a3b8")
        ax.set_ylabel("Cumulative Probability", color="#94a3b8")
        ax.set_title("ECDF of Delta (A − B)", color="#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        ax.set_facecolor("#1e2530")
        fig.patch.set_facecolor("#0f172a")
        ax.grid(True, linestyle="--", alpha=0.3, color="#475569")
        ax.legend(facecolor="#1e2530", labelcolor="#e2e8f0", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("📖 How to read the Delta ECDF", expanded=False):
            st.markdown("""
- Plots the distribution of **(A minus B)** computed pairwise
- Curve mostly **left of zero** → A tends to be *smaller* than B
- Curve mostly **right of zero** → A tends to be *larger* than B
- The annotation shows % of paired differences where A < B vs A > B
- Wide spread → high variability in how A and B differ case by case
            """)


# =============================================================================
# MULTI-FILTER BUILDER (used by CSV mode)
# =============================================================================

def build_csv_filters(df, key_prefix):
    """
    Lets user add multiple filters to a dataframe.
    Each filter can be categorical (multiselect) or numeric (range inputs).
    Returns the filtered dataframe.
    """
    st.markdown("**Filters**")

    # Track how many filter rows the user has added
    n_key = "{}_n_filters".format(key_prefix)
    if n_key not in st.session_state:
        st.session_state[n_key] = 0

    col_add, col_clear = st.columns([1, 1])
    with col_add:
        if st.button("＋ Add filter", key="{}_add".format(key_prefix)):
            st.session_state[n_key] += 1
    with col_clear:
        if st.button("✕ Clear all filters", key="{}_clear".format(key_prefix)):
            st.session_state[n_key] = 0

    out = df.copy()
    all_cols = df.columns.tolist()

    for i in range(st.session_state[n_key]):
        fk = "{}_f{}".format(key_prefix, i)
        with st.container():
            fc1, fc2 = st.columns([1, 2])
            with fc1:
                chosen_col = st.selectbox(
                    "Column #{}".format(i + 1),
                    options=["— select —"] + all_cols,
                    key="{}_col".format(fk),
                )
            if chosen_col == "— select —" or chosen_col not in df.columns:
                continue

            col_data = df[chosen_col].dropna()
            is_numeric = pd.api.types.is_numeric_dtype(col_data)

            with fc2:
                if is_numeric:
                    lo, hi = num_input_range(
                        chosen_col,
                        col_data.min(), col_data.max(),
                        "{}_lo".format(fk),
                        "{}_hi".format(fk),
                    )
                    out = out[(out[chosen_col] >= lo) & (out[chosen_col] <= hi)]
                else:
                    unique_vals = sorted(col_data.unique().tolist(), key=str)
                    selected_vals = st.multiselect(
                        "Values to include",
                        options=unique_vals,
                        default=[],
                        key="{}_vals".format(fk),
                    )
                    if selected_vals:
                        out = out[out[chosen_col].isin(selected_vals)]

    return out


# =============================================================================
# MODE TOGGLE
# =============================================================================

mode = st.radio(
    "Comparison Mode",
    options=["📁 CSV Field Comparison", "🌐 Regime Bucket Heatmap"],
    horizontal=True,
    help=(
        "CSV mode: compare any two numeric fields from any CSV files with custom filters. "
        "Regime Heatmap: define two market regime buckets and compare mean/median ATR Gain "
        "by setup tag in a combined heatmap."
    ),
)

st.divider()

# =============================================================================
# MODE 1: CSV FIELD COMPARISON
# =============================================================================

if mode == "📁 CSV Field Comparison":

    csv_files = list_csv_files()
    if not csv_files:
        st.warning("No CSV files found in the data/ directory.")
        st.stop()

    colA, colB = st.columns(2)

    with colA:
        st.subheader("🔵 Distribution A")
        datasetA = st.selectbox("Dataset A", csv_files, key="datasetA")
        dfA_raw  = load_csv(os.path.join(DATA_DIR, datasetA))
        dfA      = build_csv_filters(dfA_raw, "csvA")
        st.caption("{:,} rows after filters".format(len(dfA)))
        fieldA   = st.selectbox("Field to compare (numeric)", get_numeric_columns(dfA), key="fieldA")

    with colB:
        st.subheader("🟡 Distribution B")
        datasetB = st.selectbox("Dataset B", csv_files, key="datasetB")
        dfB_raw  = load_csv(os.path.join(DATA_DIR, datasetB))
        dfB      = build_csv_filters(dfB_raw, "csvB")
        st.caption("{:,} rows after filters".format(len(dfB)))
        fieldB   = st.selectbox("Field to compare (numeric)", get_numeric_columns(dfB), key="fieldB")

    if st.button("▶ Run Comparison", type="primary", key="csv_run"):
        st.session_state["dist_seriesA"] = dfA[fieldA].dropna()
        st.session_state["dist_seriesB"] = dfB[fieldB].dropna()
        st.session_state["dist_labelA"]  = "{} / {} (n={:,})".format(datasetA, fieldA, len(dfA))
        st.session_state["dist_labelB"]  = "{} / {} (n={:,})".format(datasetB, fieldB, len(dfB))
        st.session_state["dist_mode"]    = "csv"

    if st.session_state.get("dist_mode") == "csv" and st.session_state.get("dist_seriesA") is not None:
        st.divider()
        run_comparison(
            st.session_state["dist_seriesA"],
            st.session_state["dist_seriesB"],
            st.session_state["dist_labelA"],
            st.session_state["dist_labelB"],
        )


# =============================================================================
# MODE 2: REGIME BUCKET HEATMAP
# =============================================================================

elif mode == "🌐 Regime Bucket Heatmap":

    if not cache_exists():
        st.warning(
            "No causal analysis cache found. "
            "Run the analysis first on the **Causal Analysis** page."
        )
        st.stop()

    @st.cache_data
    def get_setup_df():
        return load_from_cache()["df"]

    df = get_setup_df()
    all_regime_cols = [c for c in (REGIME_POSITION_COLS + STREAK_ZSCORE_COLS) if c in df.columns]

    st.markdown(
        "Define two market regime buckets. The heatmap shows **mean or median ATR Gain Close** "
        "by setup tag for each bucket side by side, plus the difference (B − A)."
    )

    # ── Metric toggle ─────────────────────────────────────────────────────────
    metric_choice = st.radio(
        "Show metric",
        options=["Mean", "Median"],
        horizontal=True,
        key="heatmap_metric",
    )
    metric_fn = "mean" if metric_choice == "Mean" else "median"

    # ── Shared base filters ───────────────────────────────────────────────────
    with st.expander("⚙️ Shared Filters (applied to both buckets)", expanded=True):
        sh1, sh2, sh3 = st.columns(3)
        with sh1:
            shared_tags = st.multiselect(
                "Tags (blank = all)",
                options=sorted(df["tag"].unique()),
                default=[],
                key="shared_tags",
            )
        with sh2:
            date_min = df["Buy Date"].min().date()
            date_max = df["Buy Date"].max().date()
            date_range = st.date_input(
                "Date Range",
                value=(date_min, date_max),
                key="shared_dates",
            )
        with sh3:
            min_gain = st.number_input(
                "Min ATR Gain Close",
                min_value=0.0, value=0.0, step=0.5,
                key="shared_min_gain",
            )

    base_df = df.copy()
    if shared_tags:
        base_df = base_df[base_df["tag"].isin(shared_tags)]
    if len(date_range) == 2:
        base_df = base_df[
            (base_df["Buy Date"].dt.date >= date_range[0]) &
            (base_df["Buy Date"].dt.date <= date_range[1])
        ]
    base_df = base_df[base_df[OUTCOME_COL] >= min_gain]
    st.caption("Shared base: {:,} rows ({:,} unique setups)".format(
        len(base_df),
        base_df.drop_duplicates(subset=["Stock", "Buy Date"]).shape[0],
    ))

    # ── Bucket builder ────────────────────────────────────────────────────────
    def build_regime_bucket(label, key_prefix, base):
        st.markdown("### {}".format(label))

        extra_tags = st.multiselect(
            "Further filter by tag (optional)",
            options=sorted(base["tag"].unique()),
            default=[],
            key="{}_tags".format(key_prefix),
        )

        selected_cols = st.multiselect(
            "Regime conditions to filter on",
            options=all_regime_cols,
            default=[],
            key="{}_cols".format(key_prefix),
            help="Positive ATR distance = index above that MA. Z-score > 0 = trending above longer than average.",
        )

        out = base.copy()
        if extra_tags:
            out = out[out["tag"].isin(extra_tags)]

        if selected_cols:
            for col_name in selected_cols:
                col_min = round(float(base[col_name].min()), 3)
                col_max = round(float(base[col_name].max()), 3)
                lo, hi  = num_input_range(
                    col_name, col_min, col_max,
                    "{}_{}_lo".format(key_prefix, col_name),
                    "{}_{}_hi".format(key_prefix, col_name),
                )
                out = out[(out[col_name] >= lo) & (out[col_name] <= hi)]

        n_unique = out.drop_duplicates(subset=["Stock", "Buy Date"]).shape[0]
        st.success("{:,} rows · {:,} unique setups".format(len(out), n_unique))
        return out

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        b1_df = build_regime_bucket("🔵 Bucket A", "b1", base_df)
    with col_b2:
        b2_df = build_regime_bucket("🟡 Bucket B", "b2", base_df)

    if st.button("▶ Build Heatmap", type="primary", key="regime_run"):
        st.session_state["heatmap_b1"] = b1_df
        st.session_state["heatmap_b2"] = b2_df

    # ── Render heatmap ────────────────────────────────────────────────────────
    if "heatmap_b1" in st.session_state and st.session_state["heatmap_b1"] is not None:

        b1  = st.session_state["heatmap_b1"]
        b2  = st.session_state["heatmap_b2"]
        mfn = metric_fn   # use live radio value so toggle updates without re-run

        st.divider()
        st.markdown("#### {} ATR Gain Close by Setup Tag — Bucket A vs Bucket B".format(metric_choice))

        # Aggregate per tag
        def agg_tag(bdf, suffix):
            grp = bdf.groupby("tag")[OUTCOME_COL].agg(
                **{
                    "val_{}".format(suffix):   mfn,
                    "n_{}".format(suffix):     "count",
                }
            ).reset_index()
            return grp

        a_agg = agg_tag(b1, "A")
        b_agg = agg_tag(b2, "B")

        merged = pd.merge(a_agg, b_agg, on="tag", how="outer")
        merged["val_A"]  = merged["val_A"].fillna(0)
        merged["val_B"]  = merged["val_B"].fillna(0)
        merged["n_A"]    = merged["n_A"].fillna(0).astype(int)
        merged["n_B"]    = merged["n_B"].fillna(0).astype(int)
        merged["diff"]   = merged["val_B"] - merged["val_A"]
        merged = merged.sort_values("diff", ascending=False).reset_index(drop=True)

        tags = merged["tag"].tolist()

        # Build vertical heatmap: tags as rows, Bucket A | Bucket B as columns
        z_vals = np.array([merged["val_A"].values, merged["val_B"].values]).T  # shape: (n_tags, 2)

        text_ann = []
        for i, tag in enumerate(tags):
            row = []
            for j, (val_col, n_col) in enumerate([("val_A", "n_A"), ("val_B", "n_B")]):
                v = merged[val_col].iloc[i]
                n = merged[n_col].iloc[i]
                row.append("{:.2f}<br><sub>n={}</sub>".format(float(v), int(float(n))))
            text_ann.append(row)

        fig_heat = go.Figure(data=go.Heatmap(
            z=z_vals,
            x=["Bucket A", "Bucket B"],
            y=tags,
            text=text_ann,
            texttemplate="%{text}",
            textfont=dict(size=11, color="white"),
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(
                title=dict(
                    text="{} ATR Gain".format(metric_choice),
                    font=dict(color="#94a3b8"),
                ),
                tickfont=dict(color="#94a3b8"),
            ),
        ))

        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            height=max(300, len(tags) * 50 + 100),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(title="Bucket", tickfont=dict(size=12), side="bottom"),
            yaxis=dict(title="Setup Tag", tickfont=dict(size=12), autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── Summary table ─────────────────────────────────────────────────────
        with st.expander("📋 Full Summary Table", expanded=False):
            summary_tbl = merged.rename(columns={
                "tag":   "Tag",
                "val_A": "{} ATR Gain — Bucket A".format(metric_choice),
                "n_A":   "N — Bucket A",
                "val_B": "{} ATR Gain — Bucket B".format(metric_choice),
                "n_B":   "N — Bucket B",
                "diff":  "Difference (B − A)",
            })
            st.dataframe(
                summary_tbl.style.format({
                    "{} ATR Gain — Bucket A".format(metric_choice): "{:.3f}",
                    "{} ATR Gain — Bucket B".format(metric_choice): "{:.3f}",
                    "Difference (B − A)":                            "{:+.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # ── Quick interpretation ──────────────────────────────────────────────

        st.caption(
            "N shown in each cell — treat thin cells (low N) with caution. "
            "Brighter green = higher {} ATR Gain for that tag/bucket combination.".format(metric_choice.lower())
        )
