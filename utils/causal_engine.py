"""
causal_engine.py
================
All data prep, model fitting, and results generation for the
Causal Analysis module.

Two analyses:
  Q1 - A/B: Does setup TAG affect ATR Gain Close, controlling for market conditions?
  Q2 - Dose-Response: Does MA extension at entry affect ATR Gain Close?

Caching:
  Results are saved to data/causal_cache/ as parquet/json files.
  The Streamlit page loads from cache on startup and only re-runs
  when explicitly requested via button.

Dependencies:
    pip install econml scikit-learn pandas numpy plotly joblib pyarrow
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from typing import List, Dict, Optional, Tuple

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from econml.dml import LinearDML, CausalForestDML

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this file so they work from any working directory
_HERE     = Path(__file__).resolve().parent        # utils/
_ROOT     = _HERE.parent                           # trading_app/
DATA_DIR  = _ROOT / "data"
CACHE_DIR = _ROOT / "data" / "causal_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "buy_details_with_indices.csv"

# ── Cache file paths ──────────────────────────────────────────────────────────
CACHE_SUMMARY_FILE = CACHE_DIR / "summary.json"
CACHE_AB_FILE      = CACHE_DIR / "ab_results.parquet"
CACHE_REGIME_FILE  = CACHE_DIR / "regime_hte.parquet"
CACHE_RAWDIST_FILE = CACHE_DIR / "raw_dist.parquet"
CACHE_DR_SUMMARY   = CACHE_DIR / "dr_summary.parquet"
CACHE_DR_CURVES    = str(CACHE_DIR / "dr_curves_{}.parquet")   # .format(safe_col_name)
CACHE_DR_SCATTER   = str(CACHE_DIR / "dr_scatter_{}.parquet")
CACHE_DF_FILE      = CACHE_DIR / "analytical_df.parquet"
CACHE_META_FILE    = CACHE_DIR / "cache_meta.json"

# ── Constants ─────────────────────────────────────────────────────────────────
VALID_TAGS = ["10MA", "21MA", "50MA", "5MA", "EP", "HL-21MA", "HL-50MA", "10WMA", "HL-10WMA"]

EXTENSION_COLS = [
    "10 EMA ATR Distance",
    "21 EMA ATR Distance",
    "50 SMA ATR Distance",
    "200 SMA ATR Distance",
]

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

STOCK_LEVEL_CONFOUNDERS = ["20 Day ADR%"]
ALL_CONFOUNDERS = REGIME_POSITION_COLS + STREAK_ZSCORE_COLS + STOCK_LEVEL_CONFOUNDERS  # 21 features

OUTCOME_COL = "ATR Gain Close"
TAG_COL     = "Tags"
DATE_COL    = "Buy Date"
STOCK_COL   = "Stock"


# =============================================================================
# CACHE HELPERS
# =============================================================================

def _safe_col(col_name):
    """Convert a column name to a safe filename fragment."""
    return col_name.replace(" ", "_").replace("%", "pct").replace("/", "_")


def cache_exists():
    # type: () -> bool
    """Return True if all required cache files are present."""
    required = [
        CACHE_SUMMARY_FILE,
        CACHE_AB_FILE,
        CACHE_REGIME_FILE,
        CACHE_RAWDIST_FILE,
        CACHE_DR_SUMMARY,
        CACHE_DF_FILE,
        CACHE_META_FILE,
    ]
    return all(f.exists() for f in required)


def get_cache_meta():
    # type: () -> dict
    """Return metadata about when cache was last built."""
    if CACHE_META_FILE.exists():
        with open(CACHE_META_FILE, "r") as f:
            return json.load(f)
    return {}


def save_cache_meta(selected_tags, min_atr_gain):
    # type: (List[str], float) -> None
    """Save metadata about the current cache run."""
    import datetime
    meta = {
        "last_run":      datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_tags": selected_tags,
        "min_atr_gain":  min_atr_gain,
        "data_file":     str(DATA_FILE),
    }
    with open(CACHE_META_FILE, "w") as f:
        json.dump(meta, f, indent=2)


def load_from_cache():
    # type: () -> dict
    """Load all results from cached parquet/json files."""
    with open(CACHE_SUMMARY_FILE, "r") as f:
        summary_raw = json.load(f)

    summary = summary_raw.copy()
    summary["date_min"] = pd.Timestamp(summary_raw["date_min"])
    summary["date_max"] = pd.Timestamp(summary_raw["date_max"])

    ab = {
        "ate_table":  pd.read_parquet(CACHE_AB_FILE),
        "regime_hte": pd.read_parquet(CACHE_REGIME_FILE),
        "raw_dist":   pd.read_parquet(CACHE_RAWDIST_FILE),
    }

    dr_summary = pd.read_parquet(CACHE_DR_SUMMARY) if CACHE_DR_SUMMARY.exists() else pd.DataFrame()

    curves  = {}
    scatter = {}
    for ext_col in EXTENSION_COLS:
        safe   = _safe_col(ext_col)
        c_path = Path(CACHE_DR_CURVES.format(safe))
        s_path = Path(CACHE_DR_SCATTER.format(safe))
        if c_path.exists():
            curves[ext_col]  = pd.read_parquet(c_path)
        if s_path.exists():
            scatter[ext_col] = pd.read_parquet(s_path)

    dr = {
        "curves":  curves,
        "scatter": scatter,
        "summary": dr_summary,
    }

    df = pd.read_parquet(CACHE_DF_FILE)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    return {
        "df":      df,
        "summary": summary,
        "ab":      ab,
        "dr":      dr,
    }


def save_to_cache(results, selected_tags, min_atr_gain):
    # type: (dict, List[str], float) -> None
    """Persist all results to cache files."""
    summary = results["summary"].copy()
    summary["date_min"] = str(summary["date_min"])
    summary["date_max"] = str(summary["date_max"])
    with open(CACHE_SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    ab = results["ab"]
    ab["ate_table"].to_parquet(CACHE_AB_FILE, index=False)
    ab["regime_hte"].to_parquet(CACHE_REGIME_FILE, index=False)
    ab["raw_dist"].to_parquet(CACHE_RAWDIST_FILE, index=False)

    dr = results["dr"]
    if not dr["summary"].empty:
        dr["summary"].to_parquet(CACHE_DR_SUMMARY, index=False)

    for ext_col, curve_df in dr["curves"].items():
        safe = _safe_col(ext_col)
        curve_df.to_parquet(Path(CACHE_DR_CURVES.format(safe)), index=False)

    for ext_col, scatter_df in dr["scatter"].items():
        safe = _safe_col(ext_col)
        scatter_df.to_parquet(Path(CACHE_DR_SCATTER.format(safe)), index=False)

    results["df"].to_parquet(CACHE_DF_FILE, index=False)
    save_cache_meta(selected_tags, min_atr_gain)


# =============================================================================
# DATA LOADING & PREP
# =============================================================================

def load_raw_data():
    # type: () -> pd.DataFrame
    """Load and minimally clean the combined setup+indices CSV."""
    df = pd.read_csv(DATA_FILE, low_memory=False)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    df = df[df[OUTCOME_COL].notna()].copy()
    df = df[df[OUTCOME_COL] > 0].copy()

    for col in EXTENSION_COLS + ALL_CONFOUNDERS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def explode_tags(df, selected_tags):
    # type: (pd.DataFrame, List[str]) -> pd.DataFrame
    """
    Explode comma-separated Tags into one row per tag.
    A setup tagged '21MA,50MA' becomes two rows if both tags are selected.
    """
    df = df.copy()
    df[TAG_COL] = df[TAG_COL].fillna("").astype(str)
    df["tag_list"] = df[TAG_COL].apply(
        lambda x: [t.strip() for t in x.split(",") if t.strip() in selected_tags]
    )
    df = df[df["tag_list"].map(len) > 0]
    df = df.explode("tag_list").rename(columns={"tag_list": "tag"})
    df = df.reset_index(drop=True)
    return df


def build_analytical_df(selected_tags=None, min_atr_gain=0.0):
    # type: (Optional[List[str]], float) -> pd.DataFrame
    """Full prep pipeline. Returns clean DataFrame ready for both analyses."""
    if selected_tags is None:
        selected_tags = VALID_TAGS

    df = load_raw_data()
    df = df[df[OUTCOME_COL] > min_atr_gain].copy()

    for col in ALL_CONFOUNDERS + EXTENSION_COLS:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    return explode_tags(df, selected_tags)


def get_summary_stats(df):
    # type: (pd.DataFrame) -> dict
    """Return high-level summary stats for the Streamlit header metrics."""
    return {
        "n_setups":       int(df.drop_duplicates(subset=[STOCK_COL, DATE_COL]).shape[0]),
        "n_rows":         len(df),
        "n_tags":         int(df["tag"].nunique()),
        "date_min":       df[DATE_COL].min(),
        "date_max":       df[DATE_COL].max(),
        "mean_outcome":   float(df[OUTCOME_COL].mean()),
        "median_outcome": float(df[OUTCOME_COL].median()),
    }


# =============================================================================
# A/B ANALYSIS
# =============================================================================

def run_ab_analysis(df):
    # type: (pd.DataFrame) -> dict
    """
    Estimate ATE of each setup tag on ATR Gain Close using LinearDML,
    controlling for market regime confounders.
    """
    tag_dummies = pd.get_dummies(df["tag"], prefix="tag", drop_first=False)
    tags = tag_dummies.columns.tolist()

    X = df[ALL_CONFOUNDERS].values
    Y = df[OUTCOME_COL].values

    records = []
    for tag_col in tags:
        tag_name  = tag_col.replace("tag_", "")
        T         = tag_dummies[tag_col].values.astype(float)
        n_treated = int(T.sum())
        n_control = int((1 - T).sum())

        if n_treated < 30 or n_control < 30:
            records.append({
                "tag":                  tag_name,
                "n_treated":            n_treated,
                "mean_outcome_treated": float(df.loc[tag_dummies[tag_col] == 1, OUTCOME_COL].mean()),
                "mean_outcome_control": float(df.loc[tag_dummies[tag_col] == 0, OUTCOME_COL].mean()),
                "ATE":                  np.nan,
                "CI_lower":             np.nan,
                "CI_upper":             np.nan,
                "sufficient_data":      False,
            })
            continue

        model_y = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
        ])
        model_t = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)),
        ])

        dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=5,
            random_state=42,
        )

        try:
            dml.fit(Y, T, X=X)
            ate        = float(dml.ate(X))
            ate_inf    = dml.ate_interval(X, alpha=0.05)
            ci_lower   = float(ate_inf[0])
            ci_upper   = float(ate_inf[1])
            sufficient = True
        except Exception as e:
            print("LinearDML failed for {}: {}".format(tag_name, e))
            ate, ci_lower, ci_upper = np.nan, np.nan, np.nan
            sufficient = False

        records.append({
            "tag":                  tag_name,
            "n_treated":            n_treated,
            "mean_outcome_treated": float(df.loc[tag_dummies[tag_col] == 1, OUTCOME_COL].mean()),
            "mean_outcome_control": float(df.loc[tag_dummies[tag_col] == 0, OUTCOME_COL].mean()),
            "ATE":                  ate,
            "CI_lower":             ci_lower,
            "CI_upper":             ci_upper,
            "sufficient_data":      sufficient,
        })

    ate_table = pd.DataFrame(records).sort_values("ATE", ascending=False).reset_index(drop=True)
    raw_dist  = df[["tag", OUTCOME_COL]].copy()

    df2 = df.copy()
    df2["spy_regime"] = df2["spy_50SMA ATR Distance"].apply(
        lambda x: "SPY Above 50SMA" if x > 0 else "SPY Below 50SMA"
    )
    regime_hte = (
        df2.groupby(["tag", "spy_regime"])[OUTCOME_COL]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_outcome", "median": "median_outcome", "count": "n"})
    )

    return {
        "ate_table":  ate_table,
        "regime_hte": regime_hte,
        "raw_dist":   raw_dist,
    }


# =============================================================================
# DOSE-RESPONSE ANALYSIS
# =============================================================================

def run_dose_response(df):
    # type: (pd.DataFrame) -> dict
    """
    Estimate causal dose-response curve of MA extension on ATR Gain
    using CausalForestDML.
    """
    df_unique = df.drop_duplicates(subset=[STOCK_COL, DATE_COL]).copy()
    Y         = df_unique[OUTCOME_COL].values

    tag_dummies = pd.get_dummies(df_unique["tag"], prefix="tag", drop_first=True)
    X_base      = df_unique[ALL_CONFOUNDERS].values
    X           = np.hstack([X_base, tag_dummies.values.astype(float)])

    curves  = {}
    summary = []
    scatter = {}

    for ext_col in EXTENSION_COLS:
        if ext_col not in df_unique.columns:
            continue

        T    = df_unique[ext_col].values.astype(float)
        mask = ~np.isnan(T) & ~np.isnan(Y)

        T_clean = T[mask]
        Y_clean = Y[mask]
        X_clean = X[mask]

        if len(T_clean) < 100:
            continue

        model_y = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
        ])
        model_t = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)),
        ])

        cf = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            n_estimators=200,
            cv=5,
            random_state=42,
            verbose=0,
        )

        try:
            cf.fit(Y_clean, T_clean, X=X_clean)

            t_min, t_max = np.percentile(T_clean, [2, 98])
            t_range      = np.linspace(t_min, t_max, 60)
            X_mean       = np.mean(X_clean, axis=0, keepdims=True)
            X_sweep      = np.repeat(X_mean, len(t_range), axis=0)
            t_baseline   = np.zeros(len(t_range))

            effects  = cf.effect(X_sweep, T0=t_baseline, T1=t_range)
            inf      = cf.effect_interval(X_sweep, T0=t_baseline, T1=t_range, alpha=0.05)

            curves[ext_col] = pd.DataFrame({
                "extension_val": t_range,
                "effect":        effects,
                "ci_lower":      inf[0],
                "ci_upper":      inf[1],
            })

            tag_vals = (
                df_unique.loc[mask, "tag"].values
                if "tag" in df_unique.columns
                else ["unknown"] * int(mask.sum())
            )
            scatter[ext_col] = pd.DataFrame({
                "extension_val": T_clean,
                "outcome":       Y_clean,
                "tag":           tag_vals,
            })

            summary.append({
                "extension_col":  ext_col,
                "n":              int(mask.sum()),
                "mean_extension": float(np.mean(T_clean)),
                "p10":            float(np.percentile(T_clean, 10)),
                "p50":            float(np.percentile(T_clean, 50)),
                "p90":            float(np.percentile(T_clean, 90)),
                "corr_raw":       float(np.corrcoef(T_clean, Y_clean)[0, 1]),
            })

        except Exception as e:
            print("CausalForestDML failed for {}: {}".format(ext_col, e))
            continue

    return {
        "curves":  curves,
        "scatter": scatter,
        "summary": pd.DataFrame(summary) if summary else pd.DataFrame(),
    }


# =============================================================================
# TOP-LEVEL RUNNER
# =============================================================================

def run_full_analysis(selected_tags=None, min_atr_gain=0.0):
    # type: (Optional[List[str]], float) -> dict
    """
    Run full pipeline and save results to cache.
    Always runs fresh — cache loading is handled by the Streamlit page.
    """
    if selected_tags is None:
        selected_tags = VALID_TAGS

    df         = build_analytical_df(selected_tags=selected_tags, min_atr_gain=min_atr_gain)
    summary    = get_summary_stats(df)
    ab_results = run_ab_analysis(df)
    dr_results = run_dose_response(df)

    results = {
        "df":      df,
        "summary": summary,
        "ab":      ab_results,
        "dr":      dr_results,
    }

    save_to_cache(results, selected_tags, min_atr_gain)
    return results
