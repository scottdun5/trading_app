"""
data_manager.py
Handles all data loading, joining, and persistence for the Voice Memo Intelligence system.

Corrected funnel:
  Screener → watchlist_curation.csv (pre-memo, manual)
           → stock_thoughts.csv     (auto from pipeline — the memo analysis layer)
           → trade_log.csv          (outcomes)
           → master_journal.csv     (joined, all derived flags)
           → daily_protocol.csv     (auto-generated adherence tracking)
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
# BASE_DIR resolves to the data/ folder sitting next to utils/ (i.e. trading_app/data/)
# This works regardless of where Streamlit is launched from.
_THIS_DIR = Path(__file__).parent          # .../trading_app/utils/
_APP_DIR  = _THIS_DIR.parent               # .../trading_app/
BASE_DIR  = Path(os.environ.get("JOURNAL_DATA_DIR", str(_APP_DIR / "data")))
STOCK_THOUGHTS_PATH  = BASE_DIR / "stock_thoughts.csv"
MARKET_THOUGHTS_PATH = BASE_DIR / "market_thoughts.csv"
WATCHLIST_PATH       = BASE_DIR / "watchlist_curation.csv"
TRADE_LOG_PATH       = BASE_DIR / "raw_trades.csv"
CORRECTIONS_PATH     = BASE_DIR / "ticker_corrections.json"
MASTER_PATH          = BASE_DIR / "master_journal.csv"
PROTOCOL_PATH        = BASE_DIR / "daily_protocol.csv"

TIGERS         = ["Tightness","Ignition","Group","Earnings","RS"]
EMOTION_ENUM   = ["confident","focused","hesitant","anxious","frustrated",
                  "overconfident","fomo","revenge_trading","distracted","neutral"]
SENTIMENT_ENUM = ["bullish","bearish","neutral","uncertain"]
SENTIMENT_SCORE = {"bullish": 1, "neutral": 0, "uncertain": 0, "bearish": -1}

# Keywords that signal external influence in transcripts / influences field
EXTERNAL_INFLUENCE_KEYWORDS = [
    "twitter","tweet","fintwit","x.com","chat","alert","alerts",
    "newsletter","discord","slack","reddit","youtube","saw a post",
    "someone posted","someone mentioned","heard from","chat room",
]

# ─────────────────────────────────────────────────────────────────────────────
# TICKER CORRECTIONS
# ─────────────────────────────────────────────────────────────────────────────

def load_corrections(path=CORRECTIONS_PATH) -> dict:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return {}

def save_corrections(corrections: dict, path=CORRECTIONS_PATH):
    with open(path, "w") as f:
        json.dump(corrections, f, indent=2)

def apply_corrections(df: pd.DataFrame, corrections: dict, col="stock") -> pd.DataFrame:
    if corrections and col in df.columns:
        df = df.copy()
        df[col] = df[col].apply(
            lambda x: corrections.get(str(x).upper(), x) if pd.notna(x) else x
        )
    return df

def add_correction(wrong: str, correct: str, path=CORRECTIONS_PATH):
    corrections = load_corrections(path)
    corrections[wrong.upper()] = correct.upper()
    save_corrections(corrections, path)
    return corrections

def apply_corrections_to_csv(csv_path, corrections: dict, col="stock"):
    df = pd.read_csv(csv_path)
    df = apply_corrections(df, corrections, col)
    df.to_csv(csv_path, index=False)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_stock_thoughts(path=STOCK_THOUGHTS_PATH, corrections=None) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Normalise ticker column — pipeline may have saved it as 'ticker' or 'symbol'
    if "stock" not in df.columns:
        for alt in ["ticker", "symbol", "Ticker", "Symbol"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "stock"})
                break

    if "stock" not in df.columns:
        print(f"⚠ stock_thoughts.csv has no stock/ticker/symbol column. Columns: {df.columns.tolist()}")
        return pd.DataFrame()

    for c in TIGERS:
        if c not in df.columns:
            df[c] = ""
    df["tigers_score"] = df[TIGERS].apply(
        lambda r: sum(1 for v in r if isinstance(v, str) and v.strip()), axis=1
    )
    if "ticker_confidence"         not in df.columns: df["ticker_confidence"]         = "high"
    if "trade_rationale_explained" not in df.columns: df["trade_rationale_explained"] = False
    if corrections:
        df = apply_corrections(df, corrections)
    df["stock"] = df["stock"].str.upper().str.strip()
    return df


def load_market_thoughts(path=MARKET_THOUGHTS_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["sentiment_score"] = df["market_sentiment"].apply(
        lambda x: SENTIMENT_SCORE.get(str(x).lower().strip(), 0) if pd.notna(x) else 0
    )
    if "emotional_intensity" not in df.columns:
        df["emotional_intensity"] = 3
    # Auto-detect external influence days from influences field + transcript
    influence_text = (
        df.get("influences",       pd.Series("", index=df.index)).fillna("").str.lower() + " " +
        df.get("raw_transcript",   pd.Series("", index=df.index)).fillna("").str.lower()
    )
    df["external_influence_day"] = influence_text.apply(
        lambda t: any(kw in t for kw in EXTERNAL_INFLUENCE_KEYWORDS)
    )
    return df


def load_watchlist(path=WATCHLIST_PATH) -> pd.DataFrame:
    """
    Binary watchlist built from Focus CSV files via build_watchlist.py.
    Schema: date, stock, focus_list_exists, setup_type, conviction, watchlist_reason
    """
    schema_dtypes = {
        "date":               "datetime64[ns]",
        "stock":              "str",
        "focus_list_exists":  "Int64",   # 1 = came from a Focus file, 0 = manual
        "setup_type":         "str",
        "conviction":         "str",
        "watchlist_reason":   "str",
    }
    if not Path(path).exists():
        df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in schema_dtypes.items()})
        df.to_csv(path, index=False)
        return df
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"]  = pd.to_datetime(df["date"]).dt.normalize()
    df["stock"] = df["stock"].str.upper().str.strip()
    for col, dtype in schema_dtypes.items():
        if col not in df.columns:
            df[col] = pd.NA
    return df[[c for c in schema_dtypes if c in df.columns]].copy()


def load_trade_log(path=TRADE_LOG_PATH) -> pd.DataFrame:
    """Loads raw_trades.csv and normalises to date, stock, win."""
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Map exact column names from raw_trades.csv
    df["date"]  = pd.to_datetime(df["buy_date"], errors="coerce").dt.normalize()
    df["stock"] = df["symbol"].str.upper().str.strip()
    df["win"]   = (df["gain"] > 0).astype(int)

    # Keep original columns too so P&L data is available for analytics
    return df.dropna(subset=["date", "stock"])


def load_daily_protocol(path=PROTOCOL_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MASTER JOIN — corrected funnel
# ─────────────────────────────────────────────────────────────────────────────

def build_master_journal(
    stock_path=STOCK_THOUGHTS_PATH,
    market_path=MARKET_THOUGHTS_PATH,
    watchlist_path=WATCHLIST_PATH,
    trade_log_path=TRADE_LOG_PATH,
    corrections_path=CORRECTIONS_PATH,
    save_path=MASTER_PATH,
    protocol_path=PROTOCOL_PATH,
) -> pd.DataFrame:
    """
    Build master_journal.csv — one row per stock per date.

    Three types of rows:
      A) Watchlist stock + discussed in memo   → memo_coverage=1
      B) Watchlist stock + NOT in memo         → watchlist_uncovered=1 (silent skip)
      C) Memo stock + NOT on watchlist         → organic_idea=1 (spontaneous idea)

    Key derived flags:
      off_watchlist_entry  — traded a stock that was NOT on watchlist (impulsive)
      no_analysis_trade    — off-watchlist AND no verbal rationale (highest risk)
      uncovered_trade      — put on watchlist, never discussed, then traded anyway
    """
    corrections = load_corrections(corrections_path)
    sdf  = load_stock_thoughts(stock_path, corrections)
    mdf  = load_market_thoughts(market_path)
    wdf  = load_watchlist(watchlist_path)
    tdf  = load_trade_log(trade_log_path)

    master = _join_all(sdf, mdf, wdf, tdf)

    if save_path and not master.empty:
        master.to_csv(save_path, index=False)

    if not master.empty:
        protocol = build_daily_protocol(master, mdf, save_path=protocol_path)

    return master


def _join_all(sdf, mdf, wdf, tdf) -> pd.DataFrame:
    """Core join logic shared by build_master_journal and demo loader."""
    if sdf.empty and wdf.empty:
        return pd.DataFrame()

    # Normalize dates in source frames before any slicing
    if not sdf.empty:
        sdf = sdf.copy()
        sdf["date"] = pd.to_datetime(sdf["date"]).dt.normalize()
        sdf["stock"] = sdf["stock"].str.upper().str.strip()

    if not wdf.empty:
        wdf = wdf.copy()
        wdf["date"] = pd.to_datetime(wdf["date"]).dt.normalize()
        wdf["stock"] = wdf["stock"].str.upper().str.strip()

    memo_keys  = sdf[["date","stock"]].copy()  if not sdf.empty  else pd.DataFrame(columns=["date","stock"])
    watch_keys = wdf[["date","stock"]].copy()  if not wdf.empty  else pd.DataFrame(columns=["date","stock"])

    # Universe = union of both sources
    all_keys = pd.concat([memo_keys, watch_keys], ignore_index=True).drop_duplicates(["date","stock"])
    all_keys["date"] = pd.to_datetime(all_keys["date"]).dt.normalize()

    # Base: start with all_keys, left-join in memo analysis
    master = all_keys.merge(sdf, on=["date","stock"], how="left")

    # ── Funnel flags ──────────────────────────────────────────────────────────
    memo_set  = set(zip(memo_keys["date"],  memo_keys["stock"]))
    watch_set = set(zip(watch_keys["date"], watch_keys["stock"]))

    master["on_watchlist"]         = master.apply(lambda r: int((r["date"],r["stock"]) in watch_set), axis=1)
    master["in_voice_memo"]        = master.apply(lambda r: int((r["date"],r["stock"]) in memo_set),  axis=1)
    master["organic_idea"]         = ((master["in_voice_memo"]==1) & (master["on_watchlist"]==0)).astype(int)
    master["memo_coverage"]        = ((master["on_watchlist"]==1) & (master["in_voice_memo"]==1)).astype(int)
    master["watchlist_uncovered"]  = ((master["on_watchlist"]==1) & (master["in_voice_memo"]==0)).astype(int)

    # ── Watchlist metadata ────────────────────────────────────────────────────
    if not wdf.empty:
        master = master.merge(
            wdf[["date","stock","setup_type","conviction","watchlist_reason"]],
            on=["date","stock"], how="left"
        )

    # ── Market context ────────────────────────────────────────────────────────
    if not mdf.empty:
        mday = mdf[[
            "date","market_sentiment","emotional_state","emotional_intensity",
            "sentiment_score","influences","key_themes","external_influence_day"
        ]].copy()
        mday.columns = [
            "date","day_market_sentiment","day_emotional_state","day_emotional_intensity",
            "day_sentiment_score","day_influences","day_key_themes","external_influence_day"
        ]
        master = master.merge(mday, on="date", how="left")

    # ── Trade outcomes ────────────────────────────────────────────────────────
    if not tdf.empty and "stock" in tdf.columns and "date" in tdf.columns:
        trade_keys = set(zip(tdf["date"], tdf["stock"]))
        master["did_trade"] = master.apply(lambda r: int((r["date"],r["stock"]) in trade_keys), axis=1)
        if "win" in tdf.columns:
            master = master.merge(
                tdf[["date","stock","win"]].drop_duplicates(["date","stock"]),
                on=["date","stock"], how="left"
            )
        else:
            master["win"] = pd.NA
    else:
        master["did_trade"] = 0
        master["win"]       = pd.NA

    # ── High-risk pattern flags ───────────────────────────────────────────────
    master["off_watchlist_entry"] = (
        (master["did_trade"]==1) & (master["on_watchlist"]==0)
    ).astype(int)

    rationale = master.get("trade_rationale_explained", pd.Series(False, index=master.index))
    master["no_analysis_trade"] = (
        (master["off_watchlist_entry"]==1) & (~rationale.fillna(False))
    ).astype(int)

    master["uncovered_trade"] = (
        (master["watchlist_uncovered"]==1) & (master["did_trade"]==1)
    ).astype(int)

    # ── Outcome category (5 buckets) ──────────────────────────────────────────
    def categorize(row):
        traded = int(row.get("did_trade", 0) or 0)
        on_wl  = int(row.get("on_watchlist", 0) or 0)
        should = row.get("should_have_traded", np.nan)
        if traded == 1 and on_wl == 0:  return "impulsive_entry"
        if pd.isna(should):             return "uncategorized"
        should = int(should)
        if traded==1 and should==1: return "correct_select"
        if traded==0 and should==0: return "correct_skip"
        if traded==1 and should==0: return "false_positive"
        if traded==0 and should==1: return "missed"
        return "uncategorized"

    master["outcome_category"] = master.apply(categorize, axis=1)
    master["week"]  = master["date"].dt.to_period("W").astype(str)
    master["month"] = master["date"].dt.to_period("M").astype(str)
    master["dow"]   = master["date"].dt.day_name()

    return master.sort_values(["date","stock"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# DAILY PROTOCOL — fully auto-generated, never manually edited
# ─────────────────────────────────────────────────────────────────────────────

def build_daily_protocol(
    master: pd.DataFrame,
    market_df: pd.DataFrame,
    save_path=PROTOCOL_PATH,
    coverage_threshold: float = 0.80,
) -> pd.DataFrame:
    """
    One row per trading day. Entirely derived — never manually edited
    (except the optional free-text 'notes' column).

    Protocol score 0–3:
      +1  memo recorded
      +1  coverage_rate >= coverage_threshold (discussed ≥80% of watchlist)
      +1  all watchlist trades had rationale explained in memo
    """
    if master.empty:
        return pd.DataFrame()

    rows = []
    for day, grp in master.groupby("date"):
        memo_recorded       = int(grp["in_voice_memo"].any())
        watchlist_count     = int(grp["on_watchlist"].sum())
        memo_cov_count      = int(grp["memo_coverage"].sum())
        coverage_rate       = memo_cov_count / watchlist_count if watchlist_count > 0 else np.nan
        trades_taken        = int(grp["did_trade"].sum())
        off_wl_count        = int(grp["off_watchlist_entry"].sum())
        uncovered_trades    = int(grp["uncovered_trade"].sum())
        no_analysis_trades  = int(grp["no_analysis_trade"].sum())

        wl_trades = grp[(grp["on_watchlist"]==1) & (grp["did_trade"]==1)]
        if not wl_trades.empty and "trade_rationale_explained" in wl_trades.columns:
            rationale_count = int(wl_trades["trade_rationale_explained"].fillna(False).sum())
            all_rationale   = int(rationale_count == len(wl_trades))
        else:
            rationale_count = 0
            all_rationale   = 1 if len(wl_trades) == 0 else 0

        protocol_score = (
            int(memo_recorded) +
            int(pd.notna(coverage_rate) and coverage_rate >= coverage_threshold) +
            int(all_rationale)
        )

        ext_inf = False
        if not market_df.empty and "external_influence_day" in market_df.columns:
            day_row = market_df[market_df["date"] == day]
            if not day_row.empty:
                ext_inf = bool(day_row["external_influence_day"].iloc[0])

        # Focus list flag — did a Focus file exist for this day?
        focus_exists = int(grp["focus_list_exists"].fillna(0).astype(int).any()) \
                       if "focus_list_exists" in grp.columns else 0
        no_focus_trade = int(trades_taken > 0 and focus_exists == 0)

        # Update protocol score: also penalise trading without a focus list
        # Score components:
        #   +1  memo recorded
        #   +1  coverage_rate >= threshold
        #   +1  all watchlist trades had rationale explained
        # Violations tracked separately (don't subtract, just flag):
        #   no_focus_trade  — traded without a focus list
        #   off_watchlist_count > 0 — traded off watchlist

        rows.append({
            "date":                   day,
            "memo_recorded":          memo_recorded,
            "focus_list_exists":      focus_exists,
            "watchlist_count":        watchlist_count,
            "memo_coverage_count":    memo_cov_count,
            "coverage_rate":          round(coverage_rate, 3) if pd.notna(coverage_rate) else np.nan,
            "trades_taken":           trades_taken,
            "off_watchlist_count":    off_wl_count,
            "uncovered_trades":       uncovered_trades,
            "no_analysis_trades":     no_analysis_trades,
            "no_focus_trade":         no_focus_trade,
            "trades_with_rationale":  rationale_count,
            "all_rationale":          all_rationale,
            "external_influence_day": int(ext_inf),
            "protocol_score":         protocol_score,
            "notes":                  "",
        })

    protocol = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if save_path:
        # Preserve existing notes on rebuild — never overwrite them
        p = Path(save_path)
        if p.exists():
            existing = pd.read_csv(p, parse_dates=["date"])
            existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()
            if "notes" in existing.columns:
                notes_map = existing.set_index("date")["notes"].dropna().to_dict()
                protocol["notes"] = protocol["date"].map(notes_map).fillna("")
        protocol.to_csv(save_path, index=False)

    return protocol

# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_watchlist_entry(entry: dict, path=WATCHLIST_PATH):
    """Upsert a single row on (date, stock). Only valid schema columns written."""
    df = load_watchlist(path)
    entry["date"]  = pd.to_datetime(entry["date"]).normalize()
    entry["stock"] = str(entry["stock"]).upper().strip()
    valid = {"date","stock","focus_list_exists","setup_type","conviction","watchlist_reason"}
    entry = {k: v for k, v in entry.items() if k in valid}

    mask = (df["date"] == entry["date"]) & (df["stock"] == entry["stock"])
    if mask.any():
        for k, v in entry.items():
            if k in df.columns:
                df.loc[mask, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df.sort_values(["date","stock"]).reset_index(drop=True).to_csv(path, index=False)
    return df


def delete_watchlist_entry(date_val, stock: str, path=WATCHLIST_PATH):
    df = load_watchlist(path)
    date_val = pd.to_datetime(date_val).normalize()
    df = df[~((df["date"]==date_val) & (df["stock"]==stock.upper().strip()))]
    df.to_csv(path, index=False)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# DEMO DATA — corrected funnel
# ─────────────────────────────────────────────────────────────────────────────

def generate_demo_data(n_days=90, seed=42):
    rng    = np.random.default_rng(seed)
    dates  = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
    TICKERS = ["NVDA","APP","PLTR","RKLB","TSLA","META","CRWD","SMCI","RXRX","NBIS",
               "TTD","IONQ","HIMS","UBER","CELH","VRT","AVGO","ARM","ANET","MSTR"]
    THEMES  = ["AI momentum","Tech rotation","Rate fears","Earnings season",
               "Sector leadership","Macro uncertainty","Small cap rotation","Crypto corr"]
    SETUPS  = ["VCP","Flat base","Cup with handle","High tight flag","IPO base","3-weeks-tight"]
    INFL    = ["Twitter","Personal analysis","Chat room","Podcast","News","Alerts",
               "Personal analysis","Personal analysis"]  # weighted toward personal

    mrows, srows, wrows, trows = [], [], [], []

    for d in dates:
        infl = str(rng.choice(INFL))
        mrows.append({
            "date":               d,
            "market_sentiment":   str(rng.choice(SENTIMENT_ENUM, p=[0.4,0.2,0.25,0.15])),
            "emotional_state":    str(rng.choice(EMOTION_ENUM)),
            "emotional_intensity": int(rng.integers(1,6)),
            "influences":         infl,
            "key_themes":         ", ".join(rng.choice(THEMES, size=2, replace=False).tolist()),
            "technical_analysis": str(rng.choice(["SPY: above 21EMA","QQQ: extended","IWM: lagging","SPY: below 50SMA"])),
            "raw_transcript":     (
                f"Demo transcript for {d.date()}. Market feeling {rng.choice(['strong','heavy','mixed'])}. "
                + ("Saw something interesting on Twitter about the sector today." if "Twitter" in infl else "")
            ),
        })

        # Watchlist: 4–7 stocks per day (pre-memo)
        n_wl      = int(rng.integers(4, 8))
        wl_stocks = rng.choice(TICKERS, size=n_wl, replace=False).tolist()

        for tk in wl_stocks:
            wrows.append({
                "date": d, "stock": tk,
                "focus_list_exists": 1,
                "setup_type":       str(rng.choice(SETUPS)),
                "conviction":       str(rng.choice(["high","med","low"])),
                "watchlist_reason": "Tight setup with strong RS and group acting well.",
            })

        # Memo: covers most watchlist stocks + 0–2 organic ideas
        n_covered  = int(rng.integers(max(1, n_wl - 2), n_wl + 1))
        covered    = wl_stocks[:n_covered]
        organic_pool = [t for t in TICKERS if t not in wl_stocks]
        n_organic  = int(rng.integers(0, 3))
        organic    = rng.choice(organic_pool, size=min(n_organic, len(organic_pool)), replace=False).tolist()
        memo_stocks = covered + organic

        for tk in memo_stocks:
            ts      = int(rng.integers(0, 6))
            factors = rng.choice(TIGERS, size=ts, replace=False).tolist() if ts > 0 else []
            is_organic = tk in organic
            # Watchlist stocks more likely to be traded; organic ones more impulsive/worse
            traded    = bool(rng.random() < (0.25 if is_organic else 0.38))
            rationale = traded and bool(rng.random() < 0.68)
            srows.append({
                "date": d, "stock": tk,
                "raw_thoughts": (
                    f"{'Tight action ' if 'Tightness' in factors else ''}"
                    f"{'Strong RS ' if 'RS' in factors else ''}"
                    f"{'Breakout potential' if 'Ignition' in factors else 'Watching for setup'}."
                    + (" Entry triggered at key level, stop below base." if rationale else "")
                ),
                "summarized_thoughts":       f"{tk} discussed in memo.",
                "Tightness": "Tight consolidation near highs" if "Tightness" in factors else "",
                "Ignition":  "Volume expansion on breakout"   if "Ignition"  in factors else "",
                "Group":     "Sector acting well"             if "Group"     in factors else "",
                "Earnings":  "Strong recent earnings"         if "Earnings"  in factors else "",
                "RS":        "Outperforming SPY"              if "RS"        in factors else "",
                "tigers_score":             ts,
                "ticker_confidence":        str(rng.choice(["high","high","high","low"])),
                "trade_rationale_explained": rationale,
            })
            if traded:
                # Organic / impulsive entries have meaningfully worse win rates
                win_rate = 0.33 if is_organic else 0.44
                trows.append({"date": d, "stock": tk, "win": int(rng.random() < win_rate)})

    mdf = pd.DataFrame(mrows)
    mdf["sentiment_score"] = mdf["market_sentiment"].map(SENTIMENT_SCORE)
    mdf["external_influence_day"] = mdf["influences"].str.lower().apply(
        lambda x: any(kw in x for kw in EXTERNAL_INFLUENCE_KEYWORDS)
    )
    sdf = pd.DataFrame(srows)
    wdf = pd.DataFrame(wrows)
    tdf = pd.DataFrame(trows) if trows else pd.DataFrame(columns=["date","stock","win"])
    return mdf, sdf, wdf, tdf


def load_or_generate_demo():
    """Return (mdf, sdf, wdf, tdf, master, protocol) from in-memory demo data."""
    mdf, sdf, wdf, tdf = generate_demo_data()
    master   = _join_all(sdf, mdf, wdf, tdf)
    protocol = build_daily_protocol(master, mdf, save_path=None)
    return mdf, sdf, wdf, tdf, master, protocol
