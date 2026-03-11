"""
build_watchlist.py
==================
Parses all "M_DD Focus.csv" files from your watchlists folder into a clean
watchlist_curation.csv with columns: date, stock, focus_list_exists

Year disambiguation:
  Months >= CUTOVER_MONTH → YEAR_A (2025)
  Months <  CUTOVER_MONTH → YEAR_B (2026)

  !! UPDATE THESE TWO LINES EACH NEW YEAR !!
  When you start accumulating 2027 files, set YEAR_A=2026, YEAR_B=2027
  and adjust CUTOVER_MONTH to the first month you have 2026 files for.

Usage:
    python build_watchlist.py
    python build_watchlist.py --watchlist_dir data/watchlists --output data/watchlist_curation.csv
    python build_watchlist.py --dry_run        # parse only, print summary, write nothing
"""

import os
import re
import argparse
import pandas as pd
from pathlib import Path
from datetime import date

# ── !! CHANGE THESE AT EACH YEAR BOUNDARY !! ──────────────────────────────────
CUTOVER_MONTH = 9    # months 9–12 belong to YEAR_A; months 1–8 belong to YEAR_B
YEAR_A        = 2025 # year for months >= CUTOVER_MONTH
YEAR_B        = 2026 # year for months <  CUTOVER_MONTH
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_WATCHLIST_DIR = Path("data/watchlists")
DEFAULT_OUTPUT        = Path("data/watchlist_curation.csv")

# Regex: optional leading/trailing text, M_DD or MM_DD somewhere in the filename
FILENAME_RE = re.compile(r"(\d{1,2})_(\d{1,2})")


def infer_year(month: int) -> int:
    """Map a month number to the correct year using the cutover constant."""
    return YEAR_A if month >= CUTOVER_MONTH else YEAR_B


def parse_filename(filename: str) -> date | None:
    """
    Extract a date from a filename like '2_26 Focus.csv' or '11_3 Focus.csv'.
    Returns None if no valid date can be parsed.
    """
    stem  = Path(filename).stem          # '2_26 Focus'
    match = FILENAME_RE.search(stem)
    if not match:
        return None
    month, day = int(match.group(1)), int(match.group(2))
    year = infer_year(month)
    try:
        return date(year, month, day)
    except ValueError:
        return None  # e.g. month=13 or day=32


def load_focus_file(filepath: Path) -> list[str]:
    """
    Load tickers from a single Focus CSV.
    Expects a 'Symbol' column. Cleans and uppercases values.
    Skips blank rows and non-ticker values (numbers, headers).
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"  ⚠  Could not read {filepath.name}: {e}")
        return []

    # Find Symbol column — case-insensitive in case of variation
    sym_col = next(
        (c for c in df.columns if c.strip().lower() == "symbol"), None
    )
    if sym_col is None:
        # Fallback: if only one column, use it regardless of name
        if len(df.columns) == 1:
            sym_col = df.columns[0]
        else:
            print(f"  ⚠  No 'Symbol' column found in {filepath.name} — skipping")
            return []

    tickers = []
    for val in df[sym_col].dropna():
        cleaned = str(val).strip().upper()
        # Skip blanks, pure numbers, and values that are clearly not tickers
        if (cleaned and
                cleaned.isalpha() and       # tickers are letters only
                1 <= len(cleaned) <= 6):    # ticker length sanity check
            tickers.append(cleaned)

    return tickers


def build_watchlist(
    watchlist_dir: Path = DEFAULT_WATCHLIST_DIR,
    output_path: Path   = DEFAULT_OUTPUT,
    dry_run: bool       = False,
) -> pd.DataFrame:
    """
    Main function: scan watchlist_dir for Focus CSV files, parse dates,
    load tickers, and write watchlist_curation.csv.

    Output schema:
        date              — trading date (YYYY-MM-DD)
        stock             — ticker symbol
        focus_list_exists — always 1 for rows from this script
        setup_type        — empty (populated later from voice memo join)
        conviction        — empty (populated later from voice memo join)
        watchlist_reason  — empty (populated later from voice memo join)
    """
    watchlist_dir = Path(watchlist_dir)
    output_path   = Path(output_path)

    if not watchlist_dir.exists():
        print(f"❌ Watchlist directory not found: {watchlist_dir}")
        print(f"   Set --watchlist_dir to the correct path.")
        return pd.DataFrame()

    # ── Scan for Focus files ───────────────────────────────────────────────────
    focus_files = sorted([
        f for f in watchlist_dir.iterdir()
        if f.suffix.lower() == ".csv" and "focus" in f.name.lower()
    ])

    if not focus_files:
        print(f"❌ No Focus CSV files found in {watchlist_dir}")
        print(f"   Expected filenames like '2_26 Focus.csv', '11_3 Focus.csv'")
        return pd.DataFrame()

    print(f"\n{'DRY RUN — ' if dry_run else ''}Processing {len(focus_files)} Focus files")
    print(f"Year logic: months {CUTOVER_MONTH}–12 → {YEAR_A} | months 1–{CUTOVER_MONTH-1} → {YEAR_B}")
    print(f"{'─'*60}")

    rows        = []
    skipped     = []
    date_counts = {}   # date → ticker count, for summary

    for f in focus_files:
        parsed_date = parse_filename(f.name)

        if parsed_date is None:
            print(f"  ⚠  Could not parse date from '{f.name}' — skipping")
            skipped.append(f.name)
            continue

        tickers = load_focus_file(f)

        if not tickers:
            print(f"  ⚠  No valid tickers in '{f.name}' ({parsed_date}) — skipping")
            skipped.append(f.name)
            continue

        date_counts[parsed_date] = len(tickers)
        print(f"  ✓  {f.name:<30} → {parsed_date}  ({len(tickers)} tickers: {', '.join(tickers[:5])}{'...' if len(tickers)>5 else ''})")

        for ticker in tickers:
            rows.append({
                "date":             parsed_date,
                "stock":            ticker,
                "focus_list_exists": 1,
                "setup_type":       "",
                "conviction":       "",
                "watchlist_reason": "",
            })

    if not rows:
        print("\n❌ No valid data parsed from any file.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","stock"]).reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"✅ Parsed {len(df)} total rows across {len(date_counts)} trading days")
    print(f"   Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"   Avg tickers per focus list: {sum(date_counts.values())/len(date_counts):.1f}")
    if skipped:
        print(f"   Skipped {len(skipped)} files: {skipped}")

    # ── Year sanity check ─────────────────────────────────────────────────────
    year_counts = df["date"].dt.year.value_counts().sort_index()
    print(f"\n   Year distribution:")
    for yr, cnt in year_counts.items():
        print(f"     {yr}: {cnt} rows ({cnt//df['stock'].nunique() if df['stock'].nunique() else cnt} days approx)")
    print()

    if dry_run:
        print("DRY RUN — nothing written to disk.")
        return df

    # ── Merge with existing file (preserve manual edits) ──────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing = pd.read_csv(output_path, parse_dates=["date"])
        existing["date"] = pd.to_datetime(existing["date"]).dt.normalize()

        # Keep existing rows that came from manual entry (focus_list_exists=0 or missing)
        # and rows for dates/stocks NOT in the new parsed data
        new_keys = set(zip(df["date"].dt.normalize(), df["stock"]))
        manual_rows = existing[
            ~existing.apply(lambda r: (r["date"], r["stock"]) in new_keys, axis=1)
        ]
        if not manual_rows.empty:
            print(f"   Preserving {len(manual_rows)} existing rows not in new Focus files")
            df = pd.concat([df, manual_rows], ignore_index=True)

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date","stock"]).drop_duplicates(["date","stock"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Written to {output_path}  ({len(df)} total rows)")

    return df


# ── UPDATE data_manager to use focus_list_exists ─────────────────────────────
def patch_protocol_with_focus_flag(
    master: pd.DataFrame,
    protocol: pd.DataFrame,
    trade_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds focus_list_exists and no_focus_trade columns to daily_protocol.

    no_focus_trade = True on days where:
      - trades were taken (trades_taken > 0)
      - BUT no focus list exists for that day (focus_list_exists = 0)

    This is a protocol violation — trading without having done your narrowing process.
    Call this after build_daily_protocol().
    """
    if protocol.empty or master.empty:
        return protocol

    # Days that have at least one watchlist row with focus_list_exists=1
    if "focus_list_exists" in master.columns:
        focus_days = set(
            master[master["focus_list_exists"]==1]["date"].dt.normalize().unique()
        )
    else:
        # Fallback: any day with on_watchlist=1
        focus_days = set(
            master[master["on_watchlist"]==1]["date"].dt.normalize().unique()
        )

    protocol = protocol.copy()
    protocol["focus_list_exists"] = protocol["date"].dt.normalize().apply(
        lambda d: int(d in focus_days)
    )
    protocol["no_focus_trade"] = (
        (protocol["trades_taken"] > 0) & (protocol["focus_list_exists"] == 0)
    ).astype(int)

    return protocol


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build watchlist_curation.csv from Focus CSV files"
    )
    parser.add_argument(
        "--watchlist_dir", default=str(DEFAULT_WATCHLIST_DIR),
        help=f"Folder containing Focus CSV files (default: {DEFAULT_WATCHLIST_DIR})"
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Parse and print summary only — don't write any files"
    )
    args = parser.parse_args()

    build_watchlist(
        watchlist_dir=Path(args.watchlist_dir),
        output_path=Path(args.output),
        dry_run=args.dry_run,
    )
