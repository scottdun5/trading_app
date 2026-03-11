import os
import csv
import json
import shutil
import openai
import pandas as pd
from datetime import datetime
from pathlib import Path
from pydub import AudioSegment, silence
from utils.config import OPENAI_API_KEY


# =========================
# CONFIGURATION SECTION
# =========================

openai.api_key = OPENAI_API_KEY

# Folder paths
VOICE_MEMOS_FOLDER = 
JOURNAL_FOLDER = Path("trading_app/data")
AUDIO_FOLDER = JOURNAL_FOLDER / "audio"
PROCESSED_LOG = JOURNAL_FOLDER / "processed_files.csv"
RAW_TRANSCRIPTS = JOURNAL_FOLDER / "raw_transcripts.csv"

# Output files
MARKET_THOUGHTS = JOURNAL_FOLDER / "market_thoughts.csv"
STOCK_THOUGHTS = JOURNAL_FOLDER / "stock_thoughts.csv"
MARKET_ANALYSIS = JOURNAL_FOLDER / "market_bias_analysis.csv"
STOCK_ANALYSIS = JOURNAL_FOLDER / "stock_bias_analysis.csv"

MAX_DURATION_SEC = 1400

# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_MARKET = """
You are analyzing a trader's overall market thoughts and mental state from a single voice memo.

From the transcript below, extract the following fields:

- market_sentiment: the trader's overall market view.
  MUST be one of: ["bullish", "bearish", "neutral", "uncertain"]
  Look for language like "market looks strong" → "bullish", "can't get a read" → "uncertain".

- emotional_state: the trader's dominant mindset or feeling related to trading.
  MUST be one of: ["confident", "focused", "hesitant", "anxious", "frustrated",
                   "overconfident", "fomo", "revenge_trading", "distracted", "neutral"]
  Use context: urgency/chasing = "fomo", second-guessing = "hesitant", calm/systematic = "focused".

- emotional_intensity: rate the emotional charge of the memo on a scale of 1–5.
  1 = completely calm/analytical. 5 = highly charged, emotional, reactive.

- influences: list external sources mentioned that shaped the trader's thinking.
  Examples: Twitter, chat room, newsletter, alerts, podcast, news, other traders.
  Return as a comma-separated string. If none mentioned, return "".

- key_themes: the main market narratives or macro factors discussed.
  Examples: rotation into tech, AI momentum, rate uncertainty, earnings season, sector breakdown.
  Return as a comma-separated string.

- technical_analysis: any index/ETF setup observations.
  Format each as "INSTRUMENT: observation". Example: "SPY: above 21EMA, testing breakout level".
  Instruments to watch for: SPY, QQQ, QQQE, RSP, IWM, VIX, DIA, credit spreads, breadth indicators.

- raw_transcript: copy the full transcript verbatim, unchanged.

RULES:
- If a field has no supporting evidence in the transcript, return "" — never infer or fabricate.
- For market_sentiment and emotional_state, you MUST choose from the provided options.
- Return ONLY a valid JSON object with no additional text, preamble, or markdown.

Return format:
{{
  "market_sentiment": "",
  "emotional_state": "",
  "emotional_intensity": 0,
  "influences": "",
  "key_themes": "",
  "technical_analysis": "",
  "raw_transcript": ""
}}

Transcript:
{transcript}
"""


PROMPT_STOCKS = """
You are extracting structured stock-specific observations from a trader's voice memo.

STEP 1 — TICKER IDENTIFICATION
Scan the transcript for any company names, ticker symbols, or phonetic references to stocks.
Convert each to its publicly listed ticker symbol. Rules:
  - Letters spelled out: "P-L-T-R" → "PLTR"
  - Phonetic mishears (common ones):
      "Rocket Lab" or "Rocket Labs"  → "RKLB"
      "Lending Club"                  → "LC"
      "Palantir"                      → "PLTR"
      "Garden Health"                 → "GH"
      "MBIS" or "N-BIS"              → "NBIS"
      "Samsara"                       → "IOT"
      "Applovin" or "App Lovin"      → "APP"
      "Carvana"                       → "CVNA"
      "Hims" or "Hims and Hers"      → "HIMS"
      "Veev"                          → "VEEV"
      "Vertiv"                        → "VRT"
  - If you are unsure about a ticker, still include your best guess but set ticker_confidence to "low".
  - If you are confident, set ticker_confidence to "high".

STEP 2 — FOR EACH TICKER, EXTRACT:
  - stock: the resolved ticker symbol (uppercase)
  - ticker_confidence: "high" or "low"
  - raw_thoughts: copy the trader's EXACT spoken words about this stock verbatim.
    Do not clean up, rephrase, or correct grammar. Preserve every word as spoken.
  - summarized_thoughts: a 1–3 sentence summary of the trader's reasoning or observations.
  - trade_rationale_explained: true if the trader explicitly states a trade decision or
    intention to act — not just setup observation. Look for language like:
      "I'm going to buy/sell", "I took a position", "entering here", "I bought/sold",
      "adding to my position", "I'm in this", "taking the trade", "got long/short",
      "I pulled the trigger", "initiating", "I entered", "placing an order",
      "scaling in", "trimming", "taking profits", "stopped out", "I exited",
      "cutting the position", "I filled", "filled my order".
    Set to false if the trader only describes the setup, chart action, or says they
    are "watching", "monitoring", "on my radar", or "interested" without a clear
    statement of having taken or decided to take a specific trade action.

STEP 3 — TIGERS FRAMEWORK EXTRACTION
For each stock, scan raw_thoughts for language matching each TIGERS factor.
ONLY populate a factor if you find explicit supporting language. Never infer or fabricate.

  Tightness:  tight, consolidating, coiling, low volatility, VCP, compressed, RPP, RMV,
              basing, range-bound, calm, not moving much, tight action
  Ignition:   breakout, breaking out, surging, explosive, volume spike, momentum, launching,
              clearing, triggered, running, move starting, above resistance
  Group:      sector, group, industry, peers, similar stocks moving, theme, names in space,
              whole group acting well
  Earnings:   earnings, EPS, revenue, sales growth, guidance, beat, fundamental, valuation
  RS:         relative strength, holding up, outperforming, leading, strongest, RS line,
              underperforming, weak vs market, lagging

Return a VALID JSON array. No extra text, preamble, or markdown.

[
  {{
    "stock": "",
    "ticker_confidence": "high",
    "raw_thoughts": "",
    "summarized_thoughts": "",
    "trade_rationale_explained": false,
    "Tightness": "",
    "Ignition": "",
    "Group": "",
    "Earnings": "",
    "RS": ""
  }}
]

Transcript:
{transcript}
"""


PROMPT_MARKET_ANALYSIS = """
You are a trading coach analyzing behavioral patterns, biases, and decision-making trends
in a trader's voice memo data over time.

SOURCE DATA: market_thoughts.csv
CRITICAL: Base ALL analysis ONLY on the raw_transcript field.
Do NOT use AI-generated fields (market_sentiment, emotional_state, key_themes, etc.) as evidence.
Only use those fields for grouping/filtering context.

DATE RANGE CONTEXT: The data spans {date_start} to {date_end}.
Anchor all trend observations to actual date ranges (e.g., "In early November..." or "During the week of...").

ANALYSIS TASK — Think step by step before producing JSON:

Step 1: Identify the four behavioral scenario groups:
  - Missed Opportunities: should_have_traded = 1 AND traded = 0
  - False Positives (Overtrading): should_have_traded = 0 AND traded = 1
  - Correct Positive Decisions: should_have_traded = 1 AND traded = 1
  - Correct Avoidance: should_have_traded = 0 AND traded = 0

Step 2: For each group, find 3–5 transcript quotes (5–15 words each) that illustrate
  the trader's reasoning, language patterns, or emotional tone. Include the date.

Step 3: Identify recurring biases — look for:
  - Confirmation bias (only hearing what supports existing view)
  - Recency bias (recent market action dominating thinking)
  - External influence overweighting (acting on Twitter/alerts vs own analysis)
  - Anchoring (returning to same levels or ideas repeatedly)
  - Emotional contagion (market mood dictating personal mood)

Step 4: Identify weekly trends (week-over-week sentiment/emotion changes)
  and monthly trends (month-over-month shifts in themes and psychology).

Step 5: Generate 3–5 actionable improvement recommendations with supporting evidence.

Return ONLY a valid JSON object — no preamble, no markdown, no explanation outside the JSON:

{{
  "behavioral_patterns": {{
    "missed_opportunities": {{
      "summary": "1–3 sentence summary of observed reasoning patterns",
      "examples": [{{ "quote": "exact short quote", "date": "YYYY-MM-DD" }}]
    }},
    "overtrading_false_positives": {{
      "summary": "1–3 sentence summary",
      "examples": [{{ "quote": "exact short quote", "date": "YYYY-MM-DD" }}]
    }},
    "correct_positive": {{
      "summary": "1–3 sentence summary",
      "examples": [{{ "quote": "exact short quote", "date": "YYYY-MM-DD" }}]
    }},
    "correct_avoidance": {{
      "summary": "1–3 sentence summary",
      "examples": [{{ "quote": "exact short quote", "date": "YYYY-MM-DD" }}]
    }}
  }},
  "observed_biases": {{
    "summary": "1–3 sentence summary of recurring biases",
    "bias_types": ["confirmation_bias", "recency_bias"],
    "examples": [{{ "quote": "exact short quote", "date": "YYYY-MM-DD", "bias_type": "" }}]
  }},
  "suggested_focus_areas": {{
    "summary": "actionable recommendations",
    "items": [
      {{ "recommendation": "specific suggestion", "supporting_quote": "short quote", "date": "YYYY-MM-DD" }}
    ]
  }},
  "trends_over_time": {{
    "weekly": {{
      "summary": "week-over-week pattern summary",
      "examples": [{{ "quote": "short quote", "date": "YYYY-MM-DD", "week": "YYYY-WW" }}]
    }},
    "monthly": {{
      "summary": "month-over-month pattern summary",
      "examples": [{{ "quote": "short quote", "date": "YYYY-MM-DD", "month": "YYYY-MM" }}]
    }}
  }}
}}
"""


PROMPT_STOCK_ANALYSIS = """
You are a trading coach analyzing biases in a trader's stock selection process over time.

SOURCE DATA: stock_thoughts.csv
CRITICAL: Base ALL analysis ONLY on the raw_thoughts field.
Do NOT use summarized_thoughts or AI-generated TIGERS columns as evidence.
Only use those for supplementary grouping context.

DATE RANGE CONTEXT: Data spans {date_start} to {date_end}.
Reference specific time periods when describing trends.

TIGERS FRAMEWORK (use as analytical lens — look for these in raw_thoughts):
  T - Tightness:   consolidation, tight, VCP, coiling, compressed, low volatility
  I - Ignition:    breakout, surging, momentum, triggering, volume spike
  G - Group:       sector/theme mentions, group acting, peers moving
  E - Earnings:    revenue, EPS, growth, fundamental quality
  R - RS:          relative strength, leading, outperforming, holding up

ANALYSIS TASK — Think step by step:

Step 1: Categorize all entries into four outcome groups:
  - False Positive (selected=1, should_have_selected=0)
  - Missed (selected=0, should_have_selected=1)
  - Correct Select (selected=1, should_have_selected=1)
  - Correct Skip (selected=0, should_have_selected=0)

Step 2: For each group, analyze which TIGERS factors are mentioned vs absent.
  Does the trader over-rely on RS/Ignition while ignoring Tightness/Group?

Step 3: Identify weekly patterns (last 5 trading days vs prior week):
  Which TIGERS factors were emphasized? Did focus shift?

Step 4: Identify monthly patterns (this month vs prior month):
  Has vocabulary or emphasis changed? More fundamental vs technical?

Step 5: Identify the 3 most important biases in stock selection.

Return ONLY a valid JSON object — no preamble, no markdown, no explanation outside the JSON:

{{
  "outcome_analysis": {{
    "false_positives": {{
      "summary": "what reasoning errors led to bad picks",
      "tigers_pattern": "which factors were overemphasized or missing",
      "examples": [{{ "stock": "", "quote": "", "date": "" }}]
    }},
    "missed_opportunities": {{
      "summary": "what was missed or underweighted",
      "tigers_pattern": "which factors were absent or ignored",
      "examples": [{{ "stock": "", "quote": "", "date": "" }}]
    }},
    "correct_selections": {{
      "summary": "what reasoning patterns produced good picks",
      "tigers_pattern": "which factor combination correlated with good outcomes",
      "examples": [{{ "stock": "", "quote": "", "date": "" }}]
    }},
    "correct_skips": {{
      "summary": "what triggered proper avoidance",
      "tigers_pattern": "",
      "examples": [{{ "stock": "", "quote": "", "date": "" }}]
    }}
  }},
  "observed_biases": {{
    "summary": "top 3 recurring selection biases",
    "biases": [
      {{ "bias": "bias name", "description": "1–2 sentences", "example_quote": "", "date": "" }}
    ]
  }},
  "suggested_focus_areas": {{
    "summary": "top recommendations for improving stock selection",
    "items": [
      {{ "recommendation": "", "rationale": "", "example_quote": "", "date": "" }}
    ]
  }},
  "trends_over_time": {{
    "weekly": {{
      "summary": "last week vs prior week TIGERS emphasis and reasoning shifts",
      "examples": [{{ "stock": "", "quote": "", "date": "" }}]
    }},
    "monthly": {{
      "summary": "this month vs prior month approach changes",
      "examples": [{{ "stock": "", "quote": "", "date": "" }}]
    }}
  }}
}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def parse_json_response(content: str, fallback):
    """
    Robustly parse a JSON response from the LLM.
    Strips markdown fences (```json ... ``` or ``` ... ```) before parsing.
    Returns fallback value on failure instead of crashing.
    """
    content = content.strip()
    if content.startswith("```"):
        # Drop the first line (```json or ```) and last line (```)
        lines = content.splitlines()
        content = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"  ⚠  JSON parse error: {e}")
        print(f"  Raw content (first 300 chars): {content[:300]}")
        return fallback


def get_existing_dates(csv_path: Path) -> set:
    """Return the set of date strings already in a CSV. Returns empty set if file missing."""
    if not csv_path.exists():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["date"])
        return set(df["date"].astype(str).unique())
    except Exception:
        return set()


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE FUNCTIONS  (same names/signatures as original — drop-in compatible)
# ══════════════════════════════════════════════════════════════════════════════

def get_new_files(force: bool = False):
    """Detects .m4a files not yet processed. If force=True, returns ALL files."""
    if force:
        return list(VOICE_MEMOS_FOLDER.glob("*.m4a"))
    processed = set()
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG) as f:
            reader = csv.reader(f)
            processed = {row[0] for row in reader}
    new_files = [f for f in VOICE_MEMOS_FOLDER.glob("*.m4a") if f.name not in processed]
    return new_files


def move_and_log_new_files(new_files):
    """Copies new audio files to journal/audio and logs them."""
    AUDIO_FOLDER.mkdir(parents=True, exist_ok=True)
    JOURNAL_FOLDER.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_LOG, "a", newline="") as log:
        writer = csv.writer(log)
        for f in new_files:
            dest = AUDIO_FOLDER / f.name
            shutil.copy2(f, dest)
            writer.writerow([f.name])
    print(f"✅ Moved {len(new_files)} new audio files.")


def transcribe_audio(file_path):
    """Transcribes a single .m4a file using Whisper."""
    with open(file_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
        )
    return transcript.text


def merge_daily_transcripts():
    """
    Transcribes all audio in AUDIO_FOLDER and merges by date into raw_transcripts.csv.
    Skips dates already present in the file (incremental — won't re-transcribe old audio).
    """
    existing_dates = get_existing_dates(RAW_TRANSCRIPTS)

    rows = []
    for f in sorted(AUDIO_FOLDER.glob("*.m4a"), key=lambda x: x.name):
        print(f.name)
        date_str = f.name.split(" ")[0][:8]  # Extract YYYYMMDD
        try:
            date = datetime.strptime(date_str, "%Y%m%d").date().isoformat()
        except ValueError:
            print(f"  ⚠  Could not parse date from filename: {f.name}")
            continue

        if date in existing_dates:
            print(f"  ↩  {date} already transcribed — skipping")
            continue

        try:
            text = transcribe_audio(f)
            rows.append({"date": date, "raw_transcript": text})
            print(f"  ✓  Transcribed → {date}")
        except Exception as e:
            print(f"  ⚠  Transcribe error for {f.name}: {e}")

    if not rows:
        print("No new audio to transcribe.")
        return

    new_df = pd.DataFrame(rows)
    merged = new_df.groupby("date")["raw_transcript"].apply(lambda x: " ".join(x)).reset_index()

    # Append to existing file rather than overwrite
    if RAW_TRANSCRIPTS.exists():
        existing = pd.read_csv(RAW_TRANSCRIPTS)
        merged = pd.concat([existing, merged], ignore_index=True)
        merged = merged.drop_duplicates("date").sort_values("date")

    merged.to_csv(RAW_TRANSCRIPTS, index=False)
    print(f"✅ Merged transcripts saved — {len(rows)} new file(s) processed.")


def analyze_transcripts(force: bool = False):
    """
    Runs PROMPT_MARKET and PROMPT_STOCKS on each transcript.
    Skips dates already present in market_thoughts.csv / stock_thoughts.csv.
    If force=True, re-processes ALL dates, overwriting existing rows.
    """
    if not RAW_TRANSCRIPTS.exists():
        print("No raw_transcripts.csv found. Run merge_daily_transcripts() first.")
        return

    raw_df = pd.read_csv(RAW_TRANSCRIPTS)

    if force:
        existing_market_dates = set()
        existing_stock_dates  = set()
        dates_to_process = raw_df
        print("⚠️  force=True — re-processing ALL dates, existing rows will be overwritten.")
    else:
        existing_market_dates = get_existing_dates(MARKET_THOUGHTS)
        existing_stock_dates  = get_existing_dates(STOCK_THOUGHTS)

        # Process a date if it's missing from either output file
        dates_to_process = raw_df[
            ~raw_df["date"].astype(str).isin(existing_market_dates & existing_stock_dates)
        ]

        if dates_to_process.empty:
            print("All transcript dates already analyzed — nothing to do.")
            return

    print(f"Processing {len(dates_to_process)} new date(s)...")

    market_rows = []
    stock_rows  = []

    for _, row in dates_to_process.iterrows():
        date = str(row["date"])
        text = str(row["raw_transcript"])
        print(f"\n  → {date}")

        # ── Market thoughts ───────────────────────────────────────────────────
        if date not in existing_market_dates:
            m_resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": PROMPT_MARKET.format(transcript=text)}],
                temperature=0,
            )
            m_json = parse_json_response(
                m_resp.choices[0].message.content,
                fallback={
                    "market_sentiment": "", "emotional_state": "",
                    "emotional_intensity": 0, "influences": "",
                    "key_themes": "", "technical_analysis": "",
                    "raw_transcript": text,
                }
            )
            m_json["date"] = date
            market_rows.append(m_json)
            print(f"     market: {m_json.get('market_sentiment','')} / "
                  f"{m_json.get('emotional_state','')} / "
                  f"intensity={m_json.get('emotional_intensity','')}")

        # ── Stock thoughts ────────────────────────────────────────────────────
        if date not in existing_stock_dates:
            s_resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": PROMPT_STOCKS.format(transcript=text)}],
                temperature=0,
            )
            s_json = parse_json_response(
                s_resp.choices[0].message.content,
                fallback=[{
                    "stock": "", "ticker_confidence": "low",
                    "raw_thoughts": text, "summarized_thoughts": "",
                    "trade_rationale_explained": False,
                    "Tightness": "", "Ignition": "", "Group": "", "Earnings": "", "RS": "",
                }]
            )
            if isinstance(s_json, dict):   # model occasionally returns object instead of array
                s_json = [s_json]
            for s in s_json:
                s["date"] = date
            stock_rows.extend(s_json)
            tickers = [s.get("stock", "?") for s in s_json]
            low_conf = [s.get("stock","?") for s in s_json if s.get("ticker_confidence") == "low"]
            print(f"     stocks ({len(tickers)}): {', '.join(tickers)}"
                  + (f"  ⚠ low-conf: {', '.join(low_conf)}" if low_conf else ""))

    # ── Append to existing CSVs ───────────────────────────────────────────────
    if market_rows:
        new_market = pd.DataFrame(market_rows)
        if MARKET_THOUGHTS.exists() and not force:
            new_market = pd.concat([pd.read_csv(MARKET_THOUGHTS), new_market], ignore_index=True)
        elif MARKET_THOUGHTS.exists() and force:
            existing = pd.read_csv(MARKET_THOUGHTS)
            reprocessed_dates = set(new_market["date"].astype(str))
            existing = existing[~existing["date"].astype(str).isin(reprocessed_dates)]
            new_market = pd.concat([existing, new_market], ignore_index=True)
        new_market = new_market.drop_duplicates("date", keep="last").sort_values("date")
        new_market.to_csv(MARKET_THOUGHTS, index=False)
        verb = "Overwrote" if force else "Appended"
        print(f"\n✅ {verb} {len(market_rows)} row(s) in market_thoughts.csv")

    if stock_rows:
        new_stocks = pd.DataFrame(stock_rows)
        if STOCK_THOUGHTS.exists() and not force:
            new_stocks = pd.concat([pd.read_csv(STOCK_THOUGHTS), new_stocks], ignore_index=True)
        elif STOCK_THOUGHTS.exists() and force:
            existing = pd.read_csv(STOCK_THOUGHTS)
            reprocessed_dates = set(new_stocks["date"].astype(str))
            existing = existing[~existing["date"].astype(str).isin(reprocessed_dates)]
            new_stocks = pd.concat([existing, new_stocks], ignore_index=True)
        new_stocks = new_stocks.drop_duplicates(["date", "stock"], keep="last").sort_values(["date", "stock"])
        new_stocks.to_csv(STOCK_THOUGHTS, index=False)
        verb = "Overwrote" if force else "Appended"
        print(f"✅ {verb} {len(stock_rows)} row(s) in stock_thoughts.csv")


def run_bias_analyses():
    """
    Runs PROMPT_MARKET_ANALYSIS and PROMPT_STOCK_ANALYSIS on full historical data.
    Injects actual date range from the CSV into each prompt.
    Saves output as proper JSON files (was writing raw string in original).
    """
    # Pull date range from market_thoughts for context injection
    date_start, date_end = "unknown", "unknown"
    if MARKET_THOUGHTS.exists():
        mdf = pd.read_csv(MARKET_THOUGHTS)
        if not mdf.empty and "date" in mdf.columns:
            dates = pd.to_datetime(mdf["date"], errors="coerce").dropna()
            if not dates.empty:
                date_start = dates.min().strftime("%Y-%m-%d")
                date_end   = dates.max().strftime("%Y-%m-%d")
    print(f"Running bias analyses over {date_start} → {date_end}")

    # ── Market bias analysis ───────────────────────────────────────────────────
    if MARKET_THOUGHTS.exists():
        with open(MARKET_THOUGHTS) as f:
            market_data = f.read()

        m_resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                PROMPT_MARKET_ANALYSIS.format(
                    date_start=date_start,
                    date_end=date_end,
                ) + f"\n\nDATA:\n{market_data}"
            }],
            temperature=0,
        )
        m_result = parse_json_response(m_resp.choices[0].message.content, fallback={})
        with open(MARKET_ANALYSIS, "w") as f:
            json.dump(m_result, f, indent=2)
        print(f"✅ Market bias analysis saved to {MARKET_ANALYSIS}")

    # ── Stock bias analysis ────────────────────────────────────────────────────
    if STOCK_THOUGHTS.exists():
        with open(STOCK_THOUGHTS) as f:
            stock_data = f.read()

        s_resp = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content":
                PROMPT_STOCK_ANALYSIS.format(
                    date_start=date_start,
                    date_end=date_end,
                ) + f"\n\nDATA:\n{stock_data}"
            }],
            temperature=0,
        )
        s_result = parse_json_response(s_resp.choices[0].message.content, fallback={})
        with open(STOCK_ANALYSIS, "w") as f:
            json.dump(s_result, f, indent=2)
        print(f"✅ Stock bias analysis saved to {STOCK_ANALYSIS}")


def main(force: bool = False):
    """
    Run the full pipeline.

    force=False (default): incremental — only processes new audio files and
                           new transcript dates. Safe to run daily.

    force=True:            re-processes everything from scratch. Clears
                           processed_files.csv, re-transcribes all audio,
                           and overwrites all rows in market_thoughts.csv
                           and stock_thoughts.csv.
    """
    if force:
        print("⚠️  FORCE MODE — re-processing all audio and transcripts")
        if PROCESSED_LOG.exists():
            PROCESSED_LOG.unlink()
            print(f"   Cleared {PROCESSED_LOG.name}")
        if RAW_TRANSCRIPTS.exists():
            RAW_TRANSCRIPTS.unlink()
            print(f"   Cleared {RAW_TRANSCRIPTS.name}")

    new_files = get_new_files(force=force)
    if new_files:
        move_and_log_new_files(new_files)
        merge_daily_transcripts()
        analyze_transcripts(force=force)
        run_bias_analyses()
        print("\n🏁 All journal processing complete.")
    elif force:
        # force=True but audio folder is empty — still re-analyze existing transcripts
        if RAW_TRANSCRIPTS.exists():
            print("No audio files found, but re-analyzing existing transcripts (force=True)")
            analyze_transcripts(force=True)
            run_bias_analyses()
        else:
            print("No audio files and no transcripts found.")
    else:
        print("No new voice memos to process.")


# ── Jupyter usage ──────────────────────────────────────────────────────────────
# Incremental (normal daily use):
#   analyze_transcripts()           ← new dates only
#   run_bias_analyses()             ← always full history
#
# Force re-run everything:
#   main(force=True)                ← nuclear option, re-does all audio + analysis
#
# Force re-analyze transcripts only (keep audio cache):
#   analyze_transcripts(force=True) ← re-runs LLM on all dates, keeps transcriptions

if __name__ == "__main__" and not any("ipykernel" in a for a in __import__("sys").argv):
    import argparse
    parser = argparse.ArgumentParser(description="Voice memo journal pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Re-process everything from scratch (ignores existing outputs)")
    args = parser.parse_args()
    main(force=args.force)
