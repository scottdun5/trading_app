import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import pandas as pd
import os

st.set_page_config(layout="wide")

# --- Paths ---
DATA_DIR = "data"
IMG_DIR = "images"

# --- Load master dataframe ---
master_df = pd.read_csv(os.path.join(DATA_DIR, "master_df.csv"))

# master_df contains:
#   date: real datetime (string in CSV)
#   date_str: display-only version ("YYYY-MM-DD")

master_df["date"] = pd.to_datetime(master_df["date"])
master_df["date_str"] = master_df["date"].dt.strftime("%Y-%m-%d")
trades_path = os.path.join(DATA_DIR, "raw_trades.csv")

def load_stock_study():
    #path = os.path.join(DATA_DIR, "Stock Study.xlsx")
    #df = pd.read_excel(path, sheet_name="Stock Study")
    path = os.path.join(DATA_DIR, "buy_details.csv")
    df = pd.read_csv(path)

    #df.rename(columns={"Buy Date": "date", "Stock": "ticker"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    #df["ATR_gain"] = None  # placeholder for future data
    return df

#@st.cache_data
setups_df = load_stock_study()
trades_df = pd.read_csv(trades_path)
trades_df["date"] = pd.to_datetime(trades_df["buy_date"]).dt.strftime("%Y-%m-%d")

# ========================================================================
#   LAYOUT
# ========================================================================
top_left, top_right = st.columns([1, 1])

# -----------------------------
#  TOP LEFT: MASTER TABLE
# -----------------------------
with top_left:
    st.subheader("Master Table")

    # Column definitions — pin BOTH left
    column_defs = [
        {"headerName": "date", "field": "date_str", "pinned": "left", "width": 120},
        {"headerName": "setup_count", "field": "setup_count", "pinned": "left", "width": 110},
    ]

    # Add all other columns dynamically
    for col in master_df.columns:
        if col not in ["date", "date_str", "setup_count"]:
            column_defs.append({"headerName": col, "field": col})

    grid_options = {
        "columnDefs": column_defs,
        "rowSelection": "single",
        "suppressRowClickSelection": False,
    }

    grid_response = AgGrid(
        master_df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        height=600
    )

    # --- YOUR ORIGINAL WORKING SELECTION LOGIC ---
    selected = grid_response.get("selected_rows", [])

    if selected is not None and not selected.empty:
        row = selected.iloc[0]
        selected_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
    else:
        selected_date = None


# -----------------------------
#  TOP RIGHT: INDEX/BREADTH IMAGES
# -----------------------------
with top_right:
    st.subheader("Charts for Selected Date")

    img_types = ['qqq', 'spy', 'iwm', 'rsp', 'qqqe',
                 'qqq_breadth', 'spy_breadth']

    if selected_date:
        cols_per_row = 2
        row_cols = None

        for i, img_type in enumerate(img_types):
            # Start a new row every cols_per_row images
            if i % cols_per_row == 0:
                row_cols = st.columns(cols_per_row)

            img_name = f"{img_type}_{selected_date}.jpg"
            img_path = os.path.join(IMG_DIR, img_type, img_name)

            if os.path.exists(img_path):
                cap = img_type.upper()
                # Caption inside the column
                row_cols[i % cols_per_row].markdown(f"<div style='text-align: center; font-weight: bold;'>{cap}</div>", unsafe_allow_html=True)
                row_cols[i % cols_per_row].image(img_path, use_container_width=True)

            else:
                row_cols[i % cols_per_row].write(f"Image not found: {img_name}")
    else:
        st.write("Select a date to view charts.")



# ========================================================================
#  SECOND ROW — SETUPS TABLE + PER-TICKER CHARTS
# ========================================================================

bottom_left, bottom_right = st.columns([1, 1])

# -----------------------------
#  BOTTOM LEFT: TICKERS FOR SELECTED DATE
# -----------------------------
with bottom_left:
    st.subheader("Setups for Selected Date")

    # Columns we WANT to show in this table
    DESIRED_COLS = ["Stock", "date", "ATR Gain Close", "% Gain Close", "21 EMA ATR Distance", "50 SMA ATR Distance"]

    # AgGrid column definitions
    column_defs2 = [
        {"headerName": "Ticker", "field": "Stock", "pinned": "left", "width": 120},
        {"headerName": "Date", "field": "date", "width": 110},
        {"headerName": "ATR Gain", "field": "ATR Gain Close", "width": 110},
        {"headerName": "% Gain", "field": "% Gain Close", "width": 110},
        {"headerName": "21 EMA ATR Distance", "field": "21 EMA ATR Distance", "width": 110},
        {"headerName": "50 SMA ATR Distance", "field": "50 SMA ATR Distance", "width": 110},
    ]

    # ----------------------------------------------------------------
    # Load data if a date is selected
    # ----------------------------------------------------------------
    if selected_date:

        setups_df["date"] = pd.to_datetime(setups_df["date"]).dt.strftime("%Y-%m-%d")
        filtered_df = setups_df[setups_df["date"] == selected_date]

        if not filtered_df.empty:

            # ---- Keep only the columns we want in the grid ----
            filtered_df = filtered_df[DESIRED_COLS].copy()

            # ---- Sort by ATR Gain descending ----
            filtered_df = filtered_df.sort_values("ATR Gain Close", ascending=False)

            # ---- Build the AgGrid options ----
            gb2 = GridOptionsBuilder.from_dataframe(filtered_df)
            gb2.column_defs = column_defs2

            # Hide any columns not explicitly listed
            for col in filtered_df.columns:
                if col not in [c["field"] for c in column_defs2]:
                    gb2.configure_column(col, hide=True)

            gb2.configure_selection("single", use_checkbox=False)
            gb2.configure_grid_options(suppressRowClickSelection=False)

            grid2 = AgGrid(
                filtered_df,
                gridOptions=gb2.build(),
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                allow_unsafe_jscode=True,
            )

            selected_setup = grid2.get("selected_rows", [])
            if selected_setup is not None and not selected_setup.empty:
                ticker_row = selected_setup.iloc[0]
                selected_ticker = ticker_row["Stock"]
            else:
                selected_ticker = None
        else:
            st.write("No setups for this date.")
            selected_ticker = None

    else:
        st.write("Select a date above.")
        selected_ticker = None


# -----------------------------
#  BOTTOM RIGHT: DAILY + WEEKLY CHARTS FOR TICKER
# -----------------------------
with bottom_right:
    st.subheader("Ticker Charts")

    if selected_date and selected_ticker:
        #date_fmt = selected_date.replace("-", "_")

        # DAILY chart
        daily_file = f"{selected_ticker}_{selected_date}_daily.jpg"
        daily_path = os.path.join(IMG_DIR, "setup_charts", "daily", daily_file)

        # WEEKLY chart
        weekly_file = f"{selected_ticker}_{selected_date}_weekly.jpg"
        weekly_path = os.path.join(IMG_DIR, "setup_charts", "weekly", weekly_file)

        col1, col2 = st.columns(2)

        if os.path.exists(daily_path):
            col1.caption("Daily Chart") 
            col1.image(daily_path, use_container_width=True)
        else:
            col1.write(f"Daily chart missing: {daily_file}")

        if os.path.exists(weekly_path):
            col2.caption("Weekly Chart") 
            col2.image(weekly_path, use_container_width=True)
        else:
            col2.write(f"Weekly chart missing: {weekly_file}")

    else:
        st.write("Select a ticker to view its charts.")


# ========================================================================
#  THIRD ROW — TRADES TABLE + PER-TICKER CHARTS
# ========================================================================

bottom_left, bottom_right = st.columns([1, 1])

# -----------------------------
#  BOTTOM LEFT: TICKERS FOR SELECTED DATE
# -----------------------------
with bottom_left:
    st.subheader("Trades on Selected Date")

    # Columns we WANT to show in this table
    DESIRED_COLS = ["symbol", "date"] #, "ATR Gain Close", "% Gain Close"]

    # AgGrid column definitions
    column_defs2 = [
        {"headerName": "Ticker", "field": "symbol", "pinned": "left", "width": 120},
        {"headerName": "Date", "field": "date", "width": 110},
       # {"headerName": "ATR Gain", "field": "ATR Gain Close", "width": 110},
       # {"headerName": "% Gain", "field": "% Gain Close", "width": 110},
    ]

    # ----------------------------------------------------------------
    # Load data if a date is selected
    # ----------------------------------------------------------------
    if selected_date:

        trades_df["date"] = pd.to_datetime(trades_df["date"]).dt.strftime("%Y-%m-%d")
        filtered_df = trades_df[trades_df["date"] == selected_date]

        if not filtered_df.empty:

            # ---- Keep only the columns we want in the grid ----
            filtered_df = filtered_df[DESIRED_COLS].copy()


            # ---- Build the AgGrid options ----
            gb2 = GridOptionsBuilder.from_dataframe(filtered_df)
            gb2.column_defs = column_defs2

            # Hide any columns not explicitly listed
            for col in filtered_df.columns:
                if col not in [c["field"] for c in column_defs2]:
                    gb2.configure_column(col, hide=True)

            gb2.configure_selection("single", use_checkbox=False)
            gb2.configure_grid_options(suppressRowClickSelection=False)

            grid2 = AgGrid(
                filtered_df,
                gridOptions=gb2.build(),
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                height=300,
                allow_unsafe_jscode=True,
            )

            selected_setup = grid2.get("selected_rows", [])
            if selected_setup is not None and not selected_setup.empty:
                ticker_row = selected_setup.iloc[0]
                selected_ticker = ticker_row["symbol"]
            else:
                selected_ticker = None
        else:
            st.write("No setups for this date.")
            selected_ticker = None

    else:
        st.write("Select a date above.")
        selected_ticker = None


# -----------------------------
#  BOTTOM RIGHT: DAILY + WEEKLY CHARTS FOR TICKER
## need to edit this to check both trades folder and normal folder for if the chart exists
# -----------------------------
with bottom_right:
	st.subheader("Ticker Charts")

	if selected_date and selected_ticker:
		# DAILY chart
		daily_file = f"{selected_ticker}_{selected_date}_daily.jpg"
		daily_path = os.path.join(IMG_DIR, "setup_charts", "daily", daily_file)
		daily_path2 = os.path.join(IMG_DIR, "setup_charts", "daily", "trades", daily_file)

		# WEEKLY chart
		weekly_file = f"{selected_ticker}_{selected_date}_weekly.jpg"
		weekly_path = os.path.join(IMG_DIR, "setup_charts", "weekly", weekly_file)
		weekly_path2 = os.path.join(IMG_DIR, "setup_charts", "weekly", "trades", weekly_file)

		col1, col2 = st.columns(2)

		if os.path.exists(daily_path):
			col1.caption("Daily Chart")
			col1.image(daily_path, use_container_width=True)
		elif os.path.exists(daily_path2):
			col1.caption("Daily Chart")
			col1.image(daily_path2, use_container_width=True)
		else:
			col1.write(f"Daily chart missing: {daily_file}")

		if os.path.exists(weekly_path):
			col2.caption("Weekly Chart")
			col2.image(weekly_path, use_container_width=True)
		elif os.path.exists(weekly_path2):
			col2.caption("Weekly Chart")
			col2.image(weekly_path2, use_container_width=True)
		else:
			col2.write(f"Weekly chart missing: {weekly_file}")

	else:
		st.write("Select a ticker to view its charts.")
