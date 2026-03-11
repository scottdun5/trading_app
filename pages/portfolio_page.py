import streamlit as st
import pandas as pd
from pathlib import Path

from utils.etrade import (
    etrade_authorize_start,
    etrade_authorize_finish,
    current_portfolio_metrics,
    get_trade_list
)
from utils.config import CONSUMER_KEY, CONSUMER_SECRET, ACCOUNT_ID_KEY, OPEN_HEAT_THRESHOLD, NER_THRESHOLD, BASE_URL
from utils.trade_annotator import merge_annotations

# ── Path to raw_trades.csv ─────────────────────────────────────────────────────
# Adjust this if your data folder lives elsewhere relative to this file
DATA_DIR       = Path(__file__).parent.parent / "data"
TRADE_LOG_PATH = DATA_DIR / "raw_trades.csv"


def portfolio_page():
    st.title("📊 Portfolio Dashboard")

    # -----------------------
    # E*TRADE Login
    # -----------------------
    if "etrade_session" not in st.session_state:
        st.session_state.etrade_session = None

    if st.session_state.etrade_session is None:
        st.info("Please authenticate with E*TRADE to load your portfolio data.")

        if st.button("🔐 Start E*TRADE Login"):
            try:
                auth_url, oauth, key, secret = etrade_authorize_start(CONSUMER_KEY, CONSUMER_SECRET)
                st.session_state.oauth = oauth
                st.session_state.resource_owner_key = key
                st.session_state.resource_owner_secret = secret
                st.markdown(f"[Click here to authorize E*TRADE]({auth_url})")
                st.info("After authorizing, paste the verifier code below.")
            except Exception as e:
                st.error(f"Failed to start login: {e}")
                return

        verifier = st.text_input("Enter E*TRADE verifier code (PIN):")
        if verifier and "oauth" in st.session_state:
            try:
                session, access_token, access_token_secret = etrade_authorize_finish(
                    CONSUMER_KEY,
                    CONSUMER_SECRET,
                    st.session_state.oauth,
                    st.session_state.resource_owner_key,
                    st.session_state.resource_owner_secret,
                    verifier
                )
                st.session_state.etrade_session = session
                st.success("E*TRADE session authorized successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Authorization failed: {e}")
                return

        st.stop()

    session = st.session_state.etrade_session

    # -----------------------
    # Portfolio Metrics
    # -----------------------
    st.header("Current Portfolio Metrics")

    if st.button("🔄 Refresh Portfolio Data"):
        try:
            total_open_heat, total_ner, total_value, all_risk_total_value, merged_df = current_portfolio_metrics(ACCOUNT_ID_KEY, session, BASE_URL)
            st.session_state.portfolio_metrics = {
                "total_open_heat": total_open_heat,
                "total_ner": total_ner,
                "total_value": total_value,
                "all_risk_total_value": all_risk_total_value,
                "merged_df": merged_df,
            }
            st.success("Portfolio metrics updated!")
        except Exception as e:
            st.error(f"Failed to refresh portfolio metrics: {e}")

    if "portfolio_metrics" in st.session_state:
        metrics = st.session_state.portfolio_metrics
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Open Heat",
            f"{metrics['total_open_heat']:.2f}",
            None,
            delta_color="inverse" if metrics["total_open_heat"] > OPEN_HEAT_THRESHOLD else "normal",
        )
        col2.metric(
            "Total NER",
            f"{metrics['total_ner']:.2f}",
            None,
            delta_color="inverse" if metrics["total_ner"] > NER_THRESHOLD else "normal",
        )
        col3.metric("Total Value", f"${metrics['total_value']:,.0f}")

        if metrics["total_open_heat"] > OPEN_HEAT_THRESHOLD:
            st.warning("⚠️ Total Open Heat exceeds threshold — consider reducing exposure.")
        if metrics["total_ner"] > NER_THRESHOLD:
            st.warning("⚠️ Total NER exceeds threshold — review risk balance.")

        st.dataframe(metrics["merged_df"], use_container_width=True)

    # -----------------------
    # Historical Performance
    # -----------------------
    st.header("📈 Historical Trades and Performance")

    if st.button("📥 Load Trade History"):
        try:
            trades_df, monthly_df = get_trade_list(ACCOUNT_ID_KEY, session)

            # Add month column if not present (used for filtering below)
            if "month" not in trades_df.columns:
                trades_df["buy_date"] = pd.to_datetime(trades_df["buy_date"])
                trades_df["month"] = trades_df["buy_date"].dt.to_period("M").astype(str)

            # ── Preserve manual annotations before overwriting CSV ─────────────
            trades_df = merge_annotations(trades_df, TRADE_LOG_PATH)

            # ── Save to CSV ────────────────────────────────────────────────────
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            save_df = trades_df.copy()
            if "buy_date" in save_df.columns:
                save_df["buy_date"] = pd.to_datetime(save_df["buy_date"]).dt.strftime("%m/%d/%y")
            save_df.to_csv(TRADE_LOG_PATH, index=False)

            st.session_state.trade_data   = trades_df
            st.session_state.monthly_data = monthly_df
            st.success(f"✅ Trade history loaded and saved to `{TRADE_LOG_PATH}` ({len(trades_df)} trades). Manual annotations preserved.")

        except Exception as e:
            st.error(f"Failed to load trade history: {e}")

    # Monthly summary
    if "monthly_data" in st.session_state:
        st.subheader("Monthly Performance Summary")
        st.dataframe(st.session_state.monthly_data, use_container_width=True)

    # Trade list with filters
    if "trade_data" in st.session_state:
        trades_df = st.session_state.trade_data
        st.subheader("Trade History")

        col1, col2 = st.columns(2)
        months  = sorted(trades_df["month"].dropna().unique())
        symbols = sorted(trades_df["symbol"].dropna().unique())

        selected_month  = col1.selectbox("Filter by Month",  ["All"] + list(months))
        selected_symbol = col2.selectbox("Filter by Ticker", ["All"] + list(symbols))

        filtered = trades_df.copy()
        if selected_month != "All":
            filtered = filtered[filtered["month"] == selected_month]
        if selected_symbol != "All":
            filtered = filtered[filtered["symbol"] == selected_symbol]

        st.dataframe(filtered, use_container_width=True, height=500)

    # ── Equity Curve ──────────────────────────────────────────────────────────
    st.header("📈 Equity Curve")

    try:
        from utils.config import STARTING_BALANCE
    except ImportError:
        STARTING_BALANCE = None

    if STARTING_BALANCE is None:
        st.info(
            "Run `calculate_starting_balance.py` once to compute your starting balance, "
            "then add `STARTING_BALANCE = <value>` to `utils/config.py`."
        )
    elif not TRADE_LOG_PATH.exists():
        st.warning(f"No trade log found at `{TRADE_LOG_PATH}`. Load trade history above first.")
    else:
        import plotly.graph_objects as go

        ec_df = pd.read_csv(TRADE_LOG_PATH, sep=None, engine="python")
        ec_df.columns = ec_df.columns.str.strip().str.lower()

        # Estimate close date from buy_date + avg_days_in_trade
        ec_df["buy_date"] = pd.to_datetime(ec_df["buy_date"], errors="coerce")
        ec_df["avg_days_in_trade"] = pd.to_numeric(ec_df["avg_days_in_trade"], errors="coerce").fillna(1)
        ec_df["close_date"] = ec_df["buy_date"] + pd.to_timedelta(ec_df["avg_days_in_trade"].round(), unit="D")
        ec_df["gain"] = pd.to_numeric(ec_df["gain"], errors="coerce").fillna(0.0)

        # Sort by estimated close date, assign trade number
        ec_df = ec_df.dropna(subset=["close_date"]).sort_values("close_date").reset_index(drop=True)
        ec_df["trade_num"] = ec_df.index + 1

        # Cumulative P&L → equity → % gain from starting balance
        ec_df["cum_pnl"]      = ec_df["gain"].cumsum()
        ec_df["equity"]       = STARTING_BALANCE + ec_df["cum_pnl"]
        ec_df["pct_from_start"] = (ec_df["equity"] - STARTING_BALANCE) / STARTING_BALANCE * 100

        # Anchor point: day before first trade, 0%
        anchor = pd.DataFrame([{
            "close_date":     ec_df["close_date"].iloc[0] - pd.Timedelta(days=1),
            "pct_from_start": 0.0,
            "equity":         STARTING_BALANCE,
            "cum_pnl":        0.0,
            "trade_num":      0,
            "symbol":         "",
            "gain":           0.0,
        }])
        plot_df = pd.concat([anchor, ec_df], ignore_index=True)

        # Color: green above 0, red below
        current_pct  = plot_df["pct_from_start"].iloc[-1]
        line_color   = "#26a69a"                  if current_pct >= 0 else "#ef5350"
        fill_color   = "rgba(38, 166, 154, 0.15)" if current_pct >= 0 else "rgba(239, 83, 80, 0.15)"

        fig = go.Figure()

        # Zero reference line
        fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.25)", line_width=1)

        # Filled area under curve
        fig.add_trace(go.Scatter(
            x=plot_df["close_date"],
            y=plot_df["pct_from_start"],
            mode="lines",
            line=dict(color=line_color, width=2),
            fill="tozeroy",
            fillcolor=fill_color,
            name="Equity %",
            customdata=plot_df[["equity", "cum_pnl", "trade_num", "symbol", "gain"]].values,
            hovertemplate=(
                "<b>%{x|%b %d, %Y}</b><br>"
                "Return: <b>%{y:.2f}%</b><br>"
                "Equity: $%{customdata[0]:,.0f}<br>"
                "Total P&L: $%{customdata[1]:,.0f}<br>"
                "Trade #%{customdata[2]}: %{customdata[3]} "
                "($%{customdata[4]:,.0f})<extra></extra>"
            ),
        ))

        # Drawdown shading: highlight regions below 0
        below_zero = plot_df[plot_df["pct_from_start"] < 0]
        if not below_zero.empty:
            fig.add_trace(go.Scatter(
                x=plot_df["close_date"],
                y=plot_df["pct_from_start"].clip(upper=0),
                mode="none",
                fill="tozeroy",
                fillcolor="rgba(239,83,80,0.2)",
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=0, r=0, t=40, b=0),
            title=dict(
                text=f"Equity Curve — {current_pct:+.2f}% from starting balance (${STARTING_BALANCE:,.0f})",
                font=dict(size=14),
            ),
            xaxis=dict(title="", showgrid=False, zeroline=False),
            yaxis=dict(title="% Gain / Loss", ticksuffix="%", showgrid=True,
                       gridcolor="rgba(255,255,255,0.07)"),
            hovermode="x unified",
            showlegend=False,
        )

        st.plotly_chart(fig, use_container_width=True)

        # Summary metrics below chart
        total_trades  = len(ec_df)
        winners       = (ec_df["gain"] > 0).sum()
        win_rate      = winners / total_trades * 100 if total_trades else 0
        peak_pct      = plot_df["pct_from_start"].max()
        trough_pct    = plot_df["pct_from_start"].min()
        max_drawdown  = plot_df["pct_from_start"].min() - plot_df["pct_from_start"].cummax().min()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return",  f"{current_pct:+.2f}%")
        m2.metric("Win Rate",      f"{win_rate:.1f}%", f"{winners}W / {total_trades - winners}L")
        m3.metric("Peak Return",   f"{peak_pct:+.2f}%")
        m4.metric("Max Drawdown",  f"{trough_pct:.2f}%")


if __name__ == "__main__":
    portfolio_page()
