"""
voice_analytics.py
All visualization tabs — corrected funnel, off-watchlist discipline charts,
protocol score trends, full TIGERS analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
import re

TIGERS = ["Tightness","Ignition","Group","Earnings","RS"]

EMOTION_COLORS = {
    "confident":       "#4ade80", "focused":       "#22d3ee",
    "neutral":         "#9ca3af", "hesitant":      "#fbbf24",
    "anxious":         "#f97316", "frustrated":    "#ef4444",
    "fomo":            "#f59e0b", "overconfident": "#a855f7",
    "distracted":      "#6b7280", "revenge_trading":"#dc2626",
}

OUTCOME_COLORS = {
    "correct_select":  "#4ade80",
    "correct_skip":    "#22d3ee",
    "false_positive":  "#f59e0b",
    "missed":          "#ff6b6b",
    "impulsive_entry": "#dc2626",
    "uncategorized":   "#374151",
}
OUTCOME_LABELS = {
    "correct_select":  "✅ Correct Select",
    "correct_skip":    "✅ Correct Skip",
    "false_positive":  "⚠️ False Positive",
    "missed":          "❌ Missed",
    "impulsive_entry": "🚨 Impulsive Entry",
    "uncategorized":   "— Uncategorized",
}


def _dark(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#e8eaf0"),
        margin=dict(l=16, r=16, t=40, b=16),
    )
    fig.update_xaxes(gridcolor="#1f2330", zerolinecolor="#252830")
    fig.update_yaxes(gridcolor="#1f2330", zerolinecolor="#252830")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# KPIs
# ═════════════════════════════════════════════════════════════════════════════

def render_kpis(master: pd.DataFrame, mdf: pd.DataFrame, protocol: pd.DataFrame = None):
    if master.empty:
        return
    cols = st.columns(7)
    with cols[0]:
        st.metric("Memos", mdf["date"].nunique() if not mdf.empty else 0)
    with cols[1]:
        avg = mdf["sentiment_score"].mean() if not mdf.empty and "sentiment_score" in mdf.columns else 0
        st.metric("Avg Sentiment",
                  "Bullish" if avg > 0.1 else "Bearish" if avg < -0.1 else "Neutral",
                  f"{avg:+.2f}")
    with cols[2]:
        te = (mdf["emotional_state"].value_counts().idxmax()
              if not mdf.empty and "emotional_state" in mdf.columns else "—")
        st.metric("Top Emotion", str(te).title())
    with cols[3]:
        st.metric("Tickers Tracked", master["stock"].nunique())
    with cols[4]:
        st.metric("Avg TIGERS", f"{master['tigers_score'].mean():.1f}/5"
                  if "tigers_score" in master.columns else "—")
    with cols[5]:
        if "win" in master.columns:
            wr = master.dropna(subset=["win"])["win"].mean()
            st.metric("Win Rate", f"{wr*100:.0f}%", delta=f"{wr*100-50:.0f}pp vs 50%")
    with cols[6]:
        if protocol is not None and not protocol.empty:
            owl = int(protocol.get("off_watchlist_count", pd.Series(0)).sum())
            nft = int(protocol.get("no_focus_trade", pd.Series(0)).sum())
            # Show the more serious violation if both exist
            if nft > 0:
                st.metric("No-Focus Trades", nft,
                          delta=f"⚠ {nft} day(s)", delta_color="inverse")
            else:
                st.metric("Off-WL Trades", owl,
                          delta=f"-{owl}" if owl > 0 else "✓ Clean",
                          delta_color="inverse")


# ═════════════════════════════════════════════════════════════════════════════
# TAB: MARKET PULSE
# ═════════════════════════════════════════════════════════════════════════════

def render_market_pulse(master: pd.DataFrame, mdf: pd.DataFrame):
    if mdf.empty:
        st.info("Load market_thoughts.csv to begin.")
        return

    st.markdown("### Sentiment Drift Index")
    st.caption("10-day rolling sentiment vs SPY cumulative return. Are you leading or lagging the market?")
    mdf_s = mdf.copy().sort_values("date")
    mdf_s["sentiment_roll"] = mdf_s["sentiment_score"].rolling(10, min_periods=1).mean()
    spy = _mock_spy(mdf_s["date"])
    spy_ret = spy.pct_change().fillna(0).cumsum() * 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=mdf_s["date"], y=spy_ret, name="SPY Cumulative %",
                              line=dict(color="#4ade80", width=1.5), opacity=0.7), secondary_y=False)
    fig.add_trace(go.Scatter(x=mdf_s["date"], y=mdf_s["sentiment_roll"],
                              name="Your Sentiment (10d avg)",
                              line=dict(color="#00d4ff", width=2.5),
                              fill="tozeroy", fillcolor="rgba(0,212,255,0.06)"), secondary_y=True)
    fig.add_hline(y=0, line_dash="dot", line_color="#374151")
    fig.update_yaxes(title_text="SPY Return %", secondary_y=False, gridcolor="#1f2330")
    fig.update_yaxes(title_text="Sentiment", secondary_y=True, range=[-1.2,1.2],
                      gridcolor="rgba(0,0,0,0)")
    _dark(fig)
    fig.update_layout(height=300, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    # Temporal sentiment stream
    st.markdown("### Temporal Sentiment Stream")
    st.caption("Stacked weekly emotion composition — reveals emotional cycles.")
    mdf_e = mdf.copy()
    mdf_e["week"] = mdf_e["date"].dt.to_period("W").dt.start_time
    weekly_em = pd.crosstab(mdf_e["week"], mdf_e["emotional_state"])
    weekly_em = weekly_em.reindex(columns=[e for e in EMOTION_COLORS if e in weekly_em.columns], fill_value=0)
    fig2 = go.Figure()
    for em in weekly_em.columns:
        hex_color = EMOTION_COLORS.get(em, "#6b7280")
        # Convert hex to rgba for Plotly compatibility (older versions reject 8-digit hex)
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        fill_rgba = f"rgba({r},{g},{b},0.8)"
        fig2.add_trace(go.Scatter(
            x=weekly_em.index, y=weekly_em[em],
            name=em.replace("_"," ").title(), mode="lines",
            stackgroup="one", line=dict(width=0.5),
            fillcolor=fill_rgba,
        ))
    _dark(fig2)
    fig2.update_layout(height=260, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig2, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("### Sentiment Distribution")
        vc = mdf["market_sentiment"].value_counts().reset_index()
        vc.columns = ["sentiment","count"]
        cmap = {"bullish":"#4ade80","bearish":"#ff6b6b","neutral":"#9ca3af","uncertain":"#fbbf24"}
        fig3 = px.bar(vc, x="sentiment", y="count", color="sentiment", color_discrete_map=cmap)
        _dark(fig3); fig3.update_layout(showlegend=False, height=240)
        st.plotly_chart(fig3, use_container_width=True)
    with cb:
        st.markdown("### Influence Sources")
        infl = mdf["influences"].value_counts().reset_index()
        infl.columns = ["source","count"]
        fig4 = px.pie(infl, names="source", values="count", hole=0.55)
        _dark(fig4)
        fig4.update_traces(textfont_size=11, marker=dict(line=dict(color="#0a0c10", width=2)))
        fig4.update_layout(height=240)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Key Themes")
    themes = []
    for t in mdf["key_themes"].dropna():
        themes.extend([x.strip() for x in str(t).split(",") if x.strip()])
    top = dict(sorted(Counter(themes).items(), key=lambda x: -x[1])[:15])
    fig5 = px.bar(x=list(top.values()), y=list(top.keys()), orientation="h",
                   color=list(top.values()),
                   color_continuous_scale=["#1a1d25","#7c5cfc","#00d4ff"])
    _dark(fig5)
    fig5.update_layout(showlegend=False, height=340, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig5, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB: PSYCHOLOGY
# ═════════════════════════════════════════════════════════════════════════════

def render_psychology(master: pd.DataFrame, mdf: pd.DataFrame):
    if mdf.empty:
        st.info("Load market_thoughts.csv to begin.")
        return

    # Word clouds or fallback bar charts
    st.markdown("### Word Cloud by Emotional State")
    st.caption("Language patterns reveal cognitive habits under different emotional states.")

    from collections import Counter
    import math

    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        import io

        # Comprehensive stopwords — English function words + trading filler
        STOP = {
            # Articles / determiners
            "a","an","the","this","that","these","those","some","any","each","every",
            "both","few","more","most","other","such","no","nor","not","only","own",
            "same","so","than","too","very","s","t","will","just","don","should",
            # Pronouns
            "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
            "yourself","yourselves","he","him","his","himself","she","her","hers",
            "herself","it","its","itself","they","them","their","theirs","themselves",
            "what","which","who","whom","whose",
            # Prepositions / conjunctions
            "and","but","or","nor","for","yet","so","at","by","in","of","on","to",
            "up","as","be","do","if","is","am","are","was","were","be","been","being",
            "have","has","had","having","do","does","did","doing","would","could",
            "should","may","might","must","shall","will","can","need","dare","ought",
            "used","about","above","after","again","against","all","also","although",
            "among","around","because","before","between","during","from","here",
            "how","into","like","near","now","off","once","out","over","own","same",
            "since","then","there","through","under","until","upon","when","where",
            "while","with","without","within","down","per","via","re","get","got",
            "go","going","gone","come","came","know","think","look","looking","want",
            "use","using","make","making","said","say","says","even","back","still",
            "way","well","new","good","high","right","big","little","great","long",
            # Common speech fillers
            "um","uh","yeah","yep","okay","ok","like","really","pretty","kind",
            "basically","actually","literally","definitely","probably","maybe",
            "something","anything","everything","nothing","thing","things","lot",
            "lots","little","bit","bit","quite","rather","already","always","never",
            "often","usually","sometimes","today","yesterday","tomorrow","week",
            "month","year","day","time","ago","later","next","last","first","second",
            # Trading-specific filler (words that appear everywhere, add no signal)
            "stock","stocks","market","trade","trading","trader","chart","price",
            "share","shares","ticker","position","positions","name","names","one",
            "two","three","four","five","ten","hundred","thousand","million","point",
        }

        ems = mdf["emotional_state"].dropna().unique().tolist()
        sel = st.multiselect("Select emotions", ems, default=ems[:3] if len(ems)>=3 else ems)
        view_mode = st.radio("View as", ["☁️ Word Cloud", "📊 Bar Chart"], horizontal=True)

        if sel and "raw_transcript" in mdf.columns:

            # TF-IDF: compute IDF across all emotion groups
            all_docs = {}
            for em in ems:
                text = " ".join(mdf[mdf["emotional_state"]==em]["raw_transcript"].dropna().astype(str))
                words = [w for w in re.sub(r"[^a-zA-Z\s]"," ",text.lower()).split()
                         if w not in STOP and len(w) > 3]
                all_docs[em] = words

            N = len(all_docs)
            doc_freq = Counter()
            for words in all_docs.values():
                for w in set(words):
                    doc_freq[w] += 1
            idf = {w: math.log(N / df) for w, df in doc_freq.items()}

            cols = st.columns(min(len(sel), 3))
            for i, em in enumerate(sel[:3]):
                words = all_docs.get(em, [])
                if not words:
                    continue
                tf = Counter(words)
                tfidf = {w: tf[w] * idf.get(w, 0) for w in tf if idf.get(w, 0) > 0}
                if not tfidf:
                    continue

                c = EMOTION_COLORS.get(em, "#6b7280")
                with cols[i % 3]:
                    st.markdown(
                        f'<div style="text-align:center;font-family:Space Mono,monospace;'
                        f'font-size:11px;color:{c};margin-bottom:4px">{em.upper()}</div>',
                        unsafe_allow_html=True
                    )
                    if "Word Cloud" in view_mode:
                        wc = WordCloud(
                            width=500, height=260,
                            background_color="#111318",
                            colormap="cool" if i % 2 == 0 else "autumn",
                            max_words=50,
                            prefer_horizontal=0.8,
                        ).generate_from_frequencies(tfidf)
                        buf = io.BytesIO()
                        plt.figure(figsize=(5, 2.6), facecolor="#111318")
                        plt.imshow(wc, interpolation="bilinear")
                        plt.axis("off")
                        plt.tight_layout(pad=0)
                        plt.savefig(buf, format="png", bbox_inches="tight",
                                    facecolor="#111318", dpi=120)
                        plt.close()
                        buf.seek(0)
                        st.image(buf, use_container_width=True)
                    else:
                        top = sorted(tfidf.items(), key=lambda x: -x[1])[:20]
                        fw, fc = zip(*top)
                        fig_w = px.bar(x=list(fc), y=list(fw), orientation="h",
                                       color=list(fc),
                                       color_continuous_scale=["#1a1d25", c])
                        _dark(fig_w)
                        fig_w.update_layout(showlegend=False, coloraxis_showscale=False,
                                            yaxis=dict(autorange="reversed"), height=320,
                                            margin=dict(l=0,r=0,t=0,b=0))
                        st.plotly_chart(fig_w, use_container_width=True)

    except ImportError:
        st.info("Install `wordcloud matplotlib` for word cloud visuals. Showing bar chart instead.")
        em_sel = st.selectbox("Emotion", mdf["emotional_state"].dropna().unique())
        text = " ".join(mdf[mdf["emotional_state"]==em_sel]["raw_transcript"].dropna().astype(str))
        freq = Counter(w for w in re.sub(r"[^a-zA-Z\s]"," ",text.lower()).split()
                       if w not in STOP and len(w) > 3).most_common(20)
        if freq:
            fw, fc = zip(*freq)
            fig = px.bar(x=list(fc), y=list(fw), orientation="h",
                          color=list(fc), color_continuous_scale=["#1a1d25","#00d4ff"])
            _dark(fig)
            fig.update_layout(showlegend=False, coloraxis_showscale=False,
                               yaxis=dict(autorange="reversed"), height=320)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Emotion × Win Rate
    if "win" in master.columns and "day_emotional_state" in master.columns:
        st.markdown("### Emotional State vs Win Rate")
        st.caption("Win rate of trades entered under each emotional state. Your most important behavioral edge signal.")
        traded = master[master.get("did_trade", pd.Series(1, index=master.index)) == 1].dropna(subset=["win","day_emotional_state"])
        if not traded.empty:
            ewr = traded.groupby("day_emotional_state").agg(
                win_rate=("win","mean"), n=("win","count")
            ).reset_index()
            ewr = ewr[ewr["n"] >= 3]
            fig = go.Figure()
            fig.add_hline(y=0.5, line_dash="dot", line_color="#374151",
                           annotation_text="50%", annotation_position="right")
            fig.add_trace(go.Bar(
                x=ewr["day_emotional_state"], y=ewr["win_rate"],
                text=[f"{r:.0%} (n={n})" for r,n in zip(ewr["win_rate"],ewr["n"])],
                textposition="outside",
                marker_color=[EMOTION_COLORS.get(e,"#6b7280") for e in ewr["day_emotional_state"]],
            ))
            _dark(fig)
            fig.update_layout(yaxis=dict(tickformat=".0%", range=[0,1.15]), height=300)
            st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Emotion × Sentiment Cross")
        cross = pd.crosstab(mdf["emotional_state"], mdf["market_sentiment"])
        fig = px.imshow(cross, color_continuous_scale=["#0a0c10","#7c5cfc","#00d4ff"],
                         text_auto=True, aspect="auto")
        _dark(fig); fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Emotion Distribution")
        ec = mdf["emotional_state"].value_counts().reset_index()
        ec.columns = ["emotion","count"]
        fig = px.bar(ec, x="count", y="emotion", orientation="h",
                      color="emotion", color_discrete_map=EMOTION_COLORS)
        _dark(fig); fig.update_layout(showlegend=False, height=300, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB: SELECTION FUNNEL (corrected direction)
# ═════════════════════════════════════════════════════════════════════════════

def render_selection_funnel(master: pd.DataFrame, protocol: pd.DataFrame = None):
    if master.empty:
        st.info("Load data to begin.")
        return

    st.markdown("### The Corrected Selection Funnel")
    st.caption("Screener → Watchlist → Voice Memo Analysis → Trade Decision → Outcome")

    # ── Sankey: Watchlist → Memo → Trade → Win ─────────────────────────────
    st.markdown("#### Idea Lifecycle Sankey")
    wl_total      = int(master["on_watchlist"].sum())
    memo_covered  = int(master["memo_coverage"].sum())
    wl_uncovered  = int(master["watchlist_uncovered"].sum())
    organic_total = int(master["organic_idea"].sum())
    traded_wl     = int(master[(master["on_watchlist"]==1) & (master["did_trade"]==1)].shape[0])
    skipped_wl    = int(wl_total - traded_wl)
    traded_org    = int(master[(master["organic_idea"]==1) & (master["did_trade"]==1)].shape[0])

    won_wl  = int(master[(master["on_watchlist"]==1) & (master["did_trade"]==1) & (master["win"]==1)].shape[0]) if "win" in master.columns else 0
    lost_wl = traded_wl - won_wl
    won_org = int(master[(master["organic_idea"]==1) & (master["did_trade"]==1) & (master["win"]==1)].shape[0]) if "win" in master.columns else 0
    lost_org = traded_org - won_org

    # Node order: 0=Watchlist, 1=Covered in Memo, 2=Uncovered, 3=Organic Ideas,
    #             4=Traded (WL), 5=Skipped, 6=Traded (Organic),
    #             7=Won, 8=Lost
    node_labels = ["Watchlist","Covered in Memo","Silent Skip","Organic Ideas",
                   "Traded (Planned)","Skipped","Impulsive Trade","Won","Lost"]
    node_colors = ["#7c5cfc","#4ade80","#ff6b6b","#f59e0b",
                   "#22d3ee","#374151","#dc2626","#4ade80","#ff6b6b"]

    sources = [0, 0, 1, 1,  2,  3,  4,  4,  6,  6]
    targets = [1, 2, 4, 5,  4,  6,  7,  8,  7,  8]
    values  = [
        max(memo_covered,1), max(wl_uncovered,1),
        max(traded_wl,1),    max(skipped_wl,1),
        max(wl_uncovered,1), max(organic_total,1),
        max(won_wl,1),       max(lost_wl,1),
        max(won_org,1),      max(lost_org,1),
    ]
    link_colors = [
        "rgba(124,92,252,0.3)","rgba(255,107,107,0.25)",
        "rgba(34,211,238,0.3)","rgba(55,65,81,0.25)",
        "rgba(220,38,38,0.25)","rgba(220,38,38,0.3)",
        "rgba(74,222,128,0.4)","rgba(255,107,107,0.4)",
        "rgba(74,222,128,0.3)","rgba(255,107,107,0.3)",
    ]
    fig_sk = go.Figure(go.Sankey(
        node=dict(pad=20, thickness=18,
                  line=dict(color="#0a0c10", width=0.5),
                  label=node_labels, color=node_colors),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
    ))
    _dark(fig_sk)
    fig_sk.update_layout(height=380, title="Stock Idea Lifecycle: Watchlist → Win/Loss")
    st.plotly_chart(fig_sk, use_container_width=True)

    # ── Off-watchlist discipline ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚨 Off-Watchlist Discipline")
    st.caption("Any trade not on your watchlist is flagged here. The goal is zero over time.")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Off-Watchlist Entries Over Time")
        if protocol is not None and not protocol.empty and "off_watchlist_count" in protocol.columns:
            prot = protocol.copy().sort_values("date")
            prot["week"] = prot["date"].dt.to_period("W").dt.start_time
            weekly_owl = prot.groupby("week")["off_watchlist_count"].sum().reset_index()
            fig_owl = go.Figure()
            fig_owl.add_trace(go.Bar(
                x=weekly_owl["week"], y=weekly_owl["off_watchlist_count"],
                marker_color=[
                    "#ff6b6b" if v > 0 else "#1f2330"
                    for v in weekly_owl["off_watchlist_count"]
                ],
                name="Off-WL trades",
            ))
            # Rolling trend
            weekly_owl["roll"] = weekly_owl["off_watchlist_count"].rolling(4, min_periods=1).mean()
            fig_owl.add_trace(go.Scatter(
                x=weekly_owl["week"], y=weekly_owl["roll"],
                mode="lines", name="4-week avg",
                line=dict(color="#f59e0b", width=2, dash="dot"),
            ))
            _dark(fig_owl)
            fig_owl.update_layout(height=260,
                                   title="Weekly off-watchlist trade count (goal = 0)")
            st.plotly_chart(fig_owl, use_container_width=True)
        else:
            st.info("Protocol data not available yet.")

    with c2:
        st.markdown("#### Off-WL Frequency on High-Influence Days")
        st.caption("Do Twitter/alert days correlate with impulsive entries?")
        if protocol is not None and not protocol.empty and "external_influence_day" in protocol.columns:
            prot = protocol.copy()
            inf_grp = prot.groupby("external_influence_day").agg(
                avg_owl=("off_watchlist_count","mean"),
                days=("off_watchlist_count","count"),
            ).reset_index()
            inf_grp["label"] = inf_grp["external_influence_day"].map(
                {0:"No External Influence", 1:"External Influence Day"}
            )
            fig_inf = px.bar(inf_grp, x="label", y="avg_owl",
                              color="label",
                              color_discrete_map={
                                  "No External Influence": "#374151",
                                  "External Influence Day": "#ff6b6b",
                              },
                              text=[f"{v:.2f}/day" for v in inf_grp["avg_owl"]],
                              labels={"avg_owl":"Avg Off-WL Trades/Day","label":""})
            _dark(fig_inf); fig_inf.update_layout(showlegend=False, height=260)
            fig_inf.update_traces(textposition="outside")
            st.plotly_chart(fig_inf, use_container_width=True)
        else:
            st.info("Protocol + market data needed for this chart.")

    # Win rate: planned vs impulsive
    if "outcome_category" in master.columns and "win" in master.columns:
        st.markdown("#### Win Rate: Planned vs Impulsive Trades")
        planned   = master[(master["on_watchlist"]==1) & (master["did_trade"]==1)].dropna(subset=["win"])
        impulsive = master[master["outcome_category"]=="impulsive_entry"].dropna(subset=["win"])

        if not planned.empty or not impulsive.empty:
            wr_data = []
            if not planned.empty:
                wr_data.append({"type":"Planned (On Watchlist)",
                                "win_rate": planned["win"].mean(),
                                "n": len(planned)})
            if not impulsive.empty:
                wr_data.append({"type":"Impulsive (Off Watchlist)",
                                "win_rate": impulsive["win"].mean(),
                                "n": len(impulsive)})
            wr_df = pd.DataFrame(wr_data)
            fig_wr = go.Figure()
            fig_wr.add_hline(y=0.5, line_dash="dot", line_color="#374151",
                              annotation_text="50%")
            fig_wr.add_trace(go.Bar(
                x=wr_df["type"], y=wr_df["win_rate"],
                text=[f"{r:.0%} (n={n})" for r,n in zip(wr_df["win_rate"],wr_df["n"])],
                textposition="outside",
                marker_color=["#4ade80","#dc2626"],
            ))
            _dark(fig_wr)
            fig_wr.update_layout(yaxis=dict(tickformat=".0%", range=[0,1.1]), height=260)
            st.plotly_chart(fig_wr, use_container_width=True)

    # ── Protocol score trend ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Protocol Score Trend")
    st.caption("Days where you follow the full process (score=3) vs. cutting corners. Does it correlate with outcomes?")
    if protocol is not None and not protocol.empty:
        prot = protocol.copy().sort_values("date")
        prot["week"] = prot["date"].dt.to_period("W").dt.start_time
        weekly_score = prot.groupby("week").agg(
            avg_score=("protocol_score","mean"),
            off_wl=("off_watchlist_count","sum"),
            no_focus=("no_focus_trade","sum") if "no_focus_trade" in prot.columns else ("protocol_score","count"),
        ).reset_index()

        fig_ps = make_subplots(specs=[[{"secondary_y": True}]])
        fig_ps.add_trace(go.Bar(
            x=weekly_score["week"], y=weekly_score["avg_score"],
            name="Avg Protocol Score",
            marker_color=[
                "#4ade80" if s >= 2.5 else "#f59e0b" if s >= 1.5 else "#ff6b6b"
                for s in weekly_score["avg_score"]
            ],
        ), secondary_y=False)
        fig_ps.add_trace(go.Scatter(
            x=weekly_score["week"], y=weekly_score["off_wl"],
            mode="lines+markers", name="Off-WL Trades",
            line=dict(color="#f59e0b", width=2),
            marker=dict(size=5),
        ), secondary_y=True)
        if "no_focus" in weekly_score.columns:
            fig_ps.add_trace(go.Scatter(
                x=weekly_score["week"], y=weekly_score["no_focus"],
                mode="lines+markers", name="No-Focus Trades ⚠",
                line=dict(color="#dc2626", width=2, dash="dot"),
                marker=dict(size=6, symbol="x"),
            ), secondary_y=True)
        fig_ps.update_yaxes(title_text="Avg Protocol Score (0–3)", secondary_y=False,
                             gridcolor="#1f2330", range=[0, 3.2])
        fig_ps.update_yaxes(title_text="Violation Count", secondary_y=True,
                             gridcolor="rgba(0,0,0,0)")
        _dark(fig_ps)
        fig_ps.update_layout(height=320, legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig_ps, use_container_width=True)

        # No-focus trade callout table
        if "no_focus_trade" in prot.columns:
            nft_days = prot[prot["no_focus_trade"] == 1][["date","trades_taken","off_watchlist_count"]]
            if not nft_days.empty:
                st.error(f"⚠️ **{len(nft_days)} day(s) with trades but no focus list** — your most serious protocol violation.")
                st.dataframe(nft_days.sort_values("date", ascending=False),
                             use_container_width=True, hide_index=True)
            else:
                st.success("✅ No days with trades taken without a focus list.")

    # ── Memo coverage analysis ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Memo Coverage Analysis")
    st.caption("Gap 1: which watchlist stocks did you silently skip in your memo?")

    if "watchlist_uncovered" in master.columns:
        uncov = master[master["watchlist_uncovered"]==1]
        if not uncov.empty:
            uc1, uc2 = st.columns(2)
            with uc1:
                top_uncov = uncov["stock"].value_counts().head(15).reset_index()
                top_uncov.columns = ["stock","times_uncovered"]
                fig_uc = px.bar(top_uncov, x="stock", y="times_uncovered",
                                 color="times_uncovered",
                                 color_continuous_scale=["#1f2330","#f59e0b","#ff6b6b"],
                                 title="Most-skipped watchlist stocks")
                _dark(fig_uc); fig_uc.update_layout(showlegend=False, height=280)
                st.plotly_chart(fig_uc, use_container_width=True)
            with uc2:
                # Uncovered stocks that were then traded — highest risk pattern
                uncov_traded = master[master["uncovered_trade"]==1]
                if not uncov_traded.empty:
                    st.markdown("#### ⚠️ Traded Without Discussing")
                    st.caption("On watchlist, not covered in memo, but traded anyway — your highest-risk pattern.")
                    st.dataframe(
                        uncov_traded[["date","stock","tigers_score","win"]]
                        .sort_values("date", ascending=False)
                        .head(20),
                        use_container_width=True, hide_index=True
                    )
                else:
                    st.success("✅ No instances of trading without prior memo discussion found.")
        else:
            st.success("✅ All watchlist stocks covered in memos.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB: STOCK INTELLIGENCE
# ═════════════════════════════════════════════════════════════════════════════

def render_stock_intelligence(master: pd.DataFrame):
    if master.empty:
        st.info("Load data to begin.")
        return

    cf, cg = st.columns([2,1])
    with cf:
        st.markdown("### Ticker Mention Frequency")
        tc = master.groupby("stock").size().reset_index(name="mentions")
        tc = tc.sort_values("mentions", ascending=False).head(25)
        fig = px.bar(tc, x="stock", y="mentions", color="mentions",
                      color_continuous_scale=["#1a1d25","#7c5cfc","#00d4ff"])
        _dark(fig); fig.update_layout(showlegend=False, height=280)
        st.plotly_chart(fig, use_container_width=True)
    with cg:
        st.markdown("#### Top 10")
        for _, row in tc.head(10).iterrows():
            pct = row["mentions"] / tc["mentions"].max()
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:5px 0;border-bottom:1px solid #1f2330;">'
                f'<span style="font-family:Space Mono,monospace;font-size:11px;background:rgba(0,212,255,0.1);'
                f'color:#00d4ff;border:1px solid rgba(0,212,255,0.25);border-radius:5px;padding:2px 7px">'
                f'{row["stock"]}</span>'
                f'<div style="display:flex;align-items:center;gap:6px;">'
                f'<div style="width:{int(pct*70)}px;height:3px;background:#00d4ff;border-radius:2px"></div>'
                f'<span style="font-family:Space Mono,monospace;font-size:10px;color:#9ca3af">{int(row["mentions"])}</span>'
                f'</div></div>', unsafe_allow_html=True
            )

    # Weekly activity heatmap
    st.markdown("### Weekly Ticker Activity Heatmap")
    mw = master.copy()
    mw["week"] = mw["date"].dt.to_period("W").astype(str)
    heat = mw.groupby(["stock","week"]).size().reset_index(name="count")
    top_t = master["stock"].value_counts().head(15).index
    heat = heat[heat["stock"].isin(top_t)]
    pivot = heat.pivot(index="stock", columns="week", values="count").fillna(0)
    fig_h = px.imshow(pivot, color_continuous_scale=["#111318","#7c5cfc","#00d4ff"], aspect="auto")
    _dark(fig_h); fig_h.update_layout(height=400)
    st.plotly_chart(fig_h, use_container_width=True)

    # Organic vs watchlist breakdown per ticker
    if "organic_idea" in master.columns:
        st.markdown("### Organic vs Watchlist Origin")
        st.caption("Which stocks tend to show up as spontaneous ideas vs. pre-planned watchlist picks?")
        origin = master.groupby(["stock","organic_idea"]).size().reset_index(name="count")
        origin["type"] = origin["organic_idea"].map({0:"Watchlist",1:"Organic Idea"})
        top_stocks = master["stock"].value_counts().head(20).index
        origin = origin[origin["stock"].isin(top_stocks)]
        fig_o = px.bar(origin, x="stock", y="count", color="type",
                        color_discrete_map={"Watchlist":"#7c5cfc","Organic Idea":"#f59e0b"},
                        barmode="stack")
        _dark(fig_o); fig_o.update_layout(height=300)
        st.plotly_chart(fig_o, use_container_width=True)

    # Per-ticker deep dive
    st.markdown("### Ticker Deep Dive")
    sel = st.selectbox("Select ticker", sorted(master["stock"].dropna().unique()))
    sub = master[master["stock"]==sel]
    t_cols = st.columns(5)
    for i, f in enumerate(TIGERS):
        with t_cols[i]:
            pop = (sub[f].astype(str).str.strip()!="").mean() if f in sub.columns else 0
            c = "#4ade80" if pop>0.6 else "#f59e0b" if pop>0.3 else "#ff6b6b"
            st.markdown(f'<div style="background:#1a1d25;border:1px solid {c}33;border-radius:10px;'
                        f'padding:12px 8px;text-align:center;">'
                        f'<div style="font-family:Space Mono,monospace;font-size:10px;color:{c}">{f}</div>'
                        f'<div style="font-size:24px;font-weight:700;color:{c}">{pop*100:.0f}%</div>'
                        f'</div>', unsafe_allow_html=True)
    with st.expander("View all entries"):
        cols_show = ["date","on_watchlist","organic_idea","memo_coverage",
                     "tigers_score","outcome_category","win"] + TIGERS
        st.dataframe(sub[[c for c in cols_show if c in sub.columns]].sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB: TIGERS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def render_tigers_analysis(master: pd.DataFrame):
    if master.empty:
        st.info("Load data to begin.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### TIGERS Score Distribution")
        sd = master["tigers_score"].value_counts().sort_index().reset_index()
        sd.columns = ["score","count"]
        fig = px.bar(sd, x="score", y="count", color="score",
                      color_continuous_scale=["#ff6b6b","#f59e0b","#fbbf24","#22d3ee","#4ade80","#00d4ff"])
        _dark(fig); fig.update_layout(showlegend=False, height=260)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("### Factor Population Rate")
        pop = {c: (master[c].astype(str).str.strip()!="").mean()*100
               for c in TIGERS if c in master.columns}
        fig = go.Figure(go.Bar(
            x=list(pop.values()), y=list(pop.keys()), orientation="h",
            marker_color=["#00d4ff","#7c5cfc","#4ade80","#f59e0b","#ff6b6b"],
            text=[f"{v:.0f}%" for v in pop.values()], textposition="outside",
        ))
        _dark(fig); fig.update_layout(xaxis_title="% setups", yaxis=dict(autorange="reversed"), height=260)
        st.plotly_chart(fig, use_container_width=True)

    # TIGERS completion funnel: score vs win rate
    st.markdown("### TIGERS Completion Funnel — Score vs Win Rate")
    st.caption("The core hypothesis: more factors articulated = better-quality setup = better outcomes.")
    if "win" in master.columns:
        traded = master[master.get("did_trade", pd.Series(1, index=master.index)) == 1].dropna(subset=["win"])
        if not traded.empty:
            score_wr = traded.groupby("tigers_score").agg(
                win_rate=("win","mean"), n=("win","count")
            ).reset_index()
            fig_f = make_subplots(specs=[[{"secondary_y": True}]])
            fig_f.add_trace(go.Bar(x=score_wr["tigers_score"], y=score_wr["n"],
                                    name="# Trades", marker_color="rgba(124,92,252,0.4)"),
                             secondary_y=False)
            fig_f.add_trace(go.Scatter(x=score_wr["tigers_score"], y=score_wr["win_rate"],
                                        name="Win Rate", mode="lines+markers",
                                        line=dict(color="#4ade80", width=2.5),
                                        marker=dict(size=8)),
                             secondary_y=True)
            fig_f.add_hline(y=0.5, line_dash="dot", line_color="#374151", secondary_y=True)
            fig_f.update_xaxes(title_text="TIGERS Factors Articulated", dtick=1)
            fig_f.update_yaxes(title_text="# Trades", secondary_y=False, gridcolor="#1f2330")
            fig_f.update_yaxes(title_text="Win Rate", secondary_y=True,
                                tickformat=".0%", range=[0,1], gridcolor="rgba(0,0,0,0)")
            _dark(fig_f); fig_f.update_layout(height=300, legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig_f, use_container_width=True)

    # Radar by outcome
    st.markdown("### TIGERS Radar by Outcome Category")
    if "outcome_category" in master.columns:
        outcomes = [o for o in OUTCOME_COLORS if o in master["outcome_category"].values and o != "uncategorized"]
        fig_r = go.Figure()
        for i, outcome in enumerate(outcomes):
            sub = master[master["outcome_category"]==outcome]
            vals = [(sub[c].astype(str).str.strip()!="").mean() if c in sub.columns else 0 for c in TIGERS]
            c = list(OUTCOME_COLORS.values())[i % len(OUTCOME_COLORS)]
            h = c.lstrip("#")
            r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
            fig_r.add_trace(go.Scatterpolar(
                r=vals+[vals[0]], theta=TIGERS+[TIGERS[0]],
                fill="toself", name=OUTCOME_LABELS.get(outcome,outcome),
                line_color=c, fillcolor=f"rgba({r},{g},{b},0.12)",
            ))
        _dark(fig_r)
        fig_r.update_layout(
            polar=dict(bgcolor="rgba(0,0,0,0)",
                       radialaxis=dict(visible=True, range=[0,1], gridcolor="#252830",
                                       tickfont=dict(size=9, color="#6b7280")),
                       angularaxis=dict(gridcolor="#252830")),
            height=380,
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # Factor emphasis over time
    st.markdown("### Factor Emphasis Over Time")
    st.caption("Month-over-month shifts in which TIGERS factors you articulate — reveals evolving analytical habits.")
    master_t = master.copy()
    master_t["month"] = master_t["date"].dt.to_period("M").astype(str)
    mf = []
    for month, grp in master_t.groupby("month"):
        for f in TIGERS:
            if f in grp.columns:
                mf.append({"month":month, "factor":f,
                            "rate":(grp[f].astype(str).str.strip()!="").mean()})
    if mf:
        mf_df = pd.DataFrame(mf)
        fig_mf = px.line(mf_df, x="month", y="rate", color="factor",
                          color_discrete_map={"Tightness":"#00d4ff","Ignition":"#7c5cfc",
                                               "Group":"#4ade80","Earnings":"#f59e0b","RS":"#ff6b6b"},
                          labels={"rate":"% Setups Mentioning","month":"Month"})
        _dark(fig_mf); fig_mf.update_layout(height=280, yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_mf, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB: MEMO EXPLORER
# ═════════════════════════════════════════════════════════════════════════════

def render_memo_explorer(master: pd.DataFrame, mdf: pd.DataFrame):
    st.markdown("### Search & Explore Voice Memos")
    query       = st.text_input("🔍 Search transcripts", placeholder="NVDA breakout, anxious, rotation...")
    sent_filter = st.multiselect("Filter sentiment",
                                  mdf["market_sentiment"].dropna().unique().tolist() if not mdf.empty else [])
    disp = mdf.copy() if not mdf.empty else pd.DataFrame()
    if not disp.empty:
        if query:
            disp = disp[disp["raw_transcript"].astype(str).str.contains(query, case=False, na=False)]
        if sent_filter:
            disp = disp[disp["market_sentiment"].isin(sent_filter)]
        st.markdown(f"**{len(disp)}** entries match")
        for _, row in disp.sort_values("date", ascending=False).head(15).iterrows():
            ss = row.get("sentiment_score", 0)
            sc = "#4ade80" if ss>0 else "#ff6b6b" if ss<0 else "#6b7280"
            em = str(row.get("emotional_state",""))
            ec = EMOTION_COLORS.get(em.lower(), "#6b7280")
            txt = str(row.get("raw_transcript",""))
            if query: txt = _hl(txt, query)
            st.markdown(f"""
            <div style="background:#1a1d25;border:1px solid #252830;border-radius:10px;
                        padding:16px 20px;margin-bottom:8px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-family:'Space Mono',monospace;font-size:13px;color:#e8eaf0">
                  {pd.to_datetime(row['date']).strftime('%b %d, %Y') if pd.notna(row.get('date')) else '—'}
                </span>
                <div>
                  <span style="font-family:Space Mono,monospace;font-size:10px;padding:2px 8px;
                               border-radius:4px;background:{sc}22;color:{sc};border:1px solid {sc}44;
                               margin-right:4px">{row.get('market_sentiment','')}</span>
                  <span style="font-family:Space Mono,monospace;font-size:10px;padding:2px 8px;
                               border-radius:4px;background:{ec}22;color:{ec};border:1px solid {ec}44">
                    {em}</span>
                </div>
              </div>
              <div style="font-size:13px;color:#9ca3af;line-height:1.7;font-style:italic">
                "{txt[:500]}{"..." if len(str(row.get("raw_transcript","")))>500 else ''}"
              </div>
            </div>""", unsafe_allow_html=True)

    if not master.empty:
        st.markdown("---")
        st.markdown("### Stock Thought Table")
        sel_s = st.selectbox("Filter ticker", ["All"] + sorted(master["stock"].dropna().unique().tolist()))
        view  = master if sel_s=="All" else master[master["stock"]==sel_s]
        show  = ["date","stock","on_watchlist","organic_idea","memo_coverage",
                 "tigers_score","outcome_category","win"] + TIGERS
        st.dataframe(view[[c for c in show if c in view.columns]].sort_values("date", ascending=False),
                     use_container_width=True, hide_index=True)


def _hl(text, query):
    try:
        return re.sub(f"({re.escape(query)})",
                      r'<mark style="background:#7c5cfc44;color:#e8eaf0;border-radius:3px">\1</mark>',
                      text, flags=re.IGNORECASE)
    except Exception:
        return text


def _mock_spy(dates):
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0004, 0.01, len(dates))
    return pd.Series(450 * np.exp(np.cumsum(returns)), index=dates.values)