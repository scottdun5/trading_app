import streamlit as st

st.set_page_config(page_title="Trading Dashboard", layout="wide")

st.sidebar.title("Navigation")
st.sidebar.page_link("pages/portfolio_page.py", label="Portfolio")
st.sidebar.page_link("pages/journal.py", label="AI Journal")
st.sidebar.page_link("pages/gallery2.py", label="Gallery")
st.sidebar.page_link("pages/causal_analysis.py", label="Causal Analysis")

