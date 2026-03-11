"""
pages/kpi_dashboard.py
Streamlit multi-page entry point for the KPI Dashboard.
"""
import sys
from pathlib import Path

# Make utils importable
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import streamlit as st

st.set_page_config(
    page_title="KPI Dashboard",
    page_icon="📊",
    layout="wide",
)

from kpi_page import render_kpi_page

render_kpi_page()
